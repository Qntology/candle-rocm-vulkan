# Shared helpers of the CI scripts. Every step is recorded in $OUT/summary.md as PASS / FAIL / NOT RUN (+ reason);
# NOT RUN is never counted as a pass.
OUT="${OUT:?OUT must be set}"
mkdir -p "$OUT"
SUMMARY="$OUT/summary.md"
FAILS=0
[ -f "$SUMMARY" ] || printf '| step | result | log |\n|---|---|---|\n' > "$SUMMARY"
SUMMARY_SEEN="$(wc -l < "$SUMMARY")"   # the rows of earlier phases were published by their own step

# run <label> <cmd...> : runs the command, tees to $OUT/<label>.log, records PASS/FAIL.
PHASE_LABELS=()
run() {
  local label="$1"; shift
  PHASE_LABELS+=("$label")
  local log="$OUT/$label.log"
  echo "::group::$label"
  echo "+ $*" | tee "$log"
  set +e
  "$@" 2>&1 | tee -a "$log"
  local rc=${PIPESTATUS[0]}
  set -e
  echo "::endgroup::"
  local counts; counts="$(test_counts "$log")"
  if [ "$rc" -eq 0 ] && grep -qE '^running 0 tests$' "$log" 2>/dev/null; then
    # Guard against stale/mismatched test filters silently passing with 0 executed tests (P5-O01).
    local passed_total; passed_total="$(awk '/^test result:/ { for (i = 2; i <= NF; i++) if ($i ~ /^passed/) p += $(i - 1) } END { print p + 0 }' "$log")"
    if [ "$passed_total" -eq 0 ]; then
      echo "| $label | FAIL (0 tests ran) | $label.log |" >> "$SUMMARY"
      echo "::error title=$label::cargo test ran 0 tests (check test filter)"
      FAILS=$((FAILS + 1))
      return 0
    fi
  fi
  if [ "$rc" -eq 0 ]; then
    echo "| $label | PASS${counts:+ ($counts)} | $label.log |" >> "$SUMMARY"
  else
    echo "| $label | FAIL (exit $rc${counts:+, $counts}) | $label.log |" >> "$SUMMARY"
    annotate_log "$label (exit $rc)" "$log"
    FAILS=$((FAILS + 1))
  fi
  return 0
}

# test_counts <log> : totals of the `test result:` lines of a log (several test binaries each print one).
test_counts() {
  awk '/^test result:/ {
         for (i = 2; i <= NF; i++) {
           if ($i ~ /^passed/) p += $(i - 1)
           if ($i ~ /^failed/) f += $(i - 1)
           if ($i ~ /^ignored/) g += $(i - 1)
         }
         n = 1
       }
       END { if (n) printf "%d passed, %d failed, %d ignored", p, f, g }' "$1"
}

# failed_tests <log> : names from the closing `failures:` lists. With --nocapture the `test x ... FAILED` lines are
# torn apart by the output of the tests; these lists are not.
failed_tests() {
  awk '/^failures:$/ { list = 1; next }
       list && /^    [^ ]+$/ { print $1; next }
       list && NF { list = 0 }' "$1" | sort -u
}

# escape : one annotation message per input; `%` and line ends would otherwise be read as commands.
escape() { tr -d '\r' | sed 's/%/%25/g'; }

# annotate_log <label> <log>: surface the important lines of a failed step as annotations
# (readable through the checks API even where artifact / log downloads are blocked).
annotate_log() {
  # One annotation per failed step (GitHub keeps only 10 error annotations per step): the error /
  # panic lines with their next lines, then the log tail, joined with " | ".
  local label="$1" log="$2"
  local body
  body="$(grep -E -A4 "^error(\[|:)|^Error|: error:|panicked at|^failures:|assertion .*failed|FAILED| case [0-9]+[ :]|failed to run custom build|--- stderr|not found|STATUS_|fatal" "$log" | grep -vE "^--$|^\s*[0-9]+: |^\s+at " | head -40 | sed 's/$/ |/' | tr -d '\n' | cut -c1-2800 | escape)"
  echo "::error title=$label::$body || tail: $(tail -n 8 "$log" | sed 's/$/ |/' | tr -d '\n' | cut -c1-1000 | escape)"
  local names; names="$(failed_tests "$log")"
  if [ -n "$names" ]; then
    echo "::error title=$label failed tests ($(echo "$names" | wc -l | tr -d ' '))::$(echo "$names" | tr '\n' ' ' | cut -c1-3600 | escape)"
  fi
}

# not_run <label> <reason>
not_run() {
  echo "| $1 | NOT RUN — $2 | |" >> "$SUMMARY"
  echo "::warning::$1 not run: $2"
}

# fail_step <label> <reason> : a failure that is not the exit code of a command.
fail_step() {
  echo "| $1 | FAIL — $2 | |" >> "$SUMMARY"
  echo "::error title=$1::$2"
  FAILS=$((FAILS + 1))
}

# require_measured <run-label> <pattern> : a run that "passed" by skipping itself (no measurement line in its
# log) is rewritten from PASS to NOT RUN in the summary — a skipped run never counts as a pass.
require_measured() {
  local label="$1" pat="$2" log="$OUT/$1.log"
  if ! grep -qE "$pat" "$log" 2>/dev/null; then
    sed -i.bak -E "s/^\| $label \| PASS[^|]*\|/| $label | NOT RUN — test skipped itself (no measurement line) |/" "$SUMMARY" && rm -f "$SUMMARY.bak"
    echo "::warning::$label: test produced no measurement (skipped)"
  fi
}

# refuse_skip <run-label> <pattern> : tests that report `ok` after skipping themselves print why. This job exists to run
# them, so such a line fails the step. Needs --nocapture in the run.
refuse_skip() {
  local label="$1" pat="$2" hit
  hit="$(grep -E "$pat" "$OUT/$label.log" 2>/dev/null | grep -v '^+ ' | sort -u | head -5 | cut -c1-300 | tr '\n' ' ' || true)"
  if [ -n "$hit" ]; then fail_step "$label" "tests skipped themselves: $(echo "$hit" | escape)"; fi
}

# Measurement lines and the summary table as a few long notices (annotations are readable through the
# checks API; artifact / log blobs may not be reachable from where the results are read).
publish_notices() {
  # Key lines first (one-line summaries), each tagged with its run label, then the detail lines.
  local tmp="$OUT/.measure.txt"
  local key="DEVICE_PROBE|-M-INV\] 측정 결과|-M-INV\] arch|^test result: FAILED"
  local detail="\[VK-HOST-SMALL-M\]|candle gated delta rule: Metal|candle flash-attn: fused kernel unavailable"
  : > "$tmp"
  for tag in ${PHASE_LABELS[@]+"${PHASE_LABELS[@]}"}; do
    local f="$OUT/$tag.log"
    [ -f "$f" ] || continue
    grep -E "$key" "$f" 2>/dev/null | grep -v '^+ ' | cut -c1-500 | sed "s|^|<$tag> |" >> "$tmp" || true
  done
  for tag in ${PHASE_LABELS[@]+"${PHASE_LABELS[@]}"}; do
    local f="$OUT/$tag.log"
    [ -f "$f" ] || continue
    grep -E "$detail" "$f" 2>/dev/null | grep -vE "$key" | grep -v '^+ ' | cut -c1-300 | sed "s|^|<$tag> |" >> "$tmp" || true
  done
  local rows; rows="$(tail -n +$((SUMMARY_SEEN + 1)) "$SUMMARY" | sed 's/$/ ;;/' | tr -d '\n' | cut -c1-1800 | escape)"
  if [ -n "$rows" ]; then echo "::notice title=summary::$rows"; fi
  # Annotation messages are cut around 4 KB: keep each chunk well below that (Korean is 3 bytes/char).
  local chunk="" n=0
  while IFS= read -r line; do
    if [ $(( ${#chunk} + ${#line} )) -gt 1100 ] && [ -n "$chunk" ]; then
      echo "::notice title=measure-$n::$(echo "$chunk" | escape)"; n=$((n + 1)); chunk=""
      [ $n -ge 8 ] && break
    fi
    chunk="$chunk$line ;; "
  done < "$tmp"
  [ -n "$chunk" ] && [ $n -lt 8 ] && echo "::notice title=measure-$n::$(echo "$chunk" | escape)"
  return 0
}

finish() {
  publish_notices
  echo; cat "$SUMMARY"
  [ "$FAILS" -eq 0 ]
}
