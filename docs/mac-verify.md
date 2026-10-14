# Mac 검증 기록 — 개발 환경 · 진행 컨텍스트 · 테스트 결과

이 문서는 logis-center `docs/mac-verify-prompt.md` 작업(Apple Silicon CPU · Metal · MoltenVK 검증과 개선)을 하면서 얻은 내용을 정리합니다. 시행착오로 알게 된 것도 함께 남깁니다.

| 항목 | 위치 |
|---|---|
| 포크 작업 브랜치 | `claude/mac-verify` |
| logis-center 작업 브랜치 | `claude/mac-verify` |
| 앱 쪽 체크리스트 | logis-center `docs/mac-verify.md` |
| CI 워크플로 | logis-center `.github/workflows/mac-verify.yml` |
| CI 스크립트 | logis-center `.github/scripts/{lib,app-tests,fork-tests,apply-fork-patches,use-fork-path}.sh` |

---

## 0. 개발 진행 방식 — 개발은 리눅스에서, 실제 검증만 GitHub Mac에서

Mac 러너 1회는 빌드 포함 1~4시간이 걸리고, 결과는 잡이 끝나야 보입니다. 그래서 **코드 작성 · 타입 검사 · 돌릴 수 있는 테스트는 전부 리눅스 VM에서 끝냅니다.** Mac에서만 확인할 수 있는 것(실제 Metal · MoltenVK 장치에서의 실행, 수치, 속도)만 GitHub macOS 러너로 보냅니다.

| 단계 | 어디서 | 무엇을 |
|---|---|---|
| 1. 구현 | 리눅스 VM | 포크(`../candle-rocm-vulkan`)와 앱을 수정합니다. 앱은 로컬에서만 path 의존으로 포크를 봅니다(커밋 금지) |
| 2. 타입 검사 | 리눅스 VM | 바뀐 모든 대상을 `cargo check`합니다. CPU: `-p candle-core` · `-p candle-nn`. Vulkan: `--features vulkan`. **Metal: `--target aarch64-apple-darwin --features metal`**(1-5절, 링크 없이 Rust 쪽 오류를 다 잡음). 앱: `--test it` + 필요한 feature |
| 3. 리눅스 실행 검증 | 리눅스 VM | CPU 경로 테스트, lavapipe(`CANDLE_VULKAN_ALLOW_CPU=1`)로 Vulkan 경로의 **정확성**. lavapipe는 CPU 장치라 속도 · 메모리 반환은 대표하지 않습니다 |
| 4. 커밋 · push | 리눅스 VM | 포크 `claude/mac-verify` push → logis-center `patches/` 재생성(`format-patch origin/master..claude/mac-verify`) → logis-center 커밋에 **`[skip ci]`** (push로 전체 런이 돌지 않게) |
| 5. Mac 검증 1회 | GitHub macOS | **바뀐 경로만** dispatch합니다. 예: `gh api -X POST repos/sungu-che/logis-center/actions/workflows/mac-verify.yml/dispatches -f ref=claude/mac-verify -f 'inputs[phases]=accel' -f 'inputs[app]="metal"' -f 'inputs[fork]="metal"' -f 'inputs[accel_tests]=gdn,qwen4' -f 'inputs[full]=false'` |
| 6. 결과 읽기 | 리눅스 VM | 잡이 끝날 때마다 `gh api repos/sungu-che/logis-center/check-runs/<job-id>/annotations`로 읽습니다(로그 · 아티팩트는 403). 오래 막아 두는 대기 대신 주기적으로 확인합니다 |
| 7. 판단 | — | 통과하면 문서(logis-center `docs/mac-verify.md`, 이 파일)에 수치를 기록하고 끝냅니다. 실패하면 원인을 annotation에서 찾아 1단계로 돌아갑니다. 같은 테스트를 그대로 다시 돌리지 않습니다 |

Mac에서만 확인할 수 있는 것과 리눅스에서 끝내는 것:

| 리눅스 VM에서 끝냄 | GitHub Mac에서만 |
|---|---|
| Rust 컴파일 · 타입 오류 (macOS 대상 Metal 코드 포함) | MSL 소스 컴파일(런타임 `new_library_with_source`), GLSL → MoltenVK 변환 |
| CPU 경로의 정확성 · 비트 동일성 (x86) | aarch64 NEON 결과, Apple Silicon digest |
| Vulkan 경로의 정확성 (lavapipe) | MoltenVK · Metal 실제 실행 · 수치 · 속도 |
| 테스트 · CI 스크립트 문법 (`bash -n`, `actionlint`) | 장치 메모리 반환(통합 메모리), 실모델 다운로드가 필요한 판정 일치 |

이번 작업에서 이 원칙을 지키지 않아 생긴 문제와 그 교훈:
- **Metal 코드를 Mac에서 처음 컴파일:** 리눅스에서 macOS 대상으로 타입 검사할 수 있었습니다(1-5절). 그전에는 Mac 런에서 처음 오류를 봐서 왕복이 늘었습니다.
- **같은 테스트를 여러 번 돌림:** 전체 런을 반복해 Mac 러너 시간을 낭비했습니다. 이제 dispatch 입력으로 바뀐 백엔드 · 단계 · 항목만 고릅니다.
- **debug 빌드 앱 테스트 시간으로 커널 속도를 판단:** 그 결과 macOS packed CPU 커널 기본 ON을 넣었다가 되돌렸습니다. 속도 판단은 release 벤치로 합니다.
- **Mac에서만 보이는 수치 문제:** Metal fast math로 생긴 오차는 Mac에서만 보입니다. 새 Metal 커널은 처음부터 정밀 수학과 런타임 자가 점검을 같이 넣습니다.

---

## 1. 개발 환경 — 알아두어야 할 것

### 1-1. 작업 구성

- 개발 세션은 **리눅스 x86 클라우드 VM**입니다(Xeon Cascade Lake AVX-512, 4 vCPU, 16 GB).
  - Mac 검증은 전부 **GitHub Actions macOS 러너**에서 합니다.
  - 리눅스에서는 CPU 검증, lavapipe Vulkan, 컴파일 검사를 합니다.
- 포크 변경은 두 가지로 둡니다.
  1. 포크 `claude/mac-verify` 브랜치에 push합니다(모드 A).
  2. 같은 커밋을 `git format-patch origin/master..claude/mac-verify`로 logis-center `patches/candle-rocm-vulkan/`에 내보냅니다(모드 B).
- CI는 포크 **master**를 체크아웃하고 시리즈를 `git am --3way`로 적용합니다.
  - master에 이미 들어간 패치는 "No changes -- Patch already applied"로 건너뜁니다. 그래서 PR 병합 뒤에도 시리즈가 깨지지 않습니다.
  - 앱 `Cargo.toml`은 잡 안에서만 path 형태로 바꿉니다(`use-fork-path.sh`).
- 커밋된 앱 `Cargo.toml`은 **Qntology master git 줄을 유지**합니다. 로컬 path 전환은 절대 커밋하지 않습니다.
  - 따라서 포크 수정이 Qntology master에 들어가기 전에는, CI 밖의 일반 빌드에 포크 수정이 반영되지 않습니다.
  - 앱 코드는 새 포크 API에 의존하지 않게 씁니다. 예: `backends.metal` 필드를 앱에서 직접 쓰지 않음.

### 1-2. GitHub 접근 (이 세션 환경)

- 저장소마다 push 권한을 받아야 합니다(세션의 add_repo). 권한이 없으면 push와 `gh api`가 403입니다.
- `gh`는 **REST만** 됩니다. GraphQL 기반 `gh pr` · `gh issue`는 403입니다.
- **잡 로그와 아티팩트는 읽을 수 없습니다.** blob 저장소(`productionresultssa*.blob.core.windows.net`)가 403입니다.
  - 그래서 결과는 **annotation**으로 내보내고 `gh api repos/O/R/check-runs/<job-id>/annotations`로 읽습니다.
  - annotation은 **잡이 끝나야** 보입니다. 단계 안의 진행 상황은 볼 수 없습니다.
  - 한 메시지가 약 4 KB에서 잘립니다(한글은 3바이트). 그래서 1100자 단위로 나누고, 요약 줄을 먼저 넣습니다(`lib.sh publish_notices`).
  - 실패 단계는 단계당 error annotation이 10개로 제한되므로, 압축한 error 하나로 냅니다(`annotate_log`).
- `.github/workflows/` 수정 push에는 workflow 권한이 필요합니다(이 세션에서는 허용됐습니다).
- `workflow_dispatch`는 워크플로 파일이 **기본 브랜치에도** 있어야 동작합니다(PR #62 병합 뒤 가능).
  - push로 전체 런이 돌지 않게 하려면 커밋 메시지에 `[skip ci]`를 넣고 dispatch로 필요한 부분만 돌립니다.
- `concurrency: cancel-in-progress: false`에서는 같은 ref의 새 런이 **대기**합니다.
  - 대기 중인 런 위에 새 런을 넣으면 이전 대기 런이 취소됩니다.

### 1-3. macOS 러너

| 러너 | 실제 | GPU |
|---|---|---|
| `macos-15` | "Apple M1 (Virtual)" 3코어(perflevel0만), L1D 128 KB · L1I 192 KB · L2 12 MB · 캐시라인 128 B · RAM 7.5 GB | **Metal "Apple Paravirtual device"** (unified, recommendedMaxWorkingSetSize 5.01 GB), **MoltenVK INTEGRATED_GPU** (API 1.4.357) |
| `macos-14` | — | Metal 장치 없음 → 환경 기록용 |

- 가상 GPU라 **명령 제출 1회 왕복이 비쌉니다.**
  - MoltenVK에서 `submit + wait`가 약 0.8 ms입니다. 연산 수가 시간을 지배합니다.
  - Metal도 명령 버퍼 동기화가 비쌉니다. 그래서 이 VM에서는 GDN 디코드가 CPU 융합 커널보다 느립니다.
- 시간 측정은 같은 잡 안의 상대 비교로만 씁니다. VM 편차가 ±15~40%입니다.
- macOS arm64 CI 큐가 길 수 있습니다("capacity constraints"). queued는 실패가 아닙니다.

### 1-4. 빌드 함정과 해결

| 증상 | 원인 | 해결 |
|---|---|---|
| ring 0.17.14 const assert 실패 (`src/cpu/arm/darwin.rs`) | 앱 `.cargo/config.toml`의 `target-cpu=native`가 가상 M1에서 generic aarch64로 떨어짐(aes/sha2 없음). Rust 버전 문제가 아님 | 잡에 `RUSTFLAGS=-C target-cpu=apple-m1` |
| lance-encoding 빌드 스크립트 실패 | protoc 없음 | `brew install protobuf` |
| onnxruntime-sys가 macOS arm64에서 panic | triplet 없음 | `(MacOs, Arm64)` 추가 |
| (로컬) onnxruntime-sys 다운로드 TLS 실패 | 프록시 경유 ureq | `ORT_STRATEGY=system` + 수동 다운로드한 ORT |
| 디스크 부족 (앱 debug target 27 GB) | debuginfo | `CARGO_PROFILE_DEV_DEBUG=0`, `CARGO_INCREMENTAL=0` |
| MoltenVK 장치가 안 보임 | 새 로더는 portability enumeration 필요, SIP가 `DYLD_*`를 지움 | 포크 `91a7ab2`: `VK_KHR_portability_enumeration`, `portability_subset`, `/opt/homebrew/lib` 등 직접 탐색. CI는 `brew install molten-vk vulkan-loader` + `VK_ICD_FILENAMES` |
| 앱 macOS `direct_loader`가 컴파일되지 않음 | 존재하지 않는 Metal IO API 호출 | 병렬 pread/pwrite + `F_NOCACHE`로 다시 작성 (logis-center) |

### 1-5. 리눅스에서 Metal 코드 타입 검사 (테스트 아님)

candle-core는 tokenizers를 통해 oniguruma C 빌드에 의존합니다. 가짜 pkg-config 항목으로 우회하면 macOS 대상 `cargo check`가 됩니다(링크는 안 함).

```bash
P=$(mktemp -d); mkdir -p $P/include; : > $P/include/oniguruma.h
printf "prefix=/usr\nlibdir=/usr/lib\nincludedir=$P/include\nName: oniguruma\nDescription: fake\nVersion: 6.9.9\nLibs: -lonig\nCflags: -I$P/include\n" > $P/oniguruma.pc
rustup target add aarch64-apple-darwin
PKG_CONFIG_PATH=$P PKG_CONFIG_ALLOW_CROSS=1 RUSTONIG_SYSTEM_LIBONIG=1 \
  cargo check --target aarch64-apple-darwin -p candle-core --features metal   # candle-nn 도 같은 방법
```

- MSL 소스(`new_library_with_source`)는 **Mac 런타임에서야** 컴파일됩니다. 그래서 새 Metal 커널은 런타임 자가 점검과 폴백을 같이 둡니다.

### 1-6. 수치 · 측정 함정

- **candle Metal 커널은 기본이 fast math입니다**(`CANDLE_METAL_ENABLE_FAST_MATH`, 기본 true).
  - 정밀도가 필요한 커스텀 커널은 라이브러리를 fast math 없이 만들고 `precise::exp/log/divide/sqrt`를 씁니다.
  - 첫 Metal GDN은 fast math 때문에 토큰마다 곱해지는 `exp(g)`의 오차가 누적돼 f64 기준 2e-5를 넘었습니다.
- **앱 테스트는 debug 프로필입니다.** 앱 크레이트는 opt-level 0이고, candle · gemm · embed-cpu-kernels는 3입니다.
  - 앱 크레이트 코드와 커널 크레이트 코드의 속도를 앱 테스트 시간으로 비교하면 틀립니다.
  - macOS packed CPU 커널 기본 ON을 이 오류로 넣었다가 되돌렸습니다. release 벤치에서는 M1 NEON에서 gemm 크레이트가 더 빠릅니다.
- **Apple Silicon에서 비트 동일성**: CPU gemm(M≥2), MoltenVK Gpu 모드 GEMM, Metal MLX GEMM이 테스트한 형태에서 **같은 digest `4bfc0461001d5039`** 를 냅니다. 출력 원소마다 k 순서 FMA 체인 하나씩이기 때문입니다.
  - 이것이 "작은 M은 host" 경로(`51e2256`)의 근거입니다. 그래도 장치 · 형태마다 실제 가중치로 확인한 뒤에만 씁니다.
- CPU gemm은 **M=1에서만** 경로가 다릅니다(gemv, 8누산기 트리 합). M≥2는 패킹 마이크로커널이고 split-K가 없습니다.
- aarch64 gemm-f16은 **f16으로 누산**합니다. `forward_exact`(f16 가중치)는 f32 누산으로 바꿨습니다(`c1de501`, `a08ba34`).
- GPU에 상주하는 Q8_0 matmul은 **정확한 W8A32**입니다. CPU QMatMul은 활성값을 Q8_0로 양자화합니다(llama.cpp vec_dot).
  - 테스트 기준을 경로에 맞춰야 합니다. MoltenVK "diff 0.25–0.62"는 버그가 아니라 기준 차이였습니다(`a08ba34`).
- naga 27은 GLSL `barrier()`를 subgroup 범위까지 포함해 번역합니다. 워크그룹 배리어로 고정했습니다(`c663a23`). Q8_0 차이의 원인은 아니었습니다.
- lavapipe(`CANDLE_VULKAN_ALLOW_CPU=1`)는 **CPU 장치**라 MoltenVK 성능을 대표하지 않습니다.
  - 장치 메모리가 곧 시스템 RAM이라 free 반환 측정이 의미가 없습니다.
- 전역 상태를 바꾸는 테스트는 `#[ignore]`로 두고 `--test-threads 1`로 단독 실행합니다.
  - 해당 테스트: accel 설정, GDN override, 장치 전체 메모리 측정.
- **가상 러너의 시간은 런마다 1.2~2배 흔들립니다.** 코드가 그대로인 CPU 융합 GDN이 21.0 → 25.3 · 40.0 ms, MoltenVK GDN 텐서 경로 디코드가 2.03 → 3.47 ms로 변했습니다.
  - 런 사이 절대 시간 비교로 회귀를 판단하지 않습니다. 같은 잡 안의 비율과, 바뀌지 않은 경로의 기준 시간을 함께 봅니다.
  - 두 경로의 단건 지연은 한쪽씩 몰아서 재지 않고 번갈아 재서 중앙값을 씁니다. 몰아서 잰 평균은 순간 지연이 한쪽에만 실려 57.0 / 97.6 ms처럼 뒤집힌 결과가 나왔습니다(같은 런 조건에서 번갈아 재면 87.9 / 76.8 ms).
- **시드 없는 무작위 입력 + 경계에 가까운 기준은 가끔 실패합니다.** `half_kv_storage_error_is_small`(앱)은 약 0.5% 확률로 2e-3을 넘었습니다. 고정 시드로 바꿨습니다.

### 1-7. CI 운영 원칙 (사용자 합의)

- 건너뛴 실행은 통과가 아닙니다. 측정 줄이 없으면 `require_measured`가 PASS를 NOT RUN으로 바꿉니다.
- **중복 테스트를 하지 않습니다.**
  - 바뀐 경로만 다시 돌립니다. dispatch 입력: `phases`, `app`, `fork`, `fork_tests`, `embed_backends`, `real_backends`, `embed_compare`(1 · 0 · `speed`), `accel_tests`, `runner`, `full`.
  - `embed_compare=speed`는 기본 정책의 속도 · 비트 동일 테스트만 돌립니다(판정 테스트 제외).
- 포크 브랜치가 master에 병합되면 logis-center의 패치 시리즈 일부가 이미 master에 있게 됩니다. `apply-fork-patches.sh`는 patch-id가 master 커밋과 같은 패치를 건너뜁니다. 시리즈는 항상 `-o` 절대 경로로 다시 만듭니다.
  - CPU 측정은 Metal · Vulkan 잡 안에서 함께 하므로 CPU 잡의 accel 단계는 건너뜁니다.
  - 포크 GDN 테스트는 GPU 잡에서 GPU 케이스만 돌립니다. f64 기준과 호스트 에뮬레이션은 CPU 잡에서 돌립니다.
- 앱 잡은 단계마다 별도 step과 timeout을 둡니다(build 70, suite 30, embed 75, bench 30, real 50, accel 90분). 늦은 단계가 앞 결과를 잃지 않게 하기 위해서입니다.
- 비교 정책 런(고정 모양 강제, Host · Auto 모드)은 결정 테스트를 돌리지 않습니다(3코어에서 수 시간 걸림).
  - 구성 무관성을 보장하지 않는 정책은 `LOGIS_EMBED_MEASURE_ONLY=1`로 측정만 합니다.

---

## 2. 포크 변경 (`claude/mac-verify`)

| 커밋 | 내용 | master |
|---|---|---|
| `d0121b1` | 행 불변성 테스트(M 1..300 · 위치 · 스레드) + `device_probe` 예제 | ✅ PR #10 |
| `91a7ab2` | MoltenVK: portability enumeration · portability_subset · 로더 경로 | ✅ PR #10 |
| `30e502f` | Metal `Device::mem_info` = recommendedMaxWorkingSetSize − currentAllocatedSize | ✅ PR #10 |
| `c663a23` | GLSL `barrier()` → 워크그룹 배리어 | ✅ PR #11 |
| `c1de501` | CPU f16 `forward_exact` 프리필 f32 누산 | ✅ PR #11 |
| `a08ba34` | Vulkan tests: GPU 상주 Q8_0 = 정확한 W8A32, Vulkan f16 `forward_exact` f32 누산 | ✅ PR #11 |
| `3ad5050` | **Metal FlashAttention → fused sdpa 커널** (MLX 포트, GQA, causal 오른쪽 아래 정렬, 디코드 vector / 1024키 초과 2-pass). `BackendSwitches.metal`. TurboQuant는 Metal에서 계속 꺼둠 | ✅ PR #11 |
| `f947435` | **Gated DeltaNet 융합 경로**: Vulkan(`host_shared_memory()`)은 CPU 커널을 장치 텐서 위에서 실행, Metal은 네이티브 스캔(자가 점검 후 사용). `delta_rule_tests`의 GPU 케이스가 Metal · Vulkan에서도 돎 | ✅ PR #11 |
| `51e2256` | **Vulkan 고정 GPU 모드: 작은 M(2..=32, `CANDLE_VULKAN_HOST_SMALL_M`) F32 가중치 matmul은 host로**. 장치 · 형태마다 host 결과 == GPU 결과(비트)일 때만 적용, `[VK-HOST-SMALL-M]` 로그 | ✅ PR #11 |
| `806aba6` | **Metal GDN 정밀화**: conv · SiLU · L2 norm · 게이트를 fast math 없는 커널(`gdn_conv_f32`, `gdn_scan_f32`)에서 precise 수학으로, CPU 커널과 같은 갱신 순서 | ✅ PR #11 |
| `e4a7314` | **Metal 빈 텐서**: `new_buffer_with_data`가 길이 0을 `newBufferWithBytes`(nil 반환)에 넘기던 것 → 풀 버퍼(`allocate_buffer(0)`) | ✅ PR #11 |
| `a633c43` | **Metal GDN 디코드 디스패치 감소**: conv 커널(`gdn_conv_f32`)이 새 conv 상태도 직접 씀(buffer 6, t==0 스레드). 스텝당 디스패치 2개(conv · scan), conv 상태용 별도 cat/narrow 없음 | 브랜치만 |

---

## 3. GitHub macOS 테스트 결과

| 런 | 내용 |
|---|---|
| [38499518771](https://github.com/sungu-che/logis-center/actions/runs/38499518771) | 전체 |
| [38506246088](https://github.com/sungu-che/logis-center/actions/runs/38506246088) | 포크 개선(`3ad5050` · `f947435` · `51e2256`) 검증 |
| [38513842101](https://github.com/sungu-che/logis-center/actions/runs/38513842101) | `a633c43`(Metal conv 상태) 검증: fork는 `delta-rule-metal`만 (`fork_tests`) |

### 3-1. 포크 잡

| 잡 | 결과 |
|---|---|
| fork (cpu, NEON) | M-불변성: 1스레드 · 3스레드 digest `4bfc0461001d5039` 동일, M≥2 불일치 0, M=1만 다름. `delta_rule_tests` 9 pass, `w8a16_generic_cpu`, `matmul_tests` 7, candle-nn `cpu_flash_attn` 7 |
| fork (metal) 1차 | core/nn `--features metal` 빌드, `device_probe` OK. Metal M-불변성: Linear 0 · 배치 attention 0, digest `4bfc0461001d5039`. `w8a16_generic_metal`, `matmul_tests` 12, `delta_rule_tests` 9 (당시 `gpu()`가 CUDA/ROCm만 찾아 GPU 케이스는 미실행) |
| fork (metal) 2차 (38506246088) | Metal GDN 자가 점검 통과(max rel 5.4e-7)했지만 **`gpu_matches_f64_reference` · `half_inputs` 실패**(2e-5 허용치, fast math) → `806aba6`으로 정밀화 |
| fork (metal) 3차 (38508388995) | 정밀화 후 CPU 대비 0~1.2e-7. 같은 두 테스트가 conv 폭 1 케이스에서 `Err`로 끝남: 빈 conv 상태 텐서를 Metal로 보낼 때 길이 0 버퍼 생성 실패 → `e4a7314` |
| **fork (metal) 4차 ([38508978460](https://github.com/sungu-che/logis-center/actions/runs/38508978460))** | **`delta_rule_tests` 7 pass** (Metal GPU 케이스 전부: dk 64/96/128/256, Tiled, conv 폭 1~5, dv 200, L2 없음, f16/bf16 입력, 디코드 체이닝, transformers 픽스처) |
| **fork (metal) 5차 ([38513842101](https://github.com/sungu-che/logis-center/actions/runs/38513842101))** | `a633c43` 적용 후 **`delta_rule_tests` 7 pass**, Metal 스캔 자가 점검 max rel 3.3e-7 (conv 상태를 커널이 쓰는 경로) |
| fork (vulkan, MoltenVK) 1차 | `vulkan_tests` 19/19, `fixed_exec_modes_are_batch_invariant`, `vulkan_accel_tests` 3/3, candle-nn `accel_vulkan` 4/4, `w8a16_generic_cuda::vulkan`. Gpu · Host · Auto M-불변성 불일치 0, digest `4bfc0461001d5039` |
| fork (vulkan) 2차 (38506246088) | **전부 통과.** `[VK-HOST-SMALL-M]` 384×1152 · 384×384 · 384×3072 · 1536×384 모두 host == GPU → host 경로 사용. 그 상태로 M-불변성 불일치 0, `vulkan_tests` 19/19, **`delta_rule_tests`(Vulkan GPU 케이스) 7 pass** |

### 3-2. 앱 잡 (logis-center, 포크 수정이 적용된 빌드)

**Metal (38499518771)** — 임베딩 배치 순전파, 동적 모드

| 모델 | 전체 | 짧은 구 | 단건 per-text / embed_many | 비트 동일 | 판정 |
|---|---|---|---|---|---|
| 실모델 | 2.44배 | 4.77배 | 12.5 / 12.2 ms | 320/320 | 150/150, Stage-1 154/154, NMS 4/4, Plinko 25/25 |

- 고정 모양 강제는 0.47배라 동적 모드를 기본으로 정했습니다.
- suite 195 pass. Qwen4-Exp tiny Metal: max |Δ| 3.37e-5, argmax 100%.

**Metal (38506246088, 포크 `3ad5050` · `f947435`)**

- **FA(sdpa)**
  - 앱 자가 점검 통과(F32 d64, 5개 형태, max rel 7.0e-7).
  - 목업 Qwen3 teacher-forcing(프리필 64 + 디코드 96): FA on/off max |Δlogit| 1.82e-5, KLD 1.6e-13, top-1 100%.
- **GDN (1차 커널)**
  - 텐서 경로 대비 프리필(T=256) **5.26배**(167.6 → 31.9 ms), 디코드 **2.02배**(7.60 → 3.77 ms/스텝).
  - CPU 융합 대비 상대오차 ≤ 8.2e-7.
  - 참고: 이 VM에서는 CPU 융합(21.7 ms / 0.52 ms)이 더 빠릅니다(가상 GPU의 동기화 비용).
- **GPU 메모리 (`30e502f`)**
  - total 5011 MB. 256 MiB 할당 → free −270 MB, 해제 · settle → +268 MB.
  - 목업 모델 언로드 후 장치 free가 시작 대비 +0 MB.
  - KV-PLAN: 48/512/2048 MB → Vram, 8192 MB → Ssd.
  - KV 계층 강제 VRAM · RAM: CPU 대비 5.7e-6.
- Qwen4-Exp tiny(Metal GDN 경로 포함): max |Δ| 2.69e-5, argmax 100%. suite 195 pass.

**MoltenVK (38499518771)** — Gpu 모드

| 모델 | 전체 | 짧은 구 | 단건 per-text / embed_many | 구성 무관 | 판정 |
|---|---|---|---|---|---|
| 실모델 | 2.20배 | 5.28배 | 49.0 / **58.6 ms** | 12/12 · 12/12 · 160/160 | 150/150, Stage-1 154/154, NMS 4/4, Plinko 25/25 |
| 목업 Host 모드 (비교) | 1.09배 | 1.98배 | 42.4 / 76.9 ms | 동일 | 150/150 |

- 단건 20% 손해를 `51e2256`으로 개선했습니다.
- suite 213 pass. Qwen4-Exp tiny Vulkan max |Δ| 3.00e-5, argmax 100%.

**MoltenVK (38506246088, 포크 `51e2256` · `f947435` 적용)**

| 항목 | 결과 |
|---|---|
| **단건 호출 (실모델, per-text / embed_many)** | **34.9 / 22.5 ms** (이전 49.0 / 58.6 ms) — embed_many가 1.55배 빠름 |
| 전체 · 짧은 구 (실모델) | 2.05배 · 4.36배 |
| 구성 무관 · 재현 · 판정 | 12/12 · 12/12 · 160/160 · 320/320 · 150/150 · Stage-1 154/154 · NMS 4/4 · Plinko 25/25 (유지) |
| `[VK-HOST-SMALL-M]` | 384×1152 · 384×384 · 384×3072 · 1536×384 모두 host == GPU → 적용 |
| Auto 모드 (비교, 목업) | 단건 36.1 / 35.3 ms이지만 구성 무관 1/12 · 1/12 · 121/160, 비트 동일 34/320 → 기본값 불가, 새 ExecMode 불필요 |
| GDN | 텐서 경로 419.6 ms · 2.03 ms → **융합 20.0 ms (20.9배) · 1.22 ms (1.67배)**, CPU 커널 대비 오차 0 |
| FA (Vulkan) | 자가 점검 통과(6.8e-7). FA on/off max |Δlogit| 2.19e-5, top-1 100% |
| TurboQuant KV | 자가 점검 K/V fp8 평균 오차 2.36%. teacher-forcing 평균 KLD 1.66e-4, 최대 1.55e-2, top-1 100% |
| GPU 메모리 | 256 MiB −270 → +268 MB, 모델 언로드 후 +0 MB, KV 계층 강제 VRAM · RAM 오차 0 |
| Qwen4-Exp · suite | max |Δ| 3.10e-5, argmax 100% · 213 pass |

**Metal GDN 정밀화 (38508388995, `806aba6`)**

- 텐서 경로 190.5 ms · 10.81 ms → **융합 27.0 ms (7.06배) · 2.09 ms (5.18배)**. CPU 대비 오차 ≤ 7.3e-7, 자가 점검 3.3e-7.
- Qwen4-Exp tiny(Metal GDN 경로): max |Δ| 2.9e-5 · 3.4e-5, argmax 100%.

**CPU (38499518771)**: 실모델 · 동적

| 항목 | 결과 |
|---|---|
| 전체 | 0.94배 |
| 짧은 구 | 1.57배 |
| 단건 | 44.4 / 44.9 ms |
| 비트 동일 | 320/320 |
| 판정 | 150/150 |
| suite | 194 pass |
| SSD 오프로드 콜드 읽기 | std::fs 399 MB/s → 병렬 F_NOCACHE 10.1 GB/s, 캐시 증가 +0 MB |

**개선 4건 (38513842101 · 38516449786 · 38520324489, 포크 master + `a633c43`)**

| 항목 | 결과 |
|---|---|
| Metal GDN (`a633c43`) | 융합 프리필 23.5 ms · 디코드 1.15 ms/스텝, 같은 잡 텐서 경로 대비 7.15배 · 4.97배, 오차 ≤ 7.3e-7 |
| MoltenVK GDN | 융합 프리필 18.9 ms · 디코드 0.95 ms/스텝, 텐서 경로 대비 38.78배 · 3.65배, CPU 커널과 오차 0 |
| 앱 CPU 실모델 (병렬 원소 연산, 같은 비트) | 전체 0.94배 → 1.39배, 짧은 구 1.57배 → 3.00배, 단건 중앙값 87.9 / 76.8 ms(손해 없음), 비트 동일 320/320, 판정 150/150 |
| 목업 Qwen3.5 KV (Metal · MoltenVK) | Ram↔Vram 차이 0, GDN 상태 장치 유지, SSD 계층 top-1 4/4, TurboQuant 압축 KV(MoltenVK) top-1 4/4 |
| 모델 언로드 (Metal · MoltenVK) | 장치 free 시작 대비 +0 MB, RSS 둘째 사이클 +23 · +27 MB |
| suite | Metal 196, MoltenVK 214, CPU 195 pass |

---

## 4. 리눅스 테스트 결과 (클라우드 VM)

| 항목 | 결과 |
|---|---|
| CPU M-불변성 (x86 AVX-512) | `matmul_m_invariance_tests`: M≥2 불일치 0, M=1만 다름(gemv). `RAYON_NUM_THREADS` 1~4 digest `20e4a1376e76b382` 동일 |
| lavapipe Vulkan | `fixed_exec_modes_are_batch_invariant` OK. Gpu/Host/Auto M≥2 불일치 0 (Auto만 M=1 다름) |
| FA/TQ teacher-forcing (목업 Qwen3) | CPU FA on: KLD 9.6e-14, top-1 100%, max|Δ| 2.3e-5. lavapipe FA on: KLD 5.7e-14, top-1 100%. **TurboQuant KV on (lavapipe): 평균 KLD 1.66e-4, 최대 1.55e-2, top-1 100%, max|Δlogit| 0.80** (손실 압축, CPU에서는 비활성) |
| GDN (lavapipe, 포크 수정 전) | CPU 융합 프리필 T=256 60.0 ms · 디코드 0.98 ms/스텝 vs Vulkan 텐서 경로 661 ms · 3.09 ms (0.09배 · 0.32배) → `f947435`의 근거 |
| GPU 메모리 (lavapipe) | KV 계층 강제 VRAM · RAM: CPU 대비 3.8e-6. 반환 판정은 CPU 장치라 생략 |
| 앱 suite | `cargo test --test it --no-default-features` 193 pass (기본 · packed) |
| 컴파일 검사 | 포크 CPU · `--features vulkan` · macOS 대상 `--features metal` (candle-core, candle-nn) 오류 없음 |

---

## 5. 상태

- [x] 38506246088 app (vulkan): 전부 통과 (3-2절).
- [x] Metal GDN 정밀화 · 빈 버퍼 수정: app Metal 통과(38508388995), fork Metal `delta_rule_tests` 7 pass(38508978460).
- 이 브랜치의 수정은 모두 macOS에서 검증됐습니다.
- 포크 master 반영(PR)은 저장소 관리자가 정합니다. 앱의 git 의존(Qntology master)에 들어가야 CI 밖 빌드에도 반영됩니다.
- [x] PR #11로 `c663a23`~`e4a7314`가 master에 반영됨. 남은 브랜치 커밋은 `a633c43` 하나(fork Metal 7 pass, 38513842101).
- [x] 개선 4건 Mac 검증 완료(38513842101 · 38516449786 · 38520324489, 3-2절). FP8 강등 · VRAM 재압축 · kv_half는 CUDA/ROCm 전용이라 Mac · 리눅스 VM에서 실행할 수 없습니다.
- GDN 디코드(가상 GPU): `a633c43`으로 스텝당 디스패치를 2개로 줄였습니다. 가상 GPU의 동기화 비용은 실제 Apple GPU와 다르므로, 속도 판단은 셀프호스티드 Apple Silicon 러너에서 합니다(logis-center `mac-verify.yml`의 `runner` 입력에 그 라벨).
