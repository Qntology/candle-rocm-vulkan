//! Opt-in profiler of the Vulkan backend, enabled with `CANDLE_VULKAN_PROFILE=1`.
//!
//! Every backend entry point opens a [`Scope`] named `category/name`; on drop the
//! scope adds its call, its inclusive wall time and an optional work counter
//! (bytes moved, or flops for matmuls) to a process wide table. Time spent in
//! *top-level* scopes (not nested inside another scope on the same thread) is
//! also summed on its own, so a report tells the time spent inside the backend
//! apart from the time the caller spent between backend calls.
//!
//! With the variable unset the cost of a scope is one relaxed atomic load.

use std::cell::Cell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Mutex;
use std::time::Instant;

/// 0: not decided yet, 1: off, 2: on.
static STATE: AtomicU8 = AtomicU8::new(0);

#[derive(Default, Clone, Copy)]
struct Entry {
    calls: u64,
    nanos: u128,
    work: u64,
}

#[derive(Default)]
struct Table {
    ops: HashMap<(&'static str, &'static str), Entry>,
    top_nanos: u128,
    top_calls: u64,
    since: Option<Instant>,
}

static TABLE: Mutex<Option<Table>> = Mutex::new(None);

thread_local! {
    static DEPTH: Cell<u32> = const { Cell::new(0) };
}

/// Whether profiling is on (`CANDLE_VULKAN_PROFILE`, read once, or [`set_enabled`]).
pub fn enabled() -> bool {
    match STATE.load(Ordering::Relaxed) {
        1 => false,
        2 => true,
        _ => {
            let on = std::env::var("CANDLE_VULKAN_PROFILE")
                .map(|v| {
                    let v = v.trim().to_ascii_lowercase();
                    !(v.is_empty() || v == "0" || v == "false" || v == "off" || v == "no")
                })
                .unwrap_or(false);
            STATE.store(if on { 2 } else { 1 }, Ordering::Relaxed);
            on
        }
    }
}

/// Turns profiling on or off at run time.
pub fn set_enabled(on: bool) {
    STATE.store(if on { 2 } else { 1 }, Ordering::Relaxed);
}

/// A timed region; records itself when dropped. Bind it to a named variable
/// (`let _p = scope(..)`), `let _ = scope(..)` would end it immediately.
pub(crate) struct Scope {
    key: (&'static str, &'static str),
    t0: Option<Instant>,
    work: u64,
}

#[inline]
pub(crate) fn scope(cat: &'static str, name: &'static str) -> Scope {
    if !enabled() {
        return Scope {
            key: (cat, name),
            t0: None,
            work: 0,
        };
    }
    DEPTH.with(|d| d.set(d.get() + 1));
    Scope {
        key: (cat, name),
        t0: Some(Instant::now()),
        work: 0,
    }
}

impl Scope {
    /// Adds to the work counter of this scope (bytes or flops).
    #[inline]
    pub(crate) fn work(&mut self, w: usize) {
        self.work = self.work.saturating_add(w as u64);
    }

    /// Re-labels the scope, e.g. once the code path taken is known.
    #[inline]
    pub(crate) fn rename(&mut self, cat: &'static str, name: &'static str) {
        self.key = (cat, name);
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        let Some(t0) = self.t0 else { return };
        let nanos = t0.elapsed().as_nanos();
        let top = DEPTH.with(|d| {
            let v = d.get().saturating_sub(1);
            d.set(v);
            v == 0
        });
        if let Ok(mut g) = TABLE.lock() {
            let t = g.get_or_insert_with(Table::default);
            if t.since.is_none() {
                t.since = Some(t0);
            }
            let e = t.ops.entry(self.key).or_default();
            e.calls += 1;
            e.nanos += nanos;
            e.work = e.work.saturating_add(self.work);
            if top {
                t.top_nanos += nanos;
                t.top_calls += 1;
            }
        }
    }
}

fn human(w: u64) -> String {
    let f = w as f64;
    if w == 0 {
        "-".to_string()
    } else if f >= 1e9 {
        format!("{:.2}G", f / 1e9)
    } else if f >= 1e6 {
        format!("{:.1}M", f / 1e6)
    } else if f >= 1e3 {
        format!("{:.1}K", f / 1e3)
    } else {
        format!("{w}")
    }
}

/// The table sorted by inclusive time (at most `top` rows), `None` when profiling
/// is off. `reset` clears the counters afterwards.
pub fn report(top: usize, reset: bool) -> Option<String> {
    if !enabled() {
        return None;
    }
    let mut g = TABLE.lock().ok()?;
    let t = g.get_or_insert_with(Table::default);
    let window_ms = t
        .since
        .map(|s| s.elapsed().as_secs_f64() * 1e3)
        .unwrap_or(0.0);
    let mut rows: Vec<((&'static str, &'static str), Entry)> =
        t.ops.iter().map(|(k, e)| (*k, *e)).collect();
    rows.sort_by(|a, b| b.1.nanos.cmp(&a.1.nanos));
    let mut s = format!(
        "  backend (top-level) {:.0} ms in {} calls | window {:.0} ms | {} distinct ops (inclusive times, nested scopes overlap)\n",
        t.top_nanos as f64 / 1e6,
        t.top_calls,
        window_ms,
        rows.len()
    );
    s.push_str("        calls     total ms     avg us       work  op\n");
    for ((cat, name), e) in rows.into_iter().take(top) {
        s.push_str(&format!(
            "  {:>11} {:>12.1} {:>10.1} {:>10}  {}/{}\n",
            e.calls,
            e.nanos as f64 / 1e6,
            e.nanos as f64 / 1e3 / e.calls.max(1) as f64,
            human(e.work),
            cat,
            name
        ));
    }
    if reset {
        *t = Table::default();
    }
    Some(s)
}

/// Clears the counters.
pub fn reset() {
    if let Ok(mut g) = TABLE.lock() {
        *g = None;
    }
}
