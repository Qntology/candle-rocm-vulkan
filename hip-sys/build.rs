//! Links the HIP runtime and rocBLAS, and picks the ROCm installation that the dependent crates
//! build against.
//!
//! Two kinds of installation ("tracks") are supported side by side. Both ship HIP 7.x with the
//! same `amdhip64` / `rocblas` ABI, so the FFI and the backend code are shared:
//!
//! * `legacy`: ROCm / AMD HIP SDK 6.x - 7.2 (`/opt/rocm-7.x`, `C:\Program Files\AMD\ROCm\7.x`).
//! * `core`: ROCm Core SDK built by TheRock, HIP 7.10 and later (ROCm 10.1 ships HIP 7.16):
//!   `/opt/rocm/core-10.x`, tarballs, `rocm-sdk` Python wheels, `...\AMD\ROCm\core-10.x`.
//!
//! `ROCM_PATH` / `HIP_PATH` always win. Otherwise `CANDLE_ROCM_TRACK=auto|legacy|core` picks the
//! newest installation of a track. `auto` (the default) takes `core` when every detected GPU is
//! supported by the ROCm Core SDK and `legacy` otherwise.
//!
//! The choice is exported to dependents (`links = "amdhip64"`) as `DEP_AMDHIP64_ROOT`,
//! `DEP_AMDHIP64_TRACK`, `DEP_AMDHIP64_HIP_VERSION` and `DEP_AMDHIP64_ARCHS`, so `candle-core`
//! compiles its kernels with the installation that is linked here.

use std::path::{Path, PathBuf};
use std::process::Command;

/// GPU targets supported by the ROCm Core SDK 10.1 release (Instinct, Radeon and Ryzen).
const CORE_TRACK_ARCHS: &[&str] = &[
    "gfx908", "gfx90a", "gfx942", "gfx950", "gfx1030", "gfx1100", "gfx1101", "gfx1102", "gfx1103",
    "gfx1150", "gfx1151", "gfx1152", "gfx1153", "gfx1200", "gfx1201",
];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Track {
    Legacy,
    Core,
}

impl Track {
    fn name(self) -> &'static str {
        match self {
            Track::Legacy => "legacy",
            Track::Core => "core",
        }
    }
}

struct Install {
    root: PathBuf,
    hip: Option<[u32; 3]>,
    track: Track,
}

impl Install {
    fn new(root: PathBuf) -> Self {
        let hip = hip_version(&root);
        let track = classify(&root, hip);
        Self { root, hip, track }
    }

    fn sort_key(&self) -> (Vec<u32>, Vec<u32>) {
        (self.hip.map(|v| v.to_vec()).unwrap_or_default(), version_key(&self.root))
    }
}

struct Selection {
    install: Install,
    archs: Vec<String>,
    note: Option<String>,
}

fn clean_path(v: &str) -> PathBuf {
    PathBuf::from(v.trim().trim_matches('"').trim_end_matches(['\\', '/']))
}

fn env_dir(var: &str) -> Option<PathBuf> {
    let v = std::env::var(var).ok()?;
    if v.trim().is_empty() {
        return None;
    }
    let p = clean_path(&v);
    p.is_dir().then_some(p)
}

fn version_key(p: &Path) -> Vec<u32> {
    p.file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .split(|c: char| !c.is_ascii_digit())
        .filter(|s| !s.is_empty())
        .filter_map(|s| s.parse::<u32>().ok())
        .collect()
}

fn leading_u32(s: &str) -> Option<u32> {
    let digits: String = s.trim().chars().take_while(|c| c.is_ascii_digit()).collect();
    digits.parse().ok()
}

/// HIP version of an installation, from `include/hip/hip_version.h` (or the version files of
/// runtime-only installs).
fn hip_version(root: &Path) -> Option<[u32; 3]> {
    fn slot(name: &str) -> Option<usize> {
        match name {
            "HIP_VERSION_MAJOR" => Some(0),
            "HIP_VERSION_MINOR" => Some(1),
            "HIP_VERSION_PATCH" => Some(2),
            _ => None,
        }
    }
    let mut v: [Option<u32>; 3] = [None; 3];
    if let Ok(text) = std::fs::read_to_string(root.join("include").join("hip").join("hip_version.h")) {
        for line in text.lines() {
            let mut it = line.split_whitespace();
            if it.next() != Some("#define") {
                continue;
            }
            if let (Some(i), Some(val)) = (it.next().and_then(slot), it.next()) {
                v[i] = leading_u32(val);
            }
        }
    }
    if v[0].is_none() {
        for file in [root.join("share").join("hip").join("version"), root.join("bin").join(".hipVersion")] {
            let Ok(text) = std::fs::read_to_string(file) else { continue };
            for line in text.lines() {
                if let Some((k, val)) = line.split_once('=') {
                    if let Some(i) = slot(k.trim()) {
                        v[i] = leading_u32(val);
                    }
                }
            }
            if v[0].is_some() {
                break;
            }
        }
    }
    Some([v[0]?, v[1].unwrap_or(0), v[2].unwrap_or(0)])
}

/// HIP 7.10 and later come from the ROCm Core SDK (TheRock); older ones from the legacy stream.
fn classify(root: &Path, hip: Option<[u32; 3]>) -> Track {
    if let Some([major, minor, _]) = hip {
        return if major > 7 || (major == 7 && minor >= 10) {
            Track::Core
        } else {
            Track::Legacy
        };
    }
    let name = root
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let major = version_key(root).first().copied().unwrap_or(0);
    if name.starts_with("core") || name.starts_with("runtime") || major >= 10 {
        Track::Core
    } else {
        Track::Legacy
    }
}

fn has_link_libs(root: &Path, windows: bool) -> bool {
    let (hip, blas) = if windows {
        ("amdhip64.lib", "rocblas.lib")
    } else {
        ("libamdhip64.so", "librocblas.so")
    };
    let dirs = [root.join("lib"), root.join("lib64")];
    dirs.iter().any(|d| d.join(hip).is_file()) && dirs.iter().any(|d| d.join(blas).is_file())
}

fn find_in_path(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|dir| dir.join(name))
        .find(|candidate| candidate.is_file())
}

/// Root of a pip-installed ROCm Core SDK (`rocm-sdk path --root`).
fn rocm_sdk_root(windows: bool) -> Option<PathBuf> {
    let exe = find_in_path(if windows { "rocm-sdk.exe" } else { "rocm-sdk" })?;
    let out = Command::new(exe).args(["path", "--root"]).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    let p = clean_path(stdout.lines().next()?);
    p.is_dir().then_some(p)
}

fn candidate_roots(windows: bool) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    if let Some(p) = rocm_sdk_root(windows) {
        out.push(p);
    }
    if windows {
        let program_files = std::env::var("ProgramFiles")
            .map(|s| clean_path(&s))
            .unwrap_or_else(|_| PathBuf::from(r"C:\Program Files"));
        if let Ok(entries) = std::fs::read_dir(program_files.join("AMD").join("ROCm")) {
            out.extend(entries.flatten().map(|e| e.path()));
        }
    } else {
        out.push(PathBuf::from("/opt/rocm"));
        out.push(PathBuf::from("/opt/rocm/core"));
        for (dir, prefix) in [("/opt/rocm", "core-"), ("/opt", "rocm-")] {
            if let Ok(entries) = std::fs::read_dir(dir) {
                for e in entries.flatten() {
                    if e.file_name().to_str().is_some_and(|n| n.starts_with(prefix)) {
                        out.push(e.path());
                    }
                }
            }
        }
    }
    let mut seen: Vec<PathBuf> = Vec::new();
    out.retain(|p| {
        if !p.is_dir() {
            return false;
        }
        let canonical = std::fs::canonicalize(p).unwrap_or_else(|_| p.clone());
        if seen.contains(&canonical) {
            false
        } else {
            seen.push(canonical);
            true
        }
    });
    out
}

fn parse_archs(text: &str) -> Vec<String> {
    let mut archs: Vec<String> = Vec::new();
    for token in text.split(|c: char| !(c.is_ascii_alphanumeric() || c == ':' || c == '+' || c == '-')) {
        let base = token.trim().split(':').next().unwrap_or("");
        if base.starts_with("gfx") && base.len() > 3 && base != "gfx000" && !archs.iter().any(|a| a == base) {
            archs.push(base.to_string());
        }
    }
    archs
}

/// Explicit GPU list from `CANDLE_ROCM_ARCHS` / `HIP_ARCH` (keywords such as `all` are ignored).
fn env_archs() -> Vec<String> {
    for var in ["CANDLE_ROCM_ARCHS", "HIP_ARCH"] {
        if let Ok(v) = std::env::var(var) {
            let archs = parse_archs(&v);
            if !archs.is_empty() {
                return archs;
            }
        }
    }
    Vec::new()
}

fn run_arch_tool(tool: &Path, bin: Option<&Path>) -> Vec<String> {
    let mut cmd = Command::new(tool);
    if let Some(bin) = bin {
        // Let the tool find the HIP runtime (amdhip64_*.dll / libamdhip64.so) of its installation.
        let mut paths = vec![bin.to_path_buf()];
        if let Some(p) = std::env::var_os("PATH") {
            paths.extend(std::env::split_paths(&p));
        }
        if let Ok(joined) = std::env::join_paths(paths) {
            cmd.env("PATH", joined);
        }
    }
    match cmd.output() {
        Ok(out) => parse_archs(&String::from_utf8_lossy(&out.stdout)),
        Err(_) => Vec::new(),
    }
}

/// gfx targets of the GPUs in this machine, asked from the tools of the installations.
fn detect_gpu_archs(installs: &[Install], windows: bool) -> Vec<String> {
    const TOOLS: [&str; 4] = ["amdgpu-arch", "offload-arch", "rocm_agent_enumerator", "hipInfo"];
    let exe = |n: &str| if windows { format!("{n}.exe") } else { n.to_string() };
    for install in installs {
        let bin = install.root.join("bin");
        let dirs = [
            install.root.join("lib").join("llvm").join("bin"),
            install.root.join("llvm").join("bin"),
            bin.clone(),
        ];
        for dir in dirs.iter() {
            for tool in TOOLS {
                let path = dir.join(exe(tool));
                if path.is_file() {
                    let archs = run_arch_tool(&path, Some(&bin));
                    if !archs.is_empty() {
                        return archs;
                    }
                }
            }
        }
    }
    for tool in TOOLS {
        if let Some(path) = find_in_path(&exe(tool)) {
            let archs = run_arch_tool(&path, None);
            if !archs.is_empty() {
                return archs;
            }
        }
    }
    Vec::new()
}

fn requested_track() -> Option<Track> {
    let v = std::env::var("CANDLE_ROCM_TRACK").unwrap_or_default();
    match v.trim().to_ascii_lowercase().as_str() {
        "" | "auto" => None,
        "legacy" => Some(Track::Legacy),
        "core" | "therock" => Some(Track::Core),
        other => {
            println!(
                "cargo:warning=hip-sys: unknown CANDLE_ROCM_TRACK={other:?} (expected auto, legacy or core); using auto"
            );
            None
        }
    }
}

fn select(windows: bool) -> Option<Selection> {
    let forced = requested_track();
    let listed = env_archs();

    for var in ["ROCM_PATH", "HIP_PATH"] {
        if let Some(root) = env_dir(var) {
            let install = Install::new(root);
            let note = match forced {
                Some(t) if t != install.track => Some(format!(
                    "{var} points to a {} ROCm installation while CANDLE_ROCM_TRACK={}; using {var}",
                    install.track.name(),
                    t.name()
                )),
                _ => None,
            };
            return Some(Selection { install, archs: listed, note });
        }
    }

    let mut installs: Vec<Install> = candidate_roots(windows)
        .into_iter()
        .map(Install::new)
        .filter(|i| has_link_libs(&i.root, windows))
        .collect();
    installs.sort_by_key(|i| std::cmp::Reverse(i.sort_key()));
    if installs.is_empty() {
        if forced.is_some() {
            panic!("hip-sys: CANDLE_ROCM_TRACK is set but no ROCm installation was found; set ROCM_PATH");
        }
        return None;
    }
    let newest = |t: Track| installs.iter().position(|i| i.track == t);

    let (index, archs, note) = match forced {
        Some(t) => match newest(t) {
            Some(i) => (i, listed, None),
            None => panic!(
                "hip-sys: CANDLE_ROCM_TRACK={} but no such ROCm installation was found (found: {}); \
                 install it or set ROCM_PATH",
                t.name(),
                installs
                    .iter()
                    .map(|i| format!("{} [{}]", i.root.display(), i.track.name()))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        },
        None => match (newest(Track::Core), newest(Track::Legacy)) {
            (Some(core), Some(legacy)) => {
                let archs = if listed.is_empty() {
                    detect_gpu_archs(&installs, windows)
                } else {
                    listed
                };
                let unsupported: Vec<&str> = archs
                    .iter()
                    .map(String::as_str)
                    .filter(|a| !CORE_TRACK_ARCHS.contains(a))
                    .collect();
                if unsupported.is_empty() {
                    (core, archs, None)
                } else {
                    let note = format!(
                        "{} is not supported by the ROCm Core SDK at {}, using the legacy ROCm at {} \
                         (set CANDLE_ROCM_TRACK=core to override)",
                        unsupported.join(","),
                        installs[core].root.display(),
                        installs[legacy].root.display()
                    );
                    (legacy, archs, Some(note))
                }
            }
            (Some(i), None) | (None, Some(i)) => (i, listed, None),
            (None, None) => unreachable!("installs is not empty"),
        },
    };
    let install = installs.swap_remove(index);
    Some(Selection { install, archs, note })
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    for var in [
        "ROCM_PATH",
        "HIP_PATH",
        "CANDLE_ROCM_LIB_DIR",
        "CANDLE_ROCM_TRACK",
        "CANDLE_ROCM_ARCHS",
        "HIP_ARCH",
    ] {
        println!("cargo:rerun-if-env-changed={var}");
    }

    let windows = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default() == "windows";
    let selection = select(windows);

    let mut lib_dirs: Vec<PathBuf> = Vec::new();
    if let Ok(dir) = std::env::var("CANDLE_ROCM_LIB_DIR") {
        for d in std::env::split_paths(&dir) {
            lib_dirs.push(d);
        }
    }
    if let Some(sel) = &selection {
        lib_dirs.push(sel.install.root.join("lib"));
        lib_dirs.push(sel.install.root.join("lib64"));
    }
    for dir in lib_dirs.iter().filter(|d| d.exists()) {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }

    println!("cargo:rustc-link-lib=dylib=amdhip64");
    println!("cargo:rustc-link-lib=dylib=rocblas");
    if std::env::var("CARGO_FEATURE_HIPRAND").is_ok() {
        println!("cargo:rustc-link-lib=dylib=hiprand");
    }

    if let Some(sel) = selection {
        println!("cargo:root={}", sel.install.root.display());
        println!("cargo:track={}", sel.install.track.name());
        if let Some([major, minor, patch]) = sel.install.hip {
            println!("cargo:hip_version={major}.{minor}.{patch}");
        }
        if !sel.archs.is_empty() {
            println!("cargo:archs={}", sel.archs.join(","));
        }
        if let Some(note) = sel.note {
            println!("cargo:warning=hip-sys: {note}");
        }
    }
}
