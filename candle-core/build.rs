use std::path::{Path, PathBuf};
use std::process::Command;

const KERNELS: &[&str] = &[
    "unary",
    "binary",
    "cast",
    "fill",
    "indexing",
    "reduce",
    "ternary",
    "nn",
    "quantized",
    "gemm",
];

/// Targets used when no GPU is detected, per ROCm track (see `hip-sys/build.rs`).
/// `legacy` (ROCm / HIP SDK 6.x - 7.2) keeps the RDNA2 / RDNA3 set used so far.
const LEGACY_DEFAULT_ARCHS: &[&str] = &["gfx1030", "gfx1100", "gfx1101", "gfx1102"];
/// `core` (ROCm Core SDK 10.x): every Radeon and Ryzen target of ROCm 10.1.
const CORE_DEFAULT_ARCHS: &[&str] = &[
    "gfx1030", "gfx1100", "gfx1101", "gfx1102", "gfx1103", "gfx1150", "gfx1151", "gfx1152", "gfx1153",
    "gfx1200", "gfx1201",
];
/// `CANDLE_ROCM_ARCHS=all`: every target of ROCm 10.1, Instinct included.
const ALL_ARCHS: &[&str] = &[
    "gfx908", "gfx90a", "gfx942", "gfx950", "gfx1030", "gfx1100", "gfx1101", "gfx1102", "gfx1103",
    "gfx1150", "gfx1151", "gfx1152", "gfx1153", "gfx1200", "gfx1201",
];

fn clean_path(v: &str) -> PathBuf {
    PathBuf::from(v.trim().trim_matches('"').trim_end_matches(['\\', '/']))
}

fn env_path(var: &str) -> Option<PathBuf> {
    let v = std::env::var(var).ok()?;
    if v.trim().is_empty() {
        return None;
    }
    let p = clean_path(&v);
    p.exists().then_some(p)
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

/// Root picked by `hip-sys` (exported as `DEP_AMDHIP64_ROOT`), else `ROCM_PATH` / `HIP_PATH`, else
/// the newest installation in the platform default location.
fn rocm_root() -> Option<PathBuf> {
    if let Some(p) = env_path("DEP_AMDHIP64_ROOT") {
        return Some(p);
    }
    if let Some(p) = env_path("ROCM_PATH").or_else(|| env_path("HIP_PATH")) {
        return Some(p);
    }
    if cfg!(windows) {
        let program_files = std::env::var("ProgramFiles")
            .map(|s| clean_path(&s))
            .unwrap_or_else(|_| PathBuf::from(r"C:\Program Files"));
        let mut versions: Vec<PathBuf> = std::fs::read_dir(program_files.join("AMD").join("ROCm"))
            .ok()?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.join("bin").exists())
            .collect();
        versions.sort_by_key(|p| version_key(p));
        versions.pop()
    } else {
        let p = PathBuf::from("/opt/rocm");
        p.exists().then_some(p)
    }
}

/// `legacy` or `core`, as decided by `hip-sys`, else from the HIP version of `root`.
fn rocm_track(root: Option<&Path>) -> &'static str {
    match std::env::var("DEP_AMDHIP64_TRACK").as_deref() {
        Ok("core") => return "core",
        Ok("legacy") => return "legacy",
        _ => {}
    }
    match root.and_then(hip_version) {
        Some((major, minor)) if major > 7 || (major == 7 && minor >= 10) => "core",
        Some(_) => "legacy",
        None => {
            let name = root
                .and_then(|r| r.file_name())
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            let major = root.map(version_key).and_then(|v| v.first().copied()).unwrap_or(0);
            if name.starts_with("core") || major >= 10 {
                "core"
            } else {
                "legacy"
            }
        }
    }
}

fn hip_version(root: &Path) -> Option<(u32, u32)> {
    let text = std::fs::read_to_string(root.join("include").join("hip").join("hip_version.h")).ok()?;
    let (mut major, mut minor) = (None, None);
    for line in text.lines() {
        let mut it = line.split_whitespace();
        if it.next() != Some("#define") {
            continue;
        }
        match (it.next(), it.next()) {
            (Some("HIP_VERSION_MAJOR"), Some(v)) => major = v.parse().ok(),
            (Some("HIP_VERSION_MINOR"), Some(v)) => minor = v.parse().ok(),
            _ => {}
        }
    }
    Some((major?, minor.unwrap_or(0)))
}

fn find_in_path(names: &[&str]) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        for name in names {
            let candidate = dir.join(name);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

enum Compiler {
    Hipcc(PathBuf),
    Clang(PathBuf),
}

impl Compiler {
    fn path(&self) -> &Path {
        match self {
            Compiler::Hipcc(p) | Compiler::Clang(p) => p,
        }
    }
}

/// Directories that may hold the LLVM tools: legacy HIP SDK (`bin`), legacy Linux ROCm
/// (`llvm/bin`) and the ROCm Core SDK (`lib/llvm/bin`).
fn llvm_dirs(root: &Path) -> [PathBuf; 3] {
    [
        root.join("bin"),
        root.join("llvm").join("bin"),
        root.join("lib").join("llvm").join("bin"),
    ]
}

fn find_compiler(root: Option<&Path>) -> Option<Compiler> {
    if let Some(p) = env_path("HIPCC") {
        return Some(Compiler::Hipcc(p));
    }
    let hipcc_names: &[&str] = if cfg!(windows) {
        &["hipcc.exe", "hipcc.bin.exe"]
    } else {
        &["hipcc"]
    };
    let clang_names: &[&str] = if cfg!(windows) {
        &["clang++.exe"]
    } else {
        &["clang++"]
    };
    if let Some(root) = root {
        for name in hipcc_names {
            let p = root.join("bin").join(name);
            if p.is_file() {
                return Some(Compiler::Hipcc(p));
            }
        }
        for dir in llvm_dirs(root) {
            for name in clang_names {
                let p = dir.join(name);
                if p.is_file() {
                    return Some(Compiler::Clang(p));
                }
            }
        }
    }
    find_in_path(hipcc_names).map(Compiler::Hipcc)
}

fn split_archs(s: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for a in s.split([',', ';', ' ']).map(|a| a.trim()) {
        if a.starts_with("gfx") && a != "gfx000" && !out.iter().any(|x| x == a) {
            out.push(a.to_string());
        }
    }
    out
}

fn parse_tool_output(text: &str, archs: &mut Vec<String>) {
    for token in text.split(|c: char| !(c.is_ascii_alphanumeric() || c == ':' || c == '+' || c == '-')) {
        let token = token.trim();
        if token.starts_with("gfx") && token != "gfx000" {
            let base = token.split(':').next().unwrap_or(token).to_string();
            if base.len() > 3 && !archs.contains(&base) {
                archs.push(base);
            }
        }
    }
}

fn detect_archs(root: Option<&Path>) -> Vec<String> {
    let mut tools: Vec<PathBuf> = Vec::new();
    let exe = |n: &str| {
        if cfg!(windows) {
            format!("{n}.exe")
        } else {
            n.to_string()
        }
    };
    if let Some(root) = root {
        for dir in llvm_dirs(root) {
            for n in ["amdgpu-arch", "offload-arch", "rocm_agent_enumerator", "hipInfo"] {
                tools.push(dir.join(exe(n)));
            }
        }
    }
    for n in ["amdgpu-arch", "offload-arch", "rocm_agent_enumerator"] {
        if let Some(p) = find_in_path(&[exe(n).as_str()]) {
            tools.push(p);
        }
    }
    let mut archs: Vec<String> = Vec::new();
    for tool in tools.iter().filter(|t| t.is_file()) {
        let mut cmd = Command::new(tool);
        if let Some(root) = root {
            // Let the tool load the HIP runtime of the same installation.
            let mut paths = vec![root.join("bin")];
            if let Some(p) = std::env::var_os("PATH") {
                paths.extend(std::env::split_paths(&p));
            }
            if let Ok(joined) = std::env::join_paths(paths) {
                cmd.env("PATH", joined);
            }
        }
        let Ok(out) = cmd.output() else {
            continue;
        };
        parse_tool_output(&String::from_utf8_lossy(&out.stdout), &mut archs);
        if !archs.is_empty() {
            break;
        }
    }
    archs
}

fn base_command(compiler: &Compiler, root: Option<&Path>) -> Command {
    match compiler {
        Compiler::Hipcc(p) => {
            let mut c = Command::new(p);
            c.arg("--genco");
            c
        }
        Compiler::Clang(p) => {
            let mut c = Command::new(p);
            c.args(["-x", "hip", "--cuda-device-only"]);
            if let Some(root) = root {
                c.arg(format!("--rocm-path={}", root.display()));
            }
            c
        }
    }
}

/// Drops the targets that the compiler does not know (only for the built-in target lists, an
/// older ROCm may predate some of them). A user supplied list is used as is.
fn supported_archs(compiler: &Compiler, root: Option<&Path>, out_dir: &Path, archs: Vec<String>) -> Vec<String> {
    let probe = out_dir.join("candle_arch_probe.hip");
    if std::fs::write(&probe, "extern \"C\" __global__ void candle_arch_probe() {}\n").is_err() {
        return archs;
    }
    let mut archs = archs;
    for _ in 0..archs.len() {
        let mut cmd = base_command(compiler, root);
        for a in archs.iter() {
            cmd.arg(format!("--offload-arch={a}"));
        }
        cmd.arg("-fsyntax-only").arg(&probe);
        let Ok(out) = cmd.output() else {
            return archs;
        };
        if out.status.success() {
            return archs;
        }
        let stderr = String::from_utf8_lossy(&out.stderr).to_string();
        let rejected: Vec<String> = archs
            .iter()
            .filter(|a| {
                stderr.lines().any(|l| {
                    let l = l.to_ascii_lowercase();
                    (l.contains("unsupported") || l.contains("invalid") || l.contains("unknown"))
                        && l.split(|c: char| !c.is_ascii_alphanumeric()).any(|w| w == a.as_str())
                })
            })
            .cloned()
            .collect();
        if rejected.is_empty() {
            return archs;
        }
        println!(
            "cargo:warning=candle-core: {} does not support {}; skipping",
            compiler.path().display(),
            rejected.join(",")
        );
        archs.retain(|a| !rejected.contains(a));
        if archs.is_empty() {
            return archs;
        }
    }
    archs
}

/// The output must be something `hipModuleLoadData` accepts: an offload bundle (compressed or
/// not) or a single code object.
fn is_loadable_image(path: &Path) -> bool {
    let Ok(bytes) = std::fs::read(path) else {
        return false;
    };
    bytes.starts_with(b"__CLANG_OFFLOAD_BUNDLE__") || bytes.starts_with(b"CCOB") || bytes.starts_with(b"\x7fELF")
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var("CARGO_FEATURE_ROCM").is_err() {
        return;
    }
    for var in [
        "ROCM_PATH",
        "HIP_PATH",
        "HIPCC",
        "HIP_ARCH",
        "CANDLE_ROCM_ARCHS",
        "CANDLE_ROCM_TRACK",
        "CANDLE_ROCM_SKIP_KERNEL_BUILD",
        "CANDLE_ROCM_HIPCC_FLAGS",
    ] {
        println!("cargo:rerun-if-env-changed={var}");
    }
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let kernel_dir = manifest_dir.join("kernels");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    println!("cargo:rerun-if-changed={}", kernel_dir.join("common.h").display());
    for k in KERNELS {
        println!("cargo:rerun-if-changed={}", kernel_dir.join(format!("{k}.hip")).display());
    }

    let root = rocm_root();
    let track = rocm_track(root.as_deref());
    let hip_version = std::env::var("DEP_AMDHIP64_HIP_VERSION")
        .ok()
        .filter(|v| !v.is_empty())
        .or_else(|| root.as_deref().and_then(hip_version).map(|(a, b)| format!("{a}.{b}")))
        .unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=CANDLE_ROCM_TRACK={track}");
    println!("cargo:rustc-env=CANDLE_ROCM_BUILD_HIP_VERSION={hip_version}");

    let skip = std::env::var("CANDLE_ROCM_SKIP_KERNEL_BUILD")
        .map(|v| !v.is_empty() && v != "0")
        .unwrap_or(false);
    if skip {
        for k in KERNELS {
            std::fs::write(out_dir.join(format!("{k}.co")), b"").unwrap();
        }
        println!("cargo:rustc-env=CANDLE_ROCM_ARCHS=none");
        println!("cargo:warning=candle-core: ROCm kernels were not compiled (CANDLE_ROCM_SKIP_KERNEL_BUILD)");
        return;
    }

    let compiler = find_compiler(root.as_deref()).unwrap_or_else(|| {
        panic!(
            "candle-core `rocm` feature: hipcc was not found. Install ROCm (legacy HIP SDK 6.x/7.x or \
             the ROCm Core SDK 10.x) and set ROCM_PATH or HIP_PATH (Windows: e.g. \
             C:\\Program Files\\AMD\\ROCm\\7.2, or the folder where the ROCm 10.x tarball was \
             extracted), or set HIPCC. Set CANDLE_ROCM_SKIP_KERNEL_BUILD=1 to only type-check."
        )
    });

    let requested = std::env::var("CANDLE_ROCM_ARCHS")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .or_else(|| std::env::var("HIP_ARCH").ok().filter(|s| !s.trim().is_empty()))
        .unwrap_or_default();
    let keyword = requested.trim().to_ascii_lowercase();
    let mut archs: Vec<String> = Vec::new();
    let mut builtin_list = false;
    match keyword.as_str() {
        "all" => {
            archs = ALL_ARCHS.iter().map(|s| s.to_string()).collect();
            builtin_list = true;
        }
        "default" => {}
        _ => archs = split_archs(&requested),
    }
    if archs.is_empty() && keyword != "default" {
        archs = std::env::var("DEP_AMDHIP64_ARCHS")
            .map(|s| split_archs(&s))
            .unwrap_or_default();
    }
    if archs.is_empty() && keyword != "default" {
        archs = detect_archs(root.as_deref());
    }
    if archs.is_empty() {
        let defaults = if track == "core" {
            CORE_DEFAULT_ARCHS
        } else {
            LEGACY_DEFAULT_ARCHS
        };
        archs = defaults.iter().map(|s| s.to_string()).collect();
        builtin_list = true;
        if keyword != "default" {
            println!(
                "cargo:warning=candle-core: no AMD GPU detected, compiling ROCm kernels for {} ({track} track); set HIP_ARCH to target your GPU",
                archs.join(",")
            );
        }
    }
    if builtin_list {
        archs = supported_archs(&compiler, root.as_deref(), &out_dir, archs);
        if archs.is_empty() {
            panic!(
                "candle-core: {} supports none of the requested GPU targets",
                compiler.path().display()
            );
        }
    }
    println!("cargo:rustc-env=CANDLE_ROCM_ARCHS={}", archs.join(","));

    let extra_flags: Vec<String> = std::env::var("CANDLE_ROCM_HIPCC_FLAGS")
        .map(|s| s.split_whitespace().map(|s| s.to_string()).collect())
        .unwrap_or_default();

    let compile = |k: &str, legacy_driver: bool| -> Result<(), String> {
        let src = kernel_dir.join(format!("{k}.hip"));
        let out = out_dir.join(format!("{k}.co"));
        let mut cmd = base_command(&compiler, root.as_deref());
        for a in archs.iter() {
            cmd.arg(format!("--offload-arch={a}"));
        }
        if legacy_driver {
            cmd.arg("--no-offload-new-driver");
        }
        cmd.arg("-O3")
            .arg("-std=c++17")
            .arg("-I")
            .arg(&kernel_dir)
            .args(extra_flags.iter())
            .arg("-o")
            .arg(&out)
            .arg(&src);
        match cmd.output() {
            Ok(o) if o.status.success() => {
                if is_loadable_image(&out) {
                    Ok(())
                } else {
                    Err(format!("{} is not an offload bundle or code object", out.display()))
                }
            }
            Ok(o) => Err(format!(
                "{}\n{}",
                String::from_utf8_lossy(&o.stdout),
                String::from_utf8_lossy(&o.stderr)
            )),
            Err(e) => Err(e.to_string()),
        }
    };

    let results: Vec<(String, Result<(), String>)> = std::thread::scope(|scope| {
        let handles: Vec<_> = KERNELS
            .iter()
            .map(|k| {
                let compile = &compile;
                scope.spawn(move || {
                    let mut res = compile(k, false);
                    if let Err(e) = &res {
                        if e.contains("is not an offload bundle") {
                            // A newer offload driver may package device code differently; the
                            // classic driver always emits a HIP fat binary.
                            res = compile(k, true);
                        }
                    }
                    (k.to_string(), res)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut failed = false;
    for (k, r) in results {
        if let Err(e) = r {
            failed = true;
            eprintln!("failed to compile ROCm kernel {k}.hip:\n{e}");
        }
    }
    if failed {
        panic!(
            "candle-core: ROCm kernel compilation failed (compiler: {}, track: {track}, HIP {hip_version}, archs: {})",
            compiler.path().display(),
            archs.join(",")
        );
    }
}
