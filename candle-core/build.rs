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
];

const DEFAULT_ARCHS: &[&str] = &["gfx1030", "gfx1100", "gfx1101", "gfx1102"];

fn env_path(var: &str) -> Option<PathBuf> {
    let v = std::env::var(var).ok()?;
    let p = PathBuf::from(v.trim().trim_end_matches(['\\', '/']));
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

fn rocm_root() -> Option<PathBuf> {
    if let Some(p) = env_path("ROCM_PATH").or_else(|| env_path("HIP_PATH")) {
        return Some(p);
    }
    if cfg!(windows) {
        let mut versions: Vec<PathBuf> = std::fs::read_dir(r"C:\Program Files\AMD\ROCm")
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
        for dir in [root.join("bin"), root.join("llvm").join("bin")] {
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
    s.split([',', ';', ' '])
        .map(|a| a.trim())
        .filter(|a| a.starts_with("gfx") && *a != "gfx000")
        .map(|a| a.to_string())
        .collect()
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
        for dir in [root.join("bin"), root.join("llvm").join("bin")] {
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
        let Ok(out) = Command::new(tool).output() else {
            continue;
        };
        let text = String::from_utf8_lossy(&out.stdout);
        for token in text.split(|c: char| !(c.is_ascii_alphanumeric() || c == ':' || c == '+' || c == '-')) {
            let token = token.trim();
            if token.starts_with("gfx") && token != "gfx000" {
                let base = token.split(':').next().unwrap_or(token).to_string();
                if base.len() > 3 && !archs.contains(&base) {
                    archs.push(base);
                }
            }
        }
        if !archs.is_empty() {
            break;
        }
    }
    archs
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

    let root = rocm_root();
    let compiler = find_compiler(root.as_deref()).unwrap_or_else(|| {
        panic!(
            "candle-core `rocm` feature: hipcc was not found. Install the ROCm / HIP SDK and set \
             ROCM_PATH or HIP_PATH (Windows: e.g. C:\\Program Files\\AMD\\ROCm\\6.2), or set HIPCC. \
             Set CANDLE_ROCM_SKIP_KERNEL_BUILD=1 to only type-check."
        )
    });

    let mut archs = std::env::var("CANDLE_ROCM_ARCHS")
        .or_else(|_| std::env::var("HIP_ARCH"))
        .map(|s| split_archs(&s))
        .unwrap_or_default();
    if archs.is_empty() {
        archs = detect_archs(root.as_deref());
    }
    if archs.is_empty() {
        archs = DEFAULT_ARCHS.iter().map(|s| s.to_string()).collect();
        println!(
            "cargo:warning=candle-core: no AMD GPU detected, compiling ROCm kernels for {}; set HIP_ARCH to target your GPU",
            archs.join(",")
        );
    }
    println!("cargo:rustc-env=CANDLE_ROCM_ARCHS={}", archs.join(","));

    let extra_flags: Vec<String> = std::env::var("CANDLE_ROCM_HIPCC_FLAGS")
        .map(|s| s.split_whitespace().map(|s| s.to_string()).collect())
        .unwrap_or_default();

    let results: Vec<(String, Result<(), String>)> = std::thread::scope(|scope| {
        let handles: Vec<_> = KERNELS
            .iter()
            .map(|k| {
                let src = kernel_dir.join(format!("{k}.hip"));
                let out = out_dir.join(format!("{k}.co"));
                let archs = &archs;
                let compiler = &compiler;
                let kernel_dir = &kernel_dir;
                let extra_flags = &extra_flags;
                let root = &root;
                scope.spawn(move || {
                    let mut cmd = match compiler {
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
                    };
                    for a in archs.iter() {
                        cmd.arg(format!("--offload-arch={a}"));
                    }
                    cmd.arg("-O3")
                        .arg("-std=c++17")
                        .arg("-I")
                        .arg(kernel_dir)
                        .args(extra_flags.iter())
                        .arg("-o")
                        .arg(&out)
                        .arg(&src);
                    let res = match cmd.output() {
                        Ok(o) if o.status.success() => Ok(()),
                        Ok(o) => Err(format!(
                            "{}\n{}",
                            String::from_utf8_lossy(&o.stdout),
                            String::from_utf8_lossy(&o.stderr)
                        )),
                        Err(e) => Err(e.to_string()),
                    };
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
        panic!("candle-core: ROCm kernel compilation failed (archs: {})", archs.join(","));
    }
}
