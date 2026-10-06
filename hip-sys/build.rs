use std::path::{Path, PathBuf};

fn rocm_root() -> Option<PathBuf> {
    for var in ["ROCM_PATH", "HIP_PATH"] {
        if let Ok(v) = std::env::var(var) {
            let p = PathBuf::from(v.trim().trim_end_matches(['\\', '/']));
            if p.exists() {
                return Some(p);
            }
        }
    }
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "windows" {
        let base = Path::new(r"C:\Program Files\AMD\ROCm");
        if let Ok(entries) = std::fs::read_dir(base) {
            let mut versions: Vec<PathBuf> = entries
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.join("lib").exists())
                .collect();
            versions.sort_by_key(|p| version_key(p));
            if let Some(p) = versions.pop() {
                return Some(p);
            }
        }
        None
    } else {
        let p = PathBuf::from("/opt/rocm");
        if p.exists() {
            Some(p)
        } else {
            None
        }
    }
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

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=HIP_PATH");
    println!("cargo:rerun-if-env-changed=CANDLE_ROCM_LIB_DIR");

    let mut lib_dirs: Vec<PathBuf> = Vec::new();
    if let Ok(dir) = std::env::var("CANDLE_ROCM_LIB_DIR") {
        for d in std::env::split_paths(&dir) {
            lib_dirs.push(d);
        }
    }
    if let Some(root) = rocm_root() {
        lib_dirs.push(root.join("lib"));
        lib_dirs.push(root.join("lib64"));
    }
    for dir in lib_dirs.iter().filter(|d| d.exists()) {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }

    println!("cargo:rustc-link-lib=dylib=amdhip64");
    println!("cargo:rustc-link-lib=dylib=rocblas");
    if std::env::var("CARGO_FEATURE_HIPRAND").is_ok() {
        println!("cargo:rustc-link-lib=dylib=hiprand");
    }
}
