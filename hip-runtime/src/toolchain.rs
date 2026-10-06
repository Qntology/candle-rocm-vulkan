//! Locating the ROCm / HIP SDK installation at run time.

use std::path::{Path, PathBuf};

fn version_key(p: &Path) -> Vec<u32> {
    p.file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .split(|c: char| !c.is_ascii_digit())
        .filter(|s| !s.is_empty())
        .filter_map(|s| s.parse::<u32>().ok())
        .collect()
}

/// Root of the ROCm installation (`ROCM_PATH`, then `HIP_PATH`, then the platform default).
pub fn rocm_root() -> Option<PathBuf> {
    for var in ["ROCM_PATH", "HIP_PATH"] {
        if let Ok(v) = std::env::var(var) {
            let p = PathBuf::from(v.trim().trim_end_matches(['\\', '/']));
            if p.exists() {
                return Some(p);
            }
        }
    }
    if cfg!(windows) {
        let base = Path::new(r"C:\Program Files\AMD\ROCm");
        let mut versions: Vec<PathBuf> = std::fs::read_dir(base)
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

/// Path of the `hipcc` driver (`HIPCC` env var, the ROCm root, then `PATH`).
pub fn hipcc_path() -> Option<PathBuf> {
    if let Ok(v) = std::env::var("HIPCC") {
        let p = PathBuf::from(v);
        if p.is_file() {
            return Some(p);
        }
    }
    let names: &[&str] = if cfg!(windows) {
        &["hipcc.exe", "hipcc.bin.exe"]
    } else {
        &["hipcc"]
    };
    if let Some(root) = rocm_root() {
        for name in names {
            let candidate = root.join("bin").join(name);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    find_in_path(names)
}
