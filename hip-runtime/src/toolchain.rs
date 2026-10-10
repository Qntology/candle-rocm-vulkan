//! Locating the ROCm / HIP SDK installation at run time.
//!
//! Both install layouts are recognised: the legacy ROCm / HIP SDK (`bin/clang++`, `llvm/bin`) and
//! the ROCm Core SDK built by TheRock (`lib/llvm/bin`, tarballs, `rocm-sdk` Python wheels,
//! `C:\Program Files\AMD\ROCm\core-10.x`).

use crate::track::{HipVersion, RocmTrack};
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

fn clean_path(v: &str) -> PathBuf {
    PathBuf::from(v.trim().trim_matches('"').trim_end_matches(['\\', '/']))
}

/// HIP version of an installation, read from `include/hip/hip_version.h`.
pub fn installed_hip_version(root: &Path) -> Option<HipVersion> {
    let text = std::fs::read_to_string(root.join("include").join("hip").join("hip_version.h")).ok()?;
    let mut v = [None::<u32>; 3];
    for line in text.lines() {
        let mut it = line.split_whitespace();
        if it.next() != Some("#define") {
            continue;
        }
        let slot = match it.next() {
            Some("HIP_VERSION_MAJOR") => 0,
            Some("HIP_VERSION_MINOR") => 1,
            Some("HIP_VERSION_PATCH") => 2,
            _ => continue,
        };
        v[slot] = it.next().and_then(|s| s.parse().ok());
    }
    Some(HipVersion {
        major: v[0]?,
        minor: v[1].unwrap_or(0),
        patch: v[2].unwrap_or(0),
    })
}

/// Track of an installation: from its HIP version, else from the directory name.
pub fn installed_track(root: &Path) -> RocmTrack {
    if let Some(v) = installed_hip_version(root) {
        return v.track();
    }
    let name = root
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let major = version_key(root).first().copied().unwrap_or(0);
    if name.starts_with("core") || name.starts_with("runtime") || major >= 10 {
        RocmTrack::Core
    } else {
        RocmTrack::Legacy
    }
}

fn rocm_sdk_root() -> Option<PathBuf> {
    let exe = find_in_path(&[if cfg!(windows) { "rocm-sdk.exe" } else { "rocm-sdk" }])?;
    let out = std::process::Command::new(exe).args(["path", "--root"]).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    let p = clean_path(stdout.lines().next()?);
    p.is_dir().then_some(p)
}

/// ROCm installations found on this machine, newest first.
pub fn installations() -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    if let Some(p) = rocm_sdk_root() {
        out.push(p);
    }
    if cfg!(windows) {
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
        if !p.join("bin").is_dir() {
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
    let key = |p: &PathBuf| {
        let hip = installed_hip_version(p)
            .map(|v| vec![v.major, v.minor, v.patch])
            .unwrap_or_default();
        (hip, version_key(p))
    };
    out.sort_by_key(|p| std::cmp::Reverse(key(p)));
    out
}

/// Root of the ROCm installation: `ROCM_PATH`, then `HIP_PATH`, then the newest installation of
/// the track named by `CANDLE_ROCM_TRACK` (`legacy` / `core`), then the newest one.
pub fn rocm_root() -> Option<PathBuf> {
    for var in ["ROCM_PATH", "HIP_PATH"] {
        if let Ok(v) = std::env::var(var) {
            if v.trim().is_empty() {
                continue;
            }
            let p = clean_path(&v);
            if p.exists() {
                return Some(p);
            }
        }
    }
    let all = installations();
    if let Some(track) = std::env::var("CANDLE_ROCM_TRACK")
        .ok()
        .and_then(|t| RocmTrack::parse(&t))
    {
        if let Some(p) = all.iter().find(|p| installed_track(p) == track) {
            return Some(p.clone());
        }
    }
    all.into_iter().next()
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
        let p = clean_path(&v);
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
        for dir in [root.join("bin"), root.join("lib").join("llvm").join("bin")] {
            for name in names {
                let candidate = dir.join(name);
                if candidate.is_file() {
                    return Some(candidate);
                }
            }
        }
    }
    find_in_path(names)
}
