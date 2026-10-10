//! ROCm release streams ("tracks") and HIP versions.
//!
//! * [`RocmTrack::Legacy`]: ROCm / AMD HIP SDK 6.x - 7.2.
//! * [`RocmTrack::Core`]: ROCm Core SDK built by TheRock, HIP 7.10 and later
//!   (ROCm 7.14 = HIP 7.14, ROCm 10.0 = HIP 7.15, ROCm 10.1 = HIP 7.16).
//!
//! Both keep the HIP 7 ABI (`amdhip64_7`), so one build of this crate runs on either runtime.

use std::fmt;

/// GPU targets supported by the ROCm Core SDK 10.1 release (Instinct, Radeon and Ryzen).
pub const CORE_TRACK_ARCHS: &[&str] = &[
    "gfx908", "gfx90a", "gfx942", "gfx950", "gfx1030", "gfx1100", "gfx1101", "gfx1102", "gfx1103",
    "gfx1150", "gfx1151", "gfx1152", "gfx1153", "gfx1200", "gfx1201",
];

/// Whether the ROCm Core SDK 10.1 supports `arch` (processor name, e.g. `gfx1100`).
pub fn core_track_supports(arch: &str) -> bool {
    let base = arch.split(':').next().unwrap_or(arch);
    CORE_TRACK_ARCHS.contains(&base)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RocmTrack {
    /// ROCm / AMD HIP SDK 6.x - 7.2.
    Legacy,
    /// ROCm Core SDK (TheRock), HIP 7.10 and later.
    Core,
}

impl RocmTrack {
    pub fn name(self) -> &'static str {
        match self {
            Self::Legacy => "legacy",
            Self::Core => "core",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "legacy" => Some(Self::Legacy),
            "core" | "therock" => Some(Self::Core),
            _ => None,
        }
    }

    pub fn from_hip(major: u32, minor: u32) -> Self {
        if major > 7 || (major == 7 && minor >= 10) {
            Self::Core
        } else {
            Self::Legacy
        }
    }
}

impl fmt::Display for RocmTrack {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// HIP version, `HIP_VERSION = major * 10_000_000 + minor * 100_000 + patch`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HipVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl HipVersion {
    /// Decodes the value returned by `hipRuntimeGetVersion`.
    pub fn from_raw(raw: i32) -> Self {
        let v = raw.max(0) as u32;
        Self {
            major: v / 10_000_000,
            minor: (v / 100_000) % 100,
            patch: v % 100_000,
        }
    }

    /// Parses `"7.16.0"` (missing components are 0).
    pub fn parse(s: &str) -> Option<Self> {
        let mut it = s.trim().split('.').map(|p| p.trim().parse::<u32>());
        let major = it.next()?.ok()?;
        let minor = it.next().and_then(|r| r.ok()).unwrap_or(0);
        let patch = it.next().and_then(|r| r.ok()).unwrap_or(0);
        Some(Self { major, minor, patch })
    }

    pub fn track(&self) -> RocmTrack {
        RocmTrack::from_hip(self.major, self.minor)
    }
}

impl fmt::Display for HipVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_runtime_versions() {
        let v = HipVersion::from_raw(71_600_000);
        assert_eq!((v.major, v.minor, v.patch), (7, 16, 0));
        assert_eq!(v.track(), RocmTrack::Core);
        let v = HipVersion::from_raw(70_226_014);
        assert_eq!((v.major, v.minor, v.patch), (7, 2, 26014));
        assert_eq!(v.track(), RocmTrack::Legacy);
        assert_eq!(HipVersion::from_raw(60_342_134).track(), RocmTrack::Legacy);
        assert_eq!(HipVersion::parse("7.14.1").unwrap().track(), RocmTrack::Core);
        assert_eq!(HipVersion::parse("7.16").unwrap().to_string(), "7.16.0");
    }

    #[test]
    fn core_track_targets() {
        assert!(core_track_supports("gfx1201"));
        assert!(core_track_supports("gfx90a:sramecc+:xnack-"));
        assert!(!core_track_supports("gfx1010"));
        assert!(!core_track_supports("gfx1032"));
        assert_eq!(RocmTrack::parse(" Core "), Some(RocmTrack::Core));
        assert_eq!(RocmTrack::parse("auto"), None);
    }
}
