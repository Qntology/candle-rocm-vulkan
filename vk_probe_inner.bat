@echo off
call "C:\Users\lp\Documents\GitHub\logis-center\commerce\_env_vulkan.bat"
echo [STEP] env done, cwd=%CD%
cd /d "C:\Users\lp\Documents\GitHub\candle-rocm-vulkan\candle-core"
echo ===== default (as the app runs) =====
cargo run --release --features vulkan --example vulkan_probe
echo [STEP] run1 exit=%ERRORLEVEL%
echo ===== CANDLE_VULKAN_MEMORY=device =====
set "CANDLE_VULKAN_MEMORY=device"
cargo run --release --features vulkan --example vulkan_probe
echo [STEP] run2 exit=%ERRORLEVEL%
echo VK_PROBE_DONE
