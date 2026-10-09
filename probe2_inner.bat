@echo off
call "C:\Users\lp\Documents\GitHub\logis-center\commerce\_env_rocm.bat"
cd /d "C:\Users\lp\Documents\GitHub\candle-rocm-vulkan\candle-backend"
echo ===== FIXED LIBRARY, no env set =====
set "GPU_RESOURCE_CACHE_SIZE="
cargo run --release --example vram_probe2
echo [STEP] exit=%ERRORLEVEL%
echo PROBE2_DONE
