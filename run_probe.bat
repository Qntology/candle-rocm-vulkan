@echo off
call "C:\Users\lp\Documents\GitHub\logis-center\commerce\_env_rocm.bat"
cd /d "C:\Users\lp\Documents\GitHub\candle-rocm-vulkan\candle-backend"
cargo run --release --example vram_probe > "C:\Users\lp\Documents\GitHub\candle-rocm-vulkan\probe.log" 2>&1
echo exit=%ERRORLEVEL% >> "C:\Users\lp\Documents\GitHub\candle-rocm-vulkan\probe.log"
