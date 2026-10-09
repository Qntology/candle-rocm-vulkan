@echo off
call "C:\Users\lp\Documents\GitHub\logis-center\commerce\_env_vulkan.bat"
cd /d "C:\Users\lp\Documents\GitHub\candle-rocm-vulkan\candle-core"
set "CANDLE_VULKAN_DEBUG=1"
echo ===== Q1: compute-only queue =====
set "CANDLE_VULKAN_QUEUE=compute"
cargo run --release --features vulkan --example vulkan_gpu_test
echo [STEP] Q1 exit=%ERRORLEVEL%
echo ===== Q2: graphics queue (3D) =====
set "CANDLE_VULKAN_QUEUE=graphics"
cargo run --release --features vulkan --example vulkan_gpu_test
echo [STEP] Q2 exit=%ERRORLEVEL%
echo ===== Q3: compute-only queue (repeat) =====
set "CANDLE_VULKAN_QUEUE=compute"
cargo run --release --features vulkan --example vulkan_gpu_test
echo [STEP] Q3 exit=%ERRORLEVEL%
echo ===== Q4: graphics queue (repeat) =====
set "CANDLE_VULKAN_QUEUE=graphics"
cargo run --release --features vulkan --example vulkan_gpu_test
echo [STEP] Q4 exit=%ERRORLEVEL%
echo VKGPU_DONE
