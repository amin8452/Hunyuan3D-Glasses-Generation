@echo off
echo Hunyuan3D Glasses Generation
echo ===========================

if "%~1"=="" (
    echo Error: Please provide an input image.
    echo Usage: run_glasses_generation.bat your_image.jpg [output_model.glb]
    exit /b 1
)

set INPUT_IMAGE=%~1

if "%~2"=="" (
    set OUTPUT_MODEL=output.glb
) else (
    set OUTPUT_MODEL=%~2
)

echo Input image: %INPUT_IMAGE%
echo Output model: %OUTPUT_MODEL%
echo.
echo Running glasses generation...
echo.

python run_glasses_generation.py --input_image "%INPUT_IMAGE%" --output_model "%OUTPUT_MODEL%"

echo.
if %ERRORLEVEL% EQU 0 (
    echo Process completed successfully!
) else (
    echo Process failed with error code %ERRORLEVEL%
)

pause
