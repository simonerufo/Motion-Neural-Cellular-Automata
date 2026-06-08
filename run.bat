@echo off

if "%~3"=="" goto help

set ACTION=%1
set EXPERIMENT=%2
set TARGET=%3
set SAVE_GIF=%4

echo Starting Script with Action=%ACTION% ^| Experiment=%EXPERIMENT% ^| Image=%TARGET% %SAVE_GIF%
python main.py --action %ACTION% --experiment %EXPERIMENT% --target %TARGET% %SAVE_GIF%
goto end

:help
echo Usage: run.bat [ACTION] [EXPERIMENT] [TARGET_IMAGE] [--save_gif]
echo.
echo Available Actions: train1, train2, train_all, inference1, inference2
echo Available Experiments: chemotaxis, chemotaxis_obs, ecosystem
echo.
echo Example for Training (Chemotaxis):
echo   run.bat train_all chemotaxis targets\salamander32.png
echo.
echo Examples for Inference:
echo   1. Base Chemotaxis (No obstacles):
echo      run.bat inference2 chemotaxis targets\salamander32.png
echo.
echo   2. Chemotaxis with Obstacles:
echo      run.bat inference2 chemotaxis_obs targets\salamander32.png
echo.
echo   3. Ecosystem (Random Exploration ^& Mitosis):
echo      run.bat inference2 ecosystem targets\jelly32_1.png
echo.
echo To save as GIF instead of showing on screen, add --save_gif at the end:
echo      run.bat inference2 ecosystem targets\jelly32_1.png --save_gif

:end
