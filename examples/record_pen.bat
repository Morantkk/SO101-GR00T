@echo off
rem ---------------------------------------------------------------
rem  Record a dataset for the "pick one pen" task (v1).
rem
rem  Usage:
rem     record_pen.bat                 record 50 episodes
rem     record_pen.bat --dataset.num_episodes=5     override anything
rem     record_pen.bat --resume=true   continue an interrupted session
rem                                    (all other params must be identical)
rem
rem  Requirements (see docs/01_INSTALL.md):
rem     - lerobot installed  + datasets / av / torchcodec / pyarrow
rem     - arms calibrated; ids below MUST match your calibration files
rem     - camera indices confirmed with scripts\tools\check_ports.py
rem ---------------------------------------------------------------
setlocal
cd /d "%~dp0\.."

set "PY=%USERPROFILE%\.workbuddy\binaries\python\envs\lerobot-win\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

set "DSROOT=E:/lerobot_datasets/arm101_pick_one_pen"

"%PY%" -m lerobot.scripts.lerobot_record ^
  --robot.type=so101_follower ^
  --robot.port=COM3 ^
  --robot.id=my_awesome_follower_arm ^
  --robot.cameras="{ front: {type: opencv, index_or_path: 1, width: 640, height: 360, fps: 15}, wrist: {type: opencv, index_or_path: 0, width: 1280, height: 720, fps: 15} }" ^
  --teleop.type=so101_leader ^
  --teleop.port=COM4 ^
  --teleop.id=my_awesome_leader_arm ^
  --dataset.repo_id=local/arm101_pick_one_pen ^
  --dataset.root="%DSROOT%" ^
  --dataset.single_task="Pick up a pen and put it into the box" ^
  --dataset.fps=15 ^
  --dataset.episode_time_s=40 ^
  --dataset.reset_time_s=8 ^
  --dataset.num_episodes=50 ^
  --dataset.push_to_hub=false ^
  --dataset.rgb_encoder.vcodec=h264 ^
  --dataset.rgb_encoder.crf=18 ^
  --display_data=true %*

if errorlevel 1 pause
endlocal
