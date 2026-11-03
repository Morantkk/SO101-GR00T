@echo off
rem ---------------------------------------------------------------
rem  SO101-GR00T  one-shot launcher (cmd.exe / PowerShell / double-click)
rem
rem  Usage:
rem     rtc_run.bat                          normal RTC run (unlimited steps)
rem     rtc_run.bat --no-rtc                 baseline, no RTC
rem     rtc_run.bat --dry-run --steps 32     dry check, arm does NOT move
rem     rtc_run.bat --steps 60               short run
rem
rem  Before the first run:
rem     1) copy ..\examples\conn.example.py to ..\examples\conn.py and fill in
rem        HOST / PORT / PASSWORD (or set the GR00T_* env vars below)
rem     2) set PY_CLIENT to your own python (must have lerobot installed)
rem     3) start the policy server on the GPU box
rem
rem  ASCII only on purpose: the project path may contain non-ASCII chars,
rem  so this file must not embed it. %~dp0 resolves it at runtime.
rem ---------------------------------------------------------------
setlocal
cd /d "%~dp0\.."

set "PY_CLIENT=python"
if exist "C:\Users\Fakun\.workbuddy\binaries\python\envs\lerobot-win\Scripts\python.exe" (
  set "PY_CLIENT=C:\Users\Fakun\.workbuddy\binaries\python\envs\lerobot-win\Scripts\python.exe"
)

rem ---- connection (leave empty to use examples\conn.py) ----
rem set "GR00T_HOST=connect.example.seetacloud.com"
rem set "GR00T_PORT=40009"
rem set "GR00T_PASSWORD=******"

if not exist "scripts\deploy\run_groot_rtc.py" (
  echo [ERR] scripts\deploy\run_groot_rtc.py not found. Wrong working dir?
  pause
  exit /b 1
)

echo [1/2] checking tunnel to the policy server ...
rem  A listening port does NOT mean the tunnel works: a dead SSH channel keeps
rem  the local port open but forwards nothing. rtc_up.py does a real ping and
rem  rebuilds the tunnel when needed.
"%PY_CLIENT%" scripts\deploy\rtc_up.py --local 5555
if errorlevel 1 (
  echo.
  echo [ERR] tunnel is not usable. Server down, or host/port/password changed.
  echo       start it manually:  python scripts\deploy\ssh_tunnel.py --local 5555 --remote 127.0.0.1:5555
  pause
  exit /b 1
)

echo [2/2] running client ...
echo        extra flags are appended, e.g.  rtc_run.bat --no-rtc
echo        DANGER: the arm moves as soon as it connects. Clear the area.
echo.
"%PY_CLIENT%" scripts\deploy\run_groot_rtc.py ^
  --steps 0 --rtc-fire 1 --chunk 16 --max-rel 8 ^
  --task "Pick up a pen and put it into the box" ^
  --record-video outputs\rtc_run1.mp4 --log-csv outputs\rtc_run1.csv %*

echo.
echo ---- done (exit code %ERRORLEVEL%) ----
pause
