$ErrorActionPreference = "Stop"
Set-Location "C:\Users\16946\Documents\OBJECTIVE CHECK PAPER"
C:\uav_env\Scripts\python.exe scripts\monitor_training.py --run "runs\detect\runs\baselines\scratch-yolo-100ep" --total-epochs 100 --interval 10
