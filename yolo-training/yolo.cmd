@echo off
REM yolo.exe from the Python installation is not on PATH, so typing "yolo" fails.
REM This wrapper reaches the same Ultralytics CLI entrypoint through the py launcher.
REM
REM Usage: yolo-training\yolo.cmd cfg=yolo-training\train.yaml data=D:\datasets\sort\shape\data.yaml
REM If the Scripts folder is later added to PATH, plain "yolo" works and this is unnecessary.
py -3.14 -c "from ultralytics.cfg import entrypoint; entrypoint()" %*
