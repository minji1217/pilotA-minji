@echo off
chcp 65001 >nul
REM DisasterNet(칸 단위 LOCALU)과 pilotA(A2)의 학습 · 추론 시간을 이 PC에서 잰다.
REM 실험 폴더(C:\work\pilotA-minji-claude-dazzling-babbage-8c6yd4)에서 실행. 약 10~15분 걸림.
REM pilotA 비교 브랜치(followup4-lq-J-nolabels)를 ..\pilotA-time 에 받고, pilotA 입력 파일을 복사한다.
set P=..\pilotA-time
if not exist "%P%\train.py" git clone -b followup4-lq-J-nolabels https://github.com/minji1217/pilotA.git "%P%"
if not exist "%P%\raw" mkdir "%P%\raw"
if not exist "%P%\validation" mkdir "%P%\validation"
copy /Y "raw\재난프로젝트_시정촌별_USGS.xlsx" "%P%\raw\" >nul
copy /Y "raw\재난프로젝트_시정촌별_통계데이터.xlsx" "%P%\raw\" >nul
copy /Y "validation\LS_LF 데이터자료.xlsx" "%P%\validation\" >nul
py -3.10 -m disasternet.time_compare --pilota-dir "%P%"
echo.
echo 끝. 위 결과 화면(또는 outputs\time_compare.txt)을 보내 주세요.
pause
