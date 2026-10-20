@echo off
chcp 65001 >nul
REM 칸 단위 결과(실험마다 predictions_seed*.csv)를 10분마다 GitHub pc-results 브랜치에 올린다.
REM 실험 폴더(SRC)는 git 저장소가 아니어도 된다. 이 파일은 git clone 한 폴더(C:\work\pilotA-minji-git) 안에서 실행.
set SRC=C:\work\pilotA-minji-claude-dazzling-babbage-8c6yd4\outputs\disasternet
:loop
for /d %%D in ("%SRC%\cells_max*_aw") do (
  for %%F in ("%%D\predictions_seed*.csv") do (
    if not exist "pc_results\%%~nxD" mkdir "pc_results\%%~nxD"
    copy /y "%%F" "pc_results\%%~nxD\" >nul
  )
)
git add pc_results
git diff --cached --quiet || (
  git commit -q -m "[Results] PC 칸 단위 면적 평균 결과" && git push -q -f origin HEAD:refs/heads/pc-results && echo %date% %time% 올림
)
timeout /t 600 /nobreak >nul
goto loop
