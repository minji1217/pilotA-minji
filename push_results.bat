@echo off
chcp 65001 >nul
REM 칸 단위 결과(시정촌별 최댓값 predictions_seed0.csv, 실험마다 작은 파일 1개)를 10분마다 GitHub에 올린다.
REM run_cells_aw.bat 과 따로, 두 번째 명령 프롬프트 창에서 pilotA-minji 폴더에서 실행. 끝나면 창을 닫으면 된다.
REM 올리는 곳: GitHub minji1217/pilotA-minji 의 pc-results 브랜치 (Claude가 여기서 받아 채점)
:loop
for %%D in (outputs\disasternet\cells_max_*_aw) do (
  if exist "%%D\predictions_seed0.csv" git add -f "%%D\predictions_seed0.csv"
)
git diff --cached --quiet || (
  git commit -q -m "[Results] PC 칸 단위 면적 평균 결과" && git push -q -f origin HEAD:refs/heads/pc-results && echo %date% %time% 올림
)
timeout /t 600 /nobreak >nul
goto loop
