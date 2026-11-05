@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
REM 칸 단위 DisasterNet 실험 5개(PPT 1022_8: DN, SOFTU, SOFTU_끄기만, SOFTU_분산만, LOCALU)를 seed 0~4로 다시 돌린다.
REM 예전과 같은 설정(지진 5개 동시, 스레드 2)이라 예측값이 예전과 똑같이 나오고, 이번에는 파라미터와 loss를 CSV로 남긴다.
REM 실험 폴더(C:\work\pilotA-minji-claude-dazzling-babbage-8c6yd4)에서 실행. 전부 약 35시간.
REM 새 결과: outputs\disasternet\rerun_cells\<실험>\  (예전 결과 폴더는 건드리지 않음)
REM   params_<지진>_seed<s>.csv       노드 가중치 등
REM   rowparams_<지진>_seed<s>.csv.gz 칸마다의 mu 보정값, flow 보정값
REM   loss_<지진>_seed<s>.csv         epoch마다 loss
REM   muni_ / cells_ ...             예측값 (예전과 같은 형식)
REM 중간에 꺼져도 다시 실행하면 끝난 것은 건너뛰고, 학습 중이던 것은 5분마다 저장된 지점부터 이어서 한다.
REM seed 하나가 끝날 때마다 예전 결과와 비교한 표(rerun_compare.csv)를 만들고,
REM C:\work\pilotA-minji-git 이 있으면 작은 파일(loss, params, 비교표, 기록)을 GitHub pc-results 브랜치에 올린다.

set JOBS=5
set THREADS=2
set O=outputs\disasternet
set N=%O%\rerun_cells
set G=C:\work\pilotA-minji-git
set L=--local raw\local_250m_byevent_usgs_aw.csv.gz --off-rule paper --jobs %JOBS% --threads %THREADS%
if not exist "%N%" mkdir "%N%"
set LOG=%N%\log.txt

for %%S in (0 1 2 3 4) do (
  call :stamp "===== seed %%S 시작"
  call :stamp "  LOCALU seed %%S"
  py -3.10 -m disasternet.run_cells_max %L% --seeds %%S --local-off --unc --out-dir %N%\LOCALU >> "%LOG%" 2>&1
  call :stamp "  DN seed %%S"
  py -3.10 -m disasternet.run_cells_max %L% --seeds %%S --out-dir %N%\DN >> "%LOG%" 2>&1
  call :stamp "  SOFTU seed %%S"
  py -3.10 -m disasternet.run_cells_max %L% --seeds %%S --soft --soft-mid --unc --out-dir %N%\SOFTU >> "%LOG%" 2>&1
  call :stamp "  SOFTU_끄기만 seed %%S"
  py -3.10 -m disasternet.run_cells_max %L% --seeds %%S --soft --soft-mid --unc --unc-mode off --out-dir %N%\SOFTU_off >> "%LOG%" 2>&1
  call :stamp "  SOFTU_분산만 seed %%S"
  py -3.10 -m disasternet.run_cells_max %L% --seeds %%S --soft --soft-mid --unc --unc-mode var --out-dir %N%\SOFTU_var >> "%LOG%" 2>&1
  call :compare
  call :push
  call :stamp "===== seed %%S 끝"
)
echo ALL DONE %date% %time%
echo ALL DONE %date% %time% >> "%LOG%"
pause
exit /b 0

:stamp
echo %~1  %date% %time%
echo %~1  %date% %time% >> "%LOG%"
exit /b 0

:compare
py -3.10 -m disasternet.rerun_compare --level cells --out %N%\rerun_compare.csv ^
  --pair LOCALU %O%\cells_max_local_unc_paper_aw %N%\LOCALU ^
  --pair DN %O%\cells_max_aw %N%\DN ^
  --pair SOFTU %O%\cells_max_softunc_paper_aw_mid %N%\SOFTU ^
  --pair SOFTU_끄기만 %O%\cells_max_softunc_off_paper_aw_mid %N%\SOFTU_off ^
  --pair SOFTU_분산만 %O%\cells_max_softunc_var_paper_aw_mid %N%\SOFTU_var > %N%\rerun_compare.txt 2>&1
type %N%\rerun_compare.txt
type %N%\rerun_compare.txt >> "%LOG%"
exit /b 0

:push
if not exist "%G%\.git" exit /b 0
set R=%CD%\%N%
for /d %%D in ("%R%\*") do (
  if not exist "%G%\pc_rerun\%%~nxD" mkdir "%G%\pc_rerun\%%~nxD"
  copy /y "%%D\loss_*.csv" "%G%\pc_rerun\%%~nxD\" >nul 2>&1
  copy /y "%%D\params_*.csv" "%G%\pc_rerun\%%~nxD\" >nul 2>&1
  copy /y "%%D\muni_*.csv" "%G%\pc_rerun\%%~nxD\" >nul 2>&1
)
copy /y "%R%\rerun_compare.csv" "%G%\pc_rerun\" >nul 2>&1
copy /y "%R%\log.txt" "%G%\pc_rerun\" >nul 2>&1
pushd "%G%"
git add pc_rerun
git diff --cached --quiet || (git commit -q -m "[Results] PC 칸 단위 다시 돌리기: 파라미터 · loss" && git push -q -f origin HEAD:refs/heads/pc-results && echo   GitHub에 올림)
popd
exit /b 0
