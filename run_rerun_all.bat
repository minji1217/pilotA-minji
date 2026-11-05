@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
REM PPT 1022_8의 DisasterNet 실험을 전부 seed 0~4로 다시 돌려 파라미터와 loss를 CSV로 남긴다.
REM   1부 시정촌 단위 10개: DN, SOFT, DNU, DNU_끄기만, DNU_분산만, SOFTU, SOFTU_끄기만, SOFTU_분산만, DN_같은분산0.01, DN_같은분산0.001
REM   2부 칸 단위 5개: LOCALU, DN, SOFTU, SOFTU_끄기만, SOFTU_분산만 (예전과 같은 설정: 지진 5개 동시, 스레드 2)
REM 실험 폴더(C:\work\pilotA-minji-claude-dazzling-babbage-8c6yd4)에서 실행. 1부 약 1시간, 2부 약 35시간.
REM 새 결과 (예전 결과 폴더는 건드리지 않음)
REM   outputs\disasternet\rerun_muni\<실험>\  params_seed<s>.csv, rowparams_seed<s>.csv, loss_seed<s>.csv, predictions_seed<s>.csv
REM   outputs\disasternet\rerun_cells\<실험>\ params_<지진>_seed<s>.csv, rowparams_<지진>_seed<s>.csv.gz(칸마다), loss_<지진>_seed<s>.csv, muni_ / cells_ (예측)
REM 중간에 꺼져도 다시 실행하면 끝난 것은 건너뛴다 (칸 단위는 학습 중이던 것도 5분마다 저장된 지점부터 이어서).
REM 실험이 끝날 때마다 예전 결과와 비교한 표(rerun_compare.csv)를 만들고,
REM C:\work\pilotA-minji-git 이 있으면 작은 파일을 GitHub pc-results 브랜치(pc_rerun 폴더)에 올리고,
REM Google Drive for desktop(G: 드라이브)이 있으면 결과 전체를 G:\내 드라이브\pilotA_rerun 으로 복사한다.

set O=outputs\disasternet
set M=%O%\rerun_muni
set N=%O%\rerun_cells
set P=pc_inputs
set G=C:\work\pilotA-minji-git
if not exist "%M%" mkdir "%M%"
if not exist "%N%" mkdir "%N%"
set LOG=%O%\rerun_log.txt

REM ---------------- 1부: 시정촌 단위 (seed 0~4를 5개 동시에, 각 1스레드) ----------------
set I=--input %P%\disasternet_input_v5.xlsx
set U=--usgs-muni %P%\muni_usgs_pixel.csv
set T=--terrain %P%\terrain_fractions.csv
set MJ=--no-bd --jobs 5
call :muni DN "disasternet.run_input_v5 %I% %MJ%"
call :muni SOFT "disasternet.run_input_v5 %I% %MJ% %T%"
call :muni DNU "disasternet.run_muni_unc %I% %U% %MJ%"
call :muni DNU_off "disasternet.run_muni_unc %I% %U% %MJ% --unc off"
call :muni DNU_var "disasternet.run_muni_unc %I% %U% %MJ% --unc var"
call :muni SOFTU "disasternet.run_muni_unc %I% %U% %MJ% %T%"
call :muni SOFTU_off "disasternet.run_muni_unc %I% %U% %MJ% %T% --unc off"
call :muni SOFTU_var "disasternet.run_muni_unc %I% %U% %MJ% %T% --unc var"
call :muni DN_cvar01 "disasternet.run_muni_unc %I% %U% %MJ% --unc none --const-var 0.01"
call :muni DN_cvar001 "disasternet.run_muni_unc %I% %U% %MJ% --unc none --const-var 0.001"
call :compare_muni
call :push
call :drive

REM ---------------- 2부: 칸 단위 ----------------
set L=--local raw\local_250m_byevent_usgs_aw.csv.gz --off-rule paper --jobs 5 --threads 2
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
  call :compare_cells
  call :push
  call :drive
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

:muni
REM %1 = 실험 이름, %2 = 실행할 모듈과 옵션. seed 4 결과가 있으면 끝난 것으로 보고 건너뛴다.
if exist "%M%\%1\predictions_seed4.csv" (
  call :stamp "  시정촌 %1 이미 있음"
  exit /b 0
)
call :stamp "  시정촌 %1 시작"
py -3.10 -m %~2 --out-dir %M%\%1 >> "%LOG%" 2>&1
call :stamp "  시정촌 %1 끝"
exit /b 0

:compare_muni
py -3.10 -m disasternet.rerun_compare --level muni --out %M%\rerun_compare.csv ^
  --pair DN %P%\muni_old\DN %M%\DN --pair SOFT %P%\muni_old\SOFT %M%\SOFT ^
  --pair DNU %P%\muni_old\DNU %M%\DNU --pair DNU_끄기만 %P%\muni_old\DNU_off %M%\DNU_off ^
  --pair DNU_분산만 %P%\muni_old\DNU_var %M%\DNU_var --pair SOFTU %P%\muni_old\SOFTU %M%\SOFTU ^
  --pair SOFTU_끄기만 %P%\muni_old\SOFTU_off %M%\SOFTU_off --pair SOFTU_분산만 %P%\muni_old\SOFTU_var %M%\SOFTU_var ^
  --pair DN_같은분산0.01 %P%\muni_old\DN_cvar01 %M%\DN_cvar01 --pair DN_같은분산0.001 %P%\muni_old\DN_cvar001 %M%\DN_cvar001 > %M%\rerun_compare.txt 2>&1
type %M%\rerun_compare.txt
type %M%\rerun_compare.txt >> "%LOG%"
exit /b 0

:compare_cells
py -3.10 -m disasternet.rerun_compare --level cells --out %N%\rerun_compare.csv ^
  --pair LOCALU %O%\cells_max_local_unc_paper_aw %N%\LOCALU ^
  --pair DN %O%\cells_max_aw %N%\DN ^
  --pair SOFTU %O%\cells_max_softunc_paper_aw_mid %N%\SOFTU ^
  --pair SOFTU_끄기만 %O%\cells_max_softunc_off_paper_aw_mid %N%\SOFTU_off ^
  --pair SOFTU_분산만 %O%\cells_max_softunc_var_paper_aw_mid %N%\SOFTU_var > %N%\rerun_compare.txt 2>&1
type %N%\rerun_compare.txt
type %N%\rerun_compare.txt >> "%LOG%"
exit /b 0

:drive
REM Google Drive for desktop이 있으면(G:\내 드라이브 또는 G:\My Drive) 결과 전체(칸 파라미터 포함)를 pilotA_rerun 폴더로 복사한다.
set D=
if exist "G:\내 드라이브" set D=G:\내 드라이브\pilotA_rerun
if exist "G:\My Drive" set D=G:\My Drive\pilotA_rerun
if "%D%"=="" exit /b 0
robocopy "%CD%\%M%" "%D%\muni" /E /XF ckpt_*.pt *.tmp /NFL /NDL /NJH /NJS /NP >nul
robocopy "%CD%\%N%" "%D%\cells" /E /XF ckpt_*.pt *.tmp /NFL /NDL /NJH /NJS /NP >nul
copy /y "%LOG%" "%D%\" >nul 2>&1
call :stamp "  Google Drive에 복사: %D%"
exit /b 0

:push
REM GitHub pc-results 브랜치의 pc_rerun 폴더로: 시정촌은 결과 전부, 칸은 loss · params · 시정촌 예측(칸 파라미터는 너무 커서 Drive로만)
if not exist "%G%\.git" exit /b 0
for /d %%D in ("%CD%\%M%\*") do (
  if not exist "%G%\pc_rerun\muni\%%~nxD" mkdir "%G%\pc_rerun\muni\%%~nxD"
  copy /y "%%D\*.csv" "%G%\pc_rerun\muni\%%~nxD\" >nul 2>&1
)
for /d %%D in ("%CD%\%N%\*") do (
  if not exist "%G%\pc_rerun\cells\%%~nxD" mkdir "%G%\pc_rerun\cells\%%~nxD"
  copy /y "%%D\loss_*.csv" "%G%\pc_rerun\cells\%%~nxD\" >nul 2>&1
  copy /y "%%D\params_*.csv" "%G%\pc_rerun\cells\%%~nxD\" >nul 2>&1
  copy /y "%%D\muni_*.csv" "%G%\pc_rerun\cells\%%~nxD\" >nul 2>&1
)
if exist "%M%\rerun_compare.csv" copy /y "%M%\rerun_compare.csv" "%G%\pc_rerun\muni\" >nul 2>&1
if exist "%N%\rerun_compare.csv" copy /y "%N%\rerun_compare.csv" "%G%\pc_rerun\cells\" >nul 2>&1
copy /y "%LOG%" "%G%\pc_rerun\" >nul 2>&1
pushd "%G%"
git add pc_rerun
git diff --cached --quiet || (git commit -q -m "[Results] PC 다시 돌리기: 파라미터 · loss" && git push -q -f origin HEAD:refs/heads/pc-results && echo   GitHub에 올림)
popd
exit /b 0
