@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
REM Re-run all DisasterNet experiments of PPT 1022_8 (seed 0-4) and save parameters and loss as CSV.
REM   Part 1 municipality (10): DN SOFT DNU DNU_off DNU_var SOFTU SOFTU_off SOFTU_var DN_cvar01 DN_cvar001
REM   Part 2 cells (5): LOCALU DN SOFTU SOFTU_off SOFTU_var  (same settings as before: 5 events in parallel, 2 threads)
REM Run in the experiment folder. Part 1 about 1 hour, part 2 about 35 hours.
REM Output (old result folders are not touched):
REM   outputs\disasternet\rerun_muni\<exp>\   params_seed<s>.csv rowparams_seed<s>.csv loss_seed<s>.csv predictions_seed<s>.csv
REM   outputs\disasternet\rerun_cells\<exp>\  params_<event>_seed<s>.csv rowparams_<event>_seed<s>.csv.gz loss_<event>_seed<s>.csv muni_ cells_
REM If stopped, run again: finished parts are skipped, cell training resumes from the 5-minute checkpoint.
REM After each part/seed: compare with old results (rerun_compare.csv), then disasternet.rerun_sync copies
REM small files to Google Drive (pilotA_rerun) and GitHub pc-results (pc_rerun). Keep this file ASCII only.

set O=outputs\disasternet
set M=%O%\rerun_muni
set N=%O%\rerun_cells
set P=pc_inputs
if not exist "%M%" mkdir "%M%"
if not exist "%N%" mkdir "%N%"
set LOG=%O%\rerun_log.txt

REM ---------------- Part 1: municipality (5 seeds in parallel, 1 thread each) ----------------
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
call :sync

REM ---------------- Part 2: cells ----------------
set L=--local raw\local_250m_byevent_usgs_aw.csv.gz --off-rule paper --jobs 5 --threads 2
for %%S in (0 1 2 3 4) do (
  call :stamp "===== seed %%S start"
  call :stamp "  cells LOCALU seed %%S"
  py -3.10 -m disasternet.run_cells_max %L% --seeds %%S --local-off --unc --out-dir %N%\LOCALU >> "%LOG%" 2>&1
  call :stamp "  cells DN seed %%S"
  py -3.10 -m disasternet.run_cells_max %L% --seeds %%S --out-dir %N%\DN >> "%LOG%" 2>&1
  call :stamp "  cells SOFTU seed %%S"
  py -3.10 -m disasternet.run_cells_max %L% --seeds %%S --soft --soft-mid --unc --out-dir %N%\SOFTU >> "%LOG%" 2>&1
  call :stamp "  cells SOFTU_off seed %%S"
  py -3.10 -m disasternet.run_cells_max %L% --seeds %%S --soft --soft-mid --unc --unc-mode off --out-dir %N%\SOFTU_off >> "%LOG%" 2>&1
  call :stamp "  cells SOFTU_var seed %%S"
  py -3.10 -m disasternet.run_cells_max %L% --seeds %%S --soft --soft-mid --unc --unc-mode var --out-dir %N%\SOFTU_var >> "%LOG%" 2>&1
  call :compare_cells
  call :sync
  call :stamp "===== seed %%S end"
)
call :stamp "ALL DONE"
pause
exit /b 0

:stamp
echo %~1  %date% %time%
echo %~1  %date% %time% >> "%LOG%"
exit /b 0

:muni
if exist "%M%\%1\predictions_seed4.csv" (
  call :stamp "  muni %1 already done"
  exit /b 0
)
call :stamp "  muni %1 start"
py -3.10 -m %~2 --out-dir %M%\%1 >> "%LOG%" 2>&1
call :stamp "  muni %1 end"
exit /b 0

:compare_muni
py -3.10 -m disasternet.rerun_compare --level muni --out %M%\rerun_compare.csv --pair DN %P%\muni_old\DN %M%\DN --pair SOFT %P%\muni_old\SOFT %M%\SOFT --pair DNU %P%\muni_old\DNU %M%\DNU --pair DNU_off %P%\muni_old\DNU_off %M%\DNU_off --pair DNU_var %P%\muni_old\DNU_var %M%\DNU_var --pair SOFTU %P%\muni_old\SOFTU %M%\SOFTU --pair SOFTU_off %P%\muni_old\SOFTU_off %M%\SOFTU_off --pair SOFTU_var %P%\muni_old\SOFTU_var %M%\SOFTU_var --pair DN_cvar01 %P%\muni_old\DN_cvar01 %M%\DN_cvar01 --pair DN_cvar001 %P%\muni_old\DN_cvar001 %M%\DN_cvar001 > %M%\rerun_compare.txt 2>&1
type %M%\rerun_compare.txt >> "%LOG%"
call :stamp "  muni compare saved: %M%\rerun_compare.csv"
exit /b 0

:compare_cells
py -3.10 -m disasternet.rerun_compare --level cells --out %N%\rerun_compare.csv --pair LOCALU %O%\cells_max_local_unc_paper_aw %N%\LOCALU --pair DN %O%\cells_max_aw %N%\DN --pair SOFTU %O%\cells_max_softunc_paper_aw_mid %N%\SOFTU --pair SOFTU_off %O%\cells_max_softunc_off_paper_aw_mid %N%\SOFTU_off --pair SOFTU_var %O%\cells_max_softunc_var_paper_aw_mid %N%\SOFTU_var > %N%\rerun_compare.txt 2>&1
type %N%\rerun_compare.txt
type %N%\rerun_compare.txt >> "%LOG%"
exit /b 0

:sync
py -3.10 -m disasternet.rerun_sync --root %O% >> "%LOG%" 2>&1
call :stamp "  synced to Google Drive and GitHub (see rerun_log.txt)"
exit /b 0
