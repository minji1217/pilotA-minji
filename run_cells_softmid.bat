@echo off
REM 칸 soft 다시: 중간 칸(5~10°)은 깎지 않음 (--soft-mid). ②③④만, seed 0 먼저 그다음 seed 1.
REM 결과 폴더 이름 끝에 _mid (기존 결과는 그대로 남음). 중간에 꺼져도 다시 실행하면 이어서 돌아간다.
REM raw\local_250m_byevent_usgs_aw.csv.gz 가 있어야 한다. 결과 업로드: push_results.bat
set JOBS=5
set THREADS=2
set O=outputs\disasternet
set L=--local raw\local_250m_byevent_usgs_aw.csv.gz --jobs %JOBS% --threads %THREADS% --soft --soft-mid --unc --off-rule paper

for %%S in (0 1) do (
  py -3.10 -m disasternet.run_cells_max %L% --seeds %%S --out-dir %O%\cells_max_softunc_paper_aw_mid
  py -3.10 -m disasternet.run_cells_max %L% --seeds %%S --unc-mode off --out-dir %O%\cells_max_softunc_off_paper_aw_mid
  py -3.10 -m disasternet.run_cells_max %L% --seeds %%S --unc-mode var --out-dir %O%\cells_max_softunc_var_paper_aw_mid
)
echo ALL DONE
pause
