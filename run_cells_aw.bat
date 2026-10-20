@echo off
REM 칸 단위 실험을 USGS 면적 가중평균 값으로 다시 (seed 0). raw\local_250m_byevent_usgs_aw.csv.gz 가 있어야 한다.
REM 순서: 1) LOCAL + U  2) ① DN  3) SOFTU  4) SOFTU_끄기만  5) SOFTU_분산만. 결과 폴더 이름 끝에 _aw.
REM 중간에 꺼져도 다시 실행하면 저장된 지점부터 이어서 돌아간다 (끝난 지진·실험은 건너뜀).
set JOBS=5
set THREADS=2
set O=outputs\disasternet
set L=--local raw\local_250m_byevent_usgs_aw.csv.gz --seeds 0 --jobs %JOBS% --threads %THREADS%

py -3.10 -m disasternet.run_cells_max %L% --local-off --unc --off-rule paper --out-dir %O%\cells_max_local_unc_paper_aw
py -3.10 -m disasternet.run_cells_max %L% --out-dir %O%\cells_max_aw
py -3.10 -m disasternet.run_cells_max %L% --soft --unc --off-rule paper --out-dir %O%\cells_max_softunc_paper_aw
py -3.10 -m disasternet.run_cells_max %L% --soft --unc --unc-mode off --off-rule paper --out-dir %O%\cells_max_softunc_off_paper_aw
py -3.10 -m disasternet.run_cells_max %L% --soft --unc --unc-mode var --off-rule paper --out-dir %O%\cells_max_softunc_var_paper_aw
echo ALL DONE
pause
