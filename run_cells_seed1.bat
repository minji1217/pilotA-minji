@echo off
REM 칸 단위 seed 1 두 개를 차례로: 1) LOCAL + U (가장 높은 설정)  2) ① DN (교수님 기본판)
REM 결과는 각 폴더에 _seed1 파일로 저장된다 (seed 0 파일과 같은 폴더).
REM 중간에 꺼져도 다시 실행하면 저장된 지점부터 이어서 돌아간다 (끝난 지진은 건너뜀).
set JOBS=5
set THREADS=2
set O=outputs\disasternet

py -3.10 -m disasternet.run_cells_max --seeds 1 --jobs %JOBS% --threads %THREADS% --local-off --unc --off-rule paper --out-dir %O%\cells_max_local_unc_paper
py -3.10 -m disasternet.run_cells_max --seeds 1 --jobs %JOBS% --threads %THREADS% --out-dir %O%\cells_max
echo ALL DONE
pause
