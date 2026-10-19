@echo off
REM 칸 단위 DisasterNet 필수 실험 2개: 본 모델(soft + U, 논문 기준) → 교수님 기본판(DN)
REM 중간에 꺼져도 다시 실행하면 저장된 지점부터 이어서 돌아간다 (끝난 지진은 건너뜀).
REM JOBS = 동시에 돌릴 지진 수. 메모리 16GB면 3, 8GB면 1~2, 32GB 이상이면 5.
set JOBS=5
set O=outputs\disasternet

py -3.10 -m disasternet.run_cells_max --seeds 0 --jobs %JOBS% --soft --unc --off-rule paper --out-dir %O%\cells_max_softunc_paper
py -3.10 -m disasternet.run_cells_max --seeds 0 --jobs %JOBS% --out-dir %O%\cells_max
echo ALL DONE
pause
