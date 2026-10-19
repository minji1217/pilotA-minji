@echo off
REM 집 노트북용: 칸 단위 soft + U 세 번째 실험(분산만)만 돌린다. 끄기는 논문 기준.
REM JOBS = 동시에 돌릴 지진 수 (메모리가 부족하면 줄인다), THREADS = 지진 하나당 쓸 CPU 수.
REM JOBS x THREADS가 CPU 논리 프로세서 수를 넘지 않게 한다.
REM 중간에 꺼져도 다시 실행하면 저장된 지점부터 이어서 돌아간다 (끝난 지진은 건너뜀).
set JOBS=2
set THREADS=4
set O=outputs\disasternet

py -3.10 -m disasternet.run_cells_max --seeds 0 --jobs %JOBS% --threads %THREADS% --soft --unc --unc-mode var --off-rule paper --out-dir %O%\cells_max_softunc_var_paper
echo ALL DONE
pause
