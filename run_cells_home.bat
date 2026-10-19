@echo off
REM 집 노트북용: 칸 단위 soft + U 세 번째 실험(분산만)만 돌린다. 끄기는 논문 기준.
REM 메모리가 부족하면 JOBS를 줄인다 (16 GB면 3, 8 GB면 2).
REM 중간에 꺼져도 다시 실행하면 저장된 지점부터 이어서 돌아간다 (끝난 지진은 건너뜀).
set JOBS=3
set O=outputs\disasternet

py -3.10 -m disasternet.run_cells_max --seeds 0 --jobs %JOBS% --soft --unc --unc-mode var --off-rule paper --out-dir %O%\cells_max_softunc_var_paper
echo ALL DONE
pause
