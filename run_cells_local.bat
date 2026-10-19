@echo off
REM 칸 단위 DisasterNet: soft + U (끄기는 논문 기준 = 불확실도 0·999만) 세 가지
REM   1) 끄기 + 분산 (본 모델)  2) 끄기만  3) 분산만
REM 중간에 꺼져도 다시 실행하면 저장된 지점부터 이어서 돌아간다 (끝난 지진은 건너뜀).
set JOBS=5
set O=outputs\disasternet

py -3.10 -m disasternet.run_cells_max --seeds 0 --jobs %JOBS% --soft --unc --off-rule paper --out-dir %O%\cells_max_softunc_paper
py -3.10 -m disasternet.run_cells_max --seeds 0 --jobs %JOBS% --soft --unc --unc-mode off --off-rule paper --out-dir %O%\cells_max_softunc_off_paper
py -3.10 -m disasternet.run_cells_max --seeds 0 --jobs %JOBS% --soft --unc --unc-mode var --off-rule paper --out-dir %O%\cells_max_softunc_var_paper
echo ALL DONE
pause
