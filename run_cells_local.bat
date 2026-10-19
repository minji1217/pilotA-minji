@echo off
REM 칸 단위 DisasterNet 실험 5개를 차례로 돌린다 (개인 PC용, Windows).
REM 저장소 맨 위 폴더에서 더블클릭하거나 명령 프롬프트에서 run_cells_local.bat 실행.
REM 중간에 꺼져도 다시 실행하면 저장된 지점부터 이어서 돌아간다 (끝난 지진은 건너뜀).
REM Python 버전이 다르면 아래 줄들의 "py -3.10"을 "python"으로 바꾸세요.
REM JOBS = 동시에 돌릴 지진 수. 메모리 16GB면 3, 8GB면 1~2, 32GB 이상이면 5.
set JOBS=3
set O=outputs\disasternet

py -3.10 -m disasternet.run_cells_max --seeds 0 --jobs %JOBS% --out-dir %O%\cells_max
py -3.10 -m disasternet.run_cells_max --seeds 0 --jobs %JOBS% --soft --unc --off-rule paper --out-dir %O%\cells_max_softunc_paper
py -3.10 -m disasternet.run_cells_max --seeds 0 --jobs %JOBS% --soft --unc --unc-mode off --off-rule paper --out-dir %O%\cells_max_softunc_off_paper
py -3.10 -m disasternet.run_cells_max --seeds 0 --jobs %JOBS% --soft --unc --unc-mode var --off-rule paper --out-dir %O%\cells_max_softunc_var_paper
py -3.10 -m disasternet.run_cells_max --seeds 0 --jobs %JOBS% --soft --unc --off-rule ours --out-dir %O%\cells_max_softunc
echo 모두 끝났습니다.
pause
