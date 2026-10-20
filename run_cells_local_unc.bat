@echo off
REM 칸 단위 LOCAL 끄기 + U (끄기는 논문 기준). soft 대신 끄기 지도로 비탈 칸 액상화·평지 칸 산사태를 0으로 고정.
REM JOBS x THREADS가 CPU 논리 프로세서 수(연구실 PC 12)를 넘지 않게 한다.
REM 중간에 꺼져도 다시 실행하면 저장된 지점부터 이어서 돌아간다 (끝난 지진은 건너뜀).
set JOBS=5
set THREADS=2
set O=outputs\disasternet

py -3.10 -m disasternet.run_cells_max --seeds 0 --jobs %JOBS% --threads %THREADS% --local-off --unc --off-rule paper --out-dir %O%\cells_max_local_unc_paper
echo ALL DONE
pause
