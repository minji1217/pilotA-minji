@echo off
REM 칸 단위 ① DN (교수님 기본판) seed 1. seed 0과 같은 폴더에 _seed1 파일로 저장된다.
REM seed를 더 돌리려면 --seeds 1 을 --seeds 1 2 3 4 로 바꾼다.
REM 중간에 꺼져도 다시 실행하면 저장된 지점부터 이어서 돌아간다 (끝난 지진은 건너뜀).
set JOBS=5
set THREADS=2
set O=outputs\disasternet

py -3.10 -m disasternet.run_cells_max --seeds 1 --jobs %JOBS% --threads %THREADS% --out-dir %O%\cells_max
echo ALL DONE
pause
