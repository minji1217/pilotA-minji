@echo off
REM 칸 단위 seed 1 (USGS 면적 평균). raw\local_250m_byevent_usgs_aw.csv.gz 가 있어야 한다.
REM 순서: 1) ⑤ 끄기 지도 + U  2) ① baseline. 결과 폴더 이름 끝에 _aw (seed 0과 같은 폴더, 파일 이름에 seed1).
REM 중간에 꺼져도 다시 실행하면 저장된 지점부터 이어서 돌아간다 (끝난 지진·실험은 건너뜀).
REM 결과 자동 업로드: 다른 창에서 C:\work\pilotA-minji-git\push_results.bat (seed1 파일도 올림)
set JOBS=5
set THREADS=2
set O=outputs\disasternet
set L=--local raw\local_250m_byevent_usgs_aw.csv.gz --seeds 1 --jobs %JOBS% --threads %THREADS%

py -3.10 -m disasternet.run_cells_max %L% --local-off --unc --off-rule paper --out-dir %O%\cells_max_local_unc_paper_aw
py -3.10 -m disasternet.run_cells_max %L% --out-dir %O%\cells_max_aw
echo ALL DONE
pause
