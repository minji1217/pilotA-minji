@echo off
chcp 65001 >nul
REM 칸 단위 seed 2, 3, 4 (seed 0~4 다섯 개 평균을 만들기 위해). 실험 폴더(C:\work\pilotA-minji-claude-dazzling-babbage-8c6yd4)에서 실행.
REM run_cells_softmid.bat(②③④ 고친 soft, seed 0·1)이 끝날 때까지 5분마다 확인하며 기다렸다가 시작한다. 지금 켜 두면 된다.
REM 실험: ⑤ 끄기 지도 + U, ② ③ ④ 고친 soft(--soft-mid), ① baseline. seed 하나씩 다섯 실험을 다 끝내고 다음 seed로 간다.
REM   (중간에 멈춰도 끝난 seed까지는 다섯 실험 모두 있음)
REM 결과 폴더: ⑤ ①은 기존 _aw 폴더, ② ③ ④는 _aw_mid 폴더. 파일 이름에 seed2 / seed3 / seed4.
REM 중간에 꺼져도 다시 실행하면 저장된 지점부터 이어서 돌아간다. 결과 업로드: push_results.bat
set JOBS=5
set THREADS=2
set O=outputs\disasternet
set L=--local raw\local_250m_byevent_usgs_aw.csv.gz --jobs %JOBS% --threads %THREADS% --off-rule paper
set S=--soft --soft-mid --unc

:wait
if exist "%O%\cells_max_softunc_var_paper_aw_mid\predictions_seed1.csv" goto run
echo %date% %time% run_cells_softmid.bat 끝나기를 기다리는 중 (5분마다 확인)
timeout /t 300 /nobreak >nul
goto wait

:run
for %%D in (2 3 4) do (
  echo ===== seed %%D 시작 %date% %time% =====
  py -3.10 -m disasternet.run_cells_max %L% --seeds %%D --local-off --unc --out-dir %O%\cells_max_local_unc_paper_aw
  py -3.10 -m disasternet.run_cells_max %L% --seeds %%D %S% --out-dir %O%\cells_max_softunc_paper_aw_mid
  py -3.10 -m disasternet.run_cells_max %L% --seeds %%D %S% --unc-mode off --out-dir %O%\cells_max_softunc_off_paper_aw_mid
  py -3.10 -m disasternet.run_cells_max %L% --seeds %%D --out-dir %O%\cells_max_aw
  py -3.10 -m disasternet.run_cells_max %L% --seeds %%D %S% --unc-mode var --out-dir %O%\cells_max_softunc_var_paper_aw_mid
)
echo ALL DONE
pause
