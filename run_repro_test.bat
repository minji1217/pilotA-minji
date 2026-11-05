@echo off
chcp 65001 >nul
REM 칸 단위 LOCALU를 같은 PC에서 다시 돌렸을 때 예전 결과와 똑같이 나오는지 시험 (오사카, seed 0, 약 10분).
REM 실험 폴더(C:\work\pilotA-minji-claude-dazzling-babbage-8c6yd4)에서 실행. 예전 결과 폴더 outputs\disasternet\cells_max_local_unc_paper_aw 가 있어야 함.
REM 예전 실행과 같은 설정: --local ..._aw.csv.gz --off-rule paper --local-off --unc, 스레드 2, seed 0.
REM 새 결과는 outputs\disasternet\repro_test_localu 에 따로 저장 (예전 결과는 건드리지 않음). 학습된 파라미터는 model_2018오사카_seed0.pt 로 남음.
set O=outputs\disasternet
set NEW=%O%\repro_test_localu

if not exist "%O%\cells_max_local_unc_paper_aw" (
  echo 예전 결과 폴더 %O%\cells_max_local_unc_paper_aw 가 없습니다. 실험 폴더에서 실행했는지 확인하세요.
  pause
  exit /b 1
)

echo ===== 오사카 LOCALU seed 0 다시 학습 시작 %date% %time% =====
py -3.10 -m disasternet.run_cells_max --local raw\local_250m_byevent_usgs_aw.csv.gz --off-rule paper --local-off --unc --events "2018 오사카" --seeds 0 --jobs 1 --threads 2 --out-dir %NEW%
echo ===== 학습 끝 %date% %time% =====

py -3.10 -m disasternet.repro_check --old-dir %O%\cells_max_local_unc_paper_aw --new-dir %NEW% --event "2018 오사카" --seed 0 > %NEW%\repro_check.txt 2>&1
type %NEW%\repro_check.txt
echo.
echo 위 결과를 캡처해서 보내 주세요. (같은 내용이 %NEW%\repro_check.txt 에도 저장됨)
pause
