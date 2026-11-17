#!/usr/bin/env bash
# 표 3(RQ2) 학습 · 추론 시간을 한 컴퓨터에서 같은 날 모두 다시 재는 스크립트.
# 순서: 칸 LOCALU(DN-G) → 시정촌 SOFTU_분산만(DN-M) → AdminDER ② 라벨 미사용 → ② 라벨 사용 → ① 라벨 사용.
# 11/07 측정과 같은 조건: Xeon 2.10GHz 4코어, Python 3.13.16, PyTorch 2.14.1, 4스레드, seed 0, 3000에폭, 하나씩 차례로.
#
# 사용: bash run_time_all.sh <pilotA 폴더(followup4-lq-J-nolabels, raw · validation에 xlsx 3개)> <dpm_250m_v8_slim.csv.gz>
#   이 스크립트는 pilotA-minji 저장소 안에서 실행한다 (disasternet 코드와 raw · pc_inputs가 있는 곳).
#   결과: outputs/disasternet/time_all_<날짜>/ 에 로그와 환경 기록.
set -euo pipefail
PA=$(realpath "$1"); DPM=$(realpath "$2")
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
OUT=outputs/disasternet/time_all_$(date +%Y%m%d); mkdir -p "$OUT"

# 0. CPU 확인 (2.10GHz가 아니면 멈춤)
lscpu | grep "Model name" | tee "$OUT/env.txt"
grep -q "2.10GHz" "$OUT/env.txt" || { echo "2.10GHz 컴퓨터가 아님. 측정 안 함"; exit 1; }

# 1. Python 3.13.16 가상환경 (uv 최신판이 있어야 3.13.16을 받을 수 있음)
pip install -q -U uv
uv python install 3.13.16
[ -x /root/venv31316/bin/python ] || uv venv -p 3.13.16 /root/venv31316
VIRTUAL_ENV=/root/venv31316 uv pip install -q torch==2.14.1 numpy==2.5.3 pandas==3.0.5 scikit-learn openpyxl matplotlib scipy "setuptools>=77.0.3"
PY=/root/venv31316/bin/python
{ nproc; free -g | sed -n 2p; . /etc/os-release; echo "$PRETTY_NAME"
  $PY -c "import sys,torch,numpy,pandas;print('Python',sys.version.split()[0],'torch',torch.__version__,'numpy',numpy.__version__,'pandas',pandas.__version__)"; } | tee -a "$OUT/env.txt"

# 2. 입력 자리 잡기
cp pc_inputs/disasternet_input_v5.xlsx raw/DisasterNet_입력_v5.xlsx
cp pc_inputs/terrain_fractions.csv pc_inputs/muni_usgs_pixel.csv raw/
cp "$DPM" raw/dpm_250m_v8_slim.csv.gz
# time_cells_localu.py는 이미 끝난 지진을 건너뛰므로, 예전 측정 결과는 옆으로 치운다
[ -d outputs/disasternet/time_localu_full ] && mv outputs/disasternet/time_localu_full "outputs/disasternet/time_localu_full_old_$(date +%H%M%S)"
(cd disasternet && $PY -c "import sys; sys.argv=['x']; sys.path.insert(0,'.'); from time_compare import make_area; make_area('$PA')")

# 3. 측정 (하나씩 차례로)
echo "== DN-G 칸 LOCALU" | tee "$OUT/dn_g.log";  $PY disasternet/time_measure/time_cells_localu.py 2>&1 | tee -a "$OUT/dn_g.log"
cp outputs/disasternet/time_localu_full/time_localu_full.csv "$OUT/"
echo "== DN-M 시정촌 SOFTU_분산만" | tee "$OUT/dn_m.log"; $PY disasternet/time_measure/time_muni.py 4 2>&1 | tee -a "$OUT/dn_m.log"
for a in "dpm4" "dpm4 lqlab" "all8 lqlab"; do
  f="$OUT/adminder_${a// /_}.log"; echo "== AdminDER $a" | tee "$f"
  $PY disasternet/time_measure/time_pilota.py "$PA" $a 2>&1 | grep -v "Warning\|detach\|b_lq" | tee -a "$f"
done
echo "끝. 결과: $OUT"
