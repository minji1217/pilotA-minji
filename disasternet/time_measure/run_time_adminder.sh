#!/usr/bin/env bash
# AdminDER(pilotA A2) 학습 · 추론 시간 4가지를 한 컴퓨터에서 차례로 잰다.
#   ① 8개 학습 라벨 미사용(대조) → ② 4개 학습 라벨 미사용(대조) → ① 8개 학습 액상화 라벨 사용 → ② 4개 학습 액상화 라벨 사용
# 11/07 측정과 같은 조건: Xeon 2.10GHz 4코어, Python 3.13.16, PyTorch 2.14.1, 4스레드, seed 0, 3000에폭, 하나씩 차례로.
# 사용: bash run_time_adminder.sh <pilotA 폴더(followup4-lq-J-nolabels ad44fa2, raw · validation에 xlsx 3개)>
#   pilotA-minji 저장소 안에서 실행. 결과: outputs/disasternet/time_adminder_<날짜>/
set -euo pipefail
PA=$(realpath "$1")
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
OUT=outputs/disasternet/time_adminder_$(date +%Y%m%d); mkdir -p "$OUT"
lscpu | grep "Model name" | tee "$OUT/env.txt"
grep -q "2.10GHz" "$OUT/env.txt" || { echo "2.10GHz 컴퓨터가 아님. 측정 안 함"; exit 1; }
pip install -q -U uv
uv python install 3.13.16
[ -x /root/venv31316/bin/python ] || uv venv -p 3.13.16 /root/venv31316
VIRTUAL_ENV=/root/venv31316 uv pip install -q torch==2.14.1 numpy==2.5.3 pandas==3.0.5 scikit-learn openpyxl matplotlib scipy "setuptools>=77.0.3"
PY=/root/venv31316/bin/python
{ echo "CPU 수 $(nproc)"; free -g | sed -n 2p; . /etc/os-release; echo "$PRETTY_NAME"; echo "pilotA $(git -C "$PA" log --oneline -1)"
  $PY -c "import sys,torch,numpy,pandas;print('Python',sys.version.split()[0],'torch',torch.__version__,'numpy',numpy.__version__,'pandas',pandas.__version__,'threads',4)"; } | tee -a "$OUT/env.txt"
[ -f "$PA/raw/시정촌_면적.csv" ] || (cd disasternet && $PY -c "import sys; sys.argv=['x']; sys.path.insert(0,'.'); from time_compare import make_area; make_area('$PA')")
for a in "all8" "dpm4" "all8 lqlab" "dpm4 lqlab"; do
  f="$OUT/adminder_${a// /_}.log"; echo "== AdminDER $a" | tee "$f"
  $PY disasternet/time_measure/time_pilota.py "$PA" $a 2>&1 | grep -v "Warning\|detach\|b_lq" | tee -a "$f"
done
echo "끝. 결과: $OUT"
