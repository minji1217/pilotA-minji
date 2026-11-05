#!/bin/bash
# 시정촌 단위 DisasterNet 실험 10개(PPT 1022_8)를 seed 0~4로 다시 돌려 파라미터 · loss CSV를 남기고, 예전 결과와 같은지 비교한다.
# 예전 결과를 만든 명령과 같은 설정. 새 결과: outputs/disasternet/rerun_muni/<실험>/ (예전 폴더는 건드리지 않음)
set -e
cd "$(dirname "$0")"
O=outputs/disasternet
N=$O/rerun_muni
T=raw/terrain_fractions.csv
U="--usgs-muni raw/muni_usgs_pixel.csv"
J="--jobs ${JOBS:-4}"
run() {  # run <실험 폴더 이름> <모듈과 옵션...>: seed 4 결과가 있으면 끝난 것으로 보고 건너뛴다
  local n=$1; shift
  [ -f "$N/$n/predictions_seed4.csv" ] && { echo "이미 있음 $n"; return; }
  python3 -m "$@" --out-dir "$N/$n"
}
run DN disasternet.run_input_v5 --no-bd $J
run SOFT disasternet.run_input_v5 --no-bd --terrain $T $J
run DNU disasternet.run_muni_unc $U --no-bd $J
run DNU_끄기만 disasternet.run_muni_unc $U --no-bd --unc off $J
run DNU_분산만 disasternet.run_muni_unc $U --no-bd --unc var $J
run SOFTU disasternet.run_muni_unc $U --no-bd --terrain $T $J
run SOFTU_끄기만 disasternet.run_muni_unc $U --no-bd --unc off --terrain $T $J
run SOFTU_분산만 disasternet.run_muni_unc $U --no-bd --unc var --terrain $T $J
run DN_같은분산0.01 disasternet.run_muni_unc --no-bd --unc none --const-var 0.01 $J
run DN_같은분산0.001 disasternet.run_muni_unc --no-bd --unc none --const-var 0.001 $J
python3 -m disasternet.rerun_compare --level muni \
  --pair DN $O/nobd_DN $N/DN --pair SOFT $O/nobd_SOFT $N/SOFT \
  --pair DNU $O/px_DNU $N/DNU --pair DNU_끄기만 $O/px_DNU_off $N/DNU_끄기만 --pair DNU_분산만 $O/px_DNU_var $N/DNU_분산만 \
  --pair SOFTU $O/px_SOFTU $N/SOFTU --pair SOFTU_끄기만 $O/px_SOFTU_off $N/SOFTU_끄기만 --pair SOFTU_분산만 $O/px_SOFTU_var $N/SOFTU_분산만 \
  --pair DN_같은분산0.01 $O/nobd_DN_cvar01 $N/DN_같은분산0.01 --pair DN_같은분산0.001 $O/nobd_DN_cvar001 $N/DN_같은분산0.001
echo ALLDONE
