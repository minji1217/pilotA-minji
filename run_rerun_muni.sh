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
python3 -m disasternet.run_input_v5 --no-bd $J --out-dir $N/DN
python3 -m disasternet.run_input_v5 --no-bd --terrain $T $J --out-dir $N/SOFT
python3 -m disasternet.run_muni_unc $U --no-bd $J --out-dir $N/DNU
python3 -m disasternet.run_muni_unc $U --no-bd --unc off $J --out-dir $N/DNU_끄기만
python3 -m disasternet.run_muni_unc $U --no-bd --unc var $J --out-dir $N/DNU_분산만
python3 -m disasternet.run_muni_unc $U --no-bd --terrain $T $J --out-dir $N/SOFTU
python3 -m disasternet.run_muni_unc $U --no-bd --unc off --terrain $T $J --out-dir $N/SOFTU_끄기만
python3 -m disasternet.run_muni_unc $U --no-bd --unc var --terrain $T $J --out-dir $N/SOFTU_분산만
python3 -m disasternet.run_muni_unc --no-bd --unc none --const-var 0.01 $J --out-dir $N/DN_같은분산0.01
python3 -m disasternet.run_muni_unc --no-bd --unc none --const-var 0.001 $J --out-dir $N/DN_같은분산0.001
python3 -m disasternet.rerun_compare --level muni \
  --pair DN $O/nobd_DN $N/DN --pair SOFT $O/nobd_SOFT $N/SOFT \
  --pair DNU $O/px_DNU $N/DNU --pair DNU_끄기만 $O/px_DNU_off $N/DNU_끄기만 --pair DNU_분산만 $O/px_DNU_var $N/DNU_분산만 \
  --pair SOFTU $O/px_SOFTU $N/SOFTU --pair SOFTU_끄기만 $O/px_SOFTU_off $N/SOFTU_끄기만 --pair SOFTU_분산만 $O/px_SOFTU_var $N/SOFTU_분산만 \
  --pair DN_같은분산0.01 $O/nobd_DN_cvar01 $N/DN_같은분산0.01 --pair DN_같은분산0.001 $O/nobd_DN_cvar001 $N/DN_같은분산0.001
echo ALLDONE
