# time_pilota.py와 같은 측정인데, 자료를 xlsx 대신 미리 만든 batch 파일(torch.save, gzip)에서 읽는다.
# 자료 읽기는 원래도 시간에서 빠지므로 학습 · 추론 시간 측정은 time_pilota.py와 같다.
# 사용: python time_pilota_from_batch.py <pilotA 폴더 (followup4-lq-J-nolabels)> <batch.pt.gz> [all8|dpm4] [lqlab]
import time, sys, os, io, gzip, contextlib, torch
W=os.path.abspath(sys.argv[1]); BF=os.path.abspath(sys.argv[2]); os.chdir(W); sys.path.insert(0,W)
os.makedirs("outputs",exist_ok=True)
import train as T
from loader import subset_batch
from schema import EVENT_TO_INDEX, PilotABatch
torch.set_num_threads(4)
t=time.time()
batch=PilotABatch(**torch.load(io.BytesIO(gzip.decompress(open(BF,"rb").read())),weights_only=False))
print(f'자료 읽기 {time.time()-t:.1f}s')
tt=ti=0
DPM=["2016 구마모토","2018 오사카","2018 훗카이도","2021 후쿠시마","2024 노토반도"]
MODE=sys.argv[3] if len(sys.argv)>3 else "all8"
LQLAB="lqlab" in sys.argv[4:]
idx=torch.tensor([EVENT_TO_INDEX[e] for e in DPM])
for ev in DPM:
    keep=(batch.event_idx!=EVENT_TO_INDEX[ev]) if MODE=="all8" else (torch.isin(batch.event_idx,idx)&(batch.event_idx!=EVENT_TO_INDEX[ev]))
    tb=subset_batch(batch,keep)
    t=time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        reg,like,pri,hist=T.train(batch=tb,seed=0,epochs=3000,lr=0.02,lam_gamma=10,area_mode="fixed-cov-mtn",lam_kappa=10,
                                  mtn_standardize="train",use_labels=True,lq_labels=LQLAB,lq_b_only=True)
    tr=time.time()-t
    t=time.time()
    with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
        out_r=reg(batch); out_l=like(batch,out_r.mu); log_w=T.prior_log_w(pri,batch)
        lj,lp=T.marginalize(log_w,out_l.log_L); p_ls,p_lq=T.infer(lj,lp)
    with contextlib.redirect_stdout(io.StringIO()):
        T.report_holdout(batch,reg,like,pri,p_ls,p_lq,ev,"_timing_"+MODE+("_lqlab" if LQLAB else ""))
    inf=time.time()-t
    tt+=tr; ti+=inf
    print(f'{ev} 학습행 {tb.batch_size}  학습 {tr:.1f}s  추론(α_e 맞춤 포함) {inf:.2f}s')
print(f'합계: 학습 {tt:.1f}s ({tt/60:.1f}분)  추론 {ti:.2f}s')
