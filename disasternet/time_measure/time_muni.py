# 시정촌 단위 DisasterNet(SOFTU_분산만 = 시정촌 최고)의 학습(3000 epoch)·추론(256 샘플) 시간, 지진별. seed 0, 4스레드 (pilotA·칸 측정과 같은 컴퓨터·스레드).
import time, sys, torch
sys.path.insert(0,'.')
import pandas as pd
from disasternet.run_input_v5 import load_table, attach_terrain
from disasternet.run_muni_unc import build_rows
from disasternet.model import DisasterNet, prepare_inputs, triplet_penalty
from schema import INDEX_TO_EVENT
torch.set_num_threads(int(sys.argv[1]) if len(sys.argv)>1 else 4)
df=attach_terrain(load_table('raw/DisasterNet_입력_v5.xlsx'),'raw/terrain_fractions.csv')
u=pd.read_csv('raw/muni_usgs_pixel.csv',dtype={'muni_code':str})[['이벤트','muni_code','n_on_ls','n_on_lq','var_ls','var_lq']]
df=df.merge(u,on=['이벤트','muni_code'],how='left',validate='one_to_one')
rows=build_rows(df,'DPM_배경차','LQ_prior_최대','var',None)
tot_tr=tot_inf=0
for e in sorted(rows.event_idx.unique().tolist()):
    g=rows.subset(rows.event_idx==e)
    torch.manual_seed(0); a,log_x=prepare_inputs(g,2.0)
    model=DisasterNet(g.n,num_flows=4,r0_var=1.0,use_bd=False,bd_var_learn=True); opt=torch.optim.Adam(model.parameters(),lr=0.01)
    t=time.time()
    for epoch in range(3000):
        beta=min(0.01+epoch/3000,1.0)
        post=model.posteriors(a['LS'],a['LQ'],a['BD'],8,off=g.off,r0_var=g.r0_var)
        q,keep=post['q'],post['keep']; rec=model.log_p_x(log_x,q)
        log_q=sum(post['log_q'][k].mean(1)*keep[k] for k in model.nodes)
        kl=log_q-model.log_p_z(q,a,g.log_frac,keep if g.off is not None else None)
        loss=(-rec+beta*kl).sum()+triplet_penalty(q,a,1.0); opt.zero_grad(); loss.backward(); opt.step()
    tr=time.time()-t; t=time.time()
    with torch.no_grad(): model.posteriors(a['LS'],a['LQ'],a['BD'],n_samples=256,off=g.off,r0_var=g.r0_var)
    inf=time.time()-t; tot_tr+=tr; tot_inf+=inf
    print(f'{INDEX_TO_EVENT[e]} 시정촌 {g.n}  학습 {tr:.1f}s  추론 {inf*1000:.0f}ms',flush=True)
print(f'합계: 학습 {tot_tr:.1f}s ({tot_tr/60:.1f}분)  추론 {tot_inf*1000:.0f}ms')
