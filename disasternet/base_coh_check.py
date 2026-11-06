# base상관도 기준으로 평가 시정촌을 거르면 AUC가 어떻게 바뀌는지 (USGS · DPM만 · pilotA · DisasterNet 시정촌 · 칸). 사용: python disasternet/base_coh_check.py <pilotA_latest와 basechk/localu_*.csv가 있는 폴더>
import sys, pandas as pd, numpy as np, glob
sys.path.insert(0,'.')
from sklearn.metrics import roc_auc_score as A
from disasternet.run_input_v5 import load_table
import sys as _s; S=_s.argv[1] if len(_s.argv)>1 else '.'
P=S+'/pilotA_latest/outputs/'
ls=pd.concat([pd.read_csv(f,dtype={'muni_code':str}) for f in glob.glob(P+'holdout_detail_lqnolab_holdout-*.csv')])
lq=pd.concat([pd.read_csv(f,dtype={'muni_code':str}) for f in glob.glob(P+'holdout_detail_lq_lqnolab_holdout-*.csv')])
lab=pd.concat([ls[['event','muni_code','ls_true','p_ls_alpha_fit']].rename(columns={'ls_true':'y','p_ls_alpha_fit':'pilotA'}).assign(h='LS'),
               lq[['event','muni_code','lq_true','p_lq_alpha_fit']].rename(columns={'lq_true':'y','p_lq_alpha_fit':'pilotA'}).assign(h='LQ')])
lab['muni_code']=lab.muni_code.str.zfill(5)
x=pd.read_excel('raw/DPM_municipality_v8.xlsx',sheet_name='집계'); x['muni_code']=x['시정촌코드'].astype(str).str.zfill(5)
lab=lab.merge(x[['이벤트','muni_code','base상관도']].rename(columns={'이벤트':'event'}),on=['event','muni_code'])
t=load_table('raw/DisasterNet_입력_v5.xlsx')[['이벤트','muni_code','event_idx','LS_prior_평균','LQ_prior_최대','DPM_배경차']].rename(columns={'이벤트':'event'})
lab=lab.merge(t,on=['event','muni_code'])
lab['USGS']=np.where(lab.h=='LS',lab.LS_prior_평균,lab.LQ_prior_최대)
lab['DPM만']=lab.DPM_배경차
def addpred(name,files,maxcol=True):
    vals=[]
    for f in files:
        p=pd.read_csv(f,dtype={'muni_code':str}); p['muni_code']=p.muni_code.str.zfill(5)
        vals.append(p.set_index(['event_idx','muni_code'])[['p_ls','p_lq']])
    for i,v in enumerate(vals):
        m=lab.join(v,on=['event_idx','muni_code'])
        lab[f'{name}_s{i}']=np.where(lab.h=='LS',m.p_ls,m.p_lq)
    return [f'{name}_s{i}' for i in range(len(vals))]
dn=addpred('DN시정촌',sorted(glob.glob('outputs/disasternet/px_SOFTU_var/predictions_seed*.csv')))
lc=addpred('LOCALU',sorted(glob.glob(S+'/basechk/localu_*.csv')))
print('rows',len(lab), lab.groupby('h').size().to_dict(), 'NaN', lab[dn+lc].isna().sum().sum())
def wauc(d,col):
    num=den=0;out={}
    for (e,h),g in d.groupby(['event','h']):
        if g.y.nunique()<2: continue
        a=A(g.y,g[col]); out[(e,h)]=(a,len(g)); 
    r={}
    for H in ['LS','LQ','ALL']:
        it=[(a,n) for (e,h),(a,n) in out.items() if H=='ALL' or h==H]
        r[H]=sum(a*n for a,n in it)/sum(n for a,n in it) if it else np.nan
    return r,out
models={'USGS':['USGS'],'DPM만':['DPM만'],'pilotA':['pilotA'],'DN시정촌 SOFTU_분산만':dn,'DN칸 LOCALU':lc}
for th in [None,0.3,0.4]:
    d=lab if th is None else lab[lab['base상관도']>=th]
    cnt={h:(len(g), int(g.y.sum())) for h,g in d.groupby('h')}
    print(f'\n== base 기준 {th} : 남은 행(양성) {cnt}')
    for k,cols in models.items():
        rs=[wauc(d,c)[0] for c in cols]
        print(f'  {k:22s}', ' '.join(f"{H} {np.mean([r[H] for r in rs]):.3f}" for H in ['ALL','LS','LQ']))
print('\n평가 행 중 base<0.4 비율', lab.assign(lo=lab['base상관도']<0.4).groupby('h').lo.mean().round(3).to_dict(), ' <0.3', lab.assign(lo=lab['base상관도']<0.3).groupby('h').lo.mean().round(3).to_dict())
print('\nLS 지진별 (전체 행)')
for e,g in lab[lab.h=='LS'].groupby('event'):
    if g.y.nunique()<2: continue
    print(f'  {e} n={len(g)} 양성={int(g.y.sum())} base중앙 {g["base상관도"].median():.3f}', ' '.join(f"{k} {np.mean([A(g.y,g[c]) for c in cols]):.3f}" for k,cols in models.items()))
print('LQ 지진별')
for e,g in lab[lab.h=='LQ'].groupby('event'):
    if g.y.nunique()<2: continue
    print(f'  {e} n={len(g)} 양성={int(g.y.sum())} base중앙 {g["base상관도"].median():.3f}', ' '.join(f"{k} {np.mean([A(g.y,g[c]) for c in cols]):.3f}" for k,cols in models.items()))
# DPM vs base within LS positives: correlation
for h in ['LS','LQ']:
    g=lab[lab.h==h]; print(h,'DPM-base 상관(스피어만)',round(g['DPM_배경차'].corr(g['base상관도'],method='spearman'),3), ' 양성 중 base<0.4 비율',round((g[g.y==1]['base상관도']<0.4).mean(),3),' 음성 중',round((g[g.y==0]['base상관도']<0.4).mean(),3))
print()
for th in [None,0.3,0.4]:
    d=lab if th is None else lab[lab['base상관도']>=th]
    out=[]
    for (e,h),g in d.groupby(['event','h']):
        if g.y.nunique()<2: continue
        out.append(f"{h} {e[5:]} {len(g)}({int(g.y.sum())}+)")
    print(th, ' | '.join(out))
