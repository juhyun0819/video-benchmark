import sys, json
import numpy as np, pandas as pd
act_p, resp_p, out_p = sys.argv[1:4]
act = pd.read_csv(act_p, dtype={'TUCASEID':str,'TRCODE':str})
resp = pd.read_csv(resp_p, dtype={'TUCASEID':str}, usecols=['TUCASEID','TELFS','TRDPFTPT','TRHOLIDAY','TUDIARYDAY','TRMJOCC1','TUFINLWGT'])
act['start_off'] = act['TUCUMDUR24'] - act['TUACTDUR24']   # minutes since 04:00
act['stop_off']  = act['TUCUMDUR24']
act['t2'] = act['TRCODE'].str[:4]; act['t1'] = act['TRCODE'].str[:2]
ft = resp[(resp.TELFS==1)&(resp.TRDPFTPT==1)].copy()
ft['daytype'] = np.where(ft.TUDIARYDAY.isin([1,7]),'weekend', np.where(ft.TRHOLIDAY==0,'weekday','holiday'))
ft = ft[ft.daytype!='holiday']
a = act.merge(ft[['TUCASEID','daytype','TRMJOCC1','TUFINLWGT']], on='TUCASEID')

def clock(off): # minutes since 04:00 -> minutes since 00:00 (may exceed 1440)
    return off + 240
def wq(x, w, qs=(.1,.25,.5,.75,.9)):
    x=np.asarray(x,float); w=np.asarray(w,float); m=~np.isnan(x); x,w=x[m],w[m]
    if len(x)==0: return None
    o=np.argsort(x); x,w=x[o],w[o]; c=np.cumsum(w)/w.sum()
    mean=np.average(x,weights=w); sd=np.sqrt(np.average((x-mean)**2,weights=w))
    return {'n':int(len(x)),'mean':mean,'sd':sd, **{f'p{int(q*100)}':float(np.interp(q,c,x)) for q in qs}}
def hhmm(v):
    v=int(round(v)) % 1440; return f'{v//60:02d}:{v%60:02d}'
def fmt(d, clk=True):
    if d is None: return None
    return {k:(hhmm(v) if clk and k!='n' and k!='sd' else (round(v,1) if k!='n' else v)) for k,v in d.items()}

SLEEP = a.t2=='0101'
res={}
per=[]
for cid,g in a.groupby('TUCASEID', sort=False):
    g=g.sort_values('TUACTIVITY_N'); s=SLEEP.loc[g.index].values
    r={'id':cid,'daytype':g.daytype.iat[0],'occ':g.TRMJOCC1.iat[0],'w':g.TUFINLWGT.iat[0]}
    # wake: end of first sleep block if diary starts asleep
    if s[0]:
        i=0
        while i<len(s) and s[i]: i+=1
        r['wake']=clock(g.stop_off.iat[i-1]) if i<len(s) else np.nan
    else: r['wake']=np.nan
    # bedtime: start of final sleep block if diary ends asleep
    if s[-1] and not s.all():
        j=len(s)-1
        while j>=0 and s[j]: j-=1
        r['bed']=clock(g.start_off.iat[j+1])
    else: r['bed']=np.nan
    w = g[(g.t2=='0501')&(g.TEWHERE==2)]
    if len(w):
        r['work_start']=clock(w.start_off.min()); r['work_end']=clock(w.stop_off.max())
        r['work_min']=g[g.t2=='0501'].TUACTDUR24.sum()
        first=w.TUACTIVITY_N.min(); last=w.TUACTIVITY_N.max()
        # commute: contiguous travel(18xx) right before first work / after last work
        before=g[g.TUACTIVITY_N<first].iloc[::-1]; c=0
        for _,x in before.iterrows():
            if x.t1=='18': c+=x.TUACTDUR24
            else: break
        r['commute_in']=c if c>0 else np.nan
        after=g[g.TUACTIVITY_N>last]; c=0
        for _,x in after.iterrows():
            if x.t1=='18': c+=x.TUACTDUR24
            else: break
        r['commute_out']=c if c>0 else np.nan
    any_work = (g.t2=='0501').any(); r['worked']=bool(any_work)
    # meals
    e=g[g.t2=='1101']
    for name,lo,hi in [('breakfast',300,630),('lunch',630,900),('dinner',960,1320)]:
        m=e[(clock(e.start_off)>=lo)&(clock(e.start_off)<hi)]
        r[f'{name}_has']=len(m)>0
        r[f'{name}_time']=clock(m.start_off.iat[0]) if len(m) else np.nan
        loc=m.TEWHERE.iat[0] if len(m) else np.nan
        r[f'{name}_loc']=loc
    # leisure/sports, household
    sp=g[g.t2=='1301']
    r['sport_has']=len(sp)>0
    r['sport_time']=clock(sp.start_off.iat[0]) if len(sp) else np.nan
    r['sport_outdoor']=(sp.TEWHERE==9).any() if len(sp) else np.nan
    r['sport_min']=sp.TUACTDUR24.sum()
    r['house_min']=g[g.t1=='02'].TUACTDUR24.sum()
    r['leisure_min']=g[g.t1=='12'].TUACTDUR24.sum()
    r['social_out']=bool((g.TEWHERE==3).any() or ((g.t1=='12')&(g.TEWHERE.isin([4,9,11]))).any())
    r['away_all_day']=not (g.TEWHERE==1).any()
    per.append(r)
P=pd.DataFrame(per)
P.to_csv(out_p.replace('.json','_per_respondent.csv'),index=False)
LOC={1:'home',2:'workplace',3:'others_home',4:'restaurant'}
def share(df,col): return float(np.average(df[col].astype(float),weights=df.w)) if len(df) else None
for dt in ['weekday','weekend']:
    D=P[P.daytype==dt]; W=D[D.worked] if dt=='weekday' else D
    out={'n_respondents':int(len(D)),
         'wake':fmt(wq(D.wake,D.w)),'wake_sd_min':None,'bed':fmt(wq(D.bed,D.w)),
         'share_worked':share(D,'worked')}
    ws=wq(D.wake,D.w); bs=wq(D.bed,D.w)
    out['wake_sd_min']=round(ws['sd'],1); out['bed_sd_min']=round(bs['sd'],1)
    Wk=D[D.worked & D.work_start.notna()]
    if len(Wk):
        out['work_start']=fmt(wq(Wk.work_start,Wk.w)); out['work_end']=fmt(wq(Wk.work_end,Wk.w))
        out['work_minutes']=fmt(wq(Wk.work_min,Wk.w),clk=False)
        out['commute_in_min']=fmt(wq(Wk.commute_in,Wk.w),clk=False)
        out['commute_out_min']=fmt(wq(Wk.commute_out,Wk.w),clk=False)
    base = Wk if dt=='weekday' else D
    meals={}
    for mname in ['breakfast','lunch','dinner']:
        has=share(base,f'{mname}_has'); M=base[base[f'{mname}_has']]
        locs={}
        for code,lab in LOC.items():
            locs[lab]=round(float(np.average((M[f'{mname}_loc']==code).astype(float),weights=M.w)),3) if len(M) else None
        locs['other']=round(1-sum(v for v in locs.values() if v),3)
        meals[mname]={'share_reporting':round(has,3),'time':fmt(wq(M[f'{mname}_time'],M.w)),'location_share':locs}
    out['meals_'+('workday_workers' if dt=='weekday' else 'all')]=meals
    S=D[D.sport_has]
    out['sports']={'share':round(share(D,'sport_has'),3),'time':fmt(wq(S.sport_time,S.w)),
                   'outdoor_share':round(float(np.average(S.sport_outdoor.astype(float),weights=S.w)),3) if len(S) else None,
                   'minutes_if_any':fmt(wq(S.sport_min,S.w),clk=False)}
    out['household_minutes']=fmt(wq(D.house_min,D.w),clk=False)
    out['leisure_minutes']=fmt(wq(D.leisure_min,D.w),clk=False)
    out['share_social_out']=round(share(D,'social_out'),3)
    out['share_never_home']=round(share(D,'away_all_day'),3)
    out['share_missing_wake']=round(float(D.wake.isna().mean()),3)
    out['share_missing_bed']=round(float(D.bed.isna().mean()),3)
    res[dt]=out
# per-person mean? not possible (1 day each). occupation breakdown weekday workers
OCC={1:'mgmt_business_fin',2:'professional',3:'service',4:'sales',5:'office_admin',6:'farm_fish_forest',7:'construction_extraction',8:'install_maint_repair',9:'production',10:'transport_moving'}
occ={}
Wk=P[(P.daytype=='weekday')&P.worked&P.work_start.notna()]
for k,lab in OCC.items():
    O=Wk[Wk.occ==k]
    if len(O)<30: occ[lab]={'n':int(len(O)),'note':'n<30, not reported'}; continue
    occ[lab]={'n':int(len(O)),'work_start':fmt(wq(O.work_start,O.w)),'work_end':fmt(wq(O.work_end,O.w)),
              'work_minutes_p50':round(wq(O.work_min,O.w)['p50'])}
res['weekday_by_occupation']=occ
json.dump(res,open(out_p,'w'),indent=1,ensure_ascii=False)
print(json.dumps(res,indent=1,ensure_ascii=False))
