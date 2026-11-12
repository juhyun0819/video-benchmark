"""Extra ATUS 2025 stats: (a) weekend meal priors on non-working days, (b) weekday evening at-home activity after work."""
import sys, json
import numpy as np, pandas as pd
act_p, resp_p, per_p, out_p = sys.argv[1:5]
P = pd.read_csv(per_p, dtype={'id': str})
def wavg(x, w): return float(np.average(np.asarray(x, float), weights=w)) if len(x) else None
def wq(x, w, q):
    x = np.asarray(x, float); w = np.asarray(w, float); m = ~np.isnan(x); x, w = x[m], w[m]
    o = np.argsort(x); c = np.cumsum(w[o]) / w[o].sum(); return float(np.interp(q, c, x[o]))
hh = lambda v: f"{int(round(v))//60%24:02d}:{int(round(v))%60:02d}"
res = {}
# (a) weekend, not worked that day
E = P[(P.daytype == 'weekend') & (~P.worked)]
res['weekend_nonwork_n'] = int(len(E)); res['weekend_worked_share'] = wavg(P[P.daytype=='weekend'].worked, P[P.daytype=='weekend'].w)
LOC = {1: 'home', 2: 'workplace', 3: 'others_home', 4: 'restaurant'}
meals = {}
for m in ['breakfast', 'lunch', 'dinner']:
    M = E[E[f'{m}_has']]
    loc = {lab: round(wavg(M[f'{m}_loc'] == c, M.w), 3) for c, lab in LOC.items()}
    loc['other'] = round(1 - sum(loc.values()), 3)
    meals[m] = {'share_reporting': round(wavg(E[f'{m}_has'], E.w), 3), 'n_reporting': int(len(M)),
                'time_p25_p50_p75': [hh(wq(M[f'{m}_time'], M.w, q)) for q in (.25, .5, .75)], 'location_share': loc}
res['weekend_nonwork_meals'] = meals
# (b) weekday workers: at-home awake activity after last workplace work, excluding eating, sleep, travel
act = pd.read_csv(act_p, dtype={'TUCASEID': str, 'TRCODE': str})
act['start'] = act.TUCUMDUR24 - act.TUACTDUR24 + 240; act['stop'] = act.TUCUMDUR24 + 240
W = P[(P.daytype == 'weekday') & P.work_end.notna() & P.bed.notna()][['id', 'w', 'work_end', 'bed', 'dinner_has']]
a = act.merge(W, left_on='TUCASEID', right_on='id')
ev = a[(a.start >= a.work_end) & (a.stop <= a.bed) & (a.TEWHERE == 1)
       & (~a.TRCODE.str[:4].isin(['0101', '1101'])) & (a.TRCODE.str[:2] != '18')]
mins = ev.groupby('TUCASEID').TUACTDUR24.sum().reindex(W.id).fillna(0).values
kinds = ev.assign(t1=ev.TRCODE.str[:2]).groupby(['TUCASEID', 't1']).TUACTDUR24.sum().unstack(fill_value=0).reindex(W.id).fillna(0)
w = W.w.values
res['weekday_evening_home'] = {
    'n_workers': int(len(W)),
    'share_any_home_activity_ge30min': round(wavg(mins >= 30, w), 3),
    'share_any_home_activity_ge60min': round(wavg(mins >= 60, w), 3),
    'minutes_p25_p50_p75': [round(wq(mins, w, q)) for q in (.25, .5, .75)],
    'share_with_leisure_12_ge30': round(wavg(kinds.get('12', 0) >= 30, w), 3),
    'share_with_household_02_ge30': round(wavg(kinds.get('02', 0) >= 30, w), 3),
    'definition': 'at home (TEWHERE=1), between last workplace work end and bedtime, excluding sleep(0101), eating(1101), travel(18)'}
# (c) evening time budget check: work_end -> bed available minutes
avail = (W.bed - W.work_end).values
res['weekday_after_work_minutes_p10_p50_p90'] = [round(wq(avail, w, q)) for q in (.1, .5, .9)]
json.dump(res, open(out_p, 'w'), indent=1, ensure_ascii=False)
print(json.dumps(res, indent=1, ensure_ascii=False))
