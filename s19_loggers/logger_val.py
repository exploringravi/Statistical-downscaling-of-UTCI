"""Fixed-logger validation. Args: DIR_with_site_tifs OUTDIR
Extracts Ta_10m (calibrated) and Ta_10m_raw at each logger (3x3 cell mean, ~10 m position uncertainty),
the ERA5-Land value, and the logger mean over HH:00 +-30 min. Reports level and within-site contrast."""
import sys,glob,os,re,numpy as np,pandas as pd,rasterio
from pyproj import Transformer
D,OUT=sys.argv[1],sys.argv[2]; os.makedirs(OUT,exist_ok=True)
H=os.path.dirname(os.path.abspath(__file__))
P=pd.read_csv(f'{H}/logger_points.csv')
L22=pd.read_csv(f'{H}/loggers2022.csv',parse_dates=['t'])
core={'Gallenga':'P1','Lo_Chalet':'P2','Chalet':'P2','Ripa_di_Meana':'P3','Palazzina_Valitutti':'P4','Valitutti':'P4','Galilei_succursale':'P5','Succursale_Galilei':'P5','Succursale':'P5'}
L22['id']=L22.logger.map(core); L22=L22.dropna(subset=['id'])
L21=pd.read_csv(f'{H}/loggers2021.csv',index_col=0,parse_dates=True)
def obs(id_,t):
    lo,hi=t-pd.Timedelta(minutes=30),t+pd.Timedelta(minutes=30)
    if id_.startswith('S'):
        s=L21[id_]; s=s[(s.index>=lo)&(s.index<=hi)]
    else:
        s=L22[L22.id==id_].set_index('t')['T']; s=s[(s.index>=lo)&(s.index<=hi)]
    return s.mean() if len(s)>=5 else np.nan
tr=Transformer.from_crs(4326,32633,always_xy=True)
rows=[]
for f in sorted(glob.glob(f'{D}/UTCI_PerugiaLoggers2*_components*.tif')):
    m=re.search(r'PerugiaLoggers(2\d)_(\d{8})_(\d\d)00',f); camp,day,hh=m.groups()
    t=pd.Timestamp(f'{day[:4]}-{day[4:6]}-{day[6:]} {hh}:00')
    r=rasterio.open(f); B={d:i+1 for i,d in enumerate(r.descriptions)}
    era=glob.glob(f'{D}/ERA5_PerugiaLoggers{camp}_{day}_{hh}00_coarse.tif')
    pts=P[P.id.str.startswith('S' if camp=='21' else 'P')]
    for _,p in pts.iterrows():
        x,y=tr.transform(p.lon,p.lat); row,col=r.index(x,y)
        win=rasterio.windows.Window(col-1,row-1,3,3)
        val={k:float(np.nanmean(np.where(r.read(B[k],window=win)<=-9000,np.nan,r.read(B[k],window=win)))) for k in ('Ta_10m','Ta_10m_raw','RH_10m','MRT_10m','UTCI_10m') if k in B}
        e=np.nan
        if era:
            re_=rasterio.open(era[0]); bi=[i+1 for i,d in enumerate(re_.descriptions) if d in ('Ta','Ta_coarse','temperature_2m')] or [1]
            ex,ey=x,y; rr,cc=re_.index(ex,ey); e=float(re_.read(bi[0])[rr,cc])
        rows.append(dict(campaign=camp,time=t,hour=int(hh),id=p.id,logger=p.logger,obs=obs(p.id,t),era5=e,**val))
d=pd.DataFrame(rows); d.to_csv(f'{OUT}/logger_points_model.csv',index=False)
d=d.dropna(subset=['obs','Ta_10m'])
d['period']=np.where(d.hour==4,'night (04:00)','day (14:00/18:00)')
# within-site anomalies (each hour minus mean of loggers at that hour)
for c in ['obs','Ta_10m','Ta_10m_raw','era5']:
    d[c+'_a']=d[c]-d.groupby(['campaign','time'])[c].transform('mean')
out=[]
for (camp,per),g in d.groupby(['campaign','period']):
    rm=lambda a,b: float(np.sqrt(np.mean((g[a]-g[b])**2)))
    out.append(dict(campaign='20'+camp,period=per,n=len(g),hours=g.time.nunique(),
        bias_model=round((g.Ta_10m-g.obs).mean(),2),rmse_model=round(rm('Ta_10m','obs'),2),
        bias_era5=round((g.era5-g.obs).mean(),2),rmse_era5=round(rm('era5','obs'),2),
        sd_obs_contrast=round(g.obs_a.std(),2),sd_model_contrast=round(g.Ta_10m_a.std(),2),
        rmse_contrast_model=round(rm('Ta_10m_a','obs_a'),2),rmse_contrast_uniform=round(float(np.sqrt(np.mean(g.obs_a**2))),2),
        r_contrast=round(np.corrcoef(g.Ta_10m_a,g.obs_a)[0,1],2) if g.Ta_10m_a.std()>0 else np.nan,
        r_contrast_raw=round(np.corrcoef(g.Ta_10m_raw_a,g.obs_a)[0,1],2) if g.Ta_10m_raw_a.std()>0 else np.nan))
o=pd.DataFrame(out); o['skill_contrast']=(1-(o.rmse_contrast_model/o.rmse_contrast_uniform)**2).round(2)
pd.set_option('display.width',250); print(o.to_string()); o.to_csv(f'{OUT}/logger_summary.csv',index=False)
# mean anomaly per logger and period: does the model rank the loggers the same way?
m=d.groupby(['campaign','period','id'])[['obs_a','Ta_10m_a']].mean().round(2); print(m.to_string()); m.to_csv(f'{OUT}/logger_mean_anomaly.csv')
