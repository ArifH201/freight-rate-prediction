from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import KFold
sys.path.insert(0, str(Path(__file__).parent))
from freight_model import lane_features, LGB_PARAMS
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"outputs"
tr=pd.read_csv(ROOT/"data/train_test.csv",parse_dates=["date"]); va=pd.read_csv(ROOT/"data/validation.csv",parse_dates=["date"])
TEAL,GREY,RED="#064A56","#9DAFB3","#C0392B"

# Fig 1: quote_signal vs distance correlation, by month (train) and validation months
tr["m"]=tr.date.dt.month; va["m"]=va.date.dt.month
c=[np.corrcoef(np.log(g.quote_signal),np.log(g.distance))[0,1] for _,g in tr.groupby("m")]
cv_=[np.corrcoef(np.log(g.quote_signal),np.log(g.distance))[0,1] for _,g in va.groupby("m")]
fig,ax=plt.subplots(figsize=(7.2,3.1),dpi=200)
lab=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov*","Dec*"]
ax.bar(range(10),c,color=[TEAL if v<-0.5 else (RED if v>0.5 else GREY) for v in c])
ax.bar([10,11],cv_,color=GREY,hatch="//",edgecolor="white")
ax.axhline(0,color="#455A60",lw=.8); ax.set_xticks(range(12)); ax.set_xticklabels(lab,fontsize=8)
ax.set_ylabel("corr(log quote_signal, log distance)",fontsize=8); ax.tick_params(axis="y",labelsize=8)
ax.set_title("quote_signal flips meaning month to month (and is noise in Nov/Dec)",loc="left",fontsize=10,fontweight="bold")
ax.spines[["top","right"]].set_visible(False); ax.text(0.99,0.03,"* validation months (unlabeled)",transform=ax.transAxes,ha="right",fontsize=7.5,color="#455A60")
fig.tight_layout(); fig.savefig(OUT/"fig_quote_signal.png"); plt.close(fig)

# Fig 2: split schematic
fig,ax=plt.subplots(figsize=(7.2,2.7),dpi=200)
months=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
for i,mn in enumerate(months): ax.add_patch(plt.Rectangle((i,2.4),1,0.5,fc=TEAL if i<10 else RED,ec="white")); ax.text(i+.5,2.65,mn,ha="center",va="center",color="white",fontsize=7.5)
ax.text(0,3.05,"Provided data: labeled train_test.csv (Jan-Oct)  |  unlabeled validation.csv (Nov-Dec)",fontsize=8,color="#455A60")
for r,k in enumerate([6,7,8,9,10]):
    y=1.8-r*0.42
    ax.add_patch(plt.Rectangle((0,y),k-1,0.3,fc=TEAL,alpha=.35,ec="white")); ax.add_patch(plt.Rectangle((k-1,y),1,0.3,fc="#E8A33D",ec="white"))
    ax.text(-0.15,y+.15,f"Fold {r+1}",ha="right",va="center",fontsize=7.5)
ax.text(10.1,1.2,"train on all earlier months\n(light teal) -> test on next\nmonth (orange)",fontsize=7.5,color="#455A60",va="center")
ax.set_xlim(-1.2,13.5); ax.set_ylim(-0.4,3.3); ax.axis("off")
fig.tight_layout(); fig.savefig(OUT/"fig_split.png"); plt.close(fig)

# spike share (random 5-fold OOF lane model, no date info)
X=lane_features(tr); y=np.log(tr.posted_rate.values); res=np.zeros(len(y))
for a,b in KFold(5,shuffle=True,random_state=0).split(X):
    res[b]=y[b]-lgb.LGBMRegressor(**LGB_PARAMS).fit(X.iloc[a],y[a]).predict(X.iloc[b])
print("share |resid|>0.3:",np.mean(np.abs(res-np.median(res))>0.3).round(4),"  >0.5:",np.mean(np.abs(res-np.median(res))>0.5).round(4))
print("high spikes (>+0.3):",np.mean(res>0.3).round(4)," low (<-0.3):",np.mean(res<-0.3).round(4))
