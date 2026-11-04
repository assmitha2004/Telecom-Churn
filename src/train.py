import json, sys
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

df = pd.read_csv(sys.argv[1])
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df[df.tenure > 0].copy()  # 11 brand-new rows with blank TotalCharges
y = (df.Churn == "Yes").astype(int).values
num = ["tenure", "MonthlyCharges"]   # TotalCharges ~= tenure*Monthly; dropped to keep the form short & avoid collinearity
cat = ["SeniorCitizen","Partner","Dependents","PhoneService","MultipleLines","InternetService",
       "OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies",
       "Contract","PaperlessBilling","PaymentMethod"]
df["SeniorCitizen"] = df.SeniorCitizen.map({0:"No",1:"Yes"})
# collapse "No internet service"/"No phone service" into "No" (service flag captures it already)
for c in cat:
    df[c] = df[c].replace({"No internet service":"No","No phone service":"No"})
levels = {c: sorted(df[c].unique().tolist()) for c in cat}
base = {c: df[c].mode()[0] for c in cat}
def design(d):
    X = pd.DataFrame(index=d.index)
    X["tenure"] = d.tenure; X["MonthlyCharges"] = d.MonthlyCharges
    X["log_tenure"] = np.log1p(d.tenure)
    for c in cat:
        for l in levels[c]:
            if l != base[c]: X[f"{c}={l}"] = (d[c] == l).astype(float)
    return X
X = design(df)
mu, sd = X.mean(), X.std()
Xs = (X - mu) / sd
Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, stratify=y, random_state=42)
lr = LogisticRegression(C=0.5, max_iter=5000)
cv = StratifiedKFold(5, shuffle=True, random_state=42)
res = {}
for name, m in [("Logistic Regression", lr),
                ("Random Forest", RandomForestClassifier(400, min_samples_leaf=5, random_state=42, n_jobs=-1)),
                ("Gradient Boosting", GradientBoostingClassifier(random_state=42))]:
    auc = cross_val_score(m, Xtr, ytr, cv=cv, scoring="roc_auc")
    m.fit(Xtr, ytr); p = m.predict_proba(Xte)[:,1]
    res[name] = {"cv_auc": round(auc.mean(),4), "cv_auc_sd": round(auc.std(),4), "test_auc": round(roc_auc_score(yte,p),4)}
lr.fit(Xtr, ytr); p = lr.predict_proba(Xte)[:,1]
# pick operating threshold maximizing F1
ths = np.arange(0.2,0.7,0.01); f1s = [f1_score(yte, p>=t) for t in ths]; t = float(ths[int(np.argmax(f1s))])
def mets(th):
    pr = p>=th; tn,fp,fn,tp = confusion_matrix(yte,pr).ravel()
    return dict(threshold=round(th,2), accuracy=round(accuracy_score(yte,pr),4), precision=round(precision_score(yte,pr),4),
                recall=round(recall_score(yte,pr),4), f1=round(f1_score(yte,pr),4), tp=int(tp),fp=int(fp),fn=int(fn),tn=int(tn))
# risk band calibration on test set
bands = [(0,0.3,"Low"),(0.3,0.55,"Moderate"),(0.55,0.75,"High"),(0.75,1.01,"Critical")]
band_stats = []
for lo,hi,n in bands:
    m = (p>=lo)&(p<hi); band_stats.append(dict(band=n, lo=lo, hi=min(hi,1), customers=int(m.sum()), share=round(m.mean(),4), actual_churn=round(yte[m].mean(),4) if m.sum() else None))
# refit on full data for deployment, fold scaling into raw coefficients
final = LogisticRegression(C=0.5, max_iter=5000).fit(Xs, y)
w = final.coef_[0] / sd.values; b = final.intercept_[0] - np.sum(final.coef_[0]*mu.values/sd.values)
model = {"intercept": float(b), "coef": {k: float(v) for k,v in zip(X.columns, w)},
         "std_coef": {k: float(v) for k,v in zip(X.columns, final.coef_[0])},
         "means": {k: float(v) for k,v in mu.items()}, "levels": levels, "baseline": base,
         "threshold": round(t,2), "base_rate": float(y.mean())}
# segment insights
seg = {}
for c in ["Contract","InternetService","PaymentMethod","TechSupport","OnlineSecurity","SeniorCitizen","PaperlessBilling","Partner","Dependents"]:
    seg[c] = df.groupby(c).Churn.apply(lambda s: round((s=="Yes").mean(),4)).to_dict()
df["tenure_band"] = pd.cut(df.tenure,[0,6,12,24,48,72],labels=["0-6","7-12","13-24","25-48","49-72"])
seg["tenure_band"] = df.groupby("tenure_band",observed=True).Churn.apply(lambda s: round((s=="Yes").mean(),4)).to_dict()
out = {"n": int(len(df)), "churn_rate": round(y.mean(),4), "models": res, "op": mets(t), "default": mets(0.5),
       "bands": band_stats, "segments": seg,
       "top_std": sorted(model["std_coef"].items(), key=lambda kv: -abs(kv[1]))[:12]}
json.dump(model, open(sys.argv[2]+"/model.json","w"), indent=1)
json.dump(out, open(sys.argv[2]+"/metrics.json","w"), indent=1)
print(json.dumps(out, indent=1))
