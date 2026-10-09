"""Keystroke-dynamics user identification: 3 baselines vs 1 advanced (self-supervised) model.

Task : given the 31 timing numbers of one password typing, say WHICH of the 51 users typed it.
Data : CMU Keystroke Dynamics Benchmark (Killourhy and Maxion, 2009).
Setup: sessions 1-4 = training pool (labels hidden except k samples per user),
       sessions 5-8 = test (typed on LATER days, like a real login after enrolment).

Run  : python simple_project.py --data data/DSL-StrongPasswordData.csv
"""
import argparse, copy, time, warnings
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PowerTransformer
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
torch.set_num_threads(2)
ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/DSL-StrongPasswordData.csv")
ap.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20])   # labelled samples per user
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--ssl_epochs", type=int, default=40)
ap.add_argument("--members", type=int, default=3)                   # networks in the advanced ensemble
args = ap.parse_args()

# ------------------------------------------------------------------ data
df = pd.read_csv(args.data)
feats = [c for c in df.columns if c not in ("subject", "sessionIndex", "rep")]
X = df[feats].values.astype(float)
y = pd.factorize(df.subject)[0]
pool, test = np.where(df.sessionIndex <= 4)[0], np.where(df.sessionIndex > 4)[0]
prep = PowerTransformer().fit(X[pool])           # timings are very skewed -> Yeo-Johnson; fitted without labels
Xp, Xt, yp, yt = prep.transform(X[pool]), prep.transform(X[test]), y[pool], y[test]
Xp_raw = X[pool]
N = y.max() + 1


def enrol(k, seed):
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(np.where(yp == u)[0], k, replace=False) for u in range(N)]))


def evaluate(S):
    """S = (samples x users) scores. Returns accuracy, macro-F1, mean per-user EER, per-user accuracy."""
    pred = S.argmax(1)
    eers = []
    for u in range(N):
        fpr, tpr, _ = roc_curve(yt == u, S[:, u])
        i = np.argmin(np.abs((1 - tpr) - fpr)); eers.append((fpr[i] + 1 - tpr[i]) / 2)
    per_user = np.array([(pred[yt == u] == u).mean() for u in range(N)])
    return (pred == yt).mean(), f1_score(yt, pred, average="macro"), np.mean(eers), per_user


# ------------------------------------------------------------------ advanced model: contrastive pretraining + fine-tuning
class Net(nn.Module):
    def __init__(s, d=31, h=256, e=128):
        super().__init__()
        s.f = nn.Sequential(nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Linear(h, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Linear(h, e))

    def forward(s, x):
        return s.f(x)


def augment(raw, rng):
    """Two 'views' of the same typing sample: typed a bit faster/slower overall, small jitter on each timing, a few timings blanked."""
    z = raw * np.exp(rng.normal(0, .1, (len(raw), 1))) * np.exp(rng.normal(0, .1, raw.shape))
    z = prep.transform(z)
    z[rng.random(z.shape) < .15] = 0
    return torch.tensor(z, dtype=torch.float32)


def pretrain(seed, epochs, tau=0.2, bs=512):
    """SimCLR: no labels. Two views of the same sample must be close, views of different samples far apart."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net, proj = Net(), nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 64))
    opt = torch.optim.Adam(list(net.parameters()) + list(proj.parameters()), 1e-3, weight_decay=1e-5)
    for _ in range(epochs):
        v1, v2 = augment(Xp_raw, rng), augment(Xp_raw, rng)
        for idx in torch.tensor(rng.permutation(len(Xp_raw))).split(bs):
            if len(idx) < 32: continue
            z = F.normalize(proj(net(torch.cat([v1[idx], v2[idx]]))), dim=1)
            sim = z @ z.T / tau; sim.fill_diagonal_(-1e9)
            n = len(idx); tgt = torch.cat([torch.arange(n, 2 * n), torch.arange(n)])
            loss = F.cross_entropy(sim, tgt); opt.zero_grad(); loss.backward(); opt.step()
    return net.eval()


class Finetuned:
    """Pretrained (or random) network + linear layer, trained on the few labelled samples."""

    def __init__(self, net, seed, epochs=80):
        self.net, self.seed, self.epochs = copy.deepcopy(net), seed, epochs

    def fit(self, X_, y_):
        torch.manual_seed(self.seed); rng = np.random.default_rng(self.seed)
        self.head = nn.Linear(128, N)
        opt = torch.optim.Adam(list(self.net.parameters()) + list(self.head.parameters()), 3e-4, weight_decay=1e-3)
        X_, y_ = torch.tensor(X_, dtype=torch.float32), torch.tensor(y_)
        for _ in range(self.epochs):
            self.net.train()
            for idx in torch.tensor(rng.permutation(len(X_))).split(64):
                if len(idx) < 2: continue
                loss = F.cross_entropy(self.head(F.relu(self.net(X_[idx]))), y_[idx])
                opt.zero_grad(); loss.backward(); opt.step()
        self.net.eval(); return self

    def proba(self, X_):
        with torch.no_grad():
            return F.softmax(self.head(F.relu(self.net(torch.tensor(X_, dtype=torch.float32)))), 1).numpy()


def ensemble(nets, seed, Xl, yl):
    return np.mean([Finetuned(n, seed * 100 + i).fit(Xl, yl).proba(Xt) for i, n in enumerate(nets)], axis=0)


# ------------------------------------------------------------------ run
BASE = {"SVM": (SVC(probability=False), {"C": [1, 10, 100], "gamma": [0.005, 0.02, 0.05]}),
        "Random Forest": (RandomForestClassifier(n_estimators=200, n_jobs=1, random_state=0), {"max_depth": [None, 20], "max_features": ["sqrt", 0.5]}),
        "kNN": (KNeighborsClassifier(), {"n_neighbors": [1, 3, 7], "metric": ["euclidean", "manhattan"], "weights": ["uniform", "distance"]})}
rows, peruser = [], {}
t_start = time.time()
for seed in range(args.seeds):
    t0 = time.time()
    ssl_nets = [pretrain(seed * 10 + i, args.ssl_epochs) for i in range(args.members)]     # label-free
    rnd_nets = []
    for i in range(args.members):
        torch.manual_seed(seed * 10 + i); rnd_nets.append(Net().eval())                   # same network, NO pretraining
    t_ssl = time.time() - t0
    for k in args.ks:
        idx = enrol(k, seed); Xl, yl = Xp[idx], yp[idx]
        for name, (est, grid) in BASE.items():
            t0 = time.time()
            gs = GridSearchCV(est, grid, cv=min(5, k), n_jobs=1).fit(Xl, yl)               # tuned on enrolment samples only
            m = gs.best_estimator_
            S = m.decision_function(Xt) if name == "SVM" else m.predict_proba(Xt)
            rows.append((k, seed, name, *evaluate(S)[:3], time.time() - t0)); peruser[(k, seed, name)] = evaluate(S)[3]
        for name, nets in [("Advanced: SSL-pretrained net (ensemble)", ssl_nets), ("Ablation: same net, no SSL (ensemble)", rnd_nets)]:
            t0 = time.time(); S = ensemble(nets, seed, Xl, yl); dt = time.time() - t0 + (t_ssl / args.members * len(nets) / len(args.ks) if "SSL" in name else 0)
            rows.append((k, seed, name, *evaluate(S)[:3], dt)); peruser[(k, seed, name)] = evaluate(S)[3]
    print(f"seed {seed} done ({time.time() - t_start:.0f}s total)", flush=True)

R = pd.DataFrame(rows, columns=["k", "seed", "model", "accuracy", "macro_F1", "EER", "train_s"])
order = ["SVM", "Random Forest", "kNN", "Advanced: SSL-pretrained net (ensemble)", "Ablation: same net, no SSL (ensemble)"]
for k in args.ks:
    g = R[R.k == k].groupby("model")[["accuracy", "macro_F1", "EER", "train_s"]].agg(["mean", "std"]).reindex(order)
    print(f"\n=== {k} labelled samples per user  (mean +/- std over {args.seeds} seeds; test = sessions 5-8, {len(test)} samples) ===")
    print(f"{'model':42s} {'accuracy':>16s} {'macro-F1':>16s} {'EER (lower=better)':>20s} {'train s':>8s}")
    for m_, r in g.iterrows():
        print(f"{m_:42s} {r[('accuracy','mean')]:.3f} +/- {r[('accuracy','std')]:.3f}  {r[('macro_F1','mean')]:.3f} +/- {r[('macro_F1','std')]:.3f}  {r[('EER','mean')]:.3f} +/- {r[('EER','std')]:.3f}  {r[('train_s','mean')]:8.1f}")
    # is the advanced model better than the best baseline? paired test over the 51 users
    best = g.loc[order[:3], ("accuracy", "mean")].idxmax()
    adv = order[3]
    a = np.mean([peruser[(k, s, adv)] for s in range(args.seeds)], 0); b = np.mean([peruser[(k, s, best)] for s in range(args.seeds)], 0)
    diff = a.mean() - b.mean(); p = wilcoxon(a, b).pvalue
    print(f"-> Advanced vs best baseline ({best}): accuracy {diff:+.3f}, Wilcoxon over 51 users p = {p:.4f}  =>",
          "ADVANCED IS BETTER (significant)" if diff > 0 and p < .05 else "not significantly better" if diff > 0 else "advanced is NOT better")
    ab = g.loc[order[4], ("accuracy", "mean")]
    print(f"-> Effect of self-supervised pretraining alone (advanced minus same net without SSL): {g.loc[adv, ('accuracy','mean')] - ab:+.3f}")
R.to_csv("simple_results.csv", index=False)
print(f"\nTotal time {(time.time() - t_start) / 60:.1f} min. Raw numbers saved in simple_results.csv")
