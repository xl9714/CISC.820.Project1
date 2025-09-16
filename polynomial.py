import numpy as np
import matplotlib.pyplot as plt

path_train = "traindata.txt"
path_test = "testinputs.txt"

P_LIST = [0,1,2,3,4,5,6,7,8,9,10]
KSET   = [5,8,10]

def rd(p):
    M = np.loadtxt(p, dtype=float)
    if M.ndim == 1:
        M = M.reshape(1, -1)
    return M

def mkA(D, p, mu=None, sg=None):
    if mu is None or sg is None:
        mu = D.mean(axis=0)
        sg = D.std(axis=0)
        sg[sg < 1e-12] = 1.0
    Ds = (D - mu) / sg
    n, d = Ds.shape
    cols = [np.ones((n,1))]
    i = 1
    while i <= p:
        cols.append(Ds ** i)
        i += 1
    A = np.hstack(cols)
    return A, mu, sg

def slv(A, b):
    AT = A.T
    ATA = np.dot(AT, A)
    ATb = np.dot(AT, b)
    try:
        x = np.linalg.solve(ATA, ATb)
    except np.linalg.LinAlgError:
        m = ATA.shape[0]
        x = np.linalg.solve(ATA + 1e-8*np.eye(m), ATb)
    return x

def mse(u, v):
    r = u - v
    return float(np.dot(r, r) / len(v))

def k_splits(n, K):
    K = max(2, min(int(K), n))
    idx = np.arange(n)
    sizes = np.full(K, n // K, dtype=int)
    sizes[: n % K] += 1
    out = []
    s = 0
    for sz in sizes:
        test = idx[s:s+sz]
        train_mask = np.ones(n, dtype=bool)
        train_mask[test] = False
        train = idx[train_mask]
        if len(test) > 0 and len(train) > 0:
            out.append((train, test))
        s = s + sz
    return out

def cv_errs(D, b, p, K):
    e_train = []
    e_test = []
    for train, test in k_splits(len(D), K):
        Dtrain = D[train]
        btrain = b[train]
        Dtest = D[test]
        btest = b[test]
        A_train, mu, sg = mkA(Dtrain, p)
        A_test, _, _   = mkA(Dtest, p, mu, sg)   
        x = slv(A_train, btrain)
        e_train.append(mse(np.dot(A_train, x), btrain))
        e_test.append(mse(np.dot(A_test, x), btest))
    return np.array(e_train), np.array(e_test)

train = rd(path_train)
test = rd(path_test)
X  = train[:, :-1]
y  = train[:, -1]
Xtest = test
n = len(X)
d = X.shape[1]

best_p = None
best_K = None
best_cv = None
fold_train_best = None
fold_test_best = None

RtrainK = {kk: {} for kk in KSET}
RtestK = {kk: {} for kk in KSET}
best_p_per_K = {kk: None for kk in KSET}
best_cv_per_K = {kk: None for kk in KSET}

for K in KSET:
    for p in P_LIST:
        a,b_ = cv_errs(X, y, p, K)  
        train_m = float(a.mean())
        test_m = float(b_.mean())
        RtrainK[K][p] = train_m
        RtestK[K][p] = test_m
        if (best_cv is None) or (test_m < best_cv):
            best_cv = test_m
            best_p = p
            best_K = K
            fold_train_best = a
            fold_test_best = b_
        if (best_cv_per_K[K] is None) or (test_m < best_cv_per_K[K]):
            best_cv_per_K[K] = test_m
            best_p_per_K[K] = p

print("best:", {"p": best_p, "K": best_K, "Ecv": best_cv})

A_all, mu_all, sg_all = mkA(X, best_p)
x_all = slv(A_all, y)
Ein = mse(np.dot(A_all, x_all), y)

A_test, _, _ = mkA(Xtest, best_p, mu_all, sg_all)
yhat = np.dot(A_test, x_all)

m_train = float(fold_train_best.mean())
m_test  = float(fold_test_best.mean())

with open("report.txt","w") as f:
    f.write("Polynomial\n")
    f.write(f"n={n}, d={d}\n")
    f.write(f"Selected: p={best_p}, K={best_K}\n")
    f.write(f"Ein={Ein:.6f}\n")
    f.write(f"CV train mean = {m_train:.6f}\n")
    f.write(f"Predicted test error (CV mean) = {m_test:.6f}\n")

for KK in KSET:
    ps = sorted(RtestK[KK].keys())
    if not ps:
        continue
    yy1 = [RtrainK[KK][pp] for pp in ps]
    yy2 = [RtestK[KK][pp] for pp in ps]
    plt.figure()
    plt.plot(ps, yy1, label="training error")
    plt.plot(ps, yy2, label="test error")
    plt.xlabel("p")
    plt.ylabel("R")
    plt.title("hold-out set")
    plt.legend()
    plt.grid(True)
    plt.xticks(ps)
    plt.savefig(f"holdout_R_vs_p_K{KK}.png", bbox_inches="tight")
    with open(f"holdout_R_vs_p_K{KK}.txt","w") as f:
        for pp in ps:
            f.write(f"p={pp}, R_train={RtrainK[KK][pp]:.6f}, R_test={RtestK[KK][pp]:.6f}\n")

j = 0
for KK in KSET:
    pK = best_p_per_K[KK]
    if pK is None:
        continue
    splits = k_splits(n, KK)
    train_idx, test_idx = splits[0]
    Xtrain = X[train_idx]
    ytrain = y[train_idx]
    Xho = X[test_idx]
    yho = y[test_idx]
    A_train, muK, sgK = mkA(Xtrain, pK)
    A_ho, _, _ = mkA(Xho, pK, muK, sgK)
    xK = slv(A_train, ytrain)
    R_ho = mse(np.dot(A_ho, xK), yho)

    xj_lo = float(min(Xtrain[:, j].min(), Xho[:, j].min()))
    xj_hi = float(max(Xtrain[:, j].max(), Xho[:, j].max()))
    xx = np.linspace(xj_lo, xj_hi, 200)
    tmp = np.tile(muK, (len(xx), 1))
    tmp[:, j] = xx
    A_line, _, _ = mkA(tmp, pK, muK, sgK)
    y_line = np.dot(A_line, xK)

    tmp2 = np.tile(muK, (len(Xho), 1))
    tmp2[:, j] = Xho[:, j]
    A_st, _, _ = mkA(tmp2, pK, muK, sgK)
    y_st = np.dot(A_st, xK)

    plt.figure()
    plt.title(f"R = {R_ho:.6f} (test)")
    plt.plot(xx, y_line, linewidth=1.5, label=f"order {pK} polynomial")
    plt.scatter(Xtrain[:, j], ytrain, s=25, label="train data")
    plt.scatter(Xho[:, j], yho, s=25, facecolors='none', edgecolors='r', label="test data")
    ii = 0
    while ii < len(Xho):
        xi = Xho[ii, j]
        yi = yho[ii]
        yfit = y_st[ii]
        plt.plot([xi, xi], [yi, yfit], linestyle="--", linewidth=0.8)
        ii += 1
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig(f"fit_1d_holdout_K{KK}.png", bbox_inches="tight")
