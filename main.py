"""
Social Sensitivity Centrality -- main experiment (Sec. IV).

Zachary's Karate Club, actuated stochastic Friedkin--Johnsen dynamics
    x_{t+1} = (I - Lambda)(A x_t + B u_t) + Lambda d + w_t,
regulated by infinite-horizon average-cost LQR. Computes the four SSC metrics in
closed form, compares them with Degree, PageRank, Betweenness and FJ social power,
and validates them against finite interventions that re-solve the DARE.
"""
import csv
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov
from scipy.stats import kendalltau

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
LEADERS = [0, 33]               # faction leaders: actuated agents
LAMBDA_LEADER = 0.4             # leader stubbornness
LAMBDA_FOLLOWER_MAX = 0.1       # follower stubbornness ~ U(0, LAMBDA_FOLLOWER_MAX)
Q_LEADER = 5.0                  # state weight at the leaders
Q_FOLLOWER = 10.0               # state weight at every other agent
R_WEIGHT = 1.0                  # R = R_WEIGHT * I_m
BIAS = {0: +1.0, 33: -1.0}      # internal biases (enter s only)
WEIGHT = "weight"               # Zachary interaction counts
MODEL_SEED = 42

# Finite interventions (Sec. IV-A)
TRUST_SHIFT = 0.1               # eta: fraction of trust each listener moves to the target
STUB_STEP = 0.1                 # lambda_i <- lambda_i + STUB_STEP
ACT_STEP = 0.1                  # B_i. <- B_i. + ACT_STEP * c*, c* maximizing unit direction (Def. 4)
NOISE_STEP = 0.1                # W_ii <- W_ii - NOISE_STEP
K_MAX = 5                       # largest budget
N_RANDOM = 20                   # random selections per budget
RANDOM_SEED = 0

# Robustness check: implementation noise on the interventions
NOISE_REL = 0.5                 # realized step = nominal * (1 + eps), eps ~ U(-NOISE_REL, NOISE_REL)
DIR_NOISE = 0.3                 # actuator direction c* + DIR_NOISE * g, renormalized
N_TRIALS = 20

# Actuator placement (Prop. 1): gains of the added channel
PLACE_BETAS = [0.3, 1.0]


# ------------------------------------------------------------------
# MODEL AND LQ SOLUTION
# ------------------------------------------------------------------
def build_model(G):
    n = G.number_of_nodes()
    adj = nx.to_numpy_array(G, nodelist=range(n), weight=WEIGHT)
    A = adj / adj.sum(axis=1, keepdims=True)

    rng = np.random.default_rng(MODEL_SEED)
    lam = rng.uniform(0.0, LAMBDA_FOLLOWER_MAX, n)
    lam[LEADERS] = LAMBDA_LEADER

    B = np.zeros((n, len(LEADERS)))
    B[LEADERS, range(len(LEADERS))] = 1.0

    q = np.full(n, Q_FOLLOWER)
    q[LEADERS] = Q_LEADER
    d = np.zeros(n)
    for i, v in BIAS.items():
        d[i] = v
    return dict(A=A, lam=lam, B=B, Q=np.diag(q), R=R_WEIGHT * np.eye(len(LEADERS)), W=np.eye(n), d=d)


def solve_lq(A, lam, B, Q, R, W):
    """Stabilizing DARE solution, gain, closed loop, stationary covariance, J* = Tr(PW)."""
    L = np.diag(1.0 - lam)
    At, Bt = L @ A, L @ B
    P = solve_discrete_are(At, Bt, Q, R)
    K = np.linalg.solve(R + Bt.T @ P @ Bt, Bt.T @ P @ At)
    Acl = At - Bt @ K
    Sigma = solve_discrete_lyapunov(Acl, W)
    return dict(At=At, P=P, K=K, Acl=Acl, Sigma=Sigma, J=float(np.trace(P @ W)))


def open_loop_cost(mdl):
    At = np.diag(1.0 - mdl["lam"]) @ mdl["A"]
    return float(np.trace(solve_discrete_lyapunov(At.T, mdl["Q"]) @ mdl["W"]))


# ------------------------------------------------------------------
# SOCIAL SENSITIVITY CENTRALITIES (Sec. III; higher = more stabilizing)
# ------------------------------------------------------------------
def compute_ssc(mdl):
    A, lam, B = mdl["A"], mdl["lam"], mdl["B"]
    sol = solve_lq(A, lam, B, mdl["Q"], mdl["R"], mdl["W"])
    P, K, Acl, Sigma = sol["P"], sol["K"], sol["Acl"], sol["Sigma"]
    L = np.diag(1.0 - lam)
    support = A > 0

    dJdA = 2 * L @ P @ Acl @ Sigma                                    # Thm. 1, eq. (14)
    row_mean = (dJdA * support).sum(1, keepdims=True) / support.sum(1, keepdims=True)
    S = -(dJdA - row_mean) * support                                   # Def. 1
    influence = S.sum(0) - np.diag(S)                                  # Def. 2

    Gamma = 2 * L @ P @ Acl @ Sigma @ K.T                              # Thm. 2
    actuator = np.linalg.norm(Gamma, axis=1)                           # Def. 4
    stubbornness = 2 * np.diag(P @ Acl @ Sigma @ (A - B @ K).T)        # Thm. 3
    noise = np.diag(P)                                                 # Thm. 4: C_Noise = P_ii
    # Placement Sensitivity (Prop. 1): second-order gain of a new actuator at node i, weight r = R_WEIGHT
    placement = (1 - lam) ** 2 * np.diag(P @ (Sigma - mdl["W"]) @ P) / R_WEIGHT

    ssc = {"Influence": influence, "Stubbornness": stubbornness, "Actuator": actuator, "Noise": noise}
    return ssc, dict(sol=sol, S=S, Gamma=Gamma, placement=placement)


# ------------------------------------------------------------------
# BASELINES
# ------------------------------------------------------------------
def compute_baselines(G, mdl):
    n = G.number_of_nodes()
    H = G.copy()
    for u, v, e in H.edges(data=True):
        e["distance"] = 1.0 / e[WEIGHT]                                # strong ties = short paths
    deg = np.array([H.degree(i, weight=WEIGHT) for i in range(n)], float)
    pr = nx.pagerank(H, weight=WEIGHT)
    bet = nx.betweenness_centrality(H, weight="distance")
    # FJ social power: column sums of V = (I - (I - Lambda) A)^{-1} Lambda
    V = np.linalg.solve(np.eye(n) - np.diag(1 - mdl["lam"]) @ mdl["A"], np.diag(mdl["lam"]))
    return {"Degree": deg,
            "PageRank": np.array([pr[i] for i in range(n)]),
            "Betweenness": np.array([bet[i] for i in range(n)]),
            "SocialPower": V.sum(0)}


# ------------------------------------------------------------------
# FINITE INTERVENTIONS (re-solve the DARE; return the true reduction -dJ*)
# ------------------------------------------------------------------
def make_interventions(mdl, extra, noise_rng=None):
    A, lam, B, Q, R, W = (mdl[k] for k in ("A", "lam", "B", "Q", "R", "W"))
    n, m = B.shape
    J0 = extra["sol"]["J"]
    c_star = extra["Gamma"] / np.linalg.norm(extra["Gamma"], axis=1, keepdims=True)

    if noise_rng is None:
        eta, stub_step = np.full((n, n), TRUST_SHIFT), np.full(n, STUB_STEP)
        act_step, noise_step, c_dir = np.full(n, ACT_STEP), np.full(n, NOISE_STEP), c_star
    else:
        u = lambda *s: 1 + noise_rng.uniform(-NOISE_REL, NOISE_REL, s)
        eta, stub_step = TRUST_SHIFT * u(n, n), STUB_STEP * u(n)
        act_step, noise_step = ACT_STEP * u(n), NOISE_STEP * u(n)
        c_dir = c_star + DIR_NOISE * noise_rng.standard_normal((n, m))
        c_dir /= np.linalg.norm(c_dir, axis=1, keepdims=True)

    J = lambda A_=A, lam_=lam, B_=B, W_=W: solve_lq(A_, lam_, B_, Q, R, W_)["J"]

    def trust(T):
        T, Ap = list(T), A.copy()
        for j in T:
            for i in range(n):
                if A[i, j] > 0 and i not in T:          # listeners inside the target set unchanged
                    Ap[i] = (1 - eta[i, j]) * Ap[i]
                    Ap[i, j] += eta[i, j]
        return J0 - J(A_=Ap)

    def stub(T):
        T, lp = list(T), lam.copy()
        lp[T] = np.minimum(lp[T] + stub_step[T], 0.999)
        return J0 - J(lam_=lp)

    def act(T):
        Bp = B.copy()
        for i in T:
            Bp[i] += act_step[i] * c_dir[i]
        return J0 - J(B_=Bp)

    def noise(T):
        Wp = W.copy()
        for i in T:
            Wp[i, i] -= noise_step[i]
        return J0 - J(W_=Wp)

    return {"Influence": trust, "Stubbornness": stub, "Actuator": act, "Noise": noise}


def target_score(name, ssc):
    """Top entries = best targets (for noise, the largest C_Noise = P_ii)."""
    return ssc[name]


def top(v, k):
    return list(np.argsort(-v, kind="stable")[:k])


# ------------------------------------------------------------------
# EXPERIMENTS
# ------------------------------------------------------------------
def single_agent(interventions, n):
    return {name: np.array([f([i]) for i in range(n)]) for name, f in interventions.items()}


def validity_table(ssc, base, true):
    rows = []
    for name, dJ in true.items():
        sc = target_score(name, ssc)
        row = dict(metric=name, SSC=kendalltau(sc, dJ)[0])
        row.update({b: kendalltau(v, dJ)[0] for b, v in base.items()})
        best3 = set(top(dJ, 3))
        row["top3_hits_SSC"] = len(best3 & set(top(sc, 3)))
        rows.append(row)
    return rows


def budget_table(ssc, base, interventions, n, rng):
    rows = []
    for name, f in interventions.items():
        chosen, greedy = [], []
        for _ in range(K_MAX):                                         # O(nk) DARE solves
            j = max((j for j in range(n) if j not in chosen), key=lambda j: f(chosen + [j]))
            chosen.append(j)
            greedy.append(f(chosen))
        for k in range(1, K_MAX + 1):
            g = greedy[k - 1]
            row = dict(metric=name, k=k, greedy=g, greedy_set=" ".join(map(str, chosen[:k])))
            row["SSC"] = 100 * f(top(target_score(name, ssc), k)) / g
            row["SSC_set"] = " ".join(map(str, top(target_score(name, ssc), k)))
            for b, v in base.items():
                row[b] = 100 * f(top(v, k)) / g
                row[f"{b}_set"] = " ".join(map(str, top(v, k)))
            row["BestBaseline"] = max(row[b] for b in base)
            row["Random"] = 100 * np.mean([f(list(rng.choice(n, k, replace=False)))
                                           for _ in range(N_RANDOM)]) / g
            rows.append(row)
    return rows


def robustness(mdl, extra, ssc, base, n, true_nominal):
    """Implementation noise: SSC tau vs the ceiling attainable without knowledge of the realized noise."""
    rec = {name: dict(ssc=[], ceiling=[], budget=[]) for name in true_nominal}
    for t in range(1, N_TRIALS + 1):
        iv = make_interventions(mdl, extra, noise_rng=np.random.default_rng(1000 + t))
        true = single_agent(iv, n)
        bud = budget_table(ssc, base, iv, n, np.random.default_rng(2000 + t))
        for name in true:
            rec[name]["ssc"].append(kendalltau(target_score(name, ssc), true[name])[0])
            rec[name]["ceiling"].append(kendalltau(true_nominal[name], true[name])[0])
            rec[name]["budget"] += [r["SSC"] for r in bud if r["metric"] == name]
    return [dict(metric=name, ssc_tau_mean=np.mean(r["ssc"]), ssc_tau_std=np.std(r["ssc"]),
                 ceiling_mean=np.mean(r["ceiling"]), budget_mean=np.mean(r["budget"]),
                 budget_min=np.min(r["budget"])) for name, r in rec.items()]


# ------------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------------
def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, (float, np.floating)) else v) for k, v in r.items()})


def plot_topology(G):
    pos = nx.spring_layout(G, seed=42, k=0.3, weight=None)
    gap = pos[33] - pos[32]
    pos[33] = pos[33] + 0.13 * gap / (np.linalg.norm(gap) + 1e-12)
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    groups = {
        "Instructor's faction": ([k for k, d in G.nodes(data=True) if d["club"] == "Mr. Hi" and k != 0],
                                 dict(node_color="#6baed6", node_size=110, node_shape="o")),
        "Administrator's faction": ([k for k, d in G.nodes(data=True) if d["club"] == "Officer" and k != 33],
                                    dict(node_color="#fd8d3c", node_size=110, node_shape="o")),
        "Instructor (Node 0)": ([0], dict(node_color="#2171b5", node_size=200, node_shape="s", linewidths=1.2)),
        "Administrator (Node 33)": ([33], dict(node_color="#d94801", node_size=220, node_shape="^", linewidths=1.2)),
    }
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color="gray",
                           width=[0.3 + 0.3 * G[u][v][WEIGHT] for u, v in G.edges()], ax=ax)
    for label, (nodes, style) in groups.items():
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, label=label, edgecolors="black", ax=ax, **style)
    nx.draw_networkx_labels(G, pos, font_size=5, ax=ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2, fontsize=6, frameon=False,
              markerscale=0.6, handletextpad=0.3, columnspacing=1.0)
    ax.axis("off")
    fig.savefig("topology.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    G = nx.karate_club_graph()
    n = G.number_of_nodes()
    mdl = build_model(G)
    ssc, extra = compute_ssc(mdl)
    base = compute_baselines(G, mdl)
    sol = extra["sol"]

    # --- model diagnostics ---
    J_ol, floor = open_loop_cost(mdl), float(np.trace(mdl["Q"] @ mdl["W"]))
    print(f"lambda followers in [{mdl['lam'][2:33].min():.3f}, {mdl['lam'][2:33].max():.3f}]")
    print(f"rho(A~) = {max(abs(np.linalg.eigvals(sol['At']))):.3f}, rho(Acl) = {max(abs(np.linalg.eigvals(sol['Acl']))):.3f}")
    print(f"J_open = {J_ol:.1f}, J* = {sol['J']:.1f}, floor Tr(QW) = {floor:.0f}, "
          f"excess reduction = {100 * (J_ol - sol['J']) / (J_ol - floor):.1f}%")
    print(f"max |row sum of S| = {np.abs(extra['S'].sum(1)).max():.1e}")
    for name, v in ssc.items():
        print(f"{name:13s} #pos={np.sum(v > 0):2d} #neg={np.sum(v < 0):2d} top5={top(v, 5)}")
    print(f"C_Stub < 0 at {np.flatnonzero(ssc['Stubbornness'] < 0).tolist()}")

    # --- finite-difference check of the closed forms ---
    h, errs = 1e-6, []
    Jf = lambda lam=mdl["lam"], B=mdl["B"]: solve_lq(mdl["A"], lam, B, mdl["Q"], mdl["R"], mdl["W"])["J"]
    for i in [5, 32]:
        e = np.zeros(n); e[i] = h
        errs.append(abs(-(Jf(lam=mdl["lam"] + e) - Jf(lam=mdl["lam"] - e)) / (2 * h) - ssc["Stubbornness"][i]))
        Bp, Bm = mdl["B"].copy(), mdl["B"].copy(); Bp[i, 0] += h; Bm[i, 0] -= h
        errs.append(abs(-(Jf(B=Bp) - Jf(B=Bm)) / (2 * h) - extra["Gamma"][i, 0]))
    print(f"max abs finite-difference error = {max(errs):.1e}")

    # --- Table I: predictive validity ---
    iv = make_interventions(mdl, extra)
    true = single_agent(iv, n)
    t1 = validity_table(ssc, base, true)
    write_csv("table1_validity.csv", t1)
    print("\nTABLE I (Kendall tau-b)")
    for r in t1:
        print(f"  {r['metric']:13s} SSC={r['SSC']:+.2f} " + " ".join(f"{b}={r[b]:+.2f}" for b in base)
              + f"  top3 hits={r['top3_hits_SSC']}")

    # --- Table II: budget ---
    t2 = budget_table(ssc, base, iv, n, np.random.default_rng(RANDOM_SEED))
    write_csv("table2_budget.csv", t2)
    print("\nTABLE II (% of greedy)")
    for r in t2:
        print(f"  {r['metric']:13s} k={r['k']} SSC={r['SSC']:6.1f} best={r['BestBaseline']:6.1f} rand={r['Random']:6.1f} | "
              + " ".join(f"{b}={r[b]:6.1f}" for b in base)
              + f" | greedy={r['greedy_set']} SSC={r['SSC_set']} Deg={r['Degree_set']}")

    # --- conflicting roles ---
    j = int(np.argmin(ssc["Influence"]))
    stub_rank = int(np.where(np.argsort(-ssc["Stubbornness"], kind="stable") == j)[0][0]) + 1
    print(f"\nConflicting roles: node {j}, C_Inf = {ssc['Influence'][j]:+.2f} (most negative), "
          f"C_Stub rank = {stub_rank}; trust shift toward it changes J* by {-true['Influence'][j]:+.3f}, "
          f"entrenching it changes J* by {-true['Stubbornness'][j]:+.3f}")

    # --- robustness to implementation noise ---
    rob = robustness(mdl, extra, ssc, base, n, true)
    write_csv("robustness_noise.csv", rob)
    print("\nROBUSTNESS (implementation noise)")
    for r in rob:
        print(f"  {r['metric']:13s} tau={r['ssc_tau_mean']:.2f}+/-{r['ssc_tau_std']:.2f} ceiling={r['ceiling_mean']:.2f} "
              f"budget mean={r['budget_mean']:.0f}% min={r['budget_min']:.0f}%")

    # --- actuator placement (Prop. 1): add a new channel beta * e_i with weight R_WEIGHT ---
    print("\nPLACEMENT")
    pl = extra["placement"]
    for beta in PLACE_BETAS:
        trueP = np.empty(n)
        for i in range(n):
            Bp = np.hstack([mdl["B"], np.zeros((n, 1))]); Bp[i, -1] = beta
            Rp = np.diag(np.r_[np.diag(mdl["R"]), R_WEIGHT])
            trueP[i] = sol["J"] - solve_lq(mdl["A"], mdl["lam"], Bp, mdl["Q"], Rp, mdl["W"])["J"]
        print(f"  beta={beta}: tau C_Place={kendalltau(pl, trueP)[0]:.2f}  tau C_Act={kendalltau(ssc['Actuator'], trueP)[0]:.2f}  "
              + " ".join(f"{b}={kendalltau(v, trueP)[0]:+.2f}" for b, v in base.items())
              + f"  top3 true={top(trueP, 3)} C_Place={top(pl, 3)}")
    i = 5
    for beta in [1e-2, 1e-3]:
        Bp = np.hstack([mdl["B"], np.zeros((n, 1))]); Bp[i, -1] = beta
        Rp = np.diag(np.r_[np.diag(mdl["R"]), R_WEIGHT])
        fd = (sol["J"] - solve_lq(mdl["A"], mdl["lam"], Bp, mdl["Q"], Rp, mdl["W"])["J"]) / beta ** 2
        print(f"  second-order check node {i}, beta={beta}: (J0-J)/beta^2 = {fd:.3f} vs C_Place = {pl[i]:.3f}")

    write_csv("centralities.csv", [dict(node=i, lam=mdl["lam"][i], **{k: v[i] for k, v in ssc.items()}, Placement=extra["placement"][i],
                                        **{k: v[i] for k, v in base.items()}) for i in range(n)])
    plot_topology(G)


if __name__ == "__main__":
    main()