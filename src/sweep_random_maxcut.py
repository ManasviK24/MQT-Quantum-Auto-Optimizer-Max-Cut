import argparse, time, csv, random
import networkx as nx
import matplotlib.pyplot as plt
from mqt.qao import Constraints, ObjectiveFunction, Problem, Solver, Variables

# --- helpers (reuse minimal versions here) ---
def cut_value(G, part):
    val = 0.0
    for u, v, data in G.edges(data=True):
        if part[u] != part[v]:
            val += data.get("weight", 1.0)
    return val

def brute_force_maxcut(G):
    import itertools
    nodes = list(G.nodes())
    best_val = float("-inf"); best_part = None
    for bits in itertools.product([0,1], repeat=len(nodes)):
        part = {nodes[i]: bits[i] for i in range(len(nodes))}
        val = cut_value(G, part)
        if val > best_val:
            best_val = val; best_part = part
    return best_part, best_val

def assignment_to_partition(assign, nodes_order):
    # accept either 'x_i' or internal 'b{idx}' names
    part = {}
    for node in nodes_order:
        kx = f"x_{node}"
        if kx in assign:
            part[node] = int(round(assign[kx]))
        else:
            idx = nodes_order.index(node)
            part[node] = int(round(assign[f"b{idx}"]))
    return part

# --- build Max-Cut problem in MQT-QAO from a NetworkX graph ---
def build_problem(G):
    var = Variables()
    x = {i: var.add_binary_variable(f"x_{i}") for i in G.nodes()}
    nodes_order = list(x.keys())

    obj = ObjectiveFunction()
    expr = 0
    for i, j, data in G.edges(data=True):
        w = data.get("weight", 1.0)
        expr += w * (x[i] + x[j] - 2 * (x[i] * x[j]))
    obj.add_objective_function(expr, minimization=False)

    prb = Problem()
    prb.create_problem(var, Constraints(), obj)
    return prb, nodes_order

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="number of nodes")
    ap.add_argument("--p_edge", type=float, default=0.6, help="edge probability")
    ap.add_argument("--wmin", type=float, default=0.5, help="min weight")
    ap.add_argument("--wmax", type=float, default=1.5, help="max weight")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reps", type=int, nargs="+", default=[1,2,3])
    ap.add_argument("--num_runs", type=int, default=5)
    args = ap.parse_args()

    random.seed(args.seed)
    G = nx.gnp_random_graph(args.n, args.p_edge, seed=args.seed)
    # add random weights to each edge
    for u, v in G.edges():
        G[u][v]["weight"] = random.uniform(args.wmin, args.wmax)

    prb, nodes_order = build_problem(G)

    # baseline SA
    t0 = time.perf_counter()
    sa_sol = Solver().solve_simulated_annealing(prb)
    sa_t = time.perf_counter() - t0
    sa_part = assignment_to_partition(sa_sol.best_solution_original_var, nodes_order)
    sa_cut = cut_value(G, sa_part)

    # QAOA sweep
    rows = []
    for r in args.reps:
        t0 = time.perf_counter()
        q_sol = Solver().solve_qaoa_qubo(prb, reps=r, num_runs=args.num_runs)
        q_t = time.perf_counter() - t0
        q_part = assignment_to_partition(q_sol.best_solution_original_var, nodes_order)
        q_cut = cut_value(G, q_part)
        rows.append({"reps": r, "num_runs": args.num_runs, "qaoa_cut": q_cut, "qaoa_time_s": q_t})

    # exact (only if reasonably small)
    exact_cut = None
    if args.n <= 10:  # ~2^10 = 1024 → still fine
        _, exact_cut = brute_force_maxcut(G)

    # save CSV
    csv_path = f"results/qaoa_sweep_n{args.n}_p{args.p_edge}_seed{args.seed}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["reps","num_runs","qaoa_cut","qaoa_time_s","sa_cut","exact_cut"])
        writer.writeheader()
        for row in rows:
            row["sa_cut"] = sa_cut
            row["exact_cut"] = exact_cut
            writer.writerow(row)

    # text summary
    print("\nSweep results on random graph:")
    print(f"n={args.n}, p_edge={args.p_edge}, seed={args.seed}")
    print(f"SA cut = {sa_cut:.3f}, Exact = {exact_cut if exact_cut is not None else 'N/A'}")
    for r in rows:
        print(f"reps={r['reps']}  qaoa_cut={r['qaoa_cut']:.3f}  time={r['qaoa_time_s']:.3f}s")
    print(f"Saved CSV: {csv_path}")

    # plot
    plt.figure()
    xs = [r["reps"] for r in rows]
    ys = [r["qaoa_cut"] for r in rows]
    plt.plot(xs, ys, marker="o", label="QAOA")
    plt.axhline(sa_cut, linestyle="--", label="SA baseline")
    if exact_cut is not None:
        plt.axhline(exact_cut, linestyle=":", label="Exact")
    plt.xlabel("QAOA depth (reps)")
    plt.ylabel("Cut value")
    plt.title(f"Random graph n={args.n} (num_runs={args.num_runs})")
    plt.legend()
    fig_path = f"figures/sweep_n{args.n}_p{args.p_edge}_seed{args.seed}.png"
    plt.savefig(fig_path, bbox_inches="tight")
    print(f"Saved figure: {fig_path}")

if __name__ == "__main__":
    main()
