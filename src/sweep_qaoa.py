import time, csv
import networkx as nx
import matplotlib.pyplot as plt
from mqt.qao import Constraints, ObjectiveFunction, Problem, Solver, Variables
from maxcut_utils import cut_value, brute_force_maxcut

# ----- Build the same tiny graph -----
G = nx.Graph()
edges = [
    (1, 2, 1.0),
    (2, 3, 1.2),
    (3, 4, 0.8),
    (4, 1, 1.1),
    (1, 3, 0.5),
    (2, 4, 0.7),
]
G.add_weighted_edges_from(edges)

# ----- Build Max-Cut problem in MQT-QAO -----
var = Variables()
x = {i: var.add_binary_variable(f"x_{i}") for i in G.nodes()}
nodes_order = list(x.keys())

obj = ObjectiveFunction()
expr = 0
for i, j, data in G.edges(data=True):
    w = data.get("weight", 1.0)
    expr += w * (x[i] + x[j] - 2 * (x[i] * x[j]))
obj.add_objective_function(expr, minimization=False)

cst = Constraints()
prb = Problem()
prb.create_problem(var, cst, obj)

def bit_for(assign, node):
    kx = f"x_{node}"
    if kx in assign:
        return int(round(assign[kx]))
    idx = nodes_order.index(node)
    kb = f"b{idx}"
    return int(round(assign[kb]))

def assignment_to_partition(assign):
    return {node: bit_for(assign, node) for node in G.nodes()}

# ----- Baseline: Simulated Annealing -----
t0 = time.perf_counter()
sa_sol = Solver().solve_simulated_annealing(prb)
sa_t = time.perf_counter() - t0
sa_cut = cut_value(G, assignment_to_partition(sa_sol.best_solution_original_var))

# ----- Sweep QAOA depth -----
reps_list = [1, 2, 3]      # small & fast to start
num_runs = 5               # internal restarts per setting
rows = []

for r in reps_list:
    t0 = time.perf_counter()
    q_sol = Solver().solve_qaoa_qubo(prb, reps=r, num_runs=num_runs)
    q_t = time.perf_counter() - t0
    q_cut = cut_value(G, assignment_to_partition(q_sol.best_solution_original_var))
    rows.append({"reps": r, "num_runs": num_runs, "qaoa_cut": q_cut, "qaoa_time_s": q_t})

# ----- Exact best for credibility -----
_, exact_cut = brute_force_maxcut(G)

# ----- Save CSV -----
csv_path = "results/qaoa_sweep.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["reps", "num_runs", "qaoa_cut", "qaoa_time_s", "sa_cut", "exact_cut"])
    writer.writeheader()
    for row in rows:
        row["sa_cut"] = sa_cut
        row["exact_cut"] = exact_cut
        writer.writerow(row)

# ----- Print a small table -----
print("\nSweep results (SA baseline and exact shown for reference):")
print(f"SA cut = {sa_cut:.3f}, Exact = {exact_cut:.3f}")
for r in rows:
    print(f"reps={r['reps']}  qaoa_cut={r['qaoa_cut']:.3f}  time={r['qaoa_time_s']:.3f}s")

# ----- Plot (QAOA cut vs reps) -----
plt.figure()
plt.plot([r["reps"] for r in rows], [r["qaoa_cut"] for r in rows], marker="o", label="QAOA")
plt.axhline(sa_cut, linestyle="--", label="SA baseline")
plt.axhline(exact_cut, linestyle=":", label="Exact")
plt.xlabel("QAOA depth (reps)")
plt.ylabel("Cut value")
plt.title("QAOA cut vs reps (num_runs=5)")
plt.legend()
fig_path = "figures/sweep_qaoa.png"
plt.savefig(fig_path, bbox_inches="tight")
print(f"\nSaved: {csv_path} and {fig_path}")
