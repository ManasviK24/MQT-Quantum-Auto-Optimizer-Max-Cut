import time
import networkx as nx
from mqt.qao import Constraints, ObjectiveFunction, Problem, Solver, Variables
from maxcut_utils import cut_value, brute_force_maxcut, plot_partition

# 1) Tiny weighted graph (cycle of 4 + diagonals)
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

# 2) One binary variable per node: 0 = group A, 1 = group B
var = Variables()
x = {i: var.add_binary_variable(f"x_{i}") for i in G.nodes()}
nodes_order = list(x.keys())

# 3) Max-Cut objective (maximize)
obj = ObjectiveFunction()
expr = 0
for i, j, data in G.edges(data=True):
    w = data.get("weight", 1.0)
    expr += w * (x[i] + x[j] - 2 * (x[i] * x[j]))
obj.add_objective_function(expr, minimization=False)

cst = Constraints()

# 4) Build the problem
prb = Problem()
prb.create_problem(var, cst, obj)

# Map back from solver's 'b0,b1,...' names if needed
def bit_for(assign, node):
    kx = f"x_{node}"
    if kx in assign:
        return int(round(assign[kx]))
    idx = nodes_order.index(node)
    kb = f"b{idx}"
    return int(round(assign[kb]))

def assignment_to_partition(assign):
    return {node: bit_for(assign, node) for node in G.nodes()}

# 5a) Simulated Annealing (baseline)
t0 = time.perf_counter()
sa_sol = Solver().solve_simulated_annealing(prb)
sa_t = time.perf_counter() - t0
sa_part = assignment_to_partition(sa_sol.best_solution_original_var)
print("[SA] cut value:", cut_value(G, sa_part), f"(time {sa_t:.4f}s)")

# 5b) QAOA small
t0 = time.perf_counter()
qaoa_sol = Solver().solve_qaoa_qubo(prb, reps=1, num_runs=1)
qaoa_t = time.perf_counter() - t0
qaoa_part = assignment_to_partition(qaoa_sol.best_solution_original_var)
print("[QAOA] cut value:", cut_value(G, qaoa_part), f"(time {qaoa_t:.4f}s)")

# 6) Exact check (brute force) for credibility
bf_part, bf_val = brute_force_maxcut(G)
print("[Exact] best possible cut value:", bf_val)
print("SA is optimal?   ", abs(cut_value(G, sa_part)   - bf_val) < 1e-9)
print("QAOA is optimal? ", abs(cut_value(G, qaoa_part) - bf_val) < 1e-9)

# 7) Save a figure to figures/
plot_partition(
    G, qaoa_part,
    out_path="figures/partition_qaoa.png",
    title=f"QAOA partition (cut={cut_value(G, qaoa_part):.3f})"
)
print("Saved figure: figures/partition_qaoa.png")
