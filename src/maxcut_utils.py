import itertools
import networkx as nx
import matplotlib.pyplot as plt

def cut_value(G, part):
    val = 0.0
    for u, v, data in G.edges(data=True):
        if part[u] != part[v]:
            val += data.get("weight", 1.0)
    return val

def brute_force_maxcut(G):
    nodes = list(G.nodes())
    best_val = float("-inf")
    best_part = None
    for bits in itertools.product([0, 1], repeat=len(nodes)):
        part = {nodes[i]: bits[i] for i in range(len(nodes))}
        val = cut_value(G, part)
        if val > best_val:
            best_val = val
            best_part = part
    return best_part, best_val

def plot_partition(G, part, out_path=None, title="Max-Cut Partition"):
    pos = nx.spring_layout(G, seed=42)
    A = [n for n, b in part.items() if b == 0]
    B = [n for n, b in part.items() if b == 1]

    plt.figure()
    nx.draw_networkx_nodes(G, pos, nodelist=A)
    nx.draw_networkx_nodes(G, pos, nodelist=B, node_shape="s")
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos)
    plt.title(title)

    if out_path:
        plt.savefig(out_path, bbox_inches="tight")
