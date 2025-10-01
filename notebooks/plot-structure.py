# Create a bottom-to-top hierarchical diagram with ellipsis truncation per layer
import os, textwrap
import matplotlib.pyplot as plt
import networkx as nx

# -------- Data --------
categories = {
    "Single driver": {
        "A": "\nRight roadside\n departure",
        "B": "\nLeft roadside\n departure",
        "C": "\nForward impact",
    },
    "Same trafficway,\n same direction": {
        "D": "\nRear-end",
        "E": "\nForward impact",
        "... ": "\nMore crash\n configurations"
        
    },
    "Same trafficway,\n opposite direction": {
        "G": "Head-on",
        "H": "Forward impact",
        "I": "Angle, sideswipe"
    },
    "Changing trafficway,\n vehicle turning": {
        "J": "Turn across path",
        "K": "Turn into path"
    },
    "Intersecting paths \n(vehicle damage)": {
        "L": "Straight paths"
    },
    "Miscellaneous": {
        "M": "Backing, etc."
    }
}

# Crash types (only for A demo; extend similarly for others if needed)
crash_types_map = {
    "A": {
        1: "Drive off road",
        2: "Control/traction loss",
        3: "Avoid Collision with Vehicle,\n Pedestrian, Animal",
        4: "Specifics other",
        5: "Specifics unknown"
        
    }
}
crash_types_mapB = {
    "B": {
        6: "Drive off road",
        "... ": "More crash types",
        
    }
}

# -------- Build graph with subsets --------
G = nx.DiGraph()

# Parameters
MAX_SHOW_PER_LAYER = 6  # show at most 5 nodes per layer, rest collapsed into ellipsis

# Layer 0: Categories (bottom layer)
all_cats = list(categories.keys())
show_cats = all_cats[:MAX_SHOW_PER_LAYER]
more_cats = len(all_cats) - len(show_cats)
for cat in show_cats:
    G.add_node(cat, subset=0, kind="category")
if more_cats > 0:
    ell_cat = f"… ({more_cats} more)"
    G.add_node(ell_cat, subset=0, kind="category")
    show_cats.append(ell_cat)

# Layer 1: Configurations (middle layer)
# Collect configurations; keep mapping to parent
all_confs = []
for cat in all_cats:
    for letter, desc in categories[cat].items():
        all_confs.append((cat, f"{letter}: {desc}", letter))

show_confs = all_confs[:MAX_SHOW_PER_LAYER]
more_confs = len(all_confs) - len(show_confs)

for parent, label, letter in show_confs:
    G.add_node(label, subset=1, kind="config")
    # connect to parent if parent is visible else to ellipsis
    if parent in show_cats:
        G.add_edge(parent, label) 
        

# if more_confs > 0:
#     ell_conf = f"… ({more_confs} more configs)"
#     G.add_node(ell_conf, subset=1, kind="config")
#     # connect ellipsis to a visible category node (or categories ellipsis if present)
#     if show_cats:
#         G.add_edge(show_cats[0], ell_conf)

# Layer 2: Crash types (top layer) — only for A (if it is visible among shown configs)
shown_letters = {lab.split(":")[0] for _, lab, _l in show_confs}

if "A" in crash_types_map and ("A" in shown_letters):
    parent_label = [lab for _, lab, letter in show_confs if letter=="A"][0]
    types = list(crash_types_map["A"].items())
    show_types = types[:MAX_SHOW_PER_LAYER]
    more_types = len(types) - len(show_types)
    for tid, tdesc in show_types:
        wrapped = f"{tid}\n" + "\n".join(textwrap.wrap(tdesc, width=20))
        G.add_node(wrapped, subset=2, kind="type")
        G.add_edge(parent_label, wrapped)


if "B" in crash_types_mapB and ("B" in shown_letters):
    parent_label = [lab for _, lab, letter in show_confs if letter=="B"][0]
    types = list(crash_types_mapB["B"].items())
    show_types = types[:MAX_SHOW_PER_LAYER]
    more_types = len(types) - len(show_types)
    for tid, tdesc in show_types:
        print(tid)
        wrapped = f"{tid}\n" + "\n".join(textwrap.wrap(tdesc, width=20))
        G.add_node(wrapped, subset=2, kind="type")
        G.add_edge(parent_label, wrapped)

# -------- Manual bottom-to-top layered layout --------
# We'll set y = subset (0 bottom, 1 middle, 2 top), and space x evenly per layer.
def layer_positions(nodes, y, x_start=0.0, x_step=0.8):
    pos = {}
    for i, n in enumerate(nodes):
        pos[n] = (x_start + i * x_step, y)
    return pos

# order nodes per subset
cats_layer  = [n for n,d in G.nodes(data=True) if d["subset"]==0]
confs_layer = [n for n,d in G.nodes(data=True) if d["subset"]==1]
types_layer = [n for n,d in G.nodes(data=True) if d["subset"]==2]

pos = {}
pos.update(layer_positions(cats_layer, 0))
pos.update(layer_positions(confs_layer, 1))
pos.update(layer_positions(types_layer, 2))

# -------- Draw --------
plt.figure(figsize=(16, 16))
nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="-|>", arrowsize=12, width=2, edge_color="grey")

# Node styles
def draw_nodes(layer_nodes, color, size, shape="o"):
    if layer_nodes:
        nx.draw_networkx_nodes(
            G, pos, nodelist=layer_nodes,
            node_size=size, node_color=color,
            node_shape=shape
        )


draw_nodes(cats_layer,  "#c6e2ff", 8000, shape="s")   # 方形
draw_nodes(confs_layer, "#f0f8ff", 8000, shape="s")   # 圆形
draw_nodes(types_layer, "#fffacd", 8000, shape="s")

nx.draw_networkx_labels(G, pos, font_size=13, font_weight="bold")

plt.title("Crash Taxonomy (Bottom-to-Top)\nBottom: Crash Categories  |  Middle: Crash Configurations  |  Top: Crash Types (A only)", fontsize=18, fontweight="bold")
plt.axis("off")

out_dir = "reports/crash-type-test/metrics"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "crash_taxonomy_bottom_to_top_with_ellipsis.png")
plt.tight_layout()
plt.savefig(out_path, dpi=300, bbox_inches="tight")
