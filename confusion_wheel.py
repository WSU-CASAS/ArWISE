"""
   confusion_wheel.py <confusion_matrix.csv>

   The confusion_wheel.py script generates a confusion wheel plot to visualize
   a confusion matrix set of classification results, stored in file the given CSV file.
   The wheel is stored in file `<confusion_matrix>_wheel.png`.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyArrowPatch
import sys
import os

# Generate curved Bezier edges to draw between nodes
class CurvedEdge(FancyArrowPatch):
    def __init__(self, posA, posB, base_rad=0.3, linewidth=2, node_radius=0.15, **kwargs):
        posA, posB = np.array(posA), np.array(posB)

        # Compute unit direction vector
        vec = posB - posA
        vec /= np.linalg.norm(vec)  # Normalize

        # Adjust positions to start and end at the node edge
        start = posA + vec * node_radius
        end = posB - vec * node_radius

        # Dynamic curvature based on distance between nodes
        distance = np.linalg.norm(posB - posA)
        rad = base_rad * np.exp(-distance)
        connection_style = f"arc3,rad={rad}"

        super().__init__(start, end, connectionstyle=connection_style, linewidth=linewidth, **kwargs)


# Create the confusion wheel
def generate_confusion_wheel(cmfile):
    df = pd.read_csv(cmfile, index_col=0)
    class_labels = df.index.tolist()
    num_classes = len(class_labels)
    conf_matrix = df.values

    # Compute TP, FP, FN, TN for each class
    TP = np.diag(conf_matrix)
    FP = np.sum(conf_matrix, axis=0) - TP
    FN = np.sum(conf_matrix, axis=1) - TP
    TN = np.sum(conf_matrix) - (TP + FP + FN)

    G = nx.DiGraph()
    for i in range(num_classes):
        G.add_node(i)

    max_weight = np.max(conf_matrix)
    for i in range(num_classes):
        for j in range(i + 1, num_classes):  # Only process each pair once
            if conf_matrix[i, j] > 0 or conf_matrix[j, i] > 0:
                combined_weight = (conf_matrix[i, j] + conf_matrix[j, i]) / max_weight * 15
                G.add_edge(i, j, weight=combined_weight)

    scale_factor = 0.7
    pos = nx.circular_layout(G)  # Position nodes in a circle
    pos = {k: v * scale_factor for k, v in pos.items()}

    fig, ax = plt.subplots(figsize=(12, 12))
    fig.patch.set_facecolor("white")
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    for u, v, data in G.edges(data=True): # Draw curved edges
        edge_alpha = max(0.3, min(1.0, data['weight'] / 15))  # Scale transparency
        edge = CurvedEdge(pos[u], pos[v], linewidth=data['weight'] * 10.0, alpha=edge_alpha, color="black", arrowstyle="-")
        ax.add_patch(edge)

    for i, (x, y) in pos.items(): # Labels outside nodes
        ax.text(x * 1.28, y * 1.28, class_labels[i], fontsize=10, fontweight="bold", ha="center", va="center")

    colors = ['green', 'yellow', 'red', 'paleturquoise'] # TP = Green, FP = Yellow, FN = Red, TN = Gray

    for i in range(num_classes): # Node pie charts
        x, y = pos[i]
        pie_values = [TP[i], FP[i], FN[i], TN[i]]
        ax.pie(pie_values, colors=colors, wedgeprops={'edgecolor': 'dimgrey', 'linewidth': 0.25}, center=(x, y), radius=0.15)

    #------------------
    all_x = [pos[i][0] for i in range(num_classes)]
    all_y = [pos[i][1] for i in range(num_classes)]
    node_radius = 0.15  # Same as pie chart radius

    x_margin = node_radius * 2  # Extra space for pie charts
    y_margin = node_radius * 2

    # Compute new axis limits
    x_min, x_max = min(all_x) - x_margin, max(all_x) + x_margin
    y_min, y_max = min(all_y) - y_margin, max(all_y) + y_margin

    # Apply limits to ensure all nodes fit
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    #------------------

    legend_labels = ["TP", "FP", "FN", "TN"]
    legend_colors = ["green", "yellow", "red", "paleturquoise"]

    patches = [plt.Rectangle((0, 0), 1, 1, fc=color) for color in legend_colors]
    ax.legend(patches, legend_labels, loc="upper right", bbox_to_anchor=(0.9, 0.9), frameon=False, fontsize=12)

    basefile = os.path.splitext(cmfile)[0]
    plt.savefig(basefile + '_wheel.png', dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

if __name__ == "__main__":
    cmfile = sys.argv[1]
    generate_confusion_wheel(cmfile)
