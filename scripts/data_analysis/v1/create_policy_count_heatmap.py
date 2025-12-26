"""
Generate a heatmap of policy selection count by civilization and chosen ancient policy.
X-axis: civilization
Y-axis: chosen_ancient_policy
Cell value: number of times that civilization chose that policy tree.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def main() -> None:
    csv_path = Path("data/processed/game_data.csv")
    df = pd.read_csv(csv_path)

    # Count occurrences per (civilization, policy)
    civs = sorted(df["civilization"].unique())
    policies = sorted(df["chosen_ancient_policy"].unique())

    count_matrix = (
        df.groupby(["chosen_ancient_policy", "civilization"])
        .size()
        .reindex(
            pd.MultiIndex.from_product([policies, civs], names=["chosen_ancient_policy", "civilization"])
        )
        .fillna(0)
        .unstack(fill_value=0)
    )

    plt.figure(figsize=(max(10, len(civs) * 0.4), max(4, len(policies) * 0.4)))
    
    # Use a colormap that works well for counts (higher values = brighter)
    im = plt.imshow(count_matrix, cmap="YlOrRd", aspect="auto", vmin=0)

    # Annotate each cell with the count
    for i, policy in enumerate(policies):
        for j, civ in enumerate(civs):
            val = count_matrix.loc[policy, civ]
            text = f"{int(val)}" if val > 0 else "0"
            # Use white text for darker cells, black for lighter cells
            text_color = "white" if val > count_matrix.max().max() * 0.5 else "black"
            plt.text(j, i, text, ha="center", va="center", fontsize=7, color=text_color, weight="bold")

    plt.xticks(ticks=range(len(civs)), labels=civs, rotation=90)
    plt.yticks(ticks=range(len(policies)), labels=policies)
    plt.xlabel("Civilization")
    plt.ylabel("Chosen Ancient Policy")
    plt.title("Policy Selection Count by Civilization and Ancient Policy")
    plt.colorbar(im, label="Count")
    plt.tight_layout()

    output_path = Path("tmp/heatmap_civ_policy_count.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved heatmap to {output_path}")


if __name__ == "__main__":
    main()

