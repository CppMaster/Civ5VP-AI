"""
Generate a heatmap of win percentage by civilization and chosen ancient policy.
X-axis: civilization
Y-axis: chosen_ancient_policy
Cell value: percentage of games won (0-100) for that civ/policy combo.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def main() -> None:
    csv_path = Path("data/processed/game_data.csv")
    df = pd.read_csv(csv_path)

    # Compute win rate per (civilization, policy)
    civs = sorted(df["civilization"].unique())
    policies = sorted(df["chosen_ancient_policy"].unique())

    win_rate = (
        df.groupby(["chosen_ancient_policy", "civilization"])["won"]
        .mean()
        .mul(100)
        .reindex(
            pd.MultiIndex.from_product([policies, civs], names=["chosen_ancient_policy", "civilization"])
        )
        .unstack(fill_value=np.nan)
    )

    plt.figure(figsize=(max(10, len(civs) * 0.4), max(4, len(policies) * 0.4)))
    im = plt.imshow(win_rate, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")

    # Annotate each cell with the win % (or dash if NaN)
    for i, policy in enumerate(policies):
        for j, civ in enumerate(civs):
            val = win_rate.loc[policy, civ]
            text = f"{val:.0f}%" if not np.isnan(val) else "–"
            plt.text(j, i, text, ha="center", va="center", fontsize=7, color="black")

    plt.xticks(ticks=range(len(civs)), labels=civs, rotation=90)
    plt.yticks(ticks=range(len(policies)), labels=policies)
    plt.xlabel("Civilization")
    plt.ylabel("Chosen Ancient Policy")
    plt.title("Win Percentage by Civilization and Ancient Policy")
    plt.colorbar(im, label="Win %")
    plt.tight_layout()

    output_path = Path("tmp/heatmap_civ_policy_winrate.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved heatmap to {output_path}")


if __name__ == "__main__":
    main()

