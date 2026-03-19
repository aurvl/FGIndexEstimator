import matplotlib.patches as mpatches

ZONES = [
    (0,  25, "#d73027", 0.10),
    (25, 45, "#fc8d59", 0.10),
    (45, 55, "#fee08b", 0.10),
    (55, 75, "#91cf60", 0.10),
    (75, 100,"#1a9850", 0.10),
]

RATING_COLORS = {
    "extreme fear": "#d73027",
    "fear":         "#fc8d59",
    "neutral":      "#fee08b",
    "greed":        "#91cf60",
    "extreme greed":"#1a9850",
}

RATING_LABELS = {
    "extreme fear": "Extreme Fear (0–25)",
    "fear":         "Fear (25–45)",
    "neutral":      "Neutral (45–55)",
    "greed":        "Greed (55–75)",
    "extreme greed":"Extreme Greed (75–100)",
}

def draw_zones(ax):
    for ymin, ymax, color, alpha in ZONES:
        ax.axhspan(ymin, ymax, color=color, alpha=alpha, zorder=0)
    for y in [25, 45, 55, 75]:
        ax.axhline(y, color="white", linewidth=0.4, linestyle="--", alpha=0.3, zorder=1)


def add_legend_and_titles(ax):
    legend_patches = [
        mpatches.Patch(color=RATING_COLORS[k], label=RATING_LABELS[k])
        for k in ["extreme fear", "fear", "neutral", "greed", "extreme greed"]
    ]
    ax.legend(
        handles=legend_patches,
        loc="upper left", fontsize=8.5,
        facecolor="#1a1a2e", edgecolor="#444444",
        labelcolor="white", framealpha=0.8,
    )

    ax.set_title("CNN Fear & Greed Index — Daily History", fontsize=14,
                 color="white", fontweight="bold", pad=14)
    ax.set_ylabel("Index Value", fontsize=10, color="#aaaaaa")
