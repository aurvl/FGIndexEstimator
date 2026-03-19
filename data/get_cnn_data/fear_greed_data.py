"""
─────────────────────────────────────────────────────────────────────────────
REQUIRED HTML STRUCTURE FOR PARSING
─────────────────────────────────────────────────────────────────────────────
This script expects an HTML file saved from CNN's Fear & Greed Index page:
https://edition.cnn.com/markets/fear-and-greed

The parser relies on a single element:

  <div class="market-line-chart" data-instance="...URL-encoded JSON...">

The JSON embedded in `data-instance` must contain the path:
  lines[0].data.series  →  list of { x: timestamp_ms, y: float, rating: str }

HOW TO SAVE THE PAGE CORRECTLY:
  1. Open the CNN Fear & Greed page in your browser
  2. Click the "Timeline" tab to make sure the chart is loaded
  3. Save the page (Ctrl+S) as "Webpage, Complete" or "Webpage, HTML Only"
  4. Pass the saved .html file path to this script

Everything else in the HTML (the gauge SVG, historical snapshots, etc.)
is not required and can be absent without affecting the parsing.
─────────────────────────────────────────────────────────────────────────────
"""

import json
import urllib.parse
from pathlib import Path
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
import numpy as np
from colorama import Fore, Style

BOLD  = "\033[1m"
RED   = "\033[31m"
GREEN = "\033[32m"
RESET = "\033[0m"

ent = f"""
─────────────────────────────────────────────────────────────────────────────
REQUIRED HTML STRUCTURE FOR PARSING
─────────────────────────────────────────────────────────────────────────────
This script expects an HTML file saved from CNN's Fear & Greed Index page:
https://edition.cnn.com/markets/fear-and-greed

{BOLD}{RED}!!! The parser relies on a single element:

  <div class="market-line-chart" data-instance="...URL-encoded JSON...">
{RESET}{GREEN}
The JSON embedded in `data-instance` must contain the path:
  lines[0].data.series  →  list of {{ x: timestamp_ms, y: float, rating: str }}

HOW TO SAVE THE PAGE CORRECTLY:
  1. Open the CNN Fear & Greed page in your browser
  2. Click the "Timeline" tab to make sure the chart is loaded
  3. Save the page (Ctrl+S) as "Webpage, Complete" or "Webpage, HTML Only"
  4. Pass the saved .html file path to this script

Everything else in the HTML (the gauge SVG, historical snapshots, etc.)
is not required and can be absent without affecting the parsing.
─────────────────────────────────────────────────────────────────────────────
"""

print(Fore.GREEN + ent + Style.RESET_ALL)

# ── 1. Extraction ──────────────────────────────────────────────────────────────
base_cwd = Path.cwd()
script_dir = Path(__file__).resolve().parent

save_dir_input = input("Directory to save outputs (or press Enter to use current directory): ").strip()
output_dir = Path(save_dir_input).expanduser() if save_dir_input else base_cwd
if not output_dir.is_absolute():
    output_dir = (base_cwd / output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

path_input = input("Full path to the Fear & Greed Index HTML file (or press Enter to use default): ").strip()

default_html = script_dir / "markets_fear-and-greed_19-03-25_to_19-03-26.html"
if not path_input:
    # Prefer a file next to this script; fallback to the historical absolute path.
    path = default_html if default_html.exists() else Path(
        r"C:\Users\aurel\Desktop\Projects\IndexEstimation\data\get_cnn_data\markets_fear-and-greed_19-03-25_to_19-03-26.html"
    )
else:
    candidate = Path(path_input).expanduser()
    if candidate.is_absolute():
        path = candidate
    else:
        tried = [base_cwd / candidate, script_dir / candidate, output_dir / candidate]
        path = next((p for p in tried if p.exists()), None)
        if path is None:
            tried_str = "\n".join(f"  - {p}" for p in tried)
            raise FileNotFoundError(
                f"HTML file not found: {path_input}\nTried:\n{tried_str}"
            )

with open(path, "r", encoding="utf-8", errors="replace") as f:
    html = f.read()

soup = BeautifulSoup(html, "html.parser")
chart = soup.find("div", class_="market-line-chart")
data = json.loads(urllib.parse.unquote(chart["data-instance"]))

series = data["lines"][0]["data"]["series"]

df = pd.DataFrame(series)
df["date"] = pd.to_datetime(df["x"], unit="ms")
df = df.rename(columns={"y": "value"})[["date", "value", "rating"]]
df = df.sort_values("date").reset_index(drop=True)

print(f"Données extraites : {len(df)} points")
print(f"Période : {df['date'].min().date()} → {df['date'].max().date()}")
print(df.tail())

# ── 2. Couleurs par rating ──────────────────────────────────────────────────────
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

ZONES = [
    (0,  25, "#d73027", 0.10),
    (25, 45, "#fc8d59", 0.10),
    (45, 55, "#fee08b", 0.10),
    (55, 75, "#91cf60", 0.10),
    (75, 100,"#1a9850", 0.10),
]

# ── 3. Figure ──────────────────────────────────────────────────────────────────
def style_axes(ax):
    ax.set_ylim(0, 100)
    ax.set_xlim(df["date"].min(), df["date"].max())

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=9, color="#aaaaaa")

    ax.set_yticks([0, 25, 45, 55, 75, 100])
    ax.set_yticklabels(["0", "25", "45", "55", "75", "100"], fontsize=9, color="#aaaaaa")

    ax.tick_params(axis="both", colors="#aaaaaa", length=3)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    ax.grid(axis="x", color="#333333", linewidth=0.5, linestyle="--", alpha=0.5)


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


# ── 3A. Static plot (saved to disk) ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor("#0f1117")
ax.set_facecolor("#0f1117")

draw_zones(ax)

for i in range(len(df) - 1):
    row = df.iloc[i]
    ax.plot(
        [df["date"].iloc[i], df["date"].iloc[i + 1]],
        [df["value"].iloc[i], df["value"].iloc[i + 1]],
        color=RATING_COLORS[row["rating"]],
        linewidth=1.8,
        solid_capstyle="round",
        zorder=2,
    )

scatter_colors = [RATING_COLORS[r] for r in df["rating"]]
ax.scatter(df["date"], df["value"], c=scatter_colors, s=8, zorder=3)

last = df.iloc[-1]
ax.scatter(
    last["date"], last["value"],
    color=RATING_COLORS[last["rating"]], s=80, zorder=5,
    edgecolors="white", linewidths=1.2,
)
ax.annotate(
    f"  {last['value']:.0f}\n  {last['rating'].title()}",
    xy=(last["date"], last["value"]),
    fontsize=10, color="white", fontweight="bold",
    va="center", zorder=6,
)

# ── 4. Axes & grille ───────────────────────────────────────────────────────────
style_axes(ax)

# ── 5. Légende & titres ────────────────────────────────────────────────────────
add_legend_and_titles(ax)

plt.tight_layout()
png_path = output_dir / "fear_greed_plot.png"
plt.savefig(str(png_path), dpi=150, bbox_inches="tight", facecolor="#0f1117")
plt.close(fig)

# ── 3B. Animated replay (shown on screen) ─────────────────────────────────────
fig_anim, ax_anim = plt.subplots(figsize=(14, 6))
fig_anim.patch.set_facecolor("#0f1117")
ax_anim.set_facecolor("#0f1117")

draw_zones(ax_anim)
style_axes(ax_anim)
add_legend_and_titles(ax_anim)

x_num = mdates.date2num(df["date"])
y_val = df["value"].to_numpy()

line, = ax_anim.plot([], [], color="white", linewidth=1.8, solid_capstyle="round", zorder=2)
dot = ax_anim.scatter([], [], s=80, color="white", zorder=5, edgecolors="white", linewidths=1.2)
label = ax_anim.annotate(
    "",
    xy=(0, 0),
    fontsize=10, color="white", fontweight="bold",
    va="center", zorder=6,
)


def init_anim():
    line.set_data([], [])
    dot.set_offsets(np.empty((0, 2)))
    label.set_text("")
    return line, dot, label


def update_anim(i: int):
    # i is 1..N
    idx = i - 1
    line.set_data(x_num[:i], y_val[:i])

    dot.set_offsets([[x_num[idx], y_val[idx]]])
    rating = df["rating"].iloc[idx]
    dot.set_facecolor(RATING_COLORS.get(rating, "white"))

    label.xy = (x_num[idx], y_val[idx])
    label.set_text(f"  {y_val[idx]:.0f}\n  {str(rating).title()}")
    return line, dot, label


ani = FuncAnimation(
    fig_anim,
    update_anim,
    frames=range(1, len(df) + 1),
    init_func=init_anim,
    interval=25,
    blit=False,
    repeat=False,
)

plt.tight_layout()
plt.show()

# save csv file
min_date = df["date"].min().strftime("%Y-%m-%d")
max_date = df["date"].max().strftime("%Y-%m-%d")
file_name = f"fear_greed_cnn_{min_date}_to_{max_date}.csv"
csv_path = output_dir / file_name
df.to_csv(csv_path, index=False)

print(f"Figure sauvegardée : {png_path}")
print(f"Données sauvegardées : {csv_path}")
