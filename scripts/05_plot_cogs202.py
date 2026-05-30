# Plots the cogs202 sweep DVs (produced by 04_sweep_cogs202.py) as rich-vs-lazy
# line plots, one panel per noise condition.
#
#   "transfer time": y = t (time to near-asymptotic Task-B loss)
#   "interference":  y = learn_auc + transfer_auc
#
# Each panel overlays the per-participant data (jittered scatter, fixed seed) and
# a line connecting the regime means with 95% bootstrapped CIs.
#
# Figures are written to figures/cogs202/:
#   fig_transfer_time.png       - transfer time, one panel per noise condition
#   fig_interference.png        - interference, one panel per noise condition
#   fig_noise_A{a}_B{b}.png     - transfer time + interference side by side, per noise
#
#   uv run scripts/05_plot_cogs202.py
#   uv run scripts/05_plot_cogs202.py --csv data/cogs202_results_participants.csv

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

project_root = Path(__file__).resolve().parent
while not (project_root / 'src').exists():
    if project_root == project_root.parent:
        raise RuntimeError("Project root directory not found.")
    project_root = project_root.parent
sys.path.append(str(project_root))

from src.utils.figure_settings import cm_conv  # noqa: F401  (applies mpl rcParams)

REGIME_ORDER = ['rich_50', 'lazy_50']
REGIME_LABELS = {'lazy_50': 'Splitters', 'rich_50': 'Lumpers'}
REGIME_COLOURS = {'Splitters': '#D9544D', 'Lumpers': '#56B4E9'}
ORDER = [REGIME_LABELS[r] for r in REGIME_ORDER]

SEED = 0          # fixes both the scatter jitter and the CI bootstrap resampling
N_BOOT = 10000


# (value_col, title, y-axis label) for each DV
METRICS = [
    ('t', 'Transfer Time', 'time to percentile'),
    ('interference', 'Interference', 'learn_auc'),
]


def noise_label(a_sd, b_sd):
    return f'A={a_sd}, B={b_sd}'


def draw_cell(ax, cell, value_col):
    """Jittered per-participant points + a regime-mean line with 95%
    bootstrapped CIs, on a single axis."""
    np.random.seed(SEED)  # seaborn jitter draws from the global RNG
    sns.stripplot(data=cell, x='regime', y=value_col, order=ORDER,
                  hue='regime', palette=REGIME_COLOURS, legend=False,
                  jitter=0.15, size=3, alpha=0.45, linewidth=0, ax=ax)
    sns.pointplot(data=cell, x='regime', y=value_col, order=ORDER,
                  errorbar=('ci', 95), n_boot=N_BOOT, seed=SEED,
                  color='black', markersize=4, linewidth=1,
                  capsize=0.15, err_kws={'linewidth': 1}, ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.margins(x=0.25)
    ax.spines[['top', 'right']].set_visible(False)


def noise_cells(df):
    return sorted(df[['a_error_sd', 'b_error_sd']].drop_duplicates().itertuples(index=False))


def plot_metric(df, value_col, title, ylabel, out_path):
    """One figure per metric: a panel per noise condition."""
    cells = noise_cells(df)
    fig, axes = plt.subplots(1, len(cells), figsize=[3.2 * len(cells) * cm_conv, 4.5 * cm_conv],
                             sharey=True)
    axes = np.atleast_1d(axes)

    for ax, (a_sd, b_sd) in zip(axes, cells):
        cell = df[(df['a_error_sd'] == a_sd) & (df['b_error_sd'] == b_sd)].copy()
        cell['regime'] = cell['regime'].map(REGIME_LABELS)
        draw_cell(ax, cell, value_col)
        ax.set_title(noise_label(a_sd, b_sd))

    axes[0].set_ylabel(ylabel)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=500, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_by_noise(df, out_dir):
    """One figure per noise condition: transfer time and interference side by side."""
    for a_sd, b_sd in noise_cells(df):
        cell = df[(df['a_error_sd'] == a_sd) & (df['b_error_sd'] == b_sd)].copy()
        cell['regime'] = cell['regime'].map(REGIME_LABELS)

        fig, axes = plt.subplots(1, len(METRICS), figsize=[4.5 * len(METRICS) * cm_conv, 4.5 * cm_conv])
        for ax, (value_col, title, ylabel) in zip(axes, METRICS):
            draw_cell(ax, cell, value_col)
            ax.set_title(title)
            ax.set_ylabel(ylabel)

        fig.suptitle(noise_label(a_sd, b_sd))
        fig.tight_layout()
        out_path = out_dir / f'fig_noise_A{a_sd}_B{b_sd}.png'
        fig.savefig(out_path, dpi=500, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot cogs202 sweep results')
    parser.add_argument('--csv', type=str,
                        default=str(project_root / 'data' / 'cogs202_results_participants.csv'),
                        help='Per-participant CSV from 04_sweep_cogs202.py')
    parser.add_argument('--out-dir', type=str, default=str(project_root / 'figures' / 'cogs202'))
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df['interference'] = df['learn_auc'] # + df['transfer_auc']

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_metric(df, 't', 'Transfer Time', 'time to percentile',
                out_dir / 'fig_transfer_time.png')
    plot_metric(df, 'interference', 'Interference', 'learn_auc',
                out_dir / 'fig_interference.png')
    plot_by_noise(df, out_dir)


if __name__ == '__main__':
    main()
