#!/usr/bin/env python3
# ABOUTME: Plots cost vs performance analysis for consolidated experiment results
# ABOUTME: Simplified version leveraging structured CSV schema

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# Pricing per 1000 tokens (already in $ per 1000 tokens)
PRICING = {
    "o3-mini": {"prompt": 0.55 / 1000, "completion": 4.40 / 1000},
    "o4-mini": {"prompt": 1.1 / 1000, "completion": 4.40 / 1000},
    "gemini-flash-2.5": {"prompt": 0.35 / 1000, "completion": 0.7 / 1000},
    "llama": {"prompt": 0.0001 / 1000, "completion": 0.0001 / 1000},
    "deepseek-r1": {"prompt": 1.35 / 1000, "completion": 5.4 / 1000},
}

# Display names
DATASET_NAMES = {
    "DA-KGQA": "DA-KGQA",
    "ReAct(noKG)": "ReAct\n(No KG)",
    "ReAct(100)": "ReAct\n(100 Triples)",
    "ReAct(5000)": "ReAct\n(5000 Triples)",
}

MODEL_NAMES = {
    "o3-mini": "o3-mini",
    "o4-mini": "o4-mini",
    "gemini-flash-2.5": "g/flash-2.5",
    "llama": "Llama 4",
    "deepseek-r1": "Deepseek R1",
}

MODEL_MARKERS = {
    "o3-mini": 'o',
    "o4-mini": 's',
    "gemini-flash-2.5": 'D',
    "llama": '*',
    "deepseek-r1": 'P',
}

# Define which (dataset, model) pairs to plot
APPROACHES = [
    ("DA-KGQA", "o3-mini"),
    ("DA-KGQA", "gemini-flash-2.5"),
    ("DA-KGQA", "llama"),
    ("ReAct(noKG)", "o3-mini"),
    ("ReAct(noKG)", "gemini-flash-2.5"),
    ("ReAct(noKG)", "llama"),
    ("ReAct(100)", "o3-mini"),
    ("ReAct(100)", "o4-mini"),
    ("ReAct(100)", "gemini-flash-2.5"),
    ("ReAct(100)", "llama"),
    ("ReAct(100)", "deepseek-r1"),
    ("ReAct(5000)", "o3-mini"),
    ("ReAct(5000)", "gemini-flash-2.5"),
    ("ReAct(5000)", "llama"),
]


def calculate_cost(row, model):
    """Calculate cost from token counts"""
    if pd.isna(row.get('prompt_tokens')) or pd.isna(row.get('completion_tokens')):
        return np.nan

    prices = PRICING.get(model, PRICING["o3-mini"])
    prompt_cost = (row['prompt_tokens'] / 1000) * prices['prompt']
    completion_cost = (row['completion_tokens'] / 1000) * prices['completion']
    return prompt_cost + completion_cost


def load_data(approaches):
    """Load all approach data and compute costs"""
    all_data = []

    for dataset, model in approaches:
        csv_path = os.path.join("./Experiment_Results/", dataset, f"{model}.csv")

        if not os.path.exists(csv_path):
            print(f"⚠️  Skipped: {csv_path} not found")
            continue

        print(f"📂 Loading: {dataset}/{model}.csv")
        df = pd.read_csv(csv_path)

        # Add computed columns
        df['cost'] = df.apply(lambda row: calculate_cost(row, model), axis=1)
        df['dataset'] = dataset
        df['model'] = model
        df['approach'] = f"{dataset}_{model}"  # Unique identifier

        all_data.append(df)

    return pd.concat(all_data, ignore_index=True) if all_data else None


def compute_summary(df, metric):
    """Compute mean and SEM for each approach"""
    return df.groupby('approach').agg(
        dataset=('dataset', 'first'),
        model=('model', 'first'),
        f1_mean=(metric, 'mean'),
        f1_sem=(metric, lambda x: stats.sem(x, nan_policy='omit')),
        cost_mean=('cost', 'mean'),
        cost_sem=('cost', lambda x: stats.sem(x, nan_policy='omit')),
        count=(metric, 'count'),
    ).reset_index()


def plot_data(summary_df, metric):
    """Create the scatter plot"""
    # Setup colors for datasets
    base_datasets = list(DATASET_NAMES.keys())
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0.1, 0.9, len(base_datasets)))
    dataset_colors = dict(zip(base_datasets, colors))

    # Configure plot
    import matplotlib
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    plt.style.use('seaborn-v0_8-whitegrid')

    fig, ax = plt.subplots(figsize=(6, 5))

    # Plot each point
    for _, row in summary_df.iterrows():
        dataset = row['dataset']
        model = row['model']

        marker = MODEL_MARKERS.get(model, 'x')
        color = dataset_colors.get(dataset, 'grey')

        ax.errorbar(
            x=row['cost_mean'], y=row['f1_mean'],
            xerr=row['cost_sem'], yerr=row['f1_sem'],
            fmt=marker, color=color, markersize=10, capsize=5,
            alpha=0.9, markeredgecolor='black', zorder=5
        )

        # Annotate (skip Llama to reduce clutter)
        if model != 'llama':
            label = f"{DATASET_NAMES[dataset]}\n{MODEL_NAMES[model]}"

            # Custom positioning based on dataset and model
            if dataset == "ReAct(100)" and model == "o4-mini":
                offset = (0, 10)
                ha, va = 'center', 'bottom'
                y_pos = row['f1_mean'] + row['f1_sem'] - 0.015
            elif dataset == "ReAct(100)" and model == "gemini-flash-2.5":
                offset = (0, 10)
                ha, va = 'center', 'bottom'
                y_pos = row['f1_mean'] + row['f1_sem'] - 0.018
            elif dataset == "ReAct(100)" and model == "o3-mini":
                offset = (0, 10)
                ha, va = 'center', 'top'
                y_pos = row['f1_mean'] + row['f1_sem'] - 0.018
            elif dataset == "DA-KGQA" and model == "gemini-flash-2.5":
                offset = (0, 10)
                ha, va = 'left', 'top'
                y_pos = row['f1_mean'] + row['f1_sem'] - 0.01
            elif dataset == "DA-KGQA":
                offset = (0, 10)
                ha, va = 'left', 'top'
                y_pos = row['f1_mean'] - 0.015
            else:
                offset = (0, 10)
                ha, va = 'center', 'bottom'
                y_pos = row['f1_mean'] + row['f1_sem'] - 0.018

            ax.annotate(
                label, xy=(row['cost_mean'], y_pos),
                textcoords="offset points", xytext=offset,
                ha=ha, va=va, fontsize=10, color=color,
                fontweight='demi', zorder=6, linespacing=1.1
            )

    # Axis labels
    ax.set_xlabel('Average Cost per Question ($)', fontsize=14, fontweight='bold')
    ylabel_map = {
        'row_matching_f1': 'Mean Row Matching F1',
        'entity_set_f1': 'Mean Entity Set F1',
        'exact_match_f1': 'Mean Exact Match F1',
    }
    ax.set_ylabel(ylabel_map.get(metric, f'Mean {metric}'), fontsize=14, fontweight='bold')

    # Limits and ticks
    ax.set_ylim(-0.005, 0.6)
    ax.set_xlim(-0.003, 0.088)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune='upper'))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))

    # Legends
    # Model legend (markers)
    model_legend = [
        Line2D([0], [0], marker=MODEL_MARKERS[m], color='w', label=MODEL_NAMES[m],
               markerfacecolor='gray', markeredgecolor='black', markersize=9)
        for m in ["o3-mini", "gemini-flash-2.5", "llama", "o4-mini", "deepseek-r1"]
    ]
    legend1 = ax.legend(handles=model_legend, title="Model", fontsize=10,
                       title_fontsize=12, loc='upper left')
    ax.add_artist(legend1)

    # Dataset legend (colors)
    dataset_legend = [
        Patch(facecolor=dataset_colors[d], label=DATASET_NAMES[d])
        for d in base_datasets
    ]
    ax.legend(handles=dataset_legend, title="Method", fontsize=10,
             title_fontsize=12, loc='upper right', ncols=2)

    # Grid and save
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=0)
    fig.tight_layout()

    filename = f"cost_vs_performance_{metric}.pdf"
    plt.savefig(filename, dpi=300)
    print(f"\n✅ Plot saved: {filename}")
    plt.show()


def main(metric='row_matching_f1'):
    """Main plotting function"""
    print("="*80)
    print("Cost vs Performance Analysis")
    print("="*80)

    # Load data
    df = load_data(APPROACHES)
    if df is None:
        print("❌ No data loaded. Exiting.")
        return

    # Ensure numeric
    df[metric] = pd.to_numeric(df[metric], errors='coerce').fillna(0)
    df['cost'] = pd.to_numeric(df['cost'], errors='coerce')

    # Compute summary statistics
    summary = compute_summary(df, metric)

    # Print summary
    print("\n" + "="*80)
    print("Summary Statistics")
    print("="*80)
    print(summary[['dataset', 'model', 'count', 'f1_mean', 'cost_mean']].to_string(index=False))
    print("="*80)

    # Plot
    plot_data(summary, metric)


if __name__ == "__main__":
    main(metric='row_matching_f1')
