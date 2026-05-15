"""
Generate training reward plots for all FeynRL experiments.

Data formats
------------
Time-series (x_type="time"):
    MLflow export CSV with columns: run_id, key, value, step, timestamp
    X-axis is elapsed wall-clock time in minutes.

Step-series / MLflow step (x_type="mlflow_step"):
    MLflow export CSV with columns: run_id, key, value, step, timestamp
    X-axis is training step.  Optionally filter to a single run_id per framework.

Step-series / simple (x_type="step"):
    Simple CSV with a "step" column plus one column per metric.

Usage
-----
Plot all experiments:
    python generate_plots.py

Plot one or more experiments by key:
    python generate_plots.py --experiment math_qwen3_4b
    python generate_plots.py --experiment math_qwen3_4b --experiment math_qwen2_5_1b

Override smoothing and time window:
    python generate_plots.py --smooth 10 --max-x 120

Custom output directory:
    python generate_plots.py --output-dir /tmp/plots

List available experiment keys:
    python generate_plots.py --list

To add a new experiment:
  1. Drop its CSV into the appropriate data/ subfolder.
  2. Add an entry to EXPERIMENTS below.
  3. Run: python generate_plots.py --experiment <new_key>
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

HERE = pathlib.Path(__file__).parent

# ---------------------------------------------------------------------------
# Experiment registry
# Each key maps to a plot configuration.  `out` is relative to the
# experiment's subdirectory inside examples/ (e.g. math/plots/foo.png).
# ---------------------------------------------------------------------------
EXPERIMENTS = {
    # --- Math ---------------------------------------------------------------
    "math_qwen3_4b": dict(
        out="math/plots/qwen3_4b_thinking_2507.png",
        title="Reward vs Time — Framework Comparison",
        subtitle="Model: Qwen3-4B-Thinking-2507  |  Dataset: DeepScaleR  |  Reward: binary math_reward",
        x_label="Elapsed Time (minutes)",
        y_label="Reward",
        x_type="time",          # MLflow timestamp → elapsed minutes
        max_x=220,
        smooth=5,
        frameworks=[
            dict(
                file="math/data/qwen3-4b-thinking-2507/feynrl_sync.csv",
                reward_key="rollout/avg_reward",
                label="FeynRL (Sync)",
                color="black",
                linestyle="-",
            ),
            dict(
                file="math/data/qwen3-4b-thinking-2507/feynrl_async.csv",
                reward_key="rollout/avg_reward",
                label="FeynRL (Overlap)",
                color="#0f766e",
                linestyle="--",
            ),
        ],
    ),
    "math_qwen2_5_1b": dict(
        out="math/plots/qwen2_5_1b_instruct.png",
        title="Reward vs Time — Framework Comparison",
        subtitle="Model: Qwen2.5-1.5B-Instruct  |  Dataset: GSM8K  |  Reward: binary math_reward",
        x_label="Elapsed Time (minutes)",
        y_label="Reward",
        x_type="time",
        max_x=60,
        smooth=5,
        frameworks=[
            dict(
                file="math/data/qwen2.5-1.5b-instruct/FeynRL_sync.csv",
                reward_key="rollout/avg_reward",
                label="FeynRL (Sync)",
                color="black",
                linestyle="-",
            ),
            dict(
                file="math/data/qwen2.5-1.5b-instruct/FeynRL_async.csv",
                reward_key="rollout/avg_reward",
                label="FeynRL (Overlap)",
                color="#0f766e",
                linestyle="--",
            ),
        ],
    ),
    # --- HealthBench --------------------------------------------------------
    "healthbench_qwen2_5_1b": dict(
        out="healthbench/plots/qwen2_5_1b_instruct.png",
        title="Reward vs Training Step — Algorithm Comparison",
        subtitle="Model: Qwen2.5-1.5B-Instruct  |  Dataset: HealthBench  |  Reward: LLM judge",
        x_label="Training Step",
        y_label="Avg Reward",
        x_type="mlflow_step",
        smooth=5,
        frameworks=[
            dict(
                file="healthbench/data/healthbench.csv",
                run_id="261e81f2171c47f2b762929af52463dc",
                reward_key="rollout/avg_reward",
                label="GRPO",
                color="black",
                linestyle="-",
            ),
            dict(
                file="healthbench/data/healthbench.csv",
                run_id="b72185e79f9d4fa7a808d6979855da1c",
                reward_key="rollout/avg_reward",
                label="P3O",
                color="#0f766e",
                linestyle="--",
            ),
        ],
    ),
    # --- RAR-b Science ------------------------------------------------------
    # Drop rar_science/data/rar_science.csv (MLflow export) to enable.
    "rar_science_qwen2_5_1b": dict(
        out="rar_science/plots/qwen2_5_1b_instruct.png",
        title="Reward vs Training Step — Algorithm Comparison",
        subtitle="Model: Qwen2.5-1.5B-Instruct  |  Dataset: RAR-b Science  |  Reward: LLM judge",
        x_label="Training Step",
        y_label="Avg Reward",
        x_type="mlflow_step",
        smooth=5,
        frameworks=[
            dict(
                file="rar_science/data/rar_science.csv",
                run_id="724e71d1493e44f7a70eaed57ca53621",
                reward_key="rollout/avg_reward",
                label="GRPO",
                color="black",
                linestyle="-",
            ),
            dict(
                file="rar_science/data/rar_science.csv",
                run_id="c517c84dec214cc3aecd9585aa032106",
                reward_key="rollout/avg_reward",
                label="P3O",
                color="#0f766e",
                linestyle="--",
            ),
        ],
    ),
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_time_series(fw_cfg: dict) -> list[pd.Series]:
    """MLflow CSV → list of Series indexed by elapsed wall-clock minutes."""
    path = HERE / fw_cfg["file"]
    df = pd.read_csv(path)
    mask = df["key"] == fw_cfg["reward_key"]
    if not mask.any():
        available = df["key"].unique().tolist()
        raise ValueError(
            f"[{fw_cfg['file']}] reward_key '{fw_cfg['reward_key']}' not found. "
            f"Available: {available}"
        )
    df = df.loc[mask].copy()
    runs = []
    for _, group in df.groupby("run_id", sort=False):
        group = group.sort_values("timestamp")
        elapsed_min = (group["timestamp"] - group["timestamp"].iloc[0]) / 60_000
        runs.append(pd.Series(group["value"].values, index=elapsed_min.values))
    return runs


def load_mlflow_step_series(fw_cfg: dict) -> list[pd.Series]:
    """MLflow CSV → list of Series indexed by training step, optionally filtered to one run_id."""
    path = HERE / fw_cfg["file"]
    df = pd.read_csv(path)
    run_id = fw_cfg.get("run_id")
    if run_id is not None:
        df = df[df["run_id"] == run_id]
    mask = df["key"] == fw_cfg["reward_key"]
    if not mask.any():
        available = df["key"].unique().tolist()
        raise ValueError(
            f"[{fw_cfg['file']}] reward_key '{fw_cfg['reward_key']}' not found. "
            f"Available: {available}"
        )
    df = df.loc[mask].copy()
    runs = []
    for _, group in df.groupby("run_id", sort=False):
        group = group.sort_values("step")
        runs.append(pd.Series(group["value"].values, index=group["step"].values))
    return runs


def load_step_series(fw_cfg: dict) -> list[pd.Series]:
    """Simple CSV (step, col_a, ...) → list containing one Series indexed by step."""
    path = HERE / fw_cfg["file"]
    df = pd.read_csv(path)
    col = fw_cfg["reward_key"]
    if col not in df.columns:
        raise ValueError(
            f"[{fw_cfg['file']}] column '{col}' not found. "
            f"Available: {list(df.columns)}"
        )
    return [df.set_index("step")[col].dropna()]


def load_runs(fw_cfg: dict, x_type: str) -> list[pd.Series]:
    if x_type == "time":
        return load_time_series(fw_cfg)
    if x_type == "mlflow_step":
        return load_mlflow_step_series(fw_cfg)
    return load_step_series(fw_cfg)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def smooth_series(series: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return series
    return series.rolling(window=window, min_periods=1, center=True).mean()


def truncate_series(series: pd.Series, max_x: float | None) -> pd.Series:
    """Clip series at max_x (inclusive), interpolating the boundary point."""
    if max_x is None or series.empty:
        return series
    if series.index.max() <= max_x:
        return series
    before = series[series.index <= max_x]
    after  = series[series.index > max_x]
    if after.empty:
        return before
    return pd.concat([before, pd.Series([after.iloc[0]], index=[max_x])])


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
            "legend.framealpha": 0.9,
            "legend.edgecolor": "#cccccc",
            "figure.dpi": 400,
        }
    )


def plot_experiment(
    key: str,
    cfg: dict,
    smooth_window: int | None,
    max_x: float | None,
    output_dir: pathlib.Path | None,
) -> None:
    apply_style()

    window    = smooth_window if smooth_window is not None else cfg.get("smooth", 5)
    eff_max_x = max_x        if max_x        is not None else cfg.get("max_x")
    x_type    = cfg.get("x_type", "time")

    out_path = (output_dir / pathlib.Path(cfg["out"]).name) if output_dir else (HERE / cfg["out"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for fw in cfg["frameworks"]:
        runs = load_runs(fw, x_type)
        color    = fw["color"]
        ls       = fw.get("linestyle", "-")
        alpha_sc = fw.get("alpha_scale", 1.0)

        for i, raw in enumerate(runs):
            if x_type == "time":
                raw = truncate_series(raw, eff_max_x)
            s     = smooth_series(raw, window)
            label = fw["label"] if i == 0 else "_nolegend_"

            ax.plot(
                s.index, s.values,
                label=label, color=color, linestyle=ls,
                linewidth=2.2, marker="o", markersize=3.2,
                markeredgewidth=0,
                alpha=(0.9 if len(runs) == 1 else 0.75) * alpha_sc,
            )
            if window > 1:
                ax.plot(
                    raw.index, raw.values,
                    color=color, linestyle=ls,
                    linewidth=0.8, marker="o",
                    markersize=max(3.2 - 0.8, 1.5),
                    markeredgewidth=0, alpha=0.2 * alpha_sc,
                )

    ax.set_xlabel(cfg.get("x_label", "Elapsed Time (minutes)"), fontsize=13)
    ax.set_ylabel(cfg.get("y_label", "Reward"), fontsize=13)
    ax.set_title(cfg["title"], fontsize=14, fontweight="bold", pad=18)
    ax.text(
        0.5, 1.01, cfg["subtitle"],
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=9.5, color="#5A5A5A",
    )
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.legend(loc="lower right", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate training reward plots for FeynRL experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Available experiment keys:\n  " + "\n  ".join(EXPERIMENTS),
    )
    parser.add_argument(
        "--experiment", "-e",
        action="append",
        dest="experiments",
        metavar="KEY",
        help=(
            "Experiment key to plot (may be repeated). "
            "Defaults to all experiments when omitted."
        ),
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=None,
        metavar="N",
        help="Rolling-average window (default: per-experiment value). Set 1 to disable.",
    )
    parser.add_argument(
        "--max-x",
        type=float,
        default=None,
        metavar="V",
        help=(
            "Truncate x-axis at V (minutes for time-series experiments, "
            "steps for step-series). Overrides the per-experiment default."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=None,
        metavar="PATH",
        help="Write all PNGs to PATH instead of their default locations.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print available experiment keys and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list:
        print("Available experiments:")
        for key in EXPERIMENTS:
            print(f"  {key}")
        return

    selected = args.experiments or list(EXPERIMENTS)
    unknown  = [k for k in selected if k not in EXPERIMENTS]
    if unknown:
        raise SystemExit(
            f"Unknown experiment key(s): {unknown}. "
            f"Run with --list to see available keys."
        )

    for key in selected:
        print(f"Plotting: {key}")
        plot_experiment(
            key=key,
            cfg=EXPERIMENTS[key],
            smooth_window=args.smooth,
            max_x=args.max_x,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
