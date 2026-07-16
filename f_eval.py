"""
eval.py — load a trained checkpoint, score it on a held-out split, and plot.

Reports error in PHYSICAL units (real Gamma/kappa, via decode) so the numbers
mean something, and draws two figures:
  1. pred_vs_true.png  — parity panels (Gamma on log-log, kappa linear), square
     axes with equal limits so the 45° identity line is honest
  2. error_heatmap.png — mean relative Gamma error per (Gamma, kappa) cell over
     the phase plane (expect the melting line to light up on the full dataset)

Also writes outputs/metrics.txt so the numbers survive next to the figures.

Run after training has written a checkpoint:  python3 eval.py
Depends on dataset.py's contract: read_manifest, split, decode, PlasmaDataset.
"""
from a_imports import *

# --- style tokens (light surface) --------------------------------------------
INK     = "#0b0b0b"   # primary text
INK_2   = "#52514e"   # secondary text (axis labels, annotations)
MUTED   = "#898781"   # tick labels, reference lines
GRID    = "#e1e0d9"   # hairline gridlines
AXIS    = "#c3c2b7"   # spines
SURFACE = "#fcfcfb"   # figure/axes background, marker rings
BLUE    = "#2a78d6"   # the one series color

# one-hue sequential ramp (light -> dark) for the error map
SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b",
])

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "text.color": INK,
    "axes.labelcolor": INK_2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.titleweight": "bold",
})


@torch.no_grad()
def run_inference(model, loader, device):
    """Run the model over a loader. Returns (preds, trues) in MODEL space,
    both [N, 2] numpy arrays."""
    model.eval()                       # dropout OFF, batchnorm uses running stats
    preds, trues = [], []
    for images, labels in loader:
        out = model(images.to(device)).cpu().numpy()
        preds.append(out)
        trues.append(labels.numpy())
    return np.concatenate(preds), np.concatenate(trues)


def to_physical(arr):
    """Decode an [N, 2] array of model-space labels into physical (Gamma, kappa)
    columns. Returns (gammas, kappas), each [N]."""
    gammas, kappas = [], []
    for row in arr:
        g, k = decode(row)
        gammas.append(g)
        kappas.append(k)
    return np.array(gammas), np.array(kappas)


def report(true, pred, name):
    """Compute + print MAE and R^2 for one target. Returns (mae, r2)."""
    mae = float(np.mean(np.abs(true - pred)))
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - np.mean(true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    print(f"  {name:12s}  MAE = {mae:10.4f}   R2 = {r2:7.4f}")
    return mae, r2


def _parity_panel(ax, true, pred, name, mae, r2, log=False):
    """One square parity panel: equal limits, 45° identity line, stats box."""
    if log:
        ax.set_xscale("log"); ax.set_yscale("log")
        lo = min(true.min(), pred.min()) * 0.7
        hi = max(true.max(), pred.max()) * 1.4
    else:
        lo, hi = min(true.min(), pred.min()), max(true.max(), pred.max())
        pad = max(0.15 * (hi - lo), 0.25)   # floor keeps degenerate ranges visible
        lo, hi = lo - pad, hi + pad

    # identity reference, direct-labeled (no legend box needed for one series)
    ax.plot([lo, hi], [lo, hi], ls=(0, (4, 4)), lw=1, color=MUTED, zorder=2)
    ax.scatter(true, pred, s=72, color=BLUE, alpha=0.85,
               edgecolor=SURFACE, linewidth=1.2, zorder=3)

    # equal limits + equal aspect -> the identity line is truly 45 degrees
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal")

    # bottom-right corner is the empty zone in a parity plot (data hugs the diagonal)
    ax.text(0.96, 0.05, "y = x", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9, color=MUTED)
    ax.text(0.05, 0.95, f"MAE {mae:.3g}\nR² {r2:.3f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=9, color=INK_2, linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.4", fc=SURFACE, ec=GRID))
    ax.set_xlabel(f"true {name}")
    ax.set_ylabel(f"predicted {name}")


def plot_pred_vs_true(g_true, g_pred, k_true, k_pred, stats, out_path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 5.2))
    _parity_panel(ax[0], g_true, g_pred, "$\\Gamma$", *stats["gamma"], log=True)
    ax[0].set_title("$\\Gamma$  (log–log)")
    _parity_panel(ax[1], k_true, k_pred, "$\\kappa$", *stats["kappa"])
    ax[1].set_title("$\\kappa$")
    fig.suptitle("Predicted vs true", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_error_map(g_true, k_true, g_pred, out_path):
    """Mean relative Gamma error per (Gamma, kappa) cell, on the phase plane.

    Aggregating per cell matters: the eval split holds several seeds of the same
    cell, and raw overplotting would hide all but the last-drawn point.
    """
    rel_err = np.abs(g_pred - g_true) / g_true

    cells = {}                                   # (gamma, kappa) -> list of errors
    for g, k, e in zip(g_true, k_true, rel_err):
        cells.setdefault((g, k), []).append(e)
    gs = np.array([gk[0] for gk in cells])
    ks = np.array([gk[1] for gk in cells])
    errs = np.array([100 * np.mean(v) for v in cells.values()])   # as %

    fig, ax = plt.subplots(figsize=(7.2, 5))
    sc = ax.scatter(gs, ks, c=errs, cmap=SEQ_BLUE,
                    vmin=0.0, vmax=max(float(errs.max()) * 1.05, 1e-6),
                    s=300, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.set_xscale("log")

    # ticks at the actual grid values, not generic log decades
    ax.set_xticks(sorted(set(g_true)))
    # 3 sig figs below 100, plain integers above (avoids "1.2e+03" for 1200)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:.0f}" if v >= 100 else f"{v:.3g}"))
    ax.minorticks_off()
    ax.tick_params(axis="x", rotation=30)
    ax.set_yticks(sorted(set(k_true)))

    ax.set_xlabel("$\\Gamma$")
    ax.set_ylabel("$\\kappa$")
    ax.set_title("Relative $\\Gamma$ error across the phase plane")

    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("mean $|\\Delta\\Gamma| \\, / \\, \\Gamma$  (%)", color=INK_2)
    cb.ax.tick_params(color=MUTED, labelcolor=MUTED)
    cb.outline.set_edgecolor(AXIS)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    os.makedirs(g_configs.OUT_DIR, exist_ok=True)

    # Build the eval split. Pilot has no test set, so fall back to val.
    rows = read_manifest(g_configs.MANIFEST)
    _, val_rows, test_rows = split(rows, g_configs.TRAIN_FRAC, g_configs.VAL_FRAC)
    eval_rows = test_rows if len(test_rows) > 0 else val_rows
    split_name = "test" if len(test_rows) > 0 else "val"
    print(f"evaluating on {len(eval_rows)} samples ({split_name} split)")

    ds = dataset(eval_rows, g_configs.DATA_DIR,
                       resolution=g_configs.RESOLUTION, blob_sigma=g_configs.BLOB_SIGMA,
                       augment=False)
    loader = DataLoader(ds, batch_size=g_configs.BATCH_SIZE, shuffle=False)

    model = CNN().to(g_configs.DEVICE)
    model.load_state_dict(torch.load(g_configs.CKPT_PATH, map_location=g_configs.DEVICE))

    preds, trues = run_inference(model, loader, g_configs.DEVICE)
    g_pred, k_pred = to_physical(preds)
    g_true, k_true = to_physical(trues)

    print("\n=== metrics (physical units) ===")
    stats = {
        "gamma":  report(g_true, g_pred, "Gamma"),
        "kappa":  report(k_true, k_pred, "kappa"),
        "log10g": report(np.log10(g_true), np.log10(g_pred), "log10(Gamma)"),
    }

    # persist the numbers next to the figures
    metrics_path = os.path.join(g_configs.OUT_DIR, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"split: {split_name} ({len(eval_rows)} samples)\n")
        for name, (mae, r2) in stats.items():
            f.write(f"{name:8s}  MAE = {mae:.6g}   R2 = {r2:.4f}\n")
    print(f"\nwrote {metrics_path}")

    plot_pred_vs_true(g_true, g_pred, k_true, k_pred, stats,
                      os.path.join(g_configs.OUT_DIR, "pred_vs_true.png"))
    plot_error_map(g_true, k_true, g_pred,
                   os.path.join(g_configs.OUT_DIR, "error_heatmap.png"))
