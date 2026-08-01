import argparse
import sys
from pathlib import Path

import numpy as np
import zarr

# Optional imports for plotting – only required if --plot is used
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

def check_array(name: str, obj: zarr.Array, full: bool = False):
    """Perform sanity checks on a Zarr array in chunks to prevent OOM."""
    issues = []
    
    chunk_size = 10000
    n_samples = obj.shape[0] if len(obj.shape) > 0 else 1
    
    n_total = 0
    sum_val = 0.0
    sum_sq_val = 0.0
    min_val = float('inf')
    max_val = float('-inf')
    has_nan = False
    has_inf = False
    
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        if len(obj.shape) > 0:
            arr_chunk = obj[start:end]
        else:
            arr_chunk = np.array([obj[...]])
            
        if np.isnan(arr_chunk).any():
            has_nan = True
        if np.isinf(arr_chunk).any():
            has_inf = True
            
        valid = arr_chunk[~np.isnan(arr_chunk) & ~np.isinf(arr_chunk)]
        if valid.size > 0:
            sum_val += float(np.sum(valid, dtype=np.float64))
            sum_sq_val += float(np.sum(valid**2, dtype=np.float64))
            min_val = min(min_val, float(np.min(valid)))
            max_val = max(max_val, float(np.max(valid)))
            n_total += valid.size

    if has_nan:
        issues.append("Contains NaN values")
    if has_inf:
        issues.append("Contains infinite values")

    mean_val = sum_val / n_total if n_total > 0 else float('nan')
    var_val = (sum_sq_val / n_total) - (mean_val**2) if n_total > 0 else 0
    std_val = float(np.sqrt(max(0, var_val)))

    stats = {
        "shape": obj.shape,
        "dtype": str(obj.dtype),
        "min": min_val if min_val != float('inf') else float('nan'),
        "max": max_val if max_val != float('-inf') else float('nan'),
        "mean": mean_val,
        "std": std_val,
    }

    extra = {}
    return {"issues": issues, "stats": stats, "extra": extra}

def generate_plots(name: str, extra: dict, out_dir: Path):
    if plt is None:
        return []
    written = []
    if "histogram" in extra:
        hist = extra["histogram"]
        plt.figure(figsize=(6, 4))
        plt.bar(hist["bins"][:-1], hist["counts"], width=np.diff(hist["bins"]), align="edge")
        plt.title(f"Histogram of {name}")
        plt.xlabel("Value")
        plt.ylabel("Count")
        hist_path = out_dir / f"{name}_hist.png"
        plt.savefig(hist_path)
        plt.close()
        written.append(hist_path)
    if "correlation" in extra:
        corr = np.array(extra["correlation"])
        if corr.ndim == 2:
            plt.figure(figsize=(6, 5))
            im = plt.imshow(corr, cmap="viridis", aspect="auto")
            plt.title(f"Correlation matrix of {name}")
            plt.colorbar(im)
            corr_path = out_dir / f"{name}_corr.png"
            plt.savefig(corr_path)
            plt.close()
            written.append(corr_path)
    return written

def main():
    parser = argparse.ArgumentParser(
        description="Data quality check for Zarr cache used by the Forex scaling model.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--cache-path", required=True, type=Path,
                        help="Path to the root of the Zarr store (directory).")
    parser.add_argument("--full", action="store_true",
                        help="Compute full statistics (histograms, correlation).")
    parser.add_argument("--plot", action="store_true",
                        help="Generate PNG plots for histograms / correlation matrices.")
    parser.add_argument("--output-dir", type=Path,
                        default=Path.cwd() / "artifacts",
                        help="Directory where the markdown report and any plot images will be saved.")
    args = parser.parse_args()

    if not args.cache_path.is_dir():
        sys.exit(f"Error: cache path {args.cache_path} does not exist or is not a directory.")
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    store = zarr.open(store=args.cache_path, mode="r")
    report = ["# Data Quality Report", f"Cache path: `{args.cache_path}`", ""]
    plot_paths = []
    for name in store.keys():
        obj = store[name]
        if isinstance(obj, zarr.Array):
            res = check_array(name, obj, full=args.full)
            report.append(f"## Dataset: `{name}`")
            report.append(f"**Shape:** {res['stats']['shape']}")
            report.append(f"**Dtype:** {res['stats']['dtype']}")
            report.append("**Stats:**")
            report.extend([f"- {k}: {v}" for k, v in res['stats'].items()])
            if res['issues']:
                report.append("**Issues Detected:**")
                report.extend([f"- {i}" for i in res['issues']])
            else:
                report.append("**Issues Detected:** None")
            if args.plot and res['extra']:
                plots = generate_plots(name, res['extra'], out_dir)
                for p in plots:
                    rel = p.relative_to(out_dir)
                    report.append(f"![{p.name}]({rel})")
                    plot_paths.append(p)
            report.append("---")
        else:
            report.append(f"## Group: `{name}` (skipped)")
            report.append("---")
    report_path = out_dir / "data_quality_report.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(f"Report written to {report_path}")
    if plot_paths:
        print("Plots generated:")
        for p in plot_paths:
            print(p)

if __name__ == "__main__":
    main()
