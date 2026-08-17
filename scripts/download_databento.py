import argparse
from pathlib import Path

import databento as db
import pandas as pd
from dotenv import load_dotenv

CME_MAP = {"EURUSD": "6E", "GBPUSD": "6B", "USDJPY": "6J", "AUDUSD": "6A", "USDCAD": "6C"}


def main():
    parser = argparse.ArgumentParser(description="Download Databento CME L2 Order Book data")
    parser.add_argument("--pair", type=str, required=True, help="Spot pair e.g. EURUSD")
    parser.add_argument("--start", type=str, required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--schema", type=str, default="mbp-10", help="Data schema (mbp-10 or trades)")
    parser.add_argument("--dry-run", action="store_true", help="Only estimate cost, do not download")
    parser.add_argument("--budget", type=float, default=None, help="Maximum total cost in USD (stop if exceeded)")
    parser.add_argument(
        "--max-size-gb", type=float, default=None, help="Maximum total download size in GB (stop if exceeded)"
    )
    parser.add_argument("--skip-cost-check", action="store_true", help="Skip cost estimate API call (avoids hangs)")
    parser.add_argument(
        "--api-key", type=str, required=True, help="Databento API key (pass on CLI; not stored in .env)"
    )
    args = parser.parse_args()

    load_dotenv()
    api_key = args.api_key.strip()
    if not api_key:
        print("ERROR: --api-key is empty.")
        return

    sym = CME_MAP.get(args.pair.upper())
    if not sym:
        print(f"ERROR: No CME mapping found for {args.pair}")
        return

    client = db.Historical(api_key)
    dataset = "GLBX.MDP3"
    symbols = [f"{sym}.c.0"]

    print("--- Databento Request Details ---")
    print(f"Dataset:  {dataset}")
    print(f"Symbols:  {symbols}")
    print(f"Schema:   {args.schema}")
    print(f"Date:     {args.start} -> {args.end}")

    # Calculate Total Cost
    total_estimated_cost = 0.0
    if not getattr(args, "skip_cost_check", False):
        try:
            import signal

            def _timeout_handler(signum, frame):
                raise TimeoutError("Cost estimate timed out")

            try:
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(10)
            except (AttributeError, OSError):
                pass  # Windows doesn't support SIGALRM
            cost = client.metadata.get_cost(
                dataset=dataset,
                symbols=symbols,
                schema=args.schema,
                start=args.start,
                end=args.end,
                stype_in="continuous",
            )
            try:
                signal.alarm(0)
            except (AttributeError, OSError):
                pass
            print(f"[ESTIMATE] Expected Total Cost: ${cost:.4f}")
            total_estimated_cost = cost
        except Exception as e:
            print(f"[WARN] Cost estimate failed ({e}) - proceeding without estimate. Budget cap still applies.")
            total_estimated_cost = 0.0
    else:
        print("[INFO] Skipping cost estimate (--skip-cost-check). Budget cap still applies.")
    total_cost_accum = 0.0
    total_size_bytes = 0

    if args.dry_run:
        print("[DRY RUN] Exiting.")
        return

    if args.budget is not None:
        print(f"[BUDGET] Hard cap set at ${args.budget:.2f} USD")

    # Generate week-by-week date pairs to avoid 5GB streaming limit
    start_dt = pd.to_datetime(args.start)
    end_dt = pd.to_datetime(args.end)

    # Create weekly periods
    periods = pd.date_range(start=start_dt, end=end_dt, freq="7D")
    if len(periods) == 0 or periods[-1] < end_dt:
        periods = periods.append(pd.DatetimeIndex([end_dt]))

    out_dir = Path("data/raw/databento")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compute simple per-chunk cost estimate (divide total by number of chunks)
    per_chunk_cost = total_estimated_cost / max(len(periods) - 1, 1)

    print("\n[DOWNLOAD] Starting Week-by-Week Chunked Download...")
    for idx in range(len(periods) - 1):
        chunk_start = periods[idx]
        chunk_end = periods[idx + 1]

        str_start = chunk_start.strftime("%Y-%m-%d")
        str_end = chunk_end.strftime("%Y-%m-%d")

        out_file = out_dir / f"{args.pair}_{args.schema}_{str_start}_{str_end}.parquet"

        # Skip if already downloaded
        if out_file.exists():
            print(f"  -> Skipping {str_start} to {str_end} (Already exists)")
            # Account for existing file size toward limits
            if args.max_size_gb is not None:
                total_size_bytes += out_file.stat().st_size
            continue

        print(f"  -> Downloading {str_start} to {str_end} (Chunk {idx + 1}/{len(periods) - 1})...")
        try:
            import threading

            result = [None]
            error = [None]

            def _fetch():
                try:
                    result[0] = client.timeseries.get_range(  # noqa: B023
                        dataset=dataset,
                        symbols=symbols,
                        schema=args.schema,
                        start=str_start,  # noqa: B023
                        end=str_end,  # noqa: B023
                        stype_in="continuous",
                    )
                except Exception as e:
                    error[0] = e  # noqa: B023

            t = threading.Thread(target=_fetch, daemon=True)
            t.start()
            t.join(timeout=300)  # 5-minute timeout per chunk

            if t.is_alive():
                print("     [TIMEOUT] Chunk timed out after 5 min, skipping.")
                continue
            if error[0] is not None:
                raise error[0]
            data = result[0]
            data.to_parquet(out_file)
            print("     [SUCCESS] Saved chunk.")
            # Update accumulators
            total_cost_accum += per_chunk_cost
            total_size_bytes += out_file.stat().st_size
        except Exception as e:
            print(f"     [ERROR] Failed chunk: {e}")
            continue

        # Enforce budget limits if provided
        if args.budget is not None and total_cost_accum >= args.budget:
            print(
                f"[LIMIT] Budget of ${args.budget:.2f} reached (estimated cost ${total_cost_accum:.2f}). Stopping further downloads."
            )
            break
        if args.max_size_gb is not None and (total_size_bytes / (1024**3)) >= args.max_size_gb:
            print(
                f"[LIMIT] Size limit of {args.max_size_gb:.2f} GB reached (downloaded {(total_size_bytes / 1024**3):.2f} GB). Stopping further downloads."
            )
            break

    print("\n[COMPLETE] Download session finished.")
    if args.budget is not None:
        print(f"Total estimated cost this session: ${total_cost_accum:.4f}")
    if args.max_size_gb is not None:
        print(f"Total downloaded size this session: {(total_size_bytes / 1024**3):.2f} GB")


if __name__ == "__main__":
    main()
