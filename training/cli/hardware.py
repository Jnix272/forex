"""Hardware profile application for training args."""

from __future__ import annotations

from config.settings import HARDWARE_PROFILES, PATHS


def apply_hardware_profile(args) -> None:
    """Override training paths and loader settings for a known local GPU/RAM combo."""
    name = getattr(args, "hardware_profile", None)
    if not name:
        return
    prof = HARDWARE_PROFILES.get(name)
    if not prof:
        return
    for k, v in prof.items():
        if k == "local_project_paths":
            continue
        setattr(args, k, v)
    if prof.get("local_project_paths"):
        args.checkpoint_dir = PATHS["checkpoints"]
        args.data_cache = PATHS["data_processed"]
    print(
        f"[Hardware] profile={name} | batch={args.batch_size} | workers={args.num_workers} | "
        f"chunk={args.chunk_size} | prefetch={args.prefetch_factor}"
    )
    print(f"             checkpoint_dir={args.checkpoint_dir} | data_cache={args.data_cache}")
