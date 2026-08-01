# Folder Guide: `scripts`

## Purpose
Operational scripts for data, training, export, audit, and maintenance tasks.

## Files
| File | Role | Details |
| --- | --- | --- |
| `audit_and_repair_data.py` | Python module | Functions: audit_and_repair |
| `audit_training_cache.py` | Training module/script | Functions: audit_cache, main |
| `augment_news.py` | Python module | Functions: generate_variations, augment_dataset, main |
| `backtest_mamba_full.py` | Backtesting module/script | Backward-compatible Mamba backtest entrypoint. |
| `backtest_model.py` | Backtesting module/script | Functions: log, parse_args, run_backtest |
| `backtest_true_walk_forward.py` | Backtesting module/script | Functions: log, parse_args, get_fold_checkpoint, run_true_walkforward |
| `best_date.py` | Python module | best_date.py ============ Find the best representative trading day per year by testing several candidate dates and scoring them on: - completeness : how many of the 9 pairs have ticks (0-9) - consistency : no pair is mor... Functions: third_tuesday, score_day, run |
| `check_apis.py` | Python module | scripts/check_apis.py ===================== Validates every API key and service endpoint configured in .env. Functions: check_eodhd, check_fred, check_alpha_vantage, check_wandb, check_discord, check_telegram, check_ollama, check_mlflow, check_stooq, check_timescale... |
| `check_project_structure.py` | Python module | Check for files that should not live at the repository root. Functions: main |
| `compare_models.py` | Python module | Functions: run_evaluation |
| `continuous_finetune.py` | Python module | scripts/continuous_finetune.py ============================== Automates the 'Rolling Window' Continuous Fine-Tuning pipeline. Functions: get_python_exe, write_temp_finetune_config, run_pipeline, main |
| `count_ticks.py` | Python module | count_ticks.py ============== Estimates total tick count across all pairs for 2010-2025. Functions: second_tuesday_of_january, run |
| `data_quality_check.py` | Python module | Functions: check_array, generate_plots, main |
| `discord_ollama_bot.py` | Python module | Discord <-> Ollama training Q&A bot. Functions: redact, latest_file, tail_text, parse_latest_jsonl, summarize_metrics, list_checkpoints, config_excerpt, build_context, ask_ollama, discord_post... |
| `download_2008_news.py` | Data operation script | Functions: generate_synthetic_2008 |
| `download_all.py` | Data operation script | scripts/download_all.py ======================= One-command download for all offline training data. Functions: parse_args, main |
| `download_cot.py` | Data operation script | Functions: download_cot_data, parse_args |
| `download_data.py` | Data operation script | scripts/download_data.py ======================== Standalone bulk downloader / ingester for: 1) Dukascopy tick data (data/raw/dukascopy/<PAIR>/...) 2) Cross-asset panel (data/processed/cross_asset/...) 3) Myfxbook daily... Functions: download_ticks, download_ticks_yearly, download_cross_asset, download_eodhd_forex, download_eodhd_cross_asset, ingest_myfxbook, auto_redownload_missing_data, main |
| `download_databento.py` | Data operation script | Functions: main |
| `download_fnspid.py` | Data operation script | Functions: main |
| `download_gdelt2_bulk.py` | Data operation script | Functions: generate_urls, download_and_parse_gdelt, main |
| `download_hf_news.py` | Data operation script | Functions: main |
| `download_historical_news.py` | Data operation script | Download forex-focused historical news/calendar data for offline training. Functions: fetch_official_feeds, write_rows, append_failures, resolve_failures, fetch_gdelt_pair, fetch_eodhd_news, fetch_eodhd_calendar, load_failures_to_retry, parse_args, run_post_download_sentiment... |
| `download_yearly.py` | Data operation script | scripts/download_yearly.py ========================== CLI for pair-by-pair, year-by-year Dukascopy downloads with verification. Functions: main |
| `drift_report.py` | Python module | Generate an input-distribution drift report from cached training features. Functions: parse_args, main |
| `dummy_zmq_sender.py` | Python module | Functions: main |
| `dump_api.py` | Python module | Functions: main |
| `export_onnx.py` | Python module | Functions: export_to_onnx |
| `export_parity_data.py` | Python module | Functions: generate_parity_data |
| `find_api.py` | Python module | Functions: main |
| `generate_synthetic_news_fill.py` | Python module | Generate reviewable synthetic historical-news rows for missing pair coverage. Functions: pair_currencies, parse_month, month_days, stable_pick, event_timestamp, headline_filter_sql, template_rows_for_month, rows_from_real_news, detect_missing_months, parse_args... |
| `merge_datasets.py` | Data operation script | Functions: main |
| `merge_datasets_duckdb.py` | Data operation script | Functions: main |
| `merge_massive_datasets.py` | Data operation script | Functions: merge_massive_datasets, parse_args |
| `normalize_historical_news.py` | Python module | Normalize historical_news_combined.parquet for the training news loader. Functions: normalize_news, main |
| `optuna_tune.py` | Python module | Functions: parse_args, objective, main |
| `parse_test.py` | Test module | Functions: parse |
| `promote_best_fold.py` | Python module | scripts/promote_best_fold.py =========================== Standalone utility to scan cross-validation folds and promote the best one to the primary checkpoint filename, applying a stability penalty for overfitting. Functions: promote_best_fold, main |
| `report_databento_usage.py` | Python module | Functions: get_file_size, human_readable_size, main |
| `run_feature_engineering.py` | Python module | Functions: load_config, run_pipeline |
| `run_pipeline.py` | Python module | scripts/run_pipeline.py ======================= Top-level pipeline: download data, train, or both. Functions: main |
| `sanitize_cached_labels.py` | Python module | Functions: sanitize_cache, main |
| `score_historical_news_sentiment.py` | Python module | Score historical news headlines and write sentiment_score into the CSV. Functions: print_stats, score_historical_news, parse_args, main |
| `scrape_forexfactory.py` | Data operation script | Scrape the ForexFactory economic calendar into data/raw/eco_calendar/events.csv. Functions: parse_week_html, fetch_forexfactory_week, parse_args, main |
| `scrape_forexlive.py` | Data operation script | Functions: main |
| `scrape_historical_news.py` | Data operation script | Stage resumable historical-news scraping for model training. Classes: Window Functions: parse_date, iter_windows, build_download_command, shell_join, parse_args, main |
| `test_playwright.py` | Test module | Functions: test_fetch |
| `train.py` | Training module/script | scripts/train.py ================ Simple entry point for GPU training — wraps training/train_gpu.py. Functions: check_data_paths, warn_optional_data, print_run_summary, parse_args, main |
| `train_diverse_recipes.py` | Training module/script | Functions: main |
| `train_ensemble_meta.py` | Training module/script | Train only the EnsembleMeta learner from existing base checkpoints. Functions: log, parse_args, checkpoint_state_dict, resolve_checkpoint, load_training_config, make_model_args, load_base_model, infer_cache_shape, main |
| `train_rl.py` | Training module/script | scripts/train_rl.py =================== Trains a Reinforcement Learning execution policy on top of a frozen supervised model's signal. Functions: parse_args, extract_signals, main |
| `update_readme_docs.py` | Python module | Functions: format_changelog_paragraph |
| `verify_data.py` | Python module | scripts/verify_data.py ====================== Data quality verification and auto-repair for the Dukascopy tick cache. Functions: scan_duplicates, scan_missing, analyse_gaps, redownload_hours, run_verification, verify_news_dataset, main |
| `verify_onnx_export.py` | Python module | Functions: generate_test_data |

_Generated by Codex to document folder contents._
