#!/usr/bin/env bash
# =============================================================================
# setup_ubuntu.sh — Forex Scaling Model  |  Ubuntu GPU Training Setup
# =============================================================================
# Supports: Ubuntu 20.04 LTS · 22.04 LTS · 24.04 LTS
# GPU:      Any NVIDIA GPU with CUDA 11.8+ drivers (RTX 20/30/40-series, A/H-series)
#
# What this script does
# ─────────────────────
#   1. Checks Ubuntu version, NVIDIA driver, CUDA toolkit
#   2. Installs system packages (build-essential, libpq-dev, tmux, nvtop …)
#   3. Creates a Python 3.11 virtual environment at .venv/
#   4. Auto-detects your CUDA version and installs the matching PyTorch build
#   5. Installs requirements_gpu.txt
#   6. Creates .env from the template (skips if already filled in)
#   7. Creates data/, checkpoints/, logs/, exports/ directories
#   8. Applies Linux DataLoader system-limit tuning (ulimit, vm.max_map_count)
#   9. Runs the pre-flight smoke test (all 6 models, CPU)
#  10. Prints the next-step cheat-sheet
#
# Usage
# ─────
#   chmod +x setup_ubuntu.sh
#   ./setup_ubuntu.sh                   # full setup
#   ./setup_ubuntu.sh --skip-smoke      # skip pre-flight model test at the end
#   ./setup_ubuntu.sh --cpu-only        # skip PyTorch GPU build, CPU-only install
#
# After setup
# ──────────
#   source .venv/bin/activate
#   python training/smoke_test.py --gpu --amp       # GPU pre-flight check
#   python training/train_gpu.py --config config/run_ubuntu.yaml
# =============================================================================

set -euo pipefail

# ── Colour helpers ─────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
ok()      { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*" >&2; }
section() { echo -e "\n${BOLD}${CYAN}══ $* ══${RESET}"; }

# ── Flags ─────────────────────────────────────────────────────────────────────
SKIP_SMOKE=false
CPU_ONLY=false

for arg in "$@"; do
  case "$arg" in
    --skip-smoke) SKIP_SMOKE=true ;;
    --cpu-only)   CPU_ONLY=true ;;
    --help|-h)
      grep '^#' "$0" | head -40 | sed 's/^# \?//'
      exit 0
      ;;
  esac
done

# ── Script location (project root) ────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
info "Project root: $SCRIPT_DIR"

# ── 1. System checks ──────────────────────────────────────────────────────────
section "1. System Checks"

# Ubuntu version
if [[ -f /etc/os-release ]]; then
  . /etc/os-release
  UBUNTU_VERSION="${VERSION_ID:-unknown}"
  info "OS: $PRETTY_NAME"
  if [[ "$ID" != "ubuntu" ]]; then
    warn "This script is designed for Ubuntu. Proceeding anyway — some steps may fail."
  fi
  if [[ "$UBUNTU_VERSION" != "20.04" && "$UBUNTU_VERSION" != "22.04" && "$UBUNTU_VERSION" != "24.04" ]]; then
    warn "Tested on Ubuntu 20.04/22.04/24.04. Your version ($UBUNTU_VERSION) may work but is not guaranteed."
  fi
else
  warn "Cannot detect OS. Proceeding anyway."
fi

# CPU cores / RAM
CPU_CORES=$(nproc)
RAM_GB=$(awk '/MemTotal/ {printf "%.0f", $2/1024/1024}' /proc/meminfo)
info "CPU cores: $CPU_CORES  |  RAM: ${RAM_GB} GB"

# NVIDIA driver
if command -v nvidia-smi &>/dev/null; then
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
  DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
  ok "NVIDIA GPU detected: $GPU_NAME  |  driver: $DRIVER_VER"
else
  if [[ "$CPU_ONLY" == "true" ]]; then
    warn "nvidia-smi not found — installing CPU-only PyTorch (--cpu-only flag set)."
  else
    error "nvidia-smi not found. Install NVIDIA drivers first:"
    echo "  ubuntu-drivers autoinstall   # or: sudo apt install nvidia-driver-545"
    echo "  sudo reboot"
    echo ""
    echo "  Then re-run this script, or use --cpu-only for a CPU-only install."
    exit 1
  fi
fi

# CUDA version detection
CUDA_VERSION=""
if command -v nvcc &>/dev/null; then
  CUDA_VERSION=$(nvcc --version | grep -oP "release \K[0-9]+\.[0-9]+" | head -1)
  ok "CUDA toolkit: $CUDA_VERSION"
elif [[ -f /usr/local/cuda/version.txt ]]; then
  CUDA_VERSION=$(cat /usr/local/cuda/version.txt | grep -oP "[0-9]+\.[0-9]+" | head -1)
  ok "CUDA version (from file): $CUDA_VERSION"
elif nvidia-smi &>/dev/null; then
  CUDA_VERSION=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" | head -1)
  warn "nvcc not in PATH — using driver-reported CUDA $CUDA_VERSION"
fi

# ── Map CUDA version to PyTorch wheel index URL ────────────────────────────────
if [[ "$CPU_ONLY" == "true" ]]; then
  TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
  TORCH_DESC="CPU-only"
else
  CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
  CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)
  if   [[ "$CUDA_MAJOR" -ge 13 ]] || [[ "$CUDA_MAJOR" -eq 12 && "$CUDA_MINOR" -ge 4 ]]; then
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"
    TORCH_DESC="CUDA 12.4"
  elif [[ "$CUDA_MAJOR" -eq 12 && "$CUDA_MINOR" -ge 1 ]]; then
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"
    TORCH_DESC="CUDA 12.1"
  elif [[ "$CUDA_MAJOR" -eq 12 && "$CUDA_MINOR" -eq 0 ]] || [[ "$CUDA_MAJOR" -eq 11 && "$CUDA_MINOR" -ge 8 ]]; then
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu118"
    TORCH_DESC="CUDA 11.8"
  else
    warn "CUDA $CUDA_VERSION is older than 11.8. Installing CPU PyTorch as fallback."
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
    TORCH_DESC="CPU-only (CUDA too old)"
  fi
fi
info "PyTorch build: $TORCH_DESC  ($TORCH_INDEX_URL)"

# ── 2. System packages ────────────────────────────────────────────────────────
section "2. System Packages"

sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
  build-essential \
  python3 \
  python3-venv \
  python3-dev \
  python3-pip \
  git \
  curl \
  wget \
  htop \
  nvtop \
  tmux \
  vim \
  unzip \
  libpq-dev \
  libssl-dev \
  libffi-dev \
  pkg-config \
  2>/dev/null || warn "Some system packages failed (non-fatal)."

ok "System packages installed."

# ── 3. Python virtual environment ─────────────────────────────────────────────
section "3. Python Virtual Environment"

# Pick the best available Python. 3.12 matches the run_ubuntu.yaml profile;
# fall back through 3.11/3.13 to whatever `python3` is (e.g. 3.14).
PY_BIN=""
for pv in python3.12 python3.11 python3.13 python3; do
  if command -v "$pv" &>/dev/null; then
    PY_BIN="$pv"
    break
  fi
done
if [[ -z "$PY_BIN" ]]; then
  error "No Python 3 found on PATH."
  exit 1
fi
PY_VER="$("$PY_BIN" --version)"
info "Using interpreter: $PY_BIN ($PY_VER)"

VENV_DIR="$SCRIPT_DIR/.venv"
if [[ -d "$VENV_DIR" ]]; then
  info "Virtual environment already exists at $VENV_DIR — reusing."
else
  "$PY_BIN" -m venv --copies "$VENV_DIR"
  ok "Created venv at $VENV_DIR using $PY_BIN"
fi

# Activate
source "$VENV_DIR/bin/activate"
python --version
pip install --upgrade pip setuptools wheel -q
ok "pip upgraded."

# ── 4. PyTorch install ────────────────────────────────────────────────────────
section "4. PyTorch ($TORCH_DESC)"

# Check if torch is already installed with the right CUDA variant
if python -c "import torch; assert torch.cuda.is_available() or '${CPU_ONLY}' == 'true'" 2>/dev/null; then
  TORCH_VER=$(python -c "import torch; print(torch.__version__)")
  ok "PyTorch $TORCH_VER already installed and CUDA is available — skipping reinstall."
else
  info "Installing PyTorch 2.4.x from $TORCH_INDEX_URL ..."
  pip install torch torchvision torchaudio \
    --index-url "$TORCH_INDEX_URL" \
    --quiet
  ok "PyTorch installed."
fi

# Verify
TORCH_VER=$(python -c "import torch; print(torch.__version__)")
CUDA_AVAIL=$(python -c "import torch; print(torch.cuda.is_available())")
info "Installed: torch $TORCH_VER  |  CUDA available: $CUDA_AVAIL"
if [[ "$CUDA_AVAIL" == "False" && "$CPU_ONLY" == "false" ]]; then
  warn "CUDA not available inside Python. Check driver/toolkit compatibility."
fi

# ── 5. Project requirements ───────────────────────────────────────────────────
section "5. Python Requirements"

if [[ -f "$SCRIPT_DIR/requirements_gpu.txt" ]]; then
  info "Installing requirements_gpu.txt ..."
  pip install -r "$SCRIPT_DIR/requirements_gpu.txt" \
    --quiet --no-deps 2>/dev/null || \
  pip install -r "$SCRIPT_DIR/requirements_gpu.txt" --quiet
  ok "requirements_gpu.txt installed."
else
  warn "requirements_gpu.txt not found — installing requirements.txt instead."
  pip install -r "$SCRIPT_DIR/requirements.txt" --quiet
fi

# ── 6. .env setup ────────────────────────────────────────────────────────────
section "6. Environment Variables (.env)"

ENV_FILE="$SCRIPT_DIR/.env"
if [[ -f "$ENV_FILE" ]]; then
  info ".env file found at $ENV_FILE"
  # Check if the key fields have been filled in
  if grep -q "WANDB_API_KEY=$" "$ENV_FILE" 2>/dev/null || \
     grep -q "^WANDB_API_KEY=$" "$ENV_FILE" 2>/dev/null; then
    echo ""
    warn "Your .env has empty values. Fill these in for full tracking:"
    echo ""
    echo "  nano $ENV_FILE"
    echo ""
    echo "  Key fields to fill:"
    echo "    WANDB_API_KEY       → https://wandb.ai/authorize"
    echo "    DISCORD_WEBHOOK_URL → Discord channel → Integrations → Webhooks"
    echo "    FRED_API_KEY        → https://fred.stlouisfed.org/docs/api/api_key.html"
    echo "    OANDA_API_TOKEN     → https://www.oanda.com/demo-account/ (optional)"
    echo ""
  else
    ok ".env already configured."
  fi
else
  warn ".env not found. Creating from template ..."
  cat > "$ENV_FILE" << 'ENVTEMPLATE'
# =============================================================================
# Forex Scaling Model — Environment Variables  (Ubuntu)
# =============================================================================
# Fill in your API keys and credentials below.
# This file is loaded automatically at startup — never commit it to git.
# =============================================================================

# Weights & Biases — training metrics, model comparison
# Get key: https://wandb.ai/authorize
WANDB_API_KEY=

# MLflow tracking server (blank = http://localhost:5000)
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT=forex-scaling-model
MLFLOW_FALLBACK_DIR=

# Discord webhook for live alerts (circuit breaker, model promotion, etc.)
# Create: Discord server → Edit Channel → Integrations → Webhooks → New Webhook → Copy URL
DISCORD_WEBHOOK_URL=

# FRED (Federal Reserve) — macro/yield spread features (free)
# Get key: https://fred.stlouisfed.org/docs/api/api_key.html
FRED_API_KEY=

# Alpha Vantage — economic calendar (free tier: 25 req/day)
AV_API_KEY=

# OANDA v20 REST API (optional — for live/paper trading)
OANDA_API_TOKEN=
OANDA_ACCOUNT_ID=

# Cross-asset data
CROSS_ASSET_SOURCE=auto
CROSS_ASSET_CACHE_DIR=

# Sentiment
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=mistral
SENTIMENT_CACHE_DIR=

# TimescaleDB (optional — tick storage for live trading)
TIMESCALE_HOST=localhost
TIMESCALE_PORT=5432
TIMESCALE_DB=forex_ticks
TIMESCALE_USER=forex_user
TIMESCALE_PASSWORD=

# Kafka (optional — live tick stream)
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_TICK_TOPIC=forex_ticks

# Prometheus
PROMETHEUS_PORT=8000

# Telegram alerts (alternative to Discord)
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=

# Path overrides (blank = project-relative defaults)
CHECKPOINTS_DIR=
ECO_CACHE_DIR=
TRANSFORMERS_CACHE=
ENVTEMPLATE
  ok ".env template created at $ENV_FILE"
  warn "Edit $ENV_FILE and add your API keys before training."
fi

# ── 7. Directories ────────────────────────────────────────────────────────────
section "7. Project Directories"

for dir in checkpoints logs exports data/processed data/raw; do
  mkdir -p "$SCRIPT_DIR/$dir"
  ok "  $dir/"
done

# ── 8. Linux system limits (DataLoader tuning) ────────────────────────────────
section "8. Linux System Limits"

# Increase open file limit for DataLoader workers
ULIMIT_TARGET=65536
CURRENT_ULIMIT=$(ulimit -n)
if [[ "$CURRENT_ULIMIT" -lt "$ULIMIT_TARGET" ]]; then
  ulimit -n "$ULIMIT_TARGET" 2>/dev/null && \
    ok "ulimit -n set to $ULIMIT_TARGET (was $CURRENT_ULIMIT)" || \
    warn "Could not set ulimit -n. Add to ~/.bashrc:  ulimit -n $ULIMIT_TARGET"
else
  ok "ulimit -n already $CURRENT_ULIMIT (≥ $ULIMIT_TARGET)"
fi

# Persistent ulimit via /etc/security/limits.conf
LIMITS_LINE="* soft nofile $ULIMIT_TARGET"
if ! grep -q "nofile $ULIMIT_TARGET" /etc/security/limits.conf 2>/dev/null; then
  echo "$LIMITS_LINE" | sudo tee -a /etc/security/limits.conf > /dev/null 2>&1 && \
    ok "Persistent ulimit written to /etc/security/limits.conf" || \
    warn "Could not write to /etc/security/limits.conf (non-fatal)."
fi

# vm.max_map_count — required for large memory-mapped datasets (Zarr / NPY)
VM_TARGET=262144
CURRENT_VM=$(cat /proc/sys/vm/max_map_count 2>/dev/null || echo 0)
if [[ "$CURRENT_VM" -lt "$VM_TARGET" ]]; then
  sudo sysctl -w vm.max_map_count="$VM_TARGET" > /dev/null 2>&1 && \
    ok "vm.max_map_count set to $VM_TARGET" || \
    warn "Could not set vm.max_map_count (non-fatal). Manually: sudo sysctl -w vm.max_map_count=$VM_TARGET"
  # Make it persist across reboots
  echo "vm.max_map_count=$VM_TARGET" | sudo tee -a /etc/sysctl.conf > /dev/null 2>&1 || true
else
  ok "vm.max_map_count already $CURRENT_VM (≥ $VM_TARGET)"
fi

# transparent_hugepage (disable for memmap workloads — reduces latency spikes)
if [[ -f /sys/kernel/mm/transparent_hugepage/enabled ]]; then
  echo madvise | sudo tee /sys/kernel/mm/transparent_hugepage/enabled > /dev/null 2>&1 && \
    ok "transparent_hugepage set to madvise" || \
    warn "Could not set transparent_hugepage (non-fatal)."
fi

# ── 9. GPU VRAM check & profile recommendation ────────────────────────────────
section "9. Hardware Profile Recommendation"

if command -v nvidia-smi &>/dev/null && [[ "$CPU_ONLY" == "false" ]]; then
  VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)
  info "GPU VRAM: ${VRAM_MB} MB"

  if   [[ "$VRAM_MB" -ge 20000 ]]; then
    RECOMMENDED_PROFILE="ubuntu_rtx4090_desktop"
  elif [[ "$VRAM_MB" -ge 12000 ]]; then
    RECOMMENDED_PROFILE="ubuntu_rtx4070_laptop"
  else
    RECOMMENDED_PROFILE="ubuntu_rtx_laptop"
  fi

  echo ""
  ok "Recommended hardware profile for your GPU: [${BOLD}$RECOMMENDED_PROFILE${RESET}]"
  echo ""
  echo "  Set it in config/run_ubuntu.yaml:"
  echo "    hardware:"
  echo "      profile: \"$RECOMMENDED_PROFILE\""
  echo ""
  # Auto-patch run_ubuntu.yaml if it exists
  if [[ -f "$SCRIPT_DIR/config/run_ubuntu.yaml" ]]; then
    sed -i "s/profile: \"ubuntu_rtx4090_desktop\"/profile: \"$RECOMMENDED_PROFILE\"/" \
      "$SCRIPT_DIR/config/run_ubuntu.yaml" 2>/dev/null && \
      ok "Auto-patched config/run_ubuntu.yaml → profile: $RECOMMENDED_PROFILE"
  fi
fi

# ── 10. Pre-flight smoke test ─────────────────────────────────────────────────
if [[ "$SKIP_SMOKE" == "true" ]]; then
  warn "Skipping pre-flight smoke test (--skip-smoke flag)."
else
  section "10. Pre-flight Model Smoke Test (CPU)"
  info "Testing all 6 model architectures with a tiny synthetic dataset ..."
  echo ""
  if python training/smoke_test.py; then
    ok "Smoke test passed — all models build and train correctly."
  else
    error "Smoke test failed. Fix the errors above before starting full training."
    echo ""
    echo "  To debug a specific model:"
    echo "    python training/smoke_test.py --models haelt --verbose"
    exit 1
  fi
fi

# ── Done — cheat-sheet ────────────────────────────────────────────────────────
section "Setup Complete"

cat << CHEATSHEET

${BOLD}${GREEN}✓ Ubuntu setup complete.${RESET}

${BOLD}Activate the environment${RESET}
  source .venv/bin/activate

${BOLD}Edit your API keys${RESET}
  nano .env

${BOLD}GPU pre-flight check (recommended before full training)${RESET}
  python training/smoke_test.py --gpu --amp

${BOLD}Start MLflow UI (in a separate terminal)${RESET}
  source .venv/bin/activate && mlflow ui --port 5000

${BOLD}Run full GPU training${RESET}
  python training/train_gpu.py --config config/run_ubuntu.yaml

${BOLD}Training with tmux (safe against SSH disconnect)${RESET}
  tmux new -s train
  source .venv/bin/activate
  python training/train_gpu.py --config config/run_ubuntu.yaml
  # Detach: Ctrl+B then D   |   Reattach: tmux attach -t train

${BOLD}Monitor GPU${RESET}
  nvtop                                # GPU utilisation + temperature
  watch -n1 nvidia-smi                 # VRAM + power usage

${BOLD}View training logs${RESET}
  tail -f logs/train_*.log             # structured training log
  cat  logs/train_*_summary.txt        # session summary after training

${BOLD}TensorBoard${RESET}
  tensorboard --logdir logs/tensorboard --port 6006
  # then open http://localhost:6006

${BOLD}Hardware profiles available${RESET}
  ubuntu_rtx_laptop        # RTX 30/40 laptop, 8-16 GB VRAM
  ubuntu_rtx4070_laptop    # RTX 4070 Laptop, 12 GB VRAM
  ubuntu_rtx4090_desktop   # RTX 4090, 24 GB VRAM, full throughput
  rtx_4000_ada_cloud       # Cloud GPU pod (RunPod / Vast.ai)
  a5000_24gb               # NVIDIA A5000 workstation / cloud
  a40_48gb                 # NVIDIA A40 high-VRAM cloud

CHEATSHEET
