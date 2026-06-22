#!/usr/bin/env bash
# deploy.sh — One-shot bootstrap for trading-system-v2 on a fresh Ubuntu 22.04/24.04 VPS
#
# Run as root (or a user with passwordless sudo) immediately after first SSH login:
#
#   ssh root@<your-droplet-ip>
#   git clone https://github.com/bangalore-killua/trading-system-v2.git
#   cd trading-system-v2
#   chmod +x scripts/deploy.sh
#   ./scripts/deploy.sh
#
# What it does (in order):
#   1. Install system packages (Python 3.12, PostgreSQL 16, build tools for TA-Lib)
#   2. Create a non-root service user 'trading'
#   3. Set up PostgreSQL database + role
#   4. Clone repo into /srv/trading-system-v2 (or use current dir if already cloned)
#   5. Create Python venv + install requirements with uv
#   6. Build and install TA-Lib from source (clean path on Linux — no wheel games)
#   7. Install pandas-ta pre-release
#   8. Download FinBERT model + spaCy en_core_web_sm
#   9. Write .env from prompts
#  10. Run Alembic migration
#  11. Install systemd service + enable on boot
#  12. Start the service

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }
step() { echo -e "\n${CYAN}=== $* ===${NC}"; }

REPO_URL="https://github.com/bangalore-killua/trading-system-v2.git"
INSTALL_DIR="/srv/trading-system-v2"
SERVICE_USER="trading"
PYTHON="python3.12"
SERVICE_NAME="trading-system-v2"

step "1/12 System packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq \
    python3.12 python3.12-venv python3.12-dev \
    build-essential curl git wget \
    libpq-dev \
    ta-lib \
    postgresql postgresql-contrib \
    libta-lib-dev \
    software-properties-common
ok "System packages installed"

# uv — fast pip replacement
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi
ok "uv $(uv --version)"

step "2/12 Service user"
if ! id "$SERVICE_USER" &>/dev/null; then
    useradd --system --create-home --shell /bin/bash "$SERVICE_USER"
    ok "Created user '$SERVICE_USER'"
else
    ok "User '$SERVICE_USER' already exists"
fi

step "3/12 PostgreSQL database"
systemctl enable postgresql --now
DB_NAME="trading_system_v2"
DB_USER="killua_trading"

# Prompt for DB password (shown once, stored in .env later)
read -rsp "  Set password for PostgreSQL role '$DB_USER': " DB_PASS; echo
sudo -u postgres psql -tc "SELECT 1 FROM pg_roles WHERE rolname='$DB_USER'" | grep -q 1 \
    || sudo -u postgres psql -c "CREATE ROLE $DB_USER WITH LOGIN PASSWORD '$DB_PASS';"
sudo -u postgres psql -tc "SELECT 1 FROM pg_database WHERE datname='$DB_NAME'" | grep -q 1 \
    || sudo -u postgres createdb --owner="$DB_USER" "$DB_NAME"
ok "PostgreSQL: database '$DB_NAME' owned by '$DB_USER'"

step "4/12 Clone repository"
if [ -d "$INSTALL_DIR/.git" ]; then
    warn "Repo already at $INSTALL_DIR — pulling latest"
    git -C "$INSTALL_DIR" pull --ff-only
elif [ -f "$(pwd)/run_server.py" ]; then
    # Already inside the repo (user cloned manually)
    INSTALL_DIR="$(pwd)"
    warn "Using current directory as install dir: $INSTALL_DIR"
else
    git clone "$REPO_URL" "$INSTALL_DIR"
fi
chown -R "$SERVICE_USER":"$SERVICE_USER" "$INSTALL_DIR"
ok "Repo at $INSTALL_DIR"

step "5/12 Python venv + requirements"
cd "$INSTALL_DIR"
sudo -u "$SERVICE_USER" "$PYTHON" -m venv venv
VENV="$INSTALL_DIR/venv"
sudo -u "$SERVICE_USER" "$VENV/bin/pip" install --quiet uv
sudo -u "$SERVICE_USER" "$VENV/bin/uv" pip install -r requirements.txt
ok "Requirements installed"

step "6/12 TA-Lib Python binding"
# libta-lib-dev installed above gives us the C headers; just pip-install the binding
sudo -u "$SERVICE_USER" "$VENV/bin/uv" pip install ta-lib
ok "TA-Lib $(sudo -u $SERVICE_USER $VENV/bin/python -c 'import talib; print(talib.__version__)')"

step "7/12 pandas-ta pre-release"
sudo -u "$SERVICE_USER" "$VENV/bin/uv" pip install "pandas-ta==0.4.71b0" --prerelease=allow
ok "pandas-ta installed"

step "8/12 NLP models (FinBERT + spaCy)"
sudo -u "$SERVICE_USER" bash -c "
    source '$VENV/bin/activate'
    python -c \"
import os; os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
from transformers import AutoTokenizer, AutoModelForSequenceClassification
AutoTokenizer.from_pretrained('yiyanghkust/finbert-tone')
AutoModelForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
print('FinBERT cached')
\"
    python -m spacy download en_core_web_sm
"
ok "FinBERT cached, spaCy model installed"

step "9/12 .env file"
ENV_FILE="$INSTALL_DIR/.env"
if [ -f "$ENV_FILE" ]; then
    warn ".env already exists — skipping (edit manually if needed)"
else
    read -rp  "  Alpaca API key:         " ALPACA_KEY
    read -rsp "  Alpaca secret key:      " ALPACA_SECRET; echo
    read -rp  "  Alpaca paper? (true/false, default true): " ALPACA_PAPER
    ALPACA_PAPER="${ALPACA_PAPER:-true}"
    read -rsp "  Telegram bot token:     " TG_TOKEN; echo
    read -rp  "  Telegram chat ID:       " TG_CHAT
    read -rp  "  FMP API key (optional): " FMP_KEY
    read -rp  "  NewsAPI key (optional): " NEWS_KEY
    read -rp  "  OpenBB PAT (optional):  " OPENBB_PAT

    cat > "$ENV_FILE" <<EOF
ALPACA_API_KEY=$ALPACA_KEY
ALPACA_SECRET_KEY=$ALPACA_SECRET
ALPACA_PAPER=$ALPACA_PAPER

DATABASE_URL=postgresql://$DB_USER:$DB_PASS@localhost:5432/$DB_NAME

TELEGRAM_BOT_TOKEN=$TG_TOKEN
TELEGRAM_CHAT_ID=$TG_CHAT

FMP_API_KEY=$FMP_KEY
NEWS_API_KEY=$NEWS_KEY
OPENBB_PAT=$OPENBB_PAT
EOF
    chmod 600 "$ENV_FILE"
    chown "$SERVICE_USER":"$SERVICE_USER" "$ENV_FILE"
    ok ".env written (chmod 600)"
fi

step "10/12 Alembic migration"
sudo -u "$SERVICE_USER" bash -c "
    cd '$INSTALL_DIR'
    source venv/bin/activate
    alembic upgrade head
"
ok "Database schema up to date"

step "11/12 systemd service"
cat > "/etc/systemd/system/$SERVICE_NAME.service" <<EOF
[Unit]
Description=trading-system-v2
After=network.target postgresql.service
Requires=postgresql.service

[Service]
Type=simple
User=$SERVICE_USER
WorkingDirectory=$INSTALL_DIR
EnvironmentFile=$INSTALL_DIR/.env
ExecStart=$VENV/bin/python run_server.py
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=$SERVICE_NAME

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable "$SERVICE_NAME"
ok "systemd service installed and enabled on boot"

step "12/12 Start service"
systemctl start "$SERVICE_NAME"
sleep 3
if systemctl is-active --quiet "$SERVICE_NAME"; then
    ok "Service is running"
    echo ""
    echo -e "${GREEN}=== Deploy complete ===${NC}"
    echo ""
    echo "Monitor logs:        journalctl -u $SERVICE_NAME -f"
    echo "Stop:                systemctl stop $SERVICE_NAME"
    echo "Restart:             systemctl restart $SERVICE_NAME"
    echo ""
    echo "LightGBM bootstrap (run once, ~30 min, paper trade runs in parallel):"
    echo "  sudo -u $SERVICE_USER bash -c 'cd $INSTALL_DIR && source venv/bin/activate && python -m strategy.training_pipeline --symbols AAPL MSFT NVDA GOOGL META AMZN TSLA AMD CRM NFLX --years 5'"
    echo ""
    echo "YOLOv8 training (optional):"
    echo "  See scripts/train_yolov8.sh (coming soon) or docs in scripts/train_yolov8.ps1"
else
    warn "Service did not start cleanly. Check logs:"
    echo "  journalctl -u $SERVICE_NAME -n 50"
    exit 1
fi
