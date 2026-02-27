#!/bin/bash
# ---------------------------------------------------------------------------
# user_data.sh — EC2 bootstrap script for the Bitunix grid bot
# Runs once at instance creation as root.
# ---------------------------------------------------------------------------
set -euxo pipefail

REPO_URL="${repo_url}"
REPO_BRANCH="${repo_branch}"
BOT_DIR="/home/ec2-user/AS-Grid"

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------
dnf update -y
dnf install -y git python3 python3-pip python3-devel gcc

# ---------------------------------------------------------------------------
# 2. Clone repo as ec2-user
# ---------------------------------------------------------------------------
su - ec2-user -c "
  git clone '$REPO_URL' '$BOT_DIR'
  cd '$BOT_DIR'
  git checkout '$REPO_BRANCH'
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -r requirements.txt
"

# ---------------------------------------------------------------------------
# 3. Create a placeholder .env (populated by GitHub Actions on first deploy)
# ---------------------------------------------------------------------------
cat > "$BOT_DIR/.env" << 'ENV'
# Populated by GitHub Actions deploy workflow — do not edit manually
API_KEY=
API_SECRET=
COIN_NAME=XRP
GRID_SPACING=0.004
INITIAL_QUANTITY=1
LEVERAGE=20
ENABLE_NOTIFICATIONS=false
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
ENV

chown ec2-user:ec2-user "$BOT_DIR/.env"
chmod 600 "$BOT_DIR/.env"

# ---------------------------------------------------------------------------
# 4. Install systemd service
# ---------------------------------------------------------------------------
cat > /etc/systemd/system/bitunix-bot.service << 'SERVICE'
[Unit]
Description=Bitunix Grid Bot
After=network-online.target
Wants=network-online.target

[Service]
User=ec2-user
WorkingDirectory=/home/ec2-user/AS-Grid
EnvironmentFile=/home/ec2-user/AS-Grid/.env
ExecStart=/home/ec2-user/AS-Grid/.venv/bin/python -m src.single_bot.bitunix_bot
Restart=always
RestartSec=15
StandardOutput=journal
StandardError=journal
SyslogIdentifier=bitunix-bot

[Install]
WantedBy=multi-user.target
SERVICE

systemctl daemon-reload
systemctl enable bitunix-bot
# Do NOT start yet — secrets are empty until first GitHub Actions deploy

# ---------------------------------------------------------------------------
# 5. Run backtest on first boot (no API keys needed — uses public klines)
#    Output saved to ~/AS-Grid/backtest.log for review
# ---------------------------------------------------------------------------
su - ec2-user -c "
  cd '$BOT_DIR'
  echo '=== Backtest started at '\$(date)' ===' > backtest.log
  .venv/bin/python asBack/backtest_grid_bitunix.py >> backtest.log 2>&1
  echo '=== Backtest finished at '\$(date)' ===' >> backtest.log
" || true   # don't fail the boot if backtest errors
