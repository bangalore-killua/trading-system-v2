#!/usr/bin/env bash
# setup_backup.sh — Install nightly pg_dump backup via cron
#
# Run once after deploy.sh:
#   sudo bash scripts/setup_backup.sh
#
# Creates a cron job that runs at 2am daily, keeping 7 days of backups.
# Backups land in /var/backups/trading-system-v2/

set -euo pipefail
BACKUP_DIR="/var/backups/trading-system-v2"
DB_NAME="trading_system_v2"
DB_USER="killua_trading"

mkdir -p "$BACKUP_DIR"
chown postgres:postgres "$BACKUP_DIR"

# Write backup script
cat > /usr/local/bin/backup-trading-db.sh <<'SCRIPT'
#!/usr/bin/env bash
set -euo pipefail
BACKUP_DIR="/var/backups/trading-system-v2"
DB_NAME="trading_system_v2"
DB_USER="killua_trading"
DATE=$(date +%Y-%m-%d)
OUT="$BACKUP_DIR/trading_system_v2_$DATE.sql.gz"
pg_dump -U "$DB_USER" "$DB_NAME" | gzip > "$OUT"
# Keep last 7 days only
find "$BACKUP_DIR" -name "*.sql.gz" -mtime +7 -delete
echo "Backup written: $OUT"
SCRIPT

chmod +x /usr/local/bin/backup-trading-db.sh

# Add cron job for postgres user (pg_dump runs cleanly as postgres)
CRON_LINE="0 2 * * * /usr/local/bin/backup-trading-db.sh >> /var/log/trading-backup.log 2>&1"
(crontab -u postgres -l 2>/dev/null | grep -v backup-trading-db; echo "$CRON_LINE") | crontab -u postgres -

echo "[OK] Nightly backup cron installed (2am daily, 7-day retention)"
echo "     Backups at: $BACKUP_DIR"
echo "     Test with: sudo -u postgres /usr/local/bin/backup-trading-db.sh"
