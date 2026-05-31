#!/bin/bash
# hype_monitor.sh — Run every 6 hours via cron
# Updates data + pushes to GitHub + notifies Discord
set -euo pipefail

PROJECT_DIR="$HOME/Projects/hype-monitor"
SCRIPT_DIR="$HOME/.hermes/scripts"
DATA_DIR="$PROJECT_DIR/data"
DISCORD_CHANNEL="1510673439461998622"
# Discord bot token loaded from .env
source "$HOME/.hermes/hermes-agent/venv/bin/activate"

cd "$PROJECT_DIR"

echo "=== $(date) | HYPE Monitor Run ==="

# 1. Generate fresh data with DeepSeek V4 Pro analysis
python3 "$SCRIPT_DIR/hype_monitor.py" --output-dir "$DATA_DIR" 2>&1 | tee /tmp/hype_monitor.log

# 2. Commit and push to GitHub Pages
if [ -n "$(git status --porcelain data/hype_data.json 2>/dev/null)" ]; then
    git add data/hype_data.json
    git commit -m "Update: $(date '+%Y-%m-%d %H:%M') — auto" --allow-empty
    git push origin gh-pages
    echo "Pushed to GitHub Pages"
else
    echo "No data changes"
fi

# 3. Send Discord notification
python3 - "$DISCORD_CHANNEL" "$DATA_DIR/hype_data.json" << 'PYEOF'
import sys, os, json, datetime, requests

channel = sys.argv[1]
with open(sys.argv[2]) as f: d = json.load(f)

# Read Discord token from .env
token = None
env_path = os.path.expanduser("~/.hermes/.env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            if line.startswith("DISCORD_BOT_TOKEN="):
                token = line.strip().split("=", 1)[1]
                break

if not token or "..." in token:
    print("No valid Discord token — notification skipped")
    sys.exit(0)

meta = d["meta"]
tech = d["technical"]
returns = d["returns"]

price = meta["price"]
rsi = tech["rsi14"]
rsi_sig = tech["rsi_signal"]
macd = tech["macd_hist"]
ob = d["orderbook"]

emoji = "📈" if (returns.get("1d") or 0) >= 0 else "📉"
trend = "BULLISH" if (returns.get("7d") or 0) > 0 else "BEARISH"

msg = f"""{emoji} **HYPE Monitor Update** | `{datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC`

```
Price:  ${price:.2f}  |  1D: {returns.get('1d', 'N/A'):+.2f}%  |  7D: {returns.get('7d', 'N/A'):+.2f}%  |  30D: {returns.get('30d', 'N/A'):+.2f}%
RSI:    {rsi:.1f} ({rsi_sig})
MACD:   {macd:.4f} ({'+' if macd > 0 else ''}histogram)
OB:     {ob['imbalance']:.3f} ({ob['pressure']})
Funding:{d['funding'].get('latest_8h', 'N/A')}% (8h)  |  {d['funding'].get('annualized_pct', 'N/A')}% (ann)
BTC ρ:  {d['correlation'].get('btc_30d', 'N/A')}
```

**Trend: {trend}** | [View Dashboard](https://buildandtestppg.github.io/hype-monitor/)

_{d.get('ai_analysis', 'Analysis: DeepSeek V4 Pro')[:200]}..._
"""

resp = requests.post(
    f"https://discord.com/api/v10/channels/{channel}/messages",
    headers={"Authorization": f"Bot {token}", "Content-Type": "application/json"},
    json={"content": msg},
    timeout=15
)
print(f"Discord: {resp.status_code} {'OK' if resp.status_code == 200 else resp.text[:100]}")
PYEOF

echo "=== Done $(date) ==="
