#!/usr/bin/env python3
"""
HYPE/Hyperliquid — DeepSeek V4 Pro Multi-Factor Analyst
Runs every 6h via cron. Generates analysis + dashboard data.

Usage:
    python3 hype_monitor.py --output-dir ~/Projects/hype-monitor/public/data
"""
import argparse, json, os, sys, time, datetime
import numpy as np
import requests

# ── Constants ──
HL_API = "https://api.hyperliquid.xyz/info"
COIN = "HYPE"
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "deepseek-v4-pro:cloud"

# ── Data Fetchers ──

def fetch_candles(interval, days_back):
    now_ms = int(time.time() * 1000)
    start = now_ms - days_back * 86400 * 1000
    r = requests.post(HL_API, json={
        "type": "candleSnapshot",
        "req": {"coin": COIN, "interval": interval, "startTime": start, "endTime": now_ms}
    }, timeout=20)
    return r.json()

def fetch_funding(days_back=30):
    now_ms = int(time.time() * 1000)
    start = now_ms - days_back * 86400 * 1000
    r = requests.post(HL_API, json={
        "type": "fundingHistory", "coin": COIN, "startTime": start, "endTime": now_ms
    }, timeout=15)
    return r.json()

def fetch_book():
    r = requests.post(HL_API, json={"type": "l2Book", "coin": COIN}, timeout=10)
    return r.json()

def fetch_meta():
    r = requests.post(HL_API, json={"type": "metaAndAssetCtxs"}, timeout=15)
    m = r.json()
    universe = m[0]["universe"]
    ctxs = m[1]
    for i, a in enumerate(universe):
        if a["name"] == COIN:
            return ctxs[i]
    return None

def fetch_btc_candles():
    now_ms = int(time.time() * 1000)
    r = requests.post(HL_API, json={
        "type": "candleSnapshot",
        "req": {"coin": "BTC", "interval": "1d", "startTime": now_ms - 90 * 86400 * 1000, "endTime": now_ms}
    }, timeout=15)
    return r.json()

# ── Technical Calculations ──

def calc_sma(data, period):
    result = np.full_like(data, np.nan, dtype=float)
    for i in range(period-1, len(data)):
        result[i] = np.mean(data[i-period+1:i+1])
    return result

def calc_ema(data, period):
    result = np.full_like(data, np.nan, dtype=float)
    k = 2 / (period + 1)
    start = period - 1
    result[start] = np.mean(data[:period])
    for i in range(start+1, len(data)):
        result[i] = data[i] * k + result[i-1] * (1 - k)
    return result

def calc_rsi(data, period=14):
    result = np.full_like(data, np.nan, dtype=float)
    deltas = np.diff(data)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    if len(gains) < period: return result
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    result[period] = 100 - 100 / (1 + avg_gain / (avg_loss + 1e-10))
    for i in range(period+1, len(data)):
        avg_gain = (avg_gain * (period-1) + gains[i-1]) / period
        avg_loss = (avg_loss * (period-1) + losses[i-1]) / period
        result[i] = 100 - 100 / (1 + avg_gain / (avg_loss + 1e-10))
    return result

def calc_macd(close):
    fast = calc_ema(close, 12)
    slow = calc_ema(close, 26)
    line = fast - slow
    # Signal is EMA of MACD line (skip NaN)
    valid = line[~np.isnan(line)]
    if len(valid) < 9:
        sig = np.full_like(line, np.nan, dtype=float)
    else:
        sig = calc_ema(line[~np.isnan(line)], 9)
        full_sig = np.full_like(line, np.nan, dtype=float)
        # Map back to full sized array
        vmask = ~np.isnan(line)
        vidx = np.where(vmask)[0]
        if len(vidx) >= len(sig):
            full_sig[vidx[:len(sig)]] = sig
        sig = full_sig
    return line, sig

def calc_bollinger(close, period=20, mult=2):
    sma = calc_sma(close, period)
    std = np.full_like(close, np.nan, dtype=float)
    for i in range(period-1, len(close)):
        std[i] = np.std(close[i-period+1:i+1])
    upper = sma + mult * std
    lower = sma - mult * std
    return upper, sma, lower

# ── Build Analysis Data ──

def build_analysis():
    print("[1/6] Fetching data...")

    daily = fetch_candles("1d", 120)
    btc_daily = fetch_btc_candles()
    funding_data = fetch_funding(30)
    book = fetch_book()
    ctx = fetch_meta()

    if not daily:
        print("ERROR: No daily data")
        return None

    # Process daily
    n = len(daily)
    dates = [str(datetime.date.fromtimestamp(c["t"]/1000)) for c in daily]
    opens = np.array([float(c["o"]) for c in daily])
    highs = np.array([float(c["h"]) for c in daily])
    lows = np.array([float(c["l"]) for c in daily])
    closes = np.array([float(c["c"]) for c in daily])
    volumes = np.array([float(c["v"]) for c in daily])

    # Technicals
    sma7 = calc_sma(closes, 7); sma20 = calc_sma(closes, 20); sma50 = calc_sma(closes, 50)
    ema12 = calc_ema(closes, 12); ema26 = calc_ema(closes, 26)
    rsi14 = calc_rsi(closes, 14); rsi7 = calc_rsi(closes, 7)
    bb_upper, bb_sma, bb_lower = calc_bollinger(closes)
    macd_line, macd_signal = calc_macd(closes)

    # ATR
    atr7 = np.full_like(closes, np.nan, dtype=float)
    atr14 = np.full_like(closes, np.nan, dtype=float)
    trs = np.maximum(highs[1:] - lows[1:], np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])))
    for i in range(7, len(closes)):
        atr7[i] = np.mean(trs[i-6:i])
    for i in range(14, len(closes)):
        atr14[i] = np.mean(trs[i-13:i])

    # BTC correlation
    btc_closes = np.array([float(c["c"]) for c in btc_daily])
    btc_corr_30 = round(np.corrcoef(closes[-30:], btc_closes[-30:])[0,1], 3) if len(closes) >= 30 and len(btc_closes) >= 30 else None

    # Funding
    funding_rates = [float(f.get("rate", f.get("fundingRate", 0))) for f in funding_data]
    funding_dates = [str(datetime.date.fromtimestamp(f["time"]/1000)) for f in funding_data]
    funding_summary = {
        "count": len(funding_rates),
        "avg_8h": round(sum(funding_rates)/len(funding_rates)*100, 4) if funding_rates else None,
        "latest_8h": round(funding_rates[-1]*100, 4) if funding_rates else None,
        "annualized_pct": round(funding_rates[-1] * 3 * 365 * 100, 2) if funding_rates else None,
    }

    # Orderbook
    bids = book["levels"][0] if book and "levels" in book else []
    asks = book["levels"][1] if book and len(book["levels"]) > 1 else []
    bid_vol = sum(float(lvl["sz"]) for lvl in bids[:20]) if bids else 0
    ask_vol = sum(float(lvl["sz"]) for lvl in asks[:20]) if asks else 0
    imbalance = round((bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-10), 4) if (bid_vol + ask_vol) > 0 else 0
    spread = round(float(asks[0]["px"]) - float(bids[0]["px"]), 4) if (bids and asks) else 0
    spread_bps = round(spread / float(bids[0]["px"]) * 10000, 2) if (bids and float(bids[0]["px"]) > 0) else 0
    best_bid = float(bids[0]["px"]) if bids else 0
    best_ask = float(asks[0]["px"]) if asks else 0

    # Current values
    current_price = round(closes[-1], 3)
    dist_sma7 = round(((current_price - sma7[-1]) / sma7[-1]) * 100, 1) if not np.isnan(sma7[-1]) else None
    dist_sma20 = round(((current_price - sma20[-1]) / sma20[-1]) * 100, 1) if not np.isnan(sma20[-1]) else None
    dist_sma50 = round(((current_price - sma50[-1]) / sma50[-1]) * 100, 1) if not np.isnan(sma50[-1]) else None

    # Returns
    def pct_change(d, days):
        if len(d) < days: return None
        return round(((d[-1] - d[-days]) / d[-days]) * 100, 2)

    returns = {
        "1d": pct_change(closes, 1),
        "3d": pct_change(closes, 3),
        "7d": pct_change(closes, 7),
        "14d": pct_change(closes, 14),
        "30d": pct_change(closes, 30),
        "60d": pct_change(closes, 60),
        "90d": pct_change(closes, 90),
    }

    # Volume metrics
    vol7 = np.mean(volumes[-7:])
    vol30 = np.mean(volumes[-30:])
    vol_30d = round(vol7 / vol30, 2) if vol30 > 0 else None

    # RSI regime
    rsi_val = round(rsi14[-1], 1) if not np.isnan(rsi14[-1]) else None
    rsi_signal = "OVERSOLD" if rsi_val and rsi_val < 30 else ("OVERBOUGHT" if rsi_val and rsi_val > 70 else "NEUTRAL")

    # Volatility regime
    atr_val = round(atr14[-1], 2) if not np.isnan(atr14[-1]) else None
    atr_pct = round(atr_val / current_price * 100, 2) if atr_val else None
    vol_regime = "EXPANDING" if atr_pct and atr_pct > 10 else ("CONTRACTING" if atr_pct and atr_pct < 5 else "NORMAL")

    # Key levels
    sl_2x = round(current_price - 2 * atr_val, 2) if atr_val else None
    sl_3x = round(current_price - 3 * atr_val, 2) if atr_val else None
    poc = round(np.sum(closes[-30:]*volumes[-30:]) / np.sum(volumes[-30:]), 2)

    # Market context
    oi = ctx.get("openInterest", "?") if ctx else "?"
    mark = round(float(ctx["markPx"]), 3) if ctx else None

    return {
        "meta": {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "data_points": n,
            "price": current_price,
            "mark_price": mark,
            "open_interest": str(oi) if oi else "?",
        },
        "returns": returns,
        "technical": {
            "sma7": round(sma7[-1], 2) if not np.isnan(sma7[-1]) else None,
            "sma20": round(sma20[-1], 2) if not np.isnan(sma20[-1]) else None,
            "sma50": round(sma50[-1], 2) if not np.isnan(sma50[-1]) else None,
            "rsi14": rsi_val,
            "rsi7": round(rsi7[-1], 1) if not np.isnan(rsi7[-1]) else None,
            "rsi_signal": rsi_signal,
            "macd_line": round(macd_line[-1], 4) if not np.isnan(macd_line[-1]) else None,
            "macd_signal": round(macd_signal[-1], 4) if not np.isnan(macd_signal[-1]) else None,
            "macd_hist": round((macd_line[-1] - macd_signal[-1]), 4) if not np.isnan(macd_line[-1]) and not np.isnan(macd_signal[-1]) else None,
            "bb_upper": round(bb_upper[-1], 2) if not np.isnan(bb_upper[-1]) else None,
            "bb_lower": round(bb_lower[-1], 2) if not np.isnan(bb_lower[-1]) else None,
            "atr14": atr_val,
            "atr_pct": atr_pct,
            "dist_sma7_pct": dist_sma7,
            "dist_sma20_pct": dist_sma20,
            "dist_sma50_pct": dist_sma50,
            "stop_2x": sl_2x,
            "stop_3x": sl_3x,
            "poc": poc,
        },
        "volume": {
            "latest_M": round(volumes[-1]/1e6, 2),
            "avg7d_M": round(vol7/1e6, 2),
            "avg30d_M": round(vol30/1e6, 2),
            "ratio_7v30": vol_30d,
            "poc": poc,
        },
        "orderbook": {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": spread,
            "spread_bps": spread_bps,
            "bid_vol_20": round(bid_vol, 2),
            "ask_vol_20": round(ask_vol, 2),
            "imbalance": imbalance,
            "pressure": "BULLISH" if imbalance > 0.1 else ("BEARISH" if imbalance < -0.1 else "NEUTRAL"),
        },
        "funding": funding_summary,
        "correlation": {
            "btc_30d": btc_corr_30,
        },
        "timeseries": {
            "dates": dates,
            "closes": closes[~np.isnan(closes)].tolist() + ["NaN"] * int(np.sum(np.isnan(closes))),  # hack for JSON
            "volumes": volumes.tolist(),
            "rsi": [round(x, 1) if not np.isnan(x) else None for x in rsi14.tolist()],
            "sma7": [round(x, 2) if not np.isnan(x) else None for x in sma7.tolist()],
            "sma20": [round(x, 2) if not np.isnan(x) else None for x in sma20.tolist()],
            "sma50": [round(x, 2) if not np.isnan(x) else None for x in sma50.tolist()],
            "bb_upper": [round(x, 2) if not np.isnan(x) else None for x in bb_upper.tolist()],
            "bb_lower": [round(x, 2) if not np.isnan(x) else None for x in bb_lower.tolist()],
            "macd": [round(x, 4) if not np.isnan(x) else None for x in macd_line.tolist()],
            "macd_signal": [round(x, 4) if not np.isnan(x) else None for x in macd_signal.tolist()],
        }
    }

# ── DeepSeek V4 Pro Analysis ──

def generate_ai_analysis(data):
    print("[2/6] Generating DeepSeek V4 Pro analysis...")
    price = data["meta"]["price"]
    rsi = data["technical"]["rsi14"]
    btc_corr = data["correlation"]["btc_30d"]
    ob = data["orderbook"]
    funding = data["funding"]
    volume = data["volume"]
    tech = data["technical"]
    returns = data["returns"]

    prompt = f"""You are a senior crypto trader. Based on this live HYPE/USD data, write a brief multi-factor analysis.

CONTEXT: HYPE (Hyperliquid token) is at ${price}. It is a perp DEX token with an aggressive buyback engine (97% of fees buyback HYPE). It's the dominant perp DEX with 32-44% market share. Recent ETF approval, HIP-3 tokenized stocks, and institutional accumulation (a16z $170M+) are key drivers.

DATA:
- Price: ${price} | 1D: {returns['1d']}% | 7D: {returns['7d']}% | 30D: {returns['30d']}%
- RSI(14): {rsi} ({data['technical']['rsi_signal']})
- BTC correlation (30D): {btc_corr} (decoupled if negative)
- MACD line: {tech['macd_line']} | Signal: {tech['macd_signal']} | Histogram: {tech['macd_hist']}
- Bollinger: ${tech['bb_lower']} - ${tech['bb_upper']} | B%: ${tech['poc']}
- SMA7: ${tech['sma7']} ({tech['dist_sma7_pct']}% away)
- SMA20: ${tech['sma20']} ({tech['dist_sma20_pct']}% away)
- SMA50: ${tech['sma50']} ({tech['dist_sma50_pct']}% away)
- Stop levels: 2x ATR @ ${tech['stop_2x']} | 3x ATR @ ${tech['stop_3x']}
- Orderbook: Bid/Ask imbalance {ob['imbalance']} ({ob['pressure']}) | Spread {ob['spread_bps']} bps
- Volume (latest): {volume['latest_M']}M vs 7D: {volume['avg7d_M']}M vs 30D: {volume['avg30d_M']}M
- Funding (8h avg): {funding['avg_8h']}% (annualized: {funding['annualized_pct']}%)
- Volatility (14D ATR): {tech['atr_pct']}% daily

Write EXACTLY in this format — each section must be 1-2 sentences max, very concise:

**TREND:** [bullish/neutral/bearish — state clearly with one reason]
**MOMENTUM:** [strength/weakness — RSI + MACD interpretation]
**DECOUPLING:** [is HYPE diverging from BTC? what does it mean?]
**LIQUIDITY:** [orderbook + spread + volume — institutional or retail?]
**LEVERS:** [funding + OI — are longs crowded?]
**VALUE:** [is it overbought/oversold relative to recent price action?]
**RISK:** [the #1 risk right now, specific]

End with ONE bold actionable line: what a trader should do."""

    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"num_predict": 8000, "temperature": 0.3}
        }, timeout=300)
        result = resp.json()
        return result.get("message", {}).get("content", "[Analysis unavailable — model response was empty]")
    except Exception as e:
        print(f"AI analysis failed: {e}")
        return f"[Analysis unavailable: {str(e)[:100]}]"

# ── Write Dashboard Data ──

def write_data(data, ai_text, output_dir):
    print("[3/6] Writing data...")
    os.makedirs(output_dir, exist_ok=True)
    payload = {
        **data,
        "ai_analysis": ai_text,
    }
    with open(os.path.join(output_dir, "hype_data.json"), "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Data written to: {os.path.join(output_dir, 'hype_data.json')}")

# ── Main ──

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="~/Projects/hype-monitor/public/data")
    args = parser.parse_args()
    output_dir = os.path.expanduser(args.output_dir)

    data = build_analysis()
    if not data:
        print("FAILED: Could not build analysis")
        sys.exit(1)

    ai_text = generate_ai_analysis(data)
    write_data(data, ai_text, output_dir)
    print("Done.")

if __name__ == "__main__":
    main()
