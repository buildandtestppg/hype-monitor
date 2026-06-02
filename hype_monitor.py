#!/usr/bin/env python3
"""
HYPE/Hyperliquid — DeepSeek V4 Pro Multi-Factor Analyst
Runs every 6h via cron. Generates analysis + dashboard data.

Usage:
    python3 hype_monitor.py --output-dir ~/Projects/hype-monitor/public/data
"""
import argparse, json, os, sys, time, datetime, sqlite3
import numpy as np
import requests

# ── Constants ──
HL_API = "https://api.hyperliquid.xyz/info"
COIN = "HYPE"
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "deepseek-v4-pro:cloud"
PREDICTION_MODEL = "minimax-m3:cloud"

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

# ── Comparison assets (top majors) ──
COMPARISON_ASSETS = ["BTC", "ETH", "SOL", "BNB", "XRP"]

def fetch_btc_candles():
    now_ms = int(time.time() * 1000)
    r = requests.post(HL_API, json={
        "type": "candleSnapshot",
        "req": {"coin": "BTC", "interval": "1d", "startTime": now_ms - 90 * 86400 * 1000, "endTime": now_ms}
    }, timeout=15)
    return r.json()

def fetch_asset_candles(coin, days_back=90):
    """Fetch daily candles for any Hyperliquid-listed asset."""
    now_ms = int(time.time() * 1000)
    try:
        r = requests.post(HL_API, json={
            "type": "candleSnapshot",
            "req": {"coin": coin, "interval": "1d", "startTime": now_ms - days_back * 86400 * 1000, "endTime": now_ms}
        }, timeout=15)
        data = r.json()
        if data and len(data) > 0:
            return data
    except Exception as e:
        print(f"  {coin} candle fetch failed: {e}")
    return None

def build_comparison_data(hype_closes):
    """Build relative performance comparison of HYPE vs top majors."""
    print("[1c/6] Fetching comparison assets...")
    comparison = {}
    for asset in COMPARISON_ASSETS:
        candles = fetch_asset_candles(asset)
        if not candles or len(candles) < 2:
            print(f"  {asset}: insufficient data, skipping")
            continue
        closes = np.array([float(c["c"]) for c in candles])
        dates = [str(datetime.date.fromtimestamp(c["t"]/1000)) for c in candles]

        # Returns
        def pct(d, days):
            if len(d) < days: return None
            return round(((d[-1] - d[-days]) / d[-days]) * 100, 2)

        asset_returns = {
            "1d": pct(closes, 1),
            "7d": pct(closes, 7),
            "30d": pct(closes, 30),
        }

        # Correlation with HYPE (30d)
        min_len = min(len(hype_closes), len(closes))
        corr_30 = round(np.corrcoef(hype_closes[-30:], closes[-30:])[0,1], 3) if min_len >= 30 else None

        # Normalized performance (base 100 from 30d ago)
        norm_start = closes[-30] if len(closes) >= 30 else closes[0]
        norm_days = min(30, len(closes))
        normalized = [round(((c - norm_start) / norm_start) * 100, 2) for c in closes[-norm_days:]]

        comparison[asset] = {
            "price": round(closes[-1], 2),
            "returns": asset_returns,
            "correlation_hype_30d": corr_30,
            "normalized_30d": normalized,
            "dates_30d": dates[-norm_days:] if len(dates) >= norm_days else dates,
        }
        print(f"  {asset}: ${closes[-1]:,.2f} | 1d={asset_returns['1d']}% 7d={asset_returns['7d']}% 30d={asset_returns['30d']}% corr={corr_30}")

    return comparison

def fetch_defillama_protocol():
    """Fetch Hyperliquid protocol data from DeFi Llama (TVL, metadata)"""
    try:
        r = requests.get("https://api.llama.fi/protocol/hyperliquid", timeout=20)
        return r.json()
    except Exception as e:
        print(f"  DeFi Llama protocol fetch failed: {e}")
        return None

def fetch_defillama_fees():
    """Fetch fee/revenue data from DeFi Llama summary"""
    try:
        r = requests.get("https://api.llama.fi/summary/fees/hyperliquid?dataType=dailyRevenue", timeout=20)
        data = r.json()
        return {
            "fees_24h": data.get("total24h"),
            "fees_48h": data.get("total48hto24h"),
            "fees_7d_total": data.get("total7d"),
            "fees_30d_total": data.get("total30d"),
            "fees_all_time": data.get("totalAllTime"),
            "fees_change_1d_pct": data.get("change_1d"),
        }
    except Exception as e:
        print(f"  DeFi Llama fees fetch failed: {e}")
        return None

def fetch_defillama_volume():
    """Fetch DEX volume data from DeFi Llama"""
    try:
        r = requests.get("https://api.llama.fi/summary/dexs/hyperliquid", timeout=20)
        data = r.json()
        return {
            "dex_vol_24h": data.get("total24h"),
            "dex_vol_7d_total": data.get("total7d"),
            "dex_vol_30d_total": data.get("total30d"),
            "dex_vol_all_time": data.get("totalAllTime"),
        }
    except Exception as e:
        print(f"  DeFi Llama volume fetch failed: {e}")
        return None

def fetch_coingecko_data():
    """Fetch FDV, volume, and supply data from CoinGecko"""
    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/coins/hyperliquid?localization=false",
            timeout=20
        )
        data = r.json()
        md = data.get("market_data", {})
        return {
            "fdv_usd": md.get("fully_diluted_valuation", {}).get("usd"),
            "total_volume_24h_usd": md.get("total_volume", {}).get("usd"),
            "circulating_supply": md.get("circulating_supply"),
            "total_supply": md.get("total_supply"),
            "max_supply": md.get("max_supply"),
        }
    except Exception as e:
        print(f"  CoinGecko fetch failed: {e}")
        return None

def fetch_btc_price():
    """Fetch BTC price from CoinGecko as fallback, or Hyperliquid BTC mark"""
    try:
        now_ms = int(time.time() * 1000)
        r = requests.post(HL_API, json={
            "type": "candleSnapshot",
            "req": {"coin": "BTC", "interval": "1h", "startTime": now_ms - 2 * 3600 * 1000, "endTime": now_ms}
        }, timeout=10)
        data = r.json()
        if data and len(data) > 0:
            return float(data[-1]["c"])
    except Exception:
        pass
    return None

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
    valid = line[~np.isnan(line)]
    if len(valid) < 9:
        sig = np.full_like(line, np.nan, dtype=float)
    else:
        sig = calc_ema(line[~np.isnan(line)], 9)
        full_sig = np.full_like(line, np.nan, dtype=float)
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
        std[i] = np.std(close[i-period+1:i+1], ddof=1)
    upper = sma + mult * std
    lower = sma - mult * std
    return upper, sma, lower

# ── Research Pipeline ──

def build_research_data(current_price, ctx, technical, funding_summary, orderbook, btc_corr_30):
    """Build DeFi Llama research data with narrative score"""
    print("[1c/6] Fetching DeFi Llama research...")

    dl = fetch_defillama_protocol() or {}
    fees_dl = fetch_defillama_fees() or {}
    volume_dl = fetch_defillama_volume() or {}
    cg = fetch_coingecko_data() or {}
    btc_price = fetch_btc_price()

    # Protocol stats
    tvl = dl.get("tvl", [])
    current_tvl = tvl[-1]["totalLiquidityUSD"] if tvl and isinstance(tvl, list) and len(tvl) > 0 else None

    mcap = dl.get("mcap") if dl else None
    # Use CoinGecko FDV if DeFi Llama doesn't have it
    fdv = dl.get("fdv") if dl else None
    if fdv is None and cg.get("fdv_usd"):
        fdv = cg.get("fdv_usd")
    gecko_id = dl.get("gecko_id") if dl else None

    # Calculate TVL changes
    tvl_7d_change = None
    tvl_30d_change = None
    if tvl and len(tvl) >= 2:
        try:
            current_tvl_val = tvl[-1].get("totalLiquidityUSD", 0)
            now = datetime.datetime.now(datetime.timezone.utc)
            tvl_7d = None
            tvl_30d = None
            for entry in reversed(tvl):
                entry_date = datetime.datetime.fromtimestamp(entry.get("date", 0), datetime.timezone.utc)
                days_ago = (now - entry_date).days
                if tvl_7d is None and days_ago >= 6:
                    tvl_7d = entry.get("totalLiquidityUSD")
                if tvl_30d is None and days_ago >= 28:
                    tvl_30d = entry.get("totalLiquidityUSD")
                if tvl_7d is not None and tvl_30d is not None:
                    break
            if current_tvl_val and tvl_7d:
                tvl_7d_change = round(((current_tvl_val - tvl_7d) / tvl_7d) * 100, 2)
            if current_tvl_val and tvl_30d:
                tvl_30d_change = round(((current_tvl_val - tvl_30d) / tvl_30d) * 100, 2)
        except Exception:
            pass

    # Use DeFi Llama summary endpoints for fees/volume (correct data)
    fees_24h = fees_dl.get("fees_24h")
    fees_48h = fees_dl.get("fees_48h")
    fees_7d_total = fees_dl.get("fees_7d_total")
    fees_30d_total = fees_dl.get("fees_30d_total")
    fees_all_time = fees_dl.get("fees_all_time")
    fees_change_1d = fees_dl.get("fees_change_1d_pct")

    dex_vol_24h = volume_dl.get("dex_vol_24h")
    dex_vol_7d_total = volume_dl.get("dex_vol_7d_total")
    dex_vol_30d_total = volume_dl.get("dex_vol_30d_total")
    dex_vol_all_time = volume_dl.get("dex_vol_all_time")

    # Calculate trends from change_1d
    fees_trend = fees_change_1d
    vol_trend = None  # DeFi Llama DEX endpoint doesn't give change_1d

    # HYPE perp data from Hyperliquid context
    hype_perp = {}
    if ctx:
        oi_tokens_raw = ctx.get("openInterest")
        oi_tokens = float(oi_tokens_raw) if oi_tokens_raw is not None else 0
        mark_px = float(ctx.get("markPx", 0)) if ctx.get("markPx") else 0
        prev_day_px = float(ctx.get("prevDayPx", 0)) if ctx.get("prevDayPx") else 0
        day_ntl_vol = float(ctx.get("dayNtlVlm", 0)) if ctx.get("dayNtlVlm") else 0
        day_base_vol = float(ctx.get("dayBaseVlm", 0)) if ctx.get("dayBaseVlm") else 0
        funding_rate = float(ctx.get("funding", 0)) if ctx.get("funding") else 0
        premium = float(ctx.get("premium", 0)) if ctx.get("premium") else 0
        oi_usd = oi_tokens * mark_px if oi_tokens else None
        price_change = round(((mark_px - prev_day_px) / prev_day_px) * 100, 2) if prev_day_px else None

        hype_perp = {
            "oi_tokens": round(oi_tokens, 2) if oi_tokens else None,
            "oi_usd": round(oi_usd, 2) if oi_usd else None,
            "mark_px": round(mark_px, 3) if mark_px else None,
            "prev_day_px": round(prev_day_px, 3) if prev_day_px else None,
            "day_notional_vol": round(day_ntl_vol, 2) if day_ntl_vol else None,
            "day_base_vol": round(day_base_vol, 2) if day_base_vol else None,
            "funding_rate": funding_rate,
            "premium": premium,
            "price_change_24h_pct": price_change,
        }

    # Valuation ratios
    mcap_val = mcap if mcap else None
    tvl_val = current_tvl if current_tvl else None
    fees_24h_val = fees_24h if fees_24h else None
    oi_usd_val = hype_perp.get("oi_usd")

    valuation = {}
    if mcap_val:
        if tvl_val and tvl_val > 0:
            valuation["mcap_tvl_ratio"] = round(mcap_val / tvl_val, 2)
        if fees_24h_val and fees_24h_val > 0:
            valuation["mcap_annualized_fee_ratio"] = round(mcap_val / (fees_24h_val * 365), 2)
            valuation["annualized_fees_estimate"] = round(fees_24h_val * 365)
            valuation["price_to_daily_revenue"] = round(mcap_val / fees_24h_val, 2)
        if oi_usd_val and oi_usd_val > 0:
            valuation["oi_to_mcap_pct"] = round((oi_usd_val / mcap_val) * 100, 2)

    # Narrative score (0-100)
    narrative_score = calc_narrative_score(
        tvl_7d_change, tvl_30d_change, fees_trend, vol_trend,
        technical.get("rsi14"), technical.get("macd_hist"),
        funding_summary.get("latest_8h")
    )

    protocol_stats = {
        "tvl": round(current_tvl) if current_tvl else None,
        "tvl_7d_change_pct": tvl_7d_change,
        "tvl_30d_change_pct": tvl_30d_change,
        "mcap": mcap_val,
        "fdv": fdv,
        "gecko_id": gecko_id,
        "description": dl.get("description") if dl else None,
        "twitter": dl.get("twitter") if dl else None,
        "github": dl.get("github") if dl else None,
    }

    fees_section = {
        "fees_24h": round(fees_24h) if fees_24h else None,
        "fees_7d_total": round(fees_7d_total) if fees_7d_total else None,
        "fees_30d_total": round(fees_30d_total) if fees_30d_total else None,
        "fees_all_time": round(fees_all_time) if fees_all_time else None,
        "fees_7d_avg_daily": round(fees_7d_total / 7, 1) if fees_7d_total else None,
        "fees_30d_avg_daily": round(fees_30d_total / 30, 1) if fees_30d_total else None,
        "fees_trend_pct": fees_trend,
    }

    volume_section = {
        "volume_24h": round(dex_vol_24h) if dex_vol_24h else None,
        "volume_7d_total": round(dex_vol_7d_total) if dex_vol_7d_total else None,
        "volume_30d_total": round(dex_vol_30d_total) if dex_vol_30d_total else None,
        "volume_7d_avg_daily": round(dex_vol_7d_total / 7, 1) if dex_vol_7d_total else None,
        "volume_30d_avg_daily": round(dex_vol_30d_total / 30, 1) if dex_vol_30d_total else None,
        "volume_trend_pct": vol_trend,
    }

    return {
        "protocol_stats": protocol_stats,
        "fees": fees_section,
        "volume": volume_section,
        "btc_price": btc_price,
        "hype_perp": hype_perp,
        "valuation": valuation,
        "narrative_score": narrative_score,
    }

def calc_narrative_score(tvl_7d, tvl_30d, fees_trend, vol_trend, rsi, macd_hist, funding_latest):
    """Calculate 0-100 narrative score from fundamentals + technicals"""
    score = 0
    components = {}

    # TVL growth (0-20)
    tvl_score = 0
    if tvl_30d and tvl_30d > 10:
        tvl_score = min(20, tvl_30d)
    elif tvl_30d and tvl_30d > 5:
        tvl_score = 15
    elif tvl_30d and tvl_30d > 0:
        tvl_score = 10
    elif tvl_30d and tvl_30d > -5:
        tvl_score = 5
    components["tvl_growth"] = round(tvl_score, 1)
    score += tvl_score

    # Fee/revenue trend (0-20)
    fee_score = 0
    if fees_trend and fees_trend > 20:
        fee_score = 20
    elif fees_trend and fees_trend > 10:
        fee_score = 15
    elif fees_trend and fees_trend > 0:
        fee_score = 10
    elif fees_trend and fees_trend > -10:
        fee_score = 5
    components["fee_revenue"] = round(fee_score, 1)
    score += fee_score

    # Volume trend (0-20)
    vol_score = 0
    if vol_trend and vol_trend > 30:
        vol_score = 20
    elif vol_trend and vol_trend > 15:
        vol_score = 15
    elif vol_trend and vol_trend > 0:
        vol_score = 10
    elif vol_trend and vol_trend > -15:
        vol_score = 5
    components["volume_trend"] = round(vol_score, 1)
    score += vol_score

    # RSI momentum (0-15)
    rsi_score = 0
    if rsi is not None:
        if 40 <= rsi <= 65:
            rsi_score = 15  # Healthy momentum
        elif 30 <= rsi < 40 or 65 < rsi <= 75:
            rsi_score = 10
        elif rsi < 30:
            rsi_score = 5  # Oversold but could bounce
        else:
            rsi_score = 5  # Overbought caution
    components["rsi_momentum"] = round(rsi_score, 1)
    score += rsi_score

    # MACD direction (0-15)
    macd_score = 0
    if macd_hist is not None:
        if macd_hist > 0.5:
            macd_score = 15
        elif macd_hist > 0:
            macd_score = 10
        elif macd_hist > -0.5:
            macd_score = 5
        else:
            macd_score = 0
    components["macd_direction"] = round(macd_score, 1)
    score += macd_score

    # Funding regime (0-10) — low funding = bullish (not crowded)
    funding_score = 0
    if funding_latest is not None:
        if funding_latest < 0.01:
            funding_score = 10
        elif funding_latest < 0.03:
            funding_score = 7
        elif funding_latest < 0.05:
            funding_score = 5
        else:
            funding_score = 2
    components["funding_regime"] = round(funding_score, 1)
    score += funding_score

    label = "BULLISH" if score >= 60 else ("NEUTRAL" if score >= 40 else "BEARISH")

    return {
        "score": round(score),
        "label": label,
        "components": components,
    }

# ── Prediction System ──

def generate_heuristic_prediction(price, tech, returns):
    """Generate a simple heuristic prediction when the LLM fails"""
    atr = tech.get('atr14', price * 0.05)
    rsi = tech.get('rsi14', 50)
    macd_hist = tech.get('macd_hist', 0)
    
    # Determine direction based on MACD + RSI
    bullish = macd_hist > 0 and rsi < 80
    bearish = macd_hist < 0 and rsi > 20
    
    if bullish:
        direction = "BULLISH"
        mult_6h = 0.3
        mult_12h = 0.6
        mult_24h = 1.0
    elif bearish:
        direction = "BEARISH"
        mult_6h = -0.3
        mult_12h = -0.6
        mult_24h = -1.0
    else:
        direction = "NEUTRAL"
        mult_6h = 0.1
        mult_12h = 0.2
        mult_24h = 0.3
    
    # Build reasoning string
    reasons = []
    if macd_hist > 0:
        reasons.append(f"MACD hist +{macd_hist:.2f} bullish")
    elif macd_hist < 0:
        reasons.append(f"MACD hist {macd_hist:.2f} bearish")
    
    if rsi > 70:
        reasons.append(f"RSI {rsi} overbought caps upside")
    elif rsi < 30:
        reasons.append(f"RSI {rsi} oversold supports bounce")
    else:
        reasons.append(f"RSI {rsi} neutral")
    
    reasons.append(f"ATR ${atr:.2f} guides targets")
    
    return {
        "pred_6h_direction": direction,
        "pred_6h_target": round(price + mult_6h * atr, 2),
        "pred_12h_direction": direction,
        "pred_12h_target": round(price + mult_12h * atr, 2),
        "pred_24h_direction": direction,
        "pred_24h_target": round(price + mult_24h * atr, 2),
        "confidence": 5,
        "reasoning": "; ".join(reasons) + " (heuristic fallback — LLM unavailable)"
    }


def generate_predictions(data):
    """Generate AI predictions and store to SQLite DB"""
    print("[4/6] Generating predictions...")
    price = data["meta"]["price"]
    rsi = data["technical"]["rsi14"]
    narrative = data.get("research", {}).get("narrative_score", {})
    ob = data["orderbook"]
    tech = data["technical"]
    returns = data["returns"]

    prompt = f"""You are a quantitative crypto analyst. Predict HYPE price direction and targets for 6h, 12h, 24h horizons.

Current price: ${price}
RSI(14): {rsi}
Narrative score: {narrative.get('score', 'N/A')}/100 ({narrative.get('label', 'N/A')})
Orderbook imbalance: {ob['imbalance']} ({ob['pressure']})
MACD hist: {tech['macd_hist']} (line: {tech['macd_line']}, signal: {tech['macd_signal']})
Returns: 1d {returns['1d']}%, 7d {returns['7d']}%, 30d {returns['30d']}%
ATR(14): {tech['atr14']} (${tech['atr_pct']}%)
BB upper: ${tech['bb_upper']}, lower: ${tech['bb_lower']}
Funding (8h): {tech.get('funding_8h', 'N/A')}
Volume regime: {tech.get('vol_regime', 'N/A')}

Respond ONLY with valid JSON in this exact format:
{{
  "pred_6h_direction": "BULLISH|BEARISH",
  "pred_6h_target": float,
  "pred_12h_direction": "BULLISH|BEARISH",
  "pred_12h_target": float,
  "pred_24h_direction": "BULLISH|BEARISH",
  "pred_24h_target": float,
  "confidence": int (1-10),
  "reasoning": "string"
}}

Rules:
- Targets must be realistic: roughly 0.2-0.5× ATR for 6h, 0.4-0.8× ATR for 12h, 0.8-1.5× ATR for 24h
- CONSERVATIVE targets — better to hit a small target than miss a big one
- BULLISH target > current price, BEARISH target < current price
- Confidence 1-10 scale based on signal alignment
- In reasoning, explain WHY each target was chosen: cite specific indicator levels (RSI, MACD, ATR, BB, funding) and what they imply about momentum, support/resistance, and timeframe
- Keep reasoning to 1-2 sentences per timeframe (under 300 chars total)"""

    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": PREDICTION_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"num_predict": 1200, "temperature": 0.2}
        }, timeout=120)
        result = resp.json()
        content = result.get("message", {}).get("content", "")
        
        if not content:
            print(f"  Empty response from model, result keys: {list(result.keys())}")
            print(f"  Message object: {result.get('message', {})}")
            # Fall back to a simple heuristic prediction
            pred = generate_heuristic_prediction(price, tech, returns)
        else:
            # Extract JSON from response — be more robust
            json_str = content.strip()
            
            # Remove markdown code fences
            if "```json" in json_str:
                parts = json_str.split("```json")
                for part in parts[1:]:
                    fence_end = part.find("```")
                    if fence_end != -1:
                        candidate = part[:fence_end].strip()
                        if candidate.startswith("{") and candidate.endswith("}"):
                            json_str = candidate
                            break
            elif "```" in json_str:
                parts = json_str.split("```")
                for part in parts[1:]:
                    if part.strip().startswith("{") and part.strip().endswith("}"):
                        json_str = part.strip()
                        break
            
            # Try to find JSON directly if no fences
            if not json_str.startswith("{"):
                start = content.find("{")
                end = content.rfind("}")
                if start != -1 and end != -1 and end > start:
                    json_str = content[start:end+1]

            if not json_str or not json_str.startswith("{"):
                print(f"  Could not extract JSON, using heuristic fallback")
                pred = generate_heuristic_prediction(price, tech, returns)
            else:
                pred = json.loads(json_str)
                print(f"  Parsed: 6h={pred.get('pred_6h_target')} 12h={pred.get('pred_12h_target')} 24h={pred.get('pred_24h_target')} reason={pred.get('reasoning', '')[:60]}")

        # Validate and store
        db_path = os.path.expanduser("~/Projects/hype-monitor/data/predictions.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                price_at_prediction REAL NOT NULL,
                pred_6h_direction TEXT NOT NULL,
                pred_6h_target REAL,
                pred_12h_direction TEXT NOT NULL,
                pred_12h_target REAL,
                pred_24h_direction TEXT NOT NULL,
                pred_24h_target REAL,
                confidence INTEGER,
                reasoning TEXT,
                resolved_6h INTEGER DEFAULT 0,
                resolved_12h INTEGER DEFAULT 0,
                resolved_24h INTEGER DEFAULT 0,
                actual_price_6h REAL,
                actual_price_12h REAL,
                actual_price_24h REAL,
                resolved_at_6h TEXT,
                resolved_at_12h TEXT,
                resolved_at_24h TEXT
            )
        """)
        conn.execute("""
            INSERT INTO predictions (
                created_at, price_at_prediction,
                pred_6h_direction, pred_6h_target,
                pred_12h_direction, pred_12h_target,
                pred_24h_direction, pred_24h_target,
                confidence, reasoning
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.datetime.now(datetime.timezone.utc).isoformat(),
            price,
            pred["pred_6h_direction"], float(pred["pred_6h_target"]),
            pred["pred_12h_direction"], float(pred["pred_12h_target"]),
            pred["pred_24h_direction"], float(pred["pred_24h_target"]),
            int(pred.get("confidence", 5)),
            pred.get("reasoning", "")
        ))
        conn.commit()
        new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.close()

        print(f"  Stored prediction #{new_id}")
        return pred
    except Exception as e:
        print(f"  Prediction generation failed: {e}")
        return None

def get_prediction_stats():
    """Read prediction stats from SQLite DB"""
    db_path = os.path.expanduser("~/Projects/hype-monitor/data/predictions.db")
    if not os.path.exists(db_path):
        return None

    stats = {
        "total": 0,
        "win_rate_6h": None,
        "win_rate_12h": None,
        "win_rate_24h": None,
        "resolved_6h": 0,
        "resolved_12h": 0,
        "resolved_24h": 0,
    }

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        stats["total"] = total

        for tf in ["6h", "12h", "24h"]:
            col = f"resolved_{tf}"
            wins = conn.execute(f"SELECT COUNT(*) FROM predictions WHERE {col} = 1").fetchone()[0]
            losses = conn.execute(f"SELECT COUNT(*) FROM predictions WHERE {col} = -1").fetchone()[0]
            resolved = wins + losses
            stats[f"resolved_{tf}"] = resolved
            if resolved > 0:
                stats[f"win_rate_{tf}"] = round(wins / resolved * 100, 1)

        # Active prediction (most recent)
        cursor = conn.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 1")
        active = cursor.fetchone()
        if active:
            stats["active"] = {k: active[k] for k in active.keys()}

        # Recent predictions (last 5)
        cursor = conn.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 5")
        stats["recent"] = [
            {k: row[k] for k in row.keys()}
            for row in cursor.fetchall()
        ]

    # Calculate streak
    streak = 0
    if stats.get("recent"):
        for p in stats["recent"]:
            # Use 6h resolution for streak
            res = p.get("resolved_6h")
            if res == 1:
                streak = streak + 1 if streak >= 0 else 1
            elif res == -1:
                streak = streak - 1 if streak <= 0 else -1
            else:
                break
    stats["streak"] = streak

    return stats

# ── Changelog Tracking ──

def track_changelog(data, output_dir):
    """Track signal changes between runs"""
    print("[5/6] Tracking changelog...")
    changelog_path = os.path.join(output_dir, "changelog.json")
    prev_data = None
    if os.path.exists(changelog_path):
        try:
            with open(changelog_path) as f:
                prev_entries = json.load(f)
                if prev_entries and len(prev_entries) > 0:
                    # Get last state from previous hype_data.json if available
                    data_path = os.path.join(output_dir, "hype_data.json")
                    if os.path.exists(data_path):
                        with open(data_path) as df:
                            prev_data = json.load(df)
        except Exception:
            pass

    entries = []
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    price = data["meta"]["price"]

    if prev_data:
        ptech = prev_data.get("technical", {})
        pbook = prev_data.get("orderbook", {})
        pfund = prev_data.get("funding", {})
        tech = data["technical"]
        book = data["orderbook"]
        fund = data["funding"]

        # RSI signal change
        if ptech.get("rsi_signal") != tech.get("rsi_signal"):
            entries.append({
                "timestamp": now,
                "type": "signal_flip",
                "category": "rsi_regime",
                "from": ptech.get("rsi_signal"),
                "to": tech.get("rsi_signal"),
                "price": price,
                "detail": f"RSI regime: {ptech.get('rsi_signal')} → {tech.get('rsi_signal')} (RSI {tech.get('rsi14')})"
            })

        # Orderbook pressure change
        if pbook.get("pressure") != book.get("pressure"):
            entries.append({
                "timestamp": now,
                "type": "signal_flip",
                "category": "orderbook_pressure",
                "from": pbook.get("pressure"),
                "to": book.get("pressure"),
                "price": price,
                "detail": f"Imbalance: {book.get('imbalance')}"
            })

        # MACD crossover
        p_macd = ptech.get("macd_hist")
        c_macd = tech.get("macd_hist")
        if p_macd is not None and c_macd is not None:
            if p_macd <= 0 and c_macd > 0:
                entries.append({
                    "timestamp": now,
                    "type": "signal_flip",
                    "category": "macd_crossover",
                    "from": "BEARISH",
                    "to": "BULLISH",
                    "price": price,
                    "detail": f"MACD histogram crossed positive ({c_macd})"
                })
            elif p_macd >= 0 and c_macd < 0:
                entries.append({
                    "timestamp": now,
                    "type": "signal_flip",
                    "category": "macd_crossover",
                    "from": "BULLISH",
                    "to": "BEARISH",
                    "price": price,
                    "detail": f"MACD histogram crossed negative ({c_macd})"
                })

        # Funding spike
        p_fund = pfund.get("latest_8h")
        c_fund = fund.get("latest_8h")
        if p_fund is not None and c_fund is not None:
            if abs(c_fund) > abs(p_fund) * 2 and abs(c_fund) > 0.05:
                entries.append({
                    "timestamp": now,
                    "type": "signal_change",
                    "category": "funding_spike",
                    "from": str(p_fund),
                    "to": str(c_fund),
                    "price": price,
                    "detail": f"Funding spiked to {c_fund}%"
                })

        # Price milestone — 30d high
        prev_closes = prev_data.get("timeseries", {}).get("closes", [])
        if isinstance(prev_closes, list) and len(prev_closes) > 0:
            try:
                prev_max = max([float(c) for c in prev_closes if c is not None and c != "NaN"])
                if price > prev_max:
                    entries.append({
                        "timestamp": now,
                        "type": "price_milestone",
                        "category": "30d_high",
                        "from": str(round(prev_max, 2)),
                        "to": str(price),
                        "price": price,
                        "detail": f"New 30-day high: ${price}"
                    })
            except Exception:
                pass

    # Load existing changelog
    existing = []
    if os.path.exists(changelog_path):
        try:
            with open(changelog_path) as f:
                existing = json.load(f)
        except Exception:
            pass

    all_entries = existing + entries
    # Keep last 200 entries
    all_entries = all_entries[-200:]

    with open(changelog_path, "w") as f:
        json.dump(all_entries, f, indent=2)

    print(f"  Changelog: {len(entries)} new entries, {len(all_entries)} total")
    return all_entries

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

    # ATR (trs[j] = True Range for candle index j+1)
    atr7 = np.full_like(closes, np.nan, dtype=float)
    atr14 = np.full_like(closes, np.nan, dtype=float)
    trs = np.maximum(highs[1:] - lows[1:], np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])))
    for i in range(7, len(closes)):
        atr7[i] = np.mean(trs[i-7:i])   # 7 elements
    for i in range(14, len(closes)):
        atr14[i] = np.mean(trs[i-14:i])  # 14 elements

    # BTC correlation
    btc_closes = np.array([float(c["c"]) for c in btc_daily])
    btc_corr_30 = round(np.corrcoef(closes[-30:], btc_closes[-30:])[0,1], 3) if len(closes) >= 30 and len(btc_closes) >= 30 else None

    # Multi-asset comparison (HYPE vs BTC/ETH/SOL/BNB/XRP)
    comparison = build_comparison_data(closes)

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

    # Multi-timeframe OHLCV candles for Lightweight Charts
    print("[1b/6] Fetching multi-timeframe candles...")
    def build_candle_data(interval, days_back):
        candles = fetch_candles(interval, days_back)
        result = []
        for c in candles:
            result.append({
                "time": int(c["t"] // 1000),
                "open": float(c["o"]),
                "high": float(c["h"]),
                "low": float(c["l"]),
                "close": float(c["c"]),
                "volume": float(c["v"]),
            })
        return result

    candles_1h = build_candle_data("1h", 30)
    candles_4h = build_candle_data("4h", 90)
    candles_1d = build_candle_data("1d", 120)

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

    # Add HYPE self-reference to comparison data
    hype_norm_start = closes[-30] if len(closes) >= 30 else closes[0]
    hype_norm_days = min(30, len(closes))
    comparison["HYPE"] = {
        "price": round(closes[-1], 2),
        "returns": {k: returns.get(k) for k in ["1d", "7d", "30d"]},
        "correlation_hype_30d": 1.0,
        "normalized_30d": [round(((c - hype_norm_start) / hype_norm_start) * 100, 2) for c in closes[-hype_norm_days:]],
        "dates_30d": dates[-hype_norm_days:],
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

    technical = {
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
        "vol_regime": vol_regime,
        "dist_sma7_pct": dist_sma7,
        "dist_sma20_pct": dist_sma20,
        "dist_sma50_pct": dist_sma50,
        "stop_2x": sl_2x,
        "stop_3x": sl_3x,
        "poc": poc,
    }

    research = build_research_data(current_price, ctx, technical, funding_summary, {
        "imbalance": imbalance,
        "pressure": "BULLISH" if imbalance > 0.1 else ("BEARISH" if imbalance < -0.1 else "NEUTRAL"),
        "spread_bps": spread_bps,
    }, btc_corr_30)

    return {
        "meta": {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "data_points": n,
            "price": current_price,
            "mark_price": mark,
            "open_interest": str(oi) if oi else "?",
        },
        "returns": returns,
        "technical": technical,
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
        "research": research,
        "timeseries": {
            "dates": dates,
            "closes": closes.tolist(),
            "volumes": volumes.tolist(),
            "rsi": [round(x, 1) if not np.isnan(x) else None for x in rsi14.tolist()],
            "sma7": [round(x, 2) if not np.isnan(x) else None for x in sma7.tolist()],
            "sma20": [round(x, 2) if not np.isnan(x) else None for x in sma20.tolist()],
            "sma50": [round(x, 2) if not np.isnan(x) else None for x in sma50.tolist()],
            "bb_upper": [round(x, 2) if not np.isnan(x) else None for x in bb_upper.tolist()],
            "bb_lower": [round(x, 2) if not np.isnan(x) else None for x in bb_lower.tolist()],
            "macd": [round(x, 4) if not np.isnan(x) else None for x in macd_line.tolist()],
            "macd_signal": [round(x, 4) if not np.isnan(x) else None for x in macd_signal.tolist()],
        },
        "candles": {
            "1h": candles_1h,
            "4h": candles_4h,
            "1d": candles_1d,
        },
        "comparison": comparison,
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
    research = data.get("research", {})
    narrative = research.get("narrative_score", {})
    ps = research.get("protocol_stats", {})
    fees = research.get("fees", {})
    valn = research.get("valuation", {})

    prompt = f"""You are a senior crypto trader. Based on this live HYPE/USD data, write a brief multi-factor analysis.

CONTEXT: HYPE (Hyperliquid token) is at ${price}. It is a perp DEX token with an aggressive buyback engine (97% of fees buyback HYPE). It's the dominant perp DEX with 32-44% market share. Recent ETF approval, HIP-3 tokenized stocks, and institutional accumulation (a16z $170M+) are key drivers.

PROTOCOL FUNDAMENTALS:
- TVL: ${fmt_large(ps.get('tvl'))} ({fmt_pct(ps.get('tvl_30d_change_pct'))} 30d)
- Market Cap: ${fmt_large(ps.get('mcap'))}
- Fees 24h: ${fmt_large(fees.get('fees_24h'))}
- All-Time Fees: ${fmt_large(fees.get('fees_all_time'))}
- Mcap/TVL: {valn.get('mcap_tvl_ratio')}
- Price/Daily Revenue: {valn.get('price_to_daily_revenue')}
- Narrative Score: {narrative.get('score')}/100 ({narrative.get('label')})

TECHNICAL DATA:
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

**PROTOCOL FUNDAMENTALS:** [key TVL/fee/volume trend + buyback engine impact]
**VALUATION:** [mcap/tvl, price/revenue assessment — over/under/fair]
**NARRATIVE THESIS:** [the big story driving price, institutional or retail?]
**WHAT CHANGED:** [most significant change since last run]
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

def fmt_large(val):
    if val is None: return "N/A"
    if val >= 1e9:
        return f"${val/1e9:.2f}B"
    if val >= 1e6:
        return f"${val/1e6:.2f}M"
    if val >= 1e3:
        return f"${val/1e3:.2f}K"
    return f"${val:.2f}"

def fmt_pct(val):
    if val is None: return "N/A"
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:.1f}%"

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

    # Generate and store predictions
    pred = generate_predictions(data)
    pred_stats = get_prediction_stats()
    if pred_stats:
        # Update hype_data.json with predictions
        data_path = os.path.join(output_dir, "hype_data.json")
        try:
            with open(data_path, "r") as f:
                payload = json.load(f)
            payload["predictions"] = pred_stats
            with open(data_path, "w") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            print(f"  Failed to update predictions in hype_data.json: {e}")

    # Track changelog
    changelog = track_changelog(data, output_dir)
    if changelog:
        data_path = os.path.join(output_dir, "hype_data.json")
        try:
            with open(data_path, "r") as f:
                payload = json.load(f)
            payload["changelog"] = changelog
            with open(data_path, "w") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            print(f"  Failed to update changelog in hype_data.json: {e}")

    # Add ai_meta
    data_path = os.path.join(output_dir, "hype_data.json")
    try:
        with open(data_path, "r") as f:
            payload = json.load(f)
        payload["ai_meta"] = {
            "model": OLLAMA_MODEL,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        with open(data_path, "w") as f:
            json.dump(payload, f, indent=2)
    except Exception as e:
        print(f"  Failed to update ai_meta in hype_data.json: {e}")

    print("Done.")

if __name__ == "__main__":
    main()
