import os
import asyncio
import json
import logging
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pandas as pd
import requests
from google import genai
from google.genai import types
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes


# ============================================================
# CONFIG
# ============================================================

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

OKX_BASE_URL = "https://www.okx.com"
OKX_TICKER_URL = f"{OKX_BASE_URL}/api/v5/market/ticker"
OKX_CANDLES_URL = f"{OKX_BASE_URL}/api/v5/market/candles"
OKX_BOOKS_URL = f"{OKX_BASE_URL}/api/v5/market/books"
OKX_FUNDING_URL = f"{OKX_BASE_URL}/api/v5/public/funding-rate"
OKX_OPEN_INTEREST_URL = f"{OKX_BASE_URL}/api/v5/public/open-interest"
# V4.1: fallback chain inside EACH Gemini job.
# A model failure never stops the job immediately: the next model is tried.
# Only after every model fails does the job become ERROR and pipeline moves on.
_DEFAULT_MODEL_CHAIN = [
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
]
AGENT_MODEL_CHAIN = [
    x.strip()
    for x in os.getenv("AGENT_MODEL_CHAIN", ",".join(_DEFAULT_MODEL_CHAIN)).split(",")
    if x.strip()
]
# Backward compatibility for older code paths/commands.
AGENT_MODEL = AGENT_MODEL_CHAIN[0] if AGENT_MODEL_CHAIN else "gemini-3.6-flash"

gemini_client = (
    genai.Client(
        api_key=GEMINI_API_KEY,
        # google-genai HttpOptions.timeout is milliseconds.
        # This prevents one HTTP request from waiting indefinitely.
        http_options=types.HttpOptions(timeout=30000),
    )
    if GEMINI_API_KEY
    else None
)


# ============================================================
# HELPERS
# ============================================================

def normalize_symbol(args):
    """Normalize command arguments safely to OKX SPOT format."""
    if isinstance(args, (list, tuple)):
        raw = args[0] if args else "BTC-USDT"
    else:
        raw = args or "BTC-USDT"
    symbol = str(raw).strip().upper().replace("/", "-").replace("_", "-").replace(" ", "")
    if "-" in symbol:
        parts = [x for x in symbol.split("-") if x]
        if len(parts) == 2:
            return f"{parts[0]}-{parts[1]}"
    for quote in ("USDT", "USDC", "USD", "BTC", "ETH"):
        if symbol.endswith(quote) and len(symbol) > len(quote):
            return f"{symbol[:-len(quote)]}-{quote}"
    return symbol

def okx_get(url, params):
    response = requests.get(
        url,
        params=params,
        timeout=15,
    )
    response.raise_for_status()

    payload = response.json()

    if payload.get("code") != "0":
        raise RuntimeError(
            f"OKX error {payload.get('code')}: "
            f"{payload.get('msg')}"
        )

    return payload.get("data", [])


def get_ticker(symbol):
    data = okx_get(
        OKX_TICKER_URL,
        {"instId": symbol},
    )

    if not data:
        raise RuntimeError("OKX không trả dữ liệu ticker.")

    item = data[0]

    return {
        "symbol": symbol,
        "last": float(item.get("last", 0) or 0),
        "bid": float(item.get("bidPx", 0) or 0),
        "ask": float(item.get("askPx", 0) or 0),
        "high24h": float(item.get("high24h", 0) or 0),
        "low24h": float(item.get("low24h", 0) or 0),
        "vol24h": float(item.get("vol24h", 0) or 0),
        "ts": item.get("ts"),
    }


def get_completed_candles(symbol, bar, limit=300):
    data = okx_get(
        OKX_CANDLES_URL,
        {
            "instId": symbol,
            "bar": bar,
            "limit": str(limit),
        },
    )

    rows = []

    for candle in data:
        # OKX candle:
        # ts,o,h,l,c,vol,volCcy,volCcyQuote,confirm
        if len(candle) < 9:
            continue

        if str(candle[8]) != "1":
            continue

        rows.append(
            {
                "ts": int(candle[0]),
                "open": float(candle[1]),
                "high": float(candle[2]),
                "low": float(candle[3]),
                "close": float(candle[4]),
                "volume": float(candle[5]),
            }
        )

    rows.sort(key=lambda x: x["ts"])

    if len(rows) < 210:
        raise RuntimeError(
            f"Không đủ nến hoàn tất cho {symbol} {bar}. "
            f"Chỉ có {len(rows)} nến."
        )

    return pd.DataFrame(rows)


def calculate_indicators(df):
    df = df.copy()

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # EMA
    df["ema20"] = close.ewm(
        span=20,
        adjust=False,
    ).mean()

    df["ema50"] = close.ewm(
        span=50,
        adjust=False,
    ).mean()

    df["ema200"] = close.ewm(
        span=200,
        adjust=False,
    ).mean()

    # RSI 14 - Wilder style
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14,
    ).mean()

    rs = avg_gain / avg_loss.replace(0, float("nan"))
    df["rsi14"] = 100 - (100 / (1 + rs))

    # MACD 12/26/9
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(
        span=9,
        adjust=False,
    ).mean()
    df["macd_hist"] = (
        df["macd"] - df["macd_signal"]
    )

    # Bollinger 20, 2
    df["bb_mid"] = close.rolling(20).mean()
    bb_std = close.rolling(20).std(ddof=0)
    df["bb_upper"] = df["bb_mid"] + 2 * bb_std
    df["bb_lower"] = df["bb_mid"] - 2 * bb_std

    # ATR 14 - Wilder style
    previous_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["atr14"] = tr.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14,
    ).mean()

    # Volume
    df["volume_ma20"] = volume.rolling(20).mean()

    # TSI 25/13 + signal 7
    momentum = close.diff()
    abs_momentum = momentum.abs()

    momentum_ema25 = momentum.ewm(
        span=25,
        adjust=False,
    ).mean()
    momentum_double = momentum_ema25.ewm(
        span=13,
        adjust=False,
    ).mean()

    abs_ema25 = abs_momentum.ewm(
        span=25,
        adjust=False,
    ).mean()
    abs_double = abs_ema25.ewm(
        span=13,
        adjust=False,
    ).mean()

    df["tsi"] = (
        100
        * momentum_double
        / abs_double.replace(0, float("nan"))
    )

    df["tsi_signal"] = df["tsi"].ewm(
        span=7,
        adjust=False,
    ).mean()

    df["tsi_hist"] = (
        df["tsi"] - df["tsi_signal"]
    )

    return df


def clean_number(value):
    if pd.isna(value):
        return None
    return round(float(value), 8)


def timeframe_analysis(symbol, bar):
    df = get_completed_candles(symbol, bar)
    df = calculate_indicators(df)

    last = df.iloc[-1]

    return {
        "bar": bar,
        "timestamp": int(last["ts"]),
        "open": clean_number(last["open"]),
        "high": clean_number(last["high"]),
        "low": clean_number(last["low"]),
        "close": clean_number(last["close"]),
        "volume": clean_number(last["volume"]),
        "volume_ma20": clean_number(last["volume_ma20"]),
        "rsi14": clean_number(last["rsi14"]),
        "ema20": clean_number(last["ema20"]),
        "ema50": clean_number(last["ema50"]),
        "ema200": clean_number(last["ema200"]),
        "macd": clean_number(last["macd"]),
        "macd_signal": clean_number(last["macd_signal"]),
        "macd_hist": clean_number(last["macd_hist"]),
        "bb_mid": clean_number(last["bb_mid"]),
        "bb_upper": clean_number(last["bb_upper"]),
        "bb_lower": clean_number(last["bb_lower"]),
        "atr14": clean_number(last["atr14"]),
        "tsi": clean_number(last["tsi"]),
        "tsi_signal": clean_number(last["tsi_signal"]),
        "tsi_hist": clean_number(last["tsi_hist"]),
    }


def get_market_analysis(symbol):
    return {
        "symbol": symbol,
        "15m": timeframe_analysis(symbol, "15m"),
        "1H": timeframe_analysis(symbol, "1H"),
        "4H": timeframe_analysis(symbol, "4H"),
    }


def extract_json(text: str) -> dict[str, Any]:
    if not text:
        raise ValueError("Gemini trả về nội dung rỗng.")

    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"\s*```$",
            "",
            cleaned,
        )

    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start == -1 or end == -1 or end <= start:
            raise ValueError(
                "Không tìm thấy JSON hợp lệ từ Gemini."
            )

        obj = json.loads(cleaned[start:end + 1])

    if not isinstance(obj, dict):
        raise ValueError("JSON Gemini không phải object.")

    return obj


def validate_ai_result(result):
    decision = str(
        result.get("decision", "HOLD")
    ).upper()

    if decision not in {"BUY", "SELL", "HOLD"}:
        decision = "HOLD"

    try:
        confidence = int(result.get("confidence", 0))
    except Exception:
        confidence = 0

    confidence = max(0, min(100, confidence))

    result["decision"] = decision
    result["confidence"] = confidence

    if not isinstance(result.get("reasoning"), list):
        result["reasoning"] = []

    if not isinstance(result.get("risk_notes"), list):
        result["risk_notes"] = []

    return result


def build_ai_prompt(symbol, market):
    return f"""
Bạn là bộ phận phân tích thị trường của một hệ thống
giao dịch Spot thử nghiệm.

Bạn KHÔNG được đặt lệnh, quyết định kích thước vị thế,
hoặc thay đổi Risk Engine.

Phân tích {symbol} dựa CHỈ trên dữ liệu số được cung cấp.
Không tự bịa thêm mức giá hay chỉ báo.

4H là xu hướng chính.
1H là xu hướng trung gian.
15m là động lượng ngắn hạn.

Xem xét RSI, EMA20/50/200, MACD, Bollinger Bands,
ATR, Volume và TSI.

Nếu dữ liệu xung đột hoặc tín hiệu không rõ,
ưu tiên HOLD.

Dữ liệu:
{json.dumps(market, ensure_ascii=False)}

Chỉ trả JSON hợp lệ:
{{
  "decision": "BUY hoặc SELL hoặc HOLD",
  "confidence": 0,
  "market_regime": "",
  "reasoning": [],
  "risk_notes": [],
  "invalidation": ""
}}
"""


def ask_gemini(symbol, market):
    if not GEMINI_API_KEY or gemini_client is None:
        raise RuntimeError("Chưa cấu hình GEMINI_API_KEY.")

    response = gemini_client.models.generate_content(
        model="gemini-3.6-flash",
        contents=build_ai_prompt(symbol, market),
    )

    result = extract_json(response.text)
    return validate_ai_result(result)


# ============================================================
# ADVANCED MULTI-AGENT CORE — PAPER ONLY
# AI has NO authority over money, sizing, SL/TP, or hard risk limits.
# ============================================================

AGENT_MEMORY = []
AGENT_MEMORY_MAX = 100
AGENT_REFLECTIONS = []
AGENT_REFLECTIONS_MAX = 50


def _memory_context(symbol, limit=5):
    """Return compact prior-cycle context for Critic/Master only.

    Memory can inform AI reasoning, but never controls sizing, SL/TP or execution.
    """
    rows = [x for x in AGENT_MEMORY if x.get("symbol") == symbol][-limit:]
    reflections = [x for x in AGENT_REFLECTIONS if x.get("symbol") == symbol][-3:]
    return {"recent_cycles": rows, "recent_reflections": reflections}


def _record_agent_memory(bundle, validator_warnings=None, risk=None, execution=None):
    master = bundle.get("master") or {}
    critic = bundle.get("critic") or {}
    reports = bundle.get("specialists") or []
    row = {
        "timestamp_ms": bundle.get("timestamp_ms", _ms()),
        "symbol": bundle.get("symbol"),
        "decision": master.get("decision", "HOLD"),
        "confidence": int(master.get("confidence", 0) or 0),
        "agreement": critic.get("agreement", "LOW"),
        "caution": critic.get("recommended_caution", "HIGH"),
        "biases": {r.get("agent"): r.get("bias") for r in reports},
        "validator_warnings": list(validator_warnings or [])[:5],
        "risk_allowed": bool((risk or {}).get("allowed", False)),
        "risk_reason": str((risk or {}).get("reason", ""))[:160],
        "execution": str(execution or "NO ACTION")[:80],
    }
    AGENT_MEMORY.append(row)
    if len(AGENT_MEMORY) > AGENT_MEMORY_MAX:
        del AGENT_MEMORY[:-AGENT_MEMORY_MAX]
    return row


def _reflect_cycle(bundle, memory_row):
    """Deterministic post-cycle reflection; advisory context only."""
    critic = bundle.get("critic") or {}
    master = bundle.get("master") or {}
    lessons = []
    if critic.get("conflicts"):
        lessons.append("Giữ thận trọng khi specialist còn mâu thuẫn.")
    if critic.get("needs_more_data") or master.get("needs_more_data"):
        lessons.append("Thiếu dữ liệu: ưu tiên HOLD cho đến khi bằng chứng đầy đủ hơn.")
    if memory_row.get("validator_warnings"):
        lessons.append("Validator phát hiện điểm cần kiểm chứng; không tăng confidence theo AI.")
    if not memory_row.get("risk_allowed"):
        lessons.append("Risk Engine đã BLOCK; AI không được vượt quyền ở cycle sau.")
    if not lessons:
        lessons.append("Không có cảnh báo mới; tiếp tục đánh giá độc lập theo dữ liệu cycle kế tiếp.")
    row = {
        "timestamp_ms": _ms(),
        "symbol": bundle.get("symbol"),
        "decision": master.get("decision", "HOLD"),
        "lessons": lessons[:4],
    }
    AGENT_REFLECTIONS.append(row)
    if len(AGENT_REFLECTIONS) > AGENT_REFLECTIONS_MAX:
        del AGENT_REFLECTIONS[:-AGENT_REFLECTIONS_MAX]
    return row


def _f(v, default=None):
    try:
        return float(v)
    except Exception:
        return default


def _ms():
    return int(time.time() * 1000)


def spot_to_swap(symbol):
    return f"{symbol}-SWAP"


def get_orderbook_snapshot(symbol, depth=50):
    data = okx_get(OKX_BOOKS_URL, {"instId": symbol, "sz": str(depth)})
    if not data:
        raise RuntimeError("Order book unavailable")
    x = data[0]
    bids=[(_f(r[0],0),_f(r[1],0)) for r in (x.get('bids') or []) if len(r)>=2]
    asks=[(_f(r[0],0),_f(r[1],0)) for r in (x.get('asks') or []) if len(r)>=2]
    if not bids or not asks:
        raise RuntimeError("Order book missing sides")
    bb,ba=bids[0][0],asks[0][0]
    mid=(bb+ba)/2
    bq=sum(q for _,q in bids); aq=sum(q for _,q in asks)
    total=bq+aq
    return {
        "source":"OKX_PUBLIC_ORDERBOOK","symbol":symbol,
        "timestamp_ms":int(x.get('ts') or _ms()),
        "best_bid":bb,"best_ask":ba,"mid":mid,
        "spread_bps":round((ba-bb)/mid*10000,4) if mid else 0,
        "bid_qty_depth":round(bq,8),"ask_qty_depth":round(aq,8),
        "imbalance":round((bq-aq)/total,6) if total else 0,
        "depth_levels":min(len(bids),len(asks)),"data_quality":"GOOD"
    }


def get_derivatives_snapshot(symbol):
    swap=spot_to_swap(symbol)
    out={"source":"OKX_PUBLIC_DERIVATIVES","swap_symbol":swap,
         "timestamp_ms":_ms(),"funding_rate":None,"next_funding_time":None,
         "open_interest":None,"open_interest_ccy":None,
         "data_quality":"GOOD","errors":[]}
    try:
        d=okx_get(OKX_FUNDING_URL,{"instId":swap})
        if d:
            out['funding_rate']=_f(d[0].get('fundingRate'))
            out['next_funding_time']=d[0].get('nextFundingTime')
    except Exception as e:
        out['errors'].append('funding:'+type(e).__name__)
    try:
        d=okx_get(OKX_OPEN_INTEREST_URL,{"instType":"SWAP","instId":swap})
        if d:
            out['open_interest']=_f(d[0].get('oi'))
            out['open_interest_ccy']=_f(d[0].get('oiCcy'))
    except Exception as e:
        out['errors'].append('oi:'+type(e).__name__)
    if out['errors']:
        out['data_quality']='PARTIAL' if (out['funding_rate'] is not None or out['open_interest'] is not None) else 'UNAVAILABLE'
    return out


def _gemini_generate_with_retry(prompt, label="GEMINI"):
    """
    V4.1 model fallback chain.

    Each job tries every configured Gemini model in order. A failure on one
    model (429/503/timeout/etc.) immediately falls through to the next model.
    The job is marked ERROR only when ALL models fail. The outer V4 pipeline
    then fail-forwards to the next job.
    """
    if not GEMINI_API_KEY or gemini_client is None:
        raise RuntimeError("Chưa cấu hình GEMINI_API_KEY.")
    if not AGENT_MODEL_CHAIN:
        raise RuntimeError("AGENT_MODEL_CHAIN rỗng.")

    failures = []
    total = len(AGENT_MODEL_CHAIN)
    last_exc = None

    for idx, model in enumerate(AGENT_MODEL_CHAIN, 1):
        _monitor_set(
            label, "CALLING",
            retry=f"model {idx}/{total}",
            detail=f"Đang thử {model}",
        )
        _touch_progress(f"{label}:CALLING:{model}")
        t0 = time.monotonic()
        try:
            result = gemini_client.models.generate_content(
                model=model,
                contents=prompt,
            )
            elapsed = (time.monotonic() - t0) * 1000
            failures.append(f"{model}=OK")
            _monitor_set(
                label, "OK",
                latency_ms=elapsed,
                retry=f"model {idx}/{total}",
                detail=(
                    f"Model thành công: {model}. "
                    f"Fallback: {' -> '.join(failures)}"
                ),
            )
            return result
        except Exception as exc:
            last_exc = exc
            elapsed = (time.monotonic() - t0) * 1000
            code = _error_http_code(exc) or "?"
            failures.append(f"{model}=ERR({code})")
            has_next = idx < total
            _monitor_set(
                label,
                "FALLBACK" if has_next else "ERROR",
                latency_ms=elapsed,
                error=type(exc).__name__,
                retry=f"model {idx}/{total}",
                detail=(
                    f"{model} lỗi HTTP {code}. "
                    + (f"Chuyển sang {AGENT_MODEL_CHAIN[idx]}. " if has_next else "Tất cả model đã lỗi. ")
                    + f"Chain: {' -> '.join(failures)}"
                ),
                http_code=code,
            )
            logging.warning(
                "%s model %s/%s (%s) failed HTTP %s: %s",
                label, idx, total, model, code, type(exc).__name__,
            )
            # No repeated retry of the same model. Move immediately to fallback.
            # This avoids wasting calls when a model returns quota/rate-limit 429.
            continue

    raise last_exc or RuntimeError(f"{label}: tất cả Gemini model đều lỗi")


def _unavailable_specialist(agent, reason):
    return {
        "agent": agent,
        "bias": "NEUTRAL",
        "confidence": 0,
        "summary": "DATA_UNAVAILABLE",
        "evidence": [],
        "risks": [reason],
        "data_quality": "POOR",
        "timestamp_ms": _ms(),
    }


def _specialist(agent, task, payload):
    prompt=f"""Bạn là {agent}, specialist của hệ thống Spot PAPER trading.
CHỈ dùng payload. Không bịa dữ liệu. Không quyết định tiền, size, risk %, SL/TP.
Không được sửa Risk Engine. Dữ liệu thiếu/xung đột => giảm confidence.
Nhiệm vụ: {task}
PAYLOAD: {json.dumps(payload,ensure_ascii=False)}
Chỉ JSON: {{"bias":"BULLISH|BEARISH|NEUTRAL|MIXED","confidence":0,"summary":"","evidence":[],"risks":[],"data_quality":"GOOD|PARTIAL|POOR"}}"""
    r=_gemini_generate_with_retry(prompt, agent)
    x=extract_json(r.text)
    bias=str(x.get('bias','NEUTRAL')).upper()
    if bias not in {'BULLISH','BEARISH','NEUTRAL','MIXED'}: bias='NEUTRAL'
    try: conf=max(0,min(100,int(x.get('confidence',0))))
    except Exception: conf=0
    return {"agent":agent,"bias":bias,"confidence":conf,
            "summary":str(x.get('summary','')),
            "evidence":x.get('evidence') if isinstance(x.get('evidence'),list) else [],
            "risks":x.get('risks') if isinstance(x.get('risks'),list) else [],
            "data_quality":str(x.get('data_quality','POOR')).upper(),"timestamp_ms":_ms()}


def _critic(symbol, reports):
    memory = _memory_context(symbol)
    prompt=f"""Bạn là CRITIC AGENT độc lập cho {symbol}. Vai trò của bạn là phản biện, không phải đồng thuận.
Chỉ dùng REPORTS và MEMORY_CONTEXT được cung cấp. Hãy:
1) đối chiếu Technical vs Orderbook vs Derivatives; 2) tìm bằng chứng ngược chiều;
3) phát hiện dữ liệu thiếu/cũ/yếu; 4) phát hiện confidence quá cao;
5) nêu rõ điều gì có thể làm luận điểm hiện tại sai; 6) ưu tiên HOLD khi bằng chứng chưa đủ.
Memory chỉ là lịch sử tham khảo, không được lấn át dữ liệu cycle hiện tại.
Không quyết định tiền/size/risk %/SL/TP và không được sửa Risk Engine.
REPORTS: {json.dumps(reports,ensure_ascii=False)}
MEMORY_CONTEXT: {json.dumps(memory,ensure_ascii=False)}
Chỉ JSON: {{"conflicts":[],"weak_points":[],"counter_evidence":[],"agreement":"HIGH|MEDIUM|LOW","needs_more_data":false,"recommended_caution":"LOW|MEDIUM|HIGH","summary":""}}"""
    r=_gemini_generate_with_retry(prompt, "CRITIC_AGENT")
    x=extract_json(r.text); x['agent']='CRITIC_AGENT'; x['timestamp_ms']=_ms()
    x['conflicts']=x.get('conflicts') if isinstance(x.get('conflicts'),list) else []
    x['weak_points']=x.get('weak_points') if isinstance(x.get('weak_points'),list) else []
    x['counter_evidence']=x.get('counter_evidence') if isinstance(x.get('counter_evidence'),list) else []
    x['needs_more_data']=bool(x.get('needs_more_data',False))
    return x


def _master(symbol,reports,critic):
    memory = _memory_context(symbol)
    prompt=f"""Bạn là MASTER DECISION AGENT cho Spot PAPER {symbol}.
Tổng hợp specialist SAU KHI đọc phản biện của Critic. Không được bỏ qua counter-evidence.
BUY/SELL/HOLD chỉ là quyết định HƯỚNG. SELL chỉ có nghĩa đóng BUY đang có, tuyệt đối không mở short.
Memory/reflection chỉ giúp nhận ra mẫu lặp lại; dữ liệu cycle hiện tại luôn ưu tiên hơn lịch sử.
Không quyết định tiền, size, risk %, SL/TP. Python Validator + Risk Engine có quyền ALLOW/BLOCK cuối cùng.
Nếu evidence xung đột/yếu/thiếu hoặc Critic yêu cầu thêm dữ liệu => ưu tiên HOLD và giảm confidence.
REPORTS: {json.dumps(reports,ensure_ascii=False)}
CRITIC: {json.dumps(critic,ensure_ascii=False)}
MEMORY_CONTEXT: {json.dumps(memory,ensure_ascii=False)}
Chỉ JSON: {{"decision":"BUY|SELL|HOLD","confidence":0,"market_regime":"","agreement":"HIGH|MEDIUM|LOW","reasoning":[],"risk_notes":[],"invalidation":"","needs_more_data":false}}"""
    r=_gemini_generate_with_retry(prompt, "MASTER_DECISION_AGENT")
    x=validate_ai_result(extract_json(r.text)); x['agent']='MASTER_DECISION_AGENT'; x['timestamp_ms']=_ms()
    x['needs_more_data']=bool(x.get('needs_more_data',False))
    return x


def run_advanced_agent(symbol, market=None):
    """
    Resilient Advanced Agent cycle.
    A failed specialist becomes UNAVAILABLE. Critic/Master failures fail closed
    to HOLD. AI never controls money, sizing, SL/TP or hard risk limits.
    """
    if not GEMINI_API_KEY or gemini_client is None:
        raise RuntimeError("Chưa cấu hình GEMINI_API_KEY.")

    market = market or get_market_analysis(symbol)

    try:
        book = _monitored_call("OKX_ORDERBOOK", get_orderbook_snapshot, symbol)
    except Exception as exc:
        logging.exception("ORDERBOOK ERROR")
        book = {
            "source": "OKX_PUBLIC_ORDERBOOK",
            "data_quality": "UNAVAILABLE",
            "error": type(exc).__name__,
        }

    deriv = _monitored_call("OKX_DERIVATIVES", get_derivatives_snapshot, symbol)
    reports = []

    specialist_jobs = [
        (
            "TECHNICAL_AGENT",
            "Phân tích 15m/1H/4H, EMA, RSI, MACD, Bollinger, ATR, volume, TSI.",
            market,
            True,
        ),
        (
            "ORDERBOOK_AGENT",
            "Phân tích spread, depth, imbalance và thanh khoản snapshot.",
            book,
            book.get("data_quality") != "UNAVAILABLE",
        ),
        (
            "DERIVATIVES_AGENT",
            "Phân tích funding và open interest. Không bịa liquidation/long-short ratio.",
            deriv,
            deriv.get("data_quality") != "UNAVAILABLE",
        ),
    ]

    for agent, task, payload, available in specialist_jobs:
        if not available:
            reports.append(_unavailable_specialist(agent, "Nguồn dữ liệu unavailable"))
            continue
        try:
            reports.append(_specialist(agent, task, payload))
        except Exception as exc:
            logging.exception("%s FAILED", agent)
            reports.append(
                _unavailable_specialist(
                    agent,
                    f"Gemini {type(exc).__name__} sau retry",
                )
            )

    usable = [
        r for r in reports
        if r.get("data_quality") != "POOR" and r.get("confidence", 0) > 0
    ]

    try:
        critic = _critic(symbol, reports)
    except Exception as exc:
        logging.exception("CRITIC_AGENT FAILED")
        critic = {
            "agent": "CRITIC_AGENT",
            "timestamp_ms": _ms(),
            "conflicts": [],
            "weak_points": [f"Critic unavailable: {type(exc).__name__}"],
            "agreement": "LOW",
            "needs_more_data": True,
            "recommended_caution": "HIGH",
            "summary": "CRITIC_UNAVAILABLE",
        }

    if not usable:
        master = {
            "agent": "MASTER_DECISION_AGENT",
            "timestamp_ms": _ms(),
            "decision": "HOLD",
            "confidence": 0,
            "market_regime": "UNKNOWN",
            "agreement": "LOW",
            "reasoning": ["Không có specialist AI khả dụng."],
            "risk_notes": ["Fail-closed: không giao dịch."],
            "invalidation": "",
            "needs_more_data": True,
        }
    else:
        try:
            master = _master(symbol, reports, critic)
        except Exception as exc:
            logging.exception("MASTER_DECISION_AGENT FAILED")
            master = {
                "agent": "MASTER_DECISION_AGENT",
                "timestamp_ms": _ms(),
                "decision": "HOLD",
                "confidence": 0,
                "market_regime": "UNKNOWN",
                "agreement": "LOW",
                "reasoning": [f"Master unavailable: {type(exc).__name__}"],
                "risk_notes": ["Fail-closed: Gemini lỗi, không giao dịch."],
                "invalidation": "",
                "needs_more_data": True,
            }

    if (
        critic.get("needs_more_data")
        or master.get("needs_more_data")
        or len(usable) < 2
    ):
        master["decision"] = "HOLD"
        master["confidence"] = min(master.get("confidence", 0), 69)
        master.setdefault("risk_notes", []).append(
            "Fail-closed: thiếu/không đủ dữ liệu specialist."
        )

    bundle = {
        "symbol": symbol,
        "timestamp_ms": _ms(),
        "market": market,
        "orderbook": book,
        "derivatives": deriv,
        "specialists": reports,
        "critic": critic,
        "master": master,
    }

    return bundle



async def run_advanced_agent_pipeline(application, chat_id, symbol, market):
    """V5.1 15-job pipeline. Jobs are fail-forward; Risk Engine remains outside AI."""
    reports = []
    book = {"source":"OKX_PUBLIC_ORDERBOOK","data_quality":"UNAVAILABLE"}
    deriv = {"source":"OKX_PUBLIC_DERIVATIVES","data_quality":"UNAVAILABLE"}

    def stage(n, name, state="RUNNING"):
        _touch_progress(f"V5:{n:02d}/15:{name}:{state}")

    def quant_engine():
        out = {}
        for tf in ("15m", "1H", "4H"):
            d = market.get(tf, {}) or {}
            try:
                close=float(d.get("close",0)); e20=float(d.get("ema20",0)); e50=float(d.get("ema50",0)); e200=float(d.get("ema200",0))
                rsi=float(d.get("rsi14",50)); mh=float(d.get("macd_hist",0)); tsi=float(d.get("tsi_hist",0))
                score = (1 if close>e20 else -1)+(1 if e20>e50 else -1)+(1 if e50>e200 else -1)+(1 if rsi>=50 else -1)+(1 if mh>0 else -1)+(1 if tsi>0 else -1)
                out[tf]={"score":score,"bias":"BULLISH" if score>=3 else "BEARISH" if score<=-3 else "MIXED"}
            except Exception:
                out[tf]={"score":0,"bias":"UNKNOWN"}
        return out

    def regime_engine(q):
        biases=[x.get("bias") for x in q.values()]
        if biases.count("BULLISH")>=2: regime="BULL_TREND"
        elif biases.count("BEARISH")>=2: regime="BEAR_TREND"
        elif "BULLISH" in biases and "BEARISH" in biases: regime="CONFLICTED"
        else: regime="RANGE_MIXED"
        return {"regime":regime,"timeframe_bias":biases,"source":"PYTHON_DETERMINISTIC"}

    # 1 MARKET DATA
    _job_stat_run(1); stage(1,"MARKET","OK"); _job_stat_result(1, True)
    # 2 QUANT ENGINE
    _job_stat_run(2); stage(2,"QUANT")
    quant = quant_engine(); stage(2,"QUANT","OK"); _job_stat_result(2, True)

    # 3 ORDERBOOK DATA
    _job_stat_run(3); stage(3,"ORDERBOOK_DATA")
    try:
        book = await asyncio.wait_for(asyncio.to_thread(_monitored_call,"OKX_ORDERBOOK",get_orderbook_snapshot,symbol), timeout=25)
        stage(3,"ORDERBOOK_DATA","OK"); _job_stat_result(3, True)
    except Exception as exc:
        book={"source":"OKX_PUBLIC_ORDERBOOK","data_quality":"UNAVAILABLE","error":type(exc).__name__}; stage(3,"ORDERBOOK_DATA","ERROR"); _job_stat_result(3, False)

    # 4 DERIVATIVES DATA
    _job_stat_run(4); stage(4,"DERIVATIVES_DATA")
    try:
        deriv = await asyncio.wait_for(asyncio.to_thread(_monitored_call,"OKX_DERIVATIVES",get_derivatives_snapshot,symbol), timeout=25)
        stage(4,"DERIVATIVES_DATA","OK"); _job_stat_result(4, True)
    except Exception as exc:
        deriv={"source":"OKX_PUBLIC_DERIVATIVES","data_quality":"UNAVAILABLE","error":type(exc).__name__}; stage(4,"DERIVATIVES_DATA","ERROR"); _job_stat_result(4, False)

    # 5 MARKET REGIME
    _job_stat_run(5); stage(5,"REGIME")
    regime = regime_engine(quant); stage(5,"REGIME","OK"); _job_stat_result(5, True)

    async def ai_job(n, label, fn, fallback):
        stage(n,label)
        try:
            x=await asyncio.wait_for(asyncio.to_thread(fn), timeout=45); stage(n,label,"OK"); return x
        except asyncio.TimeoutError:
            stage(n,label,"TIMEOUT"); return fallback("TIMEOUT") if callable(fallback) else fallback
        except Exception as exc:
            stage(n,label,"ERROR"); return fallback(type(exc).__name__) if callable(fallback) else fallback

    # 6 AI TECHNICAL
    _job_stat_run(6)
    technical=await ai_job(6,"AI_TECHNICAL",lambda:_specialist("TECHNICAL_AGENT","Phân tích kỹ thuật 15m/1H/4H. Bắt buộc đối chiếu số liệu; không bịa.",{"market":market,"quant":quant,"regime":regime}),lambda r:_unavailable_specialist("TECHNICAL_AGENT",r)); reports.append(technical); _job_stat_result(6, technical.get("data_quality") != "POOR")
    # 7 AI ORDERBOOK
    _job_stat_run(7)
    orderbook=await ai_job(7,"AI_MICROSTRUCTURE_ORDERBOOK",lambda:_specialist("ORDERBOOK_AGENT","Phân tích spread, depth, imbalance, liquidity. Không suy diễn dữ liệu thiếu.",book),lambda r:_unavailable_specialist("ORDERBOOK_AGENT",r)); reports.append(orderbook)
    # 8 AI DERIVATIVES / MICROSTRUCTURE
    micro=await ai_job(7,"AI_MICROSTRUCTURE_DERIVATIVES",lambda:_specialist("DERIVATIVES_AGENT","Phân tích funding, OI cùng orderbook. Không bịa liquidation/long-short ratio.",{"derivatives":deriv,"orderbook":book}),lambda r:_unavailable_specialist("DERIVATIVES_AGENT",r)); reports.append(micro); _job_stat_result(7, orderbook.get("data_quality") != "POOR" and micro.get("data_quality") != "POOR")

    # 9 CRITIC
    critic_fallback={"agent":"CRITIC_AGENT","timestamp_ms":_ms(),"conflicts":[],"weak_points":["CRITIC_UNAVAILABLE"],"agreement":"LOW","needs_more_data":True,"recommended_caution":"HIGH","summary":"CRITIC_UNAVAILABLE"}
    _job_stat_run(8)
    critic=await ai_job(8,"CRITIC",lambda:_critic(symbol,reports),critic_fallback)
    _job_stat_result(8, critic.get("summary") != "CRITIC_UNAVAILABLE")

    # 10 MASTER
    master_fallback={"agent":"MASTER_DECISION_AGENT","timestamp_ms":_ms(),"decision":"HOLD","confidence":0,"market_regime":"UNKNOWN","agreement":"LOW","reasoning":["Master unavailable; fail-safe HOLD."],"risk_notes":["Fail-closed: không giao dịch."],"invalidation":"","needs_more_data":True}
    _job_stat_run(9)
    master=await ai_job(9,"MASTER",lambda:_master(symbol,reports,critic),master_fallback)
    _job_stat_result(9, int(master.get("confidence",0) or 0) > 0)
    usable=[r for r in reports if r.get("data_quality")!="POOR" and r.get("confidence",0)>0]
    if critic.get("needs_more_data") or master.get("needs_more_data") or len(usable)<2:
        master["decision"]="HOLD"; master["confidence"]=min(int(master.get("confidence",0) or 0),69)
        master.setdefault("risk_notes",[]).append("Fail-closed: thiếu/không đủ dữ liệu specialist.")

    bundle={"symbol":symbol,"timestamp_ms":_ms(),"market":market,"quant":quant,"regime_engine":regime,"orderbook":book,"derivatives":deriv,"specialists":reports,"critic":critic,"master":master}
    stage(9,"MASTER","OK")
    return bundle

def format_agent_summary(bundle):
    if not bundle: return 'Chưa có Advanced Agent cycle.'
    lines=[f"• {r.get('agent')}: {r.get('bias')} {r.get('confidence')}%" for r in bundle.get('specialists',[])]
    c=bundle.get('critic',{}); m=bundle.get('master',{})
    lines.append(f"• CRITIC: agreement={c.get('agreement','N/A')} | caution={c.get('recommended_caution','N/A')}")
    lines.append(f"• MASTER: {m.get('decision','HOLD')} {m.get('confidence',0)}%")
    return '\n'.join(lines)


# ============================================================
# TELEGRAM BASIC COMMANDS
# ============================================================

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    text = (
        "🤖 OKX AI Agent V5.3 — 15 JOBS\n\n"
        "Lệnh:\n"
        "/status - trạng thái bot\n"
        "/price BTC-USDT - giá hiện tại\n"
        "/analyze BTC-USDT - chỉ báo 15m/1H/4H\n"
        "/ai BTC-USDT - phân tích Gemini đơn\n"
        "/agent BTC-USDT - Advanced Multi-Agent\n"
        "/signal BTC-USDT - AI + Risk Engine + paper entry\n"
        "/risk - cấu hình Risk Engine\n"
        "/position - xem/check paper position\n"
        "/closepaper - đóng paper position thủ công\n"
        "/auto BTC-USDT - BẬT tự động paper trading\n"
        "/autostatus - xem lãi/lỗ và trạng thái auto\n"
        "/performance - thống kê PAPER V5\n"
        "/jobstats - PASS/RUN của 15 jobs\n"
        "/stopauto - dừng tự động\n\n"
        "⚠️ Phiên bản này KHÔNG gửi lệnh tới OKX."
    )

    await update.message.reply_text(text)


async def status(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    gemini_status = (
        "OK"
        if GEMINI_API_KEY and gemini_client is not None
        else "CHƯA CẤU HÌNH"
    )

    await update.message.reply_text(
        "✅ Bot đang chạy.\n"
        f"Gemini: {gemini_status}\n"
        "OKX: public market data only\n"
        "Trading: PAPER ONLY"
    )


async def price(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    symbol = normalize_symbol(context.args)

    try:
        ticker = get_ticker(symbol)

        await update.message.reply_text(
            f"💰 {symbol}\n\n"
            f"Last: {ticker['last']:.4f}\n"
            f"Bid: {ticker['bid']:.4f}\n"
            f"Ask: {ticker['ask']:.4f}\n"
            f"24h High: {ticker['high24h']:.4f}\n"
            f"24h Low: {ticker['low24h']:.4f}"
        )

    except Exception as e:
        logging.exception("PRICE ERROR")
        await update.message.reply_text(
            "❌ Không lấy được giá.\n"
            f"Loại lỗi: {type(e).__name__}"
        )


async def analyze(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    symbol = normalize_symbol(context.args)

    await update.message.reply_text(
        f"📊 Đang phân tích {symbol}..."
    )

    try:
        market = get_market_analysis(symbol)

        parts = [f"📊 {symbol}"]

        for tf in ("15m", "1H", "4H"):
            d = market[tf]

            parts.append(
                f"\n⏱ {tf}\n"
                f"Close: {d['close']}\n"
                f"RSI14: {d['rsi14']}\n"
                f"EMA20: {d['ema20']}\n"
                f"EMA50: {d['ema50']}\n"
                f"EMA200: {d['ema200']}\n"
                f"MACD hist: {d['macd_hist']}\n"
                f"ATR14: {d['atr14']}\n"
                f"Volume: {d['volume']}\n"
                f"Volume MA20: {d['volume_ma20']}\n"
                f"TSI hist: {d['tsi_hist']}"
            )

        await update.message.reply_text(
            "\n".join(parts)
        )

    except Exception as e:
        logging.exception("ANALYZE ERROR")
        await update.message.reply_text(
            "❌ Không thể phân tích thị trường.\n"
            f"Loại lỗi: {type(e).__name__}"
        )


async def ai_analysis(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    symbol = normalize_symbol(context.args)

    await update.message.reply_text(
        f"🤖 Gemini đang phân tích {symbol}..."
    )

    try:
        market = get_market_analysis(symbol)
        result = ask_gemini(symbol, market)

        reasoning = result.get("reasoning", [])
        risks = result.get("risk_notes", [])

        reason_text = (
            "\n".join(f"• {x}" for x in reasoning)
            if reasoning
            else "• Không có."
        )

        risk_text = (
            "\n".join(f"• {x}" for x in risks)
            if risks
            else "• Không có."
        )

        await update.message.reply_text(
            f"🤖 AI PHÂN TÍCH {symbol}\n\n"
            f"Quyết định: {result['decision']}\n"
            f"Độ tin cậy: {result['confidence']}%\n"
            f"Trạng thái thị trường: "
            f"{result.get('market_regime', '')}\n\n"
            f"📊 Lý do:\n{reason_text}\n\n"
            f"⚠️ Rủi ro:\n{risk_text}\n\n"
            f"🛑 Điều kiện làm phân tích mất hiệu lực:\n"
            f"{result.get('invalidation', '')}\n\n"
            "ℹ️ AI chỉ phân tích; không được đặt lệnh."
        )

    except Exception as e:
        logging.exception("AI ERROR")
        await update.message.reply_text(
            "❌ Không thể chạy AI analysis.\n"
            f"Loại lỗi: {type(e).__name__}"
        )


# ============================================================
# RISK ENGINE V2
# ============================================================

RISK_CONFIG = {
    "sim_balance_usdt": 1000.0,
    "risk_per_trade_pct": 1.0,
    "daily_loss_limit_pct": None,
    "max_open_positions": 1,
    "min_confidence": 70,
    "stop_atr_multiplier": 1.5,
    "reward_risk_ratio": 2.0,
}

# Trạng thái RAM: Render restart sẽ reset.
SIM_STATE = {
    "daily_pnl_usdt": 0.0,
    "open_positions": 0,
}

PAPER_POSITION = None
TRADE_HISTORY = []

AUTO_STATE = {
    "running": False,
    "desired_running": False,
    "symbol": "BTC-USDT",
    "task": None,
    "watchdog_task": None,
    "status_task": None,
    "position_task": None,
    "cycle_count": 0,
    "validator_warnings": [],
    "validator_stats": {"runs": 0, "clean": 0, "contradictions": 0, "errors": 0, "claims_checked": 0, "claims_skipped": 0},
    "chat_id": None,
    "last_analyzed_15m_ts": None,
    "last_market": None,
    "last_ai_result": None,
    "last_risk_result": None,
    "last_ai_updated_at": None,
    "last_agent_bundle": None,
    "last_heartbeat_at": None,
    "last_progress_at": None,
    "last_progress_stage": None,
    "worker_restarts": 0,
    "last_worker_error": None,
    "trades": 0,
    "wins": 0,
    "losses": 0,
    "realized_pnl_usdt": 0.0,
    "job_stats": {},
}

AUTO_SCAN_SECONDS = 60
WORKER_STALE_SECONDS = 240
PROGRESS_STALE_SECONDS = 240
GEMINI_HTTP_TIMEOUT_MS = 30000
ADVANCED_CYCLE_TIMEOUT_SECONDS = 420
GEMINI_RETRY_ATTEMPTS = 3
GEMINI_RETRY_BASE_SECONDS = 2

API_MONITOR = {}

def _safe_error_detail(exc):
    try:
        text = str(exc)
    except Exception:
        text = type(exc).__name__
    text = re.sub(r'AIza[0-9A-Za-z_-]{20,}', '[REDACTED_API_KEY]', text)
    text = re.sub(r'(?i)(api[_ -]?key|x-goog-api-key|authorization|bearer|token|secret|passphrase)(\\s*[:=]\\s*)[^,\\s]+', r'\\1\\2[REDACTED]', text)
    text = re.sub(r'(?i)(key=)[^&\\s]+', r'\\1[REDACTED]', text)
    return " ".join(text.split())[:300]

def _error_http_code(exc):
    for attr in ("status_code", "code"):
        try:
            value = getattr(exc, attr, None)
            if value is not None:
                return str(value)
        except Exception:
            pass
    m = re.search(r'\\b(400|401|403|404|408|409|429|500|502|503|504)\\b', _safe_error_detail(exc))
    return m.group(1) if m else None

JOB_NAMES = {
    1:"MARKET DATA", 2:"QUANT ENGINE", 3:"ORDERBOOK DATA", 4:"DERIVATIVES DATA",
    5:"MARKET REGIME", 6:"AI TECHNICAL", 7:"AI MICROSTRUCTURE", 8:"CRITIC",
    9:"MASTER", 10:"PYTHON VALIDATOR", 11:"RISK ENGINE", 12:"PAPER EXECUTION",
    13:"MEMORY", 14:"REFLECTION", 15:"FINAL REPORT",
}

def _job_stat_run(n):
    row=AUTO_STATE["job_stats"].setdefault(n,{"run":0,"pass":0,"fail":0})
    row["run"] += 1

def _job_stat_result(n, ok):
    row=AUTO_STATE["job_stats"].setdefault(n,{"run":0,"pass":0,"fail":0})
    row["pass" if ok else "fail"] += 1

def format_job_stats():
    lines=["📈 JOB PASS / RUN"]
    for n in range(1,16):
        row=AUTO_STATE["job_stats"].get(n,{"run":0,"pass":0,"fail":0})
        run=row["run"]; passed=row["pass"]
        pct=(passed/run*100) if run else 0
        lines.append(f"{n:02d}. {JOB_NAMES[n]}: {passed}/{run} ({pct:.0f}%)")
    return "\n".join(lines)

def _touch_progress(stage=None):
    """V3: mark the worker as making progress from worker/helper threads."""
    now = time.time()
    AUTO_STATE["last_progress_at"] = now
    AUTO_STATE["last_heartbeat_at"] = now
    if stage:
        AUTO_STATE["last_progress_stage"] = str(stage)[:80]


def _monitor_set(name, status, *, latency_ms=None, error=None, retry=None, detail=None, http_code=None):
    _touch_progress(f"{name}:{status}")
    row = API_MONITOR.setdefault(name, {})
    row["status"] = status
    row["updated_at"] = time.time()
    if latency_ms is not None:
        row["latency_ms"] = int(latency_ms)
    if error is not None:
        row["error"] = str(error)[:80]
    elif status == "OK":
        row.pop("error", None)
    if detail is not None:
        row["detail"] = str(detail)[:300]
    elif status == "OK":
        row.pop("detail", None)
    if http_code is not None:
        row["http_code"] = str(http_code)[:16]
    elif status == "OK":
        row.pop("http_code", None)
    if retry is not None:
        row["retry"] = retry

def _monitored_call(name, fn, *args, **kwargs):
    _monitor_set(name, "CALLING")
    t0 = time.monotonic()
    try:
        value = fn(*args, **kwargs)
        _monitor_set(name, "OK", latency_ms=(time.monotonic()-t0)*1000)
        return value
    except Exception as exc:
        _monitor_set(
            name, "ERROR",
            latency_ms=(time.monotonic()-t0)*1000,
            error=type(exc).__name__,
            detail=_safe_error_detail(exc),
            http_code=_error_http_code(exc),
        )
        raise



def evaluate_signal(symbol, market, ai_result):
    decision = str(
        ai_result.get("decision", "HOLD")
    ).upper()

    try:
        confidence = int(
            ai_result.get("confidence", 0)
        )
    except Exception:
        confidence = 0

    balance = float(
        RISK_CONFIG["sim_balance_usdt"]
    )

    if decision not in {"BUY", "SELL", "HOLD"}:
        return {
            "allowed": False,
            "reason": "Decision AI không hợp lệ.",
        }

    if decision == "HOLD":
        return {"allowed": False, "reason": "AI đang HOLD."}

    # Spot-only: SELL may close an existing BUY, never open a short.
    if decision == "SELL":
        if PAPER_POSITION is None:
            return {"allowed": False, "reason": "SELL bị BLOCK: không có BUY để đóng; Spot không mở SHORT."}
        if confidence < RISK_CONFIG["min_confidence"]:
            return {"allowed": False, "reason": f"Confidence {confidence}% thấp hơn ngưỡng {RISK_CONFIG['min_confidence']}%."}
        return {"allowed": True, "reason": "SELL hợp lệ: đóng BUY PAPER hiện có.", "decision": "SELL", "confidence": confidence, "exit_reference": float((market.get("15m") or {}).get("close") or 0)}

    if SIM_STATE["open_positions"] >= RISK_CONFIG["max_open_positions"]:
        return {"allowed": False, "reason": "Đã đạt số vị thế tối đa."}

    if confidence < RISK_CONFIG["min_confidence"]:
        return {
            "allowed": False,
            "reason": (
                f"Confidence {confidence}% thấp hơn "
                f"ngưỡng "
                f"{RISK_CONFIG['min_confidence']}%."
            ),
        }

    tf15 = market.get("15m", {})

    price_value = tf15.get("close")
    atr_value = tf15.get("atr14")

    price_now = float(price_value or 0)
    atr = float(atr_value or 0)

    if price_now <= 0 or atr <= 0:
        return {
            "allowed": False,
            "reason": "Giá hoặc ATR không hợp lệ.",
        }

    stop_distance = (
        atr
        * RISK_CONFIG["stop_atr_multiplier"]
    )

    risk_amount = (
        balance
        * RISK_CONFIG["risk_per_trade_pct"]
        / 100
    )

    quantity = risk_amount / stop_distance
    position_value = quantity * price_now

    # Paper Spot: không vay/đòn bẩy.
    if position_value > balance:
        quantity = balance / price_now
        position_value = quantity * price_now
        actual_risk = quantity * stop_distance
    else:
        actual_risk = risk_amount

    stop_loss = price_now - stop_distance

    take_profit = price_now + (
        stop_distance
        * RISK_CONFIG["reward_risk_ratio"]
    )

    if stop_loss <= 0:
        return {
            "allowed": False,
            "reason": "Stop Loss tính ra không hợp lệ.",
        }

    return {
        "allowed": True,
        "reason": "Đạt các điều kiện Risk Engine V2.",
        "decision": decision,
        "confidence": confidence,
        "balance": balance,
        "entry_reference": price_now,
        "atr": atr,
        "stop_distance": stop_distance,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "quantity": quantity,
        "position_value": position_value,
        "planned_risk_usdt": actual_risk,
        "risk_per_trade_pct": (
            RISK_CONFIG["risk_per_trade_pct"]
        ),
    }


# ============================================================
# PAPER POSITION MANAGER
# ============================================================

def open_paper_position(symbol, risk):
    global PAPER_POSITION

    if PAPER_POSITION is not None:
        return False, "Đã có vị thế mô phỏng đang mở."

    PAPER_POSITION = {
        "symbol": symbol,
        "side": "BUY",
        "entry": float(risk["entry_reference"]),
        "quantity": float(risk["quantity"]),
        "stop_loss": float(risk["stop_loss"]),
        "take_profit": float(risk["take_profit"]),
        "position_value": float(risk["position_value"]),
        "planned_risk_usdt": float(
            risk["planned_risk_usdt"]
        ),
    }

    SIM_STATE["open_positions"] = 1

    return True, "Đã mở vị thế mô phỏng."


def close_paper_position(exit_price, reason):
    global PAPER_POSITION

    if PAPER_POSITION is None:
        return None

    position = PAPER_POSITION

    pnl = (
        float(exit_price) - position["entry"]
    ) * position["quantity"]

    RISK_CONFIG["sim_balance_usdt"] += pnl
    SIM_STATE["daily_pnl_usdt"] += pnl
    SIM_STATE["open_positions"] = 0

    AUTO_STATE["trades"] += 1
    AUTO_STATE["realized_pnl_usdt"] += pnl
    if pnl > 0:
        AUTO_STATE["wins"] += 1
    elif pnl < 0:
        AUTO_STATE["losses"] += 1

    result = {
        "symbol": position["symbol"],
        "entry": position["entry"],
        "exit": float(exit_price),
        "quantity": position["quantity"],
        "pnl": pnl,
        "reason": reason,
        "balance": RISK_CONFIG["sim_balance_usdt"],
    }

    PAPER_POSITION = None

    return result


def check_paper_position(current_price):
    if PAPER_POSITION is None:
        return None

    if current_price <= PAPER_POSITION["stop_loss"]:
        return close_paper_position(
            current_price,
            "STOP LOSS",
        )

    if current_price >= PAPER_POSITION["take_profit"]:
        return close_paper_position(
            current_price,
            "TAKE PROFIT",
        )

    return None



# ============================================================
# V5 DETERMINISTIC VALIDATOR + FAST PAPER POSITION MONITOR
# ============================================================
def validate_ai_against_market(market, ai_result, agent_bundle=None):
    """V5.3 deterministic validator: verify claims only against the SAME timeframe/source."""
    warnings = []
    stats = AUTO_STATE.setdefault("validator_stats", {"runs":0,"clean":0,"contradictions":0,"errors":0,"claims_checked":0,"claims_skipped":0})
    stats["runs"] += 1

    def _txt(v):
        return str(v or "").strip().lower()

    def _claim_tf(text):
        # Require an explicit timeframe. Ambiguous claims are NOT treated as contradictions.
        t = _txt(text).replace(" ", "")
        hits = []
        if "15m" in t or "15phút" in t or "15phut" in t: hits.append("15m")
        if "1h" in t or "1giờ" in t or "1gio" in t: hits.append("1H")
        if "4h" in t or "4giờ" in t or "4gio" in t: hits.append("4H")
        return hits[0] if len(set(hits)) == 1 else None

    def _has_any(text, words):
        return any(w in text for w in words)

    try:
        claims = ai_result.get("reasoning", [])
        if not isinstance(claims, list):
            claims = []

        for raw in claims:
            claim = _txt(raw)
            if not claim:
                continue
            tf = _claim_tf(claim)

            # Volume vs MA20: validate only when the claim explicitly names one timeframe.
            if _has_any(claim, ("volume", "khối lượng", "khoi luong")) and _has_any(claim, ("ma20", "trung bình", "trung binh")):
                if tf is None:
                    stats["claims_skipped"] += 1
                else:
                    d = market.get(tf, {}) or {}
                    vol = float(d.get("volume") or 0)
                    vma = float(d.get("volume_ma20") or d.get("volume_ma") or 0)
                    if vol > 0 and vma > 0:
                        stats["claims_checked"] += 1
                        ratio = vol / vma
                        says_below = _has_any(claim, ("thấp hơn", "thap hon", "below", "dưới ma20", "duoi ma20"))
                        says_above = _has_any(claim, ("cao hơn", "cao hon", "above", "trên ma20", "tren ma20"))
                        if ratio > 1.05 and says_below:
                            warnings.append(f"MASTER_VOLUME_CONTRADICTION tf={tf} ratio={ratio:.2f}")
                        elif ratio < 0.95 and says_above:
                            warnings.append(f"MASTER_VOLUME_CONTRADICTION tf={tf} ratio={ratio:.2f}")

            # Explicit RSI overbought/oversold claims, same timeframe only.
            if "rsi" in claim:
                if tf is None:
                    stats["claims_skipped"] += 1
                else:
                    d = market.get(tf, {}) or {}
                    rsi = float(d.get("rsi14") or 0)
                    if rsi > 0:
                        says_overbought = _has_any(claim, ("quá mua", "qua mua", "overbought"))
                        says_oversold = _has_any(claim, ("quá bán", "qua ban", "oversold"))
                        if says_overbought or says_oversold:
                            stats["claims_checked"] += 1
                            if says_overbought and rsi < 70:
                                warnings.append(f"MASTER_RSI_CONTRADICTION tf={tf} rsi={rsi:.2f} claim=OVERBOUGHT")
                            elif says_oversold and rsi > 30:
                                warnings.append(f"MASTER_RSI_CONTRADICTION tf={tf} rsi={rsi:.2f} claim=OVERSOLD")

        # Orderbook directional fact: source-specific, no timeframe mixing.
        if agent_bundle:
            reasoning = " ".join(_txt(x) for x in claims)
            book = agent_bundle.get("orderbook", {}) or {}
            imb = book.get("imbalance")
            if imb is not None and "imbalance" in reasoning:
                imb = float(imb)
                says_positive = _has_any(reasoning, ("imbalance dương", "imbalance duong", "positive imbalance"))
                says_negative = _has_any(reasoning, ("imbalance âm", "imbalance am", "negative imbalance"))
                if says_positive or says_negative:
                    stats["claims_checked"] += 1
                    if says_positive and imb < -0.02:
                        warnings.append(f"MASTER_ORDERBOOK_CONTRADICTION imbalance={imb:.3f} claim=POSITIVE")
                    elif says_negative and imb > 0.02:
                        warnings.append(f"MASTER_ORDERBOOK_CONTRADICTION imbalance={imb:.3f} claim=NEGATIVE")

    except Exception as exc:
        stats["errors"] += 1
        warnings.append(f"VALIDATOR_ERROR:{type(exc).__name__}")

    checked = dict(ai_result)
    contradictions = [w for w in warnings if "CONTRADICTION" in w]
    if contradictions:
        stats["contradictions"] += 1
    else:
        stats["clean"] += 1

    # Fail closed only for actionable BUY. HOLD/SELL remain non-entry decisions in Spot paper mode.
    if warnings and str(checked.get("decision", "HOLD")).upper() == "BUY":
        checked["decision"] = "HOLD"
        checked["confidence"] = 0
        checked["needs_more_data"] = True
        checked.setdefault("risk_notes", []).append(
            "Python Validator V5.3 phát hiện mâu thuẫn dữ liệu; fail-safe HOLD."
        )

    AUTO_STATE["validator_warnings"] = warnings
    return checked, warnings


def _record_trade(result):
    row = dict(result)
    row["closed_at"] = time.time()
    TRADE_HISTORY.append(row)
    if len(TRADE_HISTORY) > 500:
        del TRADE_HISTORY[:-500]


async def paper_position_monitor(application):
    """Monitor SL/TP independently from the 15-minute AI cycle."""
    logging.info("V5 PAPER POSITION MONITOR STARTED")
    try:
        while AUTO_STATE.get("desired_running"):
            if PAPER_POSITION is not None:
                try:
                    ticker = await asyncio.wait_for(
                        asyncio.to_thread(get_ticker, PAPER_POSITION["symbol"]), timeout=25
                    )
                    closed = check_paper_position(float(ticker["last"]))
                    if closed:
                        _record_trade(closed)
                        await send_auto_message(
                            application, AUTO_STATE.get("chat_id"),
                            "🏁 V5 PAPER POSITION ĐÃ ĐÓNG\n\n"
                            f"{closed['symbol']} | {closed['reason']}\n"
                            f"Entry: {closed['entry']:.4f}\nExit: {closed['exit']:.4f}\n"
                            f"PnL: {closed['pnl']:+.2f} USDT\nBalance: {closed['balance']:.2f} USDT"
                        )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logging.exception("V5 POSITION MONITOR ERROR")
            await asyncio.sleep(15)
    finally:
        logging.info("V5 PAPER POSITION MONITOR STOPPED")


def ensure_position_monitor(application):
    task = AUTO_STATE.get("position_task")
    if task is None or task.done():
        AUTO_STATE["position_task"] = application.create_task(paper_position_monitor(application))


async def performance_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    trades = AUTO_STATE.get("trades", 0)
    wins = AUTO_STATE.get("wins", 0)
    losses = AUTO_STATE.get("losses", 0)
    pnl = AUTO_STATE.get("realized_pnl_usdt", 0.0)
    winrate = (wins / trades * 100) if trades else 0.0
    open_text = "Có" if PAPER_POSITION is not None else "Không"
    warnings = AUTO_STATE.get("validator_warnings") or []
    await update.message.reply_text(
        "📈 V5 PAPER PERFORMANCE\n\n"
        f"Balance: {RISK_CONFIG['sim_balance_usdt']:.2f} USDT\n"
        f"Realized PnL: {pnl:+.2f} USDT\n"
        f"Trades: {trades} | Wins: {wins} | Losses: {losses}\n"
        f"Win rate: {winrate:.1f}%\nOpen position: {open_text}\n"
        f"Cycles: {AUTO_STATE.get('cycle_count', 0)}\n"
        f"Validator warnings latest: {len(warnings)}\n"
        f"Validator runs/contradictions: {AUTO_STATE.get('validator_stats',{}).get('runs',0)}/{AUTO_STATE.get('validator_stats',{}).get('contradictions',0)}\n\n"
        "🧪 PAPER ONLY"
    )


async def send_auto_message(application, chat_id, text):
    try:
        await application.bot.send_message(
            chat_id=chat_id,
            text=text,
        )
    except Exception:
        logging.exception("AUTO TELEGRAM MESSAGE ERROR")


async def auto_trading_loop(application, chat_id):
    """Robust paper-only automatic loop."""
    logging.info("AUTO PAPER LOOP STARTED")

    while AUTO_STATE["running"]:
        AUTO_STATE["last_heartbeat_at"] = time.time()
        try:
            symbol = AUTO_STATE["symbol"]
            AUTO_STATE["last_progress_at"] = time.time()

            if PAPER_POSITION is not None:
                ticker = await asyncio.wait_for(
                    asyncio.to_thread(
                        get_ticker,
                        PAPER_POSITION["symbol"],
                    ),
                    timeout=25,
                )
                current_price = float(ticker["last"])
                closed = check_paper_position(current_price)

                if closed:
                    _record_trade(closed)
                    icon = "🎯" if closed["reason"] == "TAKE PROFIT" else "🛑"
                    logging.info("PAPER POSITION CLOSED %s %s", closed["symbol"], closed["reason"])

            # V4.2: always run the 1→8 cycle, even if a paper position exists.
            # Risk Engine remains authoritative and will block additional exposure as configured.
            if True:
                market = await asyncio.wait_for(
                    asyncio.to_thread(
                        _monitored_call, "OKX_MARKET", get_market_analysis, symbol
                    ),
                    timeout=90,
                )
                candle_ts = market["15m"]["timestamp"]

                # V4.2 CONTINUOUS: do not wait for a new 15m candle.
                # Keep the timestamp only as metadata for the latest completed 15m candle.
                AUTO_STATE["last_analyzed_15m_ts"] = candle_ts
                if True:

                    try:
                        agent_bundle = await run_advanced_agent_pipeline(
                            application, chat_id, symbol, market
                        )
                        ai_result = agent_bundle["master"]
                        _job_stat_run(10)
                        _touch_progress("V5:10/15:VALIDATOR:RUNNING")
                        ai_result, validator_warnings = validate_ai_against_market(market, ai_result, agent_bundle)
                        _touch_progress("V5:10/15:VALIDATOR:DONE")
                        _job_stat_result(10, not any(str(w).startswith("VALIDATOR_ERROR:") for w in validator_warnings))
                        agent_bundle["master"] = ai_result
                        AUTO_STATE["last_progress_at"] = time.time()
                    except asyncio.TimeoutError:
                        logging.error("AUTO GEMINI TIMEOUT for %s", symbol)
                        await send_auto_message(
                            application,
                            chat_id,
                            "⚠️ AUTO PAPER: Gemini phản hồi quá lâu.\n\n"
                            f"{symbol}\n"
                            "Bỏ qua kết quả AI lỗi. Auto vẫn chạy và tiếp tục cycle kế tiếp."
                        )
                        ai_result = None
                    except Exception as exc:
                        logging.exception("AUTO GEMINI ERROR")
                        await send_auto_message(
                            application,
                            chat_id,
                            "⚠️ AUTO PAPER: Gemini gặp lỗi.\n\n"
                            f"{symbol}\n"
                            f"Lỗi: {type(exc).__name__}\n"
                            "Bỏ qua công việc lỗi. Auto vẫn tiếp tục cycle kế tiếp."
                        )
                        ai_result = None

                    if ai_result is not None:
                        _job_stat_run(11)
                        risk = evaluate_signal(symbol, market, ai_result)
                        _touch_progress("V5:11/15:RISK_ENGINE:DONE"); _job_stat_result(11, True)
                        # V4.3 job 7 completes silently; job 8 will summarize the whole cycle.

                        # Cache the latest completed-15m AI analysis.
                        # The 60-second Telegram report reuses this snapshot;
                        # it does NOT call Gemini again.
                        AUTO_STATE["last_market"] = market
                        AUTO_STATE["last_ai_result"] = ai_result
                        AUTO_STATE["last_agent_bundle"] = agent_bundle
                        AUTO_STATE["last_risk_result"] = risk
                        AUTO_STATE["last_ai_updated_at"] = time.time()

                        execution_detail = "NO ACTION"
                        if risk["allowed"]:
                            if str(risk.get("decision", "BUY")).upper() == "SELL":
                                closed = close_paper_position(risk.get("exit_reference") or market["15m"]["close"], "AI SELL")
                                if closed:
                                    _record_trade(closed)
                                    execution_detail = "BUY CLOSED BY AI SELL"
                                    logging.info("PAPER TRADE CLOSED BY AI SELL %s", symbol)
                            else:
                                opened, reason = open_paper_position(symbol, risk)
                                if opened:
                                    execution_detail = "BUY OPENED"
                                    logging.info("PAPER TRADE OPENED %s", symbol)
                        else:
                            decision = str(
                                ai_result.get("decision", "HOLD")
                            ).upper()
                            confidence = ai_result.get("confidence", 0)

                            logging.info(
                                "AUTO BLOCK %s: %s (%s%%) - %s",
                                symbol,
                                decision,
                                confidence,
                                risk["reason"],
                            )

                        # Jobs 12-14 completed for this cycle. PASS means the job executed without error;
                        # for Risk/Paper it does NOT mean a BUY was allowed or profitable.
                        _job_stat_run(12); _touch_progress("V8:12/15:PAPER_EXECUTION:DONE"); _job_stat_result(12, True)
                        _job_stat_run(13); _touch_progress("V8:13/15:MEMORY:RUNNING")
                        memory_row = _record_agent_memory(agent_bundle, validator_warnings, risk, execution_detail)
                        agent_bundle["memory"] = memory_row
                        _touch_progress("V8:13/15:MEMORY:DONE"); _job_stat_result(13, True)
                        _job_stat_run(14); _touch_progress("V8:14/15:REFLECTION:RUNNING")
                        agent_bundle["reflection"] = _reflect_cycle(agent_bundle, memory_row)
                        _touch_progress("V8:14/15:REFLECTION:DONE"); _job_stat_result(14, True)
                        _job_stat_run(15)
                        _touch_progress("V5:15/15:SUMMARY:RUNNING")
                        try:
                            summary_text = await build_auto_status_text()
                            # V8 preserves the V6.0.1 compact fixed 1→15 layout.
                            summary_text = summary_text
                            AUTO_STATE["cycle_count"] += 1
                            await send_auto_message(application, chat_id, summary_text)
                            _job_stat_result(15, True)
                            _touch_progress("V5:15/15:SUMMARY:DONE")
                        except Exception as exc:
                            _job_stat_result(15, False)
                            _touch_progress("V5:15/15:SUMMARY:ERROR")
                            logging.exception("V5 JOB 15 SUMMARY ERROR: %s", type(exc).__name__)

        except asyncio.CancelledError:
            logging.info("AUTO PAPER LOOP CANCELLED")
            raise
        except asyncio.TimeoutError:
            logging.exception("AUTO OKX/MARKET TIMEOUT")
            await send_auto_message(
                application,
                chat_id,
                "⚠️ AUTO PAPER: dữ liệu thị trường phản hồi quá lâu.\n"
                "Bot không dừng; sẽ chuyển sang cycle tiếp theo ngay."
            )
        except Exception as exc:
            logging.exception("AUTO PAPER LOOP ERROR")
            await send_auto_message(
                application,
                chat_id,
                "⚠️ AUTO PAPER GẶP LỖI\n\n"
                f"{type(exc).__name__}\n"
                "Bot không dừng; sẽ tự chạy cycle tiếp theo ngay."
            )

        AUTO_STATE["last_heartbeat_at"] = time.time()
        # V4.4: wait 15 minutes after each completed/failed cycle before starting the next cycle.
        # Sleep in short chunks so /stop remains responsive and heartbeat stays healthy.
        for _ in range(180):
            if not AUTO_STATE["running"]:
                break
            AUTO_STATE["last_heartbeat_at"] = time.time()
            _touch_progress("V5:WAITING_15M")
            await asyncio.sleep(5)

    logging.info("AUTO PAPER LOOP STOPPED")


async def auto_watchdog(application):
    """Restart worker if it exits OR stops heartbeating."""
    logging.info("AUTO WATCHDOG STARTED")

    while AUTO_STATE["desired_running"]:
        try:
            task = AUTO_STATE.get("task")
            now = time.time()
            heartbeat = AUTO_STATE.get("last_heartbeat_at")
            progress = AUTO_STATE.get("last_progress_at") or heartbeat
            progress_age = (now - progress) if progress is not None else None
            stale = (
                task is not None
                and not task.done()
                and progress_age is not None
                and progress_age > PROGRESS_STALE_SECONDS
            )

            if stale:
                logging.error(
                    "AUTO WORKER STALE: no progress for %.1fs (stage=%s)",
                    progress_age,
                    AUTO_STATE.get("last_progress_stage", "UNKNOWN"),
                )
                AUTO_STATE["last_worker_error"] = "STALE_HEARTBEAT"
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=10)
                except BaseException:
                    pass
                AUTO_STATE["task"] = None
                task = None

            # V4.3: no periodic Telegram status task; summary is job 8 of each cycle.
            if task is None or task.done():
                if not AUTO_STATE["desired_running"]:
                    break

                AUTO_STATE["running"] = True
                AUTO_STATE["last_heartbeat_at"] = time.time()
                AUTO_STATE["last_progress_at"] = time.time()
                AUTO_STATE["worker_restarts"] += 1
                AUTO_STATE["task"] = application.create_task(
                    auto_trading_loop(
                        application,
                        AUTO_STATE["chat_id"],
                    )
                )

                await send_auto_message(
                    application,
                    AUTO_STATE["chat_id"],
                    "♻️ AUTO PAPER ĐÃ TỰ KHỞI ĐỘNG LẠI\n\n"
                    f"📌 {AUTO_STATE['symbol']}\n"
                    f"Restart count: {AUTO_STATE['worker_restarts']}\n"
                    "Watchdog phát hiện worker dừng/treo và đã tạo worker mới.\n\n"
                    "🧪 PAPER ONLY."
                )

            await asyncio.sleep(15)

        except asyncio.CancelledError:
            logging.info("AUTO WATCHDOG CANCELLED")
            raise
        except Exception as exc:
            AUTO_STATE["last_worker_error"] = type(exc).__name__
            logging.exception("AUTO WATCHDOG ERROR")
            await asyncio.sleep(15)

    logging.info("AUTO WATCHDOG STOPPED")


def ensure_auto_watchdog(application):
    task = AUTO_STATE.get("watchdog_task")
    if task is None or task.done():
        AUTO_STATE["watchdog_task"] = application.create_task(
            auto_watchdog(application)
        )


async def auto_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    symbol = "BTC-USDT"
    if context.args:
        symbol = normalize_symbol(context.args[0])

    AUTO_STATE["symbol"] = symbol
    AUTO_STATE["chat_id"] = update.effective_chat.id
    AUTO_STATE["desired_running"] = True
    AUTO_STATE["last_heartbeat_at"] = time.time()
    AUTO_STATE["last_progress_at"] = time.time()

    task = AUTO_STATE.get("task")
    if task is None or task.done():
        AUTO_STATE["running"] = True
        AUTO_STATE["task"] = context.application.create_task(
            auto_trading_loop(
                context.application,
                AUTO_STATE["chat_id"],
            )
        )

    ensure_auto_watchdog(context.application)
    ensure_position_monitor(context.application)
    # V4.3: no 60-second reporter. Job 8 sends one summary after every completed cycle.

    await update.message.reply_text(
        "🚀 AUTO PAPER ĐÃ BẬT\n\n"
        f"Symbol: {symbol}\n"
        f"Vốn giả định: {RISK_CONFIG['sim_balance_usdt']:.2f} USDT\n"
        f"Risk/lệnh: {RISK_CONFIG['risk_per_trade_pct']}%\n"
        f"Min confidence: {RISK_CONFIG['min_confidence']}%\n"
        "Phân tích AI: 15 phút / cycle\n"
        "V6 Python Validator V2: ON\n"
        "Kiểm tra SL/TP độc lập: khoảng 15 giây/lần\n\n"
        "♻️ Watchdog: ON\n"
        "Nếu Auto worker lỗi/dừng ngoài ý muốn, bot sẽ tự bật lại.\n"
        "📨 Kết quả được tự gửi về chat Telegram này.\n"
        "📨 Telegram: gửi 1 tổng kết sau khi pipeline 15/15 công việc hoàn tất.\n"
        "🧠 Gemini API: gọi theo cycle 15 phút; position monitor không gọi Gemini.\n\n"
        "🧪 PAPER ONLY. Không gửi lệnh tới OKX.\n"
        "Dùng /stop để dừng chủ động."
    )


async def stopauto_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    AUTO_STATE["desired_running"] = False
    AUTO_STATE["running"] = False

    worker = AUTO_STATE.get("task")
    if worker is not None and not worker.done():
        worker.cancel()
    AUTO_STATE["task"] = None

    watchdog = AUTO_STATE.get("watchdog_task")
    if watchdog is not None and not watchdog.done():
        watchdog.cancel()
    AUTO_STATE["watchdog_task"] = None

    position_task = AUTO_STATE.get("position_task")
    if position_task is not None and not position_task.done():
        position_task.cancel()
    AUTO_STATE["position_task"] = None

    status_task = AUTO_STATE.get("status_task")
    if status_task is not None and not status_task.done():
        status_task.cancel()
    AUTO_STATE["status_task"] = None

    await update.message.reply_text(
        "⛔ AUTO PAPER ĐÃ DỪNG\n\n"
        "Watchdog: OFF\n"
        "Bot sẽ không tự khởi động lại Auto worker "
        "cho đến khi bạn gửi /auto.\n\n"
        "ℹ️ Vị thế paper đang mở (nếu có) không bị ép đóng."
    )


def _api_monitor_text():
    def line(name, label):
        x = API_MONITOR.get(name)
        if not x:
            return f"{label}: ⏸ WAITING"
        st = x.get("status", "WAITING")
        icon = {
            "OK":"✅", "CALLING":"⏳", "RETRY":"🔄", "FALLBACK":"🔀",
            "ERROR":"❌", "TIMEOUT":"⏱", "SKIPPED":"⏭",
        }.get(st, "•")
        extra = []
        if x.get("latency_ms") is not None:
            extra.append(f"{x['latency_ms']}ms")
        if x.get("retry"):
            extra.append(f"retry {x['retry']}")
        if x.get("error"):
            extra.append(x["error"])
        if x.get("http_code"):
            extra.append(f"HTTP {x['http_code']}")
        base = f"{label}: {icon} {st}" + (f" | {' | '.join(extra)}" if extra else "")
        if x.get("detail") and st in ("ERROR", "RETRY", "FALLBACK", "TIMEOUT"):
            base += f"\n  ↳ {x['detail']}"
        return base

    now = time.time()
    hb = AUTO_STATE.get("last_heartbeat_at")
    if hb:
        age = max(0, int(now - hb))
        hb_text = f"✅ HEALTHY | {age}s trước" if age <= WORKER_STALE_SECONDS else f"❌ STALE | {age}s"
    else:
        hb_text = "⏸ WAITING"

    # OKX 15m candles close on :00/:15/:30/:45.
    sec = int(now)
    next_close = ((sec // 900) + 1) * 900
    remain = max(0, next_close - sec)
    mm, ss = divmod(remain, 60)

    last_ai = AUTO_STATE.get("last_ai_updated_at")
    if last_ai:
        ai_age = max(0, int((now-last_ai)//60))
        last_cycle = f"{ai_age} phút trước"
    else:
        last_cycle = "chưa có"

    rows = [
        "🩺 API / SYSTEM MONITOR",
        line("OKX_MARKET", "OKX Market"),
        line("OKX_ORDERBOOK", "OKX Orderbook"),
        line("OKX_DERIVATIVES", "OKX Derivatives"),
        "",
        line("TECHNICAL_AGENT", "Gemini Technical"),
        line("ORDERBOOK_AGENT", "Gemini Orderbook"),
        line("DERIVATIVES_AGENT", "Gemini Derivatives"),
        line("CRITIC_AGENT", "Gemini Critic"),
        line("MASTER_DECISION_AGENT", "Gemini Master"),
        "",
        f"❤️ Heartbeat: {hb_text}",
        (f"🧭 Progress: {max(0, int(now-(AUTO_STATE.get('last_progress_at') or now)))}s trước | "
         f"{AUTO_STATE.get('last_progress_stage') or 'LOOP'}"),
        f"♻️ Worker restarts: {AUTO_STATE.get('worker_restarts', 0)}",
        f"🕒 Last AI cycle: {last_cycle}",
        "🔁 Cycle kế tiếp: sau 15 phút",
        "⏱ Chu kỳ AI: 15 phút / vòng.",
    ]
    return "\n".join(rows)



async def jobstats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(format_job_stats())

async def build_auto_status_text():
    """Compact V8 report preserving the V6.0.1 fixed 1→15 layout."""
    symbol = AUTO_STATE["symbol"]
    market = AUTO_STATE.get("last_market") or {}
    ai = AUTO_STATE.get("last_ai_result") or {}
    risk = AUTO_STATE.get("last_risk_result") or {}
    bundle = AUTO_STATE.get("last_agent_bundle") or {}
    warnings = AUTO_STATE.get("validator_warnings") or []

    try:
        ticker = await asyncio.wait_for(asyncio.to_thread(get_ticker, symbol), timeout=25)
        price = float(ticker.get("last") or 0)
    except Exception:
        price = 0.0

    def num(v, d=2):
        try: return f"{float(v):.{d}f}"
        except Exception: return "N/A"

    def stat(n, assume_current_final=False):
        row = AUTO_STATE.get("job_stats", {}).get(n, {"run":0,"pass":0,"fail":0})
        run, passed = int(row.get("run",0)), int(row.get("pass",0))
        # Job 15 is being rendered before Telegram send completes. Display this
        # current report as pending-success so the report does not look one cycle behind.
        if assume_current_final and n == 15 and run > passed:
            passed += 1
        pct = (passed/run*100) if run else 0
        return f"{passed}/{run} ({pct:.0f}%)"

    def okmark(n):
        row=AUTO_STATE.get("job_stats",{}).get(n,{})
        return "✅" if int(row.get("fail",0)) == 0 else "⚠️"

    q = bundle.get("quant") or {}
    regime = (bundle.get("regime_engine") or {}).get("regime", "N/A")
    book = bundle.get("orderbook") or {}
    deriv = bundle.get("derivatives") or {}
    specialists = {str(x.get("agent")):x for x in (bundle.get("specialists") or [])}
    tech = specialists.get("TECHNICAL_AGENT", {})
    ob_ai = specialists.get("ORDERBOOK_AGENT", {})
    der_ai = specialists.get("DERIVATIVES_AGENT", {})
    critic = bundle.get("critic") or {}

    tf15, tf1h, tf4h = market.get("15m",{}), market.get("1H",{}), market.get("4H",{})
    qb = lambda tf: (q.get(tf) or {}).get("bias", "N/A")
    imb = book.get("imbalance")
    funding = deriv.get("funding_rate")
    oi = deriv.get("open_interest")

    validator_detail = ("PASS — 0 contradiction" if not warnings else
                        "WARN — " + "; ".join(str(x) for x in warnings[:2]))
    decision = str(ai.get("decision","HOLD")).upper()
    confidence = int(ai.get("confidence",0) or 0)
    risk_state = "ALLOWED" if risk.get("allowed") else "BLOCKED"
    risk_reason = str(risk.get("reason","N/A"))[:90]

    if PAPER_POSITION:
        pos = f"BUY @ {num(PAPER_POSITION.get('entry'),4)} | SL {num(PAPER_POSITION.get('stop_loss'),4)} | TP {num(PAPER_POSITION.get('take_profit'),4)}"
        exec_detail = "POSITION OPEN"
    else:
        pos = "NONE"
        exec_detail = "NO ACTION" if not risk.get("allowed") else "ENTRY ATTEMPTED"

    task_alive = bool(AUTO_STATE.get("task") and not AUTO_STATE["task"].done())
    gemini_rows=[v for k,v in API_MONITOR.items() if "AGENT" in k]
    gem_ok=sum(1 for x in gemini_rows if x.get("status")=="OK")
    gem_total=len(gemini_rows)
    okx_rows=[API_MONITOR.get(k,{}) for k in ("OKX_MARKET","OKX_ORDERBOOK","OKX_DERIVATIVES")]
    okx_ok=sum(1 for x in okx_rows if x.get("status")=="OK")

    cycle_no = int(AUTO_STATE.get("cycle_count",0)) + 1
    lines = [
        f"✅ V8 — CYCLE #{cycle_no} | 15/15",
        f"₿ {symbol} | 💰 {num(price,4)} USDT",
        "",
        "━━━ 15 CÔNG VIỆC ━━━",
        f"01 MARKET DATA       {okmark(1)} {stat(1)} | RSI 15m {num(tf15.get('rsi14'),1)} · 1H {num(tf1h.get('rsi14'),1)} · 4H {num(tf4h.get('rsi14'),1)}",
        f"02 QUANT ENGINE      {okmark(2)} {stat(2)} | 15m {qb('15m')} · 1H {qb('1H')} · 4H {qb('4H')}",
        f"03 ORDERBOOK DATA    {okmark(3)} {stat(3)} | Imbalance {num(imb,3)} · Spread {num(book.get('spread_bps'),2)}bps",
        f"04 DERIVATIVES DATA  {okmark(4)} {stat(4)} | Funding {num(funding,6)} · OI {num(oi,2)}",
        f"05 MARKET REGIME     {okmark(5)} {stat(5)} | {regime}",
        f"06 AI TECHNICAL      {okmark(6)} {stat(6)} | {tech.get('bias','N/A')} {tech.get('confidence',0)}%",
        f"07 AI MICROSTRUCTURE {okmark(7)} {stat(7)} | Book {ob_ai.get('bias','N/A')} {ob_ai.get('confidence',0)}% · Deriv {der_ai.get('bias','N/A')} {der_ai.get('confidence',0)}%",
        f"08 CRITIC            {okmark(8)} {stat(8)} | Agree {critic.get('agreement','N/A')} · Caution {critic.get('recommended_caution','N/A')}",
        f"09 MASTER            {okmark(9)} {stat(9)} | {decision} {confidence}%",
        f"10 VALIDATOR V2      {'✅' if not warnings else '⚠️'} {stat(10)} | {validator_detail}",
        f"11 RISK ENGINE       {okmark(11)} {stat(11)} | {risk_state} · {risk_reason}",
        f"12 PAPER EXECUTION   {okmark(12)} {stat(12)} | {exec_detail}",
        f"13 MEMORY            {okmark(13)} {stat(13)} | Cycle recorded",
        f"14 REFLECTION        {okmark(14)} {stat(14)} | Cycle reviewed",
        f"15 FINAL REPORT      {okmark(15)} {stat(15, True)} | Telegram report",
        "",
        "━━━ QUYẾT ĐỊNH ━━━",
        f"🧠 MASTER {decision} {confidence}% | 🛡 RISK {risk_state}",
        f"🧮 VALIDATOR {'PASS' if not warnings else 'WARN'} | 📍 POSITION {pos}",
        "",
        "━━━ THỊ TRƯỜNG ━━━",
        f"15m RSI {num(tf15.get('rsi14'),1)} | MACD {num(tf15.get('macd_hist'),4)} | ATR {num(tf15.get('atr14'),2)}",
        f"VOL15 {num(tf15.get('volume'),2)} / MA20 {num(tf15.get('volume_ma20'),2)}",
        "",
        "━━━ PAPER ━━━",
        f"💵 {RISK_CONFIG['sim_balance_usdt']:.2f} USDT | PnL {AUTO_STATE['realized_pnl_usdt']:+.2f}",
        f"🎯 Trades {AUTO_STATE['trades']} | W {AUTO_STATE['wins']} | L {AUTO_STATE['losses']}",
        "",
        "━━━ HỆ THỐNG ━━━",
        f"⚙️ Worker {'✅' if task_alive else '❌'} | Watchdog {'✅' if AUTO_STATE.get('desired_running') else '❌'} | Restarts {AUTO_STATE.get('worker_restarts',0)}",
        f"🌐 OKX {okx_ok}/3 OK | 🤖 Gemini {gem_ok}/{gem_total} OK",
        "⏱ Cycle tiếp theo: sau 15 phút | 🧪 PAPER ONLY",
    ]
    return "\n".join(lines)[:4000]


async def auto_status_notifier(application):
    """Send a full agent report every 60 seconds while Auto is enabled."""
    logging.info("AUTO STATUS NOTIFIER STARTED")

    try:
        while AUTO_STATE["desired_running"]:
            await asyncio.sleep(60)

            if not AUTO_STATE["desired_running"]:
                break

            try:
                text = await build_auto_status_text()
                await send_auto_message(
                    application,
                    AUTO_STATE["chat_id"],
                    text,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logging.exception("AUTO STATUS NOTIFIER ERROR")

    except asyncio.CancelledError:
        logging.info("AUTO STATUS NOTIFIER CANCELLED")
        raise
    finally:
        logging.info("AUTO STATUS NOTIFIER STOPPED")


def ensure_auto_status_notifier(application):
    task = AUTO_STATE.get("status_task")
    if task is None or task.done():
        AUTO_STATE["status_task"] = application.create_task(
            auto_status_notifier(application)
        )


async def autostatus_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    text = await build_auto_status_text()
    await update.message.reply_text(text)


async def agent_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbol = normalize_symbol(context.args)
    await update.message.reply_text(f"🧠 Đang chạy Advanced Multi-Agent cho {symbol}...")
    try:
        bundle = await asyncio.wait_for(asyncio.to_thread(run_advanced_agent, symbol), timeout=180)
        m=bundle["master"]; c=bundle["critic"]
        text=(f"🧠 ADVANCED AGENT — {symbol}\n\n{format_agent_summary(bundle)}\n\n"
              f"Critic conflicts: {len(c.get('conflicts',[]))}\nNeeds more data: {c.get('needs_more_data',False)}\n\n"
              f"MASTER: {m.get('decision')} | {m.get('confidence')}%\nRegime: {m.get('market_regime','N/A')}\n\n"
              "⚠️ AI chỉ quyết định hướng. Tiền/size/SL/TP thuộc Python Risk Engine.\n🧪 PAPER ONLY.")
        await update.message.reply_text(text[:4000])
    except Exception as e:
        logging.exception("AGENT COMMAND ERROR")
        await update.message.reply_text(f"❌ Advanced Agent lỗi: {type(e).__name__}")


async def risk_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    balance = RISK_CONFIG["sim_balance_usdt"]

    max_risk = (
        balance
        * RISK_CONFIG["risk_per_trade_pct"]
        / 100
    )

    daily_limit = None

    text = (
        "🛡 RISK ENGINE V2\n\n"
        f"Vốn mô phỏng: {balance:.2f} USDT\n"
        f"Risk/lệnh: "
        f"{RISK_CONFIG['risk_per_trade_pct']}%\n"
        f"Risk tối đa/lệnh: {max_risk:.2f} USDT\n"
        "Daily loss limit: NONE / disabled\n"
        f"Max positions: "
        f"{RISK_CONFIG['max_open_positions']}\n"
        f"Min confidence: "
        f"{RISK_CONFIG['min_confidence']}%\n"
        f"Stop Loss: "
        f"{RISK_CONFIG['stop_atr_multiplier']} × ATR\n"
        f"Reward/Risk: "
        f"{RISK_CONFIG['reward_risk_ratio']}:1\n\n"
        f"PnL mô phỏng hôm nay: "
        f"{SIM_STATE['daily_pnl_usdt']:.2f} USDT\n"
        f"Vị thế mô phỏng đang mở: "
        f"{SIM_STATE['open_positions']}\n\n"
        "ℹ️ Gemini không có quyền thay đổi "
        "các giới hạn này."
    )

    await update.message.reply_text(text)


async def position_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if PAPER_POSITION is None:
        await update.message.reply_text(
            "📭 Không có vị thế mô phỏng đang mở."
        )
        return

    try:
        ticker = get_ticker(PAPER_POSITION["symbol"])
        current_price = float(ticker["last"])

        if current_price <= 0:
            raise ValueError("Không lấy được giá hiện tại")

        closed = check_paper_position(current_price)

        if closed:
            await update.message.reply_text(
                "🏁 PAPER POSITION ĐÃ ĐÓNG\n\n"
                f"{closed['symbol']}\n"
                f"Lý do: {closed['reason']}\n"
                f"Entry: {closed['entry']:.4f}\n"
                f"Exit: {closed['exit']:.4f}\n"
                f"PnL: {closed['pnl']:+.2f} USDT\n"
                f"Balance: {closed['balance']:.2f} USDT\n\n"
                "ℹ️ Không có lệnh nào gửi tới OKX."
            )
            return

        p = PAPER_POSITION

        unrealized = (
            current_price - p["entry"]
        ) * p["quantity"]

        current_value = (
            current_price * p["quantity"]
        )

        await update.message.reply_text(
            "📊 PAPER POSITION\n\n"
            f"Symbol: {p['symbol']}\n"
            f"Side: {p['side']}\n"
            f"Entry: {p['entry']:.4f}\n"
            f"Current: {current_price:.4f}\n"
            f"Quantity: {p['quantity']:.8f}\n"
            f"Entry value: "
            f"{p['position_value']:.2f} USDT\n"
            f"Current value: {current_value:.2f} USDT\n\n"
            f"Stop Loss: {p['stop_loss']:.4f}\n"
            f"Take Profit: {p['take_profit']:.4f}\n"
            f"Unrealized PnL: "
            f"{unrealized:+.2f} USDT\n\n"
            "ℹ️ Vị thế này hoàn toàn mô phỏng."
        )

    except Exception as e:
        logging.exception("POSITION ERROR")
        await update.message.reply_text(
            "❌ Không thể kiểm tra paper position.\n"
            f"Loại lỗi: {type(e).__name__}"
        )


async def closepaper_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if PAPER_POSITION is None:
        await update.message.reply_text(
            "📭 Không có vị thế mô phỏng để đóng."
        )
        return

    try:
        ticker = get_ticker(PAPER_POSITION["symbol"])
        current_price = float(ticker["last"])

        if current_price <= 0:
            raise ValueError("Không lấy được giá hiện tại")

        result = close_paper_position(
            current_price,
            "MANUAL CLOSE",
        )

        await update.message.reply_text(
            "🧪 PAPER POSITION ĐÃ ĐÓNG\n\n"
            f"{result['symbol']}\n"
            f"Entry: {result['entry']:.4f}\n"
            f"Exit: {result['exit']:.4f}\n"
            f"PnL: {result['pnl']:+.2f} USDT\n"
            f"Balance: {result['balance']:.2f} USDT\n\n"
            "ℹ️ Không có lệnh nào gửi tới OKX."
        )

    except Exception as e:
        logging.exception("CLOSE PAPER ERROR")
        await update.message.reply_text(
            "❌ Không thể đóng paper position.\n"
            f"Loại lỗi: {type(e).__name__}"
        )


async def signal_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    symbol = normalize_symbol(context.args)

    if not GEMINI_API_KEY or gemini_client is None:
        await update.message.reply_text(
            "❌ Chưa cấu hình GEMINI_API_KEY."
        )
        return

    await update.message.reply_text(
        f"🛡 Đang kiểm tra tín hiệu {symbol}..."
    )

    try:
        # Nếu đã có paper position thì kiểm tra SL/TP trước.
        if PAPER_POSITION is not None:
            ticker = get_ticker(PAPER_POSITION["symbol"])
            closed = check_paper_position(
                float(ticker["last"])
            )

            if closed:
                await update.message.reply_text(
                    "🏁 Paper position trước đã tự đóng khi "
                    "kiểm tra signal.\n"
                    f"Lý do: {closed['reason']}\n"
                    f"PnL: {closed['pnl']:+.2f} USDT"
                )

        market = get_market_analysis(symbol)
        agent_bundle = run_advanced_agent(symbol, market)
        ai_result = agent_bundle["master"]

        risk = evaluate_signal(
            symbol,
            market,
            ai_result,
        )

        decision = ai_result["decision"]
        confidence = ai_result["confidence"]

        if not risk["allowed"]:
            text = (
                f"🛡 RISK ENGINE V2 {symbol}\n\n"
                f"AI: {decision}\n"
                f"Confidence: {confidence}%\n\n"
                "🚫 BLOCK\n"
                f"Lý do: {risk['reason']}\n\n"
                "ℹ️ Không có lệnh nào được gửi tới OKX."
            )

        else:
            opened, open_reason = open_paper_position(
                symbol,
                risk,
            )

            if not opened:
                text = (
                    f"🛡 RISK ENGINE V2 {symbol}\n\n"
                    "🚫 BLOCK\n"
                    f"Lý do: {open_reason}\n\n"
                    "ℹ️ Không có lệnh nào được gửi tới OKX."
                )
            else:
                text = (
                    f"🛡 RISK ENGINE V2 {symbol}\n\n"
                    f"AI: {risk['decision']}\n"
                    f"Confidence: {risk['confidence']}%\n\n"
                    "✅ PAPER POSITION ĐÃ MỞ\n\n"
                    f"Vốn: {risk['balance']:.2f} USDT\n"
                    f"Entry: "
                    f"{risk['entry_reference']:.4f}\n"
                    f"ATR 15m: {risk['atr']:.4f}\n"
                    f"Stop distance: "
                    f"{risk['stop_distance']:.4f}\n"
                    f"Stop Loss: "
                    f"{risk['stop_loss']:.4f}\n"
                    f"Take Profit: "
                    f"{risk['take_profit']:.4f}\n\n"
                    f"Khối lượng: "
                    f"{risk['quantity']:.8f} BTC\n"
                    f"Giá trị vị thế: "
                    f"{risk['position_value']:.2f} USDT\n"
                    f"Rủi ro dự kiến: "
                    f"{risk['planned_risk_usdt']:.2f} USDT\n\n"
                    "🧪 Đây là vị thế giả lập.\n"
                    "Dùng /position để kiểm tra.\n"
                    "Dùng /closepaper để đóng thủ công.\n\n"
                    "⚠️ Chưa gửi lệnh tới OKX."
                )

        await update.message.reply_text(text)

    except Exception as e:
        logging.exception("SIGNAL ERROR")
        await update.message.reply_text(
            "❌ Không thể tạo signal.\n"
            f"Loại lỗi: {type(e).__name__}"
        )


# ============================================================
# RENDER HEALTH SERVER
# ============================================================

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/health"):
            body = b"OKX Advanced Multi-Agent is running"
            self.send_response(200)
            self.send_header(
                "Content-Type",
                "text/plain; charset=utf-8",
            )
            self.send_header(
                "Content-Length",
                str(len(body)),
            )
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Keep Render logs clean.
        return


def run_health_server():
    port = int(os.getenv("PORT", "10000"))

    server = HTTPServer(
        ("0.0.0.0", port),
        HealthHandler,
    )

    logging.info(
        "Render health server listening on port %s",
        port,
    )

    server.serve_forever()


def start_health_server():
    thread = threading.Thread(
        target=run_health_server,
        daemon=True,
    )
    thread.start()


# ============================================================
# MAIN
# ============================================================

def main():
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError(
            "Thiếu TELEGRAM_BOT_TOKEN trong environment."
        )

    app = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .build()
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("price", price))
    app.add_handler(CommandHandler("analyze", analyze))
    app.add_handler(CommandHandler("ai", ai_analysis))
    app.add_handler(CommandHandler("agent", agent_command))
    app.add_handler(
        CommandHandler("signal", signal_command)
    )
    app.add_handler(
        CommandHandler("risk", risk_command)
    )
    app.add_handler(
        CommandHandler("position", position_command)
    )
    app.add_handler(
        CommandHandler("closepaper", closepaper_command)
    )
    app.add_handler(
        CommandHandler("auto", auto_command)
    )
    app.add_handler(
        CommandHandler("autostatus", autostatus_command)
    )
    app.add_handler(CommandHandler("performance", performance_command))
    app.add_handler(CommandHandler("jobstats", jobstats_command))
    app.add_handler(
        CommandHandler("stopauto", stopauto_command)
    )
    app.add_handler(
        CommandHandler("stop", stopauto_command)
    )

    logging.info("OKX Advanced Multi-Agent V6 modular starting...")
    logging.info(
        "PAPER ONLY - no OKX private trading API configured."
    )

    # Render Web Service requires an HTTP port.
    # Telegram continues to use long polling independently.
    start_health_server()

    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
