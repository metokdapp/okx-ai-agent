"""BTC-USDT paper-only agent. No authenticated exchange or real-order code."""
import asyncio
import concurrent.futures
import dataclasses
import datetime as dt
import fcntl
import json
import logging
import math
import os
from pathlib import Path
import re
import signal
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

LOG = logging.getLogger('paper')
SYMBOL = 'BTC-USDT'
DEFAULT_MODELS = ('gemini-3.8-flash', 'gemini-3.7-flash', 'gemini-3.6-flash')


def number(value, low, high):
    n = float(value)
    if not math.isfinite(n) or not low <= n <= high:
        raise ValueError('Number outside allowed range')
    return n


@dataclasses.dataclass
class Config:
    initial: float = 1000
    risk: float = 0.005
    allocation: float = 0.25
    daily_loss: float = 0.03
    fee: float = 0.001
    slippage: float = 0.0005
    atr_multiple: float = 1.5
    reward_risk: float = 2
    confidence: float = 70
    report_seconds: int = 900
    ai_daily_limit: int = 1440

    @classmethod
    def env(cls):
        mapping = {
            'initial': ('INITIAL_BALANCE', 10, 1e9),
            'risk': ('RISK_PER_TRADE', 0.0001, 0.02),
            'allocation': ('MAX_ALLOCATION', 0.01, 1),
            'daily_loss': ('DAILY_LOSS_LIMIT', 0.001, 0.2),
            'fee': ('FEE_RATE', 0, 0.01),
            'slippage': ('SLIPPAGE_RATE', 0, 0.01),
            'atr_multiple': ('STOP_ATR_MULTIPLIER', 0.5, 10),
            'reward_risk': ('REWARD_RISK', 1, 5),
            'confidence': ('MIN_CONFIDENCE', 0, 100),
            'report_seconds': ('REPORT_SECONDS', 60, 86400),
            'ai_daily_limit': ('AI_DAILY_LIMIT', 1, 1440),
        }
        c = cls()
        for field, (key, low, high) in mapping.items():
            if os.getenv(key):
                setattr(c, field, number(os.environ[key], low, high))
        return c


def utc_day(now=None):
    return dt.datetime.fromtimestamp(now or time.time(), dt.timezone.utc).date().isoformat()


def http_json(url, payload=None, headers=None, timeout=12):
    data = None if payload is None else json.dumps(payload).encode()
    h = {'User-Agent': 'BTC-Paper-Agent/1.0', **(headers or {})}
    if data is not None:
        h['Content-Type'] = 'application/json'
    req = urllib.request.Request(url, data=data, headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read(2_000_000))


async def market(path, **params):
    # Fixed public market endpoints only. There is no exchange secret or order API.
    if path not in {'ticker', 'candles'}:
        raise ValueError('Not a public market endpoint')
    url = 'https://www.okx.com/api/v5/market/' + path + '?' + urllib.parse.urlencode(params)
    result = await asyncio.to_thread(http_json, url)
    if result.get('code') != '0' or not result.get('data'):
        raise ValueError('OKX response invalid')
    return result['data']


def validate_quote(row, now=None):
    now = time.time() if now is None else now
    stamp = int(row['ts']) / 1000
    bid, ask = number(row['bidPx'], 1, 1e9), number(row['askPx'], 1, 1e9)
    last = number(row['last'], 1, 1e9)
    if not -2 <= now - stamp <= 10 or ask < bid:
        raise ValueError('Stale or invalid ticker')
    return {'bid': bid, 'ask': ask, 'last': last, 'ts': stamp}


def closed_candles(rows, bar_seconds, now=None):
    now = time.time() if now is None else now
    rows = sorted((r for r in rows if str(r[8]) == '1'), key=lambda r: int(r[0]))
    if len(rows) < 100 or len({int(r[0]) for r in rows}) != len(rows):
        raise ValueError('Insufficient or duplicate candles')
    out = []
    for row in rows:
        ts = int(row[0]) / 1000
        o, h, l, c = [number(v, 0.00000001, 1e12) for v in row[1:5]]
        v = number(row[5], 0, 1e15)
        if l > min(o, c) or h < max(o, c) or h < l:
            raise ValueError('Invalid OHLC')
        if out and ts - out[-1][0] != bar_seconds:
            raise ValueError('Candle gap')
        out.append((ts, o, h, l, c, v))
    age = now - (out[-1][0] + bar_seconds)
    if not 0 <= age <= bar_seconds + 30:
        raise ValueError('Stale or future candles')
    return out


def ema(values, period):
    result = [values[0]]
    alpha = 2 / (period + 1)
    for v in values[1:]:
        result.append(result[-1] + alpha * (v - result[-1]))
    return result


def indicators(candles):
    closes = [r[4] for r in candles]
    changes = [b-a for a,b in zip(closes, closes[1:])]
    # Wilder RSI and ATR, initialized with 14-period arithmetic averages.
    gain = sum(max(x, 0) for x in changes[:14])/14
    loss = sum(max(-x, 0) for x in changes[:14])/14
    for x in changes[14:]:
        gain = (gain*13 + max(x, 0))/14
        loss = (loss*13 + max(-x, 0))/14
    rsi = 50 if gain == loss == 0 else (100 if loss == 0 else 100-100/(1+gain/loss))
    tr = [max(row[2]-row[3], abs(row[2]-prev[4]), abs(row[3]-prev[4]))
          for prev,row in zip(candles, candles[1:])]
    atr = sum(tr[:14])/14
    for x in tr[14:]:
        atr = (atr*13+x)/14
    fast, slow = ema(closes,12), ema(closes,26)
    macd = [a-b for a,b in zip(fast,slow)]
    avgvol = sum(r[5] for r in candles[-21:-1])/20
    return {'close': closes[-1], 'ema20': ema(closes,20)[-1],
            'ema50': ema(closes,50)[-1], 'rsi14': rsi, 'atr14': atr,
            'macd_hist': macd[-1]-ema(macd,9)[-1],
            'volume_ratio': candles[-1][5]/avgvol if avgvol else 0}


def parse_decision(raw):
    if not isinstance(raw, dict) or raw.get('action') not in {'BUY','SELL','HOLD'}:
        raise ValueError('Invalid AI action')
    confidence = number(raw.get('confidence'), 0, 100)
    reason = raw.get('reason')
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError('Missing AI reason')
    return {'action': raw['action'], 'confidence': confidence, 'reason': reason[:700]}


async def ask_ai(snapshot, api_key, model, timeout=12):
    if not api_key:
        raise ValueError('AI key missing')
    if not re.fullmatch(r'[a-zA-Z0-9._-]+', model):
        raise ValueError('Invalid model name')
    schema = {'type':'OBJECT', 'properties':{
        'action':{'type':'STRING','enum':['BUY','SELL','HOLD']},
        'confidence':{'type':'NUMBER'}, 'reason':{'type':'STRING'}},
        'required':['action','confidence','reason']}
    body = {
        'systemInstruction':{'parts':[{'text':
            'You are a paper trading analyst for BTC-USDT spot. Decide on a CLOSED 1m candle, '
            'using 5m/15m context. No short, leverage or real trades. Prefer HOLD when unclear. '
            'Account for supplied costs, spread, existing position and trend. SELL only exits '
            'an existing position. Confidence is subjective, NOT a win probability. '
            'Return action, confidence 0..100, and a concise Vietnamese reason. '
            'Never invent prices, external news or guaranteed profit.'}]},
        'contents':[{'role':'user','parts':[{'text':json.dumps(snapshot, allow_nan=False)}]}],
        'generationConfig':{'temperature':0.2,'maxOutputTokens':1024,
                            'responseMimeType':'application/json','responseSchema':schema}}
    result = await asyncio.to_thread(http_json,
        f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent',
        body, {'x-goog-api-key':api_key}, timeout)
    candidate = result['candidates'][0]
    if candidate.get('finishReason') != 'STOP':
        raise ValueError('Incomplete AI response')
    text = ''.join(p.get('text','') for p in candidate['content']['parts'] if not p.get('thought'))
    return parse_decision(json.loads(text))


class AIUnavailable(Exception):
    pass


def gemini_failure(error, model):
    """Return safe label, cooldown seconds and whether the whole key must wait."""
    if not isinstance(error, urllib.error.HTTPError):
        return ('TIMEOUT' if isinstance(error, TimeoutError) else 'INVALID_RESPONSE', 60, False)
    code = error.code
    try:
        body = json.loads(error.read(32768)).get('error', {})
        details = body.get('details', [])
        reasons = {str(d.get('reason','')) for d in details if isinstance(d,dict)}
    except (ValueError, AttributeError, TypeError):
        details, reasons = [], set()
    if code == 401 or reasons.intersection({'API_KEY_INVALID','API_KEY_EXPIRED','API_KEY_SERVICE_BLOCKED','SERVICE_DISABLED','BILLING_DISABLED'}):
        return ('KEY_OR_PROJECT_ERROR',3600,True)
    if code == 429:
        delay = 900.0
        try:
            delay = max(delay, float(error.headers.get('Retry-After',0)))
        except (ValueError,TypeError,AttributeError):
            pass
        violations = []
        for item in details:
            if not isinstance(item,dict):
                continue
            try:
                delay = max(delay,float(str(item.get('retryDelay','0s')).removesuffix('s')))
            except ValueError:
                pass
            violations.extend(item.get('violations',[]))
        # Shared/unknown quota applies to the whole router; do not bypass it by switching.
        model_scoped = bool(violations) and all(
            isinstance(v,dict) and v.get('quotaDimensions',{}).get('model') in {model,'models/'+model}
            for v in violations)
        if any('perday' in str(v.get('quotaId','')).lower().replace('_','') for v in violations if isinstance(v,dict)):
            delay = max(delay,86400)
        return ('QUOTA_MODEL' if model_scoped else 'QUOTA_PROJECT',delay,not model_scoped)
    if code in (403,404):
        return ('MODEL_UNAVAILABLE_'+str(code),21600,False)
    if code == 400:
        return ('MODEL_REQUEST_400',3600,False)
    return ('HTTP_'+str(code),60,False)


class Book:
    def __init__(self, path, config):
        self.c = config
        self.db = sqlite3.connect(path)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY, data TEXT NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, ts REAL, kind TEXT, data TEXT)')
        self.db.execute('CREATE TABLE IF NOT EXISTS outbox (id INTEGER PRIMARY KEY, text TEXT NOT NULL)')
        row = self.db.execute('SELECT data FROM state WHERE id=1').fetchone()
        self.s = json.loads(row[0]) if row else {
            'initial':config.initial, 'cash':config.initial, 'position':None,
            'peak':config.initial, 'max_dd':0, 'day':utc_day(),
            'day_start':config.initial, 'day_halted':False,
            'last_candle':0, 'cycles':0, 'ai_calls':0, 'ai_failures':0,
            'trades':0,'wins':0,'pnl':0, 'paused':False, 'last_exit':0,
            'last_report':0, 'tg_offset':0, 'last_decision':None,
        }
        self.save()

    def save(self, event=None, notice=None):
        with self.db:
            self.db.execute('INSERT OR REPLACE INTO state VALUES(1,?)', (json.dumps(self.s,allow_nan=False),))
            if event:
                self.db.execute('INSERT INTO events(ts,kind,data) VALUES(?,?,?)',
                                (time.time(),event[0],json.dumps(event[1],allow_nan=False)))
            if notice:
                self.db.execute('INSERT INTO outbox(text) VALUES(?)',(notice,))
            # Bound accumulated notifications while Telegram isn't configured.
            self.db.execute('DELETE FROM outbox WHERE id NOT IN (SELECT id FROM outbox ORDER BY id DESC LIMIT 100)')

    def equity(self, bid):
        p = self.s['position']
        return self.s['cash'] + (p['qty']*bid*(1-self.c.slippage)*(1-self.c.fee) if p else 0)

    def mark(self, bid, day=None):
        eq = self.equity(bid)
        day = day or utc_day()
        if day != self.s['day']:
            self.s.update(day=day, day_start=eq, day_halted=False, ai_calls=0)
        self.s['peak'] = max(self.s['peak'], eq)
        self.s['max_dd'] = max(self.s['max_dd'], 1-eq/self.s['peak'])
        if eq <= self.s['day_start']*(1-self.c.daily_loss):
            self.s['day_halted'] = True
        return eq

    def buy(self, q, atr, reason):
        c,s = self.c,self.s
        self.mark(q['bid'])
        if s['position'] or s['paused'] or s['day_halted'] or time.time()-s['last_exit'] < 60:
            return False
        entry = q['ask']*(1+c.slippage)
        # Minimum stop width avoids tiny stops overwhelmed by round-trip friction.
        distance = max(number(atr,0,1e9)*c.atr_multiple,
                       entry*(2*c.fee+2*c.slippage+0.001))
        stop = entry-distance
        if stop <= 0:
            return False
        exit_stop = stop*(1-c.slippage)
        loss_per_unit = entry*(1+c.fee)-exit_stop*(1-c.fee)
        qty = min(self.equity(q['bid'])*c.risk/loss_per_unit,
                  s['cash']*c.allocation/(entry*(1+c.fee)))
        if qty*entry < 5:
            return False
        cost = qty*entry*(1+c.fee)
        s['cash'] -= cost
        s['position'] = {'qty':qty,'entry':entry,'cost':cost,'stop':stop,
                         'target':entry+distance*c.reward_risk,'opened':time.time()}
        self.save(('BUY',dict(s['position'])),
                  f'🧪 BUY MÔ PHỎNG BTC-USDT\nGiá: {entry:,.2f}\nBTC: {qty:.8f}\nSL: {stop:,.2f} | TP: {s["position"]["target"]:,.2f}\n{reason}')
        return True

    def sell(self, q, reason):
        p = self.s['position']
        if not p:
            return False
        fill = q['bid']*(1-self.c.slippage)
        proceeds = p['qty']*fill*(1-self.c.fee)
        pnl = proceeds-p['cost']
        self.s['cash'] += proceeds
        self.s['pnl'] += pnl
        self.s['trades'] += 1
        self.s['wins'] += int(pnl > 0)
        self.s['position'] = None
        self.s['last_exit'] = time.time()
        self.mark(q['bid'])
        self.save(('SELL',{'entry':p,'exit':fill,'net_pnl':pnl,'reason':reason}),
                  f'🧪 SELL MÔ PHỎNG BTC-USDT\nGiá: {fill:,.2f}\nPnL sau phí: {pnl:+.4f} USDT\n{reason}')
        return True

    def protect(self, q):
        self.mark(q['bid'])
        p = self.s['position']
        if not p:
            return
        if q['bid'] <= p['stop']:
            self.sell(q,'Cắt lỗ; khớp theo giá bid hiện tại')
        elif q['bid'] >= p['target']:
            self.sell(q,'Chốt lời')
        elif self.s['day_halted']:
            self.sell(q,'Dừng do giới hạn lỗ ngày UTC')


class Agent:
    def __init__(self, config, book):
        self.c,self.book = config,book
        self.quote = None
        self.ai_task = None
        self.running = True
        self.market_error = None
        self.last_market_ok = 0
        self.token = os.getenv('TELEGRAM_BOT_TOKEN','').strip()
        self.chat = book.s.get('telegram_chat_id','')
        if self.token and not self.chat:
            LOG.info('TELEGRAM SETUP: Nhắn /start trong chat riêng để liên kết tài khoản đầu tiên.')
        self.ai_key = os.getenv('GEMINI_API_KEY','').strip()
        configured = os.getenv('GEMINI_MODELS','').strip()
        primary = os.getenv('GEMINI_MODEL',DEFAULT_MODELS[0]).strip()
        names = configured.split(',') if configured else [primary,*DEFAULT_MODELS]
        # Never echo arbitrary env values: a key may be pasted into the model field.
        valid = [n.strip() for n in names if re.fullmatch(r'gemini-[a-z0-9][a-z0-9.-]{0,55}',n.strip())]
        if len(valid) != len(names):
            LOG.warning('Invalid Gemini model setting ignored (value hidden); put the API key in GEMINI_API_KEY')
        self.models = list(dict.fromkeys(valid))[:8] or list(DEFAULT_MODELS)
        if not re.fullmatch(r'gemini-[a-z0-9][a-z0-9.-]{0,55}',book.s.get('ai_last_model','')):
            book.s.pop('ai_last_model',None)
        previous_error = book.s.get('ai_last_error','')
        if previous_error and not re.fullmatch(r'gemini-[a-z0-9][a-z0-9.-]{0,55}: [A-Z0-9_]+',previous_error):
            book.s['ai_last_error'] = 'Lỗi cấu hình trước đó (đã ẩn giá trị)'


    def fresh(self):
        return self.quote is not None and -2 <= time.time()-self.quote['ts'] <= 10

    def report(self):
        s,c = self.book.s,self.c
        q = self.quote
        eq = self.book.equity(q['bid']) if q else None
        p = s['position']
        decision = s['last_decision'] or {'action':'HOLD','reason':'Đang chờ nến đóng'}
        return '\n'.join([
            '🧪 AI PAPER AGENT • BTC-USDT • 1M',
            dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'),
            'Giá: '+(f'{q["last"]:,.2f}' if q else 'chưa có'),
            'Dữ liệu: '+('mới' if self.fresh() else 'cũ/gián đoạn — chặn giao dịch'),
            f'Vốn đầu: {s["initial"]:.2f} | Tiền mặt: {s["cash"]:.2f}',
            'Tổng vốn ước tính sau phí thoát: '+(f'{eq:.2f} USDT' if eq is not None else 'chưa định giá'),
            f'PnL đã chốt sau phí: {s["pnl"]:+.4f} USDT',
            'PnL tổng: '+(f'{eq-s["initial"]:+.4f} USDT' if eq is not None else 'chưa định giá'),
            f'Lệnh đóng: {s["trades"]} | Thắng: {s["wins"]} | Max DD: {s["max_dd"]:.2%}',
            'Vị thế: '+(f'{p["qty"]:.8f} BTC | SL {p["stop"]:.2f} | TP {p["target"]:.2f}' if p else 'không'),
            f'AI: {decision["action"]} — {decision["reason"]}',
            'Model thành công gần nhất: '+s.get('ai_last_model','chưa có'),
            'Model dự phòng: '+', '.join(self.models),
            'Lỗi model gần nhất: '+s.get('ai_last_error','không'),
            f'AI hôm nay: {s["ai_calls"]}/{int(c.ai_daily_limit)} | Lỗi tích lũy: {s["ai_failures"]}',
            'Trạng thái: '+('PAUSED' if s['paused'] else 'CHẶN MUA: lỗ ngày' if s['day_halted'] else 'ACTIVE'),
            'Cấu hình AI: '+('đã có key' if self.ai_key else 'CHƯA CÓ GEMINI_API_KEY — HOLD'),
            'Vốn giả. Confidence AI không phải xác suất thắng.',
        ])

    async def prices(self):
        last_save = 0
        while self.running:
            start = time.monotonic()
            try:
                q = validate_quote((await market('ticker',instId=SYMBOL))[0])
                self.quote = q
                self.book.protect(q)
                self.last_market_ok = time.time()
                self.market_error = None
                if time.time()-last_save > 10:
                    self.book.save()
                    last_save = time.time()
            except Exception as e:
                self.market_error = type(e).__name__
                # Never log raw exception text: request URLs may contain credentials elsewhere.
                LOG.warning('Market unavailable (%s); trading blocked until fresh quote',type(e).__name__)
            await asyncio.sleep(max(0.1,1-(time.monotonic()-start)))

    async def ai_decision(self, candle_id, snapshot):
        s = self.book.s
        if not self.ai_key:
            raise AIUnavailable('Chưa có GEMINI_API_KEY')
        if time.time() < s.get('ai_global_until',0):
            raise AIUnavailable('Đang chờ hạn mức/quyền API toàn project')
        cooldown = s.setdefault('ai_cooldowns',{})
        preferred = s.get('ai_last_model')
        models = sorted(self.models,key=lambda n:n != preferred)
        for model in models:
            if cooldown.get(model,0) > time.time():
                continue
            remaining = candle_id+110-time.time()
            if remaining < 2:
                break
            if s['ai_calls'] >= self.c.ai_daily_limit:
                raise AIUnavailable('Đã đạt AI_DAILY_LIMIT gồm cả lần thử dự phòng')
            s['ai_calls'] += 1
            self.book.save()  # Every attempt counted durably, including fallback failures.
            try:
                result = await ask_ai(snapshot,self.ai_key,model,timeout=min(12,remaining))
                s['ai_last_model'] = model
                cooldown.pop(model,None)
                self.book.save()
                return {**result,'model':model}
            except Exception as error:
                label,delay,global_wait = gemini_failure(error,model)
                cooldown[model] = time.time()+delay
                s['ai_last_error'] = model+': '+label
                if global_wait:
                    s['ai_global_until'] = time.time()+delay
                self.book.save(('MODEL_ERROR',{'model':model,'error':label,'cooldown_seconds':delay}))
                LOG.warning('Gemini %s: %s; cooldown %.0fs',model,label,delay)
                if global_wait:
                    break
        raise AIUnavailable('Không có model sẵn sàng; '+s.get('ai_last_error','đã hết thời gian phân tích'))

    async def analyze(self, candle_id, snapshot):
        s = self.book.s
        try:
            decision = await self.ai_decision(candle_id,snapshot)
            if time.time()-(candle_id+60) > 50:
                raise ValueError('Decision expired')
            s['last_decision'] = decision
            self.book.save(('DECISION',{'candle':candle_id,**decision}))
            if not self.fresh():
                return
            q = self.quote
            # Re-evaluate current position and protections after the AI await.
            self.book.protect(q)
            if s['paused']:
                return
            if decision['confidence'] < self.c.confidence:
                return
            if decision['action'] == 'SELL':
                self.book.sell(q,decision['reason'])
            elif decision['action'] == 'BUY' and (q['ask']-q['bid'])/q['bid'] <= 0.002:
                self.book.buy(q,snapshot['1m']['atr14'],decision['reason'])
        except Exception as e:
            s['ai_failures'] += 1
            s['last_decision'] = {'action':'HOLD','reason':str(e) if isinstance(e,AIUnavailable) else 'AI lỗi/chậm: '+type(e).__name__}
            self.book.save(('AI_ERROR',{'type':type(e).__name__}))
            LOG.warning('AI HOLD (%s)',type(e).__name__)

    async def candles(self):
        while self.running:
            try:
                if self.ai_task and not self.ai_task.done():
                    await asyncio.sleep(2)
                    continue
                one = closed_candles(await market('candles',instId=SYMBOL,bar='1m',limit=200),60)
                candle_id = int(one[-1][0])
                s = self.book.s
                if candle_id > s['last_candle'] and time.time()-(candle_id+60) < 25:
                    # Claim the candle durably BEFORE calling AI: restart cannot repeat it.
                    s['last_candle'] = candle_id
                    s['cycles'] += 1
                    self.book.save()
                    if not self.ai_key or s['paused'] or not self.fresh():
                        continue
                    self.book.mark(self.quote['bid'])
                    if s['ai_calls'] >= self.c.ai_daily_limit:
                        s['last_decision'] = {'action':'HOLD','reason':'Đã đạt AI_DAILY_LIMIT'}
                        self.book.save()
                        continue
                    five, fifteen = await asyncio.gather(
                        market('candles',instId=SYMBOL,bar='5m',limit=200),
                        market('candles',instId=SYMBOL,bar='15m',limit=200))
                    snapshot = {'symbol':SYMBOL,'1m':indicators(one),
                        '5m':indicators(closed_candles(five,300)),
                        '15m':indicators(closed_candles(fifteen,900)),
                        'position':s['position'], 'fee_per_side':self.c.fee,
                        'slippage_per_side':self.c.slippage,
                        'spread':(self.quote['ask']-self.quote['bid'])/self.quote['bid']}
                    self.ai_task = asyncio.create_task(self.analyze(candle_id,snapshot))
            except Exception as e:
                LOG.warning('Candle analysis skipped (%s)',type(e).__name__)
            await asyncio.sleep(4)

    async def telegram(self, method, payload):
        result = await asyncio.to_thread(http_json,
            f'https://api.telegram.org/bot{self.token}/{method}',payload,None,20)
        if not result.get('ok'):
            raise ValueError('Telegram API failed')
        return result['result']

    def pair_telegram(self, sender, chat_type, text):
        """First private /start binds the recipient once, as requested by the owner."""
        parts = text.split()
        if (self.chat or not self.token or chat_type != 'private'
                or not re.fullmatch(r'[1-9][0-9]*',sender)
                or not parts or parts[0].split('@')[0] != '/start'):
            return False
        self.book.s['telegram_chat_id'] = sender
        self.book.save()  # Durable binding before enabling report delivery.
        self.chat = sender
        return True

    async def commands(self):
        if not self.token:
            return
        while self.running:
            try:
                updates = await self.telegram('getUpdates',{
                    'offset':self.book.s['tg_offset'],'timeout':10,
                    'allowed_updates':['message']})
                for update in updates:
                    msg = update.get('message',{})
                    chat = msg.get('chat',{})
                    text = msg.get('text','').strip()
                    sender = str(chat.get('id',''))
                    private = chat.get('type') == 'private'
                    if self.pair_telegram(sender,chat.get('type'),text):
                        await self.telegram('sendMessage',{'chat_id':sender,'text':
                            '✅ Đã liên kết Telegram. Bot tự lưu nơi nhận báo cáo; không cần nhập Chat ID.\n'+self.report()})
                    elif private and text.split(' ')[0].split('@')[0] in {'/start','/whoami'} and not self.chat:
                        await self.telegram('sendMessage',{'chat_id':sender,'text':
                            'Chưa liên kết. Gửi /start trong chat riêng để nhận báo cáo.'})
                    elif private and self.chat and sender == self.chat:
                        cmd = text.split(' ')[0].split('@')[0]
                        response = None
                        if cmd in {'/start','/status','/report'}:
                            response = self.report()
                        elif cmd == '/pause':
                            self.book.s['paused'] = True
                            response = 'Đã dừng quyết định AI. SL/TP của vị thế đang mở vẫn được kiểm tra.'
                        elif cmd == '/resume':
                            self.book.s['paused'] = False
                            response = 'Đã bật phân tích từ nến 1m đóng tiếp theo.'
                        elif cmd == '/settings':
                            response = json.dumps(dataclasses.asdict(self.c),indent=2)+'\nĐổi biến môi trường Railway và Redeploy. Vốn đầu chỉ áp dụng cho dữ liệu mới.'
                        elif cmd == '/trades':
                            rows = self.book.db.execute("SELECT data FROM events WHERE kind='SELL' ORDER BY id DESC LIMIT 5").fetchall()
                            response = '5 lệnh đóng gần nhất:\n'+'\n'.join(f'{json.loads(r[0])["net_pnl"]:+.4f} USDT — {json.loads(r[0])["reason"]}' for r in rows) if rows else 'Chưa có lệnh đóng.'
                        elif cmd == '/help':
                            response = '/status /report /trades /settings /pause /resume /whoami'
                        elif cmd == '/whoami':
                            response = 'Chat ID: '+sender
                        if response:
                            await self.telegram('sendMessage',{'chat_id':self.chat,'text':response[:4000]})
                    self.book.s['tg_offset'] = update['update_id']+1
                    self.book.save()
            except Exception as e:
                LOG.warning('Telegram commands unavailable (%s)',type(e).__name__)
                await asyncio.sleep(5)

    async def notifications(self):
        while self.running:
            if self.token and self.chat:
                try:
                    s = self.book.s
                    if time.time()-s['last_report'] >= self.c.report_seconds:
                        s['last_report'] = time.time()
                        self.book.save(notice=self.report())
                    row = self.book.db.execute('SELECT id,text FROM outbox ORDER BY id LIMIT 1').fetchone()
                    if row:
                        await self.telegram('sendMessage',{'chat_id':self.chat,'text':row[1][:4000]})
                        with self.book.db:
                            self.book.db.execute('DELETE FROM outbox WHERE id=?',(row[0],))
                except Exception as e:
                    LOG.warning('Telegram delivery pending (%s)',type(e).__name__)
            await asyncio.sleep(3)

    async def run(self):
        tasks = [asyncio.create_task(f()) for f in
                 (self.prices,self.candles,self.commands,self.notifications)]
        try:
            while self.running:
                for task in tasks:
                    if task.done() and task.exception():
                        raise task.exception()
                await asyncio.sleep(1)
        finally:
            for task in tasks + ([self.ai_task] if self.ai_task else []):
                task.cancel()
            await asyncio.gather(*tasks, *([self.ai_task] if self.ai_task else []),return_exceptions=True)
            self.book.save()
            self.book.db.close()


def main():
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    directory = Path(os.getenv('DATA_DIR','./data'))
    directory.mkdir(parents=True,exist_ok=True)
    lock = (directory/'agent.lock').open('w')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('Another worker holds this data volume; run one replica only')
    config = Config.env()
    bot = Agent(config,Book(directory/'paper.sqlite3',config))
    class Health(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != '/health':
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type','application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'process':'running','paper_only':True,
                'market_fresh':bot.fresh(),'telegram_configured':bool(bot.token and bot.chat),
                'ai_configured':bool(bot.ai_key)}).encode())
        def log_message(self,*args):
            pass
    server = ThreadingHTTPServer(('0.0.0.0',int(os.getenv('PORT','8080'))),Health)
    Thread(target=server.serve_forever,daemon=True).start()
    LOG.info('PAPER ONLY BTC-USDT 1m; Telegram=%s AI=%s storage=%s',
             bool(bot.token and bot.chat),bool(bot.ai_key),directory)
    async def start():
        loop = asyncio.get_running_loop()
        # Separate network threads keep slow AI and Telegram from blocking price protection.
        loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=8))
        for sig in (signal.SIGTERM,signal.SIGINT):
            loop.add_signal_handler(sig,lambda:setattr(bot,'running',False))
        await bot.run()
    try:
        asyncio.run(start())
    finally:
        server.shutdown()
        lock.close()


if __name__ == '__main__':
    main()
