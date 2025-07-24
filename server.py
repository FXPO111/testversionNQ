import os
import time
import uuid
import logging, builtins
import numpy as np
import random
from collections import deque

from flask import Flask, send_from_directory, request
from flask_socketio import SocketIO, emit
from flask import Flask, send_from_directory, request, jsonify

# ── тута свечки ─────────────────────────────────────
from candle_db import init_db, insert_candle, load_candles
# ────────────────────────────────────────────────────

# ── silence per-trade spam ─────────────
#orig_print = builtins.print
#builtins.print = lambda *a, **k: None               
#logging.disable(logging.CRITICAL)                  
# ─────────────────────────────────

# ─────────────────────────────────
from trade_logger import _flush as flush_trades
# ─────────────────────────────────

from order import Order, OrderSide, OrderType
from order_book import OrderBook
from market_context import MarketContextAdvanced
from market_maker import MarketMakerManager
#from retail_trader import RetailTrader
from trend_follower import TrendFollowerAgent
from liquidity_hunter import LiquidityHunterAgent
from high_freq import HighFreqAgent
from impulsive_aggressor import ImpulsiveAggressorAgent
#from fair_value_anchor import FairValueAnchorAgent
from spread_arbitrage import SpreadArbitrageAgent
from passive_depth_provider import PassiveDepthProvider
#from base_agent import BaseAgent
from market_order_agent import MarketOrderAgent
from liquidity_manager import LiquidityManagerAgent
from liquidity_manipulator import LiquidityManipulator
from smart_money import SmartMoneyManager
#from momentum_igniter import SmartSidewaysSqueezeAgent
from bank_agent import BankAgentManager
from corporate_vwap_agent import CorporateVWAPAgent
from trend_impuls import TrendAgentManager
from bank_agent_v2 import BankAgentManagerv2

# --- Логирование ---
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(name)s: %(message)s',
)
logger = logging.getLogger("OrderBookServer")

# --- Flask + SocketIO ---
app = Flask(__name__, static_folder='.', template_folder='.')
socketio = SocketIO(app, cors_allowed_origins="*")

# --- Книга и стартовые лимитные ордера ---
order_book = OrderBook()
order_book.market_context = MarketContextAdvanced()

seed_order_buy = Order(
    order_id=str(uuid.uuid4()),
    agent_id="seed_buy",
    side=OrderSide.BID,
    volume=10.0,
    price=99.50,
    order_type=OrderType.LIMIT,
    ttl=None
)
seed_order_sell = Order(
    order_id=str(uuid.uuid4()),
    agent_id="seed_sell",
    side=OrderSide.ASK,
    volume=10.0,
    price=100.50,
    order_type=OrderType.LIMIT,
    ttl=None
)
order_book.add_order(seed_order_buy)
order_book.add_order(seed_order_sell)

market_phases_history = []
_last_phase_name = None
_last_phase_start = None
_last_microphase = None


# --- Инициализация начальных сделок ---
def inject_initial_trades():
    logger.info("[Init] Injecting initial market orders to trigger first trades")
    market_orders = [
        Order(str(uuid.uuid4()), "seeder_m1", OrderSide.BID, 2.0, None, OrderType.MARKET, None),
        Order(str(uuid.uuid4()), "seeder_m2", OrderSide.ASK, 1.0, None, OrderType.MARKET, None),
        Order(str(uuid.uuid4()), "seeder_m3", OrderSide.BID, 1.0, None, OrderType.MARKET, None),
        Order(str(uuid.uuid4()), "seeder_m4", OrderSide.ASK, 2.0, None, OrderType.MARKET, None),
    ]
    for mo in market_orders:
        order_book.add_order(mo)

# --- Список агентов ---

#retail = RetailTrader(agent_id="retail1", init_cash=1_000_000.0)
trend  = TrendFollowerAgent(agent_id="trend1", capital=20_000_000.0, num_subagents=5)
hf1    = LiquidityHunterAgent(agent_id="hf1", capital=1_000_000.0)
hf2    = HighFreqAgent(agent_id="hf2", capital=20_000.0)
aggressor = ImpulsiveAggressorAgent(agent_id="imp1", capital=30_500_000.0)
#anchor = FairValueAnchorAgent(agent_id="anchor1", capital=1_000_000.0)
arb = SpreadArbitrageAgent(agent_id="arb1", capital=1_000_000.0)
depth_provider = PassiveDepthProvider(agent_id="depth1", capital=1.0)
#base_agent = BaseAgent(agent_id="base1", total_capital=6_000_000.0, participants_per_agent=1_000)
market_order_agent = MarketOrderAgent(agent_id="market_order1", capital=10_500_000.0)
liquidity_manipulator = LiquidityManipulator(agent_id="manipulator1", capital=20_000_000.0) 
smart_money = SmartMoneyManager(agent_id="smart", total_capital=60_000_000.0, num_desks=4)
#momentum_igniter = SmartSidewaysSqueezeAgent(agent_id="momentum1", capital=20_000_000.0)
bank_agent = BankAgentManager(agent_id="bank1", total_capital=1_000_000_000.0)
market_maker = MarketMakerManager(agent_id="mm1", total_capital=70_000_000.0)
corp_vwap = CorporateVWAPAgent(agent_id="corp_vwap", total_capital=80_000_000.0, legs=5)
liquidity_manager = LiquidityManagerAgent(agent_id="liquidity_manager1", capital=1_000_000.0)
trend_imp = TrendAgentManager(agent_id="trend_imp1")
bank_agent_v2 = BankAgentManagerv2(agent_id="bankadv1", total_capital=1_000_000_000.0)

AGENTS = [
    market_maker,
    #retail,
    trend,
    hf1,
    hf2,
    aggressor,
    #anchor,
    arb,
    depth_provider,
    #base_agent,
    market_order_agent,
    liquidity_manager,
    liquidity_manipulator,
    smart_money,  
    #momentum_igniter,  
    bank_agent,
    corp_vwap,
    trend_imp,
    bank_agent_v2
]

# --- OHLC (свечи) ---

class PhaseTracker:
    def __init__(self, maxlen=300):
        self.current_phase = None
        self.start_time = None
        self.tracked_phases = deque(maxlen=maxlen)

    def update(self, market_context):
        now = int(time.time())
        phase = market_context.phase.name if market_context and market_context.phase else "undefined"
        micro = market_context.phase.reason.get("microphase", "") if market_context and market_context.phase and market_context.phase.reason else ""

        if self.current_phase != (phase, micro):
            if self.current_phase:
                self.tracked_phases.append({
                    "start": self.start_time,
                    "end": now,
                    "phase": self.current_phase[0],
                    "microphase": self.current_phase[1]
                })
            self.current_phase = (phase, micro)
            self.start_time = now

    def export(self):
        now = int(time.time())
        exported = list(self.tracked_phases)
        if self.current_phase:
            exported.append({
                "start": self.start_time,
                "end": now,
                "phase": self.current_phase[0],
                "microphase": self.current_phase[1]
            })
        return exported

phase_tracker = PhaseTracker()

# --- HTTP ---
@app.route('/')
def index():
    return send_from_directory(os.getcwd(), 'index.html')

@app.route('/styles.css')
def css():
    return send_from_directory(os.getcwd(), 'styles.css')

@app.route('/scripts.js')
def js():
    return send_from_directory(os.getcwd(), 'scripts.js')

# --- Socket.IO ---
@socketio.on('connect')
def on_connect():
    sid = request.sid
    logger.info(f"Client connected: {sid}")
    emit("orderbook_update", order_book.get_order_book_snapshot(depth=10))
    emit("history", list(order_book.trade_history)[-200:])

@socketio.on('disconnect')
def on_disconnect():
    sid = request.sid
    logger.info(f"Client disconnected: {sid}")

@socketio.on('add_order')
def handle_add_order(data):
    try:
        side = OrderSide.BID if data.get('side') == 'buy' else OrderSide.ASK
        order_type = OrderType.LIMIT if data.get('order_type', 'limit') == 'limit' else OrderType.MARKET
        order = Order(
            order_id=str(uuid.uuid4()),
            agent_id=data.get('agent_id'),
            side=side,
            volume=float(data.get('volume')),
            price=None if data.get('price') is None else float(data.get('price')),
            order_type=order_type,
            ttl=None
        )
        order_book.add_order(order)
        emit("confirmation", {"message": f"Order {order.order_id} added"})
    except Exception as e:
        logger.error(f"add_order error: {e}")
        emit("error", {"message": str(e)})

@socketio.on('cancel_order')
def handle_cancel_order(data):
    oid = data.get('order_id')
    if not oid:
        emit("error", {"message": "Missing order_id"})
        return
    if order_book.cancel_order(oid):
        emit("confirmation", {"message": f"Order {oid} cancelled"})
    else:
        emit("error", {"message": f"Order {oid} not found or inactive"})

@app.route("/candles", methods=["GET"])
def get_candles():
    tf = int(request.args.get("interval", 5))
    candles = candle_managers[tf].get_candles()
    return jsonify([
        {
            "timestamp": c.timestamp,
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "volume": c.volume
        } for c in candles
    ])

@socketio.on('candles')
def handle_candles():
    # Отправляем данные о свечах клиенту
    socketio.emit('candles', candles_data)

@app.route("/market_phases")
def get_market_phases():
    # последние 200 фаз, чтобы не грузить лишнего
    return jsonify(phase_tracker.export())



# --- Fill уведомление ---
def notify_agent_fill(trade):
    price = trade['price']
    volume = trade['volume']
    for agent in AGENTS:
        if agent.agent_id == trade['buy_agent']:
            agent.on_order_filled(trade['buy_order_id'], price, volume, OrderSide.BID)
        if agent.agent_id == trade['sell_agent']:
            agent.on_order_filled(trade['sell_order_id'], price, volume, OrderSide.ASK)

# --- Matching и трансляция ---
def match_and_broadcast():
    global _last_phase_name, _last_phase_start, _last_microphase
    flush_trades.last = time.time()

    while True:
        try:
            order_book.tick()
            trades = order_book.match()

            # ---------- агрегированная сводка раз в минуту ----------
            now = time.time()
            if now - getattr(flush_trades, "last", 0) >= 60:
                flush_trades(now)
                flush_trades.last = now

            # === ФАЗОВЫЙ КОНТЕКСТ ===
            if hasattr(order_book, "market_context"):
                ctx = order_book.market_context
                snapshot = order_book.get_order_book_snapshot()
                sweep_occurred = bool(trades)
                order_book.market_context.update(snapshot, sweep_occurred)
                phase_tracker.update(ctx)
                # Берём name и microphase
                phase = ctx.phase.name if ctx.phase else "undefined"
                micro = ctx.phase.reason.get("microphase", "") if ctx.phase and hasattr(ctx.phase, "reason") else ""
                now = int(time.time())
                if phase != _last_phase_name or micro != _last_microphase:
                    # Заканчиваем предыдущую фазу
                    if _last_phase_name is not None and _last_phase_start is not None:
                        market_phases_history.append({
                            "start": _last_phase_start,
                            "end": now,
                            "phase": _last_phase_name,
                            "microphase": _last_microphase or ""
                        })
                    _last_phase_name = phase
                    _last_phase_start = now
                    _last_microphase = micro

            if trades:
                for t in trades:
                    notify_agent_fill(t)
                    socketio.emit("trade", t)  # Отправляем сделку клиенту
                    for tf, cm in candle_managers.items():
                        cm.update(t['price'], t['volume'])  # Обновляем свечку
                        if cm.current_candle:
                            insert_candle(conn, tf, cm.current_candle)  # Сохраняем последнюю свечку

            # Отправляем свечи
            for tf, cm in candle_managers.items():
                socketio.emit("candles", [
                    {
                        "timestamp": c.timestamp,
                        "open": c.open,
                        "high": c.high,
                        "low": c.low,
                        "close": c.close,
                        "volume": c.volume
                    }
                    for c in cm.get_candles()
                ])

            socketio.emit("orderbook_update", order_book.get_order_book_snapshot(depth=10))

        except Exception as e:
            logger.error(f"match loop error: {e}")

        socketio.sleep(0.3)


# --- Отправка свечей ---


# --- Агенты: разный ритм для каждого ---
def agents_loop():
    next_call = {a.agent_id: 0 for a in AGENTS}
    while True:
        now = time.time()
        for agent in AGENTS:
            interval = 0.5
            if now >= next_call[agent.agent_id]:
                try:
                    new_orders = agent.generate_orders(order_book, order_book.market_context)
                    for order in new_orders:
                        order_book.add_order(order)
                except Exception as e:
                    logger.error(f"[Agent {agent.agent_id}] error: {e}")
                next_call[agent.agent_id] = now + interval
        socketio.sleep(0.2)


# --- Начальная ликвидность: 6 уровней по 20 ---
def inject_initial_liquidity():
    base = 100.0
    for i in range(1, 7):
        order_book.add_order(Order(
            order_id=str(uuid.uuid4()), agent_id=f"seed_buy_{i}",
            side=OrderSide.BID,
            volume=20.0,
            price=round(base - i * 0.5, 2),
            order_type=OrderType.LIMIT, ttl=None
        ))
        order_book.add_order(Order(
            order_id=str(uuid.uuid4()), agent_id=f"seed_sell_{i}",
            side=OrderSide.ASK,
            volume=20.0,
            price=round(base + i * 0.5, 2),
            order_type=OrderType.LIMIT, ttl=None
        ))


class Candle:
    def __init__(self, timestamp, open, high, low, close, volume):
        self.timestamp = timestamp
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume

class CandleManager:
    def __init__(self, interval_seconds=5):
        self.interval = interval_seconds
        self.current_candle = None
        self.history = []

    def update(self, trade_price, trade_volume, trade_time=None):
        now = trade_time or time.time()
        bucket = int(now // self.interval) * self.interval

        if not self.current_candle or self.current_candle.timestamp != bucket:
            if self.current_candle:
                self.history.append(self.current_candle)
            self.current_candle = Candle(
                timestamp=bucket,
                open=self.current_candle.close if self.current_candle else trade_price,
                high=trade_price,
                low=trade_price,
                close=trade_price,
                volume=trade_volume
            )
        else:
            c = self.current_candle
            c.high = max(c.high, trade_price)
            c.low = min(c.low, trade_price)
            c.close = trade_price
            c.volume += trade_volume

    def get_candles(self, limit=1000):
        return self.history[-limit:] + ([self.current_candle] if self.current_candle else [])

    def tick(self, last_price=None):
        now = time.time()
        bucket = int(now // self.interval) * self.interval

        if not self.current_candle or self.current_candle.timestamp != bucket:
            if self.current_candle:
                self.history.append(self.current_candle)

            ref_price = self.current_candle.close if self.current_candle else last_price or 100.0

            self.current_candle = Candle(
                timestamp=bucket,
                open=ref_price,
                high=ref_price,
                low=ref_price,
                close=ref_price,
                volume=0.0
            )

def candle_tick_loop():
    while True:
        # Предполагаем, что best mid-price = средняя между bid/ask
        bid = order_book._best_bid_price()
        ask = order_book._best_ask_price()
        mid = (bid + ask) / 2 if bid and ask else None

        for cm in candle_managers.values():
            cm.tick(last_price=mid)

        socketio.sleep(1.0)



# --- Инициализация менеджеров таймфреймов ---
conn = init_db()

candle_managers = {
    tf: CandleManager(interval_seconds=tf) for tf in [1, 5, 15, 60, 300]
}
for tf, cm in candle_managers.items():
    cm.history = load_candles(conn, tf)


# --- MAIN ---
if __name__ == '__main__':
    inject_initial_liquidity()
    inject_initial_trades()
    socketio.start_background_task(match_and_broadcast)
    socketio.start_background_task(agents_loop)
    socketio.start_background_task(candle_tick_loop)

    logger.info("Starting server on http://0.0.0.0:8000 …")
    socketio.run(app, host='0.0.0.0', port=8000)
