import uuid
import random
import time
from collections import deque
from order import Order, OrderSide, OrderType

"""smart_money.py — Real‑Money / Institutional Layer (final)
=============================================================
Полностью прокачанный SmartMoney:
    • VWAP/TWAP‑дрип,  адаптивный к глубине и времени суток.
    • Momentum‑join ― MARKET, если BID/ASK ≥ 2× и давление ≥ 2.
    • Contrarian‑fade ― MARKET против, когда выметен крупный кластер
      (≥1 M) или давление ≥ 4.
    • Iceberg‑limit 50‑400 k на кластере (order‑block).
    • Risk‑менеджмент: поза ≤ 50 % капитала, take +0.1 %, stop −0.05 %,
      авто‑флэт к 22:00 UTC.

Включается из server.py одной строкой:
    smart_money = SmartMoneyManager(agent_id="smart", total_capital=60_000_000, num_desks=4)
"""

# --------------------- ГЛОБАЛЬНЫЕ КОНСТАНТЫ -----------------------------
CHI_SIZE           = 120_000      # базовый child‑order (EUR)
VWAP_TARGET_PCT    = 0.25         # % капитала за день
VWAP_INTERVAL      = 90           # с (будет адаптировано по времени)

TREND_JOIN_PCT     = 0.05         # 5 % капитала
FADE_PCT           = 0.03         # 3 % капитала
MAX_POS_PCT        = 0.50         # max book 50 % cap
TAKE_PCT           = 0.001        # +0.10 % take‑profit
STOP_PCT           = 0.0005       # −0.05 % stop‑loss

PRESSURE_JOIN      = 2.0
PRESSURE_FADE      = 4.0

# Кластеры
CLUSTER_THRESH     = 1_000_000    # 1 M EUR на уровне
IMBALANCE_RATIO    = 2.0          # BID/ASK ≥ 2×
SWEEP_DROP         = 0.7          # кластер вымели >70 %
ICE_MIN            = 50_000
ICE_MAX            = 400_000
ICE_TTL            = 60           # c

# --------------------- DESK --------------------------------------------
class SmartMoneyDesk:
    def __init__(self, agent_id: str, capital: float):
        self.agent_id = agent_id
        self.capital  = capital
        # state
        self.pos = 0.0
        self.entry = None
        self.pnl_hist = deque(maxlen=500)
        self.cooldown = 0.0
        # VWAP state
        self.vwap_target = max(1_000_000, capital*VWAP_TARGET_PCT)
        self.vwap_done   = 0.0
        self.next_child  = time.time() + random.uniform(5, 20)
        # cluster state
        self.cluster_price = None
        self.prev_snapshot = None

    # ---------- helpers
    def _mkt(self, side, vol):
        vol = max(10_000, vol)  # safety
        return Order(uuid.uuid4().hex, self.agent_id, side, vol, None, OrderType.MARKET, None)

    @staticmethod
    def _vol_at(price, levels):
        for lvl in levels:
            if abs(lvl['price'] - price) < 1e-9:
                return lvl['volume']
        return 0.0

    # ---------- main
    def generate_orders(self, order_book, market_context=None):
        now = time.time()
        orders = []
        snap = order_book.get_order_book_snapshot(depth=10)
        if not snap['bids'] or not snap['asks']:
            return orders

        # depth‑factor (0.3..1) – адаптивный к top‑depth
        top_depth = snap['bids'][0]['volume'] + snap['asks'][0]['volume']
        depth_factor = max(0.3, min(1.0, top_depth / 1_500_000))

        # UTC‑время
        hour = time.gmtime(now).tm_hour

        # pressure
        phase = getattr(market_context, 'phase', None) if market_context else None
        pressure  = getattr(phase, 'pressure', 0.0) if phase else 0.0
        pdir      = getattr(phase, 'pressure_dir', 0) if phase else 0

        # ===== VWAP drip =====
        # адаптивный интервал по времени суток
        vwap_int = VWAP_INTERVAL
        if hour < 6:
            vwap_int *= 2.0
        elif hour >= 20:
            vwap_int *= 1.5

        if now >= self.next_child and self.vwap_done < self.vwap_target:
            side = OrderSide.BID if pdir >= 0 else OrderSide.ASK
            child = min(CHI_SIZE*depth_factor, self.vwap_target - self.vwap_done)
            orders.append(self._mkt(side, child))
            self.vwap_done += child
            self.next_child = now + vwap_int * random.uniform(0.9,1.1)

        # ===== Cluster scan & iceberg =====
        big_bid = next((lvl for lvl in snap['bids'] if lvl['volume'] >= CLUSTER_THRESH), None)
        big_ask = next((lvl for lvl in snap['asks'] if lvl['volume'] >= CLUSTER_THRESH), None)
        if big_bid and not big_ask:
            self.cluster_price = big_bid['price']
        elif big_ask and not big_bid:
            self.cluster_price = big_ask['price']

        if self.cluster_price and now >= self.cooldown:
            ice_side = OrderSide.BID if big_bid else OrderSide.ASK
            ice_vol  = max(ICE_MIN, min(ICE_MAX, ICE_MAX*depth_factor))
            orders.append(Order(uuid.uuid4().hex, self.agent_id, ice_side,
                                 ice_vol, self.cluster_price, OrderType.LIMIT, ICE_TTL))
            self.cooldown = now + 30

        # ===== Momentum join (only with imbalance) =====
        bid_vol = snap['bids'][0]['volume']; ask_vol = snap['asks'][0]['volume']
        if (pressure >= PRESSURE_JOIN and now >= self.cooldown and pdir!=0):
            imbalance_ok = (pdir==1 and bid_vol/ max(ask_vol,1) >= IMBALANCE_RATIO) or \
                           (pdir==-1 and ask_vol/ max(bid_vol,1) >= IMBALANCE_RATIO)
            if imbalance_ok:
                vol = min(self.capital * TREND_JOIN_PCT * depth_factor, 1_000_000)
                side = OrderSide.BID if pdir==1 else OrderSide.ASK
                if abs(self.pos + (vol if side==OrderSide.BID else -vol)) <= MAX_POS_PCT*self.capital:
                    orders.append(self._mkt(side, vol))
                    self.cooldown = now + random.uniform(60,120)

        # ===== Fade on sweep =====
        if self.cluster_price and self.prev_snapshot and now >= self.cooldown:
            prev_vol = self._vol_at(self.cluster_price, self.prev_snapshot['bids']+self.prev_snapshot['asks'])
            curr_vol = self._vol_at(self.cluster_price, snap['bids']+snap['asks'])
            if prev_vol >= CLUSTER_THRESH and curr_vol/prev_vol < (1-SWEEP_DROP):
                vol = min(self.capital * FADE_PCT * depth_factor, 1_000_000)
                fade_side = OrderSide.ASK if pdir==1 else OrderSide.BID
                if abs(self.pos + (vol if fade_side==OrderSide.BID else -vol)) <= MAX_POS_PCT*self.capital:
                    orders.append(self._mkt(fade_side, vol))
                    self.cooldown = now + random.uniform(120,180)
                self.cluster_price = None

        # ===== NY‑cut flat =====
        if hour >= 22 and abs(self.pos) > 0 and now >= self.cooldown:
            hedge_side = OrderSide.ASK if self.pos>0 else OrderSide.BID
            orders.append(self._mkt(hedge_side, abs(self.pos)))
            self.pos = 0; self.entry = None
            self.cooldown = now + 60

        self.prev_snapshot = snap
        return orders

    # ---------- fills ----------------------------------------------------
    def on_order_filled(self, oid, price, vol, side):
        if self.pos == 0:
            self.entry = price
        if self.pos * vol >= 0:
            self.entry = (self.entry*abs(self.pos) + price*vol)/(abs(self.pos)+vol)
        self.pos += vol if side==OrderSide.BID else -vol

        if not self.entry:
            return []
        pnl = (price - self.entry) * self.pos
        lim = MAX_POS_PCT * self.capital
        if pnl > TAKE_PCT*self.capital or pnl < -STOP_PCT*self.capital or abs(self.pos)>lim:
            hedge_side = OrderSide.ASK if self.pos>0 else OrderSide.BID
            hedge_vol  = abs(self.pos)
            self.pos = 0
            return [self._mkt(hedge_side, hedge_vol)]
        return []

# --------------------- MANAGER ------------------------------------------
class SmartMoneyManager:
    """Менеджер институционалов с общей сигнатурой (agent_id, total_capital).
    Создаёт num_desks SmartMoneyDesk и агрегирует их ордера/филлы.
    """
    def __init__(self, agent_id: str, total_capital: float, num_desks: int = 3):
        self.agent_id = agent_id
        num_desks = max(1, int(num_desks))
        # случайное распределение капитала, сумма = total_capital
        weights = [random.uniform(0.8, 1.2) for _ in range(num_desks)]
        w_sum = sum(weights)
        self.desks = []
        for i, w in enumerate(weights):
            cap = total_capital * (w / w_sum)
            desk_id = f"{agent_id}_desk_{i}"
            self.desks.append(SmartMoneyDesk(desk_id, cap))

    # ------------------------------------------------------------------
    def generate_orders(self, order_book, market_context=None):
        orders = []
        for desk in self.desks:
            orders.extend(desk.generate_orders(order_book, market_context))
        return orders

    # ------------------------------------------------------------------
    def on_order_filled(self, order_id, price, volume, side):
        hedge_orders = []
        for desk in self.desks:
            hedges = desk.on_order_filled(order_id, price, volume, side)
            if hedges:
                hedge_orders.extend(hedges)
        return hedge_orders

