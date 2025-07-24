import time
import uuid
import random
from collections import deque

import numpy as np
from order import Order, OrderSide, OrderType
from order import quant

"""AdvancedMarketMaker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
• Адаптивный спред = base + k·σ + inventory_bias
• Управление инвентарём (target_inventory → µP резерва)
• Пул заявок: Top of Book + Deep Quotes + Iceberg (TTL=1)
• Пауза перед макро (±2 min) + расширение спреда
• PnL‑/VAR‑стоп: при убытке > VAR_LIMIT — сворачиваем агрессию
"""
def _market(aid, side, vol):
    """Быстрый маркет-ордер — тот же формат, что используют другие агенты."""
    return Order(
        uuid.uuid4().hex,
        aid,
        side,
        vol,
        price=None,
        order_type=OrderType.MARKET,
        ttl=None,
    )

# ---------------------------------------------------------
#  Константы
# ---------------------------------------------------------
TOP_LEVELS        = 1           # котируем лучшую цену
DEEP_LEVELS       = 2           # 2 более глубоких
ICEBERG_LEVELS    = 1           # TTL=1, объём маленький
VAR_LIMIT         = -250_000     # дневной лимит $ per MM
MACRO_EVENTS      = {(13,30), (18,0)}
MACRO_PAUSE       = 120         # sec
ROLLING_PNL_SIZE  = 200
MAX_INV_RATIO     = 0.05        # 7% капитала

# ---------------------------------------------------------
class VolEstimator:
    def __init__(self):
        self.window = deque(maxlen=40)
        self.sigma  = 0.0
    def update(self, mid):
        self.window.append(mid)
        if len(self.window) > 2:
            self.sigma = float(np.std(np.diff(self.window)))

# ---------------------------------------------------------
class AdvancedMarketMaker:
    def __init__(self, agent_id: str, capital: float):
        self.agent_id   = agent_id
        self.capital    = capital
        self.cash       = capital
        self.inventory  = 0.0
        self.entry_wavg= None
        self.vol_est    = VolEstimator()
        self.pnl_hist   = deque(maxlen=ROLLING_PNL_SIZE)
        self.active_ord = {}
        self.ttl        = 5

    # macro‑hold flag
    def _macro_pause(self):
        now = time.time()
        utc = time.gmtime(now)
        key = (utc.tm_hour, utc.tm_min)
        for h,m in MACRO_EVENTS:
            target = time.mktime(time.struct_time((utc.tm_year,utc.tm_mon,utc.tm_mday,h,m,0,0,0,0)))
            if abs(target-now) <= MACRO_PAUSE:
                return True
        return False

    # -------------------------------------------------
    def _compute_spread(self):
        base = 0.0007
        sigma = self.vol_est.sigma
        inv_bias = abs(self.inventory) / (self.capital / 2) * 0.0015
        return base + sigma*6 + inv_bias

    # -------------------------------------------------
    def cancel_expired(self):
        now=time.time()
        drop=[oid for oid,info in self.active_ord.items() if now-info['ts']>self.ttl]
        for oid in drop:self.active_ord.pop(oid,None)

    # -------------------------------------------------
    def generate_orders(self, order_book, market_context=None):
        if self._macro_pause():
            return []

        self.cancel_expired()
        snap = order_book.get_order_book_snapshot(depth=1)
        if not snap['bids'] or not snap['asks']:
            return []
        best_bid = snap['bids'][0]['price']
        best_ask = snap['asks'][0]['price']
        mid      = (best_bid+best_ask)/2.0
        self.vol_est.update(mid)

        spread = self._compute_spread()
        top_vol = (self.capital*0.01)/mid
        deep_vol= top_vol*1.5
        ice_vol = top_vol*0.3

        # inventory guard
        inv_ratio = self.inventory*mid / self.capital
        if abs(inv_ratio) > MAX_INV_RATIO:
            # резко расширяем противоположную сторону
            spread *= 1.8

        price_levels = []
        # top level
        price_levels.append((OrderSide.BID, mid-spread/2))
        price_levels.append((OrderSide.ASK, mid+spread/2))
        # deep levels
        for i in range(1,DEEP_LEVELS+1):
            price_levels.append((OrderSide.BID, mid-spread/2 - i*0.015))
            price_levels.append((OrderSide.ASK, mid+spread/2 + i*0.015))
        # iceberg TTL=1
        for i in range(ICEBERG_LEVELS):
            price_levels.append((OrderSide.BID, mid-spread/2 - 0.002*i))
            price_levels.append((OrderSide.ASK, mid+spread/2 + 0.002*i))

        orders=[]; now=time.time()
        for side, px in price_levels:
            vol = ice_vol if side==OrderSide.BID and px>mid or side==OrderSide.ASK and px<mid else (top_vol if abs(px-mid)<spread else deep_vol)
            oid = uuid.uuid4().hex
            ttl = 1 if vol==ice_vol else self.ttl
            self.active_ord[oid]={'side':side,'price':px,'vol':vol,'ts':now}
            order = Order(oid, "mm1", side, vol, quant(px), OrderType.LIMIT, ttl, metadata={"highlight": self.agent_id})

            orders.append(order)

        return orders

    # -------------------------------------------------
    def on_order_filled(self, oid, price, qty, side):
        if oid not in self.active_ord:
            return
        info=self.active_ord.pop(oid)
        # cash / inventory update
        if side==OrderSide.BID:
            self.cash -= price*qty
            self.inventory += qty
        else:
            self.cash += price*qty
            self.inventory -= qty
        # pnl snapshot (mark‑to‑market)
        mtm = self.inventory*price + self.cash - self.capital
        self.pnl_hist.append(mtm)
        if mtm < VAR_LIMIT:
            # cut inventory aggressively
            hedge_side = OrderSide.ASK if self.inventory>0 else OrderSide.BID
            hedge_qty  = abs(self.inventory)
            if hedge_qty>0:
                return [_market(self.agent_id, hedge_side, hedge_qty)]
        return []

# ---------------------------------------------------------------------------
class MarketMakerManager:
    def __init__(self, agent_id:str,total_capital:float):
        self.agent_id=agent_id
        per_cap = total_capital/22
        self.mms = [AdvancedMarketMaker(f"advmm_{i}", per_cap) for i in range(22)]
        self.tick=0
    def generate_orders(self, order_book, market_context=None):
        orders=[]
        for mm in self.mms:
            if random.random()<0.2:continue  # stagger activity
            orders.extend(mm.generate_orders(order_book, market_context))
        self.tick+=1
        return orders
    def on_order_filled(self, oid, price, qty, side):
        extra=[]
        for mm in self.mms:
            res=mm.on_order_filled(oid, price, qty, side)
            if res:extra.extend(res)
        return extra

