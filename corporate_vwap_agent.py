import uuid, random, time
from order import Order, OrderSide, OrderType

# ───────────── helpers ─────────────
def _m(aid, side, vol):
    return Order(uuid.uuid4().hex, aid, side, vol,
                 price=None, otype=OrderType.MARKET, ttl=None)

def _l(aid, side, vol, px, ttl=6):
    return Order(uuid.uuid4().hex, aid, side, vol,
                 price=px, otype=OrderType.LIMIT, ttl=ttl)

# ───────────── один «лег» (деск) ─────────────
class _VWAPLeg:
    U_SHAPE = (0.06,0.07,0.08,0.09,0.10,0.11,0.12,
               0.11,0.10,0.09,0.08,0.07,0.06)           # 30-40-30 %

    def __init__(self, uid, notional, side, horizon,
                 start_delay, max_clip, part_cap):
        self.uid   = uid
        self.rest  = float(notional)
        self.side  = OrderSide.BID if side == "buy" else OrderSide.ASK
        now        = time.time()
        self.t0    = now + start_delay              # сдвиг, чтобы не синхронно
        self.t_end = self.t0 + horizon
        self.bins  = 20
        self.max_clip = max_clip
        self.part_cap = part_cap
        self.slip_ema = 0.0

    # — вызывается менеджером каждый тик —
    def step(self, ctx):
        # ───── защита: если market_context пустой ─────
        if ctx is None or not getattr(ctx, "mid_prices_short", None):
            return []          # пропускаем тик, ждём нормальный контекст

        now = time.time()
        if now < self.t0 or self.rest <= 0:
            return []

        mid     = ctx.mid_prices_short[-1]
        spread  = getattr(ctx, "current_spread", 0.00008)
        mvol    = getattr(ctx, "executed_volume_last_sec", 60_000)

        # bucket index по U-кривой
        idx = int((now - self.t0) / ((self.t_end - self.t0) / self.bins))
        idx = max(0, min(idx, self.bins - 1))

        want = self.U_SHAPE[idx] * (self.rest + 1e-9)
        want = min(want, self.part_cap*mvol, self.max_clip, self.rest)
        want *= random.uniform(0.9, 1.1)
        if want < 5_000:
            return []

        orders = []
        if spread < 0.00008 and self.slip_ema < 0.00010:
            # MARKET iceberg ≤ 20k
            while want > 0:
                clip = min(20_000, want)
                orders.append(_m(self.uid, self.side, clip))
                self.rest -= clip
                want -= clip
        elif spread < 0.00014 and self.slip_ema < 0.00012:
            px = mid - 0.000003 if self.side == OrderSide.BID else mid + 0.000003
            orders.append(_l(self.uid, self.side, want, px))
            self.rest -= want
        # иначе пауза

        return orders

    # — вызывай из движка после исполнения, если есть callback —
    def on_fill(self, fill_px, mid_px):
        self.slip_ema = 0.9*self.slip_ema + 0.1*abs(fill_px - mid_px)

# ───────────── менеджер под твою архитектуру ─────────────
class CorporateVWAPAgent:
    """
    CorporateVWAPAgent(agent_id, total_capital)

    • создаёт N асинхронных legs (по умолчанию 3) с разными delay-стартами,
      противоположными сторонами и разными horizon.
    • интерфейс generate_orders / on_order_filled идентичен другим агентам.
    """

    def __init__(self, agent_id: str, total_capital: float, legs: int = 3):
        self.agent_id = agent_id
        self.total_capital = total_capital
        self.legs = self._spawn_legs(legs)

    def _spawn_legs(self, n):
        legs = []
        leg_cap = self.total_capital / n
        sides = ["buy", "sell"] * (n//2 + 1)
        random.shuffle(sides)
        for i in range(n):
            uid = f"{self.agent_id}_leg{i}"
            side = sides[i]
            horizon = random.randint(60*40, 60*120)     # 40-120 мин
            delay   = random.randint(0, 60*30)          # 0-30 мин сдвиг
            clip    = random.randint(60_000, 100_000)
            part    = random.uniform(0.08, 0.15)
            legs.append(
                _VWAPLeg(uid, leg_cap, side, horizon,
                         delay, clip, part)
            )
        return legs

    def generate_orders(self, order_book, market_context=None):
        orders = []
        for leg in self.legs:
            orders.extend(leg.step(market_context))
        return orders

    def on_order_filled(self, agent_uid, fill_price, mid_price):
        # прокидываем событие в нужный лег
        for leg in self.legs:
            if agent_uid.startswith(leg.uid):
                leg.on_fill(fill_price, mid_price)
                break
