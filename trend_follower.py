import numpy as np
import uuid
import random
from collections import deque
from order import Order, OrderSide, OrderType

class TrendSubAgent:
    def __init__(self, agent_id, capital, base_agent):
        self.agent_id = agent_id
        self.capital = capital
        self.base_agent = base_agent
        self.confidence = np.random.uniform(0.4, 1.0)
        self.fomo = np.random.uniform(0.0, 0.3)
        self.fatigue = np.random.uniform(0.0, 0.2)
        self.bias = np.random.choice(["long", "short", "neutral"], p=[0.4, 0.4, 0.2])

    def decide(self, market_context, order_book):
        if random.random() < self.fatigue:
            return None

        indicators = self.base_agent.calculate_indicators()
        perception = self.base_agent.perceive_market(market_context, indicators)
        action = self.base_agent.decide_action(indicators, perception, self.bias, self.fomo, self.confidence)

        if action == "hold":
            return None

        last_price = order_book.last_trade_price
        volatility = indicators.get("atr", 0.01)
        size = self.base_agent.calculate_position_size(volatility, last_price, self.confidence)

        side = OrderSide.BID if action == "buy" else OrderSide.ASK
        use_market = random.random() < (0.4 + 0.3 * self.confidence)

        if use_market:
            return Order(
                order_id=str(uuid.uuid4()),
                agent_id=self.agent_id,
                side=side,
                volume=round(size, 4),
                price=None,
                order_type=OrderType.MARKET,
                ttl=None
            )
        else:
            offset = 0.003 * last_price * np.random.uniform(0.8, 1.2)
            limit_price = last_price - offset if side == OrderSide.BID else last_price + offset
            return Order(
                order_id=str(uuid.uuid4()),
                agent_id=self.agent_id,
                side=side,
                volume=round(size, 4),
                price=round(limit_price, 2),
                order_type=OrderType.LIMIT,
                ttl=random.randint(5, 20)
            )

class TrendFollowerAgent:
    def __init__(self, agent_id, capital, num_subagents=100):
        self.agent_id = agent_id
        self.capital = capital
        self.price_history = deque(maxlen=500)
        self.subagents = [
            TrendSubAgent(f"{agent_id}_sub_{i}", capital / num_subagents, self)
            for i in range(num_subagents)
        ]
        self.ema_fast_period = 12
        self.ema_slow_period = 26
        self.atr_period = 14
        self.adx_period = 14

    def _ema(self, data, period):
        if len(data) < period:
            return [data[-1]] * len(data)
        alpha = 2 / (period + 1)
        ema = [data[0]]
        for p in data[1:]:
            ema.append(alpha * p + (1 - alpha) * ema[-1])
        return ema

    def _atr(self, prices):
        if len(prices) < self.atr_period + 1:
            return 0.01
        high = np.array(prices) + 0.2
        low = np.array(prices) - 0.2
        close = np.array(prices)
        tr = np.maximum(high[1:], close[:-1]) - np.minimum(low[1:], close[:-1])
        return float(np.mean(tr[-self.atr_period:]))

    def _adx(self, prices):
        if len(prices) < self.adx_period + 2:
            return 10.0
        diff = np.diff(prices)
        dm_plus = np.maximum(diff, 0)
        dm_minus = -np.minimum(diff, 0)
        atr = np.mean(np.abs(diff[-self.adx_period:])) + 1e-9
        di_plus = 100 * np.mean(dm_plus[-self.adx_period:]) / atr
        di_minus = 100 * np.mean(dm_minus[-self.adx_period:]) / atr
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus + 1e-9)
        return dx

    def calculate_indicators(self):
        prices = list(self.price_history)
        price = prices[-1]
        ema_fast = self._ema(prices, self.ema_fast_period)[-1]
        ema_slow = self._ema(prices, self.ema_slow_period)[-1]
        atr = self._atr(prices)
        adx = self._adx(prices)
        slope = (ema_fast - ema_slow) / price if price > 0 else 0
        return {
            "price": price,
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "atr": atr,
            "adx": adx,
            "slope": slope,
        }

    def perceive_market(self, market_context, indicators):
        adx = indicators["adx"]
        slope = indicators["slope"]
        if market_context and hasattr(market_context, "phase"):
            phase = market_context.phase.name
            if phase in ["trend_up", "trend_down"]:
                return "strong_trend"
            elif phase in ["volatile", "noise"]:
                return "hesitation"
            elif phase == "calm":
                return "neutral"
        if adx > 25 and abs(slope) > 0.001:
            return "strong_trend"
        elif adx > 15:
            return "weak_trend"
        else:
            return "no_trend"

    def decide_action(self, indicators, perception, bias, fomo, confidence):
        if perception == "no_trend":
            return "hold"
        if indicators["ema_fast"] > indicators["ema_slow"]:
            raw_signal = "buy"
        elif indicators["ema_fast"] < indicators["ema_slow"]:
            raw_signal = "sell"
        else:
            raw_signal = "hold"

        # модификаторы поведения
        if bias == "long" and raw_signal == "sell" and random.random() < 0.5:
            return "hold"
        if bias == "short" and raw_signal == "buy" and random.random() < 0.5:
            return "hold"
        if fomo > 0.25 and perception == "weak_trend":
            return raw_signal
        if confidence < 0.5 and random.random() > confidence:
            return "hold"
        return raw_signal

    def calculate_position_size(self, volatility, price, confidence):
        if price <= 0:
            return 1.0
        base_risk = self.capital * 0.01 * confidence
        size = base_risk / (volatility + 1e-9)
        size = min(size, (self.capital / price) * 0.1)
        return max(size, (self.capital * 0.001) / price)

    def generate_orders(self, order_book, market_context=None):
        last_price = order_book.last_trade_price
        if last_price is None:
            return []

        self.price_history.append(last_price)
        orders = []
        for sub in self.subagents:
            order = sub.decide(market_context, order_book)
            if order:
                orders.append(order)
        return orders