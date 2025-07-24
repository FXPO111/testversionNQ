# файл: spread_arbitrage.py

import uuid
import time
import numpy as np

from order import Order, OrderSide, OrderType

class SpreadArbitrageAgent:
    """
    Если spread > 0.3 % (или другой порог), агент одновременно
    входит Market-буквой с обеих сторон, фиксируя небольшую прибыль и
    сужая спред.
    """

    def __init__(self, agent_id: str, capital: float, restore_threshold: float = 0.3):
        self.agent_id = agent_id
        self.capital = float(capital)
        self.last_action_time = 0.0
        self.cooldown = 2.0  # минимум 2 с между заходами
        self.trade_history = []
        self.restore_threshold = restore_threshold

    def generate_orders(self, order_book, market_context=None) -> list:

        self._market_context = market_context
        perception = self.perceive_market(market_context) if hasattr(self, "_market_context") else "neutral"

        # В фазе "паники" — избегать входа
        if perception == "avoid" and np.random.rand() < 0.8:
            return []

        # В фазе "волатильности" — быть осторожным
        if perception == "hesitate" and np.random.rand() < 0.5:
            return []

        # В фазе "арбитража" — активнее искать спред
        if perception == "arbitrage":
            threshold = 0.004  # увеличим порог для арбитража
       
        if perception == "supportive":
            threshold = 0.002

        now = time.time()
        if now - self.last_action_time < self.cooldown:
            return []

        snapshot = order_book.get_order_book_snapshot(depth=5)
        bids = snapshot.get("bids", [])
        asks = snapshot.get("asks", [])
        if not bids or not asks:
            return []

        best_bid = bids[0]["price"]
        best_ask = asks[0]["price"]
        spread_pct = (best_ask - best_bid) / (best_bid + 1e-8)

        threshold = 0.003  # 0.3%
        if spread_pct < threshold:
            return []

        mid = (best_bid + best_ask) / 2.0
        vol = max(1.0, round((self.capital * 0.001) / mid))

        buy_order = Order(
            order_id=str(uuid.uuid4()),
            agent_id=self.agent_id,
            side=OrderSide.BID,
            volume=vol,
            price=None,
            order_type=OrderType.MARKET,
            ttl=None
        )
        sell_order = Order(
            order_id=str(uuid.uuid4()),
            agent_id=self.agent_id,
            side=OrderSide.ASK,
            volume=vol,
            price=None,
            order_type=OrderType.MARKET,
            ttl=None
        )

        self.last_action_time = now
        return [buy_order, sell_order]

    def restore_capital(self):
        """
        Восстанавливаем капитал, если он упал ниже порога.
        Пополняем капитал до 50% от начального значения.
        """
        if self.capital < self.initial_capital * self.restore_threshold:
            # Восстановление части капитала
            restore_amount = self.initial_capital * 0.7  # Восстанавливаем 50% от начального капитала
            self.capital = min(self.capital + restore_amount, self.initial_capital)  # Не превышаем начальный капитал
            logger.info(f"Capital restored. New capital: {self.capital}")


    def on_order_filled(self, order_id: str, price: float, volume: float, side: OrderSide):
        self.trade_history.append({
            'order_id': order_id,
            'price': price,
            'volume': volume,
            'side': side.value,
            'timestamp': time.time()
        })

    def perceive_market(self, market_context) -> str:
        if not market_context or not market_context.is_ready():
            return "neutral"

        perceived_phase = market_context.phase.name
        if np.random.rand() < 0.15:
            perceived_phase = random.choice(["panic", "volatile", "trend_up", "trend_down", "calm"])

        if perceived_phase == "panic":
            return "avoid"
        elif perceived_phase == "volatile":
            return "hesitate"
        elif perceived_phase in ["trend_up", "trend_down"]:
            return "arbitrage"
        elif perceived_phase == "calm":
            return "supportive"
        return "neutral"

