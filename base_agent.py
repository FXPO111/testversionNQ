# Файл: base_agent.py (финальный гибридный агент уровня EUR/USD)

import uuid
import random
import numpy as np
from collections import deque
from order import Order, OrderSide, OrderType


class SubAgent:
    def __init__(self, agent_id, capital):
        self.agent_id = agent_id
        self.capital = capital
        self.position = 0.0
        self.entry_price = None
        self.cooldown = 0
        self.risk_fraction = 0.02
        self.volatility_sensitivity = 1.0
        self.bias_memory = []

    def step(self, order_book, market_context):
        if self.cooldown > 0:
            self.cooldown -= 1
            return []

        # --- Реалистичная оценка рыночного состояния ---
        # Волатильность (стандартное отклонение доходностей)
        if len(market_context.mid_prices_short) > 3:
            returns = np.diff(market_context.mid_prices_short)
            volatility = float(np.std(returns))
        else:
            volatility = 0.0

        # Давление ликвидности (кластерный дисбаланс)
        pressure = getattr(market_context, 'cluster_disbalance', 0.0)

        # Orderflow skew: микро-тренд + институциональный биас
        skew = 0.0
        if hasattr(market_context, 'micro_trends') and market_context.micro_trends:
            skew += np.mean(market_context.micro_trends[-5:])
        skew += getattr(market_context, 'institutional_bias', 0.0)

        # --- Стратегия ---
        direction = 1 if skew > 0 else -1
        side = OrderSide.BID if direction > 0 else OrderSide.ASK

        best_bid = order_book._best_bid_price()
        best_ask = order_book._best_ask_price()
        if best_bid is None or best_ask is None:
            return []
        mid = (best_bid + best_ask) / 2

        volume = min(self.capital * self.risk_fraction, 100_000)

        order = Order(
            order_id=str(uuid.uuid4()),
            agent_id=self.agent_id,
            side=side,
            price=best_ask if side == OrderSide.BID else best_bid,
            volume=volume,
            order_type=OrderType.MARKET,
        )

        self.cooldown = random.randint(5, 15)
        return [order]


class BaseAgent:
    def __init__(self, agent_id: str, total_capital: float, participants_per_agent: int = 10000):
        self.agent_id = agent_id
        self.total_capital = total_capital
        self.participants_per_agent = participants_per_agent

        self.num_subagents = participants_per_agent
        self.subagents = []
        mean_capital = total_capital / participants_per_agent

        for i in range(self.num_subagents):
            cap_i = np.random.lognormal(mean=np.log(mean_capital), sigma=0.7)
            cap_i = min(cap_i, total_capital * 0.05)
            
    def generate_orders(self, order_book, market_context=None):
        # Поддержка текущей архитектуры: fallback если нет метода get_mid_price
        if hasattr(order_book, "get_mid_price"):
            price = order_book.get_mid_price()
        elif hasattr(order_book, "last_trade_price") and order_book.last_trade_price:
            price = order_book.last_trade_price
        elif hasattr(order_book, "best_bid_price") and hasattr(order_book, "best_ask_price"):
            bid = order_book.best_bid_price()
            ask = order_book.best_ask_price()
            if bid is not None and ask is not None:
                price = (bid + ask) / 2.0
            else:
                return []
        else:
            return []

        orders = []
        for sub in self.subagents:
            sub_orders = sub.decide(market_context, price)
            if sub_orders:
                orders.extend(sub_orders)
        return orders
