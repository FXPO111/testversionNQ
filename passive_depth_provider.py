import uuid
import random
import time
import numpy as np

from order import Order, OrderSide, OrderType

class PassiveDepthProvider:
    def __init__(self, agent_id, capital, min_depth_percent=0.01, max_depth_percent=0.01, order_lifetime=10.0):
        self.agent_id = agent_id
        self.capital = capital
        self.min_depth_percent = min_depth_percent
        self.max_depth_percent = max_depth_percent
        self.order_lifetime = order_lifetime
        self.position = 0.0

        self.current_orders = []
        self.last_mid = None
        self.levels = 100  # количество уровней вверх и вниз

    def cancel_expired_orders(self):
        now = time.time()
        self.current_orders = [order for order in self.current_orders if now - order.timestamp < self.order_lifetime]

    def generate_orders(self, order_book, market_context=None):
        bid = order_book._best_bid_price()
        ask = order_book._best_ask_price()
        if bid is None or ask is None:
            return []

        mid = round((bid + ask) / 2, 5)

        self.cancel_expired_orders()

        price_tolerance = self.levels * 0.01 * 1.5
        self.current_orders = [
            order for order in self.current_orders
            if abs(order.price - mid) <= price_tolerance
        ]

        self.current_orders = []  # сбрасываем, если нужно перегенерация

        now = time.time()

        # ==== ЗОНА 1: удержание спреда ====
        spread = random.uniform(0.01, 0.02)
        bid_price = round(mid - spread / 2, 2)
        ask_price = round(mid + spread / 2, 2)

        strong_volume = max(10_000, min(self.capital * 0.00005, 25_000))

        bid_order = Order(
            order_id=str(uuid.uuid4()),
            agent_id=self.agent_id,
            side=OrderSide.BID,
            price=bid_price,
            volume=strong_volume,
            order_type=OrderType.LIMIT
        )
        bid_order.timestamp = now

        ask_order = Order(
            order_id=str(uuid.uuid4()),
            agent_id=self.agent_id,
            side=OrderSide.ASK,
            price=ask_price,
            volume=strong_volume,
            order_type=OrderType.LIMIT
        )
        ask_order.timestamp = now

        self.current_orders += [bid_order, ask_order]

        # ==== ЗОНА 2: глубина ====
        max_total_volume_per_side = 350_000
        cum_volume_bid = 0
        cum_volume_ask = 0

        for i in range(3, self.levels + 1):
            offset = round(i * 0.01, 5)

            # дырки в глубине
            if i > 50 and random.random() < 0.1:
                continue

            base = 1000 * (1.0 - (i / self.levels) ** 2)
            base = max(100, base)

            decay_bias = max(0.1, random.uniform(1.0 - i / 120, 1.0))

            if random.random() < 0.08:
                volume = random.uniform(0.01, 3.0)
            elif random.random() < 0.03:
                volume = random.uniform(5000, 10000)
            else:
                volume = base * decay_bias * random.uniform(0.4, 1.4)

            volume = max(100, min(volume, 16000))

            if i > 60 and random.random() < 0.01:
                volume *= random.uniform(1.5, 3.0)
                volume = min(volume, 20000)

            bid_price = round(mid - offset, 2)
            ask_price = round(mid + offset, 2)

            if cum_volume_bid + volume <= max_total_volume_per_side:
                bid_order = Order(
                    order_id=str(uuid.uuid4()),
                    agent_id=self.agent_id,
                    side=OrderSide.BID,
                    price=bid_price,
                    volume=volume,
                    order_type=OrderType.LIMIT
                )
                bid_order.timestamp = now
                self.current_orders.append(bid_order)
                cum_volume_bid += volume

            if cum_volume_ask + volume <= max_total_volume_per_side:
                ask_order = Order(
                    order_id=str(uuid.uuid4()),
                    agent_id=self.agent_id,
                    side=OrderSide.ASK,
                    price=ask_price,
                    volume=volume,
                    order_type=OrderType.LIMIT
                )
                ask_order.timestamp = now
                self.current_orders.append(ask_order)
                cum_volume_ask += volume

        return self.current_orders

    def on_order_filled(self, order_id, fill_price, fill_size, side):
        self.position += fill_size if side == OrderSide.BID else -fill_size

