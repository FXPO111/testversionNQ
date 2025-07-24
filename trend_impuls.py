import uuid
import random
import numpy as np
from datetime import datetime
from order import Order, OrderSide, OrderType


class ImpulseTrendCoreUnit:
    def __init__(self, uid, volume=1_000_000):
        self.uid = uid
        self.volume = volume
        self.state = "idle"
        self.trend_window = []
        self.trend_ticks = 5
        self.cooldown = 0
        self.last_side = None
        self.hold_ticks = 0

    def step(self, order_book, ctx):
        orders = []

        if self.cooldown > 0:
            self.cooldown -= 1
            return orders

        try:
            best_bid = order_book.best_bid()
            best_ask = order_book.best_ask()
        except:
            return orders

        mid = (best_bid + best_ask) / 2
        self.trend_window.append(mid)
        if len(self.trend_window) > self.trend_ticks:
            self.trend_window.pop(0)

        if len(self.trend_window) == self.trend_ticks:
            delta = self.trend_window[-1] - self.trend_window[0]
            avg_volatility = np.std(self.trend_window)

            if abs(delta) > avg_volatility * 1.5 and abs(delta) > 0.0004:
                side = OrderSide.BID if delta > 0 else OrderSide.ASK

                # блокировка повторных входов в одну сторону
                if self.last_side != side or self.hold_ticks > 10:
                    # разбивка объема на несколько маркетов
                    n_chunks = random.randint(2, 4)
                    chunk_size = self.volume // n_chunks
                    for _ in range(n_chunks):
                        orders.append(Order(
                            order_id=str(uuid.uuid4()),
                            agent_id=self.uid,
                            side=side,
                            order_type=OrderType.MARKET,
                            volume=chunk_size
                        ))

                    self.cooldown = random.randint(5, 10)
                    self.last_side = side
                    self.hold_ticks = 0
                else:
                    self.hold_ticks += 1
        return orders


class StrategicInitiatorAgent:
    def __init__(self, uid, volume=2_000_000):
        self.uid = uid
        self.volume = volume
        self.state = "idle"
        self.target_price = None
        self.entry_price = None
        self.current_position = 0
        self.lim_order_id = None
        self.cooldown = 0
        self.support_price = None
        self.ticks_since_entry = 0
        self.exit_started = False

    def step(self, order_book, ctx):
        orders = []

        if self.cooldown > 0:
            self.cooldown -= 1
            return orders

        try:
            best_bid = order_book.best_bid()
            best_ask = order_book.best_ask()
        except:
            return orders

        mid = (best_bid + best_ask) / 2

        # --- СТАДИЯ 1: постановка цели ---
        if self.state == "idle":
            # выбираем сторону движения
            bias = random.choice(["up", "down"])
            offset = random.uniform(0.0020, 0.0040)
            self.target_price = mid + offset if bias == "up" else mid - offset
            self.state = "prepare"
            self.bias = bias
            self.cooldown = 5
            return orders

        # --- СТАДИЯ 2: подготовка входа ---
        elif self.state == "prepare":
            # проверка: достаточно ли "мелкой" ликвидности перед входом
            book_depth = order_book.depth_bids() if self.bias == "up" else order_book.depth_asks()
            top_levels = book_depth[:5]
            thin = all(level[1] < self.volume * 0.2 for level in top_levels)

            if thin:
                self.state = "impulse_entry"
                return orders

        # --- СТАДИЯ 3: вход и установка лимитки поддержки ---
        elif self.state == "impulse_entry":
            entry_side = OrderSide.BID if self.bias == "up" else OrderSide.ASK
            lim_side = entry_side

            self.entry_price = mid
            self.support_price = mid - 0.0005 if self.bias == "up" else mid + 0.0005

            # лимитка поддержки
            lim_order = Order(
                order_id=str(uuid.uuid4()),
                agent_id=self.uid,
                side=lim_side,
                order_type=OrderType.LIMIT,
                volume=self.volume * 0.5,
                price=self.support_price
            )
            self.lim_order_id = lim_order.order_id
            orders.append(lim_order)

            # начальный маркет вход
            orders.append(Order(
                order_id=str(uuid.uuid4()),
                agent_id=self.uid,
                side=entry_side,
                order_type=OrderType.MARKET,
                volume=self.volume * 0.5
            ))

            self.current_position += self.volume * 0.5
            self.state = "push_and_hold"
            self.cooldown = 3
            self.ticks_since_entry = 0
            return orders

        # --- СТАДИЯ 4: контроль движения и докупка на импульсе ---
        elif self.state == "push_and_hold":
            self.ticks_since_entry += 1

            # докупка маркетом, если импульс подтверждается
            price = mid
            distance = abs(price - self.entry_price)

            if self.bias == "up" and price > self.entry_price + 0.0007 and self.current_position < self.volume:
                orders.append(Order(
                    str(uuid.uuid4()), self.uid, OrderSide.BID, OrderType.MARKET, self.volume * 0.25
                ))
                self.current_position += self.volume * 0.25
                self.entry_price = price
                self.cooldown = 2
                return orders

            if self.bias == "down" and price < self.entry_price - 0.0007 and self.current_position < self.volume:
                orders.append(Order(
                    str(uuid.uuid4()), self.uid, OrderSide.ASK, OrderType.MARKET, self.volume * 0.25
                ))
                self.current_position += self.volume * 0.25
                self.entry_price = price
                self.cooldown = 2
                return orders

            # цель достигнута?
            if (self.bias == "up" and price >= self.target_price) or (self.bias == "down" and price <= self.target_price):
                self.state = "exit"
                return orders

        # --- СТАДИЯ 5: разгрузка с помощью лимитки и подталкивания ---
        elif self.state == "exit":
            exit_side = OrderSide.ASK if self.bias == "up" else OrderSide.BID
            exit_price = mid + 0.0003 if self.bias == "up" else mid - 0.0003

            # подставляем лимитку на выход
            orders.append(Order(
                str(uuid.uuid4()), self.uid, exit_side, OrderType.LIMIT, self.current_position, price=exit_price
            ))

            # подталкиваем толпу маркетом
            orders.append(Order(
                str(uuid.uuid4()), self.uid, exit_side, OrderType.MARKET, self.volume * 0.1
            ))

            self.state = "reset"
            self.cooldown = 10
            return orders

        # --- СТАДИЯ 6: сброс и перезапуск ---
        elif self.state == "reset":
            self.target_price = None
            self.entry_price = None
            self.current_position = 0
            self.support_price = None
            self.lim_order_id = None
            self.state = "idle"
            return orders

        return orders


class TrendAgentManager:
    def __init__(self, agent_id, volume=1_000_000):
        self.agent_id = agent_id
        self.impulse_agent = ImpulseTrendCoreUnit(agent_id + "_impulse", volume)
        self.range_agent = StrategicInitiatorAgent(agent_id + "_range", volume)

    def generate_orders(self, order_book, market_context=None):
        orders = []
        orders.extend(self.impulse_agent.step(order_book, market_context))
        orders.extend(self.range_agent.step(order_book, market_context))
        return orders

    def on_order_filled(self, order_id, fill_price, fill_size, side):
        pass  # можно реализовать управление позицией, если будет нужен state tracking

