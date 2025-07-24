import numpy as np
import random
import uuid
import logging

from collections import deque
from order import Order, OrderSide, OrderType

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

class LiquidityManipulator:
    def __init__(self, agent_id: str, capital: float, volatility_threshold: float = 0.01, restore_threshold: float = 0.5):
        self.agent_id = agent_id
        self.capital = capital
        self.initial_capital = capital
        self.restore_threshold = restore_threshold

        self.position = 0.0
        self.max_position = capital * 0.2

        self.price_history = deque(maxlen=500)
        self.close_history = deque(maxlen=500)
        self.volume_history = deque(maxlen=500)
        self.long_term_trend_history = deque(maxlen=1000)  # История долгосрочных трендов

        self.cooldown_ticks = 0
        self.last_action_price = None
        self.manipulation_probability = 0.6
        self.mood = 1.0  # Психологический настрой

    def update_history(self, last_trade: dict):
        if not last_trade:
            return
        price = float(last_trade['price'])
        volume = float(last_trade['volume'])
        self.price_history.append(price)
        self.close_history.append(price)
        self.volume_history.append(volume)

        # Долгосрочный тренд
        self.long_term_trend_history.append(price)

    def calculate_volatility(self):
        if len(self.close_history) < 50:
            return 0.0
        returns = np.diff(self.close_history)[-50:]
        return float(np.std(returns))

    def calculate_trend_strength(self):
        if len(self.close_history) < 50:
            return 0.0
        returns = np.diff(self.close_history)[-50:]
        return float(np.abs(np.mean(returns)) / (np.std(returns) + 1e-6))

    def is_volume_spike(self):
        if len(self.volume_history) < 50:
            return False
        recent = np.array(self.volume_history)[-10:]
        avg = np.mean(self.volume_history[-50:-10])
        return np.any(recent > avg * 2.5)  # объём выше среднего более чем в 2.5 раза

    def is_consolidation(self):
        if len(self.close_history) < 50:
            return False
        price_range = max(self.close_history[-50:]) - min(self.close_history[-50:])
        avg_price = np.mean(self.close_history[-50:])
        norm_range = price_range / avg_price
        volatility = self.calculate_volatility()
        trend_strength = self.calculate_trend_strength()
        return norm_range < 0.012 and volatility < self.volatility_threshold and trend_strength < 0.5

    def detect_channel_bounds(self):
        recent = list(self.close_history)[-50:]
        return min(recent), max(recent)

    def analyze_orderbook_imbalance(self, order_book, depth=5):
        snapshot = order_book.get_order_book_snapshot(depth=depth)
        bids = snapshot.get("bids", [])
        asks = snapshot.get("asks", [])
        bid_volume = sum(b['volume'] for b in bids)
        ask_volume = sum(a['volume'] for a in asks)
        total = bid_volume + ask_volume + 1e-8
        return (bid_volume - ask_volume) / total

    def calculate_order_size(self, price, volatility):
        risk_budget = self.capital * 0.01
        stop_dist = max(volatility * 2, price * 0.001)
        size = risk_budget / stop_dist
        size = min(size, self.max_position - abs(self.position))
        return round(max(1.0, size), 2)

    def long_term_trend_analysis(self):
        """Анализ долгосрочного тренда на основе более широкой картины рынка"""
        if len(self.long_term_trend_history) < 50:
            return 0.0
        returns = np.diff(self.long_term_trend_history)[-50:]
        return float(np.std(returns))

    def generate_orders(self, order_book, market_context=None, last_trade=None):
        self.update_history(last_trade)
        if self.cooldown_ticks > 0:
            self.cooldown_ticks -= 1
            return []

        if len(self.close_history) < 50:
            return []

        last_price = order_book.last_trade_price
        if last_price is None:
            return []

        if self.is_volume_spike():
            return []  # избегаем резких вбросов

        volatility = self.calculate_volatility()
        imbalance = self.analyze_orderbook_imbalance(order_book)
        size = self.calculate_order_size(last_price, volatility)
        orders = []

        # Применяем стратегию для консолидации
        if self.is_consolidation():
            logger.info(f"[+] Агент {self.agent_id} обнаружил консолидацию на рынке! Стратегия: захват ликвидности.")
            orders.extend(self.execute_large_market_order(last_price, size, imbalance))

        # Используем долгосрочный тренд для манипуляции
        if self.long_term_trend_analysis() > 0.05:  # Если долгосрочный тренд указывает на изменение
            orders.extend(self.execute_in_trending_market(last_price, size, imbalance))

        num_orders = len(orders)
        logger.info(f"[{self.agent_id}] generated {num_orders} orders")

        return orders

    def execute_large_market_order(self, last_price, size, imbalance):
        """Захват позиции с помощью крупных маркет-ордеров, если рынок застыл"""
        orders = []
        if imbalance < -0.3:  # Если дисбаланс в стакане на стороне покупок
            order_size = size * 2  # Удваиваем размер ордера для более сильного захвата позиции
            logger.info(f"Манипулятор!! Агент {self.agent_id} размещает крупный ордер на продажу! Размер: {order_size} по цене: {last_price}")
            orders.append(Order(str(uuid.uuid4()), self.agent_id, OrderSide.ASK, order_size, None, OrderType.MARKET, None))  # Крупный ордер на продажу
        elif imbalance > 0.3:  # Если дисбаланс в стакане на стороне продаж
            order_size = size * 2  # Удваиваем размер ордера
            logger.info(f"Манипулятор!! Агент {self.agent_id} размещает крупный ордер на покупку! Размер: {order_size} по цене: {last_price}")
            orders.append(Order(str(uuid.uuid4()), self.agent_id, OrderSide.BID, order_size, None, OrderType.MARKET, None))  # Крупный ордер на покупку
        return orders

    def execute_in_trending_market(self, last_price, size, imbalance):
        """Стратегия для рынка с трендом"""
        orders = []
        if imbalance < -0.3:
            orders.append(Order(str(uuid.uuid4()), self.agent_id, OrderSide.ASK, size, None, OrderType.MARKET, None))
        elif imbalance > 0.3:
            orders.append(Order(str(uuid.uuid4()), self.agent_id, OrderSide.BID, size, None, OrderType.MARKET, None))
        return orders

    def on_order_filled(self, order_id, fill_price, fill_size, side):
        delta = fill_size if side == OrderSide.BID else -fill_size
        self.position += delta
        if self.last_action_price:
            move = abs(fill_price - self.last_action_price)
            if move > 0.002 * fill_price:
                self.manipulation_probability = min(1.0, self.manipulation_probability + 0.05)
            else:
                self.manipulation_probability = max(0.1, self.manipulation_probability - 0.05)

    def restore_capital(self):
        if self.capital < self.initial_capital * self.restore_threshold:
            restore_amount = self.initial_capital * 0.8
            self.capital = min(self.capital + restore_amount, self.initial_capital)
            print(f"[{self.agent_id}] Capital restored to: {self.capital}")

    def stochastic_decision(self):
        """Прогнозируем вероятность успеха манипуляции"""
        probability_of_success = self.calculate_probability_of_success()
        if random.random() < probability_of_success:
            return True  # Решение принято
        return False
