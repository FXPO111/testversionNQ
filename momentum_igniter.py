
import time
import numpy as np
import uuid
import random
from collections import deque
from order import Order, OrderSide, OrderType
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

class SmartSidewaysSqueezeAgent:
    def __init__(self, agent_id: str, capital: float, volatility_threshold: float = 0.02):
        self.agent_id = agent_id
        self.capital = capital
        self.initial_capital = capital
        self.position = 0.0
        self.entry_price = None
        self.max_position_size = capital * 0.1
        self.volatility_threshold = 0.015
        self.cooldown_ticks = 0
        self.last_action_time = 0.0
        self.orders = {}
        self.mood = 1.0
        self.stress = 0.0
        self.fatigue = 0
        self.trade_history = []
        self.loss_streak = 0
        self.win_streak = 0

        # История для различных индикаторов
        self.price_history = deque(maxlen=500)
        self.volume_history = deque(maxlen=500)
        self.long_term_trend_history = deque(maxlen=1000)
        self.position_history = deque(maxlen=200)
        
        # Параметры для манипуляции ликвидностью
        self.manipulation_probability = 0.6
        self.restore_threshold = 0.6  # Порог для восстановления капитала
        self.sideways_duration = 0

    def update_history(self, last_trade):
        """ Обновляет историю цен и объемов для анализа """
        if not last_trade:
            return
        price = float(last_trade['price'])
        volume = float(last_trade['volume'])
        self.price_history.append(price)
        self.volume_history.append(volume)
        self.position_history.append(self.position)
        # Долгосрочный тренд
        self.long_term_trend_history.append(price)

    def calculate_volatility(self):
        """ Рассчитывает волатильность на основе исторических цен """
        if len(self.price_history) < 50:
            return 0.0
        returns = np.diff(self.price_history)[-50:]
        return float(np.std(returns))

    def calculate_trend_strength(self):
        """ Рассчитывает силу тренда на основе исторических данных """
        if len(self.price_history) < 50:
            return 0.0
        returns = np.diff(self.price_history)[-50:]
        return float(np.abs(np.mean(returns)) / (np.std(returns) + 1e-6))

    def is_volume_spike(self):
        """ Проверяет на наличие всплеска объема на рынке """
        if len(self.volume_history) < 50:
            return False
        recent = np.array(self.volume_history)[-10:]
        avg = np.mean(self.volume_history[-50:-10])
        return np.any(recent > avg * 2.5)  # объём выше среднего более чем в 2.5 раза

    def is_consolidation(self):
        """ Проверяет фазу консолидации на основе цен и волатильности """
        if len(self.price_history) < 50:
            return False
        price_range = max(self.price_history[-50:]) - min(self.price_history[-50:])
        avg_price = np.mean(self.price_history[-50:])
        norm_range = price_range / avg_price
        volatility = self.calculate_volatility()
        trend_strength = self.calculate_trend_strength()
        return norm_range < 0.012 and volatility < self.volatility_threshold and trend_strength < 0.5

    def detect_channel_bounds(self):
        """ Определяет пределы ценового канала для консолидации """
        recent = list(self.price_history)[-50:]
        return min(recent), max(recent)

    def analyze_orderbook_imbalance(self, order_book, depth=5):
        """ Анализирует дисбаланс ордеров в стакане """
        snapshot = order_book.get_order_book_snapshot(depth=depth)
        bids = snapshot.get("bids", [])
        asks = snapshot.get("asks", [])
        bid_volume = sum(b['volume'] for b in bids)
        ask_volume = sum(a['volume'] for a in asks)
        total = bid_volume + ask_volume + 1e-8
        return (bid_volume - ask_volume) / total

    def calculate_order_size(self, price, volatility):
        """ Рассчитывает размер ордера с учетом волатильности и капитала """
        risk_budget = self.capital * 0.01
        stop_dist = max(volatility * 2, price * 0.001)
        size = risk_budget / stop_dist
        size = min(size, self.max_position_size - abs(self.position))
        return round(max(1.0, size), 2)

    def long_term_trend_analysis(self):
        """ Анализ долгосрочного тренда """
        if len(self.long_term_trend_history) < 50:
            return 0.0
        returns = np.diff(self.long_term_trend_history)[-50:]
        return float(np.std(returns))

    def adjust_mood_for_market_conditions(self, volatility, market_trend):
        """ Корректирует настроение агента в зависимости от рыночных условий """
        if volatility > 0.05:
            self.mood -= 0.1  # Высокая волатильность вызывает тревогу
        if market_trend == "up":
            self.mood += 0.1  # В восходящем тренде настроение улучшается

    def generate_orders(self, order_book, market_context=None, last_trade=None):
        """ Генерирует ордера на основе анализа рынка """       
        self.update_history(last_trade)
        if self.cooldown_ticks > 0:
            self.cooldown_ticks -= 1
            return []

        if len(self.price_history) < 50:
            return []

        last_price = order_book.last_trade_price
        if last_price is None:
            return []

        volatility = self.calculate_volatility()
        imbalance = self.analyze_orderbook_imbalance(order_book)
        size = self.calculate_order_size(last_price, volatility)
        market_trend = "up" if imbalance > 0 else "down"

        orders = []

        # Применяем стратегию для консолидации
        if self.is_consolidation():
            logger.info(f"[+] Агент {self.agent_id} обнаружил консолидацию на рынке! Стратегия: захват ликвидности.")
            self.sideways_duration += 1
            if self.sideways_duration >= 50:  # Длительность боковика 50 тиков
                orders.extend(self.execute_in_consolidation(last_price, size * 2, imbalance))
        else:
            self.sideways_duration = 0  # Сбрасываем таймер боковика, если он закончился

        # Используем долгосрочный тренд для манипуляции
        if self.long_term_trend_analysis() > 0.05:  # Если долгосрочный тренд указывает на изменение
            orders.extend(self.execute_in_trending_market(last_price, size, imbalance))

        # Применяем психологию рынка
        self.adjust_mood_for_market_conditions(volatility, market_trend)

        num_orders = len(orders)
        logger.info(f"[{self.agent_id}] generated {num_orders} orders")

        return orders

    def execute_in_consolidation(self, last_price, size, imbalance):
        """ Стратегия манипуляции ликвидностью в консолидации """
        orders = []
        if imbalance < -0.3:  # Если дисбаланс в сторону покупок
            orders.append(Order(str(uuid.uuid4()), self.agent_id, OrderSide.ASK, size * 1.5, last_price, OrderType.LIMIT, ttl=5))
        elif imbalance > 0.3:  # Если дисбаланс в сторону продаж
            orders.append(Order(str(uuid.uuid4()), self.agent_id, OrderSide.BID, size * 1.5, last_price, OrderType.LIMIT, ttl=5))
        return orders

    def execute_in_trending_market(self, last_price, size, imbalance):
        """ Стратегия для трендовых рынков """
        orders = []
        if imbalance < -0.3:
            orders.append(Order(str(uuid.uuid4()), self.agent_id, OrderSide.ASK, size, last_price, OrderType.LIMIT, ttl=5))
        elif imbalance > 0.3:
            orders.append(Order(str(uuid.uuid4()), self.agent_id, OrderSide.BID, size, last_price, OrderType.LIMIT, ttl=5))
        return orders

    def restore_capital(self):
        """ Восстанавливает капитал, если он падает ниже порога """
        if self.capital < self.initial_capital * self.restore_threshold:
            restore_amount = self.initial_capital * 0.8
            self.capital = min(self.capital + restore_amount, self.initial_capital)
            logger.info(f"Agent {self.agent_id} capital restored. New capital: {self.capital}")

    def update_psychology(self, pnl):
        """ Обновление психологического состояния агента """
        if pnl > 0:
            self.mood = min(1.5, self.mood + 0.1)
        else:
            self.mood = max(0.5, self.mood - 0.1)
