import uuid
import time
import numpy as np
import random
import logging
from order import Order, OrderSide, OrderType
from market_context import MarketContextAdvanced

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StrategistAgent:
    def __init__(self, agent_id: str, capital: float, volatility_threshold: float = 0.02):
        self.agent_id = agent_id
        self.capital = capital
        self.initial_capital = capital
        self.position = 0.0
        self.entry_price = None
        self.pnl = 0.0
        self.trade_history = []

        # Параметры для долгосрочной стратегии
        self.volatility_threshold = volatility_threshold
        self.cooldown_range = (1.5, 4.0)
        self.long_term_goal = np.random.choice(["up", "down"])  # Направление, куда агент хочет доставить цену
        self.long_term_plan = []  # План движения на несколько шагов вперед
        self.plan_progress = 0
        self.sideways_duration = 0

        # Психология и эмоции
        self.stress = 0.0
        self.euphoria = 0.0
        self.mood = 1.0  # Общее настроение
        self.confidence = 1.0  # Уверенность агента в своих действиях
        self.emotional_weight = 0.05  # Влияние психологии на поведение агента

        # Параметры для долгосрочного планирования
        self.time_horizon = random.randint(1, 24)  # Количество часов, на которое агент планирует (от 1 до 24 часов)
        self.learning_rate = 0.1  # Скорость обучения агента от опыта

        # Модели предсказания
        self.price_model = None  # Модель для прогнозирования цены
        self.market_model = None  # Модель для анализа рыночных фаз и трендов

        # Инициализация market_context
        self.market_context = MarketContextAdvanced()

    def update_state(self, snapshot):
        """Обновление состояния рынка для принятия решения"""
        bids = snapshot.get("bids", [])
        asks = snapshot.get("asks", [])
        if not bids or not asks:
            return
        best_bid = bids[0]["price"]
        best_ask = asks[0]["price"]
        mid = (best_bid + best_ask) / 2.0
        self.volatility = np.std(np.diff([best_bid, best_ask]))  # Вычисление волатильности на основе спроса и предложения

    def compute_imbalance(self, snapshot):
        """Анализирует дисбаланс в стакане"""
        bids = snapshot.get("bids", [])
        asks = snapshot.get("asks", [])
        bid_vol = sum([lvl["volume"] for lvl in bids[:5]])
        ask_vol = sum([lvl["volume"] for lvl in asks[:5]])
        return bid_vol - ask_vol, bid_vol, ask_vol

    def predict_price_movement(self):
        """Прогнозирование движения цены с использованием модели"""
        if self.price_model:
            prediction = self.price_model.predict()  # Модель предсказания движения цены
            return prediction
        return 0.0

    def adjust_strategy(self, market_context):
        """Адаптация стратегии на основе текущего рыночного контекста"""
        # Используем модель для предсказания рыночных фаз
        if self.market_model:
            phase_prediction = self.market_model.predict(market_context)
            return phase_prediction
        return "neutral"

    def decide_action(self, imbalance, market_context):
        """Решение о действии на основе долгосрочного плана и текущего контекста"""
        predicted_phase = self.adjust_strategy(market_context)
        if predicted_phase == "sideways" and imbalance > 20:
            return OrderSide.BID
        elif predicted_phase == "volatile" and imbalance < -20:
            return OrderSide.ASK
        return None

    def place_large_order(self, size, direction):
        """Размещение крупного ордера для достижения долгосрочной цели"""
        order = Order(
            order_id=str(uuid.uuid4()),
            agent_id=self.agent_id,
            side=direction,
            volume=size,
            price=None,
            order_type=OrderType.MARKET,
            ttl=None
        )
        self.execute_order(order)
        logger.info(f"[{self.agent_id}] placed large order to move price: {order}")

    def execute_order(self, order):
        """Исполнение ордера"""
        self.trade_history.append(order)
        self.position += order.volume if order.side == OrderSide.BID else -order.volume
        self.entry_price = order.price
        logger.info(f"[{self.agent_id}] executed order: {order}")

    def generate_orders(self, order_book):
        """Генерация ордеров на основе анализа рынка"""
        snapshot = order_book.get_order_book_snapshot(depth=5)
        if not snapshot.get("bids") or not snapshot.get("asks"):
            return []

        # Обновляем market_context с текущим состоянием snapshot
        self.market_context.update(snapshot, sweep_occurred=True)

        # Пропускаем действия, если market_context ещё не готов
        if not self.market_context or not self.market_context.phase.confidence:
            logger.info(f"[{self.agent_id}] Market context is not ready. Skipping order generation.")
            return []

        self.update_state(snapshot)
        imbalance, bid_vol, ask_vol = self.compute_imbalance(snapshot)

        self.plan_next_move()

        if self.sideways_duration >= 50:
            self.place_large_order(self.capital * 0.1, OrderSide.BID if self.long_term_goal == "up" else OrderSide.ASK)

        side = self.decide_action(imbalance, self.market_context)
        if side:
            size = self.capital * 0.1 / (self.volatility + 1e-6)
            self.place_large_order(size, side)

        self.last_action_time = time.time()
        logger.info(f"[{self.agent_id}] generated orders based on market analysis.")
        return []

    def plan_next_move(self):
        """Моделируем долгосрочную стратегию"""
        if self.plan_progress < len(self.long_term_plan):
            action = self.long_term_plan[self.plan_progress]
            if action["type"] == "move_price":
                self.place_large_order(action["size"], action["direction"])
                self.plan_progress += 1
                return []

        # Определение корректировки стратегии, если стратегия не даёт результатов
        if self.sideways_duration > 50:
            self.long_term_goal = np.random.choice(["up", "down"])
            self.plan_progress = 0
            self.long_term_plan = [{"type": "move_price", "size": self.capital * 0.1, "direction": self.long_term_goal}]

    def adapt_to_market_conditions(self, snapshot):
        """Адаптация стратегии к изменениям на рынке"""
        if self.volatility > self.volatility_threshold:
            self.confidence = max(0.5, self.confidence - self.learning_rate)
        else:
            self.confidence = min(1.0, self.confidence + self.learning_rate)
        
        # Психологическое влияние: стресс и эйфория
        if self.position > 0:
            self.euphoria = min(1.0, self.euphoria + 0.05)
        else:
            self.stress = min(1.0, self.stress + 0.05)

        # Психологические реакции на рынок
        if self.euphoria > 0.7:
            self.long_term_goal = np.random.choice(["up", "down"])
        
        if self.confidence > 0.9 and self.euphoria < 0.5:
            self.long_term_goal = np.random.choice(["up", "down"])
        elif self.stress > 0.7:
            self.confidence = max(0.2, self.confidence - self.learning_rate)
        
        if random.random() < 0.05:  # Добавление случайности для неопределенности
            self.confidence *= random.uniform(0.8, 1.2)  # Небольшая случайность для нестабильности

    def restore_capital(self):
        """Восстановление капитала, если он упал ниже порога"""
        restore_threshold = self.initial_capital * 0.6
        if self.capital < restore_threshold:
            recovery_factor = 1.0
            if self.stress > 0.5:
                recovery_factor = 0.5
            elif self.euphoria > 0.7:
                recovery_factor = 1.5

            restore_amount = (self.initial_capital * 0.7) * recovery_factor
            self.capital = min(self.capital + restore_amount, self.initial_capital)  # Не превышаем начальный капитал
            logger.info(f"[{self.agent_id}] Capital restored to {self.capital}. Stress: {self.stress}, Euphoria: {self.euphoria}")

