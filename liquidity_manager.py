import numpy as np
import uuid
from collections import deque
from order import Order, OrderSide, OrderType
import logging
import time

# Логирование
logger = logging.getLogger("OrderBookServer")

class LiquidityManagerAgent:
    def __init__(self, agent_id: str, capital: float, price_history_size: int = 20, max_levels: int = 3, restore_threshold: float = 0.5):
        self.agent_id = agent_id
        self.initial_capital = capital  # Изначальный капитал
        self.capital = capital  # Текущий капитал
        self.position = 0.0
        self.entry_price = None
        self.max_position_size = capital * 0.1  # Максимальный размер позиции

        # Параметры для восстановления капитала
        self.restore_threshold = restore_threshold  # Порог, при котором капитал восстанавливается (например, 50%)

        # История цен (используем deque с ограничением на количество элементов)
        self.price_history = deque(maxlen=price_history_size)  # Храним последние 20 цен

        # Параметры для обнаружения аномалий
        self.anomaly_multiplier = 1.0  # Множитель для аномалий
        self.trend_sensitivity = 0.05  # Порог для оценки тренда
        self.max_levels = max_levels  # Количество уровней для анализа ликвидности

        # История ордеров
        self.active_orders = {}  # активные ордера (order_id -> информация о ордере)

    def update_price_history(self, new_price: float):
        """
        Обновляем историю цен (добавляем новую цену в очередь).
        Если истории еще нет, она постепенно накапливается.
        """
        self.price_history.append(new_price)
        logger.debug(f"Price history updated. Current history: {list(self.price_history)}")

    def update_volume_stats(self, order_book_snapshot: dict):
        """
        Обновляем статистику объёмов на первых нескольких уровнях стакана.
        Агент анализирует средние объемы на различных уровнях стакана.
        """
        bids = order_book_snapshot['bids']
        asks = order_book_snapshot['asks']

        bid_volumes = [lvl['volume'] for lvl in bids]
        ask_volumes = [lvl['volume'] for lvl in asks]

        avg_bid_vol = np.mean(bid_volumes) if bid_volumes else 0
        avg_ask_vol = np.mean(ask_volumes) if ask_volumes else 0

        logger.debug(f"Average bid volume: {avg_bid_vol}, Average ask volume: {avg_ask_vol}")
        return avg_bid_vol, avg_ask_vol

    def detect_anomalies(self, order_book_snapshot: dict):
        """
        Ищем аномалии на основе объёма на уровнях стакана. Агент реагирует, если объём
        значительно превышает среднее.
        """
        avg_bid_vol, avg_ask_vol = self.update_volume_stats(order_book_snapshot)
        anomalies = []

        bids = order_book_snapshot['bids']
        asks = order_book_snapshot['asks']

        # Проверяем аномалии по биду
        for i, bid in enumerate(bids[:self.max_levels]):
            if bid['volume'] < avg_bid_vol:  # Меньше средней ликвидности
                anomalies.append(('bid', i, bid['price'], bid['volume']))
                

        # Проверяем аномалии по аску
        for i, ask in enumerate(asks[:self.max_levels]):
            if ask['volume'] < avg_ask_vol:  # Меньше средней ликвидности
                anomalies.append(('ask', i, ask['price'], ask['volume']))
                

        return anomalies

    def calculate_position_size(self, price: float, volatility: float) -> float:
        """
        Определяем размер позиции на основе текущей рыночной ситуации.
        Размер позиции адаптируется в зависимости от волатильности и доступного капитала.
        """
        size = self.capital * 0.05 / (volatility + 0.01)  # Используем небольшое добавление, чтобы избежать деления на ноль
        size = min(size, self.max_position_size)
        logger.debug(f"Calculated position size: {size} based on volatility: {volatility}")
        return size

    def add_liquidity(self, anomalies):
        """
        Создаёт ордера для поддержания ликвидности.
        Агент выставляет ордера на противоположной стороне, если ликвидности не хватает.
        """
        orders_to_add = []
        # Перебираем все аномалии и добавляем ордера на противоположной стороне
        for side, level, price, volume in anomalies:
            if side == 'ask':  # Если на уровне ask дефицит ликвидности, создаем ордер на покупку
                limit_price = price + 0.01  # Цена для ордера на покупку (BID)
                side_enum = OrderSide.BID
            elif side == 'bid':  # Если на уровне bid дефицит ликвидности, создаем ордер на продажу
                limit_price = price - 0.01  # Цена для ордера на продажу (ASK)
                side_enum = OrderSide.ASK

            size = self.calculate_position_size(limit_price, 0.05)  # Рассчитываем размер ордера в зависимости от волатильности

            orders_to_add.append(Order(
                order_id=str(uuid.uuid4()),
                agent_id=self.agent_id,
                side=side_enum,
                volume=round(size, 4),
                price=round(limit_price, 2),
                order_type=OrderType.LIMIT,
                ttl=3  # Время жизни ордера
            ))

        logger.info(f"Generated {len(orders_to_add)} orders to improve liquidity.")
        return orders_to_add

    def generate_orders(self, order_book, market_context=None):
       
        self._market_context = market_context
        perception = self.perceive_market(market_context) if hasattr(self, "_market_context") else "neutral"

        # В фазе "паники" — избегать входа
        if perception == "avoid" and np.random.rand() < 0.8:
            return []

        # В фазе "волатильности" — быть осторожным
        if perception == "hesitate" and np.random.rand() < 0.5:
            return []

        # В фазе "активности" — добавлять ордера
        if perception == "active":
            self.max_levels = 2  # Увеличиваем глубину стакана

        now = time.time()
        snapshot = order_book.get_order_book_snapshot()

        anomalies = self.detect_anomalies(snapshot)
        if anomalies:
            orders_to_add = self.add_liquidity(anomalies)
            return orders_to_add

        return []

    def restore_capital(self):
        """
        Восстанавливаем капитал, если он упал ниже порога.
        Пополняем капитал до 50% от начального значения.
        """
        if self.capital < self.initial_capital * self.restore_threshold:
            # Восстановление части капитала
            restore_amount = self.initial_capital * 0.5  # Восстанавливаем 50% от начального капитала
            self.capital = min(self.capital + restore_amount, self.initial_capital)  # Не превышаем начальный капитал
            logger.info(f"Capital restored. New capital: {self.capital}")

    def on_order_filled(self, order_id: str, fill_price: float, fill_size: float, side: OrderSide):
        """
        Обрабатываем исполнение ордера. Обновляем капитал и позицию.
        """
        if order_id not in self.active_orders:
            return

        info = self.active_orders.pop(order_id)
        pnl = 0.0

        if side == OrderSide.BID:
            pnl = - fill_price * fill_size  # Покупка - расходы
        else:
            pnl = fill_price * fill_size  # Продажа - прибыль

        self.capital += pnl

        # Восстанавливаем капитал, если он упал ниже порога
        self.restore_capital()

        # Обновляем позицию
        if side == OrderSide.BID:
            self.position += fill_size
        else:
            self.position -= fill_size

        # Регистрация ордера
        self.active_orders[order_id] = {'side': side, 'price': fill_price, 'size': fill_size}

        logger.info(f"Order {order_id} filled. PnL: {pnl}. New capital: {self.capital}")
        return pnl


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
            return "active"
        elif perceived_phase == "calm":
            return "supportive"
        return "neutral"
