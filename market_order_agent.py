import uuid
import random
import numpy as np
from collections import deque
from order import Order, OrderSide, OrderType

class MarketOrderAgent:
    def __init__(self, agent_id: str, capital: float, restore_threshold: float = 0.3):
        self.agent_id = agent_id
        self.initial_capital = float(capital)
        self.capital = float(capital)
        self.positions = []
        self.max_position_risk = capital * 0.2
        self.daily_loss_limit = capital * 0.05
        self.daily_loss = 0.0
        self.restore_threshold = restore_threshold

        self.win_streak = 0
        self.loss_streak = 0
        self.stress = 0.0
        self.fatigue = 0
        self.cooldown_ticks = 0
        self.mood = 1.0

        self.commission_rate = 0.0005
        self.slippage_rate = 0.0002

        # Периоды индикаторов
        self.ema_period_fast = 12
        self.ema_period_slow = 26
        self.macd_signal_period = 9
        self.rsi_period = 14
        self.atr_period = 14
        self.bollinger_period = 20
        self.bollinger_std_dev = 2

        self.price_history = deque(maxlen=500)
        self.high_history = deque(maxlen=500)
        self.low_history = deque(maxlen=500)
        self.close_history = deque(maxlen=500)

        self.trade_history = []
        self.daily_pnl = []

    # ───── Индикаторы ─────

    def _ema(self, data, period):
        data_arr = np.array(data)
        if len(data_arr) < period:
            return np.array([data_arr[-1]])
        alpha = 2 / (period + 1)
        ema = [data_arr[0]]
        for price in data_arr[1:]:
            ema.append(alpha * price + (1 - alpha) * ema[-1])
        return np.array(ema)

    def _macd(self):
        if len(self.close_history) < self.ema_period_slow:
            return 0.0, 0.0, 0.0
        close = np.array(self.close_history)
        ema_fast = self._ema(close, self.ema_period_fast)
        ema_slow = self._ema(close, self.ema_period_slow)
        if len(ema_fast) > len(ema_slow):
            ema_fast = ema_fast[-len(ema_slow):]
        elif len(ema_slow) > len(ema_fast):
            ema_slow = ema_slow[-len(ema_fast):]
        macd_line = ema_fast - ema_slow
        macd_signal = self._ema(macd_line, self.macd_signal_period)
        macd_hist = macd_line[-len(macd_signal):] - macd_signal
        return float(macd_line[-1]), float(macd_signal[-1]), float(macd_hist[-1])

    def _rsi(self):
        close = np.array(self.close_history)
        if len(close) < self.rsi_period + 1:
            return 50.0
        deltas = np.diff(close)
        seed = deltas[:self.rsi_period]
        up = seed[seed >= 0].sum() / self.rsi_period
        down = -seed[seed < 0].sum() / self.rsi_period
        rs = up / down if down != 0 else 0
        rsi = 100 - 100 / (1 + rs)
        for delta in deltas[self.rsi_period:]:
            upval = max(delta, 0)
            downval = -min(delta, 0)
            up = (up * (self.rsi_period - 1) + upval) / self.rsi_period
            down = (down * (self.rsi_period - 1) + downval) / self.rsi_period
            rs = up / down if down != 0 else 0
            rsi = 100 - 100 / (1 + rs)
        return float(rsi)

    def _atr(self):
        highs = np.array(self.high_history)
        lows = np.array(self.low_history)
        closes = np.array(self.close_history)
        if len(closes) < self.atr_period + 1:
            return float(np.std(closes)) if len(closes) > 0 else 0.01
        tr = np.maximum(highs[1:], closes[:-1]) - np.minimum(lows[1:], closes[:-1])
        atr = [float(np.mean(tr[:self.atr_period]))]
        for i in range(self.atr_period, len(tr)):
            atr.append((atr[-1] * (self.atr_period - 1) + float(tr[i])) / self.atr_period)
        return float(atr[-1])

    def _bollinger_bands(self):
        if len(self.close_history) < self.bollinger_period:
            return None, None
        close = np.array(self.close_history)
        rolling_mean = np.mean(close[-self.bollinger_period:])
        rolling_std = np.std(close[-self.bollinger_period:])
        upper_band = rolling_mean + (self.bollinger_std_dev * rolling_std)
        lower_band = rolling_mean - (self.bollinger_std_dev * rolling_std)
        return upper_band, lower_band

    def calculate_indicators(self):
        price = float(self.price_history[-1])
        ema_fast = float(self._ema(self.close_history, self.ema_period_fast)[-1]) if len(self.close_history) >= self.ema_period_fast else price
        ema_slow = float(self._ema(self.close_history, self.ema_period_slow)[-1]) if len(self.close_history) >= self.ema_period_slow else price
        macd, macd_signal, macd_hist = self._macd()
        rsi = self._rsi()
        atr = self._atr()
        boll_upper, boll_lower = self._bollinger_bands()

        return {
            "price": price,
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "macd": macd,
            "macd_signal": macd_signal,
            "macd_hist": macd_hist,
            "rsi": rsi,
            "atr": atr,
            "boll_upper": boll_upper,
            "boll_lower": boll_lower,
        }

    def can_trade(self) -> bool:
        return self.daily_loss < self.daily_loss_limit

    def calculate_position_size(self, volatility: float, price: float) -> float:
        buffer = np.random.uniform(0.01, 0.05)
        if volatility <= 0:
            volatility = 0.01
        stop_loss_distance = 2 * (volatility + buffer)
        risk_factor = np.random.uniform(0.8, 1.2)
        risk_amount = self.capital * 0.01 * self.mood * risk_factor
        size = risk_amount / stop_loss_distance
        max_size = (self.capital / price) * 0.1 * np.random.uniform(0.8, 1.2)
        size = min(size, max_size)
        min_size = (self.capital * 0.001) / price
        size = max(size, min_size)
        return float(size)

    def decide_action(self, indicators: dict) -> str:
        score = self.score_trade_signal(indicators)
        threshold = 1.0 * np.random.uniform(0.7, 1.3)
        if score > threshold:
            return "buy"
        elif score < -threshold:
            return "sell"
        else:
            return "hold"

    def score_trade_signal(self, indicators: dict) -> float:
        score = 0.0
        price = indicators["price"]
        ema_fast = indicators["ema_fast"]
        ema_slow = indicators["ema_slow"]
        rsi = indicators["rsi"]
        macd_hist = indicators["macd_hist"]
        boll_upper, boll_lower = indicators["boll_upper"], indicators["boll_lower"]

        if ema_fast > ema_slow:
            score += 1.0
        else:
            score -= 1.0

        if rsi < 30:
            score += 1.5
        elif rsi > 70:
            score -= 1.5

        if macd_hist > 0:
            score += 1.0
        else:
            score -= 1.0

        if boll_upper is not None and price > boll_upper:
            score -= 1.5
        if boll_lower is not None and price < boll_lower:
            score += 1.5

        score *= self.mood
        score += np.random.normal(0, 0.3)
        return float(score)

    def generate_orders(self, order_book, market_context=None) -> list:
        orders_to_send = []
        last_price = order_book.last_trade_price
        if last_price is None:
            return []

        self.high_history.append(last_price)
        self.low_history.append(last_price)
        self.close_history.append(last_price)
        self.price_history.append(last_price)

        indicators = self.calculate_indicators()
        action = self.decide_action(indicators)

        if action == "hold" or not self.can_trade():
            return []

        volatility = indicators.get("atr", 0.01)
        size = self.calculate_position_size(volatility, last_price)
        size = max(size, 1.0)

        side_enum = OrderSide.BID if action == "buy" else OrderSide.ASK

        # Всегда генерируем только маркет-ордер
        new_order = Order(
            order_id=str(uuid.uuid4()),
            agent_id=self.agent_id,
            side=side_enum,
            volume=round(size, 4),
            price=None,  # Для маркет-ордера цена не указывается
            order_type=OrderType.MARKET,  # Устанавливаем тип ордера как MARKET
            ttl=None
        )

        orders_to_send.append(new_order)
        return orders_to_send

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


