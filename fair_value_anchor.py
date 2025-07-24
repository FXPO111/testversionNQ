import time
import uuid
import numpy as np

from order import Order, OrderSide, OrderType

class FairValueAnchorAgent:
    """
    Реалистичный агент-якорь, который оценивает "справедливую" цену активa
    и с вероятностью, зависящей от отклонения рынка от этой цены, выступает
    лимитным поставщиком ликвидности, стараясь вернуть рынок к справедливому значению.
    """

    def __init__(self, agent_id: str, capital: float, alpha: float = 0.03, restore_threshold: float = 0.6):
        self.agent_id = agent_id
        self.capital = float(capital)
        self.position = 0.0
        self.entry_price = None
        self.pnl = 0.0
        self.trade_history = []
        self.restore_threshold = restore_threshold

        # Fair value и его EMA-параметр
        self.fair_value = None
        self.alpha = alpha  # сглаживание для EMA fair_value

        # История mid-цен для волатильности
        self.mid_prices = []
        self.volatility = 0.0

        # Параметры психологии
        self.confidence = 0.5       # от 0 до 1, чем выше — выше шанс действовать
        self.conviction = 0.0       # накопленная сила убеждения
        self.risk_aversion = 0.3    # от 0 до 1, чем выше — меньшие объёмы
        self.stress = 0.0
        self.fatigue = 0

        # Тайминг и детализация действий
        self.last_action_time = 0.0
        self.base_cooldown = 1.0    # секунда между действиями
        self.cooldown_noise = 0.5   # случайная флуктуация

        # Анти-флуд: избегать слишком частых коррекций
        self.avoid_ticks = 0

        # History for EMA of volatility
        self.mid_diffs = []

    def update_state(self, snapshot: dict):
        """
        Обновляет internal state: fair_value, volatility, conviction.
        """
        bids = snapshot.get("bids", [])
        asks = snapshot.get("asks", [])
        if not bids or not asks:
            return

        best_bid = bids[0]["price"]
        best_ask = asks[0]["price"]
        mid = (best_bid + best_ask) / 2.0

        # Инициализация fair_value первым mid
        if self.fair_value is None:
            self.fair_value = mid
        else:
            # EMA для fair_value
            self.fair_value = self.alpha * mid + (1 - self.alpha) * self.fair_value

        # Обновляем историю mid-цен и рассчитываем волатильность
        if self.mid_prices:
            diff = mid - self.mid_prices[-1]
            self.mid_diffs.append(diff)
            if len(self.mid_diffs) > 50:
                self.mid_diffs.pop(0)

        self.mid_prices.append(mid)
        if len(self.mid_prices) > 100:
            self.mid_prices.pop(0)

        if len(self.mid_diffs) >= 5:
            self.volatility = float(np.std(self.mid_diffs[-20:]))
        else:
            self.volatility = 0.0

        # Обновление conviction: чем дольше цена удаляется от fair_value, тем сильнее conviction
        mispricing = (mid - self.fair_value) / (self.fair_value + 1e-8)
        self.conviction = np.tanh(abs(mispricing) * 10)  # от 0 до 1
        # Чем выше mispricing, тем выше conviction

        # Психологическое состояние влияет: если стресс большой, confidence падает
        self.confidence = max(0.1, min(1.0, 0.5 + self.conviction * 0.5 - self.stress * 0.2))

    def compute_mispricing(self, snapshot: dict) -> float:
        """
        Возвращает относительное отклонение текущего mid от fair_value.
        """
        bids = snapshot.get("bids", [])
        asks = snapshot.get("asks", [])
        if not bids or not asks or self.fair_value is None:
            return 0.0

        mid = (bids[0]["price"] + asks[0]["price"]) / 2.0
        return (mid - self.fair_value) / (self.fair_value + 1e-8)

    def should_participate(self) -> bool:
        
        perception = self.perceive_market(self._market_context) if hasattr(self, "_market_context") else "neutral"

        if perception == "caution" and np.random.rand() < 0.7:
            return False

        if perception == "stressful" and np.random.rand() < 0.5:
            self.stress = min(1.0, self.stress + 0.05)
            return False

        if perception == "unfair":
            self.confidence = min(1.0, self.confidence + 0.1)

        if perception == "stable":
            self.confidence = max(0.3, self.confidence - 0.05)


        now = time.time()
        if self.avoid_ticks > 0:
            self.avoid_ticks -= 1
            return False

        # Cooldown с шумом
        interval = self.base_cooldown + np.random.uniform(-self.cooldown_noise, self.cooldown_noise)
        if now - self.last_action_time < interval:
            return False

        # Если стресс или fatigue слишком велики — реже участвует
        if self.stress > 0.7 or self.fatigue > 5:
            if np.random.rand() < 0.6:
                self.last_action_time = now
                return False

        # Иначе — участвует с вероятностью confidence
        return np.random.rand() < self.confidence

    def generate_orders(self, order_book, market_context=None) -> list:
        
        self._market_context = market_context
        perception = self.perceive_market(market_context) if hasattr(self, "_market_context") else "neutral"


        # Если ещё в avoid режиме или cooldown
        if not self.should_participate():
            return []

        snapshot = order_book.get_order_book_snapshot(depth=10)
        bids = snapshot.get("bids", [])
        asks = snapshot.get("asks", [])
        if not bids or not asks:
            return []

        self.update_state(snapshot)
        mispricing = self.compute_mispricing(snapshot)

        # Порог mispricing, при котором реагируем
        threshold = 0.01 + self.volatility * 0.5  # чем выше волатильность, тем выше порог
        if abs(mispricing) < threshold:
            return []

        best_bid = bids[0]["price"]
        best_ask = asks[0]["price"]
        mid = (best_bid + best_ask) / 2.0

        new_orders = []
        now = time.time()

        # Если mispricing > 0: цена выше fair_value → выставляем ask ниже текущих asks, чтобы подтянуть вниз
        if mispricing > threshold:
            # Рассчитываем уровни: несколько лимитов, начинаем чуть ниже лучшего ask
            levels = max(1, int(self.conviction * 3))  # от 1 до 3 уровней
            base_volume = (self.capital * (0.005 + self.conviction * 0.01)) / mid
            base_volume *= (1 - self.risk_aversion * 0.5)  # чем выше risk_aversion, тем меньше объём

            for lvl in range(levels):
                # price_offset зависит от conviction и уровня
                offset = (best_ask - self.fair_value) * (0.2 + 0.1 * lvl) + self.volatility * mid * 0.1 * lvl
                price = max(best_ask - offset, self.fair_value)
                price = round(price, 2)
                size = max(1.0, round(base_volume * (1.0 - 0.3 * lvl)))

                # Анти-слип: если price >= best_ask, отказываемся
                if price >= best_ask:
                    price = round(best_ask * (1 + self.volatility * 0.05), 2)

                oid = str(uuid.uuid4())
                self.last_action_time = now

                order = Order(
                    order_id=oid,
                    agent_id=self.agent_id,
                    side=OrderSide.ASK,
                    volume=size,
                    price=price,
                    order_type=OrderType.LIMIT,
                    ttl=5  # лимитки живут 5 сек
                )
                new_orders.append(order)

        # Если mispricing < 0: цена ниже fair_value → выставляем bid выше текущих bids, чтобы подтянуть вверх
        elif mispricing < -threshold:
            levels = max(1, int(self.conviction * 3))
            base_volume = (self.capital * (0.005 + self.conviction * 0.01)) / mid
            base_volume *= (1 - self.risk_aversion * 0.5)

            for lvl in range(levels):
                offset = (self.fair_value - best_bid) * (0.2 + 0.1 * lvl) + self.volatility * mid * 0.1 * lvl
                price = min(best_bid + offset, self.fair_value)
                price = round(price, 2)
                size = max(1.0, round(base_volume * (1.0 - 0.3 * lvl)))

                if price <= best_bid:
                    price = round(best_bid * (1 - self.volatility * 0.05), 2)

                oid = str(uuid.uuid4())
                self.last_action_time = now

                order = Order(
                    order_id=oid,
                    agent_id=self.agent_id,
                    side=OrderSide.BID,
                    volume=size,
                    price=price,
                    order_type=OrderType.LIMIT,
                    ttl=5
                )
                new_orders.append(order)

        # После выставления лимиток — вводим небольшую залипку, чтобы нельзя было «спамить»
        self.avoid_ticks = levels * 2

        return new_orders

    def restore_capital(self):
        """
        Восстанавливаем капитал, если он упал ниже порога.
        Пополняем капитал до 50% от начального значения.
        """
        if self.capital < self.initial_capital * self.restore_threshold:
            # Восстановление части капитала
            restore_amount = self.initial_capital * 0.8  # Восстанавливаем 80% от начального капитала
            self.capital = min(self.capital + restore_amount, self.initial_capital)  # Не превышаем начальный капитал
            logger.info(f"Capital restored. New capital: {self.capital}")


    def on_order_filled(self, order_id: str, price: float, volume: float, side: OrderSide):
        """
        Когда лимитка этого агента исполняется, обновляем психологию и учет позиций.
        """
        pnl = 0.0
        now = time.time()

        # Если исполнилось BUY (SID= BID): мы купили по этой цене
        if side == OrderSide.BID:
            self.position += volume
            self.capital -= price * volume
            if self.entry_price is None:
                self.entry_price = price

            # Пока удерживаем позицию, не считаем PnL
        else:
            # Если SELL: закрываем долгую позицию или открываем короткую
            if self.position > 0:
                closing_size = min(self.position, volume)
                pnl = (price - self.entry_price) * closing_size
                self.capital += price * closing_size
                self.position -= closing_size
                if self.position <= 1e-6:
                    self.position = 0.0
                    self.entry_price = None
            else:
                # Открываем шорт, пусть шорт поза учитывается
                self.position -= volume
                self.entry_price = price  # вход в шорт

        # Психологическая реакция
        if pnl > 0:
            # Радость: confidence растёт, stress падает
            self.confidence = min(1.0, self.confidence + 0.05)
            self.stress = max(0.0, self.stress - 0.05)
            self.fatigue = max(0, self.fatigue - 1)
        else:
            # Поражение: stress растёт, confidence падает
            self.stress = min(1.0, self.stress + 0.1)
            self.confidence = max(0.0, self.confidence - 0.05)
            self.fatigue += 1

        self.pnl += pnl
        self.trade_history.append({
            'order_id': order_id,
            'price': price,
            'volume': volume,
            'side': side.value,
            'pnl': pnl,
            'capital': self.capital,
            'position': self.position,
            'timestamp': now
        })

    def perceive_market(self, market_context) -> str:
        if not market_context or not market_context.is_ready():
            return "neutral"
        phase = market_context.phase.name
        if np.random.rand() < 0.15:
            phase = random.choice(["panic", "volatile", "trend_up", "trend_down", "calm"])
        if phase == "panic":
            return "caution"
        elif phase == "volatile":
            return "stressful"
        elif phase in ["trend_up", "trend_down"]:
            return "unfair"
        elif phase == "calm":
            return "stable"
        return "neutral"

