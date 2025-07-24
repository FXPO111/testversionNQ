# Файл: retail_trader.py

import time
import numpy as np
import uuid
import random

from order import Order, OrderSide, OrderType

class RetailTrader:
    def __init__(
        self,
        agent_id: str,
        init_cash: float = 10_000,
        max_risk_per_trade: float = 0.2,
        restore_threshold: float = 0.3
    ):
        self.agent_id = agent_id
        self.cash = float(init_cash)
        self.position = 0.0
        self.entry_price = None
        self.pnl = 0.0
        self.trade_history = []
        self.max_risk_per_trade = float(max_risk_per_trade)  # доля капитала на одну сделку
        self.restore_threshold = restore_threshold

        # История «свечей» для расчёта индикаторов
        self.candles = []           # элементы: {'open','high','low','close','volume','timestamp'}
        self.rolling_window = 50    # максимальное число свечей для расчётов

        # Параметры индикаторов (будут немного колебаться в процессе)
        self.short_ma_period = 10
        self.long_ma_period = 30
        self.rsi_period = 14

        # Базовые стоп/тейк (±10% будут стохастически меняться)
        self.stop_loss_pct = 0.02   # 2%
        self.take_profit_pct = 0.04  # 4%

        # Ограничитель частоты (с небольшим рандомом)
        self.last_action_time = 0.0
        self.base_cooldown = 0.3    # секунда между новыми решениями

        # Состояние «психологии»
        self.loss_streak = 0
        self.win_streak = 0
        self.next_participation_time = None
        self.fatigue = 0            # нарастает при частых неудачах
        self.stress = 0.0           # нарастает при убытках, падает при победах
        self.mood = 1.0             # от 0.3 до 1.5 (чем ниже – тем более осторожно)

        # [MOD] Добавляем память о кластерном imbalance
        self.cluster_imbalance = 0.0

        # [MOD] Скользящее окно для волатильности
        self.vol_window = 20

    def add_candle(self, candle: dict):
        """
        Добавляет новую «свечу» (из последней сделки).
        Удаляет старую, если длина > rolling_window.
        С вероятностью 5% слегка подправляет параметры SMA/RSI.
        """
        self.candles.append(candle)
        if len(self.candles) > self.rolling_window:
            self.candles.pop(0)

        # 5% шанс внести небольшие изменения в периоды индикаторов
        if np.random.rand() < 0.05:
            self.short_ma_period = max(5, self.short_ma_period + np.random.randint(-1, 2))
            self.long_ma_period = max(
                self.short_ma_period + 5,
                self.long_ma_period + np.random.randint(-1, 2)
            )
            self.rsi_period = max(7, self.rsi_period + np.random.randint(-1, 2))

    def calculate_sma(self, period: int):
        """
        Простая скользящая средняя по последним period закрытиям.
        """
        if len(self.candles) < period:
            return None
        closes = [c['close'] for c in self.candles[-period:]]
        return float(np.mean(closes))

    def calculate_rsi(self, period: int):
        """
        RSI за period последних свечей.
        """
        if len(self.candles) < period + 1:
            return None
        closes = np.array([c['close'] for c in self.candles[-(period + 1):]])
        deltas = np.diff(closes)
        ups = deltas[deltas > 0].sum() / period
        downs = -deltas[deltas < 0].sum() / period
        if downs == 0:
            return 100.0
        rs = ups / downs
        return float(100 - (100 / (1 + rs)))

    def calculate_volatility(self, period: int):
        """
        Стандартное отклонение закрытия за period последних свечей.
        """
        if len(self.candles) < period:
            return None
        closes = np.array([c['close'] for c in self.candles[-period:]])
        return float(np.std(closes))

    def update_position_and_pnl(self, last_price: float):
        """
        Обновляет нереализованную PnL, если позиция открыта.
        """
        if self.position == 0:
            self.pnl = 0.0
            self.entry_price = None
            return
        self.pnl = float(self.position * (last_price - self.entry_price))

    def risk_management(self, last_price: float):
        """
        Проверяет, попали ли мы в стоп или тейк (±10% стохастики).
        Если да, возвращает 'close', иначе – None.
        """
        if self.position == 0 or self.entry_price is None:
            return None

        sl_pct = self.stop_loss_pct * np.random.uniform(0.9, 1.1)
        tp_pct = self.take_profit_pct * np.random.uniform(0.9, 1.1)

        if self.position > 0:
            loss_threshold = self.entry_price * (1 - sl_pct)
            profit_threshold = self.entry_price * (1 + tp_pct)
            if last_price <= loss_threshold or last_price >= profit_threshold:
                return 'close'
        else:
            loss_threshold = self.entry_price * (1 + sl_pct)
            profit_threshold = self.entry_price * (1 - tp_pct)
            if last_price >= loss_threshold or last_price <= profit_threshold:
                return 'close'

        return None

    def update_psychology(self, pnl: float):
        """
        Корректирует win_streak/loss_streak, stress, fatigue и mood.
        """
        if pnl > 0:
            self.win_streak += 1
            self.loss_streak = 0
            self.fatigue = max(0, self.fatigue - 1)
            self.stress = max(0.0, self.stress - 0.05)
        else:
            self.loss_streak += 1
            self.win_streak = 0
            self.fatigue += 1
            self.stress += 0.1

        # Если подряд 3 победы — настроение растёт
        if self.win_streak >= 3:
            self.mood = min(1.5, self.mood + 0.1)
        # Если подряд 3 поражения — настроение падает
        elif self.loss_streak >= 3:
            self.mood = max(0.3, self.mood - 0.1)

        # Учитываем стресс/усталость
        self.mood = max(0.3, min(1.5, self.mood - self.stress * 0.05 - self.fatigue * 0.02))

    def should_participate(self, market_context=None) -> bool:
        perception = self.perceive_market(market_context)
        now = time.time()
        
        if self.next_participation_time and now < self.next_participation_time:
            return False

        # Фазовая реакция
        if perception == "fear" and np.random.rand() < 0.6:
            self.next_participation_time = now + np.random.uniform(1.0, 2.0)
            return False
        
        if perception == "hesitation" and np.random.rand() < 0.4:
            self.next_participation_time = now + np.random.uniform(0.5, 1.5)
            return False

        if self.loss_streak >= 3:
            self.next_participation_time = now + np.random.uniform(1.0, 2.0)
            return False

        vol = self.calculate_volatility(self.long_ma_period)
        if vol is not None and vol > 0.05 and np.random.rand() < 0.5:
            self.next_participation_time = now + np.random.uniform(1.0, 2.0)
            return False

        # [MOD] Если кластерный imbalance слишком против позиции (если он есть), делаем паузу
        if abs(self.cluster_imbalance) > (self.cash * 0.001) and \
           ((self.position > 0 and self.cluster_imbalance < 0) or (self.position < 0 and self.cluster_imbalance > 0)):
            # слишком против: пауза 1–2 сек
            self.next_participation_time = now + np.random.uniform(1.0, 2.0)
            return False

        return True

    def _random_limit_sell(self, last_price: float):
        """
        Небольшая лимитная SELL, чтобы подпитать стакан (20% шанс вне позиции).
        """
        if self.position == 0 and np.random.rand() < 0.2:
            small_size = max(1, int(self.cash * 0.0005 / last_price))
            if small_size > 0:
                oid = str(uuid.uuid4())
                # [MOD] Price offset более динамичен: зависит от волатильности
                vol = self.calculate_volatility(self.vol_window) or 0.01
                price_offset = (0.001 + vol * 0.5) * last_price * np.random.uniform(0.8, 1.2)
                lim_price = round(last_price * (1 + price_offset / last_price), 2)
                return Order(
                    order_id=oid,
                    agent_id=self.agent_id,
                    side=OrderSide.ASK,
                    volume=small_size,
                    price=lim_price,
                    order_type=OrderType.LIMIT,
                    ttl=5
                )
        return None

    def _random_limit_buy(self, last_price: float):
        """
        Небольшая лимитная BUY, чтобы подпитать стакан (10% шанс вне позиции).
        """
        if self.position == 0 and np.random.rand() < 0.1:
            small_size = max(1, int(self.cash * 0.0005 / last_price))
            if small_size > 0:
                oid = str(uuid.uuid4())
                vol = self.calculate_volatility(self.vol_window) or 0.01
                price_offset = (0.001 + vol * 0.5) * last_price * np.random.uniform(0.8, 1.2)
                lim_price = round(last_price * (1 - price_offset / last_price), 2)
                return Order(
                    order_id=oid,
                    agent_id=self.agent_id,
                    side=OrderSide.BID,
                    volume=small_size,
                    price=lim_price,
                    order_type=OrderType.LIMIT,
                    ttl=5
                )
        return None

    def decide_trade(self, last_price: float):
        
        actions = []
        now = time.time()
        perception = self.perceive_market(self._market_context) if hasattr(self, "_market_context") else "neutral"
        aggression_boost = 1.2 if perception in ["follow", "opportunity", "greed"] else 1.0
        caution_factor = 0.5 if perception in ["fear", "hesitation"] else 1.0

        cooldown = self.base_cooldown * np.random.uniform(0.9, 1.1)
        if now - self.last_action_time < cooldown:
            return actions

        if perception == "fear" and np.random.rand() < 0.7:
            return actions 
       
        position_size = 0

        # 2) Случайная лимитная SELL
        if self.fatigue < 5:
            rnd_sell = self._random_limit_sell(last_price)
            if rnd_sell:
                actions.append(rnd_sell)
                self.last_action_time = now
                return actions

        # 2.1) Случайная лимитная BUY (чтобы подпитать bids)
        rnd_buy = self._random_limit_buy(last_price)
        if rnd_buy:
            actions.append(rnd_buy)
            self.last_action_time = now
            return actions

        # 3) Вычисляем индикаторы
        sma_short = self.calculate_sma(self.short_ma_period)
        sma_long  = self.calculate_sma(self.long_ma_period)
        rsi       = self.calculate_rsi(self.rsi_period)
        volatility = self.calculate_volatility(self.long_ma_period)

        if sma_short is None or sma_long is None or rsi is None or volatility is None:
            return actions

        # 4) Обновляем нереализованный PnL
        self.update_position_and_pnl(last_price)

        # 5) Стоп/тейк: mix Market vs. Limit (50/50)
        stop_take = self.risk_management(last_price)
        if stop_take == "close" and self.position != 0:
            side = OrderSide.ASK if self.position > 0 else OrderSide.BID
            size = abs(self.position)
            oid = str(uuid.uuid4())

            if np.random.rand() < 0.5:
                # MARKET‐exit
                order = Order(
                    order_id=oid,
                    agent_id=self.agent_id,
                    side=side,
                    volume=size,
                    price=None,
                    order_type=OrderType.MARKET,
                    ttl=None
                )
            else:
                # LIMIT‐exit: выходим чуть дальше от цены с учётом волатильности
                vol = volatility if volatility > 0 else 0.01
                price_offset = (0.002 + vol * 0.3) * last_price * np.random.uniform(0.8, 1.2)
                lim_price = (
                    last_price + price_offset if side == OrderSide.ASK else last_price - price_offset
                )
                order = Order(
                    order_id=oid,
                    agent_id=self.agent_id,
                    side=side,
                    volume=size,
                    price=round(lim_price, 2),
                    order_type=OrderType.LIMIT,
                    ttl=5
                )

            actions.append(order)

            # 5.1) Корректируем психологию
            if self.position > 0:
                pnl_val = size * (last_price - self.entry_price)
            else:
                pnl_val = size * (self.entry_price - last_price)
            self.update_psychology(pnl_val)

            self.position = 0
            self.entry_price = None
            self.last_action_time = now
            return actions

        # 6) Кроссовер SMA + RSI + учёт волатильности + кластерного imbalance
        if len(self.candles) < self.long_ma_period + 2:
            return actions

        closes = [c['close'] for c in self.candles]
        prev_short_vals = closes[-(self.short_ma_period + 1):-1]
        prev_long_vals  = closes[-(self.long_ma_period + 1):-1]

        prev_short_val = (
            float(np.mean(prev_short_vals)) if len(prev_short_vals) == self.short_ma_period else sma_short
        )
        prev_long_val = (
            float(np.mean(prev_long_vals)) if len(prev_long_vals) == self.long_ma_period else sma_long
        )

        crossover_up = (sma_short > sma_long) and (prev_short_val <= prev_long_val)
        crossover_down = (sma_short < sma_long) and (prev_short_val >= prev_long_val)

        # Если вне позиции, пробуем войти
        if self.position == 0:
            # 6.1) BUY
            if crossover_up and rsi < 70:
                # Размер позиции зависит от риска, волатильности и настроения
                risk_amount = self.cash * self.max_risk_per_trade * self.mood
                vol_factor = volatility if volatility > 1e-6 else 0.01
                raw_size = (risk_amount / (self.stop_loss_pct * last_price * vol_factor)) * np.random.uniform(0.9, 1.1)
                position_size = int(raw_size * aggression_boost * caution_factor)

                position_size = max(1, position_size)
                cost_est = position_size * last_price * (1 + 0.0005)  # плюс маленькая комиссия
                # [MOD] Если кластерный imbalance против, уменьшаем размер вдвое
                if self.cluster_imbalance < 0:
                    position_size = max(1, position_size // 2)

                if cost_est < self.cash and position_size > 0:
                    oid = str(uuid.uuid4())
                    # Если волатильность высокая, иногда вход ограниченным лимитом
                    if volatility > 0.03 and np.random.rand() < 0.5:
                        vol = volatility
                        price_offset = (0.002 + vol * 0.3) * last_price * np.random.uniform(0.8, 1.2)
                        lim_price = round(last_price - price_offset, 2)
                        order = Order(
                            order_id=oid,
                            agent_id=self.agent_id,
                            side=OrderSide.BID,
                            volume=position_size,
                            price=lim_price,
                            order_type=OrderType.LIMIT,
                            ttl=5
                        )
                    else:
                        order = Order(
                            order_id=oid,
                            agent_id=self.agent_id,
                            side=OrderSide.BID,
                            volume=position_size,
                            price=None,
                            order_type=OrderType.MARKET,
                            ttl=None
                        )
                    actions.append(order)
                    self.position = position_size
                    self.entry_price = last_price
                    self.last_action_time = now

            # 6.2) SELL (SHORT)
            elif crossover_down and rsi > 30:
                risk_amount = self.cash * self.max_risk_per_trade * self.mood
                vol_factor = volatility if volatility > 1e-6 else 0.01
                raw_size = (risk_amount / (self.stop_loss_pct * last_price * vol_factor)) * np.random.uniform(0.9, 1.1)
                position_size = int(raw_size * aggression_boost * caution_factor)

                position_size = max(1, position_size)
                # [MOD] Если кластерный imbalance против, уменьшаем размер вдвое
                if self.cluster_imbalance > 0:
                    position_size = max(1, position_size // 2)

                if position_size > 0:
                    oid = str(uuid.uuid4())
                    if volatility > 0.03 and np.random.rand() < 0.5:
                        vol = volatility
                        price_offset = (0.002 + vol * 0.3) * last_price * np.random.uniform(0.8, 1.2)
                        lim_price = round(last_price + price_offset, 2)
                        order = Order(
                            order_id=oid,
                            agent_id=self.agent_id,
                            side=OrderSide.ASK,
                            volume=position_size,
                            price=lim_price,
                            order_type=OrderType.LIMIT,
                            ttl=5
                        )
                    else:
                        order = Order(
                            order_id=oid,
                            agent_id=self.agent_id,
                            side=OrderSide.ASK,
                            volume=position_size,
                            price=None,
                            order_type=OrderType.MARKET,
                            ttl=None
                        )
                    actions.append(order)
                    self.position = -position_size
                    self.entry_price = last_price
                    self.last_action_time = now

        return actions

    def generate_orders(self, order_book, market_context=None) -> list:
        
        self._market_context = market_context

        last_price = order_book.last_trade_price
        if last_price is None:
            return []

        # 2) Формируем «свечу»
        candle = {
            'open': last_price,
            'high': last_price,
            'low': last_price,
            'close': last_price,
            'volume': 0.0,
            'timestamp': time.time(),
        }
        self.add_candle(candle)

        # 4) Обновляем кластерный imbalance
        snapshot = order_book.get_order_book_snapshot(depth=5)
        bid_vol = sum(lvl['volume'] for lvl in snapshot['bids'][:5])
        ask_vol = sum(lvl['volume'] for lvl in snapshot['asks'][:5])
        self.cluster_imbalance = bid_vol - ask_vol

        if not self.should_participate():
            return []

        return self.decide_trade(last_price)

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


    def on_order_filled(self, fill: dict):
        """
        Callback: вызывается в match_and_update_loop(), когда наш ордер исполнился.
        fill = {'side':'buy'/'sell','price':float,'volume':float,'timestamp':float}
        Обновляем позицию, капитал, считаем PnL, корректируем психологию, сохраняем трейд.
        """
        side = fill.get('side')
        volume = float(fill.get('volume', 0))
        price = float(fill.get('price', 0))
        timestamp = fill.get('timestamp', time.time())

        pnl_val = 0.0
        if side == 'buy':
            if self.position >= 0:
                # Усреднение в лонг
                if self.position == 0:
                    self.entry_price = price
                else:
                    total_vol = self.position + volume
                    self.entry_price = (
                        (self.entry_price or price) * self.position + price * volume
                    ) / total_vol
                self.position += volume
            else:
                # Закрытие шорта
                close_vol = min(abs(self.position), volume)
                pnl_val = close_vol * (self.entry_price - price)
                self.cash += pnl_val
                self.position += volume
                if self.position > 0:
                    self.entry_price = price
                elif self.position == 0:
                    self.entry_price = None

                self.update_psychology(pnl_val)

        else:  # side == 'sell'
            if self.position <= 0:
                # Усреднение в шорт
                if self.position == 0:
                    self.entry_price = price
                else:
                    total_vol = abs(self.position) + volume
                    self.entry_price = (
                        (self.entry_price or price) * abs(self.position) + price * volume
                    ) / total_vol
                self.position -= volume
            else:
                # Закрытие лонга
                close_vol = min(self.position, volume)
                pnl_val = close_vol * (price - self.entry_price)
                self.cash += pnl_val
                self.position -= volume
                if self.position < 0:
                    self.entry_price = price
                elif self.position == 0:
                    self.entry_price = None

                self.update_psychology(pnl_val)

        # Обновляем нереализованный PnL
        self.update_position_and_pnl(price)

        # Сохраняем историю трейдов
        self.trade_history.append({
            'type': 'fill',
            'side': side,
            'price': price,
            'volume': volume,
            'timestamp': timestamp,
            'position': self.position,
            'cash': self.cash,
            'pnl': self.pnl,
        })


    def perceive_market(self, market_context) -> str:
        if not market_context or not market_context.is_ready():
            return "neutral"

        perceived_phase = market_context.phase.name

        # С 35% шансом — неправильная интерпретация
        if np.random.rand() < 0.35:
            perceived_phase = random.choice(["calm", "volatile", "panic", "trend_up", "trend_down"])

        if perceived_phase == "panic":
            return "fear" if self.stress > 0.2 or self.fatigue > 3 else "alert"
        elif perceived_phase == "volatile":
            return "hesitation" if self.mood < 0.8 else "opportunity"
        elif perceived_phase in ["trend_up", "trend_down"]:
            return "follow" if self.mood > 1.0 and self.stress < 0.3 else "doubt"
        elif perceived_phase == "calm":
            return "relaxed"
        return "neutral"

