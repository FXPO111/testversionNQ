# file: liquidity_hunter.py

import numpy as np
import time
import uuid

from order import Order, OrderSide, OrderType

class LiquidityHunterAgent:
    def __init__(self, agent_id: str, capital: float, restore_threshold: float = 0.6):
        self.agent_id = agent_id
        # Капитал и торговые параметры
        self.capital = capital
        self.position = 0.0
        self.entry_price = None
        self.max_position_size = capital * 0.1
        self.restore_threshold = restore_threshold
        self.initial_capital = capital



        # Параметры для анализа стакана
        self.depth_levels = 5
        self.avg_volume_per_level = []
        self.prev_volume_per_level = []

        # Эмоции и "живость"
        self.win_streak = 0
        self.loss_streak = 0
        self.mood = 1.0  # 1 — нейтральное, <1 — осторожность, >1 — агрессия
        self.stress = 0.0
        self.fatigue = 0
        self.cooldown_ticks = 0

        # Комиссии и проскальзывание
        self.commission_rate = 0.0005
        self.slippage_rate = 0.0003

        # История торговли
        self.last_pnl = 0.0
        self.pnl_history = []
        self.trade_history = []

        # Направление текущей позиции: 1=лонг, -1=шорт, 0=нет
        self.direction = 0

        # Храним активные лимитные ордера от этого агента
        self.active_orders = {}  # order_id -> order_info

        # Дополнительно: минимальный порог аномалии (с дробью)
        self.anomaly_multiplier = 2.0

    def update_volume_stats(self, order_book_snapshot: dict):
        """
        Обновляем скользящее среднее объёмов на первых depth_levels уровнях стакана.
        Добавлена адаптивная глубина: depth_levels может слегка меняться ±1.
        """
        # С вероятностью 5% варьируем depth_levels на ±1, но в пределах [3, 7]
        if np.random.rand() < 0.05:
            self.depth_levels = max(3, min(7, self.depth_levels + np.random.randint(-1, 2)))

        bids = order_book_snapshot['bids'][:self.depth_levels]
        asks = order_book_snapshot['asks'][:self.depth_levels]
        bid_volumes = [lvl['volume'] for lvl in bids]
        ask_volumes = [lvl['volume'] for lvl in asks]
        all_volumes = bid_volumes + ask_volumes

        if not self.avg_volume_per_level:
            # Инициализация при первом вызове
            self.avg_volume_per_level = all_volumes.copy()
            self.prev_volume_per_level = all_volumes.copy()
        else:
            alpha = 0.2 + np.random.uniform(-0.05, 0.05)  # стохастика α
            alpha = max(0.1, min(0.3, alpha))
            # EWMA
            self.avg_volume_per_level = [
                alpha * curr + (1 - alpha) * prev
                for curr, prev in zip(all_volumes, self.avg_volume_per_level)
            ]
            self.prev_volume_per_level = all_volumes.copy()

    def detect_anomalies(self, order_book_snapshot: dict):
        """
        Ищем аномалии: если объём на каком-либо уровне превышает anomaly_multiplier * средний,
        добавляем в список anomalies. Порог anomaly_multiplier слегка варьируется.
        """
        # Варьируем порог ±10%
        threshold = self.anomaly_multiplier * np.random.uniform(0.9, 1.1)

        anomalies = []
        bids = order_book_snapshot['bids'][:self.depth_levels]
        asks = order_book_snapshot['asks'][:self.depth_levels]

        # Проверяем биды
        for i, (bid, avg_vol) in enumerate(zip(bids, self.avg_volume_per_level[:self.depth_levels])):
            if avg_vol > 0 and bid['volume'] > threshold * avg_vol:
                anomalies.append(('bid', i, bid['price'], bid['volume']))

        # Проверяем аски
        for i, (ask, avg_vol) in enumerate(zip(asks, self.avg_volume_per_level[self.depth_levels:])):
            if avg_vol > 0 and ask['volume'] > threshold * avg_vol:
                anomalies.append(('ask', i, ask['price'], ask['volume']))

        return anomalies

    def calculate_delta_volume(self, order_book_snapshot: dict) -> float:
        """
        Считаем дельту объёмов на первых depth_levels уровнях (bid_volume - ask_volume).
        Добавлена стохастическая взвешенная комбинация текущих и предыдущих дельт.
        """
        bids = order_book_snapshot['bids'][:self.depth_levels]
        asks = order_book_snapshot['asks'][:self.depth_levels]
        delta_volume = 0.0

        for bid, ask in zip(bids, asks):
            buy_vol = bid.get('buy_volume', bid['volume'])
            sell_vol = ask.get('sell_volume', ask['volume'])
            delta_volume += buy_vol - sell_vol

        # Добавляем часть предыдущей дельты (гладим)
        prev_delta = getattr(self, 'prev_delta_vol', 0.0)
        weight = 0.7 + np.random.uniform(-0.1, 0.1)
        weight = max(0.5, min(0.9, weight))
        smoothed = weight * delta_volume + (1 - weight) * prev_delta
        self.prev_delta_vol = smoothed
        return float(smoothed)

    def update_mood(self):
        """
        Пересчёт настроения (mood) на основе серии выигрышей/потерь, стресса и усталости.
        Добавлена случайная корректировка mood ±5%.
        """
        if self.win_streak >= 3:
            self.mood = min(1.5, self.mood + 0.1)
        elif self.loss_streak >= 3:
            self.mood = max(0.5, self.mood - 0.1)

        # Учитываем стресс и усталость
        base_mood = self.mood - self.stress * 0.05 - self.fatigue * 0.02
        # Стохастический фактор ±5%
        base_mood *= np.random.uniform(0.95, 1.05)
        self.mood = max(0.3, min(1.5, base_mood))

    def calculate_position_size(self, price: float, volatility: float) -> float:
        """
        Размер позиции зависит от капитала, настроения (mood) и волатильности:
         - base_size = 5% капитала * mood * рандомизированный (0.8..1.2)
         - далее делим на волатильность + случайную «буферную» константу, 
           ограничиваем максимальным размером, и учитываем минимальный размер (1% капитала).
        """
        # Стохастический базовый размер ±20%
        size_factor = np.random.uniform(0.8, 1.2)
        base_size = self.capital * 0.05 * self.mood * size_factor

        # «Буфер» против нулевой или очень низкой волатильности
        buffer = np.random.uniform(0.01, 0.05)
        size = base_size / (volatility + buffer)
        size = min(size, self.max_position_size)
        size = max(size, self.capital * 0.01)  # минимум 1% капитала

        # Если стоимости больше, чем капитал, уменьшаем
        cost = size * price * (1 + self.commission_rate + self.slippage_rate)
        if cost > self.capital:
            size = self.capital / (price * (1 + self.commission_rate + self.slippage_rate))
        return float(size)

    def hunt_liquidity(self, order_book_snapshot: dict):
        """
        Основная логика «охоты за ликвидностью»:
         - обновляем статистику объёмов
         - детектим аномалии
         - считаем дельту объёмов и спрэд
         - если есть аномалия, входим против её стороны, иначе по дельте (при малом спрэдe).
         Добавлена случайная корректировка порога спрэдa и выбор уровня.
        """
        self.update_volume_stats(order_book_snapshot)
        anomalies = self.detect_anomalies(order_book_snapshot)
        delta_vol = self.calculate_delta_volume(order_book_snapshot)

        # Если нет котировок, остаёмся в режиме «hold»
        if not order_book_snapshot['bids'] or not order_book_snapshot['asks']:
            return 'hold', {}

        best_bid = order_book_snapshot['bids'][0]['price']
        best_ask = order_book_snapshot['asks'][0]['price']
        spread = best_ask - best_bid

        # Стохастический порог спрэдa: 0.5%..1.5% от mid
        mid_price = (best_bid + best_ask) / 2.0
        spread_threshold = mid_price * np.random.uniform(0.005, 0.015)

        # Решение 
        if anomalies:
            # Выбираем самую крупную аномалию с вероятность выбора по объёму
            volumes = np.array([vol for (_, _, _, vol) in anomalies])
            idx = np.argmax(volumes * np.random.uniform(0.8, 1.2, size=len(volumes)))
            side, level, price, volume = anomalies[idx]
            action = 'sell' if side == 'bid' else 'buy'
        else:
            # Торгуем по дельте, если спрэд < spread_threshold
            if delta_vol > 0 and spread < spread_threshold:
                action = 'buy'
            elif delta_vol < 0 and spread < spread_threshold:
                action = 'sell'
            else:
                action = 'hold'

        return action, {
            'spread': spread,
            'delta_vol': delta_vol,
            'anomalies': anomalies,
            'spread_threshold': spread_threshold
        }

    def generate_orders(self, order_book, market_context=None) -> list:
        """
        Вызывается из agent_loop() для генерации ордеров.
        Получаем срез стакана, определяем действие, формируем рыночный ордер.
        Добавлены случайные паузы при loss_streak и большой волатильности.
        """
        snapshot = order_book.get_order_book_snapshot(depth=self.depth_levels)
  
        # Фазовое восприятие
        perception = self.perceive_market(market_context)
        
        # === Трек фазы flat ===
        phase_name = ""
        if market_context and market_context.is_ready():
            phase_name = market_context.phase.name

        if phase_name == "calm" or phase_name == "flat":
            if self.last_flat_phase == phase_name:
                self.flat_phase_ticks += 1
            else:
                self.flat_phase_ticks = 1
                self.last_flat_phase = phase_name
        else:
            self.flat_phase_ticks = 0
            self.last_flat_phase = phase_name

        # === Fake breakout логика ===
        fake_breakout_orders = []
        if self.flat_phase_ticks > np.random.randint(16, 38) and np.random.rand() < 0.14:
            direction = np.random.choice(["up", "down"])
            best_bid = snapshot['bids'][0]['price']
            best_ask = snapshot['asks'][0]['price']
            mid = (best_bid + best_ask) / 2.0
            fake_size = self.calculate_position_size(mid, 0.05) * np.random.uniform(1.1, 1.6)
            if direction == "up":
                fake_price = best_ask + np.random.uniform(0.02, 0.04) * best_ask
                order = Order(
                    order_id=str(uuid.uuid4()),
                    agent_id=self.agent_id,
                    side=OrderSide.BID,
                    volume=fake_size,
                    price=round(fake_price, 2),
                    order_type=OrderType.MARKET,
                    ttl=None
                )
                fake_breakout_orders.append(order)
            else:
                fake_price = best_bid - np.random.uniform(0.02, 0.04) * best_bid
                order = Order(
                    order_id=str(uuid.uuid4()),
                    agent_id=self.agent_id,
                    side=OrderSide.ASK,
                    volume=fake_size,
                    price=round(fake_price, 2),
                    order_type=OrderType.MARKET,
                    ttl=None
                )
                fake_breakout_orders.append(order)
            self.cooldown_ticks = np.random.randint(2, 6)
            self.flat_phase_ticks = 0
            print(f"[{self.agent_id}] FAKE BREAKOUT {direction} в flat, объем {fake_size:.2f}")
            return fake_breakout_orders


        # Если в «кулдауне» — снижаем счетчик и отменяем все активные ордера
        if self.cooldown_ticks > 0:
            self.cooldown_ticks -= 1
            self.active_orders.clear()
            return []

        # Пересчитываем настроение
        self.update_mood()

        # Если испугался — воздерживаемся с 60% вероятностью
        if perception == "fear" and np.random.rand() < 0.6:
            self.cooldown_ticks = np.random.randint(2, 4)
            return []

        # При фазе "hesitation" — с 40% вероятностью пассивность
        if perception == "hesitation" and np.random.rand() < 0.4:
            return []

        # При фазе "opportunity" усиливаем желание действовать
        aggression_boost = 1.2 if perception in ["opportunity", "follow"] else 1.0

        # Анализ волатильности и возможная стохастическая пауза
        prices_for_vol = [lvl['price'] for lvl in snapshot['bids'] + snapshot['asks']]
        vol = np.std(prices_for_vol) if prices_for_vol else 0.0
        if vol > 0.07 and np.random.rand() < 0.3:
            self.cooldown_ticks = np.random.randint(1, 3)
            return []
        
        action, info = self.hunt_liquidity(snapshot)

        new_orders = []
        last_price = order_book.last_trade_price or 0.0
        volatility = abs(getattr(self, 'prev_delta_vol', 1.0)) or 1.0

        # Логика входа/выхода
        if action == 'buy' and self.position <= 0:
            size = self.calculate_position_size(last_price, volatility) * aggression_boost
            if size > 0:
                oid = str(uuid.uuid4())
                self.active_orders[oid] = {'side': 'buy', 'price': last_price, 'size': size}
                # Рыночный ордер
                order = Order(
                    order_id=oid,
                    agent_id=self.agent_id,
                    side=OrderSide.BID,
                    volume=size,
                    price=last_price,
                    order_type=OrderType.MARKET,
                    ttl=None
                )
                new_orders.append(order)
                # Обновляем внутренние данные о позиции
                self.position = size
                self.entry_price = last_price

        elif action == 'sell' and self.position >= 0:
            size = self.calculate_position_size(last_price, volatility) * aggression_boost
            if size > 0:
                oid = str(uuid.uuid4())
                self.active_orders[oid] = {'side': 'sell', 'price': last_price, 'size': size}
                order = Order(
                    order_id=oid,
                    agent_id=self.agent_id,
                    side=OrderSide.ASK,
                    volume=size,
                    price=last_price,
                    order_type=OrderType.MARKET,
                    ttl=None
                )
                new_orders.append(order)
                self.position = -size
                self.entry_price = last_price

        return new_orders

    def restore_capital(self):
        """
        Восстанавливаем капитал, если он упал ниже порога.
        Пополняем капитал до 80% от начального значения.
        """
        if self.capital < self.initial_capital * self.restore_threshold:
            # Восстановление части капитала
            restore_amount = self.initial_capital * 0.8  # Восстанавливаем 80% от начального капитала
            self.capital = min(self.capital + restore_amount, self.initial_capital)  # Не превышаем начальный капитал
            logger.info(f"Capital restored. New capital: {self.capital}")


    def on_order_filled(self, order_id: str, fill_price: float, fill_size: float, side: OrderSide):
        """
        Вызывается из match_and_update_loop(), когда Matching Engine сообщает о заполнении.
        Обновляем PnL, позицию, капитал, психологию.
        """
        if order_id not in self.active_orders:
            return
        info = self.active_orders.pop(order_id)

        pnl = 0.0
        if self.position == 0:
            # Открытие новой позиции
            self.entry_price = fill_price
            self.position = fill_size if side == OrderSide.BID else -fill_size
            self.direction = 1 if side == OrderSide.BID else -1
        else:
            # Если закрываем (позиция >0 и пришёл sell) или (позиция <0 и пришёл buy)
            if (self.position > 0 and side == OrderSide.ASK) or (self.position < 0 and side == OrderSide.BID):
                closing_size = min(abs(self.position), fill_size)
                pnl = closing_size * (fill_price - self.entry_price) * np.sign(self.position)
                self.position += fill_size if side == OrderSide.BID else -fill_size
                if abs(self.position) < 1e-6:
                    self.position = 0.0
                    self.entry_price = None
                    self.direction = 0
            else:
                # Усреднение позиции, если входим в ту же сторону
                total_pos = abs(self.position) + fill_size
                self.entry_price = (
                    self.entry_price * abs(self.position) + fill_price * fill_size
                ) / total_pos
                self.position += fill_size if side == OrderSide.BID else -fill_size
                self.direction = 1 if self.position > 0 else -1

        # Корректировка капитала с учётом комиссии/проскальзывания
        cost = fill_price * fill_size * (1 + self.commission_rate + self.slippage_rate)
        if side == OrderSide.BID:
            self.capital -= cost
        else:
            self.capital += fill_price * fill_size * (1 - self.commission_rate - self.slippage_rate)

        # Регистрация PnL и эмоций
        self.last_pnl = float(pnl)
        self.pnl_history.append(pnl)
        if pnl > 0:
            self.win_streak += 1
            self.loss_streak = 0
            self.stress = max(0.0, self.stress - 0.1)
        else:
            self.loss_streak += 1
            self.win_streak = 0
            self.stress += 0.1
            # Увеличиваем усталость при серии убытков
            if self.loss_streak >= 3:
                self.fatigue += 1
                # Случайный «перерыв» 3–7 тиков
                self.cooldown_ticks = np.random.randint(3, 8)

        # Логируем сделку
        self.trade_history.append({
            'order_id': order_id,
            'price': fill_price,
            'size': fill_size,
            'side': side.value,
            'pnl': pnl,
            'capital': self.capital,
            'position': self.position,
            'timestamp': time.time()
        })



    def perceive_market(self, market_context) -> str:
        if not market_context or not market_context.is_ready():
            return "neutral"

        perceived_phase = market_context.phase.name
        # Агент с 30% шансом может неверно интерпретировать фазу
        if np.random.rand() < 0.30:
            perceived_phase = np.random.choice(["calm", "volatile", "panic", "trend_up", "trend_down"])

        if perceived_phase == "panic":
            if self.stress > 0.3 or self.fatigue > 2:
                return "fear"
            else:
                return "alert"

        elif perceived_phase == "volatile":
            if self.mood > 1.1 and self.stress < 0.2:
                return "opportunity"
            else:
                return "hesitation"

        elif perceived_phase in ["trend_up", "trend_down"]:
            if self.mood > 1.0 and self.stress < 0.4:
                return "follow"
            else:
                return "doubt"

        return "neutral"
