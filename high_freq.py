# Файл: high_freq.py

import numpy as np
import time
import uuid
import random

from order import Order, OrderSide, OrderType

class HighFreqAgent:
    def __init__(self, agent_id: str, capital: float, restore_threshold: float = 0.6):
        self.agent_id = agent_id
        self.capital = float(capital)
        self.position = 0.0             # + для лонга, - для шорта
        self.entry_price = None
        self.max_position_size = capital * 0.1
        self.restore_threshold = restore_threshold
        self.initial_capital = capital

        # Инициализация pnl_history
        self.pnl_history = []

        # Инициализация trade_history
        self.trade_history = []


        # Целевой спред (0.05%) с небольшими вариациями
        self.spread_target = 0.0005
        self.order_lifetime = 5           # тики жизни лимитных ордеров
        self.orders = {}                  # { order_id: { 'side', 'price', 'size', 'timestamp' } }

        # Комиссии и проскальзывания
        self.commission_rate = 0.0002
        self.slippage_rate = 0.0001

        # Эмоции / статистика
        self.win_streak = 0
        self.loss_streak = 0
        self.mood = 1.0
        self.stress = 0.0
        self.fatigue = 0

        # Контроль частоты квотирования
        self.next_quote_time = time.time()
        self.cooldown_ticks = 0

        # [MOD] Хранение истории mid_price для анализа микротенденций
        self.mid_price_history = []
        self.trend_strength = 0.0
        self.trend_direction = 0

        # [MOD] Кластерный imbalance
        self.cluster_imbalance = 0.0

        # [MOD] Скрытые ордера для «фиктивной» глубины
        self.hidden_orders = {}

    def update_mood(self):
        """
        Пересчёт параметра mood:
          - если много выигрышей подряд, mood растёт
          - если много убытков подряд, mood падает
          - стресс и усталость тоже отнимают балл к mood
          - добавлена стохастическая корректировка ±5%.
        """
        if self.win_streak >= 5:
            self.mood = min(1.5, self.mood + 0.1)
        elif self.loss_streak >= 5:
            self.mood = max(0.5, self.mood - 0.1)

        base_mood = self.mood - self.stress * 0.05 - self.fatigue * 0.02
        self.mood = max(0.3, min(1.5, base_mood * np.random.uniform(0.95, 1.05)))

    def cancel_expired_orders(self, current_time: float):
        """
        Удалить просроченные собственные лимитки (self.orders) и скрытые лимитки.
        """
        expired = [oid for oid, info in self.orders.items()
                   if current_time - info['timestamp'] > self.order_lifetime]
        for oid in expired:
            del self.orders[oid]

        expired_hidden = [hid for hid, info in self.hidden_orders.items()
                          if current_time - info['timestamp'] > self.order_lifetime * 0.5]
        for hid in expired_hidden:
            del self.hidden_orders[hid]

    def should_quote(self) -> bool:
        """
        Стохастический контроль частоты квотирования:
          - минимум 0.1–0.2 сек между квотами
          - 10% шанс пропуска квоты (имитация «засопелости»)
          - если в «кулдауне» (cooldown_ticks > 0), не квотируем
          - если кластерный imbalance слишком против текущей позиции, пропускаем
        """
        now = time.time()
        if self.cooldown_ticks > 0:
            self.cooldown_ticks -= 1
            return False

        if now < self.next_quote_time:
            return False

        # Обновляем время до следующей квоты: 0.1–0.2 сек / mood
        interval = np.random.uniform(0.1, 0.2) / self.mood
        self.next_quote_time = now + interval

        # 10% шанс пропустить квоту
        if random.random() < 0.1:
            return False

        # Если кластерный imbalance сильно против текущей позиции, не квотируем
        if abs(self.cluster_imbalance) > (self.capital * 0.001):
            if (self.position > 0 and self.cluster_imbalance < 0) or \
               (self.position < 0 and self.cluster_imbalance > 0):
                return False

        return True

    def update_trend(self, mid_price: float):
        """
        Обновляем историю mid_price для анализа микротенденций.
        Рассчитываем trend_strength и trend_direction по последним 10 точкам.
        """
        self.mid_price_history.append(mid_price)
        if len(self.mid_price_history) > 10:
            self.mid_price_history.pop(0)

        if len(self.mid_price_history) >= 2:
            diffs = np.diff(self.mid_price_history)
            self.trend_strength = float(np.mean(diffs))
            self.trend_direction = int(np.sign(self.trend_strength))
        else:
            self.trend_strength = 0.0
            self.trend_direction = 0

    def generate_hidden_quote(self, mid_price: float, spread: float, max_size: float):
        """
        Иногда добавляем скрытую лимитку, чтобы создать иллюзию глубины: 
        - цена внутри спрэдового коридора, но не видимая сразу.
        """
        if self.trend_strength == 0 and random.random() < 0.2:
            hid = str(uuid.uuid4())
            # выбираем случайно bid или ask
            if random.random() < 0.5:
                price = mid_price * (1 - spread / 4.0) * np.random.uniform(0.98, 1.02)
                side = 'buy'
            else:
                price = mid_price * (1 + spread / 4.0) * np.random.uniform(0.98, 1.02)
                side = 'sell'
            size = max_size * np.random.uniform(0.1, 0.3)
            self.hidden_orders[hid] = {
                'side': side,
                'price': price,
                'size': size,
                'timestamp': time.time()
            }

    def generate_quotes(self, snapshot: dict, market_context=None) -> list:
        bids = snapshot.get('bids', [])
        asks = snapshot.get('asks', [])
        if not bids or not asks:
            return []

        best_bid = bids[0]['price']
        best_ask = asks[0]['price']
        mid_price = (best_bid + best_ask) / 2.0

        self.update_trend(mid_price)

        bid_vol = sum(level['volume'] for level in bids[:5])
        ask_vol = sum(level['volume'] for level in asks[:5])
        self.cluster_imbalance = bid_vol - ask_vol

        if not self.should_quote():
            return []

        # Реакция на рыночную фазу
        perception = self.perceive_market(market_context)

        # Настройка параметров по восприятию
        spread_scale = 1.0
        size_factor = np.random.uniform(0.8, 1.2)
        skip_quote = False

        if perception == "fear":
            spread_scale = 1.5
            size_factor *= 0.5
        elif perception == "opportunity":
            spread_scale = 0.8
            size_factor *= 1.3
        elif perception == "hesitation":
            if random.random() < 0.5:
                skip_quote = True
        elif perception == "follow":
            size_factor *= 1.2

        if skip_quote:
            return []

        trend_factor = 1 - min(0.5, abs(self.trend_strength) * 10)
        spread_noise = np.random.uniform(0.9, 1.1)
        current_spread = self.spread_target * self.mood * spread_noise * trend_factor * spread_scale

        target_bid_price = mid_price * (1 - current_spread)
        target_ask_price = mid_price * (1 + current_spread)

        max_order_size = self.max_position_size - abs(self.position)
        max_order_size = max(0.0, max_order_size)

        new_orders = []

        has_bid = any(info['side'] == 'buy' for info in self.orders.values())
        has_ask = any(info['side'] == 'sell' for info in self.orders.values())

        # --- BID ---
        if not has_bid and max_order_size > 0:
            base_size = max_order_size * 0.5
            size = base_size * size_factor
            min_size = (self.capital * 0.001) / target_bid_price
            size = max(size, min_size)
            if self.cluster_imbalance < 0:
                size = max(min_size, size * 0.5)

            oid = str(uuid.uuid4())
            self.orders[oid] = {
                'side': 'buy',
                'price': target_bid_price,
                'size': size,
                'timestamp': time.time()
            }
            new_orders.append(Order(
                order_id=oid,
                agent_id=self.agent_id,
                side=OrderSide.BID,
                volume=size,
                price=round(target_bid_price, 5),
                order_type=OrderType.LIMIT,
                ttl=self.order_lifetime
            ))

        # --- ASK ---
        if not has_ask and max_order_size > 0:
            base_size = max_order_size * 0.5
            size = base_size * size_factor
            min_size = (self.capital * 0.001) / target_ask_price
            size = max(size, min_size)
            if self.cluster_imbalance > 0:
                size = max(min_size, size * 0.5)

            oid = str(uuid.uuid4())
            self.orders[oid] = {
                'side': 'sell',
                'price': target_ask_price,
                'size': size,
                'timestamp': time.time()
            }
            new_orders.append(Order(
                order_id=oid,
                agent_id=self.agent_id,
                side=OrderSide.ASK,
                volume=size,
                price=round(target_ask_price, 5),
                order_type=OrderType.LIMIT,
                ttl=self.order_lifetime
            ))

        self.generate_hidden_quote(mid_price, current_spread, max_order_size)

        return new_orders


    def generate_orders(self, order_book, market_context=None) -> list:
        """
        Основной метод: вызывается из agent_loop().
        1) Если в «кулдауне», снимаем ордера и пропускаем.
        2) Удаляем просроченные лимитки.
        3) Обновляем настроение.
        4) Формируем квоты через generate_quotes().
        """
        # 1) Кулдаун после серии убытков
        if self.cooldown_ticks > 0:
            self.cooldown_ticks -= 1
            self.orders.clear()
            return []

        snapshot = order_book.get_order_book_snapshot(depth=1)
        current_time = time.time()

        # 2) Отменяем устаревшие и скрытые ордера
        self.cancel_expired_orders(current_time)

        # 3) Обновляем психологию
        self.update_mood()

        # 4) Генерируем новые «квоты»
        return self.generate_quotes(snapshot, market_context)

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


    def on_order_filled(self, order_id: str, fill_price: float, fill_size: float, side: OrderSide):
        """
        Callback из match_and_update_loop(): наша лимитка исполнилась.
        Обновляем позицию, капитал, PnL, психологию и историю.
        """
        if order_id not in self.orders:
            return
        info = self.orders.pop(order_id)

        pnl = 0.0
        # Если не было позиции, открываем новую
        if self.position == 0:
            self.entry_price = fill_price
            self.position = fill_size if side == OrderSide.BID else -fill_size
        else:
            # Если закрываем часть позиции
            if (self.position > 0 and side == OrderSide.ASK) or (self.position < 0 and side == OrderSide.BID):
                closing_size = min(abs(self.position), fill_size)
                pnl = closing_size * (fill_price - self.entry_price) * np.sign(self.position)
                self.position += fill_size if side == OrderSide.BID else -fill_size
                if abs(self.position) < 1e-6:
                    self.position = 0.0
                    self.entry_price = None
            else:
                # Усреднение в ту же сторону
                total_pos = abs(self.position) + fill_size
                self.entry_price = (self.entry_price * abs(self.position) + fill_price * fill_size) / total_pos
                self.position += fill_size if side == OrderSide.BID else -fill_size

        # Корректируем капитал с учётом комиссии/проскальзывания
        cost = fill_price * fill_size * (1 + self.commission_rate + self.slippage_rate)
        if side == OrderSide.BID:
            self.capital -= cost
        else:
            self.capital += fill_price * fill_size * (1 - self.commission_rate - self.slippage_rate)

        # Обновляем эмоции и стрики
        self.last_pnl = float(pnl)
        self.pnl_history.append(pnl)
        if pnl > 0:
            self.win_streak += 1
            self.loss_streak = 0
            self.stress = max(0.0, self.stress - 0.05)
        else:
            self.loss_streak += 1
            self.win_streak = 0
            self.stress += 0.1
            if self.loss_streak >= 3:
                # «Кулдаун» 3–5 тик/итераций
                self.cooldown_ticks = random.randint(3, 5)
        self.fatigue += 1

        # Случайно меняем order_lifetime ±1 тик с шансом 10%
        if random.random() < 0.1:
            self.order_lifetime = max(2, self.order_lifetime + random.choice([-1, 1]))

        # Логируем сделку
        self.trade_history.append({
            'order_id': order_id,
            'price': float(fill_price),
            'size': float(fill_size),
            'side': side.value,
            'pnl': pnl,
            'capital': self.capital,
            'position': self.position,
            'timestamp': time.time()
        })


    def perceive_market(self, context):
        if context is None or not context.is_ready():
            return "neutral"

        perceived_phase = context.phase.name

        if random.random() < 0.1:
            perceived_phase = random.choice(["calm", "volatile", "panic", "trend_up", "trend_down"])

        if perceived_phase == "panic":
            return "fear"
        elif perceived_phase == "volatile":
            return "hesitation"
        elif perceived_phase in ["trend_up", "trend_down"]:
            return "follow"
        else:
            return "neutral"

