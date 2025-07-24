import uuid
import time
import numpy as np
from order import Order, OrderSide, OrderType

class ImpulsiveAggressorAgent:
    def __init__(self, agent_id: str, capital: float, restore_threshold: float = 0.6):
        self.agent_id = agent_id
        self.capital = float(capital)
        self.position = 0.0
        self.entry_price = None
        self.pnl = 0.0
        self.trade_history = []
        self.restore_threshold = restore_threshold

        # Психология
        self.stress = 0.0
        self.euphoria = 0.0
        self.last_action_time = 0.0
        self.cooldown_range = (1.5, 4.0)

        # Для тренда и волатильности
        self.mid_prices = []
        self.volatility = 0.0
        self.trend_strength = 0.0
        self.trend_direction = 0

        # Анти-флуд: запоминаем последнюю сторону агрессии
        self.last_side = None
        self.avoid_repeat_ticks = 0

        # Для боковика
        self.sideways_duration = 0  # время боковика (когда рынок не двигается)

    def update_state(self, snapshot):
        bids = snapshot.get("bids", [])
        asks = snapshot.get("asks", [])

        if not bids or not asks:
            return

        best_bid = bids[0]["price"]
        best_ask = asks[0]["price"]
        mid = (best_bid + best_ask) / 2.0
        self.mid_prices.append(mid)

        if len(self.mid_prices) > 30:
            self.mid_prices.pop(0)

        if len(self.mid_prices) >= 10:
            diffs = np.diff(self.mid_prices)
            self.volatility = float(np.std(diffs))
            self.trend_strength = float(np.mean(diffs[-5:]))
            self.trend_direction = int(np.sign(self.trend_strength))

    def compute_imbalance(self, snapshot):
        bids = snapshot.get("bids", [])
        asks = snapshot.get("asks", [])
        bid_vol = sum([lvl["volume"] for lvl in bids[:5]])
        ask_vol = sum([lvl["volume"] for lvl in asks[:5]])
        return bid_vol - ask_vol, bid_vol, ask_vol

    def should_act(self, imbalance, trend_dir):
        base_prob = 0.5

        # Усилим поведение если есть тренд
        if trend_dir == 1 and imbalance > 10:
            base_prob += 0.2
        elif trend_dir == -1 and imbalance < -10:
            base_prob += 0.2

        # Эйфория усиливает агрессию
        base_prob += self.euphoria * 0.2
        base_prob -= self.stress * 0.2

        return np.random.rand() < base_prob

    def generate_orders(self, order_book, market_context=None):

        self._market_context = market_context
        perception = self.perceive_market(market_context) if hasattr(self, "_market_context") else "neutral"

        if perception == "fear" and np.random.rand() < 0.8:
            return []

        if perception == "boredom" and np.random.rand() < 0.5:
            return []

        if perception == "opportunity":
            self.euphoria = min(1.0, self.euphoria + 0.1)

        if perception == "hesitation":
            self.stress = min(1.0, self.stress + 0.05)

        now = time.time()
        if now - self.last_action_time < np.random.uniform(*self.cooldown_range):
            return []
        if self.avoid_repeat_ticks > 0:
            self.avoid_repeat_ticks -= 1
            return []

        snapshot = order_book.get_order_book_snapshot(depth=5)
        if not snapshot.get("bids") or not snapshot.get("asks"):
            return []

        self.update_state(snapshot)
        imbalance, bid_vol, ask_vol = self.compute_imbalance(snapshot)
        mid = (snapshot['bids'][0]['price'] + snapshot['asks'][0]['price']) / 2.0

        # Проверка на боковик (застой)
        if self.sideways_duration >= 50:  # если застой длится больше 50 тиков
            self.place_large_order(snapshot, mid)

        # Определяем сторону агрессии
        side = None
        if imbalance > 15 and self.trend_direction >= 0:
            side = OrderSide.BID
        elif imbalance < -15 and self.trend_direction <= 0:
            side = OrderSide.ASK
        else:
            return []

        if side == self.last_side:
            self.avoid_repeat_ticks = 3
            return []

        if not self.should_act(imbalance, self.trend_direction):
            return []

        # Размер сделки — адаптивный, зависит от эмоций и волатильности
        risk_unit = self.capital * np.random.uniform(0.004, 0.01)  # 0.4–1%
        size = risk_unit / (mid + self.volatility * 10 + 1e-6)
        size = max(1.0, round(size))

        order = Order(
            order_id=str(uuid.uuid4()),
            agent_id=self.agent_id,
            side=side,
            volume=size,
            price=None,
            order_type=OrderType.MARKET,
            ttl=None
        )

        self.last_action_time = now
        self.last_side = side
        return [order]

    def place_large_order(self, snapshot, mid):
        """ Размещение крупного ордера для выхода из боковика """
        # С вероятностью 50% размещаем крупный ордер
        if np.random.rand() < 0.5:
            # Определяем направление ордера
            direction = OrderSide.BID if np.random.rand() < 0.5 else OrderSide.ASK
            # Размер ордера пропорционален капиталу агента
            size = self.capital * 0.05 / (mid + self.volatility * 10 + 1e-6)  # более крупный ордер
            size = max(1.0, round(size))
            
            # Создаем ордер
            order = Order(
                order_id=str(uuid.uuid4()),
                agent_id=self.agent_id,
                side=direction,
                volume=size,
                price=None,
                order_type=OrderType.MARKET,
                ttl=None
            )
            
            # Выполнение ордера (имитация исполнения)
            self.execute_order(order)
            print(f"[{self.agent_id}] placed large order to break out: {order}")

    def execute_order(self, order):
        """ Исполнение ордера (псевдокод) """
        self.trade_history.append(order)
        self.position += order.volume if order.side == OrderSide.BID else -order.volume
        self.entry_price = order.price  # Предположим, что цена исполнилась по рыночной цене

    def perceive_market(self, market_context) -> str:
        if not market_context or not market_context.is_ready():
            return "neutral"

        perceived_phase = market_context.phase.name
        if np.random.rand() < 0.15:
            perceived_phase = random.choice(["panic", "volatile", "trend_up", "trend_down", "calm"])

        if perceived_phase == "panic":
            return "fear" if self.stress > 0.3 else "confusion"
        elif perceived_phase == "volatile":
            return "opportunity" if self.euphoria > 0.3 else "hesitation"
        elif perceived_phase in ["trend_up", "trend_down"]:
            return "follow"
        elif perceived_phase == "calm":
            return "boredom"
        return "neutral"

