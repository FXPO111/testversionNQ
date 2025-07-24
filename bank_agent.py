# ── bank_agent.py  ────────────────────────────────

import uuid
import random
import numpy as np

import logging

from order import Order, OrderSide, OrderType
from datetime import datetime


# === Постоянный клиентский поток ===
class ClientFlowSimulator:
    def __init__(self, uid):
        self.uid = uid
        self.position = 0.0
        self.capital = 20_000_000

    def get_session_lambda(self):
        hour = datetime.utcnow().hour + datetime.utcnow().minute / 60.0
        if 8 <= hour < 11:
            return 0.35  # активная сессия
        elif 11 <= hour < 12:
            return 0.25
        elif 14.5 <= hour < 22:
            return 0.5   # пик США
        elif 22 <= hour or hour < 6:
            return 0.5  # почти тишина
        else:
            return 0.20

    def step(self, order_book, market_context=None):
        best_bid = order_book._best_bid_price()
        best_ask = order_book._best_ask_price()
        if best_bid is None or best_ask is None:
            return []

        λ = self.get_session_lambda()
        if random.random() > λ:
            return []

        side = OrderSide.BID if random.random() < 0.5 else OrderSide.ASK
        volume = np.random.pareto(1.5) * 25000 * random.uniform(0.5, 1.5)
        volume = max(1000, min(volume, 10_500_000))

        order = Order(
            order_id=str(uuid.uuid4()),
            agent_id=self.uid,
            side=side,
            order_type=OrderType.MARKET,
            volume=volume,
            price=None
        )
        return [order]

    def on_order_filled(self, order_id, fill_price, fill_size, side):
        pnl = -fill_price * fill_size if side == OrderSide.BID else fill_price * fill_size
        self.capital += pnl
        self.position += fill_size if side == OrderSide.BID else -fill_size

        print(f"[{self.uid}] FILLED {side.name} {fill_size:.0f}@{fill_price:.5f}, PnL={pnl:.2f}, Capital={self.capital:.2f}")

        

# === Умный CoreUnit с фазовым выходом маркетами ===
class CoreUnit:
    def __init__(self, uid, brain):
        self.uid = uid
        self.brain = brain
        self.state = "idle"
        self.position = 0.0
        self.capital = 300_000_000  # начальный капитал
        self.entry_price = None
        self.pnl = 0.0
        self.entry_tick = 0
        self.cooldown = 0
        self.hold_duration = 25
        self.max_position = 250_000

    def step(self, order_book, ctx):
        orders = []
        if self.cooldown > 0:
            self.cooldown -= 1
            return orders

        try:
            best_bid = order_book.best_bid()
            best_ask = order_book.best_ask()
        except AttributeError:
            best_bid = order_book.get_best_bid() if hasattr(order_book, 'get_best_bid') else None
            best_ask = order_book.get_best_ask() if hasattr(order_book, 'get_best_ask') else None

        mid = (best_bid + best_ask) / 2 if best_bid and best_ask else None
        if not mid:
            return orders

        fair_value = ctx.fair_value if hasattr(ctx, "fair_value") else mid
        phase = ctx.phase.name if hasattr(ctx, "phase") else "undefined"
        tick = ctx.tick if hasattr(ctx, "tick") else 0
        spread = best_ask - best_bid if best_ask and best_bid else 0.01

        if self.state == "idle":
            near_fv = abs(mid - fair_value) < spread * 2
            pressure = ctx.order_flow_pressure() if hasattr(ctx, "order_flow_pressure") else 0.0
            if near_fv and phase in ["balanced", "compression", "accumulation"]:
                side = OrderSide.BID if pressure > 0 else OrderSide.ASK
                vol = min(self.max_position, 100_000 + int(abs(pressure) * 100_000))
                price = best_bid if side == OrderSide.BID else best_ask
                orders.append(Order(self.uid, side, OrderType.LIMIT, vol, price))
                self.entry_price = price
                self.position = vol if side == OrderSide.BID else -vol
                self.entry_tick = tick
                self.state = "holding"

        elif self.state == "holding":
            pnl_unreal = (mid - self.entry_price) * self.position
            duration = tick - self.entry_tick

            phase_breakout = phase in ["breakout", "runaway", "momentum"]
            sustained = duration > self.hold_duration
            pressure = ctx.order_flow_pressure() if hasattr(ctx, "order_flow_pressure") else 0.0
            breakout_condition = abs(pressure) > 0.8 or phase_breakout

            if sustained and breakout_condition:
                side = OrderSide.ASK if self.position > 0 else OrderSide.BID
                vol = abs(self.position)
                orders.append(Order(self.uid, side, OrderType.MARKET, vol))
                self.state = "cooldown"
                self.cooldown = 10
                self.position = 0.0
                self.entry_price = None

        elif self.state == "cooldown":
            self.cooldown -= 1
            if self.cooldown <= 0:
                self.state = "idle"

        
        return orders

    def on_order_filled(self, order_id: str, fill_price: float, fill_size: float, side: OrderSide):
        if side == OrderSide.BID:
            self.position += fill_size
            pnl = -fill_price * fill_size
        else:
            self.position -= fill_size
            pnl = fill_price * fill_size

        self.capital += pnl

        
class AdvancedCoreUnit:
    def __init__(self, uid, brain):
        self.uid = uid
        self.brain = brain

        self.state = "idle"
        self.side = None
        self.position = 0.0
        self.entry_price = None
        self.built_volume = 0.0
        self.entry_tick = 0
        
        self.capital = 500_000_000

        self.target_volume = 0
        self.limit_levels = []  # [(price, volume)]
        self.price_push_attempted = False

        self.stop_loss = None
        self.take_profit = None

        self.cooldown = 0
        self.max_position = 50_000_000
        self.min_position = 10_000_000
        self.level_count = 3
        self.level_spacing = 0.003  # расстояние между уровнями (0.3%)

        self.hold_ticks = 30
        self.exit_triggered = False
        self.defensive_mode = False

    def step(self, order_book, ctx):
        orders = []
        if self.cooldown > 0:
            self.cooldown -= 1
            return orders

        try:
            best_bid = order_book.best_bid()
            best_ask = order_book.best_ask()
        except:
            return orders

        mid = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        fair_value = getattr(ctx, "fair_value", mid)
        pressure = getattr(ctx, "order_flow_pressure", lambda: 0.0)()
        tick = getattr(ctx, "tick", 0)
        phase = getattr(ctx, "phase", None)
        phase_name = phase.name if phase else "undefined"

        # =================== ВХОД ======================
        if self.state == "idle":
            if phase_name not in ["accumulation", "compression", "balanced"]:
                return orders
            if abs(mid - fair_value) > spread * 2:
                return orders
            if random.random() > 0.6:
                return orders

            self.side = OrderSide.BID if pressure > 0 else OrderSide.ASK
            base_price = best_bid if self.side == OrderSide.BID else best_ask
            self.target_volume = random.randint(self.min_position, self.max_position)
            volume_per_level = self.target_volume / self.level_count

            # Каскад лимиток
            self.limit_levels = []
            for i in range(self.level_count):
                offset = i * self.level_spacing
                price = base_price - offset if self.side == OrderSide.BID else base_price + offset
                self.limit_levels.append((price, volume_per_level))

            self.built_volume = 0.0
            self.entry_tick = tick
            self.entry_price = None
            self.price_push_attempted = False
            self.state = "building"
            print(f"[{self.uid}] 🔵 ENTER {self.side.name} | volume={self.target_volume:,} | phase={phase_name} | mid={mid:.5f}")

            return [Order(self.uid, self.side, OrderType.LIMIT, vol, pr, metadata={"source": "advcore"}) for pr, vol in self.limit_levels]

        # =================== СБОР ПОЗИЦИИ ======================
        elif self.state == "building":
            filled = abs(self.position)
            if filled >= self.min_position:
                self.entry_price = mid
                rr = 2.0
                delta = spread * rr
                self.take_profit = mid + delta if self.side == OrderSide.BID else mid - delta
                self.stop_loss = mid - delta / rr if self.side == OrderSide.BID else mid + delta / rr
                self.state = "holding"
                self.entry_tick = tick
            elif not self.price_push_attempted and random.random() < 0.5:
                # Пробуем опустить цену к лимиткам
                side_push = OrderSide.ASK if self.side == OrderSide.BID else OrderSide.BID
                push_vol = min(1_050_000, self.target_volume * 0.1)
                orders.append(Order(self.uid, exit_side, OrderType.MARKET, guard_vol, metadata={"source": "advcore"}))

                self.price_push_attempted = True

        # =================== УДЕРЖАНИЕ ======================
        elif self.state == "holding":
            duration = tick - self.entry_tick
            unreal_pnl = (mid - self.entry_price) * self.position

            if not self.exit_triggered:
                # Тейк
                if (self.side == OrderSide.BID and mid >= self.take_profit) or \
                   (self.side == OrderSide.ASK and mid <= self.take_profit):
                    self.exit_triggered = True

                # Стоп
                elif (self.side == OrderSide.BID and mid <= self.stop_loss) or \
                     (self.side == OrderSide.ASK and mid >= self.stop_loss):
                    self.defensive_mode = True
                    self.exit_triggered = True

                # Слишком долго сидим
                elif duration > 3 * self.hold_ticks:
                    self.exit_triggered = True

            if self.exit_triggered:
                exit_side = OrderSide.ASK if self.position > 0 else OrderSide.BID
                exit_vol = abs(self.position)
                orders.append(Order(self.uid, exit_side, OrderType.MARKET, exit_vol, metadata={"source": "advcore"}))

                exit_type = "DEFENSIVE" if self.defensive_mode else "NORMAL"
                print(f"[{self.uid}] 🔴 EXIT {exit_type} | pos={exit_vol:.0f} | mid={mid:.5f}")

                self.state = "cooldown"
                self.cooldown = 20
                self.position = 0.0
                self.built_volume = 0.0
                self.exit_triggered = False
                self.defensive_mode = False
                self.stop_loss = None
                self.take_profit = None
                return orders
            
                
            # Дополнительная защита (если PnL в минусе)
            if unreal_pnl < -1_000_000 and not self.defensive_mode:
                guard_vol = self.target_volume * 0.2
                exit_side = OrderSide.ASK if self.position > 0 else OrderSide.BID
                orders.append(Order(self.uid, side_push, OrderType.MARKET, push_vol, metadata={"source": "advcore"}))

                self.defensive_mode = True

    
        return orders

    def on_order_filled(self, order_id: str, fill_price: float, fill_size: float, side: OrderSide):
        if side == OrderSide.BID:
            self.position += fill_size
            pnl = -fill_price * fill_size
        else:
            self.position -= fill_size
            pnl = fill_price * fill_size

        self.capital += pnl


class ReactiveFlowSimulator:
    def __init__(self, uid, style="breakout", sensitivity=0.002, delay_ticks=3):
        self.uid = uid
        self.style = style  # breakout / retest / fomo / fade
        self.prices = []
        self.lookback = 400
        self.delay_ticks = delay_ticks
        self.last_action_tick = -150
        self.active_level = None
        self.activation_tick = None
        self.activated = False
        self.sensitivity = sensitivity  # чувствительность к пробою
        self.cooldown = 15
        self.max_volume = 15_000_000 
        self.min_volume = 100_000

    def step(self, order_book, ctx):
        orders = []
        tick = getattr(ctx, "tick", 0)
        if tick - self.last_action_tick < self.cooldown:
            return orders
        
        mid = (order_book._best_bid_price() + order_book._best_ask_price()) / 2
        self.prices.append(mid)
        if len(self.prices) > self.lookback:
            self.prices.pop(0)

        if not self.activated:
            local_high = max(self.prices)
            local_low = min(self.prices)

            if self.style == "breakout":
                if mid > local_high * (1 + self.sensitivity):
                    self.active_level = local_high
                    self.activation_tick = tick
                    self.activated = True
                elif mid < local_low * (1 - self.sensitivity):
                    self.active_level = local_low
                    self.activation_tick = tick
                    self.activated = True

        else:
            # Ждём подтверждение в течение delay_ticks
            if tick - self.activation_tick >= self.delay_ticks:
                breakout_up = mid > self.active_level * (1 + self.sensitivity)
                breakout_down = mid < self.active_level * (1 - self.sensitivity)

                if breakout_up or breakout_down:
                    side = OrderSide.bid if breakout_up else OrderSide.ask
                    vol = int(random.triangular(self.min_volume, self.max_volume, self.max_volume * 0.3))

                    orders.append(Order(
                        order_id=str(uuid.uuid4()),
                        agent_id=self.uid,
                        side=side,
                        order_type=OrderType.MARKET,
                        volume=vol,
                        price=None
                    ))

                    print(f"[REACTIVE-{self.uid}] 🧠 {side.name} breakout @ {mid:.5f} vol={vol}")
                    self.last_action_tick = tick

                self.activated = False
                self.active_level = None

        return orders


# === Менеджер Банка ===
class BankAgentManager:
    def __init__(self, agent_id, total_capital):
        self.agent_id = agent_id
        self.total_capital = total_capital
        self.brain = type("BrainStub", (), {"mode": "normal"})()

        self.clients = [ClientFlowSimulator(f"{agent_id}_client_{i}") for i in range(10)]
        self.cores = [CoreUnit(f"{agent_id}_core_{i}", self.brain) for i in range(5)]
        self.advanced_cores = [AdvancedCoreUnit(f"{agent_id}_advcore_{i}", self.brain) for i in range(3)]
        self.reactive_flows = [ReactiveFlowSimulator(f"{agent_id}_reactive_{i}", style=random.choice(["breakout"]))
            for i in range(5)
]


    def generate_orders(self, order_book, market_context=None):
        orders = []

        for client in self.clients:
            client_orders = client.step(order_book, market_context)
            for o in client_orders:
                price_str = f"{o.price:.5f}" if o.price is not None else "MKT"
                print(f"[CLIENT] {o.side.name} {o.volume:.0f} @ {price_str} [{o.order_type.name}]")
            orders.extend(client_orders)


        for core in self.cores:
            core_orders = core.step(order_book, market_context)
            for o in core_orders:
                print(f"[CORE] {o.side.name} {o.volume:.0f} @ {o.price:.5f} [{o.order_type.name}]")
            orders.extend(core_orders)

        for adv in self.advanced_cores:
            orders.extend(adv.step(order_book, market_context))

        for reactive in self.reactive_flows:
            reactive_orders = reactive.step(order_book, market_context)
            for o in reactive_orders:
                price_str = f"{o.price:.5f}" if o.price is not None else "MKT"
                print(f"[REACTIVE] {o.side.name} {o.volume:.0f} @ {price_str} [{o.order_type.name}]")
            orders.extend(reactive_orders)


        return orders



#───────────────────────────────────────────────────────────────────────────────────
