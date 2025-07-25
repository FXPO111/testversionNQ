import heapq
import time
from collections import deque
from typing import Optional, List, Dict, Any
import logging
from order import Order, OrderSide, OrderType
from trade_logger import log as tlog

# Настройка логирования
logging.basicConfig(
    filename='matching_engine.log',  # Лог файл
    level=logging.DEBUG,  # Уровень логирования
    format='%(asctime)s - %(levelname)s - %(message)s',  # Формат логов
)

class PriceLevel:
    def __init__(self, price: float):
        self.price = price
        self.orders: deque[Order] = deque()

    def add_order(self, order: Order):
        logging.debug(f"[DEBUG] add_order called with {order.order_type.name} for Order {order.order_id}")
        self.orders.append(order)

    def remove_filled_or_inactive_orders(self):
        while self.orders and not self.orders[0].is_active():
            removed_order = self.orders.popleft()
            logging.debug(f"[DEBUG] Removing inactive order {removed_order.order_id}")

    def total_volume(self) -> float:
        return sum(order.remaining_volume() for order in self.orders if order.is_active())

    def __bool__(self):
        self.remove_filled_or_inactive_orders()
        return len(self.orders) > 0

    def __repr__(self):
        return f"PriceLevel(price={self.price}, orders_count={len(self.orders)})"

class MatchingEngine:
    def __init__(self):
        self.bids: Dict[float, PriceLevel] = {}
        self.asks: Dict[float, PriceLevel] = {}
        self.bid_prices: List[float] = []  # max-heap via negative prices
        self.ask_prices: List[float] = []  # min-heap
        self.orders: Dict[str, Order] = {}
        self.last_trade_price: Optional[float] = None
        self.trade_history: deque[Dict[str, Any]] = deque(maxlen=1000)

    def _add_price_level(self, side: OrderSide, price: float):
        book = self.bids if side == OrderSide.BID else self.asks
        prices_heap = self.bid_prices if side == OrderSide.BID else self.ask_prices
        if price not in book:
            book[price] = PriceLevel(price)
            if side == OrderSide.BID:
                heapq.heappush(prices_heap, -price)
            else:
                heapq.heappush(prices_heap, price)
            logging.info(f"[INFO] Added price level {price} for {side.name}")

    def _remove_price_level_if_empty(self, side: OrderSide, price: float):
        book = self.bids if side == OrderSide.BID else self.asks
        prices_heap = self.bid_prices if side == OrderSide.BID else self.ask_prices
        level = book.get(price)
        if level and not level:
            del book[price]
            if side == OrderSide.BID:
                prices_heap.remove(-price)
            else:
                prices_heap.remove(price)
            heapq.heapify(prices_heap)
            logging.info(f"[INFO] Removed empty price level {price} for {side.name}")

    def _best_bid_price(self) -> Optional[float]:
        while self.bid_prices:
            price = -self.bid_prices[0]
            if price in self.bids and self.bids[price]:
                return price
            heapq.heappop(self.bid_prices)
        return None

    def _best_ask_price(self) -> Optional[float]:
        while self.ask_prices:
            price = self.ask_prices[0]
            if price in self.asks and self.asks[price]:
                return price
            heapq.heappop(self.ask_prices)
        return None

    def add_order(self, order: Order):
        trades = self.order_book.add_order(order)
        if trades:
            logging.info(f"[MATCHED] Order {order.order_id} executed: {len(trades)} trades")
            for trade in trades:
                self.notify_agent_fill(trade)
        else:
            logging.info(f"[BOOK] Order {order.order_id} placed in book")
            for trade in trades:
                self.notify_agent_fill(trade)

    def _match_market_order(self, market_order: Order):
        logging.debug(f"[DEBUG] _match_market_order called for {market_order.order_id} (side={market_order.side.name}, qty={market_order.volume})")

        trades = []
        volume_to_match = market_order.volume
        side = market_order.side
        opposite_book = self.asks if side == OrderSide.BID else self.bids

        while volume_to_match > 0 and opposite_book:
            logging.debug(f"[MATCH_MKT_ORDER] Remaining to match: {volume_to_match}")
            best_prices = sorted(opposite_book.keys())
            if side == OrderSide.BID:
                live_prices = sorted(p for p in best_prices if p in self.asks and self.asks[p] and self.asks[p].total_volume() > 0)
                if not live_prices:
                    logging.info("[MATCH_MKT_ORDER] No valid price levels to match against.")
                    break
                best_price = min(live_prices)
                logging.debug(f"[MATCH_MKT_ORDER] Best price to match against: {best_price}")
            else:
                live_prices = sorted(p for p in best_prices if p in self.bids and self.bids[p] and self.bids[p].total_volume() > 0)
                if not live_prices:
                    break
                best_price = max(live_prices)

            level = opposite_book.get(best_price)
            if level is None:
                logging.error(f"[MATCH_MKT_ORDER] Error: best price {best_price} not found in book")
                break

            level = opposite_book[best_price]
            level.remove_filled_or_inactive_orders()

            if level.total_volume() <= 0:
                self._remove_price_level_if_empty(
                    OrderSide.ASK if side == OrderSide.BID else OrderSide.BID,
                    best_price
                )
                continue

            for order in list(level.orders):
                if not order.is_active():
                    logging.debug(f"[MATCH_MKT_ORDER] Skipping inactive order {order.order_id}")
                    continue

                match_volume = min(volume_to_match, order.remaining_volume())
                trade_price = order.price

                order.fill(match_volume)
                market_order.fill(match_volume)
                volume_to_match -= match_volume

                logging.info(f"[TRADE_EXECUTED] Price={trade_price}, Volume={match_volume}, "
                             f"Buy={market_order.order_id if side == OrderSide.BID else order.order_id}, "
                             f"Sell={order.order_id if side == OrderSide.BID else market_order.order_id}")

                trades.append({
                    'price': trade_price,
                    'volume': match_volume,
                    'buy_order_id': market_order.order_id if side == OrderSide.BID else order.order_id,
                    'sell_order_id': order.order_id if side == OrderSide.BID else market_order.order_id
                })

                if volume_to_match <= 0:
                    break

            level.remove_filled_or_inactive_orders()
            if level.total_volume() <= 0:
                self._remove_price_level_if_empty(
                    OrderSide.ASK if side == OrderSide.BID else OrderSide.BID,
                    best_price
                )

        if market_order.remaining_volume() > 0:
            logging.warning(f"[UNFILLED_MARKET_ORDER] Market order {market_order.order_id} unfilled qty: {market_order.remaining_volume()}")

        return trades

    def match(self) -> List[Dict[str, Any]]:
        trades = []
        while True:
            best_bid = self._best_bid_price()
            best_ask = self._best_ask_price()
            if best_bid is None or best_ask is None or best_bid < best_ask:
                break
            bid_level = self.bids[best_bid]
            ask_level = self.asks[best_ask]
            bid_order = bid_level.orders[0]
            ask_order = ask_level.orders[0]
            trade_price = ask_order.price if ask_order.timestamp < bid_order.timestamp else bid_order.price
            trade_qty = min(bid_order.remaining_volume(), ask_order.remaining_volume())
            bid_order.fill(trade_qty)
            ask_order.fill(trade_qty)

            # Логирование трейда
            logging.info(f"[TRADE_EXECUTED] Price={trade_price}, Volume={trade_qty}, "
                         f"Buy={bid_order.order_id}, Sell={ask_order.order_id}")

            self.last_trade_price = trade_price
            timestamp = time.time()
            trade = {
                'price': trade_price,
                'volume': trade_qty,
                'buy_order_id': bid_order.order_id,
                'sell_order_id': ask_order.order_id,
                'buy_agent': bid_order.agent_id,
                'sell_agent': ask_order.agent_id,
                'timestamp': timestamp
            }
            trades.append(trade)
            self.trade_history.append(trade)
            if not bid_order.is_active():
                bid_level.remove_filled_or_inactive_orders()
                self._remove_price_level_if_empty(OrderSide.BID, best_bid)
            if not ask_order.is_active():
                ask_level.remove_filled_or_inactive_orders()
                self._remove_price_level_if_empty(OrderSide.ASK, best_ask)
        return trades

    def cancel_order(self, order_id: str):
        order = self.orders.get(order_id)
        if order and order.is_active():
            order.cancel()
            logging.info(f"[CANCEL_ORDER] Order {order_id} cancelled")

    def tick(self):
        # 1. Обновляем TTL и отменяем истёкшие
        to_cancel = []
        for order in self.orders.values():
            if order.ttl is not None and order.is_active():
                order.tick_ttl()
                if order.ttl == 0:
                    to_cancel.append(order.order_id)
        for oid in to_cancel:
            self.cancel_order(oid)

        # 2. Чистим неактивные ордера
        for oid in list(self.orders):
            if not self.orders[oid].is_active():
                del self.orders[oid]

        # 3. Чистим пустые уровни
        for price in list(self.bids.keys()):
            self._remove_price_level_if_empty(OrderSide.BID, price)
        for price in list(self.asks.keys()):
            self._remove_price_level_if_empty(OrderSide.ASK, price)


    def get_order_book_snapshot(self, depth: int = 10) -> dict:
        bids = []
        asks = []
        for price in sorted(self.bids.keys(), reverse=True)[:depth]:
            lvl = self.bids[price]
            vol = lvl.total_volume()
            if vol > 0:
                bids.append({'price': price, 'volume': vol})
        for price in sorted(self.asks.keys())[:depth]:
            lvl = self.asks[price]
            vol = lvl.total_volume()
            if vol > 0:
                asks.append({'price': price, 'volume': vol})
        return {'bids': bids, 'asks': asks}

