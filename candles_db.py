import sqlite3
import os

DB_FILE = 'candles.db'

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS candles (
            interval TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL,
            PRIMARY KEY (interval, timestamp)
        )
    ''')
    conn.commit()
    return conn


def save_candle(conn, interval, candle):
    cur = conn.cursor()
    cur.execute('''
        INSERT OR REPLACE INTO candles (interval, timestamp, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        interval,
        candle['timestamp'],
        candle['open'],
        candle['high'],
        candle['low'],
        candle['close'],
        candle['volume']
    ))
    conn.commit()


def fetch_candles(conn, interval, from_ts, to_ts):
    cur = conn.cursor()
    cur.execute('''
        SELECT timestamp, open, high, low, close, volume
        FROM candles
        WHERE interval = ? AND timestamp BETWEEN ? AND ?
        ORDER BY timestamp ASC
    ''', (interval, from_ts, to_ts))
    rows = cur.fetchall()
    return [
        {
            'timestamp': row[0],
            'open': row[1],
            'high': row[2],
            'low': row[3],
            'close': row[4],
            'volume': row[5],
        }
        for row in rows
    ]
