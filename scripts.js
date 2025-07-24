// ==== Инициализация графика (LightweightCharts) ====

const chartContainer = document.getElementById('price-chart');
const bidsTbody = document.querySelector('#bids-table tbody');
const asksTbody = document.querySelector('#asks-table tbody');
const tradesList = document.getElementById('trades-list');

// === фазы ===
let showPhases = false;
let marketPhases = [];

// Кнопка фаз
document.getElementById('toggle-phase-btn').addEventListener('click', function() {
    showPhases = !showPhases;
    this.textContent = showPhases ? 'Скрыть фазы рынка' : 'Показать фазы рынка';
    renderPhasesOverlay();
});

const chart = LightweightCharts.createChart(chartContainer, {
    width: chartContainer.clientWidth,
    height: chartContainer.clientHeight,
    layout: {
        backgroundColor: '#0e1117',
        textColor: '#c9d1d9',
        fontFamily: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
    },
    grid: {
        vertLines: { color: '#222' },
        horzLines: { color: '#222' },
    },
    crosshair: {
        mode: LightweightCharts.CrosshairMode.Normal,
        vertLine: { visible: false },
        horzLine: { visible: false },
    },
    timeScale: {
        borderColor: '#30363d',
        visible: true,
        rightOffset: 5,
        barSpacing: 12,  // Начальная ширина свечей
        minBarSpacing: 12,  // Минимальное расстояние между свечами
        fixLeftEdge: false,  // Отключаем фиксированную левую сторону
        fixRightEdge: false,  // Отключаем фиксированную правую сторону
        lockVisibleTimeRangeOnResize: true,
        rightBarStaysOnScroll: false, // Чтобы бар не оставался в одном месте при прокрутке
    },
    priceScale: {
        borderColor: '#30363d',
    },
});

// Устанавливаем отрисовку свечей
const candleSeries = chart.addCandlestickSeries({
    upColor: '#3fb950',
    downColor: '#f85149',
    borderVisible: false,
    wickVisible: true,
    borderColor: '#3fb950',
    wickColor: (candle) => {
        if (candle.close < candle.open) {
            return '#f85149';
        } else if (candle.close > candle.open) {
            return '#3fb950';
        }
        return '#3fb950';
    }
});

// ==== ФАЗЫ РЫНКА: загрузка с сервера и overlay ====

// Загрузка фаз с сервера
function fetchPhases() {
    fetch('/market_phases')
        .then(res => res.json())
        .then(data => {
            marketPhases = data;
            renderPhasesOverlay();
        });
}
// Запускаем обновление фаз
fetchPhases();
setInterval(fetchPhases, 3000);

function renderPhasesOverlay() {
    if (!showPhases || !marketPhases.length) return;

    const paneWidget = chart._internal__paneWidgets?.[0];
    if (!paneWidget || !paneWidget._internal__canvasBinding?._canvas) return;

    const overlayTarget = paneWidget._internal__canvasBinding._canvas.parentElement;

    overlayTarget.querySelectorAll('.phase-rect').forEach(el => el.remove());

    const chartHeight = overlayTarget.clientHeight;

    marketPhases.forEach(phase => {
        const x1 = chart.timeScale().timeToCoordinate(phase.start);
        const x2 = chart.timeScale().timeToCoordinate(phase.end);
        if (x1 == null || x2 == null) return;

        const left = Math.min(x1, x2);
        const width = Math.abs(x2 - x1);
        const color = getPhaseColor(phase.phase, phase.microphase);

        const div = document.createElement('div');
        div.className = 'phase-rect';
        div.style.cssText = `
            position: absolute;
            left: ${left}px;
            width: ${width}px;
            top: 0;
            height: ${chartHeight}px;
            background: ${color};
            opacity: 0.18;
            pointer-events: none;
            z-index: 0;
            border-radius: 2px;
        `;

        const label = document.createElement('div');
        label.textContent = `${phase.phase}${phase.microphase ? ` (${phase.microphase})` : ''}`;
        label.style.cssText = `
            position: absolute;
            top: 6px;
            left: 8px;
            font-size: 0.85rem;
            color: #fff;
            opacity: 0.75;
        `;

        div.appendChild(label);
        overlayTarget.appendChild(div);
    });
}

function updateChart(data) {
    candleSeries.setData(data);
    if (showPhases) renderPhasesOverlay(); // <-- ПРАВИЛЬНЫЙ
}

function getPhaseColor(phase, microphase) {
    if (phase === 'panic') return '#cb2424';
    if (phase === 'trend_up') return '#19bc65';
    if (phase === 'trend_down') return '#f8a627';
    if (phase === 'flat') {
        if (microphase === 'flat_squeeze') return '#d4b01e';
        if (microphase === 'flat_microtrend_up' || microphase === 'flat_microtrend_down') return '#3d71fc';
        return '#334c77';
    }
    if (phase === 'volatile') return '#9452f7';
    return '#2e3d4f';
}


// Обработка ресайза окна
window.addEventListener('resize', () => {
    chart.applyOptions({
        width: chartContainer.clientWidth,
        height: chartContainer.clientHeight,
    });
});

// === Перетаскивание/скролл графика ===

let isDragging = false;
let dragStartX = 0;

chartContainer.style.cursor = 'grab';

chartContainer.addEventListener('mousedown', e => {
    isDragging = true;
    dragStartX = e.clientX;
    chartContainer.style.cursor = 'grabbing';
    e.preventDefault();
});

window.addEventListener('mouseup', () => {
    isDragging = false;
    chartContainer.style.cursor = 'grab';
});

window.addEventListener('mousemove', e => {
    if (!isDragging) return;
    const dx = dragStartX - e.clientX;
    dragStartX = e.clientX;

    const timeScale = chart.timeScale();
    const range = timeScale.getVisibleRange();
    if (!range) return;

    const barsCount = range.to - range.from;
    const offset = (dx / chartContainer.clientWidth) * barsCount * 1.5;

    // Прокрутка графика как влево, так и вправо
    const minTime = 0; // Минимальный таймштамп (начало данных)
    const maxTime = Date.now(); // Максимальный таймштамп (текущее время)

    const newFrom = timeScale.getPositionRange().from + offset;
    if (newFrom < minTime) {
        timeScale.scrollToPosition(minTime);
    } else if (newFrom > maxTime) {
        timeScale.scrollToPosition(maxTime);
    } else {
        timeScale.scrollToPosition(newFrom);
    }
});

// Обработка колесика мыши для зума и скролла
chartContainer.addEventListener('wheel', e => {
    e.preventDefault();
    const timeScale = chart.timeScale();
    const zoomFactor = e.deltaY < 0 ? 1.15 : 0.85;
    const rect = chartContainer.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    timeScale.zoomTimeScale(zoomFactor, mouseX);
}, { passive: false });

// ==== Анимация полей объёма в стакане ====

function animateWidth(element, targetWidthPercent, duration = 400) {
    let start = null;
    if (!element.parentElement) return;
    const parentWidth = element.parentElement.clientWidth;
    let initialWidth = 0;
    try {
        initialWidth = parseFloat(getComputedStyle(element).width) / parentWidth * 100 || 0;
    } catch {
        initialWidth = 0;
    }

    function step(timestamp) {
        if (!start) start = timestamp;
        const progress = Math.min((timestamp - start) / duration, 1);
        const currentWidth = initialWidth + (targetWidthPercent - initialWidth) * progress;
        element.style.width = `${currentWidth}%`;
        if (progress < 1) {
            requestAnimationFrame(step);
        }
    }
    requestAnimationFrame(step);
}

function updateOrderbook(bids, asks) {
    if (!asks.length || !bids.length) return;
    
    const asksContainer = document.getElementById('asks-container');
    const bidsContainer = document.getElementById('bids-container');
    const midDiv = document.getElementById('mid-price');

    // ОЧИСТКА ПЕРЕД ОТРИСОВКОЙ
    asksContainer.innerHTML = '';
    bidsContainer.innerHTML = '';
 
    const bestAsk = asks.length ? asks[0].price : null;
    const bestBid = bids.length ? bids[0].price : null;

    if (bestAsk !== null && bestBid !== null) {
        const newMid = ((bestAsk + bestBid) / 2).toFixed(5);

        const prevMid = parseFloat(midDiv.dataset.prevMid || 0);
        midDiv.dataset.prevMid = newMid;
        midDiv.textContent = newMid;

        if (newMid > prevMid) {
            midDiv.style.color = '#3fb950';  // зелёный
        } else if (newMid < prevMid) {
            midDiv.style.color = '#f85149';  // красный
        } else {
            midDiv.style.color = '#c9d1d9';  // нейтральный
        }

    } else {
        midDiv.textContent = '—';
        midDiv.style.color = '#8b949e';
    }

    

    const maxAskVol = Math.max(...asks.map(a => a.volume), 1);
    const maxBidVol = Math.max(...bids.map(b => b.volume), 1);
    
    const MAX_LEVELS = 10;

    // Ask (обратный порядок: сверху вниз)
    asks.slice(-MAX_LEVELS).reverse().forEach(({ price, volume, agent_id }) => {

        const row = document.createElement('div');
        row.className = 'orderbook-row sell';

        const priceDiv = document.createElement('div');
        priceDiv.className = 'price';
        priceDiv.textContent = price.toFixed(2);

        const volumeDiv = document.createElement('div');
        volumeDiv.className = 'volume';
        volumeDiv.textContent = volume.toFixed(3);

        const line = document.createElement('div');
        line.className = 'volume-line ask';

        if (agent_id && agent_id.startsWith('advcore')) {
            row.classList.add('advcore-order');
        }

        if (agent_id && agent_id.includes("advmm")) {
            row.classList.add("advmm-order");
        }


        line.style.width = `${(volume / maxAskVol) * 100}%`;


        row.appendChild(priceDiv);
        row.appendChild(volumeDiv);
        row.appendChild(line);
        asksContainer.appendChild(row);
    });

    // Bid (прямой порядок: снизу вверх)
    bids.slice(0, MAX_LEVELS).forEach(({ price, volume, agent_id }) => {
        const row = document.createElement('div');
        row.className = 'orderbook-row buy';

        const priceDiv = document.createElement('div');
        priceDiv.className = 'price';
        priceDiv.textContent = price.toFixed(2);

        const volumeDiv = document.createElement('div');
        volumeDiv.className = 'volume';
        volumeDiv.textContent = volume.toFixed(3);

        const line = document.createElement('div');
        line.className = 'volume-line bid';

        if (agent_id && agent_id.includes("advmm")) {
            row.classList.add("advmm-order");
        }


        line.style.width = `${(volume / maxBidVol) * 100}%`;

        row.appendChild(priceDiv);
        row.appendChild(volumeDiv);
        row.appendChild(line);
        bidsContainer.appendChild(row);
    });
}




// ==== Лента сделок (Trade Tape) ====

function addTrade(trade) {
    const li = document.createElement('li');
    const sideClass = trade.initiator_side === 'buy' ? 'buy' : 'sell';
    li.classList.add(sideClass);

    const priceStr = trade.price.toFixed(2);
    const volStr = trade.volume.toFixed(4);
    li.textContent = `${priceStr}  ${volStr}`;

    li.style.opacity = '0';
    tradesList.prepend(li);

    requestAnimationFrame(() => {
        li.style.transition = 'opacity 0.5s ease';
        li.style.opacity = '1';
    });

    if (tradesList.children.length > 100) {
        tradesList.removeChild(tradesList.lastChild);
    }
    tradesList.scrollTop = 0;
}

// ==== Инициализация Socket.IO-клиента ====

let socket = null;

function initSocketIO() {
    socket = io();

    socket.on('connect', () => {
        console.log('Socket.IO connected, SID=', socket.id);
    });

    socket.on('orderbook_update', (data) => {
        console.log('Order book update:', data);
        updateOrderbook(data.bids, data.asks);
    });

    socket.on('trade', (trade) => {
        console.log('Trade:', trade);
        addTrade(trade);
    });

    socket.on('history', (historyArray) => {
        console.log('Trade history:', historyArray);
        historyArray.forEach(oldTrade => addTrade(oldTrade));
    });

    socket.on('candles', (ohlcData) => {
        updateChart(ohlcData);
    });

    socket.on('confirmation', (msg) => {
        console.log('Server confirmation:', msg.message);
    });

    socket.on('error', (err) => {
        console.error('Server error:', err.message || err);
    });

    socket.on('disconnect', () => {
        console.warn('Socket.IO disconnected, reconnecting in 3s...');
        setTimeout(() => {
            initSocketIO();
        }, 3000);
    });
}

document.addEventListener('DOMContentLoaded', () => {
    initSocketIO();
    fetchAndRenderCandles();
});

// ==== Обновление графика (candles) ====

function updateChart(data) {
    candleSeries.setData(data);
}

let selectedTF = 5;

// Устанавливаем период обновления данных (например, каждую 500 миллисекунд)
setInterval(() => {
    fetchAndRenderCandles();
}, 500);  // 500 миллисекунд = 0.5 секунда

document.querySelectorAll('#tf-selector button').forEach(btn => {
    btn.addEventListener('click', () => {
        selectedTF = parseInt(btn.dataset.tf);
        fetchAndRenderCandles();
    });
});

function fetchAndRenderCandles() {
    fetch(`/candles?interval=${selectedTF}`)
        .then(res => res.json())
        .then(data => {
            
            candleSeries.setData(data.map(c => ({
                time: c.timestamp,
                open: c.open,
                high: c.high,
                low: c.low,
                close: c.close
            })));
        });
}

chart.timeScale().subscribeVisibleTimeRangeChange(() => {
    if (showPhases) renderPhasesOverlay();
});
