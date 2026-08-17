"""
Web-based Live Training Dashboard.

FastAPI + WebSocket real-time dashboard for training monitoring.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from monitoring.event_bus import HandlerPriority, get_event_bus
from monitoring.events import EventType, Severity, TrainingEvent
from monitoring.unified_logger import UnifiedLogger

# Global state
logger: UnifiedLogger | None = None
active_connections: list[WebSocket] = []
metrics_history: dict[str, list[dict]] = {}
max_history = 1000


class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections[:]:
            try:
                await connection.send_text(message)
            except Exception:
                self.active_connections.remove(connection)


manager = ConnectionManager()


def create_dashboard_app(logger_instance: UnifiedLogger | None = None) -> FastAPI:
    """Create FastAPI app with dashboard routes."""
    global logger
    logger = logger_instance

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        if logger and not logger._started:
            await logger.start()
        yield
        # Shutdown
        if logger:
            await logger.stop()

    app = FastAPI(
        title="Forex Training Dashboard",
        description="Real-time training monitoring dashboard",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Serve static files if directory exists
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # ===== REST API =====

    @app.get("/api/health")
    async def health():
        return {"status": "healthy", "timestamp": datetime.now(UTC).isoformat()}

    @app.get("/api/logger/stats")
    async def logger_stats():
        if logger:
            return logger.get_stats()
        return {"error": "Logger not initialized"}

    @app.get("/api/metrics")
    async def get_metrics(metric: str = Query(None)):
        global metrics_history
        if metric:
            return {metric: metrics_history.get(metric, [])}
        return metrics_history

    @app.get("/api/events")
    async def query_events(
        run_id: str = Query(None),
        event_type: str = Query(None),
        severity: str = Query(None),
        since: str = Query(None),
        limit: int = Query(100),
    ):
        bus = get_event_bus()
        try:
            et = EventType(event_type) if event_type else None
            sev = Severity(severity) if severity else None
            events = await bus.query_events(
                run_id=run_id,
                event_type=et,
                severity=sev,
                since=since,
                limit=limit,
            )
            return {"events": [e.to_dict() for e in events]}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/api/checks/results")
    async def get_check_results(phase: str = Query(None)):
        from monitoring.checks import get_engine

        engine = get_engine()
        if phase:
            try:
                from monitoring.events import CheckPhase

                ph = CheckPhase(phase)
                results = engine.get_results(ph)
                return {k: [r.to_dict() for r in v] for k, v in results.items()}
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid phase: {phase}")
        results = engine.get_results()
        return {k: [r.to_dict() for r in v] for k, v in results.items()}

    @app.get("/api/checks/summary")
    async def get_check_summary():
        from monitoring.checks import get_engine

        engine = get_engine()
        return engine.get_summary()

    @app.get("/api/runs")
    async def list_runs():
        """List recent training runs."""
        bus = get_event_bus()
        # Query unique run_ids from events
        events = await bus.query_events(limit=1000)
        runs = {}
        for e in events:
            if e.run_id and e.run_id not in runs:
                runs[e.run_id] = {
                    "run_id": e.run_id,
                    "first_seen": e.timestamp,
                    "model": e.model_name,
                    "event_count": 1,
                }
            elif e.run_id in runs:
                runs[e.run_id]["event_count"] += 1
        return {"runs": list(runs.values())}

    # ===== WebSocket =====

    # Background task to broadcast metric events to WebSocket clients
    async def broadcast_metrics():
        """Background task to push metrics to WebSocket clients."""
        bus = get_event_bus()

        # Register a handler that broadcasts metric events
        async def metric_handler(event: TrainingEvent):
            if event.event_type == EventType.METRIC:
                name = event.payload.get("name", "")
                if name:
                    entry = {
                        "name": name,
                        "value": event.payload.get("value"),
                        "epoch": event.epoch,
                        "batch": event.batch,
                        "model": event.model_name,
                        "unit": event.payload.get("unit", ""),
                        "timestamp": event.timestamp,
                    }
                    metrics_history[name].append(entry)
                    if len(metrics_history[name]) > max_history:
                        metrics_history[name] = metrics_history[name][-max_history:]

                    # Broadcast to all connected WebSocket clients
                    await manager.broadcast(
                        json.dumps(
                            {
                                "type": "metric",
                                "name": name,
                                **entry,
                            }
                        )
                    )

        bus.register_handler(
            metric_handler,
            event_types={EventType.METRIC},
            priority=HandlerPriority.NORMAL,
        )

        # Keep running
        while True:
            await asyncio.sleep(3600)  # Run indefinitely

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await manager.connect(websocket)
        try:
            # Send initial state
            await websocket.send_json(
                {
                    "type": "init",
                    "metrics": metrics_history,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )

            while True:
                data = await websocket.receive_text()
                try:
                    msg = json.loads(data)
                    if msg.get("type") == "ping":
                        await websocket.send_json({"type": "pong", "timestamp": datetime.now(UTC).isoformat()})
                    elif msg.get("type") == "subscribe":
                        # Handle subscription to specific metrics/events
                        pass
                except Exception:
                    pass
        except WebSocketDisconnect:
            manager.disconnect(websocket)
        except Exception:
            manager.disconnect(websocket)

    # Start background metrics broadcaster
    @app.on_event("startup")
    async def start_metrics_broadcaster():
        asyncio.create_task(broadcast_metrics())  # noqa: RUF006

    # ===== Dashboard HTML =====

    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        return DASHBOARD_HTML

    return app


# Dashboard HTML template
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Forex Training Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #1a1a2e; color: #eee; line-height: 1.6; }
        .header { background: #16213e; padding: 1rem 2rem; border-bottom: 1px solid #0f3460; display: flex; justify-content: space-between; align-items: center; position: sticky; top: 0; z-index: 100; }
        .header h1 { font-size: 1.5rem; font-weight: 600; }
        .status { display: flex; gap: 1rem; align-items: center; }
        .badge { padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; }
        .badge-connected { background: #10b981; color: white; }
        .badge-disconnected { background: #ef4444; color: white; }
        .container { max-width: 1800px; margin: 0 auto; padding: 2rem; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 1.5rem; }
        .card { background: #16213e; border: 1px solid #0f3460; border-radius: 12px; padding: 1.5rem; }
        .card h2 { font-size: 1.1rem; margin-bottom: 1rem; color: #93c5fd; border-bottom: 1px solid #0f3460; padding-bottom: 0.5rem; }
        .metric-row { display: flex; justify-content: space-between; padding: 0.5rem 0; border-bottom: 1px solid rgba(255,255,255,0.05); }
        .metric-row:last-child { border-bottom: none; }
        .metric-label { color: #94a3b8; }
        .metric-value { font-weight: 600; font-family: monospace; }
        .metric-value.ok { color: #10b981; }
        .metric-value.warn { color: #f59e0b; }
        .metric-value.error { color: #ef4444; }
        .chart-container { height: 300px; position: relative; }
        .event-log { max-height: 400px; overflow-y: auto; font-family: monospace; font-size: 0.8rem; }
        .event-entry { padding: 0.25rem 0; border-bottom: 1px solid rgba(255,255,255,0.03); display: flex; gap: 0.5rem; }
        .event-time { color: #64748b; min-width: 80px; }
        .event-type { min-width: 100px; text-transform: uppercase; font-weight: 600; }
        .event-type.LOG { color: #64748b; }
        .event-type.METRIC { color: #3b82f6; }
        .event-type.CHECK { color: #10b981; }
        .event-type.ALERT { color: #ef4444; }
        .event-type.CHECKPOINT { color: #f59e0b; }
        .event-source { color: #64748b; min-width: 150px; }
        .event-message { flex: 1; }
        .tabs { display: flex; gap: 0.5rem; margin-bottom: 1rem; }
        .tab { padding: 0.5rem 1rem; background: transparent; border: 1px solid #0f3460; border-radius: 6px; color: #94a3b8; cursor: pointer; }
        .tab.active { background: #0f3460; color: #fff; border-color: #3b82f6; }
        .tab:hover { background: #1e293b; }
        .hidden { display: none !important; }
        .loading { text-align: center; padding: 3rem; color: #64748b; }
    </style>
</head>
<body>
    <header class="header">
        <h1>📊 Forex Training Dashboard</h1>
        <div class="status">
            <span class="badge badge-disconnected" id="ws-status">Disconnected</span>
            <span id="run-info"></span>
        </div>
    </header>

    <div class="container">
        <div class="tabs">
            <button class="tab active" data-tab="metrics">📈 Metrics</button>
            <button class="tab" data-tab="checks">✅ Checks</button>
            <button class="tab" data-tab="events">📋 Events</button>
            <button class="tab" data-tab="system">🖥️ System</button>
        </div>

        <div id="tab-metrics" class="tab-content">
            <div class="grid">
                <div class="card">
                    <h2>Training Metrics</h2>
                    <div class="chart-container"><canvas id="loss-chart"></canvas></div>
                </div>
                <div class="card">
                    <h2>Validation Metrics</h2>
                    <div class="chart-container"><canvas id="val-chart"></canvas></div>
                </div>
                <div class="card">
                    <h2>Learning Rate</h2>
                    <div class="chart-container"><canvas id="lr-chart"></canvas></div>
                </div>
                <div class="card">
                    <h2>Gradient Norm</h2>
                    <div class="chart-container"><canvas id="grad-chart"></canvas></div>
                </div>
            </div>
        </div>

        <div id="tab-checks" class="tab-content hidden">
            <div class="grid">
                <div class="card">
                    <h2>Check Results</h2>
                    <div id="check-summary"></div>
                    <div id="check-details"></div>
                </div>
            </div>
        </div>

        <div id="tab-events" class="tab-content hidden">
            <div class="card">
                <h2>Event Log</h2>
                <div class="event-log" id="event-log"></div>
            </div>
        </div>

        <div id="tab-system" class="tab-content hidden">
            <div class="grid">
                <div class="card">
                    <h2>System Resources</h2>
                    <div class="chart-container"><canvas id="gpu-mem-chart"></canvas></div>
                    <div class="chart-container"><canvas id="cpu-mem-chart"></canvas></div>
                </div>
                <div class="card">
                    <h2>Run Info</h2>
                    <div id="run-details"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Chart.js defaults
        Chart.defaults.color = '#94a3b8';
        Chart.defaults.borderColor = 'rgba(255,255,255,0.1)';
        Chart.defaults.font.family = 'monospace';

        // State
        let ws = null;
        let charts = {};
        const maxPoints = 100;

        // Initialize charts
        function initCharts() {
            const common = {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 200 },
                scales: {
                    x: { display: false },
                    y: { beginAtZero: false, grid: { color: 'rgba(255,255,255,0.05)' } }
                },
                plugins: { legend: { display: false } },
            };

            charts.loss = new Chart('loss-chart', { type: 'line', data: { labels: [], datasets: [{ label: 'Train Loss', data: [], borderColor: '#3b82f6', fill: false, tension: 0.2 }] }, options: common });
            charts.val = new Chart('val-chart', { type: 'line', data: { labels: [], datasets: [{ label: 'Val Loss', data: [], borderColor: '#f59e0b', fill: false, tension: 0.2 }, { label: 'Val Sharpe', data: [], borderColor: '#10b981', fill: false, tension: 0.2, yAxisID: 'y1' }] }, options: { ...common, scales: { ...common.scales, y1: { type: 'linear', display: true, position: 'right', grid: { drawOnChartArea: false } } } });
            charts.lr = new Chart('lr-chart', { type: 'line', data: { labels: [], datasets: [{ label: 'LR', data: [], borderColor: '#8b5cf6', fill: false }] }, options: common });
            charts.grad = new Chart('grad-chart', { type: 'line', data: { labels: [], datasets: [{ label: 'Grad Norm', data: [], borderColor: '#ef4444', fill: false }] }, options: common });
            charts.gpuMem = new Chart('gpu-mem-chart', { type: 'line', data: { labels: [], datasets: [{ label: 'GPU Mem %', data: [], borderColor: '#3b82f6', fill: false }] }, options: common });
            charts.cpuMem = new Chart('cpu-mem-chart', { type: 'line', data: { labels: [], datasets: [{ label: 'CPU Mem %', data: [], borderColor: '#10b981', fill: false }] }, options: common });
        }

        // Add point to chart
        function addPoint(chart, x, y, maxPts = 100) {
            if (!chart.data.labels) chart.data.labels = [];
            if (!chart.data.datasets[0].data) chart.data.datasets[0].data = [];

            chart.data.labels.push(x);
            chart.data.datasets[0].data.push(y);

            if (chart.data.datasets[1]) {
                chart.data.datasets[1].data.push(y);
            }

            if (chart.data.labels.length > maxPts) {
                chart.data.labels.shift();
                chart.data.datasets.forEach(ds => ds.data.shift());
            }
            chart.update('none');
        }

        // WebSocket
        function connect() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

            ws.onopen = () => {
                document.getElementById('ws-status').textContent = 'Connected';
                document.getElementById('ws-status').className = 'badge badge-connected';
            };

            ws.onclose = () => {
                document.getElementById('ws-status').textContent = 'Disconnected';
                document.getElementById('ws-status').className = 'badge badge-disconnected';
                setTimeout(connect, 3000);
            };

            ws.onmessage = (event) => {
                try {
                    const msg = JSON.parse(event.data);
                    handleMessage(msg);
                } catch (e) {
                    console.error('WS message error:', e);
                }
            };
        }

        function handleMessage(msg) {
            switch (msg.type) {
                case 'init':
                    if (msg.metrics) {
                        Object.entries(msg.metrics).forEach(([name, points]) => {
                            if (points.length) {
                                points.slice(-100).forEach(p => addMetricPoint(name, p));
                            }
                        });
                    }
                    break;
                case 'metric':
                    addMetricPoint(msg.name, msg);
                    break;
                case 'check':
                    updateCheck(msg);
                    break;
                case 'event':
                    addEvent(msg);
                    break;
                case 'stats':
                    updateStats(msg);
                    break;
            }
        }

        function addMetricPoint(name, point) {
            const epoch = point.epoch || 0;
            const value = point.value;

            // Route to appropriate chart
            if (name.includes('loss') && !name.includes('val')) addPoint(charts.loss, epoch, value);
            else if (name.includes('val_loss')) addPoint(charts.val, epoch, value);
            else if (name.includes('val_sharpe')) addPoint(charts.val, epoch, value);
            else if (name.includes('lr') || name.includes('learning_rate')) addPoint(charts.lr, epoch, value);
            else if (name.includes('grad_norm')) addPoint(charts.grad, epoch, value);
            else if (name.includes('gpu_mem')) addPoint(charts.gpuMem, epoch, value);
            else if (name.includes('cpu_mem')) addPoint(charts.cpuMem, epoch, value);
        }

        function updateCheck(msg) {
            const container = document.getElementById('check-details');
            if (!container) return;

            const status = msg.payload.passed ? '✅' : '❌';
            const el = document.getElementById(`check-${msg.payload.name}`);
            const html = `<div class="metric-row">
                <span class="metric-label">${msg.payload.name}</span>
                <span class="metric-value ${msg.payload.passed ? 'ok' : 'error'}">${status} ${msg.payload.message}</span>
            </div>`;

            if (el) el.outerHTML = `<div id="check-${msg.payload.name}">${html}</div>`;
            else container.insertAdjacentHTML('afterbegin', `<div id="check-${msg.payload.name}">${html}</div>`);

            // Update summary
            updateCheckSummary();
        }

        function updateCheckSummary() {
            const container = document.getElementById('check-summary');
            const checks = container.querySelectorAll('[id^="check-"]');
            let passed = 0, failed = 0;
            checks.forEach(c => {
                if (c.querySelector('.ok')) passed++;
                else if (c.querySelector('.error')) failed++;
            });
            document.getElementById('check-summary').innerHTML =
                `<div class="metric-row"><span class="metric-label">Total</span><span class="metric-value">${checks.length}</span></div>
                 <div class="metric-row"><span class="metric-label">Passed</span><span class="metric-value ok">${passed}</span></div>
                 <div class="metric-row"><span class="metric-label">Failed</span><span class="metric-value error">${failed}</span></div>`;
        }

        function addEvent(msg) {
            const log = document.getElementById('event-log');
            if (!log) return;

            const time = new Date(msg.timestamp).toLocaleTimeString();
            const type = msg.event_type || 'LOG';
            const severity = msg.severity || 'info';
            const source = msg.source || '';
            const message = msg.payload.message || JSON.stringify(msg.payload);

            const colors = { LOG: '#64748b', METRIC: '#3b82f6', CHECK: '#10b981', ALERT: '#ef4444', CHECKPOINT: '#f59e0b', PROGRESS: '#f59e0b', HEARTBEAT: '#64748b' };
            const color = colors[type] || '#64748b';

            const div = document.createElement('div');
            div.className = 'event-entry';
            div.innerHTML = `<span class="event-time">${time}</span><span class="event-type" style="color:${color}">${type}</span><span class="event-source">${source}</span><span class="event-message">${message}</span>`;

            log.insertBefore(div, log.firstChild);

            // Limit entries
            while (log.children.length > 200) log.removeChild(log.lastChild);
        }

        function updateStats(msg) {
            document.getElementById('run-info').textContent = `Run: ${msg.run_id} | Model: ${msg.model || 'N/A'}`;
        }

        // Tabs
        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', () => {
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(c => c.classList.add('hidden'));
                tab.classList.add('active');
                document.getElementById(`tab-${tab.dataset.tab}`).classList.remove('hidden');
            });
        });

        // Init
        initCharts();
        connect();

        // Poll REST for initial data
        async function poll() {
            try {
                const [stats, runs] = await Promise.all([
                    fetch('/api/logger/stats').then(r => r.json()),
                    fetch('/api/runs').then(r => r.json()),
                ]);
                if (stats.run_id) document.getElementById('run-info').textContent = `Run: ${stats.run_id}`;
            } catch (e) {}
            setTimeout(poll, 5000);
        }
        poll();

        // Reconnect on visibility change
        document.addEventListener('visibilitychange', () => {
            if (!document.hidden && (!ws || ws.readyState !== WebSocket.OPEN)) connect();
        });
    </script>
</body>
</html>
"""


def run_dashboard(
    host: str = "0.0.0.0",
    port: int = 9090,
    logger_instance: UnifiedLogger | None = None,
):
    """Run the dashboard server."""
    import uvicorn

    app = create_dashboard_app(logger_instance)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    run_dashboard()
