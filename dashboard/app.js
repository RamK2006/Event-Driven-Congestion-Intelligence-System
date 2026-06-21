/**
 * Event Command Center — Dashboard JavaScript
 * Event Impact & Response Intelligence Platform
 * 
 * Handles: tab navigation, map rendering (Leaflet + MarkerCluster),
 * prediction form submission, workload charts, historical trends,
 * methodology metrics display. All data fetched from Flask API.
 */

// ============================================================================
// CONFIGURATION
// ============================================================================
// In production, set window.APP_CONFIG = { API_BASE: 'https://your-backend.onrender.com' }
// in index.html before this script loads. Falls back to same-origin for local dev.
const API_BASE = (window.APP_CONFIG && window.APP_CONFIG.API_BASE) || window.location.origin;
const BENGALURU_CENTER = [12.9716, 77.5946];
const DEFAULT_ZOOM = 12;

// Chart.js global defaults for dark theme
Chart.defaults.color = '#94a3b8';
Chart.defaults.borderColor = 'rgba(255,255,255,0.06)';
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.font.size = 11;
Chart.defaults.plugins.legend.labels.usePointStyle = true;
Chart.defaults.plugins.legend.labels.pointStyleWidth = 10;

// Color palette
const COLORS = {
    primary: '#6366f1',
    primaryLight: 'rgba(99, 102, 241, 0.2)',
    secondary: '#8b5cf6',
    secondaryLight: 'rgba(139, 92, 246, 0.2)',
    success: '#10b981',
    successLight: 'rgba(16, 185, 129, 0.2)',
    warning: '#f59e0b',
    warningLight: 'rgba(245, 158, 11, 0.2)',
    danger: '#ef4444',
    dangerLight: 'rgba(239, 68, 68, 0.2)',
    info: '#06b6d4',
    infoLight: 'rgba(6, 182, 212, 0.2)',
    orange: '#f97316',
    orangeLight: 'rgba(249, 115, 22, 0.2)',
    chartPalette: [
        '#6366f1', '#8b5cf6', '#06b6d4', '#10b981', '#f59e0b',
        '#ef4444', '#f97316', '#ec4899', '#14b8a6', '#a78bfa',
        '#fbbf24', '#34d399', '#fb7185', '#38bdf8'
    ]
};

// ============================================================================
// STATE
// ============================================================================
let map = null;
let markerClusterGroup = null;
let hotspotLayers = [];
let chartInstances = {};
let filtersLoaded = false;
let summaryData = null;

// ============================================================================
// INITIALIZATION
// ============================================================================
document.addEventListener('DOMContentLoaded', () => {
    // Set default datetime to now
    const now = new Date();
    const localISO = new Date(now.getTime() - now.getTimezoneOffset() * 60000)
        .toISOString().slice(0, 16);
    document.getElementById('inputDatetime').value = localISO;

    // Initialize map
    initMap();

    // Load data
    loadSummary();
    loadEvents();
    loadHotspots();
    loadFilters();
});


// ============================================================================
// TAB NAVIGATION
// ============================================================================
function switchTab(btn) {
    // Deactivate all tabs and panels
    document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));

    // Activate clicked tab and panel
    btn.classList.add('active');
    const panelId = btn.getAttribute('data-panel');
    document.getElementById(panelId).classList.add('active');

    // Lazy-load panel data
    if (panelId === 'workload-panel') {
        loadWorkload();
    } else if (panelId === 'trends-panel') {
        if (!filtersLoaded) loadFilters();
        loadTrends();
        loadCauseBreakdown();
    } else if (panelId === 'methodology-panel') {
        loadMethodologyMetrics();
    } else if (panelId === 'map-panel') {
        // Invalidate map size after tab switch
        setTimeout(() => { if (map) map.invalidateSize(); }, 100);
    }
}


// ============================================================================
// MAP INITIALIZATION
// ============================================================================
function initMap() {
    map = L.map('incidentMap', {
        zoomControl: true,
        attributionControl: true,
    }).setView(BENGALURU_CENTER, DEFAULT_ZOOM);

    // Dark tile layer
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>',
        maxZoom: 19,
    }).addTo(map);

    // Create marker cluster group
    markerClusterGroup = L.markerClusterGroup({
        maxClusterRadius: 40,
        spiderfyOnMaxZoom: true,
        showCoverageOnHover: false,
        iconCreateFunction: function(cluster) {
            const count = cluster.getChildCount();
            let size = 'small';
            let px = 36;
            if (count > 100) { size = 'large'; px = 50; }
            else if (count > 30) { size = 'medium'; px = 42; }

            return L.divIcon({
                html: `<div style="
                    background: linear-gradient(135deg, rgba(99,102,241,0.85), rgba(139,92,246,0.85));
                    color: white;
                    border-radius: 50%;
                    width: ${px}px;
                    height: ${px}px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-family: 'JetBrains Mono', monospace;
                    font-size: ${count > 100 ? 12 : 11}px;
                    font-weight: 700;
                    border: 2px solid rgba(255,255,255,0.3);
                    box-shadow: 0 0 15px rgba(99,102,241,0.4);
                ">${count}</div>`,
                className: 'custom-cluster-icon',
                iconSize: [px, px]
            });
        }
    });
    map.addLayer(markerClusterGroup);
}


// ============================================================================
// LOAD EVENTS (MAP MARKERS)
// ============================================================================
async function loadEvents() {
    try {
        const res = await fetch(`${API_BASE}/api/events`);
        const events = await res.json();

        markerClusterGroup.clearLayers();

        events.forEach(evt => {
            const lat = parseFloat(evt.latitude);
            const lon = parseFloat(evt.longitude);
            if (isNaN(lat) || isNaN(lon) || lat === 0 || lon === 0) return;

            const isHigh = evt.priority_target === 'High';
            const color = isHigh ? COLORS.danger : COLORS.success;
            const colorGlow = isHigh ? 'rgba(239,68,68,0.4)' : 'rgba(16,185,129,0.4)';

            const marker = L.circleMarker([lat, lon], {
                radius: 4,
                fillColor: color,
                color: color,
                weight: 1,
                opacity: 0.8,
                fillOpacity: 0.6,
            });

            const popupContent = `
                <div style="font-family: 'Inter', sans-serif; font-size: 12px; min-width: 200px;">
                    <div style="font-weight: 700; margin-bottom: 6px; font-size: 13px;">
                        ${evt.event_cause || 'Unknown'}
                        <span style="
                            display: inline-block;
                            padding: 1px 6px;
                            border-radius: 999px;
                            font-size: 9px;
                            font-weight: 600;
                            margin-left: 6px;
                            background: ${isHigh ? 'rgba(239,68,68,0.2)' : 'rgba(16,185,129,0.2)'};
                            color: ${color};
                            border: 1px solid ${isHigh ? 'rgba(239,68,68,0.3)' : 'rgba(16,185,129,0.3)'};
                        ">${evt.priority_target || 'N/A'}</span>
                    </div>
                    <div style="color: #94a3b8; line-height: 1.6;">
                        <strong>Type:</strong> ${evt.event_type || 'N/A'}<br>
                        <strong>Corridor:</strong> ${evt.corridor || 'N/A'}<br>
                        <strong>Station:</strong> ${evt.police_station || 'N/A'}<br>
                        <strong>Status:</strong> ${evt.status || 'N/A'}<br>
                        <strong>Vehicle:</strong> ${evt.veh_type || 'N/A'}<br>
                        <strong>Closure:</strong> ${evt.closure_target == 1 ? 'Yes' : 'No'}<br>
                        ${evt.clearance_time_minutes && evt.clearance_time_minutes !== 'N/A'
                            ? `<strong>Clearance:</strong> ${parseFloat(evt.clearance_time_minutes).toFixed(0)} min<br>` : ''}
                        <strong>Time:</strong> ${evt.start_datetime || 'N/A'}
                    </div>
                </div>
            `;
            marker.bindPopup(popupContent);
            markerClusterGroup.addLayer(marker);
        });

        console.log(`Loaded ${events.length} events on map`);
    } catch (err) {
        console.error('Error loading events:', err);
    }
}


// ============================================================================
// LOAD SUMMARY STATS
// ============================================================================
async function loadSummary() {
    try {
        const res = await fetch(`${API_BASE}/api/events/summary`);
        summaryData = await res.json();

        document.getElementById('totalEvents').textContent =
            summaryData.total_events?.toLocaleString() || '—';
        document.getElementById('highPriority').textContent =
            (summaryData.priority_dist?.High || 0).toLocaleString();
        document.getElementById('lowPriority').textContent =
            (summaryData.priority_dist?.Low || 0).toLocaleString();

        // Count road closures from closure distribution
        const closures = summaryData.total_events || 0;
        // We'll get this from the actual data
        document.getElementById('closureCount').textContent = '—';

    } catch (err) {
        console.error('Error loading summary:', err);
    }
}


// ============================================================================
// LOAD HOTSPOTS
// ============================================================================
async function loadHotspots() {
    try {
        const res = await fetch(`${API_BASE}/api/hotspots`);
        const data = await res.json();

        document.getElementById('hotspotCount').textContent =
            data.total_clusters || '—';

        // Add hotspot overlays to map
        if (data.hotspots && data.hotspots.length > 0) {
            data.hotspots.forEach(h => {
                // Draw a semi-transparent circle for each hotspot
                const radius = Math.min(Math.max(h.event_count * 3, 200), 2000);
                const circle = L.circle([h.center_latitude, h.center_longitude], {
                    radius: radius,
                    fillColor: COLORS.warning,
                    color: COLORS.warning,
                    weight: 1.5,
                    opacity: 0.6,
                    fillOpacity: 0.15,
                    className: 'hotspot-circle'
                });

                circle.bindPopup(`
                    <div style="font-family: 'Inter', sans-serif; font-size: 12px; min-width: 220px;">
                        <div style="font-weight: 700; color: ${COLORS.warning}; margin-bottom: 6px;">
                            🔥 Hotspot Cluster #${h.rank}
                        </div>
                        <div style="color: #94a3b8; line-height: 1.6;">
                            <strong>Events:</strong> ${h.event_count}<br>
                            <strong>Top Cause:</strong> ${h.dominant_event_cause} (${h.dominant_cause_pct}%)<br>
                            <strong>Priority:</strong> ${h.dominant_priority} (${h.dominant_priority_pct}%)<br>
                            <strong>Station:</strong> ${h.associated_police_station}<br>
                            <strong>Corridor:</strong> ${h.associated_corridor}<br>
                            <strong>Zone:</strong> ${h.associated_zone}
                        </div>
                    </div>
                `);
                circle.addTo(map);
                hotspotLayers.push(circle);
            });
        }

        // Also update closure count from workload data
        try {
            const wRes = await fetch(`${API_BASE}/api/workload`);
            const wData = await wRes.json();
            const totalClosures = wData.reduce((sum, s) => sum + (s.road_closures || 0), 0);
            document.getElementById('closureCount').textContent = totalClosures.toLocaleString();
        } catch (e) { /* ignore */ }

    } catch (err) {
        console.error('Error loading hotspots:', err);
    }
}


// ============================================================================
// LOAD FILTERS
// ============================================================================
async function loadFilters() {
    try {
        const res = await fetch(`${API_BASE}/api/filters`);
        const data = await res.json();

        // Populate trend filters
        const causeSelect = document.getElementById('trendCauseFilter');
        const corridorSelect = document.getElementById('trendCorridorFilter');
        const zoneSelect = document.getElementById('trendZoneFilter');

        if (data.causes) {
            data.causes.forEach(c => {
                const opt = document.createElement('option');
                opt.value = c; opt.textContent = c;
                causeSelect.appendChild(opt);
            });
        }

        if (data.corridors) {
            data.corridors.forEach(c => {
                const opt = document.createElement('option');
                opt.value = c; opt.textContent = c;
                corridorSelect.appendChild(opt);
            });
        }

        if (data.zones) {
            data.zones.forEach(z => {
                const opt = document.createElement('option');
                opt.value = z; opt.textContent = z;
                zoneSelect.appendChild(opt);
            });
        }

        filtersLoaded = true;
    } catch (err) {
        console.error('Error loading filters:', err);
    }
}


// ============================================================================
// PREDICTION FORM SUBMISSION
// ============================================================================
async function submitPrediction() {
    const btn = document.getElementById('predictBtn');
    btn.disabled = true;
    btn.innerHTML = '<div class="spinner" style="width:16px;height:16px;border-width:2px;"></div> Predicting...';

    const payload = {
        latitude: parseFloat(document.getElementById('inputLat').value),
        longitude: parseFloat(document.getElementById('inputLon').value),
        event_cause: document.getElementById('inputCause').value,
        veh_type: document.getElementById('inputVehType').value,
        event_type: document.getElementById('inputEventType').value,
        datetime_str: new Date(document.getElementById('inputDatetime').value).toISOString(),
    };

    try {
        const res = await fetch(`${API_BASE}/api/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();

        // Show results
        document.getElementById('predictionResults').style.display = 'block';

        // Priority
        if (data.priority && !data.priority.error) {
            const pVal = document.getElementById('predPriorityValue');
            pVal.textContent = data.priority.prediction;
            pVal.className = 'prediction-value ' + data.priority.prediction.toLowerCase();
            document.getElementById('predPriorityConf').textContent =
                `Confidence: ${(data.priority.confidence * 100).toFixed(1)}%`;
        } else {
            document.getElementById('predPriorityValue').textContent = 'Error';
            document.getElementById('predPriorityConf').textContent =
                data.priority?.error || 'Model unavailable';
        }

        // Closure
        if (data.closure && !data.closure.error) {
            const cVal = document.getElementById('predClosureValue');
            cVal.textContent = data.closure.prediction;
            cVal.className = 'prediction-value ' + data.closure.prediction.toLowerCase();
            document.getElementById('predClosureConf').textContent =
                `Probability: ${(data.closure.probability * 100).toFixed(1)}%`;
        } else {
            document.getElementById('predClosureValue').textContent = 'Error';
            document.getElementById('predClosureConf').textContent =
                data.closure?.error || 'Model unavailable';
        }

        // Clearance time
        if (data.clearance_time && !data.clearance_time.error) {
            const mins = data.clearance_time.predicted_minutes;
            if (mins >= 60) {
                document.getElementById('predClearanceValue').textContent =
                    `${data.clearance_time.predicted_hours}h`;
            } else {
                document.getElementById('predClearanceValue').textContent =
                    `${mins.toFixed(0)} min`;
            }
            document.getElementById('predClearanceConf').textContent =
                `≈ ${data.clearance_time.predicted_hours} hours`;
        } else {
            document.getElementById('predClearanceValue').textContent = 'Error';
            document.getElementById('predClearanceConf').textContent =
                data.clearance_time?.error || 'Model unavailable';
        }

        // Location context
        if (data.input) {
            document.getElementById('locationContext').innerHTML = `
                <div style="line-height: 1.8;">
                    <strong>Corridor:</strong> ${data.input.inferred_corridor}<br>
                    <strong>Police Station:</strong> ${data.input.inferred_police_station}<br>
                    <strong>Zone:</strong> ${data.input.inferred_zone}<br>
                    <strong>Coordinates:</strong> ${data.input.latitude}, ${data.input.longitude}
                </div>
            `;
        }

        // Diversion
        if (data.diversion && !data.diversion.error) {
            const suggestions = data.diversion.suggestions || [];
            let divHtml = `<div style="margin-bottom: 8px;"><strong>Corridor:</strong> ${data.diversion.corridor}</div>`;
            if (suggestions.length > 0) {
                divHtml += '<div style="line-height: 1.8;">';
                suggestions.forEach((s, i) => {
                    divHtml += `
                        <div class="diversion-card">
                            <span class="corridor-name">→ ${s.corridor}</span>
                            <div class="diversion-suggestion">
                                Co-occurrence ratio: ${(s.co_occurrence_ratio * 100).toFixed(1)}%
                                · ${s.total_events} historical events
                            </div>
                        </div>
                    `;
                });
                divHtml += '</div>';
            } else {
                divHtml += '<div style="color: var(--text-tertiary);">No diversion data available for this corridor.</div>';
            }
            document.getElementById('diversionContext').innerHTML = divHtml;
        }

        // Scroll to results
        document.getElementById('predictionResults').scrollIntoView({ behavior: 'smooth' });

    } catch (err) {
        console.error('Prediction error:', err);
        alert('Error: Could not get predictions. Is the server running?');
    } finally {
        btn.disabled = false;
        btn.innerHTML = '⚡ Get Predictions';
    }
}


// ============================================================================
// WORKLOAD PANEL
// ============================================================================
async function loadWorkload() {
    try {
        let url = `${API_BASE}/api/workload`;
        const params = [];
        const startDate = document.getElementById('workloadStartDate')?.value;
        const endDate = document.getElementById('workloadEndDate')?.value;
        if (startDate) params.push(`start_date=${startDate}`);
        if (endDate) params.push(`end_date=${endDate}`);
        if (params.length) url += '?' + params.join('&');

        const res = await fetch(url);
        const data = await res.json();

        // Update table
        const tbody = document.getElementById('workloadTableBody');
        tbody.innerHTML = data.map(row => `
            <tr>
                <td style="font-weight: 500;">${row.police_station}</td>
                <td><span style="font-family: var(--font-mono); font-weight: 600;">${row.total_events}</span></td>
                <td><span class="badge badge-high">${row.high_priority}</span></td>
                <td><span class="badge badge-low">${row.low_priority}</span></td>
                <td>${row.road_closures || 0}</td>
            </tr>
        `).join('');

        // Update chart
        const top20 = data.slice(0, 20);
        renderWorkloadChart(top20);

    } catch (err) {
        console.error('Error loading workload:', err);
    }
}

function renderWorkloadChart(data) {
    const ctx = document.getElementById('workloadChart');
    if (!ctx) return;

    if (chartInstances.workload) chartInstances.workload.destroy();

    chartInstances.workload = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.map(d => d.police_station.length > 18
                ? d.police_station.substring(0, 16) + '…' : d.police_station),
            datasets: [
                {
                    label: 'High Priority',
                    data: data.map(d => d.high_priority),
                    backgroundColor: COLORS.dangerLight,
                    borderColor: COLORS.danger,
                    borderWidth: 1,
                    borderRadius: 3,
                },
                {
                    label: 'Low Priority',
                    data: data.map(d => d.low_priority),
                    backgroundColor: COLORS.successLight,
                    borderColor: COLORS.success,
                    borderWidth: 1,
                    borderRadius: 3,
                }
            ]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'top' },
                title: { display: true, text: 'Top 20 Police Stations by Event Load', color: '#f1f5f9' }
            },
            scales: {
                x: {
                    stacked: true,
                    grid: { color: 'rgba(255,255,255,0.04)' }
                },
                y: {
                    stacked: true,
                    grid: { display: false },
                    ticks: { font: { size: 10 } }
                }
            }
        }
    });
}


// ============================================================================
// HISTORICAL TRENDS
// ============================================================================
async function loadTrends() {
    try {
        const cause = document.getElementById('trendCauseFilter')?.value || 'all';
        const corridor = document.getElementById('trendCorridorFilter')?.value || 'all';
        const zone = document.getElementById('trendZoneFilter')?.value || 'all';

        const url = `${API_BASE}/api/trends?cause=${cause}&corridor=${corridor}&zone=${zone}`;
        const res = await fetch(url);
        const data = await res.json();

        renderMonthlyChart(data.monthly || []);
        renderHourlyChart(data.hourly || []);
        renderDailyChart(data.daily || []);

    } catch (err) {
        console.error('Error loading trends:', err);
    }
}

function renderMonthlyChart(data) {
    const ctx = document.getElementById('monthlyChart');
    if (!ctx) return;
    if (chartInstances.monthly) chartInstances.monthly.destroy();

    chartInstances.monthly = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.map(d => d.year_month),
            datasets: [{
                label: 'Events',
                data: data.map(d => d.count),
                borderColor: COLORS.primary,
                backgroundColor: COLORS.primaryLight,
                fill: true,
                tension: 0.4,
                pointRadius: 3,
                pointHoverRadius: 6,
                pointBackgroundColor: COLORS.primary,
                borderWidth: 2,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
            },
            scales: {
                x: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { maxRotation: 45 } },
                y: { grid: { color: 'rgba(255,255,255,0.04)' }, beginAtZero: true }
            }
        }
    });
}

function renderHourlyChart(data) {
    const ctx = document.getElementById('hourlyChart');
    if (!ctx) return;
    if (chartInstances.hourly) chartInstances.hourly.destroy();

    const colors = data.map((_, i) => {
        const h = (i / 24) * 360;
        return `hsla(${h}, 70%, 60%, 0.7)`;
    });

    chartInstances.hourly = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.map(d => `${d.hour_of_day}:00`),
            datasets: [{
                label: 'Events',
                data: data.map(d => d.count),
                backgroundColor: colors,
                borderColor: colors.map(c => c.replace('0.7', '1')),
                borderWidth: 1,
                borderRadius: 4,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { grid: { display: false } },
                y: { grid: { color: 'rgba(255,255,255,0.04)' }, beginAtZero: true }
            }
        }
    });
}

function renderDailyChart(data) {
    const ctx = document.getElementById('dailyChart');
    if (!ctx) return;
    if (chartInstances.daily) chartInstances.daily.destroy();

    const dayNames = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];

    chartInstances.daily = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.map(d => dayNames[d.day_of_week] || d.day_of_week),
            datasets: [{
                label: 'Events',
                data: data.map(d => d.count),
                backgroundColor: data.map((d, i) =>
                    i >= 5 ? COLORS.warningLight : COLORS.primaryLight
                ),
                borderColor: data.map((d, i) =>
                    i >= 5 ? COLORS.warning : COLORS.primary
                ),
                borderWidth: 1.5,
                borderRadius: 6,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { grid: { display: false } },
                y: { grid: { color: 'rgba(255,255,255,0.04)' }, beginAtZero: true }
            }
        }
    });
}

async function loadCauseBreakdown() {
    try {
        const res = await fetch(`${API_BASE}/api/events/summary`);
        const data = await res.json();
        const causeDist = data.event_cause_dist || {};

        const sorted = Object.entries(causeDist).sort((a, b) => b[1] - a[1]);

        const ctx = document.getElementById('causeChart');
        if (!ctx) return;
        if (chartInstances.cause) chartInstances.cause.destroy();

        chartInstances.cause = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: sorted.map(([k]) => k),
                datasets: [{
                    data: sorted.map(([, v]) => v),
                    backgroundColor: COLORS.chartPalette.slice(0, sorted.length),
                    borderColor: 'rgba(10, 14, 26, 0.8)',
                    borderWidth: 2,
                    hoverOffset: 8,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '55%',
                plugins: {
                    legend: {
                        position: 'right',
                        labels: {
                            padding: 8,
                            font: { size: 10 },
                            boxWidth: 12,
                        }
                    }
                }
            }
        });
    } catch (err) {
        console.error('Error loading cause breakdown:', err);
    }
}


// ============================================================================
// METHODOLOGY METRICS
// ============================================================================
async function loadMethodologyMetrics() {
    try {
        const res = await fetch(`${API_BASE}/api/evaluation`);
        const data = await res.json();

        const container = document.getElementById('methodologyMetrics');
        let html = '';

        // Model A metrics
        if (data.model_a) {
            const report = data.model_a.classification_report || {};
            html += `
                <div class="metric-card">
                    <h5>📊 Model A — Priority Classifier (LightGBM)</h5>
                    <div class="metric-row">
                        <span class="metric-label">ROC-AUC</span>
                        <span class="metric-value">${data.model_a.roc_auc || '—'}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Train / Test Size</span>
                        <span class="metric-value">${data.model_a.train_size} / ${data.model_a.test_size}</span>
                    </div>
                    ${renderClassReport(report)}
                    <div style="margin-top: 8px; font-size: 0.7rem; color: var(--text-tertiary);">
                        Best params: ${JSON.stringify(data.model_a.best_params || {})}
                    </div>
                </div>
            `;
        }

        // Model B metrics
        if (data.model_b) {
            const report = data.model_b.classification_report || {};
            html += `
                <div class="metric-card">
                    <h5>🚧 Model B — Closure Classifier (LightGBM)</h5>
                    <div class="metric-row">
                        <span class="metric-label">ROC-AUC</span>
                        <span class="metric-value">${data.model_b.roc_auc || '—'}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Train / Test Size</span>
                        <span class="metric-value">${data.model_b.train_size} / ${data.model_b.test_size}</span>
                    </div>
                    ${renderClassReport(report)}
                    <div style="margin-top: 8px; font-size: 0.7rem; color: var(--text-tertiary);">
                        Best params: ${JSON.stringify(data.model_b.best_params || {})}
                    </div>
                </div>
            `;
        }

        // Model C metrics
        if (data.model_c) {
            html += `
                <div class="metric-card">
                    <h5>⏱️ Model C — Clearance Time Regressor (LightGBM)</h5>
                    <div class="metric-row">
                        <span class="metric-label">MAE</span>
                        <span class="metric-value">${data.model_c.test_mae_minutes || '—'} min</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">RMSE</span>
                        <span class="metric-value">${data.model_c.test_rmse_minutes || '—'} min</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Train / Test Size</span>
                        <span class="metric-value">${data.model_c.train_size} / ${data.model_c.test_size}</span>
                    </div>
                    <div style="margin-top: 8px; font-size: 0.7rem; color: var(--text-tertiary);">
                        Best params: ${JSON.stringify(data.model_c.best_params || {})}
                    </div>
                    ${data.model_c.error_by_event_cause ? renderErrorByCause(data.model_c.error_by_event_cause) : ''}
                </div>
            `;
        }

        container.innerHTML = html || '<div style="color: var(--text-tertiary);">Evaluation metrics not yet available. Run model training first.</div>';

    } catch (err) {
        console.error('Error loading methodology metrics:', err);
        document.getElementById('methodologyMetrics').innerHTML =
            '<div style="color: var(--text-tertiary);">Could not load evaluation metrics from server.</div>';
    }
}

function renderClassReport(report) {
    let html = '';
    const classes = Object.keys(report).filter(k =>
        !['accuracy', 'macro avg', 'weighted avg'].includes(k)
    );

    classes.forEach(cls => {
        const m = report[cls];
        if (m && typeof m === 'object') {
            html += `
                <div class="metric-row">
                    <span class="metric-label">${cls} — P/R/F1</span>
                    <span class="metric-value">
                        ${(m.precision || 0).toFixed(3)} / ${(m.recall || 0).toFixed(3)} / ${(m['f1-score'] || 0).toFixed(3)}
                    </span>
                </div>
            `;
        }
    });

    if (report['weighted avg']) {
        const wa = report['weighted avg'];
        html += `
            <div class="metric-row" style="border-top: 1px solid var(--border-subtle); padding-top: 4px; margin-top: 4px;">
                <span class="metric-label">Weighted Avg — P/R/F1</span>
                <span class="metric-value">
                    ${(wa.precision || 0).toFixed(3)} / ${(wa.recall || 0).toFixed(3)} / ${(wa['f1-score'] || 0).toFixed(3)}
                </span>
            </div>
        `;
    }

    return html;
}

function renderErrorByCause(errorData) {
    let html = '<div style="margin-top: 8px; border-top: 1px solid var(--border-subtle); padding-top: 8px;">';
    html += '<div style="font-size: 0.7rem; color: var(--text-tertiary); margin-bottom: 4px; font-weight: 600;">Error by Event Cause:</div>';

    const sorted = Object.entries(errorData).sort((a, b) => b[1].count - a[1].count);
    sorted.forEach(([cause, metrics]) => {
        html += `
            <div class="metric-row">
                <span class="metric-label">${cause} (n=${metrics.count})</span>
                <span class="metric-value">MAE: ${metrics.mae} min</span>
            </div>
        `;
    });

    html += '</div>';
    return html;
}
