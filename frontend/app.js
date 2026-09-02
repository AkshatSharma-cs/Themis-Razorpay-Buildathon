/**
 * app.js — Themis Modern Multi-Page Experience & Live Risk Engine Integration
 */
(function () {
    'use strict';

    // -------------------------------------------------------------------------
    // 1. STATE & CONSTANTS
    // -------------------------------------------------------------------------
    const API_BASE = 'https://themis-razorpay-buildathon.onrender.com';

    const PRESETS = {
        mule: {
            payer: 'anand.sharma@okaxis',
            payee: 'fast.cashback.991@ybl',
            amount: 48500,
            category: 'other',
            call_overlap: true,
            screen_share: true,
            otp_share: false
        },
        remote: {
            payer: 'sunita.patel@okhdfcbank',
            payee: 'discom.billdesk.urgent@paytm',
            amount: 12400,
            category: 'utility_bill',
            call_overlap: true,
            screen_share: false,
            otp_share: true
        },
        grocery: {
            payer: 'rohit.kumar@icici',
            payee: 'kirana.store.blr@upi',
            amount: 650,
            category: 'groceries',
            call_overlap: false,
            screen_share: false,
            otp_share: false
        },
        flight: {
            payer: 'priya.nair@sbi',
            payee: 'makemytrip.merchant@hdfcbank',
            amount: 34200,
            category: 'travel',
            call_overlap: false,
            screen_share: false,
            otp_share: false
        }
    };

    // -------------------------------------------------------------------------
    // 2. PAGE NAVIGATION ROUTER
    // -------------------------------------------------------------------------
    window.switchPage = function (pageId) {
        document.querySelectorAll('.page-view').forEach(page => {
            page.classList.remove('active');
        });
        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.remove('active');
        });
        const targetPage = document.getElementById(`page-${pageId}`);
        const targetNav = document.getElementById(`nav-${pageId}`);
        if (targetPage) {
            targetPage.classList.add('active');
            window.scrollTo({ top: 0, behavior: 'smooth' });
        }
        if (targetNav) {
            targetNav.classList.add('active');
        }
    };

    // -------------------------------------------------------------------------
    // 3. INTERACTIVE PARTICLE CANVAS BACKGROUND
    // -------------------------------------------------------------------------
    function initParticleCanvas() {
        const canvas = document.getElementById('bgCanvas');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        let width = canvas.width = window.innerWidth;
        let height = canvas.height = window.innerHeight;
        window.addEventListener('resize', () => {
            width = canvas.width = window.innerWidth;
            height = canvas.height = window.innerHeight;
        });
        const particles = [];
        const particleCount = Math.min(Math.floor(window.innerWidth / 20), 45);
        for (let i = 0; i < particleCount; i++) {
            particles.push({
                x: Math.random() * width,
                y: Math.random() * height,
                vx: (Math.random() - 0.5) * 0.4,
                vy: (Math.random() - 0.5) * 0.4,
                radius: Math.random() * 2 + 1,
                alpha: Math.random() * 0.4 + 0.1
            });
        }
        function render() {
            ctx.clearRect(0, 0, width, height);
            for (let i = 0; i < particles.length; i++) {
                const p = particles[i];
                p.x += p.vx;
                p.y += p.vy;
                if (p.x < 0) p.x = width;
                if (p.x > width) p.x = 0;
                if (p.y < 0) p.y = height;
                if (p.y > height) p.y = 0;
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(14, 165, 233, ${p.alpha})`;
                ctx.fill();
                for (let j = i + 1; j < particles.length; j++) {
                    const p2 = particles[j];
                    const dist = Math.hypot(p.x - p2.x, p.y - p2.y);
                    if (dist < 140) {
                        ctx.beginPath();
                        ctx.moveTo(p.x, p.y);
                        ctx.lineTo(p2.x, p2.y);
                        ctx.strokeStyle = `rgba(37, 99, 235, ${0.15 * (1 - dist / 140)})`;
                        ctx.lineWidth = 1;
                        ctx.stroke();
                    }
                }
            }
            requestAnimationFrame(render);
        }
        render();
    }

    // -------------------------------------------------------------------------
    // 4. PRESETS & FORM HANDLING
    // -------------------------------------------------------------------------
    window.loadPreset = function (presetKey) {
        const p = PRESETS[presetKey];
        if (!p) return;
        document.querySelectorAll('.preset-chip').forEach(btn => btn.classList.remove('active'));
        if (event && event.currentTarget) {
            event.currentTarget.classList.add('active');
        }
        document.getElementById('form_payer').value = p.payer;
        document.getElementById('form_payee').value = p.payee;
        document.getElementById('form_amount').value = p.amount;
        document.getElementById('form_category').value = p.category;
        document.getElementById('form_call_overlap').checked = p.call_overlap;
        document.getElementById('form_screen_share').checked = p.screen_share;
        document.getElementById('form_otp_share').checked = p.otp_share;
    };

    function getPayload() {
        return {
            payer_vpa: document.getElementById('form_payer').value.trim(),
            payee_vpa: document.getElementById('form_payee').value.trim(),
            amount: parseFloat(document.getElementById('form_amount').value) || 0,
            shopping_category: document.getElementById('form_category').value,
            instrument_type: 'upi_p2p',
            call_overlap_flag: document.getElementById('form_call_overlap').checked,
            screen_share_flag: document.getElementById('form_screen_share').checked,
            otp_share_flag: document.getElementById('form_otp_share').checked
        };
    }

    // -------------------------------------------------------------------------
    // 5. LIVE DECISION & SCORING INTEGRATION
    // -------------------------------------------------------------------------
    async function apiRequest(endpoint, body, method = 'POST') {
        const opts = {
            method,
            headers: { 'Content-Type': 'application/json' }
        };
        if (method !== 'GET') {
            opts.body = JSON.stringify(body);
        }
        const res = await fetch(`${API_BASE}${endpoint}`, opts);
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `HTTP ${res.status}`);
        }
        return await res.json();
    }

    window.executeDecision = async function () {
        const payload = getPayload();
        const btn = document.getElementById('btnExecuteDecision');
        const probText = document.getElementById('probScoreText');
        const fillBar = document.getElementById('gaugeFill');
        const badge = document.getElementById('verdictBadge');
        const reasonCopy = document.getElementById('verdictReasonCopy');
        const thText = document.getElementById('thresholdText');

        btn.disabled = true;
        probText.textContent = '...';

        try {
            const data = await apiRequest('/decision', payload);
            const probPct = (data.probability * 100).toFixed(1);
            probText.textContent = `${probPct}%`;
            thText.textContent = data.threshold ? data.threshold.toFixed(3) : '—';
            fillBar.style.width = `${Math.min(Math.max(data.probability * 100, 0), 100)}%`;

            if (data.action_type === 'cooling_off') {
                fillBar.className = 'gauge-bar-fill high-risk';
                badge.className = 'verdict-pill-badge badge-cooling';
                badge.innerHTML = `<i class="fas fa-clock"></i> <span>${data.duration_hours}H Cooling-Off Delay</span>`;
            } else if (data.action_type === 'advisory_only') {
                fillBar.className = 'gauge-bar-fill high-risk';
                badge.className = 'verdict-pill-badge badge-advisory';
                badge.innerHTML = '<i class="fas fa-triangle-exclamation"></i> <span>Advisory Notice</span>';
            } else {
                fillBar.className = 'gauge-bar-fill';
                badge.className = 'verdict-pill-badge badge-safe';
                badge.innerHTML = '<i class="fas fa-check-circle"></i> <span>Cleared (Nominal Flow)</span>';
            }

            reasonCopy.textContent = data.narration || data.reason || 'Decision computed by LightGBM model.';
            appendTransactionRow(data.tx_id, payload.amount, payload.payee_vpa, data.probability, data.action_type);

        } catch (err) {
            probText.textContent = 'ERROR';
            fillBar.style.width = '0%';
            fillBar.className = 'gauge-bar-fill';
            badge.className = 'verdict-pill-badge badge-advisory';
            badge.innerHTML = '<i class="fas fa-triangle-exclamation"></i> <span>Backend Unreachable</span>';
            reasonCopy.textContent = `Could not reach the model: ${err.message}. Check that the backend is awake and the API URL is correct.`;
            thText.textContent = '—';
        } finally {
            btn.disabled = false;
        }
    };

    window.executeScoreOnly = async function () {
        const payload = getPayload();
        const probText = document.getElementById('probScoreText');
        const fillBar = document.getElementById('gaugeFill');
        const badge = document.getElementById('verdictBadge');
        const reasonCopy = document.getElementById('verdictReasonCopy');

        probText.textContent = '...';

        try {
            const data = await apiRequest('/score', payload);
            const probPct = (data.probability * 100).toFixed(1);
            probText.textContent = `${probPct}%`;
            fillBar.style.width = `${Math.min(Math.max(data.probability * 100, 0), 100)}%`;
            fillBar.className = 'gauge-bar-fill';
            badge.className = 'verdict-pill-badge badge-safe';
            badge.innerHTML = '<i class="fas fa-calculator"></i> <span>Score Computed</span>';
            reasonCopy.textContent = `Raw LightGBM model score: ${(data.probability * 100).toFixed(2)}%`;
        } catch (err) {
            probText.textContent = 'ERROR';
            fillBar.style.width = '0%';
            badge.className = 'verdict-pill-badge badge-advisory';
            badge.innerHTML = '<i class="fas fa-triangle-exclamation"></i> <span>Backend Unreachable</span>';
            reasonCopy.textContent = `Could not reach the model: ${err.message}.`;
        }
    };

    function appendTransactionRow(txId, amount, payee, prob, actionType) {
        const tbody = document.getElementById('txnTableBody');
        if (!tbody) return;
        const tr = document.createElement('tr');
        let statusHtml = '<span class="status-tag safe">Safe</span>';
        let actionIcon = '<i class="fas fa-check" style="color: var(--success);"></i>';
        if (actionType === 'cooling_off') {
            statusHtml = '<span class="status-tag danger">Cooling-Off</span>';
            actionIcon = '<i class="fas fa-clock" style="color: var(--accent);"></i>';
        } else if (actionType === 'advisory_only') {
            statusHtml = '<span class="status-tag flagged">Advisory</span>';
            actionIcon = '<i class="fas fa-triangle-exclamation" style="color: var(--warning);"></i>';
        }
        tr.innerHTML = `
            <td class="font-mono-val">#${txId.slice(0, 10)}</td>
            <td class="font-mono-val">₹${amount.toLocaleString('en-IN')}</td>
            <td>${payee}</td>
            <td class="font-mono-val">${prob.toFixed(2)}</td>
            <td>${statusHtml}</td>
            <td>${actionIcon}</td>
        `;
        tbody.insertBefore(tr, tbody.firstChild);
    }

    window.verifyAuditChain = async function () {
        try {
            const res = await fetch(`${API_BASE}/audit/verify/chain`);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();
            if (data.ok) {
                alert(`✓ Tamper-Evident Audit Verification\n\nRows Checked: ${data.rows_checked}\nIntegrity: VALID — chain intact`);
            } else {
                alert(`✗ Tamper Detected\n\nRows Checked: ${data.rows_checked}\nFirst Broken Row: #${data.first_broken_row_id}\nReason: ${data.reason}`);
            }
        } catch (err) {
            alert(`Could not verify audit chain — backend unreachable.\n\nError: ${err.message}`);
        }
    };

    // Health check polling
    async function checkHealth() {
        const healthDot = document.getElementById('healthDot');
        const healthText = document.getElementById('healthText');
        try {
            const res = await fetch(`${API_BASE}/health`);
            if (res.ok) {
                const data = await res.json();
                healthDot.style.background = data.is_dummy_model ? 'var(--warning)' : 'var(--success)';
                healthText.textContent = data.is_dummy_model
                    ? 'DUMMY MODEL — CHECK DEPLOY'
                    : `MODEL: ${data.model_version.toUpperCase()} · ${data.n_features} FEATURES`;
            } else {
                throw new Error(`HTTP ${res.status}`);
            }
        } catch (e) {
            healthDot.style.background = 'var(--secondary)';
            healthText.textContent = 'BACKEND UNREACHABLE';
        }
    }

    // -------------------------------------------------------------------------
    // 6. INIT
    // -------------------------------------------------------------------------
    function init() {
        initParticleCanvas();
        checkHealth();
        setInterval(checkHealth, 15000);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();