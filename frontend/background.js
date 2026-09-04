/**
 * background.js — Wuthering Waves Ethereal Resonance Strings Engine
 *
 * Updates:
 * 1. High thread density: 15 threads distributed smoothly from baseY 0.05 to 0.95.
 * 2. Fully randomized and independent per-thread initialization (modes, phases, speeds, glow timers).
 * 3. Traveling comet-tail glow segment along each thread (sharp leading edge, soft smooth trailing fade,
 *    sub-path gradient stroke with luminous bloom).
 * 4. Optimized step density (step = 18) for silky 60 FPS performance.
 */

(function () {
  'use strict';

  function initWutheringBackground() {
    return; // Background canvas animation disabled
    let canvas = document.getElementById('wutheringCanvas');
    if (!canvas) {
      canvas = document.createElement('canvas');
      canvas.id = 'wutheringCanvas';
      document.body.prepend(canvas);
    }
    const ctx = canvas.getContext('2d');

    let width = 0;
    let height = 0;
    let dpr = window.devicePixelRatio || 1;

    function resize() {
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      width = window.innerWidth;
      height = window.innerHeight;
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      canvas.style.width = width + 'px';
      canvas.style.height = height + 'px';
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    resize();
    window.addEventListener('resize', resize, { passive: true });

    // Interactive subtle mouse repulsion
    let mouse = { x: -1000, y: -1000, targetX: -1000, targetY: -1000, active: false };
    window.addEventListener('mousemove', (e) => {
      mouse.targetX = e.clientX;
      mouse.targetY = e.clientY;
      mouse.active = true;
    }, { passive: true });

    window.addEventListener('mouseleave', () => {
      mouse.active = false;
    });

    // ── 15 Thread Base Configurations Across Full Viewport Height (0.05 - 0.95) ──
    const stringConfigs = [
      { baseY: 0.05, amp: 42, freq: 0.0016, harm: 0.0032, color: 'rgba(0, 174, 239, 0.14)', glow: 'rgba(0, 127, 196, 0.08)', width: 1.4 },
      { baseY: 0.12, amp: 55, freq: 0.0014, harm: 0.0028, color: 'rgba(0, 57, 93, 0.10)',  glow: 'rgba(0, 57, 93, 0.06)',   width: 1.3 },
      { baseY: 0.19, amp: 70, freq: 0.0012, harm: 0.0024, color: 'rgba(0, 174, 239, 0.18)', glow: 'rgba(0, 174, 239, 0.10)', width: 1.8 },
      { baseY: 0.27, amp: 85, freq: 0.0011, harm: 0.0022, color: 'rgba(0, 127, 196, 0.13)', glow: 'rgba(0, 127, 196, 0.08)', width: 1.5 },
      { baseY: 0.34, amp: 65, freq: 0.0013, harm: 0.0027, color: 'rgba(0, 174, 239, 0.12)', glow: 'rgba(0, 174, 239, 0.07)', width: 1.4 },
      { baseY: 0.41, amp: 95, freq: 0.0010, harm: 0.0020, color: 'rgba(0, 174, 239, 0.16)', glow: 'rgba(0, 174, 239, 0.09)', width: 1.9 },
      { baseY: 0.48, amp: 60, freq: 0.0009, harm: 0.0018, color: 'rgba(0, 57, 93, 0.08)',   glow: 'rgba(0, 57, 93, 0.05)',   width: 1.2 },
      { baseY: 0.55, amp: 90, freq: 0.0012, harm: 0.0025, color: 'rgba(0, 127, 196, 0.15)', glow: 'rgba(0, 127, 196, 0.08)', width: 1.6 },
      { baseY: 0.62, amp: 105, freq: 0.0010, harm: 0.0019, color: 'rgba(0, 174, 239, 0.20)', glow: 'rgba(0, 174, 239, 0.11)', width: 1.8 },
      { baseY: 0.69, amp: 75, freq: 0.0014, harm: 0.0029, color: 'rgba(0, 127, 196, 0.12)', glow: 'rgba(0, 127, 196, 0.07)', width: 1.4 },
      { baseY: 0.76, amp: 55, freq: 0.0009, harm: 0.0019, color: 'rgba(0, 57, 93, 0.09)',   glow: 'rgba(0, 57, 93, 0.05)',   width: 1.1 },
      { baseY: 0.82, amp: 68, freq: 0.0015, harm: 0.0031, color: 'rgba(0, 174, 239, 0.13)', glow: 'rgba(0, 127, 196, 0.07)', width: 1.3 },
      { baseY: 0.88, amp: 80, freq: 0.0013, harm: 0.0026, color: 'rgba(0, 127, 196, 0.16)', glow: 'rgba(0, 174, 239, 0.09)', width: 1.6 },
      { baseY: 0.93, amp: 50, freq: 0.0017, harm: 0.0035, color: 'rgba(0, 174, 239, 0.11)', glow: 'rgba(0, 174, 239, 0.06)', width: 1.3 },
      { baseY: 0.98, amp: 38, freq: 0.0018, harm: 0.0038, color: 'rgba(0, 57, 93, 0.08)',   glow: 'rgba(0, 57, 93, 0.04)',   width: 1.2 }
    ];

    const MODES = ['pulsating', 'current', 'drift'];

    // ── Independent, Fully-Randomized Thread Instantiation ──
    const threads = stringConfigs.map((cfg) => {
      // Truly random behavior mode per thread
      const mode = MODES[Math.floor(Math.random() * MODES.length)];

      return {
        ...cfg,
        mode: mode,
        // Independent, non-patterned wave parameters
        phase: Math.random() * Math.PI * 2,
        phaseShift: Math.random() * Math.PI * 2,
        speed: 0.00045 + Math.random() * 0.00045,
        vertDriftSpeed: 0.0002 + Math.random() * 0.00035,
        pulseSpeed: 0.015 + Math.random() * 0.025,            // For pulsating mode (2-4s)
        currentCycleSpeed: 0.0007 + Math.random() * 0.0010,   // For current-pushed mode (8-15s)

        // Traveling Comet-Tail Glow State (Sharp head + Soft trail)
        glowSweep: {
          active: false,
          progress: 0,
          // Traversal takes ~1.5 to 2.4s (60fps -> 90-144 frames)
          speed: 0.007 + Math.random() * 0.005,
          trailRatio: 0.18 + Math.random() * 0.07, // Trail covers 18% to 25% of total width
          nextTriggerTime: 30 + Math.floor(Math.random() * 600), // Random initial stagger
          cooldownMin: 240,
          cooldownRange: 480
        }
      };
    });

    // ── Floating Stardust Particles ──
    const particleCount = Math.max(25, Math.min(Math.floor(window.innerWidth / 24), 45));
    const particles = [];

    for (let i = 0; i < particleCount; i++) {
      particles.push({
        x: Math.random() * (window.innerWidth || 1200),
        y: Math.random() * (window.innerHeight || 800),
        radius: Math.random() * 1.9 + 0.6,
        vx: (Math.random() - 0.5) * 0.3,
        vy: -(Math.random() * 0.4 + 0.15),
        baseAlpha: Math.random() * 0.4 + 0.2,
        alpha: Math.random() * 0.4 + 0.2,
        maxAlpha: Math.random() * 0.35 + 0.55,
        pulseSpeed: Math.random() * 0.02 + 0.008,
        pulseOffset: Math.random() * Math.PI * 2,
        color: Math.random() > 0.3 ? 'rgba(0, 174, 239, 0.5)' : (Math.random() > 0.5 ? 'rgba(0, 127, 196, 0.4)' : 'rgba(0, 57, 93, 0.3)')
      });
    }

    let frameCount = 0;

    // ── Render Loop (60 FPS) ──
    function render() {
      ctx.clearRect(0, 0, width, height);
      frameCount++;

      // Mouse smooth lerp
      if (mouse.active) {
        mouse.x += (mouse.targetX - mouse.x) * 0.06;
        mouse.y += (mouse.targetY - mouse.y) * 0.06;
      }

      // Sampling step density (step = 18 delivers 60fps across 15 threads)
      const step = 18;
      const totalSteps = Math.ceil((width + 40) / step) + 2;

      for (let s = 0; s < threads.length; s++) {
        const th = threads[s];
        const sampledPoints = [];
        let curX = -20;

        let drawAlpha = 1.0;
        let drawWidth = th.width;
        let yCenter = height * th.baseY;

        // ── 1. Calculate Wave Points Based on Independent Mode ──
        if (th.mode === 'pulsating') {
          // MODE A: Mostly stationary wave position; breathes opacity and thickness
          const pulseSine = Math.sin(frameCount * th.pulseSpeed + th.phase);
          drawAlpha = 0.40 + 0.60 * (pulseSine * 0.5 + 0.5);
          drawWidth = th.width * (0.8 + 0.45 * (pulseSine * 0.5 + 0.5));

          const staticT = th.phase;
          for (let j = 0; j < totalSteps; j++) {
            let waveY =
              yCenter +
              Math.sin(curX * th.freq + staticT) * (th.amp * 0.85) +
              Math.cos(curX * th.harm + th.phaseShift) * (th.amp * 0.30);

            if (mouse.active) {
              const dx = curX - mouse.x;
              const dy = waveY - mouse.y;
              const dist = Math.sqrt(dx * dx + dy * dy);
              if (dist < 140) {
                const force = (1 - dist / 140) * 22;
                waveY += (dy / (dist || 1)) * force;
              }
            }

            sampledPoints.push({ x: curX, y: waveY });
            curX += step;
            if (curX > width + 20) break;
          }

        } else if (th.mode === 'current') {
          // MODE B: Current-pushed wave — surges and eases over 8-15s envelope
          const currentCycle = Math.sin(frameCount * th.currentCycleSpeed + th.phase);
          const dynamicAmp = th.amp * (0.60 + 0.70 * (currentCycle * 0.5 + 0.5));
          const dynamicFreq = th.freq * (0.85 + 0.30 * (currentCycle * 0.5 + 0.5));
          const t = frameCount * th.speed + th.phase;
          const vertOsc = Math.sin(frameCount * th.currentCycleSpeed * 0.7 + th.phaseShift) * (th.amp * 0.16);

          for (let j = 0; j < totalSteps; j++) {
            let waveY =
              (yCenter + vertOsc) +
              Math.sin(curX * dynamicFreq + t) * dynamicAmp +
              Math.cos(curX * th.harm - t * 0.75 + th.phaseShift) * (dynamicAmp * 0.38) +
              Math.sin(curX * 0.0006 + t * 0.4) * (dynamicAmp * 0.22);

            if (mouse.active) {
              const dx = curX - mouse.x;
              const dy = waveY - mouse.y;
              const dist = Math.sqrt(dx * dx + dy * dy);
              if (dist < 160) {
                const force = (1 - dist / 160) * 28;
                waveY += (dy / (dist || 1)) * force;
              }
            }

            sampledPoints.push({ x: curX, y: waveY });
            curX += step;
            if (curX > width + 20) break;
          }

        } else {
          // MODE C: Steady drift — continuous smooth wave motion
          const t = frameCount * th.speed + th.phase;
          const vertOsc = Math.sin(frameCount * 0.00035 + th.phaseShift) * (th.amp * 0.14);

          for (let j = 0; j < totalSteps; j++) {
            let waveY =
              (yCenter + vertOsc) +
              Math.sin(curX * th.freq + t) * th.amp +
              Math.cos(curX * th.harm - t * 0.70 + th.phaseShift) * (th.amp * 0.35);

            if (mouse.active) {
              const dx = curX - mouse.x;
              const dy = waveY - mouse.y;
              const dist = Math.sqrt(dx * dx + dy * dy);
              if (dist < 150) {
                const force = (1 - dist / 150) * 25;
                waveY += (dy / (dist || 1)) * force;
              }
            }

            sampledPoints.push({ x: curX, y: waveY });
            curX += step;
            if (curX > width + 20) break;
          }
        }

        // ── 2. Draw Base Thread ──
        ctx.save();
        ctx.globalAlpha = drawAlpha;
        ctx.beginPath();
        ctx.lineWidth = drawWidth;
        ctx.strokeStyle = th.color;
        ctx.shadowColor = th.glow;
        ctx.shadowBlur = 12;

        if (sampledPoints.length > 0) {
          ctx.moveTo(sampledPoints[0].x, sampledPoints[0].y);
          for (let k = 1; k < sampledPoints.length; k++) {
            ctx.lineTo(sampledPoints[k].x, sampledPoints[k].y);
          }
        }
        ctx.stroke();

        // Shimmering core highlight
        ctx.beginPath();
        ctx.lineWidth = Math.max(0.5, drawWidth * 0.36);
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.32)';
        ctx.shadowBlur = 4;
        ctx.shadowColor = '#ffffff';

        if (sampledPoints.length > 0) {
          ctx.moveTo(sampledPoints[0].x, sampledPoints[0].y);
          for (let k = 1; k < sampledPoints.length; k += 2) {
            ctx.lineTo(sampledPoints[k].x, sampledPoints[k].y);
          }
        }
        ctx.stroke();
        ctx.restore();

        // ── 3. Traveling Glow Segment with Asymmetric Fade Trail ──
        const gs = th.glowSweep;

        // Check trigger scheduler
        if (!gs.active && frameCount >= gs.nextTriggerTime) {
          gs.active = true;
          gs.progress = 0;
        }

        if (gs.active) {
          gs.progress += gs.speed;

          const trailPx = width * gs.trailRatio; // ~18-25% width trailing comet tail
          const headX = gs.progress * (width + trailPx * 1.5) - trailPx * 0.5;
          const tailX = headX - trailPx;

          // Overall fade envelope as it enters and leaves canvas edges
          let sweepAlpha = 1.0;
          if (gs.progress < 0.10) {
            sweepAlpha = gs.progress / 0.10;
          } else if (gs.progress > 0.90) {
            sweepAlpha = Math.max(0, (1.0 - gs.progress) / 0.10);
          }

          // Extract only the sampled points within the active comet window [tailX, headX]
          const cometPts = [];
          for (let p = 0; p < sampledPoints.length; p++) {
            const pt = sampledPoints[p];
            if (pt.x >= tailX - step && pt.x <= headX + step) {
              cometPts.push(pt);
            }
          }

          if (cometPts.length >= 2 && sweepAlpha > 0.02) {
            const firstPt = cometPts[0];
            const lastPt = cometPts[cometPts.length - 1];

            // Linear gradient along the comet path:
            // Tail (soft fade to 0) -> Mid trail -> Head (maximum blue brightness)
            const grad = ctx.createLinearGradient(firstPt.x, firstPt.y, lastPt.x, lastPt.y);
            grad.addColorStop(0.0, 'rgba(0, 174, 239, 0.0)');
            grad.addColorStop(0.4, `rgba(0, 174, 239, ${0.25 * sweepAlpha})`);
            grad.addColorStop(0.75, `rgba(0, 127, 196, ${0.65 * sweepAlpha})`);
            grad.addColorStop(0.95, `rgba(0, 174, 239, ${0.85 * sweepAlpha})`);
            grad.addColorStop(1.0, `rgba(180, 230, 250, ${1.0 * sweepAlpha})`);

            // Primary trailing glow stroke
            ctx.save();
            ctx.beginPath();
            ctx.lineWidth = th.width * 2.1;
            ctx.strokeStyle = grad;
            ctx.shadowColor = '#00AEEF';
            ctx.shadowBlur = 18;

            ctx.moveTo(cometPts[0].x, cometPts[0].y);
            for (let q = 1; q < cometPts.length; q++) {
              ctx.lineTo(cometPts[q].x, cometPts[q].y);
            }
            ctx.stroke();

            // Brilliant leading-edge core highlight (sharp front crest)
            const frontIndex = Math.max(0, cometPts.length - 5);
            ctx.beginPath();
            ctx.lineWidth = th.width * 1.2;
            ctx.strokeStyle = `rgba(200, 240, 255, ${0.85 * sweepAlpha})`;
            ctx.shadowColor = '#00AEEF';
            ctx.shadowBlur = 10;

            ctx.moveTo(cometPts[frontIndex].x, cometPts[frontIndex].y);
            for (let q = frontIndex + 1; q < cometPts.length; q++) {
              ctx.lineTo(cometPts[q].x, cometPts[q].y);
            }
            ctx.stroke();
            ctx.restore();
          }

          // Complete sweep -> schedule next randomized trigger
          if (gs.progress >= 1.0) {
            gs.active = false;
            gs.nextTriggerTime = frameCount + gs.cooldownMin + Math.floor(Math.random() * gs.cooldownRange);
          }
        }
      }

      // ── 4. Drifting Ethereal Stardust Particles ──
      for (let i = 0; i < particles.length; i++) {
        const p = particles[i];
        p.x += p.vx;
        p.y += p.vy;

        const pulse = Math.sin(frameCount * p.pulseSpeed + p.pulseOffset);
        p.alpha = p.baseAlpha + pulse * 0.25;
        if (p.alpha < 0.1) p.alpha = 0.1;
        if (p.alpha > p.maxAlpha) p.alpha = p.maxAlpha;

        if (p.y < -15) {
          p.y = height + 15;
          p.x = Math.random() * width;
        }
        if (p.x < -15) p.x = width + 15;
        if (p.x > width + 15) p.x = -15;

        ctx.save();
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
        ctx.fillStyle = p.color;
        ctx.globalAlpha = p.alpha;
        ctx.shadowColor = p.color;
        ctx.shadowBlur = 10;
        ctx.fill();
        ctx.restore();
      }

      requestAnimationFrame(render);
    }

    render();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initWutheringBackground);
  } else {
    initWutheringBackground();
  }
})();
