/* ═══════════════════════════════════════════════════════════
   SmartResist — Frontend Logic
   ═══════════════════════════════════════════════════════════ */

// ── 1. Floating Bio Canvas (DNA, Bacteria, Antibiotics) ──
(function initBioCanvas() {
  const canvas = document.getElementById('bio-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  let W, H, particles = [];

  function resize() {
    W = canvas.width = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }
  window.addEventListener('resize', resize);
  resize();

  // Particle types
  function createParticle() {
    const type = Math.random();
    if (type < 0.4) return makeDNA();
    if (type < 0.7) return makeBacterium();
    return makePill();
  }

  function makeDNA() {
    return {
      kind: 'dna', x: Math.random() * W, y: Math.random() * H,
      vx: (Math.random() - 0.5) * 0.3, vy: -0.15 - Math.random() * 0.25,
      phase: Math.random() * Math.PI * 2, size: 18 + Math.random() * 14,
      opacity: 0.15 + Math.random() * 0.2
    };
  }
  function makeBacterium() {
    return {
      kind: 'bact', x: Math.random() * W, y: Math.random() * H,
      vx: (Math.random() - 0.5) * 0.4, vy: (Math.random() - 0.5) * 0.3,
      size: 8 + Math.random() * 10, rot: Math.random() * Math.PI * 2,
      rotSpeed: (Math.random() - 0.5) * 0.01,
      opacity: 0.12 + Math.random() * 0.15
    };
  }
  function makePill() {
    return {
      kind: 'pill', x: Math.random() * W, y: Math.random() * H,
      vx: (Math.random() - 0.5) * 0.25, vy: -0.1 - Math.random() * 0.2,
      size: 10 + Math.random() * 8, rot: Math.random() * Math.PI,
      rotSpeed: (Math.random() - 0.5) * 0.008,
      opacity: 0.13 + Math.random() * 0.18
    };
  }

  // Spawn
  const count = Math.min(Math.floor(W * H / 25000), 45);
  for (let i = 0; i < count; i++) particles.push(createParticle());

  function drawDNA(p, t) {
    ctx.save();
    ctx.translate(p.x, p.y);
    ctx.globalAlpha = p.opacity;
    const rungs = 6;
    for (let i = 0; i < rungs; i++) {
      const yy = (i - rungs / 2) * 8;
      const wave = Math.sin(p.phase + t * 0.001 + i * 0.7) * p.size * 0.5;
      // Left strand dot
      ctx.beginPath();
      ctx.arc(-wave, yy, 3, 0, Math.PI * 2);
      ctx.fillStyle = '#6366f1';
      ctx.fill();
      // Right strand dot
      ctx.beginPath();
      ctx.arc(wave, yy, 3, 0, Math.PI * 2);
      ctx.fillStyle = '#06b6d4';
      ctx.fill();
      // Rung line
      ctx.beginPath();
      ctx.moveTo(-wave, yy);
      ctx.lineTo(wave, yy);
      ctx.strokeStyle = 'rgba(99,102,241,0.25)';
      ctx.lineWidth = 1;
      ctx.stroke();
    }
    ctx.restore();
  }

  function drawBacterium(p) {
    ctx.save();
    ctx.translate(p.x, p.y);
    ctx.rotate(p.rot);
    ctx.globalAlpha = p.opacity;
    // Body (oval)
    ctx.beginPath();
    ctx.ellipse(0, 0, p.size, p.size * 0.6, 0, 0, Math.PI * 2);
    ctx.strokeStyle = '#8b5cf6';
    ctx.lineWidth = 1.5;
    ctx.stroke();
    // Flagella
    ctx.beginPath();
    ctx.moveTo(-p.size, 0);
    ctx.quadraticCurveTo(-p.size - 10, -8, -p.size - 16, 2);
    ctx.strokeStyle = 'rgba(139,92,246,0.4)';
    ctx.lineWidth = 1;
    ctx.stroke();
    // Inner dot
    ctx.beginPath();
    ctx.arc(2, 0, 2.5, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(139,92,246,0.35)';
    ctx.fill();
    ctx.restore();
  }

  function drawPill(p) {
    ctx.save();
    ctx.translate(p.x, p.y);
    ctx.rotate(p.rot);
    ctx.globalAlpha = p.opacity;
    const w = p.size, h = p.size * 0.45, r = h;
    // Left half
    ctx.beginPath();
    ctx.moveTo(0, -h);
    ctx.lineTo(-w * 0.1, -h);
    ctx.arc(-w * 0.1, 0, h, -Math.PI / 2, Math.PI / 2, true);
    ctx.lineTo(0, h);
    ctx.closePath();
    ctx.fillStyle = 'rgba(16,185,129,0.5)';
    ctx.fill();
    // Right half
    ctx.beginPath();
    ctx.moveTo(0, -h);
    ctx.lineTo(w * 0.1, -h);
    ctx.arc(w * 0.1, 0, h, -Math.PI / 2, Math.PI / 2, false);
    ctx.lineTo(0, h);
    ctx.closePath();
    ctx.fillStyle = 'rgba(6,182,212,0.45)';
    ctx.fill();
    ctx.restore();
  }

  function animate(t) {
    ctx.clearRect(0, 0, W, H);
    particles.forEach(p => {
      p.x += p.vx;
      p.y += p.vy;
      if (p.rot !== undefined) p.rot += p.rotSpeed;
      // Wrap
      if (p.y < -60) { p.y = H + 60; p.x = Math.random() * W; }
      if (p.y > H + 60) { p.y = -60; p.x = Math.random() * W; }
      if (p.x < -60) p.x = W + 60;
      if (p.x > W + 60) p.x = -60;

      if (p.kind === 'dna') drawDNA(p, t);
      else if (p.kind === 'bact') drawBacterium(p);
      else drawPill(p);
    });
    requestAnimationFrame(animate);
  }
  requestAnimationFrame(animate);
})();


// ── 2. DNA Helix in Hero ──
(function buildHeroDNA() {
  const helix = document.getElementById('dna-helix');
  if (!helix) return;
  const rungs = 16;
  for (let i = 0; i < rungs; i++) {
    const rung = document.createElement('div');
    rung.className = 'dna-rung';
    rung.style.top = (i / rungs * 100) + '%';
    rung.style.animationDelay = (i * 0.18) + 's';
    rung.style.background = `linear-gradient(90deg, rgba(99,102,241,0.3), rgba(6,182,212,0.3))`;
    helix.appendChild(rung);
  }
})();


// ── 3. Navbar Scroll ──
window.addEventListener('scroll', () => {
  document.getElementById('navbar').classList.toggle('scrolled', window.scrollY > 40);
});


// ── 4. Scroll Reveal ──
const reveals = document.querySelectorAll('.reveal');
const revealObserver = new IntersectionObserver((entries) => {
  entries.forEach(e => { if (e.isIntersecting) e.target.classList.add('visible'); });
}, { threshold: 0.15 });
reveals.forEach(el => revealObserver.observe(el));


// ── 5. Counter Animation ──
let countersAnimated = false;
function animateCounters() {
  if (countersAnimated) return;
  countersAnimated = true;
  document.querySelectorAll('.stat-number[data-target]').forEach(el => {
    const target = parseInt(el.dataset.target);
    const duration = 2000;
    const start = performance.now();
    function tick(now) {
      const elapsed = now - start;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      el.textContent = Math.floor(target * eased).toLocaleString();
      if (progress < 1) requestAnimationFrame(tick);
      else el.textContent = target.toLocaleString();
    }
    requestAnimationFrame(tick);
  });
}

const statsSection = document.getElementById('stats');
if (statsSection) {
  new IntersectionObserver((entries) => {
    if (entries[0].isIntersecting) animateCounters();
  }, { threshold: 0.3 }).observe(statsSection);
}


// ── 6. Bar Chart Reveal ──
const barFills = document.querySelectorAll('.bar-fill[data-width]');
const barObserver = new IntersectionObserver((entries) => {
  entries.forEach(e => {
    if (e.isIntersecting) e.target.style.width = e.target.dataset.width;
  });
}, { threshold: 0.3 });
barFills.forEach(el => barObserver.observe(el));


// ── 7. Load live stats from API ──
fetch('/api/status')
  .then(r => r.json())
  .then(data => {
    if (data.dataset_stats) {
      const s = data.dataset_stats;
      const samplesEl = document.getElementById('stat-samples');
      const genesEl = document.getElementById('stat-genes');
      const drugsEl = document.getElementById('stat-drugs');
      const accEl = document.getElementById('stat-accuracy');
      if (samplesEl) samplesEl.dataset.target = s.total_samples || 147237;
      if (genesEl) genesEl.dataset.target = s.total_genes || 1034;
      if (drugsEl) drugsEl.dataset.target = s.total_drugs || 139;
      if (accEl) {
        const acc = s.model_accuracy;
        if (acc && acc !== 'N/A' && acc !== 'Awaiting Training...') {
          accEl.textContent = (typeof acc === 'number' ? (acc * 100).toFixed(1) : acc) + '%';
        } else {
          accEl.textContent = '—';
        }
      }
    }
  })
  .catch(() => {});


// ── 8. Autocomplete (Gene) ──
const geneInput = document.getElementById('gene-input');
const acList = document.getElementById('autocomplete-list');
let acTimeout;

if (geneInput && acList) {
  geneInput.addEventListener('input', () => {
    clearTimeout(acTimeout);
    const q = geneInput.value.trim();
    if (q.length < 1) { acList.classList.remove('active'); acList.innerHTML = ''; return; }
    acTimeout = setTimeout(() => {
      fetch(`/api/autocomplete?q=${encodeURIComponent(q)}&type=gene`)
        .then(r => r.json())
        .then(items => {
          acList.innerHTML = '';
          if (!items.length) { acList.classList.remove('active'); return; }
          items.forEach(g => {
            const div = document.createElement('div');
            div.textContent = g;
            div.addEventListener('click', () => {
              geneInput.value = g;
              acList.classList.remove('active');
              acList.innerHTML = '';
            });
            acList.appendChild(div);
          });
          acList.classList.add('active');
        })
        .catch(() => acList.classList.remove('active'));
    }, 200);
  });

  document.addEventListener('click', e => {
    if (!geneInput.contains(e.target) && !acList.contains(e.target)) {
      acList.classList.remove('active');
    }
  });
}


// ── 9. Allergy Tags ──
const allergyInput = document.getElementById('allergy-input');
const allergyTags = document.getElementById('allergy-tags');
const allergyAC = document.getElementById('allergy-autocomplete');
let allergies = [];
let allergyTimeout;

if (allergyInput) {
  allergyInput.addEventListener('keydown', e => {
    if (e.key === 'Enter') {
      e.preventDefault();
      const val = allergyInput.value.trim();
      if (val && !allergies.includes(val)) {
        allergies.push(val);
        renderAllergyTags();
      }
      allergyInput.value = '';
      if (allergyAC) allergyAC.classList.remove('active');
    }
  });

  allergyInput.addEventListener('input', () => {
    clearTimeout(allergyTimeout);
    const q = allergyInput.value.trim();
    if (q.length < 1 || !allergyAC) { if (allergyAC) allergyAC.classList.remove('active'); return; }
    allergyTimeout = setTimeout(() => {
      fetch(`/api/autocomplete?q=${encodeURIComponent(q)}&type=drug`)
        .then(r => r.json())
        .then(items => {
          allergyAC.innerHTML = '';
          if (!items.length) { allergyAC.classList.remove('active'); return; }
          items.forEach(d => {
            const div = document.createElement('div');
            div.textContent = d;
            div.addEventListener('click', () => {
              if (!allergies.includes(d)) { allergies.push(d); renderAllergyTags(); }
              allergyInput.value = '';
              allergyAC.classList.remove('active');
            });
            allergyAC.appendChild(div);
          });
          allergyAC.classList.add('active');
        })
        .catch(() => {});
    }, 200);
  });
}

function renderAllergyTags() {
  if (!allergyTags) return;
  allergyTags.innerHTML = '';
  allergies.forEach((a, i) => {
    const tag = document.createElement('span');
    tag.className = 'allergy-tag';
    tag.innerHTML = `${a} <span class="remove-tag" data-idx="${i}">&times;</span>`;
    allergyTags.appendChild(tag);
  });
  allergyTags.querySelectorAll('.remove-tag').forEach(btn => {
    btn.addEventListener('click', () => {
      allergies.splice(parseInt(btn.dataset.idx), 1);
      renderAllergyTags();
    });
  });
}


// ── 10. Prediction ──
let predChart = null;
const predictBtn = document.getElementById('predict-btn');

if (predictBtn) {
  predictBtn.addEventListener('click', runPrediction);
}

function runPrediction() {
  const gene = geneInput ? geneInput.value.trim() : '';
  const rs = parseFloat(document.getElementById('start-input').value) || 0;
  const re = parseFloat(document.getElementById('end-input').value) || 1000;

  if (!gene) { showError('⚠ Please enter a gene symbol.'); return; }

  hideAll();
  document.getElementById('loader').classList.add('active');
  predictBtn.disabled = true;

  fetch('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      genes: [{ gene, region_start: rs, region_end: re }],
      allergies: allergies
    })
  })
    .then(r => r.json())
    .then(data => {
      document.getElementById('loader').classList.remove('active');
      predictBtn.disabled = false;

      const geneResult = data.genes && data.genes[0];
      if (!geneResult || geneResult.error) {
        showError('⚠ ' + (geneResult ? geneResult.error : 'No response from server') +
          (geneResult && geneResult.reason ? ` — ${geneResult.reason}` : ''));
        return;
      }

      const recs = geneResult.recommendations || [];
      if (!recs.length) { showError('⚠ No recommendations available.'); return; }

      // Show results
      document.getElementById('results-content').classList.add('active');
      document.getElementById('results-placeholder').style.display = 'none';

      // Top recommendation
      document.getElementById('top-drug').textContent = recs[0].drug;
      document.getElementById('top-score').textContent = recs[0].susceptibility.toFixed(1) + '%';

      // Table
      const tbody = document.getElementById('results-body');
      tbody.innerHTML = '';
      recs.forEach(r => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td style="font-weight:600;">${r.drug}</td>
          <td><span class="badge-${r.label.toLowerCase()}">${r.label === 'S' ? 'Susceptible' : 'Resistant'}</span></td>
          <td>${r.susceptibility.toFixed(1)}%</td>
          <td>${r.support}</td>`;
        tbody.appendChild(tr);
      });

      // Chart
      buildChart(recs);
    })
    .catch(err => {
      document.getElementById('loader').classList.remove('active');
      predictBtn.disabled = false;
      showError('⚠ Network error — is the Flask server running?');
    });
}

function buildChart(recs) {
  const ctx = document.getElementById('predictionChart').getContext('2d');
  if (predChart) predChart.destroy();

  const labels = recs.map(r => r.drug);
  const values = recs.map(r => r.susceptibility);
  const colors = recs.map(r =>
    r.label === 'S' ? 'rgba(16,185,129,0.75)' : 'rgba(239,68,68,0.65)'
  );
  const borderColors = recs.map(r =>
    r.label === 'S' ? '#059669' : '#dc2626'
  );

  predChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Susceptibility (%)',
        data: values,
        backgroundColor: colors,
        borderColor: borderColors,
        borderWidth: 1.5,
        borderRadius: 8,
        barPercentage: 0.6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 1000, easing: 'easeOutQuart' },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#1e293b',
          titleFont: { family: 'Inter', size: 13 },
          bodyFont: { family: 'JetBrains Mono', size: 12 },
          cornerRadius: 8,
          padding: 12,
          callbacks: {
            label: ctx => `${ctx.parsed.y.toFixed(1)}% susceptibility`
          }
        }
      },
      scales: {
        y: {
          beginAtZero: true, max: 100,
          grid: { color: 'rgba(0,0,0,0.04)' },
          ticks: { font: { family: 'Inter', size: 11 }, color: '#94a3b8',
            callback: v => v + '%' }
        },
        x: {
          grid: { display: false },
          ticks: { font: { family: 'Inter', size: 11 }, color: '#64748b' }
        }
      }
    }
  });
}

function showError(msg) {
  hideAll();
  const el = document.getElementById('error-msg');
  el.textContent = msg;
  el.style.display = 'flex';
}

function hideAll() {
  document.getElementById('loader').classList.remove('active');
  document.getElementById('results-content').classList.remove('active');
  document.getElementById('error-msg').style.display = 'none';
  document.getElementById('results-placeholder').style.display = 'none';
}
