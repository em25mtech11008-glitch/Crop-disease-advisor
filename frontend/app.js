/* ═══════════════════════════════════════════════════════════════════════════
   CROP DISEASE ADVISOR — Frontend Application
   Handles upload, API calls, results rendering, report generation, PDF export
   ═══════════════════════════════════════════════════════════════════════════ */

// ── DOM References ───────────────────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const els = {
  // Navbar
  apiStatus:     $('#api-status'),
  apiStatusText: $('#api-status-text'),

  // Upload
  dropZone:       $('#drop-zone'),
  dropContent:    $('#drop-zone-content'),
  fileInput:      $('#file-input'),
  previewContainer: $('#preview-container'),
  previewImage:   $('#preview-image'),
  clearImageBtn:  $('#clear-image-btn'),

  // Config
  regionSelect:  $('#region-select'),
  seasonSelect:  $('#season-select'),
  analyzeBtn:    $('#analyze-btn'),
  analyzeBtnText: $('#analyze-btn-text'),
  analyzeBtnLoading: $('#analyze-btn-loading'),

  // Severity
  sevAutoBtn:     $('#severity-auto-btn'),
  sevManualBtn:   $('#severity-manual-btn'),
  sevAutoHint:    $('#severity-auto-hint'),
  sevManualCtrl:  $('#severity-manual-controls'),
  sevSlider:      $('#severity-slider'),
  sevValue:       $('#severity-value'),

  // Sections
  uploadSection:   $('#upload-section'),
  loadingSection:  $('#loading-section'),
  resultsSection:  $('#results-section'),

  // Loading steps
  stepVision: $('#step-vision'),
  stepReport: $('#step-report'),

  // Stats
  statPlant:      $('#stat-plant'),
  statDisease:    $('#stat-disease'),
  statConfidence: $('#stat-confidence'),
  statSeverity:   $('#stat-severity'),
  statUrgency:    $('#stat-urgency'),

  // Results
  resultImage:       $('#result-image'),
  top5Bars:          $('#top5-bars'),
  diagnosisDisease:  $('#diagnosis-disease'),
  diagnosisCrop:     $('#diagnosis-crop'),
  diagnosisDesc:     $('#diagnosis-description'),
  diagnosisBadges:   $('#diagnosis-badges'),

  // Tabs
  tabBtns:  $$('.tab-btn'),
  tabOrganic:    $('#tab-organic'),
  tabChemical:   $('#tab-chemical'),
  tabPreventive: $('#tab-preventive'),
  tabAdvisory:   $('#tab-advisory'),

  // Report
  reportContainer: $('#report-container'),

  // Toast
  toast: $('#toast'),
};

// ── State ────────────────────────────────────────────────────────────────────
let state = {
  file: null,
  imageDataUrl: null,
  severityMode: 'auto',    // 'auto' | 'manual'
  manualSeverity: 30,
  result: null,
  isLoading: false,
};

// ── Disease Descriptions (brief, for common PlantVillage classes) ─────────
const DISEASE_DESCRIPTIONS = {
  'Bacterial_spot':    'A bacterial infection causing dark, water-soaked spots on leaves, stems, and fruits. Common in warm, humid conditions.',
  'Early_blight':      'Fungal disease producing concentric ring-shaped brown spots on lower leaves. Thrives in warm, moist weather.',
  'Late_blight':       'Aggressive fungal disease causing large, dark, water-soaked lesions. Spreads rapidly in cool, wet conditions.',
  'Leaf_Mold':         'Fungal infection forming yellow patches on upper leaf surface and olive-green mold underneath.',
  'Septoria_leaf_spot':'Fungal disease producing numerous small, circular spots with gray centers and dark borders.',
  'Spider_mites':      'Tiny arachnids that feed on leaf cells causing stippled, bronzed appearance and fine webbing.',
  'Target_Spot':       'Fungal disease causing brown spots with concentric rings resembling a target pattern.',
  'Yellow_Leaf_Curl_Virus': 'Viral disease transmitted by whiteflies causing severe leaf curling, yellowing, and stunted growth.',
  'Mosaic_virus':      'Viral infection causing mottled light and dark green patterns on leaves with distortion.',
  'healthy':           'No disease detected. The plant appears to be in healthy condition with normal growth.',
  'Powdery_mildew':    'Fungal disease forming white, powdery coating on leaves and stems. Favored by dry conditions.',
  'Black_rot':         'Fungal disease causing V-shaped brown lesions from leaf margins. Can affect fruits and stems.',
  'Cercospora_leaf_spot': 'Fungal disease producing small, circular spots with light gray centers surrounded by dark borders.',
  'Common_rust':       'Fungal disease forming small, reddish-brown pustules on both leaf surfaces.',
  'Northern_Leaf_Blight': 'Fungal disease causing long, elliptical gray-green lesions on leaves.',
  'Haunglongbing':     'Bacterial disease causing asymmetric yellowing of leaves and misshapen, bitter fruits.',
  'Cedar_apple_rust':  'Fungal disease producing bright orange-yellow spots on upper leaf surfaces.',
  'Scab':              'Fungal disease causing dark, scabby lesions on fruits, leaves, and twigs.',
  'Esca':              'A complex of fungal diseases causing tiger-stripe patterns on leaves and wood decay.',
  'Leaf_blight':       'Fungal disease causing rapid browning and death of leaf tissue.',
  'Isariopsis_Leaf_Spot': 'Fungal infection producing dark brown angular spots on leaves.',
};

function getDescription(diseaseName) {
  const lower = diseaseName.toLowerCase();
  if (lower.includes('healthy')) return DISEASE_DESCRIPTIONS['healthy'];
  for (const [key, desc] of Object.entries(DISEASE_DESCRIPTIONS)) {
    if (lower.includes(key.toLowerCase())) return desc;
  }
  return 'A plant disease detected by the AI model. Consult the treatment plan below for recommended actions.';
}


// ═══════════════════════════════════════════════════════════════════════════
// API
// ═══════════════════════════════════════════════════════════════════════════

async function checkApiHealth() {
  try {
    const resp = await fetch('/health', { signal: AbortSignal.timeout(5000) });
    if (resp.ok) {
      els.apiStatus.className = 'status-dot status-dot--online';
      els.apiStatusText.textContent = 'API Online';
      return true;
    }
  } catch {}
  els.apiStatus.className = 'status-dot status-dot--offline';
  els.apiStatusText.textContent = 'API Offline';
  return false;
}

async function predictDisease(file, region, season, severity) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('region', region);
  formData.append('season', season);
  formData.append('severity', severity);

  const resp = await fetch('/predict', {
    method: 'POST',
    body: formData,
  });

  if (!resp.ok) {
    const errText = await resp.text();
    throw new Error(`API Error ${resp.status}: ${errText}`);
  }

  return resp.json();
}


// ═══════════════════════════════════════════════════════════════════════════
// IMAGE UPLOAD
// ═══════════════════════════════════════════════════════════════════════════

function setupUpload() {
  const zone = els.dropZone;

  // Click to select
  zone.addEventListener('click', (e) => {
    if (e.target === els.clearImageBtn || els.clearImageBtn.contains(e.target)) return;
    els.fileInput.click();
  });

  // File selected via input
  els.fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) handleFile(e.target.files[0]);
  });

  // Drag events
  zone.addEventListener('dragover', (e) => {
    e.preventDefault();
    zone.classList.add('drop-zone--active');
  });

  zone.addEventListener('dragleave', () => {
    zone.classList.remove('drop-zone--active');
  });

  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.classList.remove('drop-zone--active');
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
  });

  // Clear image
  els.clearImageBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    clearImage();
  });
}

function handleFile(file) {
  // Validate
  const validTypes = ['image/jpeg', 'image/png', 'image/webp'];
  if (!validTypes.includes(file.type)) {
    showToast('⚠ Please upload a JPG, PNG, or WebP image.');
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    showToast('⚠ Image must be under 10 MB.');
    return;
  }

  state.file = file;

  // Preview
  const reader = new FileReader();
  reader.onload = (e) => {
    state.imageDataUrl = e.target.result;
    els.previewImage.src = e.target.result;
    els.dropContent.style.display = 'none';
    els.previewContainer.style.display = 'block';
    els.analyzeBtn.disabled = false;
  };
  reader.readAsDataURL(file);
}

function clearImage() {
  state.file = null;
  state.imageDataUrl = null;
  els.fileInput.value = '';
  els.previewContainer.style.display = 'none';
  els.dropContent.style.display = 'flex';
  els.analyzeBtn.disabled = true;
}


// ═══════════════════════════════════════════════════════════════════════════
// SEVERITY TOGGLE
// ═══════════════════════════════════════════════════════════════════════════

function setupSeverity() {
  els.sevAutoBtn.addEventListener('click', () => setSeverityMode('auto'));
  els.sevManualBtn.addEventListener('click', () => setSeverityMode('manual'));

  els.sevSlider.addEventListener('input', () => {
    const val = parseInt(els.sevSlider.value);
    state.manualSeverity = val;
    updateSliderDisplay(val);
  });
}

function setSeverityMode(mode) {
  state.severityMode = mode;

  if (mode === 'auto') {
    els.sevAutoBtn.classList.add('toggle-btn--active');
    els.sevManualBtn.classList.remove('toggle-btn--active');
    els.sevAutoHint.style.display = 'block';
    els.sevManualCtrl.style.display = 'none';
  } else {
    els.sevManualBtn.classList.add('toggle-btn--active');
    els.sevAutoBtn.classList.remove('toggle-btn--active');
    els.sevAutoHint.style.display = 'none';
    els.sevManualCtrl.style.display = 'block';
  }
}

function updateSliderDisplay(val) {
  els.sevValue.textContent = `${val}%`;

  // Color gradient: green → yellow → red
  let color;
  if (val <= 30) {
    color = interpolateColor([34, 197, 94], [234, 179, 8], val / 30);
  } else if (val <= 60) {
    color = interpolateColor([234, 179, 8], [239, 68, 68], (val - 30) / 30);
  } else {
    color = interpolateColor([239, 68, 68], [185, 28, 28], (val - 60) / 40);
  }
  els.sevValue.style.color = `rgb(${color.join(',')})`;
  els.sevSlider.style.setProperty('--thumb-color', `rgb(${color.join(',')})`);
}

function interpolateColor(c1, c2, t) {
  return c1.map((v, i) => Math.round(v + (c2[i] - v) * Math.min(1, Math.max(0, t))));
}

function getSeverityLabel(pct) {
  if (pct <= 20) return 'Mild (0–30%)';
  if (pct <= 50) return 'Moderate (30–60%)';
  return 'Severe (60–100%)';
}

function getSelectedSeverity() {
  if (state.severityMode === 'manual') {
    return getSeverityLabel(state.manualSeverity);
  }
  return 'Moderate';  // Default, API will override based on confidence
}


// ═══════════════════════════════════════════════════════════════════════════
// TAB NAVIGATION
// ═══════════════════════════════════════════════════════════════════════════

function setupTabs() {
  els.tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      els.tabBtns.forEach(b => b.classList.remove('tab-btn--active'));
      btn.classList.add('tab-btn--active');
      $$('.tab-panel').forEach(p => p.classList.remove('tab-panel--active'));
      $(`#tab-${btn.dataset.tab}`).classList.add('tab-panel--active');
    });
  });
}


// ═══════════════════════════════════════════════════════════════════════════
// LOADING ANIMATION
// ═══════════════════════════════════════════════════════════════════════════

function showLoading() {
  els.loadingSection.style.display = 'block';
  els.resultsSection.style.display = 'none';

  // Reset steps
  [els.stepVision, els.stepReport].forEach(s => {
    if (s) s.className = 'loading-step';
  });

  // Animate steps sequentially
  setTimeout(() => { if (els.stepVision) els.stepVision.classList.add('loading-step--active'); }, 200);
  setTimeout(() => {
    if (els.stepVision) els.stepVision.classList.replace('loading-step--active', 'loading-step--done');
    if (els.stepReport) els.stepReport.classList.add('loading-step--active');
  }, 2000);
}

function hideLoading() {
  if (els.stepReport) els.stepReport.classList.replace('loading-step--active', 'loading-step--done');
  setTimeout(() => {
    els.loadingSection.style.display = 'none';
  }, 300);
}


// ═══════════════════════════════════════════════════════════════════════════
// ANALYSIS
// ═══════════════════════════════════════════════════════════════════════════

async function runAnalysis() {
  if (!state.file || state.isLoading) return;

  state.isLoading = true;
  els.analyzeBtn.disabled = true;
  els.analyzeBtnText.textContent = 'Analyzing...';
  els.analyzeBtnLoading.style.display = 'inline-block';

  showLoading();
  window.scrollTo({ top: els.loadingSection.offsetTop - 80, behavior: 'smooth' });

  try {
    const region = els.regionSelect.value;
    const season = els.seasonSelect.value;
    const severity = getSelectedSeverity();

    const result = await predictDisease(state.file, region, season, severity);

    // Apply manual severity override if in manual mode
    if (state.severityMode === 'manual') {
      result.severity = getSeverityLabel(state.manualSeverity);
    }

    // Enrich result with extra context
    result._region = region;
    result._season = season;
    result._severityMode = state.severityMode;
    result._manualPct = state.manualSeverity;

    state.result = result;

    hideLoading();
    setTimeout(() => {
      renderResults(result);
      els.resultsSection.style.display = 'block';
      window.scrollTo({ top: els.resultsSection.offsetTop - 80, behavior: 'smooth' });
    }, 400);

  } catch (err) {
    hideLoading();
    showToast(`❌ ${err.message}`);
    console.error('Analysis failed:', err);
  } finally {
    state.isLoading = false;
    els.analyzeBtn.disabled = false;
    els.analyzeBtnText.textContent = '🔍 Analyze Disease';
    els.analyzeBtnLoading.style.display = 'none';
  }
}


// ═══════════════════════════════════════════════════════════════════════════
// RENDER RESULTS
// ═══════════════════════════════════════════════════════════════════════════

function cleanLabel(label) {
  // 'label' looks like "Tomato___Spider_mites Two-spotted_spider_mite"
  // Split by the explicit '___' delimiter used in PlantVillage datasets
  const parts = label.split('___');
  
  if (parts.length > 1) {
    let rawDisease = parts[1];
    
    // Replace underscores with spaces
    let diseasePart = rawDisease.replace(/_/g, ' ').trim();

    // Convert to proper Title Case
    diseasePart = diseasePart.split(' ')
      .map(word => {
        if (!word) return '';
        // Special case for parentheses like "(Citrus_greening)"
        if (word.startsWith('(')) {
          return '(' + word.charAt(1).toUpperCase() + word.slice(2).toLowerCase();
        }
        return word.charAt(0).toUpperCase() + word.slice(1).toLowerCase();
      })
      .join(' ');
      
    return diseasePart;
  }
  
  // Fallback if no '___' is found
  return label.replace(/_/g, ' ');
}

function renderResults(r) {
  // Now disease will just be "Cedar Apple Rust" instead of "Apple - Cedar apple rust"
  const disease = cleanLabel(r.disease);
  const crop = r.crop;
  const confidence = r.confidence;
  const severity = r.severity;
  const tp = r.treatment_plan || {};
  const urgency = tp.action_urgency || '—';

  // Stats row
  els.statPlant.textContent = crop;
  els.statDisease.textContent = disease;
  els.statConfidence.textContent = `${confidence.toFixed(1)}%`;
  els.statSeverity.textContent = severity.split('(')[0].trim();
  els.statUrgency.textContent = urgency;

  // Result image
  if (state.imageDataUrl) {
    els.resultImage.src = state.imageDataUrl;
  }

  // Top-5 predictions
  renderTop5(r.top5 || []);

  // Diagnosis card
  els.diagnosisDisease.textContent = disease;
  els.diagnosisCrop.innerHTML = `Detected in <strong>${crop}</strong>`;
  els.diagnosisDesc.textContent = getDescription(r.disease);

  // Badges
  const sevKey = severity.split(' ')[0].toLowerCase();
  const sevBadgeClass = sevKey === 'severe' ? 'badge--severity-severe' :
                        sevKey === 'moderate' ? 'badge--severity-moderate' :
                        severity.toLowerCase().includes('healthy') ? 'badge--healthy' :
                        'badge--severity-mild';

  els.diagnosisBadges.innerHTML = `
    <span class="badge ${sevBadgeClass}">⚠ ${severity.split('(')[0].trim()}</span>
    <span class="badge badge--confidence">🎯 ${confidence.toFixed(1)}% Confidence</span>
    ${urgency !== '—' ? `<span class="badge badge--urgency">⏱ ${urgency}</span>` : ''}
  `;

  // Treatment tabs
  renderTreatmentTabs(tp);

  // Report
  renderReport(r);
}

function formatTop5Label(rawLabel) {
  const parts = rawLabel.split('___');
  if (parts.length > 1) {
    let cropPart = parts[0].replace(/_/g, ' ').trim();
    // Convert crop to title case
    cropPart = cropPart.split(' ').map(w => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase()).join(' ');

    let diseasePart = cleanLabel(rawLabel); // cleanLabel already returns perfectly capitalized disease
    return `${cropPart} — ${diseasePart}`;
  }
  return rawLabel.replace(/_/g, ' ');
}

function renderTop5(top5) {
  els.top5Bars.innerHTML = top5.map((p, i) => {
    const label = formatTop5Label(p.label);
    const conf = p.confidence;
    const isTop = i === 0;
    return `
      <div class="top5-item">
        <div class="top5-header">
          <span class="top5-label ${isTop ? 'top5-label--top' : ''}">${label}</span>
          <span class="top5-conf ${isTop ? 'top5-conf--top' : ''}">${conf.toFixed(1)}%</span>
        </div>
        <div class="top5-track">
          <div class="top5-fill ${isTop ? 'top5-fill--top' : ''}" style="width: 0%"></div>
        </div>
      </div>
    `;
  }).join('');

  // Animate bars
  requestAnimationFrame(() => {
    setTimeout(() => {
      els.top5Bars.querySelectorAll('.top5-fill').forEach((bar, i) => {
        bar.style.width = `${top5[i].confidence}%`;
      });
    }, 50);
  });
}

function renderTreatmentTabs(tp) {
  // Organic
  const organics = tp.organic_treatments || [];
  els.tabOrganic.innerHTML = organics.length ? organics.map(t => `
    <div class="treat-item">
      <div class="treat-item__title">🌿 ${t.method || 'Treatment'}</div>
      <div class="treat-item__row">
        <span class="treat-item__label">Application:</span>
        <span class="treat-item__value">${t.application || '—'}</span>
      </div>
      <div class="treat-item__row">
        <span class="treat-item__label">Frequency:</span>
        <span class="treat-item__value">${t.frequency || '—'}</span>
      </div>
    </div>
  `).join('') : '<div class="treat-empty">No organic treatments listed for this condition.</div>';

  // Chemical
  const chemicals = tp.chemical_treatments || [];
  els.tabChemical.innerHTML = chemicals.length ? chemicals.map(t => `
    <div class="treat-item">
      <div class="treat-item__title">💊 ${t.product || 'Treatment'}</div>
      <div class="treat-item__row">
        <span class="treat-item__label">Dosage:</span>
        <span class="treat-item__value">${t.dosage || '—'}</span>
      </div>
      ${t.safety_note ? `<div class="treat-item__warning">⚠ Safety: ${t.safety_note}</div>` : ''}
    </div>
  `).join('') : '<div class="treat-empty">No chemical treatments listed for this condition.</div>';

  // Preventive
  const preventive = tp.preventive_measures || [];
  els.tabPreventive.innerHTML = preventive.length ? `
    <ul class="prevent-list">
      ${preventive.map(m => `
        <li class="prevent-item">
          <span class="prevent-check">✓</span>
          <span>${m}</span>
        </li>
      `).join('')}
    </ul>
  ` : '<div class="treat-empty">No preventive measures listed.</div>';

  // Advisory
  const regional = tp.regional_notes || '—';
  const seasonal = tp.seasonal_notes || '—';
  const yieldImpact = tp.yield_impact_estimate || '';
  els.tabAdvisory.innerHTML = `
    <div class="advisory-grid">
      <div class="advisory-box">
        <div class="advisory-box__title">🌍 Regional Notes</div>
        <div class="advisory-box__text">${regional}</div>
      </div>
      <div class="advisory-box">
        <div class="advisory-box__title">🌦 Seasonal Notes</div>
        <div class="advisory-box__text">${seasonal}</div>
      </div>
    </div>
    ${yieldImpact ? `
      <div class="yield-impact">
        <strong>📉 Estimated Yield Impact:</strong> ${yieldImpact}
      </div>
    ` : ''}
  `;
}


// ═══════════════════════════════════════════════════════════════════════════
// REPORT — Professional Light-Themed Clinical Report
// ═══════════════════════════════════════════════════════════════════════════

function renderReport(r) {
  const disease = cleanLabel(r.disease);
  const crop = r.crop;
  const confidence = r.confidence;
  const severity = r.severity;
  const tp = r.treatment_plan || {};
  const urgency = tp.action_urgency || '—';
  const region = r._region || '—';
  const season = r._season || '—';
  const now = new Date();
  const dateStr = now.toLocaleDateString('en-IN', { day: '2-digit', month: 'short', year: 'numeric' });
  const timeStr = now.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' });
  const reportId = `CDA-${now.getFullYear()}${String(now.getMonth()+1).padStart(2,'0')}${String(now.getDate()).padStart(2,'0')}-${String(now.getHours()).padStart(2,'0')}${String(now.getMinutes()).padStart(2,'0')}${String(now.getSeconds()).padStart(2,'0')}`;

  const sevKey = severity.split(' ')[0];
  const sevColor = sevKey === 'Severe' ? '#dc2626' : sevKey === 'Moderate' ? '#d97706' : '#059669';
  const sevBg = sevKey === 'Severe' ? '#fef2f2' : sevKey === 'Moderate' ? '#fffbeb' : '#f0fdf4';
  const urgColor = urgency === 'Immediate' ? '#dc2626' : urgency.includes('3 days') ? '#ea580c' : urgency.includes('week') ? '#d97706' : '#059669';

  // Inline style constants for light-themed print-ready report
  const S = {
    wrap: 'font-family:Inter,system-ui,sans-serif;background:#ffffff;color:#1e293b;',
    // Banner
    banner: 'background:linear-gradient(135deg,#064e3b 0%,#065f46 40%,#047857 100%);padding:32px 36px 28px;text-align:center;',
    bannerTitle: 'font-size:1.5rem;font-weight:900;color:#ffffff;letter-spacing:0.5px;margin-bottom:4px;',
    bannerSub: 'font-size:0.85rem;font-weight:500;color:#a7f3d0;margin-bottom:8px;',
    bannerMeta: 'font-size:0.72rem;color:#6ee7b7;letter-spacing:0.03em;',
    // Content area
    body: 'padding:28px 32px 24px;',
    // Section headers
    secHeader: 'display:flex;align-items:center;gap:8px;padding:10px 16px;border-radius:8px;font-size:0.82rem;font-weight:700;color:#ffffff;letter-spacing:0.03em;margin-bottom:14px;page-break-after:avoid;page-break-inside:avoid;',
    // Meta grid
    metaGrid: 'display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:#e2e8f0;border:1px solid #e2e8f0;border-radius:8px;overflow:hidden;margin-bottom:14px;page-break-inside:avoid;',
    metaCell: 'background:#f8fafc;padding:14px 16px;',
    metaLabel: 'font-size:0.62rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#64748b;margin-bottom:4px;',
    metaValue: 'font-size:0.9rem;font-weight:700;color:#0f172a;',
    // Severity bar
    sevBar: 'display:grid;grid-template-columns:1fr 1fr;gap:3px;border-radius:8px;overflow:hidden;margin-bottom:24px;page-break-inside:avoid;',
    sevCell: 'padding:11px 16px;text-align:center;font-size:0.78rem;font-weight:700;color:#ffffff;letter-spacing:0.05em;',
    // Description
    descBox: 'padding:14px 18px;background:#f0fdf4;border-left:4px solid #059669;border-radius:0 8px 8px 0;font-size:0.85rem;line-height:1.7;color:#334155;margin-bottom:24px;page-break-inside:avoid;',
    // Tables
    table: 'width:100%;border-collapse:separate;border-spacing:0;border:1px solid #e2e8f0;border-radius:8px;overflow:hidden;margin-bottom:8px;page-break-inside:avoid;',
    th: 'color:#ffffff;font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;padding:10px 14px;text-align:left;',
    td: 'padding:10px 14px;font-size:0.82rem;color:#334155;border-bottom:1px solid #f1f5f9;page-break-inside:avoid;',
    tdAlt: 'padding:10px 14px;font-size:0.82rem;color:#334155;border-bottom:1px solid #f1f5f9;background:#f8fafc;page-break-inside:avoid;',
    // Preventive list
    prevWrap: 'border:1px solid #e2e8f0;border-radius:8px;overflow:hidden;margin-bottom:20px;page-break-inside:avoid;',
    prevItem: 'display:flex;align-items:flex-start;gap:10px;padding:10px 16px;font-size:0.82rem;color:#334155;line-height:1.55;',
    prevItemAlt: 'display:flex;align-items:flex-start;gap:10px;padding:10px 16px;font-size:0.82rem;color:#334155;line-height:1.55;background:#f8fafc;',
    // Advisory
    advGrid: 'display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:16px;page-break-inside:avoid;',
    advBox: 'padding:16px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;page-break-inside:avoid;',
    advTitle: 'font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#64748b;margin-bottom:8px;',
    advText: 'font-size:0.84rem;color:#334155;line-height:1.6;',
    // Yield
    yieldBox: 'padding:14px 18px;background:#fffbeb;border:1px solid #fde68a;border-radius:8px;font-size:0.84rem;color:#92400e;line-height:1.6;margin-bottom:16px;',
    // Disclaimer
    disclaim: 'padding:12px 18px;background:#fef2f2;border:1px solid #fecaca;border-radius:8px;font-size:0.78rem;color:#991b1b;font-style:italic;margin-top:24px;',
    // Footer
    footer: 'display:flex;justify-content:space-between;padding-top:14px;margin-top:16px;border-top:1px solid #e2e8f0;font-size:0.7rem;color:#94a3b8;',
    // Image
    img: 'max-width:180px;border-radius:8px;border:1px solid #e2e8f0;',
    // Divider
    divider: 'border:none;border-top:1px solid #e2e8f0;margin:20px 0;',
  };

  let html = `<div style="${S.wrap}">`;

  // ── Banner ──
  html += `
    <div style="${S.banner}">
      <div style="${S.bannerTitle}">🌾  CROP DISEASE ADVISOR</div>
      <div style="${S.bannerSub}">Clinical Agronomic Diagnosis & Treatment Report</div>
      <div style="${S.bannerMeta}">${reportId} · ${dateStr}, ${timeStr}</div>
    </div>
  `;

  // ── Body ──
  html += `<div style="${S.body}">`;

  // ── Image + Quick Summary Row ──
  html += `<div style="display:flex;gap:24px;align-items:flex-start;margin-bottom:24px;">`;
  if (state.imageDataUrl) {
    html += `<img src="${state.imageDataUrl}" alt="Analyzed leaf" style="${S.img}" />`;
  }
  html += `
    <div style="flex:1;">
      <div style="font-size:0.65rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#64748b;margin-bottom:6px;">DISEASE IDENTIFIED</div>
      <div style="font-size:1.4rem;font-weight:800;color:#0f172a;letter-spacing:-0.5px;margin-bottom:6px;">${disease}</div>
      <div style="font-size:0.88rem;color:#64748b;margin-bottom:14px;">Detected in <strong style="color:#059669;">${crop}</strong></div>
      <div style="display:flex;gap:8px;flex-wrap:wrap;">
        <span style="display:inline-block;padding:4px 14px;border-radius:99px;font-size:0.72rem;font-weight:700;background:${sevBg};color:${sevColor};border:1.5px solid ${sevColor};">⚠ ${sevKey}</span>
        <span style="display:inline-block;padding:4px 14px;border-radius:99px;font-size:0.72rem;font-weight:700;background:#eff6ff;color:#2563eb;border:1.5px solid #3b82f6;">🎯 ${confidence.toFixed(1)}% Confidence</span>
        ${urgency !== '—' ? `<span style="display:inline-block;padding:4px 14px;border-radius:99px;font-size:0.72rem;font-weight:700;background:#fff7ed;color:#c2410c;border:1.5px solid #f97316;">⏱ ${urgency}</span>` : ''}
      </div>
    </div>
  </div>`;

  // ── Diagnosis Summary ──
  html += `<div style="${S.secHeader}background:#059669;">📋  Diagnosis Summary</div>`;
  html += `<div style="${S.metaGrid}">`;
  const metaItems1 = [
    ['Disease Detected', disease], ['Crop Species', crop],
    ['AI Confidence', `${confidence.toFixed(1)}%`], ['Severity Level', severity],
  ];
  metaItems1.forEach(([label, value]) => {
    html += `<div style="${S.metaCell}"><div style="${S.metaLabel}">${label}</div><div style="${S.metaValue}">${value}</div></div>`;
  });
  html += `</div>`;

  html += `<div style="${S.metaGrid}">`;
  const metaItems2 = [
    ['Region', region], ['Season', season],
    ['Action Urgency', urgency], ['Status', 'COMPLETED'],
  ];
  metaItems2.forEach(([label, value]) => {
    html += `<div style="${S.metaCell}"><div style="${S.metaLabel}">${label}</div><div style="${S.metaValue}">${value}</div></div>`;
  });
  html += `</div>`;

  // Severity + Urgency colored bar
  html += `
    <div style="${S.sevBar}">
      <div style="${S.sevCell}background:${sevColor};">● SEVERITY: ${sevKey.toUpperCase()}</div>
      <div style="${S.sevCell}background:${urgColor};">⏱ ACTION: ${urgency.toUpperCase()}</div>
    </div>
  `;

  // ── Disease Description ──
  html += `<div style="${S.secHeader}background:#2563eb;">📖  Disease Description</div>`;
  html += `<div style="${S.descBox}">${getDescription(r.disease)}</div>`;

  // ── Organic Treatments Table ──
  const organics = tp.organic_treatments || [];
  if (organics.length) {
    html += `<div style="${S.secHeader}background:#0d9488;">🌿  Organic / Biological Treatments</div>`;
    html += `<table style="${S.table}"><thead><tr>
      <th style="${S.th}background:#0d9488;">Method / Agent</th>
      <th style="${S.th}background:#0d9488;">Application Instructions</th>
      <th style="${S.th}background:#0d9488;">Frequency</th>
    </tr></thead><tbody>`;
    organics.forEach((t, i) => {
      const sty = i % 2 === 0 ? S.td : S.tdAlt;
      html += `<tr><td style="${sty}font-weight:600;">${t.method || '—'}</td><td style="${sty}">${t.application || '—'}</td><td style="${sty}">${t.frequency || '—'}</td></tr>`;
    });
    html += `</tbody></table><hr style="${S.divider}">`;
  }

  // ── Chemical Treatments Table ──
  const chemicals = tp.chemical_treatments || [];
  if (chemicals.length) {
    html += `<div style="${S.secHeader}background:#dc2626;">⚗️  Chemical Treatments</div>`;
    html += `<table style="${S.table}"><thead><tr>
      <th style="${S.th}background:#dc2626;">Product / Fungicide</th>
      <th style="${S.th}background:#dc2626;">Dosage & Dilution</th>
      <th style="${S.th}background:#dc2626;">Safety Note</th>
    </tr></thead><tbody>`;
    chemicals.forEach((t, i) => {
      const sty = i % 2 === 0 ? S.td : S.tdAlt;
      html += `<tr><td style="${sty}font-weight:600;">${t.product || '—'}</td><td style="${sty}">${t.dosage || '—'}</td><td style="${sty}color:#991b1b;">${t.safety_note || '—'}</td></tr>`;
    });
    html += `</tbody></table><hr style="${S.divider}">`;
  }

  // ── Preventive Measures ──
  const preventive = tp.preventive_measures || [];
  if (preventive.length) {
    html += `<div style="${S.secHeader}background:#7c3aed;">🛡️  Preventive Measures</div>`;
    html += `<div style="${S.prevWrap}">`;
    preventive.forEach((m, i) => {
      const sty = i % 2 === 0 ? S.prevItem : S.prevItemAlt;
      html += `<div style="${sty}"><span style="color:#7c3aed;font-weight:700;flex-shrink:0;">✓</span><span>${m}</span></div>`;
    });
    html += `</div>`;
  }

  // ── Yield Impact ──
  if (tp.yield_impact_estimate) {
    html += `<div style="${S.secHeader}background:#d97706;">📉  Estimated Yield Impact</div>`;
    html += `<div style="${S.yieldBox}"><strong>📉 Impact:</strong> ${tp.yield_impact_estimate}</div>`;
  }

  // ── Regional & Seasonal Advisory ──
  if (tp.regional_notes || tp.seasonal_notes) {
    html += `<div style="${S.secHeader}background:#2563eb;">🌍  Regional & Seasonal Advisory</div>`;
    html += `<div style="${S.advGrid}">`;
    html += `<div style="${S.advBox}"><div style="${S.advTitle}">🌍 Regional Notes</div><div style="${S.advText}">${tp.regional_notes || '—'}</div></div>`;
    html += `<div style="${S.advBox}"><div style="${S.advTitle}">🌦 Seasonal Notes</div><div style="${S.advText}">${tp.seasonal_notes || '—'}</div></div>`;
    html += `</div>`;
  }

  // ── Disclaimer ──
  html += `<div style="${S.disclaim}">⚠ DISCLAIMER: AI-generated advisory only. Consult a licensed agronomist before applying any treatment.</div>`;

  // ── Footer ──
  html += `
    <div style="${S.footer}">
      <span>Crop Disease Advisor v2.0</span>
      <span>Shubham Haraniya · Vidhan Savaliya</span>
      <span>AI-Powered Diagnostics</span>
    </div>
  `;

  html += `</div></div>`; // close body + wrap

  els.reportContainer.innerHTML = html;
}


// ═══════════════════════════════════════════════════════════════════════════
// PDF EXPORT
// ═══════════════════════════════════════════════════════════════════════════

function downloadPdf() {
  const element = els.reportContainer;
  const now = new Date();
  const filename = `crop_disease_report_${now.getFullYear()}${String(now.getMonth()+1).padStart(2,'0')}${String(now.getDate()).padStart(2,'0')}_${String(now.getHours()).padStart(2,'0')}${String(now.getMinutes()).padStart(2,'0')}${String(now.getSeconds()).padStart(2,'0')}.pdf`;

  const opt = {
    margin:       10,
    filename:     filename,
    image:        { type: 'jpeg', quality: 0.98 },
    html2canvas:  { scale: 2, useCORS: true, backgroundColor: '#ffffff', scrollY: 0 },
    jsPDF:        { unit: 'mm', format: 'a4', orientation: 'portrait' },
    pagebreak:    { mode: ['avoid-all', 'css', 'legacy'] }
  };

  showToast('📄 Generating PDF...');

  // @ts-ignore
  html2pdf().set(opt).from(element).save().then(() => {
    showToast('✅ PDF downloaded successfully!');
  }).catch((err) => {
    showToast(`❌ PDF generation failed: ${err.message}`);
  });
}


// ═══════════════════════════════════════════════════════════════════════════
// TOAST
// ═══════════════════════════════════════════════════════════════════════════

let toastTimer = null;
function showToast(msg) {
  els.toast.textContent = msg;
  els.toast.style.display = 'block';
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    els.toast.style.display = 'none';
  }, 3500);
}


// ═══════════════════════════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════════════════════════

function init() {
  setupUpload();
  setupSeverity();
  setupTabs();

  els.analyzeBtn.addEventListener('click', runAnalysis);
  $('#download-pdf-btn').addEventListener('click', downloadPdf);

  // Health check
  checkApiHealth();
  setInterval(checkApiHealth, 30000);

  // Initial slider display
  updateSliderDisplay(30);
}

document.addEventListener('DOMContentLoaded', init);
