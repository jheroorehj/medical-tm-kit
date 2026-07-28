/* main.js — 부팅과 이벤트 배선
 * ---------------------------------------------------------------------------
 * 여기에는 판정 로직도 렌더 로직도 없습니다. 각 모듈을 연결하는 배선만 있습니다.
 *
 *   추론 1회의 흐름:
 *     프레임 → 품질 게이트 → 게이트키퍼 → 주분류기(+앙상블) → 판정 → 렌더 → 이력
 */

import { PROJECT, CLASS_IDS, POSITIVE_IDS } from '../project.config.js';
import * as store from './store.js';
import * as ui from './ui.js';
import * as hist from './history.js';
import { ModelRegistry, SLOT } from './models.js';
import { runGate, runClassify, decide, toSquareCanvas, STATUS, pct } from './infer.js';
import { measureFrame, evaluateQuality } from './quality.js';
import { TemporalStabilizer } from './temporal.js';
import { occlusionMap, drawHeatmap, interpretMap } from './saliency.js';
import { buildCalibration, describeCalibration } from './calibrate.js';
import {
  confusionMatrix, perClassMetrics, overallAccuracy, binaryMetrics,
  thresholdSweep, bestByYouden, thresholdForSensitivity,
  toCalibrationRecords, worstFailures, topConfusions, recordsToCsv, fmtPct,
} from './metrics.js';
import { collectFromDrop, collectFromInput, runBatch, verifyLabels } from './holdout.js';

const $ = ui.$;

/* ── 애플리케이션 상태 ────────────────────────────────────────────────── */
const app = {
  registry: new ModelRegistry(),
  settings: store.loadSettings(PROJECT),
  calibration: store.loadCalibration(),
  urls: store.loadModelUrls(),
  gate: { enabled: PROJECT.models.gatekeeper.enabled,
          passClasses: [...PROJECT.models.gatekeeper.passClasses],
          minConfidence: PROJECT.models.gatekeeper.minConfidence },
  ens: [],                       // [{url,label,weight}]
  stabilizer: null,
  cam: { stream: null, live: false, timer: null, busy: false },
  work: null,                    // 마지막 추론에 쓴 224 캔버스 (히트맵 재사용)
  lastDecision: null,
  lastQuality: null,
  heatToken: null,
  eval: { items: [], records: [], files: null, cm: null, sweep: [], running: null },
};

/* ── 부팅 ─────────────────────────────────────────────────────────────── */

function boot() {
  app.work = $('#work-canvas');
  applyProjectCopy();
  restoreConfigToUi();
  bindNav();
  bindInputTabs();
  bindCamera();
  bindImage();
  bindHeatmap();
  bindSettings();
  bindEval();
  bindHistory();

  app.registry.onChange(onRegistryChange);
  app.stabilizer = new TemporalStabilizer({
    window: app.settings.temporalWindow,
    confirmFrames: app.settings.confirmFrames,
    holdThreshold: app.settings.holdThreshold,
  });

  onRegistryChange();
  refreshStorageTiles();
  refreshCalibrationUi();
  autoLoadModels();
}

/** project.config.js의 카피를 화면에 주입합니다. */
function applyProjectCopy() {
  document.title = PROJECT.meta.title;
  ui.setText($('#app-title'), PROJECT.meta.title);
  ui.setText($('#app-subtitle'), PROJECT.meta.subtitle);
  ui.setText($('#app-badge'), PROJECT.meta.badge);
  $('#verdict-icon').textContent = PROJECT.meta.icon;
  $('#safety-note').innerHTML = PROJECT.copy.safetyNote;
  $('#privacy-note').innerHTML = PROJECT.copy.privacyNote;
  $('#privacy-note-2').innerHTML = PROJECT.copy.privacyNote;

  const chip = $('#eval-priority-chip');
  chip.textContent = PROJECT.copy.metricPriority === 'sensitivity'
    ? '민감도 우선' : '특이도 우선';
  chip.title = PROJECT.copy.metricPriorityReason;

  // 가이드 박스를 모델이 실제로 보는 영역과 일치시킵니다
  const box = $('#cam-guide')?.querySelector('.guide-box');
  if (box) box.style.width = `${Math.round(app.settings.inferRoi * 100)}%`;
}

/** 저장된 URL과 설정값을 폼에 되돌립니다. */
function restoreConfigToUi() {
  $('#primary-url').value = app.urls.primary || PROJECT.models.primary.url || '';
  $('#gate-url').value = app.urls.gatekeeper || PROJECT.models.gatekeeper.url || '';
  app.ens = (app.urls.ensemble?.length ? app.urls.ensemble : PROJECT.models.ensemble).map(e => ({ ...e }));

  app.gate.enabled = app.urls.gateEnabled ?? PROJECT.models.gatekeeper.enabled;
  app.gate.passClasses = app.urls.gatePass ?? PROJECT.models.gatekeeper.passClasses;
  app.gate.minConfidence = app.urls.gateMin ?? PROJECT.models.gatekeeper.minConfidence;

  $('#gate-enabled').checked = app.gate.enabled;
  $('#gate-pass').value = app.gate.passClasses.join(', ');
  $('#gate-min').value = app.gate.minConfidence;
  $('#gate-min-out').value = app.gate.minConfidence.toFixed(2);

  renderEnsembleList();
  syncSettingsToUi();
}

function syncSettingsToUi() {
  const s = app.settings;
  const set = (id, val, fmt) => {
    const el = $(id);
    if (el) el.value = val;
    const out = $(`${id}-out`);
    if (out) out.value = fmt ? fmt(val) : val;
  };
  set('#s-hold', s.holdThreshold, v => (+v).toFixed(2));
  set('#s-margin', s.marginThreshold, v => (+v).toFixed(2));
  set('#s-window', s.temporalWindow);
  set('#s-confirm', s.confirmFrames);
  set('#s-interval', s.inferIntervalMs);
  set('#s-blur', s.blurMin);
  $('#s-lumamin').value = s.lumaMin;
  $('#s-lumamax').value = s.lumaMax;
  $('#s-luma-out').value = `${s.lumaMin} – ${s.lumaMax}`;
  $('#s-qenabled').checked = s.enabled;
  $('#s-cal').checked = s.useCalibration;
  $('#eval-th').value = s.holdThreshold;
  $('#eval-th-out').value = (+s.holdThreshold).toFixed(2);
  updateMarginHint();
}

/**
 * 마진 관문(관문3)이 현재 임계값 조합에서 실제로 발동할 수 있는지 진단합니다.
 *
 * softmax는 확률의 합이 1입니다. 따라서 2위 ≤ 1 − 1위 이고,
 *   마진 = 1위 − 2위 ≥ 2 × 1위 − 1
 * 즉 1위가 0.70이면 마진은 최소 0.40입니다. 마진 기준이 그보다 작으면
 * 관문2(판단 보류)가 항상 먼저 걸러내므로 관문3은 절대 발동하지 않습니다.
 *
 * 사용자가 슬라이더를 조정했는데 아무 변화가 없는 이유를 알려 주는 것이
 * 이 진단의 목적입니다. 민감도를 위해 임계값을 낮추면 활성화됩니다.
 */
function updateMarginHint() {
  const el = $('#s-margin-hint');
  if (!el) return;
  const { holdThreshold: h, marginThreshold: m } = app.settings;
  const minPossible = 2 * h - 1;          // 1위가 h일 때 가능한 최소 마진

  if (minPossible >= m) {
    const need = ((m + 1) / 2);
    el.innerHTML = `<b style="color:var(--warn)">현재 조합에서는 발동하지 않습니다.</b> `
      + `softmax 합이 1이므로 1위가 ${(h * 100).toFixed(0)}%면 마진은 최소 `
      + `${(minPossible * 100).toFixed(0)}%p입니다. 보류 임계값을 `
      + `${(need * 100).toFixed(0)}% 아래로 내리거나 이 값을 `
      + `${(minPossible * 100).toFixed(0)}%p 위로 올리세요.`;
  } else {
    el.innerHTML = `차이가 작으면 "구분 어려움"으로 처리합니다. `
      + `<span class="muted">1위가 ${(h * 100).toFixed(0)}%~${(((m + 1) / 2) * 100).toFixed(0)}% 구간일 때 발동합니다.</span>`;
  }
}

function persistSettings() {
  store.saveSettings(app.settings);
  app.stabilizer?.configure({
    window: app.settings.temporalWindow,
    confirmFrames: app.settings.confirmFrames,
    holdThreshold: app.settings.holdThreshold,
  });
}

function persistUrls() {
  store.saveModelUrls({
    primary: $('#primary-url').value.trim(),
    gatekeeper: $('#gate-url').value.trim(),
    ensemble: app.ens,
    gateEnabled: app.gate.enabled,
    gatePass: app.gate.passClasses,
    gateMin: app.gate.minConfidence,
  });
}

/** config나 저장소에 URL이 있으면 자동으로 불러옵니다. */
async function autoLoadModels() {
  const primary = $('#primary-url').value.trim();
  if (primary) await loadPrimary();
  if (app.gate.enabled && $('#gate-url').value.trim()) await loadGate();
  for (let i = 0; i < app.ens.length; i++) {
    if (app.ens[i].url) await app.registry.load(SLOT.ens(i), app.ens[i].url, app.ens[i]);
  }
}

/* ── 모델 상태 변화 반영 ──────────────────────────────────────────────── */

function onRegistryChange() {
  const reg = app.registry;
  ui.setSlotState('#primary-state', reg.status(SLOT.PRIMARY));
  ui.setSlotState('#gate-state', reg.status(SLOT.GATE));
  app.ens.forEach((_, i) => ui.setSlotState(`#ens-state-${i}`, reg.status(SLOT.ens(i))));

  const ready = reg.has(SLOT.PRIMARY);
  const primaryStatus = reg.status(SLOT.PRIMARY);

  if (ready) {
    const n = reg.get(SLOT.PRIMARY).labels.length;
    ui.setModelPill('ok', `모델 준비됨 · 클래스 ${n}개`);
  } else if (primaryStatus.state === 'loading') {
    ui.setModelPill('loading', '모델 불러오는 중…');
  } else if (primaryStatus.state === 'error') {
    ui.setModelPill('error', '모델 로드 실패');
  } else {
    ui.setModelPill('idle', '모델 미로드');
  }

  ui.show($('#no-model-banner'), !ready);
  $('#img-run').disabled = !ready || !$('#img-preview').getAttribute('src');
  $('#cam-live').disabled = !ready || !app.cam.stream;
  $('#cam-snap').disabled = !ready || !app.cam.stream;
  $('#heat-run').disabled = !ready || !app.lastDecision;

  // 파이프라인 구성 표시
  const chip = $('#pipeline-chip');
  if (ready) {
    const parts = [];
    if (app.gate.enabled && reg.has(SLOT.GATE)) parts.push('게이트키퍼');
    const nCls = reg.classifierSlots().length;
    parts.push(nCls > 1 ? `앙상블 ${nCls}개` : '단일 모델');
    if (app.settings.useCalibration && app.calibration) parts.push('신뢰도 보정');
    chip.textContent = parts.join(' → ');
    ui.show(chip, true);
  } else {
    ui.show(chip, false);
  }

  // 클래스 정합성 리포트
  if (ready) {
    ui.renderClassCheck($('#class-check'),
      reg.verifyClasses(CLASS_IDS), reg.verifyEnsembleConsistency());
  } else {
    ui.show($('#class-check'), false);
  }
}

async function loadPrimary() {
  const url = $('#primary-url').value.trim();
  persistUrls();
  await app.registry.load(SLOT.PRIMARY, url, { label: PROJECT.models.primary.label || '주분류기' });
}

async function loadGate() {
  const url = $('#gate-url').value.trim();
  persistUrls();
  await app.registry.load(SLOT.GATE, url, { label: '게이트키퍼' });
}

/* ── 네비게이션 ───────────────────────────────────────────────────────── */

function bindNav() {
  ui.$$('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => switchView(btn.dataset.view));
  });
  ui.$$('[data-goto]').forEach(el => {
    el.addEventListener('click', () => switchView(el.dataset.goto));
  });
}

function switchView(name) {
  ui.$$('.nav-btn').forEach(b => {
    const on = b.dataset.view === name;
    b.classList.toggle('active', on);
    b.setAttribute('aria-selected', String(on));
  });
  ui.$$('.view').forEach(v => v.classList.toggle('active', v.id === `view-${name}`));
  if (name === 'history') refreshHistoryView();
  if (name === 'settings') refreshStorageTiles();
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function bindInputTabs() {
  ui.$$('.seg-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const kind = btn.dataset.input;
      ui.$$('.seg-btn').forEach(b => {
        const on = b === btn;
        b.classList.toggle('active', on);
        b.setAttribute('aria-selected', String(on));
      });
      ui.show($('#pane-cam'), false);
      ui.show($('#pane-img'), false);
      $('#pane-cam').classList.toggle('active', kind === 'cam');
      $('#pane-img').classList.toggle('active', kind === 'img');
      if (kind === 'img') stopCamera();      // 탭을 떠나면 카메라를 끕니다
    });
  });
}

/* ── 웹캠 ─────────────────────────────────────────────────────────────── */

function bindCamera() {
  $('#cam-start').addEventListener('click', startCamera);
  $('#cam-stop').addEventListener('click', stopCamera);
  $('#cam-snap').addEventListener('click', () => runOnce('webcam'));
  $('#cam-live').addEventListener('click', toggleLive);
  // 페이지를 떠날 때 카메라 LED가 남아 있지 않도록 확실히 정리합니다
  window.addEventListener('pagehide', stopCamera);
}

async function startCamera() {
  const err = $('#cam-error');
  if (!navigator.mediaDevices?.getUserMedia) {
    ui.note(err, 'danger', '이 브라우저는 카메라 API를 지원하지 않습니다.');
    err.hidden = false;
    return;
  }
  try {
    app.cam.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
  } catch (e) {
    const msg = location.protocol === 'file:'
      ? '<b>file:// 로 열면 카메라를 쓸 수 없습니다.</b> 프로젝트 폴더에서 <code>./serve.sh</code>를 실행하고 http://localhost:8000 으로 접속하세요.'
      : e?.name === 'NotAllowedError'
        ? '카메라 접근이 거부되었습니다. 브라우저 주소창의 권한 설정을 확인하세요.'
        : `카메라를 열 수 없습니다 (${e?.name ?? '알 수 없는 오류'}).`;
    err.innerHTML = msg;
    err.hidden = false;
    return;
  }

  err.hidden = true;
  const vid = $('#cam-video');
  vid.srcObject = app.cam.stream;
  await vid.play().catch(() => {});

  ui.show(vid, true);
  ui.show($('#cam-idle'), false);
  ui.show($('#cam-guide'), true);
  ui.show($('#cam-badge'), true);
  ui.show($('#qstrip'), true);
  $('#cam-start').disabled = true;
  $('#cam-stop').disabled = false;
  const ready = app.registry.has(SLOT.PRIMARY);
  $('#cam-snap').disabled = !ready;
  $('#cam-live').disabled = !ready;

  startQualityMonitor();
}

function stopCamera() {
  stopLive();
  stopQualityMonitor();
  if (app.cam.stream) {
    app.cam.stream.getTracks().forEach(t => t.stop());
    app.cam.stream = null;
  }
  const vid = $('#cam-video');
  vid.srcObject = null;
  ui.show(vid, false);
  ui.show($('#cam-idle'), true);
  ui.show($('#cam-guide'), false);
  ui.show($('#cam-badge'), false);
  ui.show($('#cam-coach'), false);
  ui.show($('#qstrip'), false);
  $('#cam-start').disabled = false;
  $('#cam-stop').disabled = true;
  $('#cam-snap').disabled = true;
  $('#cam-live').disabled = true;
}

/* 품질 모니터 — 추론과 무관하게 항상 돌면서 촬영을 코칭합니다 */
let qTimer = null;

function startQualityMonitor() {
  stopQualityMonitor();
  qTimer = setInterval(() => {
    const vid = $('#cam-video');
    const m = measureFrame(vid, app.settings.inferRoi);
    app.lastQuality = m;
    const ev = evaluateQuality(m, app.settings);
    ui.renderQuality('q', m, ev, app.settings);
    ui.renderCoach(ev);
    // 설정 탭에서도 실시간 수치를 볼 수 있게 합니다 (임계값 맞추기용)
    const lc = $('#q-live-chip');
    if (m) {
      lc.textContent = `실시간 · 초점 ${m.blur.toFixed(0)} / 밝기 ${m.luma.toFixed(0)} / 대비 ${m.contrast.toFixed(0)}`;
      lc.dataset.tone = ev.ok ? 'ok' : 'warn';
      ui.show(lc, true);
    }
  }, 250);
}

function stopQualityMonitor() {
  if (qTimer) { clearInterval(qTimer); qTimer = null; }
  ui.show($('#q-live-chip'), false);
}

function toggleLive() {
  if (app.cam.live) stopLive(); else startLive();
}

function startLive() {
  if (!app.cam.stream || !app.registry.has(SLOT.PRIMARY)) return;
  app.cam.live = true;
  app.stabilizer.reset();
  const btn = $('#cam-live');
  btn.textContent = '실시간 판독 중지';
  btn.setAttribute('aria-pressed', 'true');
  liveTick();
}

function stopLive() {
  app.cam.live = false;
  if (app.cam.timer) { clearTimeout(app.cam.timer); app.cam.timer = null; }
  const btn = $('#cam-live');
  if (btn) {
    btn.textContent = '실시간 판독 시작';
    btn.setAttribute('aria-pressed', 'false');
  }
}

async function liveTick() {
  if (!app.cam.live) return;
  if (!app.cam.busy) {
    app.cam.busy = true;
    try {
      await runOnce('live');
    } catch (e) {
      console.warn('[live] 추론 실패:', e);
    } finally {
      app.cam.busy = false;
    }
  }
  app.cam.timer = setTimeout(liveTick, app.settings.inferIntervalMs);
}

/* ── 이미지 업로드 ────────────────────────────────────────────────────── */

function bindImage() {
  const drop = $('#img-drop');
  const file = $('#img-file');

  drop.addEventListener('click', () => file.click());
  drop.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); file.click(); }
  });
  drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('drag'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('drag'));
  drop.addEventListener('drop', e => {
    e.preventDefault();
    drop.classList.remove('drag');
    const f = e.dataTransfer.files[0];
    if (f?.type.startsWith('image/')) loadImageFile(f);
    else showImgError('이미지 파일만 업로드할 수 있습니다.');
  });
  file.addEventListener('change', e => {
    if (e.target.files[0]) loadImageFile(e.target.files[0]);
  });

  $('#img-run').addEventListener('click', () => runOnce('upload'));
  $('#img-reset').addEventListener('click', resetImage);
}

function showImgError(msg) {
  const el = $('#img-error');
  el.textContent = msg;
  el.hidden = false;
}

function loadImageFile(f) {
  const img = $('#img-preview');
  const url = URL.createObjectURL(f);
  img.onload = () => {
    ui.show($('#img-stage'), true);
    ui.show($('#img-drop'), false);
    ui.show($('#img-reset'), true);
    ui.show($('#img-qstrip'), true);
    $('#img-error').hidden = true;
    $('#img-run').disabled = !app.registry.has(SLOT.PRIMARY);

    const m = measureFrame(img, app.settings.inferRoi);
    const ev = evaluateQuality(m, app.settings);
    ui.renderQuality('iq', m, ev, app.settings);
    if (!ev.ok) showImgError(`품질 경고: ${ev.coach}`);
  };
  img.onerror = () => {
    URL.revokeObjectURL(url);
    showImgError('이미지를 읽을 수 없습니다.');
  };
  img.src = url;
}

function resetImage() {
  const img = $('#img-preview');
  const old = img.getAttribute('src');
  if (old?.startsWith('blob:')) URL.revokeObjectURL(old);
  img.removeAttribute('src');            // ★ src='' 는 문서 URL로 해석되므로 속성을 지웁니다
  ui.show($('#img-stage'), false);
  ui.show($('#img-drop'), true);
  ui.show($('#img-reset'), false);
  ui.show($('#img-qstrip'), false);
  $('#img-file').value = '';
  $('#img-run').disabled = true;         // 원본 템플릿에서 빠져 있던 처리
  $('#img-error').hidden = true;
  ui.clearResult();
  app.lastDecision = null;
  $('#heat-run').disabled = true;
  ui.show($('#heat-wrap'), false);
  ui.show($('#heat-note'), false);
}

/* ── 추론 1회 ─────────────────────────────────────────────────────────── */

/**
 * @param {'webcam'|'upload'|'live'} sourceKind
 */
async function runOnce(sourceKind) {
  if (!app.registry.has(SLOT.PRIMARY)) return;

  const live = sourceKind === 'live';
  const source = sourceKind === 'upload' ? $('#img-preview') : $('#cam-video');
  const errEl = sourceKind === 'upload' ? $('#img-error') : $('#cam-error');

  // ① 품질 게이트 — 실시간 모드에서는 나쁜 프레임을 조용히 건너뜁니다
  const m = measureFrame(source, app.settings.inferRoi);
  const qev = evaluateQuality(m, app.settings);
  if (sourceKind !== 'upload') {
    ui.renderQuality('q', m, qev, app.settings);
    ui.renderCoach(qev);
  }
  if (!qev.ok) {
    if (live) return;                               // 실시간: 그냥 다음 프레임을 기다림
    if (!m) { errEl.textContent = '영상을 읽을 수 없습니다.'; errEl.hidden = false; return; }
    // 단발 촬영: 경고만 하고 진행합니다 (사용자가 의도적으로 눌렀으므로)
    errEl.textContent = `품질 경고: ${qev.coach} — 결과 신뢰도가 낮을 수 있습니다.`;
    errEl.hidden = false;
  } else {
    errEl.hidden = true;
  }

  // ② 학습 전처리와 동일한 방식으로 224 정사각 캔버스를 만듭니다
  toSquareCanvas(source, app.settings.inferRoi, 224, app.work);

  try {
    // ③ 게이트키퍼 → ④ 주분류기(+앙상블)
    const gate = await runGate(app.registry, app.work, {
      enabled: app.gate.enabled, passClasses: app.gate.passClasses,
      minConfidence: app.gate.minConfidence,
    });
    const { probs } = await runClassify(app.registry, app.work);

    // ⑤ 시간축 안정화 (실시간 모드에서만)
    let temporal = null;
    let effective = probs;
    if (live) {
      temporal = app.stabilizer.push(probs);
      effective = temporal.smoothed;
    }

    // ⑥ 판정
    const d = decide(effective, app.settings, {
      gate, calibration: app.calibration, copy: PROJECT.copy,
    });

    ui.renderVerdict(d, {
      jitter: temporal?.jitter,
      live, temporal,
      confirmFrames: app.settings.confirmFrames,
      threshold: app.settings.holdThreshold,
    });

    app.lastDecision = d;
    $('#heat-run').disabled = false;

    // ⑦ 이력 — 실시간은 "확정된 순간"에만 기록합니다 (초당 5건이 쌓이지 않도록)
    const shouldRecord = !live || temporal?.justConfirmed;
    if (shouldRecord) {
      hist.record({
        source: sourceKind, decision: d, quality: m,
        thumb: hist.makeThumb(app.work), settings: app.settings,
      });
    }
  } catch (e) {
    console.error('[infer] 실패:', e);
    errEl.textContent = '추론 중 오류가 발생했습니다. 콘솔을 확인하세요.';
    errEl.hidden = false;
  }
}

/* ── 근거 히트맵 ──────────────────────────────────────────────────────── */

function bindHeatmap() {
  $('#heat-run').addEventListener('click', async () => {
    if (!app.lastDecision) return;
    const btn = $('#heat-run');

    // 이미 돌고 있으면 중단
    if (app.heatToken) { app.heatToken.aborted = true; return; }

    const entry = app.registry.get(SLOT.PRIMARY);
    const token = { aborted: false };
    app.heatToken = token;
    btn.textContent = '중단';
    ui.show($('#heat-progress'), true);
    ui.show($('#heat-note'), false);
    stopLive();                     // 실시간 중에는 프레임이 바뀌므로 멈춥니다

    try {
      const map = await occlusionMap(entry.model, app.work, {
        grid: 8,
        targetId: app.lastDecision.top.id,
        onProgress: (done, total) => {
          $('#heat-fill').style.width = `${(done / total * 100).toFixed(0)}%`;
          ui.setText($('#heat-count'), `${done}/${total}`);
        },
        token,
      });

      const cv = $('#heat-canvas');
      const ctx = cv.getContext('2d');
      ctx.drawImage(app.work, 0, 0, cv.width, cv.height);
      if (!map.aborted) drawHeatmap(cv, map);
      ui.show($('#heat-wrap'), true);

      const interp = interpretMap(map);
      ui.note($('#heat-note'), interp.tone,
        `${interp.text}<br><span class="muted">기준 유사도 ${pct(map.base)} · `
        + `최대 하락 ${pct(Math.max(0, map.max))} · 대상 클래스 ${ui.esc(app.lastDecision.top.def.label)}`
        + (map.aborted ? ' · <b>중단됨</b>' : '') + '</span>');
    } catch (e) {
      console.error('[saliency] 실패:', e);
      ui.note($('#heat-note'), 'danger', '근거 분석 중 오류가 발생했습니다.');
    } finally {
      app.heatToken = null;
      btn.textContent = '근거 분석';
      ui.show($('#heat-progress'), false);
    }
  });
}

/* ── 설정 ─────────────────────────────────────────────────────────────── */

function bindSettings() {
  $('#primary-load').addEventListener('click', loadPrimary);
  $('#gate-load').addEventListener('click', loadGate);

  $('#gate-enabled').addEventListener('change', e => {
    app.gate.enabled = e.target.checked;
    persistUrls();
    if (app.gate.enabled && $('#gate-url').value.trim() && !app.registry.has(SLOT.GATE)) loadGate();
    else onRegistryChange();
  });
  $('#gate-pass').addEventListener('change', e => {
    app.gate.passClasses = e.target.value.split(',').map(s => s.trim()).filter(Boolean);
    persistUrls();
  });
  bindRange('#gate-min', v => {
    app.gate.minConfidence = v;
    $('#gate-min-out').value = v.toFixed(2);
    persistUrls();
  });

  $('#ens-add').addEventListener('click', () => {
    app.ens.push({ url: '', label: `앙상블 ${app.ens.length + 1}`, weight: 1 });
    renderEnsembleList();
    persistUrls();
  });

  // 판정 정책
  bindRange('#s-hold', v => {
    app.settings.holdThreshold = v;
    $('#s-hold-out').value = v.toFixed(2);
    $('#eval-th').value = v;
    $('#eval-th-out').value = v.toFixed(2);
    updateMarginHint();
    persistSettings();
    if (app.eval.records.length) refreshEvalMetrics();
  });
  bindRange('#s-margin', v => {
    app.settings.marginThreshold = v;
    $('#s-margin-out').value = v.toFixed(2);
    updateMarginHint();
    persistSettings();
  });
  bindRange('#s-window', v => {
    app.settings.temporalWindow = Math.round(v);
    $('#s-window-out').value = app.settings.temporalWindow;
    persistSettings();
  });
  bindRange('#s-confirm', v => {
    app.settings.confirmFrames = Math.round(v);
    $('#s-confirm-out').value = app.settings.confirmFrames;
    persistSettings();
  });
  bindRange('#s-interval', v => {
    app.settings.inferIntervalMs = Math.round(v);
    $('#s-interval-out').value = app.settings.inferIntervalMs;
    persistSettings();
  });
  $('#s-cal').addEventListener('change', e => {
    app.settings.useCalibration = e.target.checked;
    persistSettings();
    onRegistryChange();
  });

  // 품질 게이트
  $('#s-qenabled').addEventListener('change', e => {
    app.settings.enabled = e.target.checked;
    persistSettings();
  });
  bindRange('#s-blur', v => {
    app.settings.blurMin = Math.round(v);
    $('#s-blur-out').value = app.settings.blurMin;
    persistSettings();
  });
  const lumaSync = () => {
    let lo = +$('#s-lumamin').value, hi = +$('#s-lumamax').value;
    if (lo >= hi) lo = hi - 1;
    app.settings.lumaMin = lo;
    app.settings.lumaMax = hi;
    $('#s-luma-out').value = `${lo} – ${hi}`;
    persistSettings();
  };
  $('#s-lumamin').addEventListener('input', lumaSync);
  $('#s-lumamax').addEventListener('input', lumaSync);

  $('#policy-reset').addEventListener('click', () => {
    store.resetSettings();
    app.settings = store.loadSettings(PROJECT);
    syncSettingsToUi();
    persistSettings();
    applyProjectCopy();
  });

  // 저장 데이터
  $('#wipe-cal').addEventListener('click', () => {
    store.clearCalibration();
    app.calibration = null;
    refreshCalibrationUi();
    refreshStorageTiles();
    onRegistryChange();
  });
  $('#wipe-holdout').addEventListener('click', () => {
    store.clearHoldout();
    app.eval = { items: [], records: [], files: null, cm: null, sweep: [], running: null };
    ui.show($('#eval-results'), false);
    refreshStorageTiles();
  });
  $('#wipe-all').addEventListener('click', () => {
    if (!confirm('저장된 설정·이력·보정표를 모두 삭제합니다. 계속할까요?')) return;
    store.wipeAll();
    location.reload();
  });
}

/** range 입력을 숫자 콜백으로 감쌉니다. */
function bindRange(sel, fn) {
  const el = $(sel);
  if (!el) return;
  el.addEventListener('input', () => fn(+el.value));
}

function renderEnsembleList() {
  const list = $('#ens-list');
  if (!app.ens.length) {
    list.innerHTML = `<p class="hint">등록된 앙상블 모델이 없습니다. 단일 모델로 동작합니다.</p>`;
    return;
  }
  list.innerHTML = app.ens.map((e, i) => `
    <div class="ens-item">
      <div class="ens-item-head">
        <span>${ui.esc(e.label)}</span>
        <span class="state" id="ens-state-${i}" data-state="idle">미로드</span>
      </div>
      <div class="url-row">
        <input class="input mono" data-ens-url="${i}" type="url" spellcheck="false"
               value="${ui.esc(e.url)}" placeholder="https://teachablemachine.withgoogle.com/models/…/" />
        <button class="btn btn-sm" data-ens-load="${i}">불러오기</button>
        <button class="btn btn-sm btn-danger" data-ens-del="${i}">삭제</button>
      </div>
      <div class="field">
        <label>보팅 가중치 <output class="mono">${e.weight.toFixed(2)}</output></label>
        <input type="range" data-ens-w="${i}" min="0.1" max="2" step="0.05" value="${e.weight}" />
      </div>
    </div>`).join('');

  list.querySelectorAll('[data-ens-url]').forEach(inp => {
    inp.addEventListener('change', () => {
      app.ens[+inp.dataset.ensUrl].url = inp.value.trim();
      persistUrls();
    });
  });
  list.querySelectorAll('[data-ens-load]').forEach(btn => {
    btn.addEventListener('click', async () => {
      const i = +btn.dataset.ensLoad;
      await app.registry.load(SLOT.ens(i), app.ens[i].url, app.ens[i]);
    });
  });
  list.querySelectorAll('[data-ens-del]').forEach(btn => {
    btn.addEventListener('click', () => {
      const i = +btn.dataset.ensDel;
      app.registry.unload(SLOT.ens(i));
      app.ens.splice(i, 1);
      // 슬롯 번호가 밀리므로 전체를 다시 로드합니다
      app.registry.classifierSlots()
        .filter(s => s.startsWith('ensemble:'))
        .forEach(s => app.registry.unload(s));
      renderEnsembleList();
      persistUrls();
      app.ens.forEach((e, k) => { if (e.url) app.registry.load(SLOT.ens(k), e.url, e); });
    });
  });
  list.querySelectorAll('[data-ens-w]').forEach(inp => {
    inp.addEventListener('input', () => {
      const i = +inp.dataset.ensW;
      app.ens[i].weight = +inp.value;
      inp.closest('.field').querySelector('output').value = app.ens[i].weight.toFixed(2);
      const entry = app.registry.get(SLOT.ens(i));
      if (entry) entry.weight = app.ens[i].weight;
      persistUrls();
    });
  });
}

function refreshStorageTiles() {
  const h = hist.list();
  ui.renderTiles($('#storage-tiles'), [
    { k: '판독 이력', v: `${h.length}건` },
    { k: '보정표', v: app.calibration ? `표본 ${app.calibration.n}` : '없음',
      tone: app.calibration ? 'ok' : undefined },
    { k: '평가 결과', v: app.eval.records.length ? `${app.eval.records.length}건` : '없음' },
    { k: '저장 용량', v: ui.fmtBytes(store.storageUsage()) },
  ]);
}

function refreshCalibrationUi() {
  const desc = describeCalibration(app.calibration);
  ui.note($('#cal-summary'), desc.tone, desc.text);
  ui.renderCalibration($('#cal-table'), app.calibration);
  ui.setText($('#s-cal-hint'), app.calibration
    ? `보정표 준비됨 (표본 ${app.calibration.n}건, ECE ${(app.calibration.ece * 100).toFixed(1)}%p)`
    : '보정표가 있어야 적용됩니다 — 성능 평가 탭에서 생성하세요');
}

/* ── 홀드아웃 평가 ────────────────────────────────────────────────────── */

function bindEval() {
  const drop = $('#eval-drop');
  const input = $('#eval-files');

  drop.addEventListener('click', () => input.click());
  drop.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); }
  });
  drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('drag'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('drag'));
  drop.addEventListener('drop', async e => {
    e.preventDefault();
    drop.classList.remove('drag');
    const items = await collectFromDrop(e.dataTransfer);
    startEval(items);
  });
  input.addEventListener('change', e => startEval(collectFromInput(e.target.files)));

  $('#eval-cancel').addEventListener('click', () => {
    if (app.eval.running) app.eval.running.aborted = true;
  });

  bindRange('#eval-th', v => {
    $('#eval-th-out').value = v.toFixed(2);
    refreshEvalMetrics();
  });
  $('#eval-policy').addEventListener('change', refreshEvalMetrics);

  $('#eval-apply-th').addEventListener('click', () => {
    const v = +$('#eval-th').value;
    app.settings.holdThreshold = v;
    syncSettingsToUi();
    persistSettings();
    switchView('settings');
  });
  $('#eval-youden').addEventListener('click', () => {
    const best = bestByYouden(app.eval.sweep);
    if (!best) return;
    $('#eval-th').value = best.threshold;
    $('#eval-th-out').value = best.threshold.toFixed(2);
    refreshEvalMetrics();
  });
  $('#eval-sens95').addEventListener('click', () => {
    const t = thresholdForSensitivity(app.eval.sweep, 0.95);
    if (!t) {
      ui.note($('#eval-labelcheck'), 'warn',
        '민감도 95%를 만족하는 임계값이 없습니다. 클래스 설계나 학습 데이터를 재검토해야 합니다.');
      return;
    }
    $('#eval-th').value = t.threshold;
    $('#eval-th-out').value = t.threshold.toFixed(2);
    refreshEvalMetrics();
  });

  $('#cal-build').addEventListener('click', () => {
    const table = buildCalibration(toCalibrationRecords(app.eval.records));
    if (!table) {
      ui.note($('#cal-summary'), 'warn', '표본이 부족해 보정표를 만들 수 없습니다 (최소 10건 필요).');
      return;
    }
    app.calibration = table;
    store.saveCalibration(table);
    refreshCalibrationUi();
    refreshStorageTiles();
    onRegistryChange();
  });

  $('#eval-csv').addEventListener('click', () => {
    if (!app.eval.records.length) return;
    hist.download(`holdout-eval-${stamp()}.csv`,
      recordsToCsv(app.eval.records, CLASS_IDS));
  });
}

async function startEval(items) {
  const err = $('#eval-error');
  err.hidden = true;

  if (!app.registry.has(SLOT.PRIMARY)) {
    err.innerHTML = '<b>모델이 로드되지 않았습니다.</b> 설정 탭에서 모델을 먼저 불러오세요.';
    err.hidden = false;
    return;
  }
  if (!items.length) {
    err.textContent = '이미지 파일을 찾지 못했습니다. 클래스별 하위 폴더가 있는 폴더를 선택하세요.';
    err.hidden = false;
    return;
  }

  // 폴더명 ↔ 클래스 id 정합성 — 실습에서 가장 흔한 사고 지점
  const lc = verifyLabels(items, CLASS_IDS);
  const lines = [`총 <b>${items.length}장</b> · `
    + lc.counts.map(c => `${ui.esc(c.label)} ${c.n}`).join(' / ')];
  if (lc.unknown.length) {
    lines.push(`<b>알 수 없는 폴더</b>: ${lc.unknown.map(l => `<code>${ui.esc(l)}</code>`).join(', ')}`
      + ` — <code>project.config.js</code>의 클래스 id와 폴더명을 일치시키세요. 이 폴더는 오분류로 집계됩니다.`);
  }
  if (lc.missing.length) {
    lines.push(`폴더가 없는 클래스: ${lc.missing.map(l => `<code>${ui.esc(l)}</code>`).join(', ')}`);
  }
  ui.note($('#eval-labelcheck'), lc.unknown.length ? 'danger' : 'ok', lines.join('<br>'));

  stopLive();
  const token = { aborted: false };
  app.eval.running = token;
  ui.show($('#eval-progress'), true);

  const { records, failed, aborted, files } = await runBatch(app.registry, items, {
    roi: app.settings.inferRoi,
    onProgress: (done, total) => {
      $('#eval-fill').style.width = `${(done / total * 100).toFixed(1)}%`;
      ui.setText($('#eval-count'), `${done}/${total}`);
    },
    token,
  });

  app.eval.running = null;
  ui.show($('#eval-progress'), false);

  if (!records.length) {
    err.textContent = '추론에 성공한 이미지가 없습니다.';
    err.hidden = false;
    return;
  }

  app.eval.records = records;
  app.eval.files = files;
  store.saveHoldout({
    n: records.length, at: new Date().toISOString(),
    // 파일 핸들과 확률 벡터 전체는 저장하지 않습니다 (용량)
    summary: { failed, aborted },
  });

  if (failed || aborted) {
    ui.note($('#eval-labelcheck'), 'warn',
      `${lines.join('<br>')}<br><b>${aborted ? '중단됨 · ' : ''}${failed}장은 읽기/추론에 실패해 제외되었습니다.</b>`);
  }

  renderEvalResults();
  refreshStorageTiles();
}

function renderEvalResults() {
  const recs = app.eval.records;
  app.eval.cm = confusionMatrix(recs, CLASS_IDS);
  ui.renderConfusion($('#eval-cm'), app.eval.cm);
  ui.renderPerClass($('#eval-perclass'), perClassMetrics(app.eval.cm));

  const conf = topConfusions(app.eval.cm, 5);
  ui.note($('#eval-confusions'), conf.length ? 'warn' : 'ok',
    conf.length
      ? '<b>가장 자주 헷갈리는 조합</b><br>' + conf.map(c =>
          `${ui.esc(labelOf(c.trueId))} → ${ui.esc(labelOf(c.predId))} · ${c.count}건`).join('<br>')
        + '<br><span class="muted">이 조합이 임상적으로 구분할 필요가 없다면 두 클래스를 합치는 것이 정확도와 신뢰도를 동시에 올립니다.</span>'
      : '오분류가 없습니다. 표본이 적거나 과적합을 의심해 보세요.');

  ui.renderGallery($('#fail-gallery'), worstFailures(recs, 24), app.eval.files);
  refreshEvalMetrics();
  ui.show($('#eval-results'), true);
}

function refreshEvalMetrics() {
  const recs = app.eval.records;
  if (!recs.length) return;

  const th = +$('#eval-th').value;
  const policy = $('#eval-policy').value;

  app.eval.sweep = thresholdSweep(recs, POSITIVE_IDS, policy);
  const m = binaryMetrics(recs, POSITIVE_IDS, th, policy);
  const acc = overallAccuracy(app.eval.cm);
  const priority = PROJECT.copy.metricPriority;

  ui.renderTiles($('#eval-tiles'), [
    { k: '표본', v: `${recs.length}`, sub: 'TM에 넣지 않은 홀드아웃' },
    { k: '전체 정확도', v: fmtPct(acc), sub: 'argmax 기준 · 임계값 무시' },
    { k: '민감도', v: fmtPct(m.sensitivity),
      sub: `놓친 이상 ${m.fn}건`, tone: priority === 'sensitivity' ? 'accent' : undefined },
    { k: '특이도', v: fmtPct(m.specificity),
      sub: `위양성 ${m.fp}건`, tone: priority === 'specificity' ? 'accent' : undefined },
    { k: '양성 예측도', v: fmtPct(m.ppv), sub: '양성이라 했을 때 맞을 확률' },
    { k: '보류율', v: fmtPct(m.holdRate),
      sub: `${m.held}건이 사람에게 넘어감`,
      tone: m.holdRate > 0.4 ? 'warn' : undefined },
  ]);

  ui.renderSweep($('#eval-sweep'), app.eval.sweep, th);
}

function labelOf(id) {
  return CLASS_IDS.includes(id) ? (PROJECT.classes.find(c => c.id === id)?.label ?? id) : id;
}

/* ── 이력 ─────────────────────────────────────────────────────────────── */

function bindHistory() {
  $('#hist-csv').addEventListener('click', () => {
    const entries = hist.list();
    if (!entries.length) return;
    hist.download(`판독이력-${stamp()}.csv`, hist.toCsv(entries));
  });
  $('#hist-clear').addEventListener('click', () => {
    if (!confirm('판독 이력을 모두 삭제합니다. 계속할까요?')) return;
    hist.clear();
    refreshHistoryView();
    refreshStorageTiles();
  });
}

function refreshHistoryView() {
  const entries = hist.list();
  const s = hist.summarize(entries);
  ui.renderTiles($('#hist-tiles'), [
    { k: '총 판독', v: `${s.total}건` },
    { k: '판정 성립', v: `${s.byStatus.ok ?? 0}건`, tone: 'ok' },
    { k: '판단 보류', v: `${s.byStatus.hold ?? 0}건`, tone: 'warn' },
    { k: '구분 어려움', v: `${s.byStatus.ambiguous ?? 0}건` },
    { k: '무효 입력', v: `${(s.byStatus.invalid ?? 0) + (s.byStatus.gated ?? 0)}건` },
  ]);
  ui.renderHistory($('#hist-table'), entries);
}

function stamp() {
  const d = new Date();
  const p = n => String(n).padStart(2, '0');
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}`;
}

/* ── 시작 ─────────────────────────────────────────────────────────────── */

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}

// 콘솔에서 상태를 들여다볼 수 있게 노출합니다 (디버깅용)
window.__mtk = app;
