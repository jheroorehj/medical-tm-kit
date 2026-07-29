/* main.js — 부팅과 이벤트 배선 (RADIOLENS)
 * ---------------------------------------------------------------------------
 * 판정 로직도 렌더 로직도 없습니다. 각 모듈을 연결하는 배선만 있습니다.
 *
 *   추론 1회의 흐름:
 *     프레임 → 품질 게이트 → 224 변환 → 게이트키퍼 → 주분류기(+앙상블)
 *          → 시간축 안정화 → 판정 → 렌더 → 이력
 */

import { PROJECT, CLASS_IDS, POSITIVE_IDS, classOf, datasetFilled } from '../project.config.js';
import * as store from './store.js';
import * as ui from './ui.js';
import * as hist from './history.js';
import { ModelRegistry, SLOT } from './models.js';
import { runGate, runClassify, decide, toSquareCanvas, pct } from './infer.js';
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

/* ── 상태 ─────────────────────────────────────────────────────────────── */
const app = {
  registry: new ModelRegistry(),
  settings: store.loadSettings(PROJECT),
  calibration: store.loadCalibration(),
  urls: store.loadModelUrls(),
  gate: { ...PROJECT.models.gatekeeper, passClasses: [...PROJECT.models.gatekeeper.passClasses] },
  ens: [],
  stabilizer: null,
  cam: { stream: null, live: false, timer: null, busy: false, qTimer: null },
  work: null,
  lastDecision: null,
  lastQuality: null,
  lastUploadName: '',
  samples: [],
  heatToken: null,
  eval: { items: [], records: [], files: null, cm: null, sweep: [], token: null },
  cmdk: { open: false, sel: 0, items: [] },
};

/* ── 부팅 ─────────────────────────────────────────────────────────────── */

function boot() {
  app.work = $('#work-canvas');

  applyCopy();
  ui.renderLanding();
  ui.renderDataset('#ds-body', '#ds-license');
  ui.renderDataset('#ds-body-2', '#ds-license-2');
  ui.renderLimitations('#ds-limits');
  loadSamples();
  restoreToUi();

  bindNav();
  bindCmdk();
  bindSource();
  bindCamera();
  bindUpload();
  bindHeatmap();
  bindEval();
  bindHistory();
  bindSettings();
  bindDetail();
  bindHelp();

  app.registry.onChange(onRegistry);
  app.stabilizer = new TemporalStabilizer({
    window: app.settings.temporalWindow,
    confirmFrames: app.settings.confirmFrames,
    holdThreshold: app.settings.holdThreshold,
  });

  ui.resetVerdict();
  applyWebcamMode();
  // 평가 탭도 데이터 없는 상태의 자리를 채워 둡니다
  ui.renderTiles(null, null);
  ui.renderLiveMetrics(null);
  ui.renderSweep([], app.settings.holdThreshold, null);
  readHash();
  onRegistry();
  refreshStore();
  refreshCalibrationUi();
  autoLoad();
}

function applyCopy() {
  const m = PROJECT.meta, c = PROJECT.copy;
  document.title = `${m.brand} · 판독 콘솔`;
  ui.setText($('#brand-name'), m.brand);
  ui.setText($('#brand-tag'), m.tagline);
  ui.setText($('#console-title'), m.title);
  ui.setText($('#console-badge'), m.badge);
  ui.setHtml($('#hero-title'), c.heroTitle);
  ui.setText($('#hero-lead'), c.heroLead);
  ui.setHtml($('#safety-long'), c.safetyLong);
  ui.setHtml($('#safety-note'), c.safetyNote);
  ui.setHtml($('#privacy-note'), c.privacyNote);

  // 가이드 박스를 모델이 실제로 보는 영역과 일치시킵니다
  const roi = `${Math.round(app.settings.inferRoi * 100)}%`;
  $('#cam-guide').style.width = roi;
  $('#up-guide').style.width = roi;
  ui.setText($('#roi-label'), `ROI ${app.settings.inferRoi.toFixed(2)} · 224px`);
  ui.setText($('#roi-readout'), app.settings.inferRoi.toFixed(2));
}

function restoreToUi() {
  $('#primary-url').value = app.urls.primary || PROJECT.models.primary.url || '';
  $('#gate-url').value = app.urls.gatekeeper || PROJECT.models.gatekeeper.url || '';
  app.ens = (app.urls.ensemble?.length ? app.urls.ensemble : PROJECT.models.ensemble)
    .map(e => ({ label: '앙상블', weight: 1, ...e }));

  app.gate.enabled = app.urls.gateEnabled ?? PROJECT.models.gatekeeper.enabled;
  app.gate.passClasses = app.urls.gatePass ?? PROJECT.models.gatekeeper.passClasses;
  app.gate.minConfidence = app.urls.gateMin ?? PROJECT.models.gatekeeper.minConfidence;

  setSwitch($('#gate-switch'), app.gate.enabled);
  $('#gate-pass').value = app.gate.passClasses.join(', ');
  $('#gate-min').value = app.gate.minConfidence;
  ui.setText($('#gate-min-out'), app.gate.minConfidence.toFixed(2));

  ui.renderEnsembleList(app.ens);
  bindEnsembleRows();
  syncSettings();
}

function syncSettings() {
  const s = app.settings;
  const set = (id, v, out) => {
    const el = $(id); if (el) el.value = v;
    if (out !== undefined) ui.setText($(`${id}-out`), out);
  };
  set('#s-hold', s.holdThreshold, s.holdThreshold.toFixed(2));
  set('#s-margin', s.marginThreshold, s.marginThreshold.toFixed(2));
  set('#s-window', s.temporalWindow, String(s.temporalWindow));
  set('#s-confirm', s.confirmFrames, String(s.confirmFrames));
  set('#s-interval', s.inferIntervalMs, `${s.inferIntervalMs} ms`);
  set('#s-blur', s.blurMin, String(s.blurMin));
  $('#s-lumamin').value = s.lumaMin;
  $('#s-lumamax').value = s.lumaMax;
  ui.setText($('#s-luma-out'), `${s.lumaMin} – ${s.lumaMax}`);
  setSwitch($('#q-switch'), s.enabled);
  setSwitch($('#cal-switch'), s.useCalibration);
  $('#eval-thr').value = s.holdThreshold;
  ui.setText($('#eval-thr-out'), s.holdThreshold.toFixed(2));
  ui.setText($('#stamp-thr'), `THRESHOLD ${s.holdThreshold.toFixed(2)}`);
  updateMarginHint();
  updateIntervalLabel();
}

/**
 * 마진 관문(관문3)이 현재 조합에서 발동 가능한지 진단합니다.
 *
 * softmax 합이 1이므로 2위 ≤ 1 − 1위, 따라서 마진 ≥ 2 × 1위 − 1.
 * 1위가 0.70이면 마진은 최소 0.40이므로 기준 0.15는 절대 걸리지 않습니다 —
 * 관문2(판단 보류)가 항상 먼저 걸러냅니다. 임계값을 낮춘 민감도 우선
 * 운영점에서만 활성화됩니다. 슬라이더를 움직였는데 아무 변화가 없는 이유를
 * 사용자에게 알려 주는 것이 이 진단의 목적입니다.
 */
function updateMarginHint() {
  const { holdThreshold: h, marginThreshold: m } = app.settings;
  const minPossible = Math.max(0, 2 * h - 1);
  const reachable = m > minPossible;
  const box = $('#margin-hint');
  box.dataset.tone = reachable ? 'ok' : 'warn';
  ui.setText($('#margin-hint-tag'), reachable ? 'REACHABLE · 도달 가능' : 'UNREACHABLE · 도달 불가');
  ui.setText($('#margin-hint-text'), reachable
    ? `현재 임계값 ${h.toFixed(2)} 에서 마진 관문은 도달 가능합니다 `
      + `(최소 마진 ${pct(minPossible, 0)} < 기준 ${pct(m, 0)}). `
      + `1위가 ${pct(h, 0)}~${pct((m + 1) / 2, 0)} 구간일 때 '구분 어려움'이 발동합니다.`
    : `softmax 합이 1이므로 1위가 ${h.toFixed(2)} 이상이면 마진은 최소 ${pct(minPossible, 0)} 입니다. `
      + `기준 ${pct(m, 0)} 은 절대 발동하지 않습니다 — 보류 임계값을 ${pct((m + 1) / 2, 0)} `
      + `아래로 내리거나 이 값을 ${pct(minPossible, 0)} 위로 올리세요.`);
}

function updateIntervalLabel() {
  ui.setText($('#interval-label'),
    `${app.cam.live ? 'RUNNING' : 'IDLE'} · ${app.settings.inferIntervalMs}ms · BUSY-GUARDED`);
}

function persistSettings() {
  store.saveSettings(app.settings);
  app.stabilizer?.configure({
    window: app.settings.temporalWindow,
    confirmFrames: app.settings.confirmFrames,
    holdThreshold: app.settings.holdThreshold,
  });
  ui.setText($('#stamp-thr'), `THRESHOLD ${app.settings.holdThreshold.toFixed(2)}`);
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

async function autoLoad() {
  if ($('#primary-url').value.trim()) await loadPrimary();
  if (app.gate.enabled && $('#gate-url').value.trim()) await loadGate();
  for (let i = 0; i < app.ens.length; i++) {
    if (app.ens[i].url) await app.registry.load(SLOT.ens(i), app.ens[i].url, app.ens[i]);
  }
}

/* ── 모델 상태 ────────────────────────────────────────────────────────── */

function onRegistry() {
  const reg = app.registry;
  const st = reg.status(SLOT.PRIMARY);
  const ready = reg.has(SLOT.PRIMARY);

  ui.setPill('#primary-state', st.state,
    st.state === 'ok' ? st.message : st.state === 'error' ? '실패' : st.state === 'loading' ? '로딩…' : '미로드');
  const gst = reg.status(SLOT.GATE);
  ui.setPill('#gate-state', gst.state,
    gst.state === 'ok' ? gst.message : gst.state === 'error' ? '실패' : gst.state === 'loading' ? '로딩…' : '미로드');
  app.ens.forEach((_, i) => {
    const s = reg.status(SLOT.ens(i));
    ui.setPill(`#ens-state-${i}`, s.state,
      s.state === 'ok' ? s.message : s.state === 'error' ? '실패' : s.state === 'loading' ? '로딩…' : '미로드');
  });

  // 헤더 배지
  ui.setPill('#pill-primary', ready ? 'ok' : st.state,
    ready ? `PRIMARY LOADED · ${reg.get(SLOT.PRIMARY).labels.length} CLASSES`
          : st.state === 'loading' ? 'PRIMARY 로딩…'
          : st.state === 'error' ? 'PRIMARY 로드 실패' : 'PRIMARY 미로드');
  const gateOn = app.gate.enabled && reg.has(SLOT.GATE);
  ui.setPill('#pill-gate', gateOn ? 'ok' : 'idle', gateOn ? 'GATE ON' : 'GATE OFF');

  // 앙상블 라벨
  const n = reg.classifierSlots().length;
  ui.setText($('#ensemble-label'),
    n === 0 ? 'NO MODEL' : n === 1 ? 'SINGLE MODEL' : `ENSEMBLE MEAN · ${n} MODELS`);

  // 버튼 가용성
  $('#up-run').disabled = !ready || !$('#up-img').getAttribute('src');
  $('#cam-live').disabled = !ready || !app.cam.stream;
  $('#cam-snap').disabled = !ready || !app.cam.stream;
  $('#heat-run').disabled = !ready || !app.lastDecision || !!app.heatToken;
  $('#eval-rerun').disabled = !ready || !app.eval.items.length;

  if (ready) {
    ui.renderClassCheck(reg.verifyClasses(CLASS_IDS), reg.verifyEnsembleConsistency());
  } else {
    ui.renderClassCheck({ missing: [], extra: [], modelLabels: [], ok: false }, null);
  }

  if (curTab === 'help') refreshHelp();
}

async function loadPrimary() {
  persistUrls();
  await app.registry.load(SLOT.PRIMARY, $('#primary-url').value.trim(),
    { label: PROJECT.models.primary.label });
}

async function loadGate() {
  persistUrls();
  await app.registry.load(SLOT.GATE, $('#gate-url').value.trim(), { label: '게이트키퍼' });
}

/* ── 네비게이션 ───────────────────────────────────────────────────────── */

function bindNav() {
  ui.$$('[data-view]').forEach(b => b.addEventListener('click', () => gotoView(b.dataset.view)));
  ui.$$('[data-goto]').forEach(b => b.addEventListener('click', () => {
    gotoView(b.dataset.goto);
    if (b.dataset.tab) gotoTab(b.dataset.tab);
  }));
  ui.$$('.tabbtn').forEach(b => b.addEventListener('click', () => gotoTab(b.dataset.tab)));
  window.addEventListener('hashchange', readHash);
}

const VIEWS = ['landing', 'console'];
const TABS = ['read', 'eval', 'hist', 'set', 'help'];
let curView = 'landing', curTab = 'read';

function gotoView(name, updateHash = true) {
  curView = VIEWS.includes(name) ? name : 'landing';
  $('#nav-landing').classList.toggle('on', curView === 'landing');
  $('#nav-console').classList.toggle('on', curView === 'console');
  $('#view-landing').classList.toggle('on', curView === 'landing');
  $('#view-console').classList.toggle('on', curView === 'console');
  if (curView === 'landing') stopLive();
  if (updateHash) writeHash();
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function gotoTab(name, updateHash = true) {
  curTab = TABS.includes(name) ? name : 'read';
  ui.$$('.tabbtn').forEach(b => b.setAttribute('aria-selected', String(b.dataset.tab === curTab)));
  ui.$$('.tab').forEach(t => t.classList.toggle('on', t.id === `tab-${curTab}`));
  if (curTab !== 'read') stopLive();
  if (curTab === 'hist') refreshHistory();
  if (curTab === 'set') refreshStore();
  if (curTab === 'help') refreshHelp();
  if (updateHash) writeHash();
}

/* 딥링크 — #console/eval 처럼 특정 화면을 바로 열 수 있습니다.
 * 시연 중 원하는 탭을 즉시 띄우거나 북마크해 두는 용도입니다. */
function writeHash() {
  const h = curView === 'console' ? `#console/${curTab}` : '#overview';
  if (location.hash !== h) history.replaceState(null, '', h);
}

function readHash() {
  const [view, tab] = location.hash.replace(/^#/, '').split('/');
  if (view === 'console') {
    gotoView('console', false);
    gotoTab(tab || 'read', false);
  } else if (view === 'overview') {
    gotoView('landing', false);
    gotoTab('read', false);
  } else {
    // 해시가 없으면 설정을 따릅니다 — 데모 시간이 짧으면 콘솔부터 열 수 있습니다
    gotoView(PROJECT.ui?.showLandingFirst === false ? 'console' : 'landing', false);
    gotoTab('read', false);
  }
}

/* ── 커맨드 팔레트 (⌘K) ──────────────────────────────────────────────── */

function commands() {
  return [
    { label: '판독 화면', en: 'READ', run: () => { gotoView('console'); gotoTab('read'); } },
    { label: '성능 평가', en: 'EVAL', run: () => { gotoView('console'); gotoTab('eval'); } },
    { label: '판독 이력', en: 'HISTORY', run: () => { gotoView('console'); gotoTab('hist'); } },
    { label: '설정', en: 'SETTINGS', run: () => { gotoView('console'); gotoTab('set'); } },
    { label: '도움말 · 사용법', en: 'HELP', run: () => { gotoView('console'); gotoTab('help'); } },
    { label: '환경 자동 점검', en: 'ENV CHECK', run: () => { gotoView('console'); gotoTab('help'); scrollToSec('diag'); } },
    { label: '오류 해결 사전', en: 'TROUBLESHOOTING', run: () => { gotoView('console'); gotoTab('help'); scrollToSec('errors'); } },
    { label: '소개 페이지', en: 'OVERVIEW', run: () => gotoView('landing') },
    ...(PROJECT.ui?.enableWebcam ? [
      { label: '카메라 시작', en: 'START CAMERA', run: startCamera },
      { label: '카메라 중지', en: 'STOP CAMERA', run: stopCamera },
      { label: '실시간 판독 켜기/끄기', en: 'TOGGLE LIVE', run: toggleLive },
      { label: '캡처 & 판독', en: 'CAPTURE', run: () => runOnce('webcam') },
    ] : [
      { label: '이 이미지 판독', en: 'RUN', run: () => runOnce('upload') },
      { label: '업로드 초기화', en: 'RESET UPLOAD', run: resetUpload },
    ]),
    { label: '근거 분석 실행', en: 'OCCLUSION MAP', run: runHeatmap },
    { label: '보정표 생성', en: 'BUILD CALIBRATION', run: buildCal },
    { label: '이력 CSV 내보내기', en: 'EXPORT HISTORY', run: exportHistory },
    { label: '평가 결과 CSV 내보내기', en: 'EXPORT EVAL', run: exportEval },
    { label: '전체 초기화', en: 'WIPE ALL', run: wipeAll },
  ];
}

function bindCmdk() {
  $('#cmdk-open').addEventListener('click', openCmdk);
  $('#cmdk-backdrop').addEventListener('click', e => {
    if (e.target === $('#cmdk-backdrop')) closeCmdk();
  });
  $('#cmdk-input').addEventListener('input', filterCmdk);
  $('#cmdk-list').addEventListener('click', e => {
    const btn = e.target.closest('[data-cmd]');
    if (btn) runCmd(+btn.dataset.cmd);
  });

  document.addEventListener('keydown', e => {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
      e.preventDefault();
      app.cmdk.open ? closeCmdk() : openCmdk();
      return;
    }
    if (!app.cmdk.open) return;
    if (e.key === 'Escape') { e.preventDefault(); closeCmdk(); }
    else if (e.key === 'ArrowDown') { e.preventDefault(); moveCmdk(1); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); moveCmdk(-1); }
    else if (e.key === 'Enter') { e.preventDefault(); runCmd(app.cmdk.sel); }
  });
}

function openCmdk() {
  app.cmdk.open = true;
  app.cmdk.sel = 0;
  $('#cmdk-input').value = '';
  filterCmdk();
  ui.show($('#cmdk-backdrop'), true);
  $('#cmdk-input').focus();
}

function closeCmdk() {
  app.cmdk.open = false;
  ui.show($('#cmdk-backdrop'), false);
}

function filterCmdk() {
  const q = $('#cmdk-input').value.trim().toLowerCase();
  app.cmdk.items = commands().filter(c =>
    !q || c.label.toLowerCase().includes(q) || c.en.toLowerCase().includes(q));
  app.cmdk.sel = 0;
  ui.renderCmdk(app.cmdk.items, app.cmdk.sel, q);
}

function moveCmdk(d) {
  const n = app.cmdk.items.length;
  if (!n) return;
  app.cmdk.sel = (app.cmdk.sel + d + n) % n;
  ui.renderCmdk(app.cmdk.items, app.cmdk.sel, $('#cmdk-input').value);
}

function runCmd(i) {
  const c = app.cmdk.items[i];
  closeCmdk();
  c?.run?.();
}

/**
 * 웹캠 사용 여부를 화면에 반영합니다.
 *
 * 이 대회의 기준 시나리오는 이미지 업로드입니다 (강의 8단원). 웹캠을 끄면
 * 입력 선택 탭과 실시간 전용 설정을 화면에서 제거합니다 —
 * 대회 1순위 기준이 "데모로 보여지지 않는 기능은 크게 감점" 이기 때문입니다.
 *
 * 코드는 그대로 남아 있으므로 enableWebcam 을 true 로 바꾸면 즉시 복귀합니다.
 */
function applyWebcamMode() {
  const on = PROJECT.ui?.enableWebcam === true;
  ui.show($('#src-seg'), on);              // WEBCAM / UPLOAD 선택 탭
  ui.show($('#realtime-fields'), on);      // 이동평균·확정조건·추론간격
  if (!on) stopCamera();
  selectSource(on ? 'cam' : 'upload');
}

/* ── 입력 소스 ────────────────────────────────────────────────────────── */

function bindSource() {
  ui.$$('[data-src]').forEach(b =>
    b.addEventListener('click', () => selectSource(b.dataset.src)));
}

function selectSource(kind) {
  ui.$$('[data-src]').forEach(x => {
    const on = x.dataset.src === kind;
    x.classList.toggle('on', on);
    x.setAttribute('aria-selected', String(on));
  });
  ui.show($('#pane-cam'), kind === 'cam');
  ui.show($('#pane-upload'), kind === 'upload');
  if (kind === 'upload') stopCamera();
}

/* ── 웹캠 ─────────────────────────────────────────────────────────────── */

function bindCamera() {
  $('#cam-start').addEventListener('click', startCamera);
  $('#cam-stop').addEventListener('click', stopCamera);
  $('#cam-snap').addEventListener('click', () => runOnce('webcam'));
  $('#cam-live').addEventListener('click', toggleLive);
  // 페이지를 떠날 때 카메라 LED가 남지 않도록 확실히 정리합니다
  window.addEventListener('pagehide', stopCamera);
}

async function startCamera() {
  const err = $('#cam-error');
  if (app.cam.stream) return;
  if (!navigator.mediaDevices?.getUserMedia) {
    err.textContent = '이 브라우저는 카메라 API를 지원하지 않습니다.';
    ui.show(err, true);
    return;
  }
  try {
    app.cam.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
  } catch (e) {
    ui.setHtml(err, location.protocol === 'file:'
      ? '<b>file:// 로 열면 카메라를 쓸 수 없습니다.</b> <code>./serve.sh</code> 실행 후 http://localhost:8000 으로 접속하세요.'
      : e?.name === 'NotAllowedError'
        ? '카메라 접근이 거부되었습니다. 브라우저 주소창의 권한 설정을 확인하세요.'
        : `카메라를 열 수 없습니다 (${e?.name ?? '알 수 없는 오류'}).`);
    ui.show(err, true);
    return;
  }

  ui.show(err, false);
  const vid = $('#cam-video');
  vid.srcObject = app.cam.stream;
  await vid.play().catch(() => {});

  ui.show(vid, true);
  ui.show($('#cam-hatch'), false);
  ui.show($('#cam-idle'), false);
  ui.show($('#cam-guide'), true);
  ui.show($('#cam-quality'), true);
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
  ui.show($('#cam-hatch'), true);
  ui.show($('#cam-idle'), true);
  ui.show($('#cam-guide'), false);
  ui.show($('#cam-quality'), false);
  ui.show($('#live-badge'), false);
  ui.show($('#cam-sweep'), false);
  $('#cam-start').disabled = false;
  $('#cam-stop').disabled = true;
  $('#cam-snap').disabled = true;
  $('#cam-live').disabled = true;
  ui.renderCoach(null, false);
}

/* 품질 모니터 — 추론과 무관하게 항상 돌면서 촬영을 코칭합니다 */
function startQualityMonitor() {
  stopQualityMonitor();
  app.cam.qTimer = setInterval(() => {
    const m = measureFrame($('#cam-video'), app.settings.inferRoi);
    app.lastQuality = m;
    const ev = evaluateQuality(m, app.settings);
    ui.renderQuality('q', m, app.settings);
    ui.renderCoach(ev, true);
    // 설정 탭에서도 실시간 수치를 보며 임계값을 맞출 수 있게 합니다
    const live = $('#q-live');
    if (m) {
      ui.setPill('#q-live', ev.ok ? 'ok' : 'error',
        `LIVE FOCUS ${m.blur.toFixed(0)} · LUMA ${m.luma.toFixed(0)}`);
      ui.show(live, true);
    }
  }, 250);
}

function stopQualityMonitor() {
  if (app.cam.qTimer) { clearInterval(app.cam.qTimer); app.cam.qTimer = null; }
  ui.show($('#q-live'), false);
}

function toggleLive() {
  if (app.cam.live) stopLive(); else startLive();
}

function startLive() {
  if (!app.cam.stream || !app.registry.has(SLOT.PRIMARY)) return;
  app.cam.live = true;
  app.stabilizer.reset();
  const btn = $('#cam-live');
  btn.textContent = '■ 실시간 판독 중';
  btn.classList.add('btn-live');
  btn.setAttribute('aria-pressed', 'true');
  ui.show($('#live-badge'), true);
  ui.show($('#cam-sweep'), true);
  updateIntervalLabel();
  liveTick();
}

function stopLive() {
  app.cam.live = false;
  if (app.cam.timer) { clearTimeout(app.cam.timer); app.cam.timer = null; }
  const btn = $('#cam-live');
  if (btn) {
    btn.textContent = '▶ 실시간 판독';
    btn.classList.remove('btn-live');
    btn.setAttribute('aria-pressed', 'false');
  }
  ui.show($('#live-badge'), false);
  ui.show($('#cam-sweep'), false);
  updateIntervalLabel();
}

/** setInterval 을 쓰지 않습니다 — 추론이 간격보다 길면 큐가 쌓입니다. */
async function liveTick() {
  if (!app.cam.live) return;
  if (!app.cam.busy) {
    app.cam.busy = true;
    try { await runOnce('live'); }
    catch (e) { console.warn('[live]', e); }
    finally { app.cam.busy = false; }
  }
  app.cam.timer = setTimeout(liveTick, app.settings.inferIntervalMs);
}

/* ── 이미지 업로드 ────────────────────────────────────────────────────── */

function bindUpload() {
  const dz = $('#up-drop'), file = $('#up-file');
  dz.addEventListener('click', () => file.click());
  dz.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); file.click(); }
  });
  dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('drag'); });
  dz.addEventListener('dragleave', () => dz.classList.remove('drag'));
  dz.addEventListener('drop', e => {
    e.preventDefault(); dz.classList.remove('drag');
    const f = e.dataTransfer.files[0];
    if (f?.type.startsWith('image/')) loadUpload(f);
    else failUpload('이미지 파일만 업로드할 수 있습니다.');
  });
  file.addEventListener('change', e => { if (e.target.files[0]) loadUpload(e.target.files[0]); });
  $('#up-run').addEventListener('click', () => runOnce('upload'));
  $('#up-reset').addEventListener('click', resetUpload);
}

/**
 * 내장 샘플을 불러옵니다.
 * web/samples.js 가 있으면 그것을(data URI) 우선 사용하고, 없으면 config의 samples를 씁니다.
 * 강의 10-3의 fallback 요구사항이자, 파일 없이 데모를 돌릴 수 있게 하는 안전장치입니다.
 */
function loadSamples() {
  const fromFile = Array.isArray(window.MTK_SAMPLES) ? window.MTK_SAMPLES : null;
  app.samples = (fromFile?.length ? fromFile : (PROJECT.samples ?? []))
    .filter(s => s && s.src);
  ui.renderSamples(app.samples);
  $('#sample-list')?.addEventListener('click', e => {
    const btn = e.target.closest('[data-sample]');
    if (btn) useSample(+btn.dataset.sample, false);
  });
}

/** 샘플 이미지를 미리보기에 올립니다. @param {boolean} auto fallback으로 자동 로드된 경우 */
function useSample(i, auto) {
  const s = app.samples[i];
  if (!s) return;
  const img = $('#up-img');
  const old = img.getAttribute('src');
  if (old?.startsWith('blob:')) URL.revokeObjectURL(old);
  app.lastUploadName = `샘플: ${s.label}`;
  img.onload = () => {
    ui.show($('#up-stage'), true);
    ui.show($('#up-drop'), false);
    ui.show($('#up-quality'), true);
    $('#up-run').disabled = !app.registry.has(SLOT.PRIMARY);
    const m = measureFrame(img, app.settings.inferRoi);
    ui.renderQuality('uq', m, app.settings);
    if (auto) {
      const el = $('#up-error');
      ui.setHtml(el, `<b>업로드에 실패해 내장 샘플을 대신 불러왔습니다.</b> `
        + `(${ui.esc(s.label)}) 그대로 판독을 진행할 수 있습니다.`);
      ui.show(el, true);
    } else {
      ui.show($('#up-error'), false);
    }
  };
  img.onerror = () => showUpErr('샘플 이미지를 불러올 수 없습니다.');
  img.src = s.src;
}

function showUpErr(msg) {
  const el = $('#up-error');
  el.textContent = msg;
  ui.show(el, true);
}

function loadUpload(f) {
  const img = $('#up-img');
  const url = URL.createObjectURL(f);
  app.lastUploadName = f.name;
  img.onload = () => {
    ui.show($('#up-stage'), true);
    ui.show($('#up-drop'), false);
    ui.show($('#up-quality'), true);
    ui.show($('#up-error'), false);
    $('#up-run').disabled = !app.registry.has(SLOT.PRIMARY);
    const m = measureFrame(img, app.settings.inferRoi);
    ui.renderQuality('uq', m, app.settings);
    const ev = evaluateQuality(m, app.settings);
    if (!ev.ok) showUpErr(`품질 경고: ${ev.coach} — 결과 신뢰도가 낮을 수 있습니다.`);
  };
  img.onerror = () => { URL.revokeObjectURL(url); failUpload('이미지를 읽을 수 없습니다.'); };
  img.src = url;
}

/**
 * 업로드 실패 처리 — 내장 샘플이 있으면 자동으로 대체합니다.
 * 강의 10-3: "업로드가 실패하면 내장 샘플 이미지가 자동으로 올라가게".
 * 데모가 중간에 끊기는 것이 실패 3대 요인 1위이므로 반드시 필요한 대비입니다.
 */
function failUpload(msg) {
  if (app.samples.length) {
    console.warn('[upload]', msg, '→ 내장 샘플로 대체');
    useSample(0, true);
  } else {
    showUpErr(msg);
  }
}

function resetUpload() {
  const img = $('#up-img');
  const old = img.getAttribute('src');
  if (old?.startsWith('blob:')) URL.revokeObjectURL(old);
  // ★ src='' 는 문서 URL로 해석되어 항상 truthy 입니다. 속성을 지워야 합니다.
  img.removeAttribute('src');
  app.lastUploadName = '';
  ui.show($('#up-stage'), false);
  ui.show($('#up-drop'), true);
  ui.show($('#up-quality'), false);
  ui.show($('#up-error'), false);
  $('#up-file').value = '';
  $('#up-run').disabled = true;          // 원본 템플릿에서 빠져 있던 처리
  app.lastDecision = null;
  $('#heat-run').disabled = true;
  ui.resetVerdict();
}

/* ── 추론 1회 ─────────────────────────────────────────────────────────── */

async function runOnce(kind) {
  if (!app.registry.has(SLOT.PRIMARY)) return;

  const live = kind === 'live';
  const source = kind === 'upload' ? $('#up-img') : $('#cam-video');
  const errEl = kind === 'upload' ? $('#up-error') : $('#cam-error');

  // ① 품질 게이트
  const m = measureFrame(source, app.settings.inferRoi);
  const qev = evaluateQuality(m, app.settings);
  if (kind !== 'upload') {
    ui.renderQuality('q', m, app.settings);
    ui.renderCoach(qev, true);
  }
  if (!qev.ok) {
    if (live) return;                          // 실시간: 조용히 다음 프레임을 기다림
    if (!m) { errEl.textContent = '영상을 읽을 수 없습니다.'; ui.show(errEl, true); return; }
    errEl.textContent = `품질 경고: ${qev.coach} — 결과 신뢰도가 낮을 수 있습니다.`;
    ui.show(errEl, true);
  } else {
    ui.show(errEl, false);
  }

  // ② 학습 전처리와 동일한 방식으로 224 정사각 캔버스 생성
  toSquareCanvas(source, app.settings.inferRoi, 224, app.work);

  try {
    // ③ 게이트키퍼 → ④ 주분류기(+앙상블)
    const gate = await runGate(app.registry, app.work, app.gate);
    const { probs } = await runClassify(app.registry, app.work);

    // ⑤ 시간축 안정화 (실시간 모드에서만)
    let temporal = null, effective = probs;
    if (live) {
      temporal = app.stabilizer.push(probs);
      effective = temporal.smoothed;
    }

    // ⑥ 판정
    const d = decide(effective, app.settings, {
      gate, calibration: app.calibration, copy: PROJECT.copy,
    });

    ui.renderVerdict(d, {
      live, temporal,
      confirmFrames: app.settings.confirmFrames,
      threshold: app.settings.holdThreshold,
      marginThreshold: app.settings.marginThreshold,
      jitter: temporal?.jitter,
      flash: !live || temporal?.justConfirmed,
    });

    app.lastDecision = d;
    $('#heat-run').disabled = !!app.heatToken;

    // ⑦ 이력 — 실시간은 확정된 순간에만 기록 (초당 여러 건이 쌓이지 않도록)
    if (!live || temporal?.justConfirmed) {
      const notes = [];
      if (live) notes.push(`확정 ${app.settings.confirmFrames}프레임`);
      if (kind === 'upload' && app.lastUploadName) notes.push(app.lastUploadName);
      if (!qev.ok) notes.push(`품질 경고 후 진행 (${qev.issues[0]?.code ?? '-'})`);
      hist.record({
        source: kind === 'live' ? 'webcam·live' : kind === 'webcam' ? 'webcam·capture' : 'upload',
        decision: d, quality: m, thumb: hist.makeThumb(app.work),
        settings: app.settings, note: notes.join(' · '),
      });
    }
  } catch (e) {
    console.error('[infer]', e);
    errEl.textContent = '추론 중 오류가 발생했습니다. 콘솔을 확인하세요.';
    ui.show(errEl, true);
  }
}

/* ── 판독 화면의 전문 지표 접기 ───────────────────────────────────────── *
 * 최종 사용자가 처음 보는 것은 판정·유사도·권고·안전문구 네 가지로 충분합니다.
 * 엔트로피·jitter 같은 용어는 기본으로 접어 두고, 필요한 사람이 펼치게 합니다.
 * 선택은 저장되므로 데모 중에 한 번 펼치면 그대로 유지됩니다.
 *
 * (이 버튼은 이전까지 클릭 핸들러가 없어 눌러도 아무 일이 없었습니다.)
 * ------------------------------------------------------------------------ */

function bindDetail() {
  const btn = $('#detail-toggle');
  const box = $('#detail-block');
  if (!btn || !box) return;

  if (app.settings.detailOpen === undefined) {
    app.settings.detailOpen = PROJECT.ui?.showDetailByDefault === true;
  }

  const apply = () => {
    const on = !!app.settings.detailOpen;
    ui.show(box, on);
    btn.textContent = on ? '자세히 접기 —' : '자세히 보기 · 불확실성 5개 지표 +';
    btn.setAttribute('aria-expanded', String(on));
  };

  btn.addEventListener('click', () => {
    app.settings.detailOpen = !app.settings.detailOpen;
    persistSettings();
    apply();
  });
  apply();
}

/* ── 근거 히트맵 ──────────────────────────────────────────────────────── */

function bindHeatmap() {
  $('#heat-run').addEventListener('click', runHeatmap);
  $('#heat-abort').addEventListener('click', () => {
    if (app.heatToken) app.heatToken.aborted = true;
  });
}

async function runHeatmap() {
  if (!app.lastDecision || app.heatToken) return;
  const entry = app.registry.get(SLOT.PRIMARY);
  if (!entry) return;

  stopLive();                            // 프레임이 계속 바뀌면 무의미합니다
  const token = { aborted: false };
  app.heatToken = token;
  $('#heat-run').disabled = true;
  $('#heat-abort').disabled = false;
  ui.show($('#heat-prog'), true);

  try {
    const map = await occlusionMap(entry.model, app.work, {
      grid: 8,
      targetId: app.lastDecision.top.id,
      onProgress: (done, total) => {
        ui.setFill($('#heat-fill'), done / total);
        ui.setText($('#heat-count'), `${done} / ${total} · ${Math.round(done / total * 100)}%`);
      },
      token,
    });

    const cv = $('#heat-canvas');
    const ctx = cv.getContext('2d');
    ctx.clearRect(0, 0, cv.width, cv.height);
    ctx.drawImage(app.work, 0, 0, cv.width, cv.height);
    if (!map.aborted) drawHeatmap(cv, map);

    const it = interpretMap(map);
    const box = $('#heat-interp');
    box.dataset.tone = it.tone === 'ok' ? 'ok' : it.tone === 'danger' ? 'danger' : 'warn';
    ui.setHtml(box, `<div class="notice-k">AUTO INTERPRETATION · interpretMap()</div>
      <div class="notice-d">${ui.esc(it.text)}<br>
      <span class="mono" style="font-size:10.5px">기준 유사도 ${pct(map.base)} · 최대 하락 ${pct(Math.max(0, map.max))}
      · 대상 ${ui.esc(classOf(map.targetId).label)}${map.aborted ? ' · 중단됨' : ''}</span></div>`);
  } catch (e) {
    console.error('[saliency]', e);
    const box = $('#heat-interp');
    box.dataset.tone = 'danger';
    ui.setHtml(box, '<div class="notice-d">근거 분석 중 오류가 발생했습니다.</div>');
  } finally {
    app.heatToken = null;
    $('#heat-run').disabled = false;
    $('#heat-abort').disabled = true;
    ui.show($('#heat-prog'), false);
  }
}

/* ── 성능 평가 ────────────────────────────────────────────────────────── */

function bindEval() {
  const dz = $('#eval-drop'), input = $('#eval-files');
  dz.addEventListener('click', () => input.click());
  dz.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); }
  });
  dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('drag'); });
  dz.addEventListener('dragleave', () => dz.classList.remove('drag'));
  dz.addEventListener('drop', async e => {
    e.preventDefault(); dz.classList.remove('drag');
    startEval(await collectFromDrop(e.dataTransfer));
  });
  input.addEventListener('change', e => startEval(collectFromInput(e.target.files)));

  $('#eval-cancel').addEventListener('click', () => {
    if (app.eval.token) app.eval.token.aborted = true;
  });
  $('#eval-rerun').addEventListener('click', () => {
    if (app.eval.items.length) startEval(app.eval.items);
  });

  $('#eval-thr').addEventListener('input', () => {
    ui.setText($('#eval-thr-out'), (+$('#eval-thr').value).toFixed(2));
    refreshEvalMetrics();
  });
  $('#pol-escalate').addEventListener('click', () => setPolicy('escalate'));
  $('#pol-exclude').addEventListener('click', () => setPolicy('exclude'));

  $('#eval-apply').addEventListener('click', () => {
    app.settings.holdThreshold = +$('#eval-thr').value;
    syncSettings();
    persistSettings();
    gotoTab('read');
  });
  $('#eval-youden').addEventListener('click', () => {
    const b = bestByYouden(app.eval.sweep);
    if (b) setThrSlider(b.threshold);
  });
  $('#eval-sens95').addEventListener('click', () => {
    const t = thresholdForSensitivity(app.eval.sweep, 0.95);
    if (t) setThrSlider(t.threshold);
    else ui.setText($('#thr-advice'),
      '민감도 95%를 만족하는 임계값이 없습니다. 클래스 설계나 학습 데이터를 재검토해야 합니다.');
  });

  $('#eval-detail-toggle').addEventListener('click', () => {
    const box = $('#eval-detail');
    const on = box.hidden;
    ui.show(box, on);
    const b = $('#eval-detail-toggle');
    b.setAttribute('aria-expanded', String(on));
    b.textContent = on
      ? '혼동행렬 · 캘리브레이션 · 실패 사례 접기 —'
      : '자세히 보기 · 혼동행렬 · 캘리브레이션 · 실패 사례 +';
  });

  $('#cal-build').addEventListener('click', buildCal);
  $('#eval-csv').addEventListener('click', exportEval);
}

function setThrSlider(v) {
  $('#eval-thr').value = v;
  ui.setText($('#eval-thr-out'), v.toFixed(2));
  refreshEvalMetrics();
}

let holdPolicy = 'escalate';
function setPolicy(p) {
  holdPolicy = p;
  $('#pol-escalate').classList.toggle('on', p === 'escalate');
  $('#pol-exclude').classList.toggle('on', p === 'exclude');
  refreshEvalMetrics();
}

async function startEval(items) {
  const err = $('#eval-error');
  ui.show(err, false);

  if (!app.registry.has(SLOT.PRIMARY)) {
    ui.setHtml(err, '<b>모델이 로드되지 않았습니다.</b> 설정 탭에서 모델을 먼저 불러오세요.');
    ui.show(err, true);
    return;
  }
  if (!items.length) {
    err.textContent = '이미지 파일을 찾지 못했습니다. 클래스별 하위 폴더가 있는 폴더를 선택하세요.';
    ui.show(err, true);
    return;
  }

  app.eval.items = items;
  const folders = new Set(items.map(i => i.label)).size;
  ui.setText($('#eval-filecount'), `${items.length} FILES · ${folders} FOLDERS LOADED`);

  // 폴더명 ↔ 클래스 id 정합성 — 실습에서 가장 흔한 사고 지점
  const lc = verifyLabels(items, CLASS_IDS);
  const cards = [];
  if (!lc.unknown.length) {
    cards.push({ tone: 'ok', k: 'LABELS OK · 정합성 통과',
      d: `${folders}개 폴더가 모두 클래스 id와 일치합니다. `
         + lc.counts.map(c => `${ui.esc(c.label)} ${c.n}`).join(' · ') });
  } else {
    cards.push({ tone: 'danger', k: `${lc.unknown.length} UNKNOWN · 폴더명 불일치`,
      d: lc.unknown.map(l => `<code>${ui.esc(l)}</code>`).join(', ')
         + ' 는 클래스 id와 다릅니다 — 대소문자까지 일치해야 합니다. 실습 최다 사고 지점입니다.' });
  }
  if (lc.missing.length) {
    cards.push({ tone: 'warn', k: 'MISSING FOLDER · 폴더 없음',
      d: lc.missing.map(l => `<code>${ui.esc(l)}</code>`).join(', ') + ' 클래스의 폴더가 없습니다.' });
  }
  ui.setHtml($('#eval-labelcheck'), cards.map(c =>
    `<div class="notice" data-tone="${c.tone}"><div class="notice-k">${c.k}</div>
     <div class="notice-d">${c.d}</div></div>`).join(''));
  ui.show($('#eval-labelcheck'), true);

  stopLive();
  const token = { aborted: false };
  app.eval.token = token;
  $('#eval-cancel').disabled = false;
  $('#eval-rerun').disabled = true;
  ui.setText($('#batch-total'), `/ ${items.length} COMPLETE`);

  const { records, failed, aborted, files } = await runBatch(app.registry, items, {
    roi: app.settings.inferRoi,
    onProgress: (done, total) => {
      ui.setText($('#batch-done'), String(done));
      ui.setFill($('#batch-fill'), done / total);
    },
    token,
  });

  app.eval.token = null;
  $('#eval-cancel').disabled = true;
  $('#eval-rerun').disabled = false;
  ui.setText($('#batch-meta'),
    `YIELDS EVERY 20 FILES · ${failed} FAILED${aborted ? ' · ABORTED' : ''}`);

  if (!records.length) {
    err.textContent = '추론에 성공한 이미지가 없습니다.';
    ui.show(err, true);
    return;
  }

  app.eval.records = records;
  app.eval.files = files;
  app.eval.cm = confusionMatrix(records, CLASS_IDS);
  store.saveHoldout({ n: records.length, at: new Date().toISOString(), failed, aborted });

  ui.renderConfusion(app.eval.cm);
  ui.renderConfusions(topConfusions(app.eval.cm, 4));
  ui.renderFailures(worstFailures(records, 8), files);
  $('#cal-build').disabled = false;
  $('#eval-csv').disabled = false;
  refreshEvalMetrics();
  refreshStore();
}

function refreshEvalMetrics() {
  const recs = app.eval.records;
  if (!recs.length) return;
  const thr = +$('#eval-thr').value;

  app.eval.sweep = thresholdSweep(recs, POSITIVE_IDS, holdPolicy);
  const m = binaryMetrics(recs, POSITIVE_IDS, thr, holdPolicy);

  ui.renderTiles(m, overallAccuracy(app.eval.cm));
  ui.renderLiveMetrics(m);
  ui.renderSweep(app.eval.sweep, thr, m);

  ui.setText($('#thr-advice'), holdPolicy === 'escalate'
    ? `보류를 양성 의심으로 올렸습니다 — 임계값을 높이면 민감도가 오르고 `
      + `사람이 확인할 건수(${m.held}건)가 늘어납니다.`
    : `보류 ${m.held}건을 제외한 순수 모델 성능입니다 — `
      + `임상 운영 지표로 쓰려면 escalate 로 보세요.`);

  // 랜딩의 지표도 실제 값으로 갱신합니다
  ui.setText($('#land-sens'), fmtPct(m.sensitivity));
  ui.setText($('#land-spec'), fmtPct(m.specificity));
}

function buildCal() {
  if (!app.eval.records.length) return;
  const t = buildCalibration(toCalibrationRecords(app.eval.records));
  if (!t) {
    const el = $('#calib-note');
    el.dataset.tone = 'warn';
    ui.setHtml(el, '<div class="notice-d">표본이 부족해 보정표를 만들 수 없습니다 (최소 10건 필요).</div>');
    ui.show(el, true);
    return;
  }
  app.calibration = t;
  store.saveCalibration(t);
  refreshCalibrationUi();
  refreshStore();
}

function refreshCalibrationUi() {
  const desc = describeCalibration(app.calibration);
  ui.renderCalibration(app.calibration, desc);
  ui.setText($('#cal-hint'), app.calibration
    ? `TABLE PRESENT · ${app.calibration.n} SAMPLES`
    : 'NO TABLE — 성능 평가 탭에서 생성');
}

function exportEval() {
  if (!app.eval.records.length) return;
  hist.download(`holdout-eval-${stamp()}.csv`, recordsToCsv(app.eval.records, CLASS_IDS));
}

/* ── 이력 ─────────────────────────────────────────────────────────────── */

function bindHistory() {
  $('#hist-csv').addEventListener('click', exportHistory);
  $('#hist-clear').addEventListener('click', () => {
    if (!confirm('판독 이력을 모두 삭제합니다. 계속할까요?')) return;
    hist.clear();
    refreshHistory();
    refreshStore();
  });
}

function exportHistory() {
  const e = hist.list();
  if (!e.length) return;
  hist.download(`판독이력-${stamp()}.csv`, hist.toCsv(e));
}

function refreshHistory() {
  const e = hist.list();
  ui.renderHistTiles(hist.summarize(e));
  ui.renderHistory(e);
}

/* ── 설정 ─────────────────────────────────────────────────────────────── */

function setSwitch(el, on) {
  if (el) el.setAttribute('aria-checked', String(!!on));
}

function bindSettings() {
  $('#primary-load').addEventListener('click', loadPrimary);
  $('#gate-load').addEventListener('click', loadGate);

  $('#gate-switch').addEventListener('click', () => {
    app.gate.enabled = !app.gate.enabled;
    setSwitch($('#gate-switch'), app.gate.enabled);
    persistUrls();
    if (app.gate.enabled && $('#gate-url').value.trim() && !app.registry.has(SLOT.GATE)) loadGate();
    else onRegistry();
  });
  $('#gate-pass').addEventListener('change', e => {
    app.gate.passClasses = e.target.value.split(',').map(s => s.trim()).filter(Boolean);
    persistUrls();
  });
  $('#gate-min').addEventListener('input', e => {
    app.gate.minConfidence = +e.target.value;
    ui.setText($('#gate-min-out'), app.gate.minConfidence.toFixed(2));
    persistUrls();
  });

  $('#ens-add').addEventListener('click', () => {
    app.ens.push({ url: '', label: `앙상블 #${app.ens.length + 1}`, weight: 1 });
    ui.renderEnsembleList(app.ens);
    bindEnsembleRows();
    persistUrls();
    onRegistry();
  });

  // 판정 정책
  range('#s-hold', v => {
    app.settings.holdThreshold = v;
    ui.setText($('#s-hold-out'), v.toFixed(2));
    $('#eval-thr').value = v;
    ui.setText($('#eval-thr-out'), v.toFixed(2));
    updateMarginHint();
    persistSettings();
    if (app.eval.records.length) refreshEvalMetrics();
  });
  range('#s-margin', v => {
    app.settings.marginThreshold = v;
    ui.setText($('#s-margin-out'), v.toFixed(2));
    updateMarginHint();
    persistSettings();
  });
  range('#s-window', v => {
    app.settings.temporalWindow = Math.round(v);
    ui.setText($('#s-window-out'), String(app.settings.temporalWindow));
    persistSettings();
  });
  range('#s-confirm', v => {
    app.settings.confirmFrames = Math.round(v);
    ui.setText($('#s-confirm-out'), String(app.settings.confirmFrames));
    persistSettings();
  });
  range('#s-interval', v => {
    app.settings.inferIntervalMs = Math.round(v);
    ui.setText($('#s-interval-out'), `${app.settings.inferIntervalMs} ms`);
      updateIntervalLabel();
    persistSettings();
  });
  $('#cal-switch').addEventListener('click', () => {
    app.settings.useCalibration = !app.settings.useCalibration;
    setSwitch($('#cal-switch'), app.settings.useCalibration);
    persistSettings();
  });

  // 품질 게이트
  $('#q-switch').addEventListener('click', () => {
    app.settings.enabled = !app.settings.enabled;
    setSwitch($('#q-switch'), app.settings.enabled);
    persistSettings();
  });
  range('#s-blur', v => {
    app.settings.blurMin = Math.round(v);
    ui.setText($('#s-blur-out'), String(app.settings.blurMin));
    persistSettings();
  });
  const luma = () => {
    let lo = +$('#s-lumamin').value, hi = +$('#s-lumamax').value;
    if (lo >= hi) lo = hi - 1;
    app.settings.lumaMin = lo;
    app.settings.lumaMax = hi;
    ui.setText($('#s-luma-out'), `${lo} – ${hi}`);
    persistSettings();
  };
  $('#s-lumamin').addEventListener('input', luma);
  $('#s-lumamax').addEventListener('input', luma);

  $('#policy-reset').addEventListener('click', () => {
    store.resetSettings();
    app.settings = store.loadSettings(PROJECT);
    syncSettings();
    persistSettings();
    applyCopy();
  });

  // 저장 데이터
  $('#store-rows').addEventListener('click', e => {
    const btn = e.target.closest('[data-wipe]');
    if (!btn) return;
    const what = btn.dataset.wipe;
    if (what === 'history') { hist.clear(); refreshHistory(); }
    if (what === 'calibration') { store.clearCalibration(); app.calibration = null; refreshCalibrationUi(); }
    if (what === 'holdout') {
      store.clearHoldout();
      app.eval = { items: [], records: [], files: null, cm: null, sweep: [], token: null };
      $('#cal-build').disabled = true;
      $('#eval-csv').disabled = true;
    }
    if (what === 'settings') { store.resetSettings(); location.reload(); return; }
    refreshStore();
  });
  $('#wipe-all').addEventListener('click', wipeAll);
}

function wipeAll() {
  if (!confirm('저장된 설정·이력·보정표를 모두 삭제합니다. 계속할까요?')) return;
  store.wipeAll();
  location.reload();
}

function range(sel, fn) {
  const el = $(sel);
  if (el) el.addEventListener('input', () => fn(+el.value));
}

function bindEnsembleRows() {
  const list = $('#ens-list');
  list.querySelectorAll('[data-ens-url]').forEach(inp => {
    inp.addEventListener('change', () => {
      app.ens[+inp.dataset.ensUrl].url = inp.value.trim();
      persistUrls();
    });
  });
  list.querySelectorAll('[data-ens-load]').forEach(b => {
    b.addEventListener('click', async () => {
      const i = +b.dataset.ensLoad;
      await app.registry.load(SLOT.ens(i), app.ens[i].url, app.ens[i]);
    });
  });
  list.querySelectorAll('[data-ens-del]').forEach(b => {
    b.addEventListener('click', () => {
      // 슬롯 번호가 밀리므로 전체 언로드 후 재로드합니다
      app.registry.classifierSlots()
        .filter(s => s.startsWith('ensemble:'))
        .forEach(s => app.registry.unload(s));
      app.ens.splice(+b.dataset.ensDel, 1);
      ui.renderEnsembleList(app.ens);
      bindEnsembleRows();
      persistUrls();
      app.ens.forEach((e, k) => { if (e.url) app.registry.load(SLOT.ens(k), e.url, e); });
      onRegistry();
    });
  });
  list.querySelectorAll('[data-ens-w]').forEach(inp => {
    inp.addEventListener('input', () => {
      const i = +inp.dataset.ensW;
      app.ens[i].weight = +inp.value;
      ui.setText(list.querySelector(`[data-ens-wout="${i}"]`), app.ens[i].weight.toFixed(2));
      const entry = app.registry.get(SLOT.ens(i));
      if (entry) entry.weight = app.ens[i].weight;
      persistUrls();
    });
  });
}

function refreshStore() {
  const h = hist.list();
  const ho = store.loadHoldout();
  ui.renderStore([
    { id: 'history', k: '판독 이력', n: h.length,
      meta: h.length ? `${h.length} RECORDS · 96px THUMBS` : 'EMPTY' },
    { id: 'calibration', k: '보정표 Calibration', n: app.calibration ? 1 : 0,
      meta: app.calibration
        ? `${app.calibration.bins.filter(b => b.n).length} BINS · ${app.calibration.n} SAMPLES · ECE ${app.calibration.ece.toFixed(3)}`
        : 'NO TABLE' },
    { id: 'holdout', k: '평가 결과 Holdout', n: app.eval.records.length || (ho ? 1 : 0),
      meta: app.eval.records.length ? `${app.eval.records.length} RECORDS (메모리)`
            : ho ? `${ho.n} RECORDS · ${new Date(ho.at).toLocaleDateString('ko-KR')}` : 'NONE' },
    { id: 'settings', k: '설정 · 모델 URL', n: 1,
      meta: `POLICY + ${1 + (app.gate.enabled ? 1 : 0) + app.ens.length} SLOTS` },
  ], ui.fmtBytes(store.storageUsage()));
}


/* ── 도움말 ───────────────────────────────────────────────────────────── */

const STEPS = [
  { id: 'serve', num: 'STEP 01',
    t: '로컬 서버로 열기 (이미 하셨다면 체크)',
    body: `<p>HTML 파일을 직접 열면 <b>카메라가 절대 동작하지 않습니다.</b>
      브라우저가 안전한 주소에서만 카메라를 허용하기 때문입니다.</p>
      <pre class="doc-code">cd medical-tm-kit
./serve.sh          <span class="c"># 브라우저가 자동으로 열립니다</span></pre>
      <p>주소창이 <code>http://localhost:8000</code> 이면 정상입니다.
      위 <b>환경 자동 점검</b>의 첫 항목이 초록색인지 확인하세요.</p>
      <p>발표용으로 외부에 공개하려면 Vercel에 배포하세요 (<code>npx vercel --prod</code>).
      배포본은 https라서 그대로 동작합니다 — 자세한 절차는 <code>HACKATHON.md</code> 4장에 있습니다.</p>` },

  { id: 'data', num: 'STEP 02',
    t: '학습용 이미지를 클래스별로 모으기',
    body: `<p>분류하고 싶은 종류마다 폴더를 만들고 이미지를 담습니다.
      <b>클래스는 3~4개</b>가 적당합니다. 많을수록 헷갈립니다.</p>
      <p><b>여기서 가장 중요한 것</b> — '무효 입력' 클래스를 반드시 하나 만드세요.
      배경, 손, 벽, 초점이 안 맞은 사진을 모아 담습니다.
      이게 없으면 카메라에 손을 비춰도 모델이 병명을 말합니다.</p>
      <p>클래스마다 <b>장수를 비슷하게</b> 맞추세요. 한쪽이 3배 이상 많으면 그쪽으로 치우칩니다.</p>` },

  { id: 'train', num: 'STEP 03',
    t: 'Teachable Machine에서 학습시키고 URL 받기',
    body: `<p><a href="https://teachablemachine.withgoogle.com" target="_blank" rel="noopener">teachablemachine.withgoogle.com</a>
      → <b>Image Project</b> → <b>Standard image model</b></p>
      <ul class="doc-ul">
        <li class="doc-li">클래스 이름을 입력하고 폴더별로 이미지를 업로드합니다.</li>
        <li class="doc-li"><b>클래스 이름을 정확히 기억하세요</b> — 다음 단계에서 글자 하나까지 맞춰야 합니다.</li>
        <li class="doc-li"><b>Train Model</b> 클릭 (탭을 닫지 마세요).</li>
        <li class="doc-li"><b>Advanced → Under the hood</b> 에서 혼동행렬을 꼭 확인하세요. 어느 클래스끼리 헷갈리는지 여기서 처음 보입니다.</li>
        <li class="doc-li"><b>Export Model → Tensorflow.js → Upload my model</b> → 나오는 주소를 복사합니다.</li>
      </ul>
      <div class="callout" data-tone="warn"><div class="callout-k">NOTE</div>
      <div class="callout-d"><b>Upload my model을 누르지 않으면 주소가 동작하지 않습니다.</b>
      다운로드만 하면 이 프로그램에서 불러올 수 없습니다.</div></div>` },

  { id: 'config', num: 'STEP 04',
    t: '클래스 이름을 설정 파일에 맞추기',
    body: `<p><code>web/project.config.js</code> 를 열어 <code>classes</code> 의 <code>id</code> 를
      TM에서 만든 클래스 이름과 <b>똑같이</b> 바꿉니다. 대소문자·공백까지 동일해야 합니다.</p>
      <pre class="doc-code">{ id: <span class="h">'normal'</span>, label: '정상', labelEn: 'Normal',
  kind: <span class="h">'negative'</span>, color: '#05603F', advice: '…' },</pre>
      <p><code>kind</code> 는 세 가지입니다 —
      <b>positive</b>(발견해야 하는 소견), <b>negative</b>(정상),
      <b>invalid</b>(무효 입력). invalid는 반드시 하나 있어야 합니다.</p>
      <p>저장하고 새로고침하면 됩니다. 빌드 과정은 없습니다.</p>` },

  { id: 'load', num: 'STEP 05',
    t: '설정 탭에서 모델 URL 넣고 불러오기',
    body: `<p><b>설정</b> 탭 → <b>주분류기</b> 칸에 03단계에서 복사한 주소를 붙이고
      <b>불러오기</b>를 누릅니다.</p>
      <ul class="doc-ul">
        <li class="doc-li good">상단 배지가 <b>PRIMARY LOADED · N CLASSES</b> 로 바뀌면 성공입니다.</li>
        <li class="doc-li good">같은 탭의 <b>클래스 정합성</b>이 <b>초록색</b>이어야 합니다.
          빨간색이면 어느 이름이 다른지 알려 주니 그것만 고치세요.</li>
      </ul>
      <p>주소는 브라우저에 저장되므로 다음에는 자동으로 불러옵니다.</p>` },

  { id: 'run', num: 'STEP 06',
    t: '판독해 보고, 반드시 이 두 가지를 시험하기',
    body: `<p><b>판독</b> 탭 → 이미지를 끌어다 놓거나 <b>SAMPLE</b> 버튼 클릭 → <b>이 이미지 판독</b>.</p>
      <p>그다음 <b>꼭 이 두 가지를 해 보세요.</b> 심사위원이 가장 먼저 시도할 것들입니다.</p>
      <ul class="doc-ul">
        <li class="doc-li bad"><b>판독 대상이 아닌 이미지를 넣어 보세요.</b>
          고양이 사진, 빈 배경, 문서 스캔 같은 것을요.
          '무효 입력' 또는 '판단 보류'가 나와야 정상입니다.
          병명이 자신 있게 나오면 02단계의 무효 클래스를 보강해 재학습해야 합니다.
          <br><span class="muted">강의 4-3: "모델은 '모르겠다'를 말하지 못하고 고양이 사진을 넣어도
          억지로 학습된 클래스 중 하나로 답한다" — 그걸 막았는지 확인하는 시험입니다.</span></li>
        <li class="doc-li bad"><b>판독 후 '분석 시작'을 눌러 근거 히트맵을 보세요.</b>
          붉은 영역이 병변이 아니라 이미지 구석·글자·촬영 프레임에 몰려 있으면
          모델이 엉뚱한 걸 학습한 것입니다.</li>
      </ul>
      <p>여기까지 되면 <b>성능 평가</b> 탭에서 TM에 넣지 않은 폴더로 검증하세요 —
      그 수치가 발표에 쓸 값입니다.</p>` },
];

function bindHelp() {
  // 목차 클릭 → 해당 섹션으로 스크롤
  ui.$$('.toc-item').forEach(b => {
    b.addEventListener('click', () => scrollToSec(b.dataset.sec));
  });

  $('#diag-refresh').addEventListener('click', refreshHelp);

  // 체크리스트 (진행 상황은 localStorage에 저장됩니다)
  $('#steps').addEventListener('click', e => {
    const btn = e.target.closest('[data-step]');
    if (!btn) return;
    const done = store.loadChecklist();
    done[btn.dataset.step] = !done[btn.dataset.step];
    store.saveChecklist(done);
    ui.renderSteps(STEPS, done);
  });

  // 스크롤에 따라 목차 활성 항목 갱신
  const secs = ui.$$('.doc-sec');
  if ('IntersectionObserver' in window) {
    const io = new IntersectionObserver(entries => {
      if ($('#tab-help')?.classList.contains('on') !== true) return;
      const visible = entries.filter(x => x.isIntersecting)
        .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0];
      if (visible) ui.setTocActive(visible.target.id.replace(/^sec-/, ''));
    }, { rootMargin: '-160px 0px -60% 0px', threshold: 0 });
    secs.forEach(s => io.observe(s));
  }
}

function scrollToSec(sec) {
  const el = $(`#sec-${sec}`);
  if (!el) return;
  // 상단 고정 헤더(48 + 콘솔헤더) 높이만큼 여유를 둡니다
  const top = el.getBoundingClientRect().top + window.scrollY - 158;
  window.scrollTo({ top, behavior: 'smooth' });
  ui.setTocActive(sec);
}

function refreshHelp() {
  ui.renderDiagnostics(checkEnvironment());
  ui.renderSteps(STEPS, store.loadChecklist());
}

/**
 * 환경 자동 점검 — "왜 안 되는지"를 사용자가 스스로 알 수 있게 합니다.
 * ok: 1 통과 · 0 반드시 해결 · 'warn' 있으면 좋음
 */
function checkEnvironment() {
  const rows = [];

  // 1. secure context — 카메라의 전제 조건
  const isFile = location.protocol === 'file:';
  const secure = window.isSecureContext && !isFile;
  rows.push({
    k: '안전한 주소로 접속 (localhost 또는 https)',
    ok: secure ? 1 : 0,
    v: isFile ? 'file://' : location.origin,
    d: secure
      ? '카메라를 쓸 수 있는 환경입니다.'
      : '<b>카메라가 동작하지 않습니다.</b> 파일을 직접 열지 말고 <code>./serve.sh</code> 를 실행한 뒤 <code>http://localhost:8000</code> 으로 접속하세요.',
  });

  // 2. 추론 라이브러리
  const hasTf = typeof tf !== 'undefined';
  const hasTm = typeof tmImage !== 'undefined';
  rows.push({
    k: '추론 라이브러리 로드 (TensorFlow.js · Teachable Machine)',
    ok: hasTf && hasTm ? 1 : 0,
    v: hasTf && hasTm ? 'READY' : hasTf ? 'tmImage 없음' : '없음',
    d: hasTf && hasTm
      ? '정상적으로 불러왔습니다.'
      : '<b>인터넷 연결을 확인하세요.</b> 이 프로그램은 라이브러리를 인터넷에서 받아오므로 오프라인에서는 동작하지 않습니다. 시연 전에 반드시 확인하세요.',
  });

  // 3. 카메라 API — 웹캠을 쓰지 않으면 검사에서 제외합니다 (이 대회는 업로드 기준)
  if (PROJECT.ui?.enableWebcam) {
    const hasCam = !!navigator.mediaDevices?.getUserMedia;
    rows.push({
      k: '카메라 API 사용 가능',
      ok: hasCam ? 1 : 0,
      v: hasCam ? 'getUserMedia' : '미지원',
      d: hasCam
        ? '웹캠 판독을 쓸 수 있습니다. 처음 시작할 때 권한을 물으면 <b>허용</b>하세요.'
        : '이 브라우저는 카메라 API를 지원하지 않습니다. Chrome 최신 버전을 쓰세요.',
    });
  }

  // 3'. 내장 샘플 — 데모 안정성의 핵심 (강의 10-3의 fallback 요구)
  rows.push({
    k: '내장 샘플 이미지',
    ok: app.samples.length ? 1 : 'warn',
    v: app.samples.length ? `${app.samples.length}장` : 'NONE',
    d: app.samples.length
      ? '업로드가 실패해도 자동으로 대체되고, 파일 없이 바로 데모를 돌릴 수 있습니다.'
      : '없어도 동작하지만 <b>데모가 끊길 위험이 있습니다.</b> '
        + '<code>cd prep &amp;&amp; python3 make_samples.py</code> 로 만드세요 — '
        + '대회에서 요구한 업로드 실패 대비 항목입니다.',
  });

  // 3''. 데이터셋 표기 — 대회 필수 항목
  rows.push({
    k: '데이터셋 출처·라이선스 표기',
    ok: datasetFilled() ? 1 : 0,
    v: datasetFilled() ? 'OK' : '미기재',
    d: datasetFilled()
      ? `${ui.esc(PROJECT.dataset.name)} · ${ui.esc(PROJECT.dataset.license)}`
      : '<b>대회 필수 표기 항목입니다.</b> <code>project.config.js</code> 의 '
        + '<code>dataset</code> 블록(출처·라이선스·라벨링·한계)을 채우세요. '
        + '빠뜨리면 감점됩니다.',
  });

  // 4. 저장소
  let hasLs = false;
  try { localStorage.setItem('mtk.__t', '1'); localStorage.removeItem('mtk.__t'); hasLs = true; } catch { /* 시크릿 모드 등 */ }
  rows.push({
    k: '브라우저 저장소 사용 가능',
    ok: hasLs ? 1 : 'warn',
    v: hasLs ? ui.fmtBytes(store.storageUsage()) : '차단됨',
    d: hasLs
      ? '설정·모델 주소·이력·보정표가 이 브라우저에 저장됩니다. 브라우저를 바꾸면 따라오지 않습니다.'
      : '시크릿 모드이거나 저장이 차단되어 <b>설정과 이력이 유지되지 않습니다.</b> 일반 창에서 사용하세요.',
  });

  // 5. 주분류기
  const ready = app.registry.has(SLOT.PRIMARY);
  const st = app.registry.status(SLOT.PRIMARY);
  rows.push({
    k: '주분류기 모델 로드',
    ok: ready ? 1 : 0,
    v: ready ? `${app.registry.get(SLOT.PRIMARY).labels.length} CLASSES` : (st.state === 'error' ? 'ERROR' : '미로드'),
    d: ready
      ? '모델이 준비되었습니다. 판독을 시작할 수 있습니다.'
      : '<b>설정 탭에서 Teachable Machine 모델 주소를 넣고 불러오세요.</b> 아래 05단계를 참고하세요.'
        + (st.state === 'error' ? ` <span style="color:var(--red)">(${ui.esc(st.message)})</span>` : ''),
  });

  // 6. 클래스 정합성
  if (ready) {
    const v = app.registry.verifyClasses(CLASS_IDS);
    rows.push({
      k: '클래스 이름 정합성 (모델 ↔ 설정)',
      ok: v.ok ? 1 : 0,
      v: v.ok ? 'MATCH' : `${v.missing.length + v.extra.length} MISMATCH`,
      d: v.ok
        ? `${v.modelLabels.length}개 클래스가 정확히 일치합니다.`
        : '<b>이름이 어긋나 결과가 잘못 표시됩니다.</b> 설정 탭의 클래스 정합성 항목에서 어느 이름이 다른지 확인하고 <code>project.config.js</code> 를 고치세요.',
    });

    // 7. 무효 입력 클래스 — 있으면 크게 좋아지는 항목
    const hasInvalid = PROJECT.classes.some(c => c.kind === 'invalid'
      && app.registry.get(SLOT.PRIMARY).labels.includes(c.id));
    rows.push({
      k: "'무효 입력' 클래스 존재",
      ok: hasInvalid ? 1 : 'warn',
      v: hasInvalid ? 'YES' : 'NONE',
      d: hasInvalid
        ? '손·벽 같은 판독 대상이 아닌 입력을 걸러낼 수 있습니다.'
        : '없어도 동작하지만, <b>카메라에 아무것도 안 비춰도 모델이 병명을 말합니다.</b> 배경·손·초점불량 이미지를 모아 클래스를 추가하고 재학습하는 것을 강력히 권합니다.',
    });
  }

  // 8. 보정표
  rows.push({
    k: '신뢰도 보정표',
    ok: app.calibration ? 1 : 'warn',
    v: app.calibration ? `${app.calibration.n} SAMPLES` : 'NONE',
    d: app.calibration
      ? `표본 ${app.calibration.n}건 기준. 판독 화면에 보정 신뢰도가 함께 표시됩니다.`
      : '없어도 동작합니다. <b>성능 평가</b> 탭에서 홀드아웃 폴더를 넣고 <b>보정표 생성</b>을 누르면, 모델이 얼마나 과신하는지 교정된 값을 볼 수 있습니다.',
  });

  return rows;
}

/* ── 유틸 ─────────────────────────────────────────────────────────────── */

function stamp() {
  const d = new Date(), p = n => String(n).padStart(2, '0');
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}`;
}

/* ── 시작 ─────────────────────────────────────────────────────────────── */

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}

window.__mtk = app;
