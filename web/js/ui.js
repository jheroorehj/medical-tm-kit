/* ui.js — 렌더링 계층 (RADIOLENS)
 * ---------------------------------------------------------------------------
 * DOM을 만지는 모든 코드가 여기 모여 있습니다. main.js는 이벤트 배선만 합니다.
 * 판정 로직은 갖지 않습니다 — infer.decide()의 결과를 그리기만 합니다.
 *
 * 디자인 프로토타입은 판정 상태를 모노스페이스 글리프 한 글자로 표시합니다.
 * infer.js 는 이모지를 돌려주지만, 그건 로직 계층의 기본값이고 표현은 여기서
 * 결정합니다 (검증된 로직 파일을 건드리지 않기 위해).
 */

import { PROJECT, CLASS_IDS, classOf, fullLabel, datasetFilled } from '../project.config.js';
import { pct, STATUS } from './infer.js';
import { fmtPct } from './metrics.js';
import { qualityScore } from './quality.js';

export const $ = sel => document.querySelector(sel);
export const $$ = sel => [...document.querySelectorAll(sel)];

/**
 * 가시성 토글.
 *
 * ★ el.hidden = bool 을 쓰지 않는 이유:
 *   SVGElement 에는 hidden IDL 프로퍼티가 없습니다 (HTMLElement 에만 있습니다).
 *   SVG 요소에 el.hidden = true 를 대입하면 속성이 아니라 JS 확장 프로퍼티만
 *   만들어져 CSS 의 [hidden] 규칙이 전혀 적용되지 않습니다.
 *   속성을 직접 조작하면 HTML·SVG 양쪽에서 동일하게 동작하고,
 *   HTML 요소의 el.hidden 게터도 그대로 속성을 반영합니다.
 */
export function show(el, on = true) {
  if (!el) return;
  if (on) el.removeAttribute('hidden');
  else el.setAttribute('hidden', '');
}
export function setText(el, t) { if (el) el.textContent = t; }

/**
 * 채우기 막대의 값을 넣습니다. @param {number} ratio 0~1
 *
 * width 가 아니라 transform: scaleX 를 씁니다.
 *   · width 는 레이아웃과 페인트를 유발하고, transform 은 GPU 에서 끝납니다.
 *   · 값이 애니메이션 중에 다시 바뀔 때(실시간 판독·배치 진행) transform 은
 *     현재 위치에서 자연스럽게 재조준되지만 width 는 끊겨 보입니다.
 * CSS 쪽은 width:100% + transform-origin:left 로 맞춰져 있습니다.
 */
export function setFill(el, ratio) {
  if (!el) return;
  const r = Math.max(0, Math.min(1, Number(ratio) || 0));
  el.style.transform = `scaleX(${r.toFixed(4)})`;
}
export function setHtml(el, h) { if (el) el.innerHTML = h; }

/** 모델 metadata에서 온 문자열을 innerHTML에 넣기 전에 반드시 통과시킵니다. */
export function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, c => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}

export function fmtBytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(2)} MB`;
}

/* ── 판정 상태 → 표현 ────────────────────────────────────────────────── */

const INK = '#48545F', RED = '#B3261E', AMBER = '#8A5300', GREEN = '#05603F';

/** decide() 결과를 디자인의 글리프·색·라벨로 변환합니다. */
export function presentation(d) {
  switch (d.status) {
    case STATUS.GATED:
      return { glyph: '·', color: INK, label: 'GATED · 판독 대상 없음' };
    case STATUS.INVALID:
      return { glyph: '×', color: INK, label: 'INVALID · 무효 입력' };
    case STATUS.HOLD:
      return { glyph: '?', color: RED, label: 'HOLD · 판단 보류' };
    case STATUS.AMBIGUOUS:
      return { glyph: '≈', color: AMBER, label: 'AMBIGUOUS · 구분 어려움' };
    default: {
      const positive = d.top.def.kind === 'positive';
      return {
        glyph: positive ? '!' : '✓',
        color: positive ? RED : GREEN,
        label: 'OK · 판정 성립',
      };
    }
  }
}

/* ── 상단 상태 배지 ──────────────────────────────────────────────────── */

export function setPill(sel, state, text) {
  const el = $(sel);
  if (!el) return;
  el.dataset.state = state;
  const span = el.querySelector('span');
  if (span) span.textContent = text;
}

/* ── 판정 카드 ────────────────────────────────────────────────────────── */

let flashParity = 0;

/**
 * @param {object} d infer.decide() 결과
 * @param {object} x {live, temporal, confirmFrames, threshold, jitter, flash}
 */
export function renderVerdict(d, x = {}) {
  const p = presentation(d);
  const head = $('#verdict-head');
  $('#verdict').style.borderColor = p.color;
  head.style.background = p.color;

  const icon = $('#verdict-icon');
  icon.textContent = p.glyph;
  icon.style.color = p.color;

  setText($('#verdict-status'), p.label);
  setText($('#verdict-trust'), d.trustworthy ? 'TRUSTWORTHY' : 'NOT TRUSTWORTHY');
  setText($('#verdict-headline'), d.headline);
  show($('#verdict-num'), true);
  setText($('#verdict-pct'), pct(d.top.prob));
  setText($('#verdict-detail'), d.detail);
  setText($('#verdict-advice'), d.advice || '—');

  // 판정이 갱신될 때마다 링 플래시.
  // 링은 .verdict::after 에 그려지므로 인라인 style 로는 겨냥할 수 없어
  // 클래스를 토글합니다. 동일 키프레임 두 개를 번갈아 붙여 재시작시킵니다.
  if (x.flash) {
    flashParity ^= 1;
    const v = $('#verdict');
    v.style.setProperty('--flash', p.color);
    v.classList.remove('flash-a', 'flash-b');
    v.classList.add(flashParity ? 'flash-a' : 'flash-b');
  }

  // 실시간 확정 진행
  const cr = $('#confirm-row');
  if (x.live && x.temporal) {
    show(cr, true);
    const cf = x.confirmFrames ?? 1;
    setText($('#streak-label'), `연속 ${x.temporal.streak} / ${cf} 프레임`);
    const done = x.temporal.confirmed;
    const lab = $('#confirm-label');
    lab.textContent = done ? '판정 확정' : '확정 대기';
    lab.style.color = done ? GREEN : AMBER;
    lab.style.fontWeight = '700';
    const fill = $('#confirm-fill');
    setFill(fill, x.temporal.progress);
    fill.style.background = done ? GREEN : AMBER;
  } else {
    show(cr, false);
  }

  renderBars(d.ranked);
  renderUncertainty(d, x);

  setText($('#stamp-time'), new Date().toLocaleString('ko-KR', { hour12: false }));
  setText($('#stamp-thr'), `THRESHOLD ${(x.threshold ?? 0).toFixed(2)}`);
  setText($('#heat-target'), `TARGET: ${d.top.def.label}`);
}

export function resetVerdict() {
  const v = $('#verdict');
  v.classList.remove('flash-a', 'flash-b');
  v.style.removeProperty('--flash');
  v.style.borderColor = 'var(--t2)';
  $('#verdict-head').style.background = 'var(--t2)';
  const icon = $('#verdict-icon');
  icon.textContent = '·';
  icon.style.color = INK;
  setText($('#verdict-status'), 'READY · 대기');
  setText($('#verdict-trust'), '');
  setText($('#verdict-headline'), '판독 대기');
  setText($('#verdict-pct'), '—');
  setText($('#verdict-detail'), PROJECT.copy.idleDetail ?? '모델을 불러오고 이미지를 판독하세요.');
  setText($('#verdict-advice'), '—');
  show($('#confirm-row'), false);
  show($('#verdict-num'), false);
  setHtml($('#bars'), '<div class="empty">모델을 불러오고 판독하면 클래스별 유사도가 표시됩니다</div>');
  setHtml($('#uncertainty'), PLACEHOLDER_SIGNALS.map(r => `
    <div class="unc-row">
      <div style="min-width:0">
        <div class="unc-name">${r.k}</div>
        <div class="unc-en">${r.en}</div>
      </div>
      <div class="unc-v" style="color:var(--t4)">—</div>
    </div>`).join(''));
  setText($('#heat-target'), 'TARGET: —');
}

const PLACEHOLDER_SIGNALS = [
  { k: '유사도', en: 'TOP SIMILARITY' },
  { k: '보정 신뢰도', en: 'CALIBRATED' },
  { k: '1·2위 차이', en: 'MARGIN' },
  { k: '불확실성', en: 'NORMALIZED ENTROPY' },
  { k: '흔들림', en: 'JITTER' },
];

/* ── 유사도 막대 ──────────────────────────────────────────────────────── */

/**
 * 노드를 재사용하고 width만 갱신합니다.
 * innerHTML로 매번 새로 만들면 초기값이 곧 최종값이라 트랜지션이 재생되지 않습니다.
 */
export function renderBars(ranked) {
  const box = $('#bars');
  const wanted = ranked.map(r => r.id).join(',');
  if (box.dataset.keys !== wanted) {
    box.dataset.keys = wanted;
    box.innerHTML = ranked.map(r => `
      <div class="bar" data-id="${esc(r.id)}">
        <div class="bar-name">
          <span class="bar-swatch" style="background:${r.def.color}"></span>
          <span class="bar-label">${esc(fullLabel(r.id))}</span>
        </div>
        <div class="bar-track"><div class="bar-fill" style="background:${r.def.color}"></div></div>
        <div class="bar-pct">0.0%</div>
      </div>`).join('');
  }

  requestAnimationFrame(() => {
    [...box.children].forEach((el, i) => {
      const r = ranked[i];
      if (!r || !el.classList.contains('bar')) return;
      el.classList.toggle('top', i === 0);
      setFill(el.querySelector('.bar-fill'), r.prob);
      const pctEl = el.querySelector('.bar-pct');
      pctEl.textContent = pct(r.prob);
      pctEl.style.color = i === 0 ? r.def.color : 'var(--t2)';
    });
  });
}

/* ── 불확실성 5지표 ──────────────────────────────────────────────────── */

export function renderUncertainty(d, x = {}) {
  const p = presentation(d);
  const cal = d.calibrated;
  const rows = [
    { k: '유사도', en: 'TOP SIMILARITY', v: pct(d.top.prob), c: p.color,
      t: '1위 클래스의 원시 출력값' },
    { k: '보정 신뢰도',
      en: cal?.n ? `CALIBRATED · n=${cal.n}` : 'CALIBRATED · 없음',
      v: cal?.n ? pct(cal.value) : '—', c: 'var(--ink)',
      t: cal?.n ? `${cal.binLabel} 구간 표본 ${cal.n}건 기준 실측 정확도${cal.borrowed ? ' (인접 구간 차용)' : ''}`
               : '성능 평가 탭에서 보정표를 먼저 생성하세요' },
    { k: '1·2위 차이', en: `MARGIN · 기준 ${pct(x.marginThreshold ?? 0, 0)}`,
      v: pct(d.margin), c: d.margin < (x.marginThreshold ?? 0) ? AMBER : 'var(--ink)',
      t: '1위와 2위 유사도의 차' },
    { k: '불확실성', en: 'NORMALIZED ENTROPY', v: d.entropy.toFixed(2),
      c: d.entropy > 0.5 ? AMBER : 'var(--ink)', t: '정규화 엔트로피 0~1' },
    { k: '흔들림', en: x.live ? 'JITTER · 실시간' : 'JITTER · 단발',
      v: x.jitter != null ? x.jitter.toFixed(3) : '—', c: 'var(--ink)',
      t: '최근 프레임 간 1위 유사도 표준편차' },
  ];
  setHtml($('#uncertainty'), rows.map(r => `
    <div class="unc-row" title="${esc(r.t)}">
      <div style="min-width:0">
        <div class="unc-name">${esc(r.k)}</div>
        <div class="unc-en">${esc(r.en)}</div>
      </div>
      <div class="unc-v" style="color:${r.c}">${esc(r.v)}</div>
    </div>`).join(''));
}

/* ── 품질 ─────────────────────────────────────────────────────────────── */

const COACH_TONE = { ok: 'ok', warn: 'warn', danger: 'danger', idle: 'idle' };

/** @param {'q'|'uq'} prefix 웹캠 / 업로드 스트립 구분 */
export function renderQuality(prefix, m, settings) {
  const set = (id, v, tone) => {
    const el = $(`#${prefix}-${id}`);
    if (!el) return;
    el.textContent = v;
    if (tone) el.dataset.tone = tone; else delete el.dataset.tone;
  };
  if (!m) { ['score', 'blur', 'luma', 'contrast'].forEach(k => set(k, '—')); return; }
  const score = qualityScore(m, settings);
  set('score', String(score), score >= 70 ? 'good' : 'bad');
  set('blur', m.blur.toFixed(0), m.blur >= settings.blurMin ? 'good' : 'bad');
  set('luma', m.luma.toFixed(0),
      m.luma >= settings.lumaMin && m.luma <= settings.lumaMax ? 'good' : 'bad');
  set('contrast', m.contrast.toFixed(0), m.contrast >= 12 ? 'good' : 'bad');
}

export function renderCoach(ev, camOn) {
  const box = $('#cam-coach');
  const guide = $('#cam-guide');
  if (!camOn) {
    box.dataset.tone = COACH_TONE.idle;
    setText($('#coach-icon'), '·');
    setText($('#coach-text'), '카메라를 시작하세요');
    return;
  }
  const ok = ev?.ok !== false;
  const worst = ev?.issues?.[0];
  const tone = ok ? 'ok' : (worst?.severity >= 3 ? 'danger' : 'warn');
  box.dataset.tone = COACH_TONE[tone];
  setText($('#coach-icon'), ok ? '✓' : '!');
  setText($('#coach-text'), ok ? '촬영 상태 양호 — 그대로 유지하세요' : (ev.coach || '촬영을 조정하세요'));
  guide?.classList.toggle('bad', !ok);
}

/* ── 지표 타일 6개 ───────────────────────────────────────────────────── */

export function renderTiles(m, accuracy) {
  const pri = PROJECT.copy.metricPriority;
  const v = x => (m ? x : '—');
  const tiles = [
    { k: '표본', en: 'SAMPLES', v: v(m && String(m.evaluated + m.excluded)), tone: '', p: 0 },
    { k: '전체 정확도', en: 'ACCURACY', v: v(fmtPct(accuracy)), tone: '', p: 0 },
    { k: '민감도', en: 'SENSITIVITY', v: v(fmtPct(m?.sensitivity)), tone: 'red',
      p: pri === 'sensitivity' ? 1 : 0 },
    { k: '특이도', en: 'SPECIFICITY', v: v(fmtPct(m?.specificity)), tone: 'blue',
      p: pri === 'specificity' ? 1 : 0 },
    { k: '양성예측도', en: 'PPV · PRECISION', v: v(fmtPct(m?.ppv)), tone: '', p: 0 },
    { k: '보류율', en: 'HOLD RATE', v: v(fmtPct(m?.holdRate)), tone: 'amber', p: 0 },
  ];
  setHtml($('#eval-tiles'), tiles.map(t => `
    <div class="tile"${t.tone ? ` data-tone="${t.tone}"` : ''} data-pri="${t.p}">
      <div class="tile-head">
        <div class="tile-k">${esc(t.k)}</div>
        ${t.p ? '<div class="tile-pri">우선</div>' : ''}
      </div>
      <div class="tile-v">${esc(t.v)}</div>
      <div class="tile-en">${esc(t.en)}</div>
    </div>`).join(''));
}

export function renderLiveMetrics(m) {
  const rows = [
    { k: '민감도 Sensitivity', v: m ? fmtPct(m.sensitivity) : '—', c: RED },
    { k: '특이도 Specificity', v: m ? fmtPct(m.specificity) : '—', c: 'var(--blue)' },
    { k: '보류 건수 Hold', v: m ? `${m.held}건` : '—', c: AMBER },
    { k: '위음성 FN', v: m ? `${m.fn}건` : '—', c: 'var(--ink)' },
  ];
  setHtml($('#eval-livemetrics'), rows.map(r => `
    <div><div class="k">${esc(r.k)}</div><div class="v" style="color:${r.c}">${esc(r.v)}</div></div>
  `).join(''));
}

/* ── 임계값 스윕 곡선 (720×200 좌표계) ───────────────────────────────── */

const X = t => ((t - 0.05) / 0.9) * 720;
const Y = v => 200 - (v ?? 0) * 190;

export function renderSweep(sweep, current, m) {
  // 데이터가 없으면 곡선과 커서를 지웁니다 — 좌표 0,0 에 점이 찍히지 않도록
  const hasData = sweep.length > 0 && m;
  ['#sw-dot-sens', '#sw-dot-spec', '#sw-cursor'].forEach(s => show($(s), hasData));
  if (!hasData) {
    ['#sw-sens', '#sw-spec', '#sw-hold'].forEach(s => $(s).setAttribute('points', ''));
    return;
  }
  const pts = key => sweep
    .filter(d => d[key] != null)
    .map(d => `${X(d.threshold).toFixed(1)},${Y(d[key]).toFixed(1)}`)
    .join(' ');

  $('#sw-sens').setAttribute('points', pts('sensitivity'));
  $('#sw-spec').setAttribute('points', pts('specificity'));
  $('#sw-hold').setAttribute('points', pts('holdRate'));

  const cx = X(current).toFixed(1);
  const cur = $('#sw-cursor');
  cur.setAttribute('x1', cx); cur.setAttribute('x2', cx);
  $('#sw-dot-sens').setAttribute('cx', cx);
  $('#sw-dot-sens').setAttribute('cy', Y(m.sensitivity).toFixed(1));
  $('#sw-dot-spec').setAttribute('cx', cx);
  $('#sw-dot-spec').setAttribute('cy', Y(m.specificity).toFixed(1));
}

/* ── 혼동행렬 ─────────────────────────────────────────────────────────── */

export function renderConfusion(cm) {
  const { matrix, classIds } = cm;
  const n = classIds.length;
  const cols = `112px repeat(${n}, 1fr) 66px`;

  let html = `<div class="cm-grid" style="grid-template-columns:${cols}">`
    + '<div></div>'
    + classIds.map(id => `<div class="cm-head">${esc(classOf(id).label)}</div>`).join('')
    + '<div class="cm-head right">RECALL</div>'
    + '</div>';

  matrix.forEach((row, i) => {
    const total = row.reduce((a, b) => a + b, 0);
    const recall = total > 0 ? row[i] / total : null;
    html += `<div class="cm-grid" style="grid-template-columns:${cols};margin-bottom:3px">`
      + `<div class="cm-rowlabel">${esc(fullLabel(classIds[i]))}</div>`
      + row.map((v, j) => {
          const cls = i === j ? 'diag' : v === 0 ? 'zero' : v >= 10 ? 'hot' : '';
          return `<div class="cm-cell ${cls}">${v}</div>`;
        }).join('')
      + `<div class="cm-recall">${fmtPct(recall)}</div>`
      + '</div>';
  });

  setHtml($('#cm-wrap'), html);
}

export function renderConfusions(pairs) {
  const el = $('#cm-confusions');
  if (!pairs.length) { show(el, false); return; }
  setHtml(el, `<div class="notice-k">TOP CONFUSIONS · 가장 자주 헷갈리는 조합</div>
    <div class="notice-d">${pairs.map(p =>
      `${esc(classOf(p.trueId).label)} → ${esc(classOf(p.predId).label)} ${p.count}건`).join(' · ')}
      <br>두 클래스를 합칠지 검토할 근거입니다.</div>`);
  show(el, true);
}

/* ── 캘리브레이션 ────────────────────────────────────────────────────── */

export function renderCalibration(cal, desc) {
  setText($('#ece-label'), cal ? `ECE ${cal.ece.toFixed(3)}` : 'ECE —');
  if (!cal) {
    setHtml($('#calib-wrap'), '<div class="empty">평가를 실행한 뒤 <b>보정표 생성</b>을 누르세요</div>');
    show($('#calib-note'), false);
    return;
  }
  const rows = cal.bins.filter(b => b.n > 0);
  setHtml($('#calib-wrap'),
    `<div class="calib-head"><div>BIN</div><div>N</div><div>MEAN</div><div>ACTUAL</div><div>GAP</div></div>`
    + rows.map(b => {
        const gap = b.accuracy - b.avgConfidence;
        const color = Math.abs(gap) < 0.03 ? 'var(--t2)' : gap < 0 ? RED : GREEN;
        return `<div class="calib-row">
          <div class="bin">${b.lo.toFixed(1)} – ${b.hi.toFixed(1)}</div>
          <div class="n">${b.n}</div>
          <div class="mean">${b.avgConfidence.toFixed(2)}</div>
          <div class="acc">${fmtPct(b.accuracy)}</div>
          <div class="gap" style="color:${color}">${gap >= 0 ? '+' : ''}${(gap * 100).toFixed(1)}%p</div>
        </div>`;
      }).join(''));

  const el = $('#calib-note');
  el.dataset.tone = desc.tone === 'ok' ? 'ok' : desc.tone === 'warn' ? 'warn' : 'danger';
  setHtml(el, `<div class="notice-d">${esc(desc.text)} 판독 화면의 보정 신뢰도가 이 표를 씁니다.</div>`);
  show(el, true);
}

/* ── 실패 사례 ───────────────────────────────────────────────────────── */

let failUrls = [];

export function renderFailures(failures, files) {
  failUrls.forEach(u => URL.revokeObjectURL(u));
  failUrls = [];
  const box = $('#failgrid');
  if (!failures.length) {
    setHtml(box, '<div class="empty">오분류 사례가 없습니다. 표본이 적거나 과적합을 의심해 보세요.</div>');
    return;
  }
  setHtml(box, failures.map(f => {
    const file = files?.get(f.file);
    let src = '';
    if (file) { src = URL.createObjectURL(file); failUrls.push(src); }
    return `<div class="failitem">
      <div class="failthumb">
        ${src ? `<img src="${src}" alt="" loading="lazy" />` : ''}
        <div class="failconf">${f.topProb.toFixed(2)}</div>
      </div>
      <div class="failmeta">${esc(classOf(f.trueId).label)}<br>
        <span class="pred">→ ${esc(classOf(f.topId).label)}</span></div>
    </div>`;
  }).join(''));
}

/* ── 이력 ─────────────────────────────────────────────────────────────── */

const HIST = {
  ok:        { glyph: '✓', color: GREEN, label: '판정 성립' },
  hold:      { glyph: '?', color: RED,   label: '판단 보류' },
  ambiguous: { glyph: '≈', color: AMBER, label: '구분 어려움' },
  invalid:   { glyph: '×', color: INK,   label: '무효 입력' },
  gated:     { glyph: '·', color: INK,   label: '대상 없음' },
};

export function renderHistTiles(s) {
  const tiles = [
    { en: 'OK', k: '판정 성립', v: s.byStatus.ok ?? 0, tone: 'ok' },
    { en: 'HOLD', k: '판단 보류', v: s.byStatus.hold ?? 0, tone: 'hold' },
    { en: 'AMBIGUOUS', k: '구분 어려움', v: s.byStatus.ambiguous ?? 0, tone: 'ambig' },
    { en: 'INVALID', k: '무효 입력', v: (s.byStatus.invalid ?? 0) + (s.byStatus.gated ?? 0), tone: 'invalid' },
  ];
  setHtml($('#hist-tiles'), tiles.map(t => `
    <div class="hist-tile" data-tone="${t.tone}">
      <div class="en">${t.en}</div><div class="v">${t.v}</div><div class="k">${esc(t.k)}</div>
    </div>`).join(''));
}

export function renderHistory(entries) {
  const box = $('#hist-rows');
  if (!entries.length) {
    setHtml(box, '<div class="empty">아직 판독 이력이 없습니다.</div>');
    return;
  }
  setHtml(box, entries.slice(0, 100).map(e => {
    const h = HIST[e.status] ?? HIST.gated;
    const label = e.status === 'ok' ? fullLabel(e.topId) : `${h.label} · ${classOf(e.topId).label}`;
    return `<div class="histrow">
      ${e.thumb ? `<img class="histthumb" src="${e.thumb}" alt="" />` : '<div class="histthumb"></div>'}
      <div class="histdec">
        <div class="histicon" style="background:${h.color}">${h.glyph}</div>
        <div class="histlabel">${esc(label)}</div>
      </div>
      <div class="num" style="font-size:12px;color:var(--t2)">${new Date(e.ts).toLocaleTimeString('ko-KR', { hour12: false })}</div>
      <div class="num" style="font-size:13px;font-weight:700;color:${h.color}">${fmtPct(e.topProb)}</div>
      <div class="num" style="font-size:12px;color:var(--t2)">${(e.policy?.hold ?? 0).toFixed(2)}</div>
      <div class="mono" style="font-size:10.5px;color:var(--t3)">${esc(e.source)}</div>
      <div class="histnote">${esc(e.note ?? '')}</div>
    </div>`;
  }).join(''));
}

/* ── 저장 데이터 ─────────────────────────────────────────────────────── */

export function renderStore(rows, total) {
  setText($('#store-total'), total);
  setHtml($('#store-rows'), rows.map(r => `
    <div class="storerow">
      <div>
        <div class="k">${esc(r.k)}</div>
        <div class="meta">${esc(r.meta)}</div>
      </div>
      <button type="button" class="btn btn-tiny" data-wipe="${r.id}" ${r.n ? '' : 'disabled'}>삭제</button>
    </div>`).join(''));
}

/* ── 설정: 클래스 정합성 ─────────────────────────────────────────────── */

export function renderClassCheck(verify, ensCheck) {
  const el = $('#class-check');
  setHtml($('#cfg-classes'), CLASS_IDS.map(id =>
    verify.missing.includes(id) ? `<span class="bad">${esc(id)}</span>` : esc(id)).join('<br>'));

  if (!verify.modelLabels.length) {
    el.dataset.tone = 'warn';
    setHtml(el, `<div class="notice-k">NOT LOADED · 모델 로드 후 검사</div>
      <div class="notice-d">주분류기를 불러오면 TM 클래스명과 config의 클래스 id를 즉시 대조합니다.</div>`);
    setHtml($('#model-classes'), '—');
    return;
  }

  setHtml($('#model-classes'), verify.modelLabels.map(l =>
    verify.extra.includes(l) ? `<span class="bad">${esc(l)}</span>` : esc(l)).join('<br>'));

  const problems = [];
  if (verify.missing.length) {
    problems.push(`설정에는 있으나 모델에 없음: ${verify.missing.map(c => `<code>${esc(c)}</code>`).join(', ')}`);
  }
  if (verify.extra.length) {
    problems.push(`모델에는 있으나 설정에 없음: ${verify.extra.map(c => `<code>${esc(c)}</code>`).join(', ')}`);
  }
  if (ensCheck && !ensCheck.ok) {
    problems.push(`앙상블 클래스 집합 불일치: ${ensCheck.mismatched.map(esc).join(', ')} — 보팅이 왜곡됩니다`);
  }

  if (!problems.length) {
    el.dataset.tone = 'ok';
    setHtml(el, `<div class="notice-k">MATCH · 정합성 통과</div>
      <div class="notice-d">모델과 설정의 클래스 ${verify.modelLabels.length}개가 정확히 일치합니다.</div>`);
  } else {
    el.dataset.tone = 'danger';
    setHtml(el, `<div class="notice-k">${problems.length} MISMATCH · 로드 즉시 검사</div>
      <div class="notice-d">${problems.join('<br>')}<br>
      <code>project.config.js</code>를 수정하세요. 오타 하나로 몇 시간을 날리는 사고를 여기서 잡습니다.</div>`);
  }
}

/* ── 설정: 앙상블 목록 ───────────────────────────────────────────────── */

export function renderEnsembleList(ens) {
  const box = $('#ens-list');
  if (!ens.length) {
    setHtml(box, '<div class="mcap" style="font-weight:400">등록된 앙상블 모델이 없습니다 — 단일 모델로 동작합니다</div>');
    return;
  }
  setHtml(box, ens.map((e, i) => `
    <div style="margin-bottom:9px">
      <div class="slot-head" style="margin-bottom:6px">
        <div class="slot-name" style="font-size:12.5px">${esc(e.label)}</div>
        <div class="spill" id="ens-state-${i}" data-state="idle"><i></i><span>미로드</span></div>
      </div>
      <div class="slot-row">
        <input class="slot-url" data-ens-url="${i}" type="url" spellcheck="false"
               value="${esc(e.url)}" placeholder="https://teachablemachine.withgoogle.com/models/…/" />
        <button type="button" class="btn" data-ens-load="${i}">불러오기</button>
        <button type="button" class="btn btn-danger" data-ens-del="${i}">삭제</button>
      </div>
      <div style="margin-top:8px">
        <div class="field-head">
          <div class="k">보팅 가중치</div>
          <div class="v" data-ens-wout="${i}">${e.weight.toFixed(2)}</div>
        </div>
        <input type="range" data-ens-w="${i}" min="0.1" max="2" step="0.05" value="${e.weight}" />
      </div>
    </div>`).join(''));
}

/* ── 랜딩 정적 리스트 ────────────────────────────────────────────────── */

const PIPELINE = [
  { n: '01', name: '품질 검사', fn: 'measureFrame → evaluateQuality', d: '초점·밝기·대비를 재서 판독에 부적합한 영상을 걸러내고 이유를 알려 줍니다.' },
  { n: '02', name: '224 정사각 변환', fn: 'toSquareCanvas(roi, 224)', d: '모델이 실제로 보는 영역을 화면에 그대로 표시합니다.' },
  { n: '03', name: '게이트키퍼', fn: 'runGate(registry, cfg)', d: '판독 대상이 아닌 이미지를 주분류기 앞에서 막습니다.' },
  { n: '04', name: '주분류기 + 앙상블', fn: 'runClassify → {probs}', d: '여러 모델의 출력을 가중 평균해 클래스별 유사도를 만듭니다.' },
  { n: '05', name: '판정', fn: 'decide(probs, settings)', d: '4개 관문을 통과한 것만 결론으로 발표합니다. 5개 상태로 구분됩니다.' },
];

const TRUST = [
  { tag: 'ON-DEVICE', name: '영상은 나가지 않습니다', d: '추론·품질 측정·히트맵 모두 브라우저 안에서 실행됩니다. 이력에도 원본 대신 96px 썸네일과 메타데이터만 남습니다.' },
  { tag: 'REPRODUCIBLE', name: '판정 당시 임계값을 기록', d: '정책을 바꿔도 과거 판정을 그대로 재현할 수 있습니다. CSV로 내보내 보고서에 붙일 수 있습니다.' },
  { tag: 'HONEST METRICS', name: '보정표로 과신을 교정', d: '모델이 말하는 확신과 실측 정확도의 격차를 구간별로 보여주고, 판독 화면의 보정 신뢰도에 반영합니다.' },
];

export function renderLanding() {
  setHtml($('#pipeline'), PIPELINE.map(p => `
    <div class="pipe-step">
      <div class="pipe-num">${p.n}</div>
      <div class="pipe-name">${esc(p.name)}</div>
      <div class="pipe-fn">${esc(p.fn)}</div>
      <div class="pipe-desc">${esc(p.d)}</div>
    </div>`).join(''));

  setHtml($('#trustgrid'), TRUST.map(t => `
    <div>
      <div class="trust-tag">${t.tag}</div>
      <div class="trust-n">${esc(t.name)}</div>
      <div class="trust-d">${esc(t.d)}</div>
    </div>`).join(''));
}

/* ── 커맨드 팔레트 ───────────────────────────────────────────────────── */

export function renderCmdk(items, sel, query) {
  const box = $('#cmdk-list');
  if (!items.length) {
    setHtml(box, `<div class="cmdk-empty">"${esc(query)}" 에 해당하는 명령이 없습니다</div>`);
    return;
  }
  setHtml(box, items.map((it, i) => `
    <button type="button" class="cmdk-item${i === sel ? ' sel' : ''}" data-cmd="${i}">
      <span class="k">${esc(it.label)}</span>
      <span class="en">${esc(it.en)}</span>
    </button>`).join(''));
  box.querySelector('.sel')?.scrollIntoView({ block: 'nearest' });
}


/* ══ 도움말 탭 ═══════════════════════════════════════════════════════════ */

/** 환경 자동 점검 결과를 그립니다. @param {Array} rows {k, d, ok, v} */
export function renderDiagnostics(rows) {
  const fails = rows.filter(r => r.ok === 0).length;
  const warns = rows.filter(r => r.ok === 'warn').length;
  const oks = rows.filter(r => r.ok === 1).length;

  const score = $('#diag-score');
  score.textContent = `${oks}/${rows.length}`;
  score.style.color = fails ? RED : warns ? AMBER : GREEN;

  setText($('#diag-verdict'), fails
    ? `${fails}개 항목을 먼저 해결해야 합니다`
    : warns ? `동작은 하지만 ${warns}개 항목을 확인하세요` : '모든 점검 통과 — 바로 사용할 수 있습니다');
  setText($('#diag-sub'), fails
    ? '빨간 항목의 해결 방법이 아래에 적혀 있습니다.'
    : warns ? '노란 항목은 필수는 아니지만 있으면 결과가 좋아집니다.'
            : '아래 3번 항목의 6단계를 따라가세요.');

  setHtml($('#diag-list'), rows.map(r => `
    <div class="diag-row" data-ok="${r.ok}">
      <div class="diag-mark">${r.ok === 1 ? '✓' : r.ok === 'warn' ? '!' : '×'}</div>
      <div>
        <div class="diag-k">${esc(r.k)}</div>
        <div class="diag-d">${r.d}</div>
      </div>
      <div class="diag-v">${esc(r.v ?? '')}</div>
    </div>`).join(''));
}

/** 빠른 시작 단계. @param {Array} steps {id, num, t, body} @param {object} done */
export function renderSteps(steps, done) {
  const n = steps.filter(s => done[s.id]).length;
  setText($('#step-count'), `${n} / ${steps.length}`);
  setFill($('#step-fill'), n / steps.length);

  setHtml($('#steps'), steps.map(s => `
    <div class="step" data-done="${done[s.id] ? 1 : 0}">
      <button type="button" class="step-check" data-step="${s.id}"
              aria-pressed="${!!done[s.id]}" aria-label="${esc(s.t)} 완료 표시">✓</button>
      <div>
        <div class="step-num">${s.num}</div>
        <div class="step-t">${esc(s.t)}</div>
        <div class="step-b">${s.body}</div>
      </div>
    </div>`).join(''));
}

/** 목차 활성 항목 표시 */
export function setTocActive(sec) {
  $$('.toc-item').forEach(b => b.classList.toggle('on', b.dataset.sec === sec));
}


/* ══ 데이터셋 카드 · 한계 · 샘플 (대회 필수 표기) ═══════════════════════ */

/**
 * 데이터 출처·라이선스·라벨링·클래스 구성을 화면에 표기합니다.
 * 강의 3-3: 산출물에 반드시 남겨야 하는 기록입니다.
 * @param {string} bodySel 본문 컨테이너 · @param {string} licSel 라이선스 배지
 */
export function renderDataset(bodySel, licSel) {
  const d = PROJECT.dataset;
  const filled = datasetFilled();

  setText($(licSel), filled ? d.license : '미기재');

  const link = (txt, url) => url
    ? `<a href="${esc(url)}" target="_blank" rel="noopener">${esc(txt)}</a>` : esc(txt);

  const counts = Object.entries(d.counts ?? {});
  const rows = [
    ['NAME', link(d.name, d.url)],
    ['SOURCE', esc(d.source)],
    ['LICENSE', link(d.license, d.licenseUrl)],
    ['LABELING', esc(d.labeling)],
    ['CLASSES', esc(d.classNote)],
  ];
  if (counts.length) {
    rows.push(['COUNTS', `<div class="ds-counts">${counts.map(([id, n]) =>
      `<span class="ds-chip"><i style="background:${classOf(id).color}"></i>${esc(classOf(id).label)} ${n}</span>`
    ).join('')}</div>`]);
  }
  if (d.splitNote) rows.push(['SPLIT', esc(d.splitNote)]);

  setHtml($(bodySel), rows.map(([k, v]) =>
    `<div class="ds-row"><div class="ds-k">${k}</div><div class="ds-v">${v}</div></div>`).join(''));

  // 아직 안 채웠으면 화면에서 경고합니다 — 대회 필수 항목이라 놓치면 감점입니다
  const warn = $('#ds-warn');
  if (warn) {
    if (!filled) {
      setHtml(warn, '<b>데이터셋 정보가 비어 있습니다.</b> '
        + '<code>project.config.js</code> 의 <code>dataset</code> 블록을 채우세요 — '
        + '출처·라이선스·한계 표기는 대회 필수 항목입니다.');
      show(warn, true);
    } else {
      show(warn, false);
    }
  }
}

/** 한계 목록 — hard(근본적) / soft(개선 가능) 구분 표시 */
export function renderLimitations(sel) {
  const list = PROJECT.dataset.limitations ?? [];
  if (!list.length) {
    setHtml($(sel), '<div class="empty">한계를 아직 적지 않았습니다 — 대회 평가에서 가장 크게 작용하는 항목입니다.</div>');
    return;
  }
  setHtml($(sel), list.map(l => `
    <div class="limit" data-kind="${l.kind === 'hard' ? 'hard' : 'soft'}">
      <div class="limit-k">${l.kind === 'hard' ? '해결 불가' : '개선 가능'}</div>
      <div class="limit-t">${esc(l.text)}</div>
    </div>`).join(''));
}

/** 내장 샘플 버튼 — 파일 없이도 데모를 돌릴 수 있게 합니다 */
export function renderSamples(samples) {
  const box = $('#samples');
  if (!samples.length) { show(box, false); return; }
  setHtml($('#sample-list'), samples.map((s, i) => `
    <button type="button" class="sample-btn" data-sample="${i}">
      <img src="${s.src}" alt="" />
      <span>${esc(s.label)}</span>
    </button>`).join(''));
  show(box, true);
}
