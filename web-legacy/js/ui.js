/* ui.js — 렌더링 계층
 * ---------------------------------------------------------------------------
 * DOM을 만지는 모든 코드가 여기 모여 있습니다. main.js는 이벤트 배선만 합니다.
 * 이 파일은 판정 로직을 갖지 않습니다 — infer.decide()의 결과를 그리기만 합니다.
 */

import { classOf, CLASS_IDS } from '../project.config.js';
import { pct } from './infer.js';
import { fmtPct } from './metrics.js';
import { qualityScore } from './quality.js';

export const $ = sel => document.querySelector(sel);
export const $$ = sel => [...document.querySelectorAll(sel)];

export function show(el, on = true) {
  if (el) el.hidden = !on;
}

export function setText(el, text) {
  if (el) el.textContent = text;
}

/** 노트 박스에 톤과 내용을 넣고 표시합니다. */
export function note(el, tone, html) {
  if (!el) return;
  if (!html) { el.hidden = true; return; }
  el.dataset.tone = tone || '';
  el.innerHTML = html;
  el.hidden = false;
}

/* ── 헤더 상태 배지 ───────────────────────────────────────────────────── */

export function setModelPill(state, text) {
  const pill = $('#model-pill');
  pill.dataset.state = state;
  pill.querySelector('.pill-text').textContent = text;
}

export function setSlotState(id, status) {
  const el = $(id);
  if (!el) return;
  el.dataset.state = status.state;
  el.textContent = status.state === 'ok' ? `로드됨 · ${status.message}`
                 : status.state === 'loading' ? '불러오는 중…'
                 : status.state === 'error' ? '실패'
                 : '미로드';
  el.title = status.message || '';
}

/* ── 판정 결과 ────────────────────────────────────────────────────────── */

/**
 * @param {object} d infer.decide() 결과
 * @param {object} extra {jitter, live, temporal}
 */
export function renderVerdict(d, extra = {}) {
  show($('#result-empty'), false);
  show($('#result-body'), true);

  const v = $('#verdict');
  v.dataset.tone = d.tone;
  setText($('#verdict-icon'), d.icon);
  setText($('#verdict-head'), d.headline);
  setText($('#verdict-detail'), d.detail);

  // 안내 문구 — 판정이 성립했으면 클래스별 조치 안내, 아니면 경고 사유
  const adviceHtml = d.trustworthy
    ? (d.advice ? d.advice : '')
    : [d.advice].filter(Boolean).join(' ');
  const adviceEl = $('#advice');
  if (adviceHtml) {
    adviceEl.className = d.trustworthy ? 'note note-advice' : 'note';
    adviceEl.dataset.tone = d.trustworthy ? '' : (d.tone === 'danger' ? 'danger' : 'warn');
    adviceEl.innerHTML = adviceHtml;
    adviceEl.hidden = false;
  } else {
    adviceEl.hidden = true;
  }

  // 불확실성 지표
  setText($('#u-top'), pct(d.top.prob));
  setText($('#u-cal'), d.calibrated
    ? (d.calibrated.n > 0 ? `${pct(d.calibrated.value)}` : '표본 부족')
    : '—');
  $('#u-cal').title = d.calibrated?.n
    ? `${d.calibrated.binLabel} 구간 표본 ${d.calibrated.n}건 기준${d.calibrated.borrowed ? ' (인접 구간에서 차용)' : ''}`
    : '홀드아웃 평가에서 보정표를 먼저 생성하세요';
  setText($('#u-margin'), pct(d.margin));
  setText($('#u-entropy'), d.entropy.toFixed(2));
  setText($('#u-jitter'), extra.jitter != null ? extra.jitter.toFixed(3) : '—');

  renderBars($('#bars'), d.ranked);

  setText($('#result-ts'),
    new Date().toLocaleString('ko-KR', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
    + ' 기준 · 임계값 ' + pct(extra.threshold ?? 0, 0));

  // 실시간 확정 진행
  const cr = $('#confirm-row');
  if (extra.temporal && extra.live) {
    show(cr, true);
    setText($('#confirm-count'), `${extra.temporal.streak}/${extra.confirmFrames}`);
    $('#confirm-fill').style.width = `${Math.round(extra.temporal.progress * 100)}%`;
    setText($('#confirm-label'),
      extra.temporal.confirmed ? '판정 확정됨' : '판정 확정까지');
  } else {
    show(cr, false);
  }
}

/** 클래스별 유사도 막대. 노드를 재사용해 width 트랜지션이 실제로 재생되게 합니다. */
export function renderBars(container, ranked) {
  const wanted = ranked.map(r => r.id);
  const existing = [...container.children].map(c => c.dataset.id);
  const same = wanted.length === existing.length && wanted.every((id, i) => id === existing[i]);

  if (!same) {
    container.innerHTML = '';
    for (const r of ranked) {
      const kindLabel = r.def.kind === 'invalid' ? '무효'
                      : r.def.kind === 'positive' ? '이상' : '';
      const el = document.createElement('div');
      el.className = 'bar';
      el.dataset.id = r.id;
      el.innerHTML = `
        <div class="bar-meta">
          <span class="bar-label">${esc(r.def.label)}${kindLabel ? `<span class="kind">${kindLabel}</span>` : ''}</span>
          <span class="bar-pct">0.0%</span>
        </div>
        <div class="bar-track"><div class="bar-fill" style="background:${r.def.color}"></div></div>`;
      container.appendChild(el);
    }
  }

  // 노드가 이미 존재하는 상태에서 width만 바꾸므로 CSS 트랜지션이 재생됩니다.
  // (원본 템플릿은 innerHTML로 매번 새로 만들어 트랜지션이 죽어 있었습니다)
  requestAnimationFrame(() => {
    [...container.children].forEach((el, i) => {
      const r = ranked[i];
      if (!r) return;
      el.querySelector('.bar-pct').textContent = pct(r.prob);
      el.querySelector('.bar-fill').style.width = `${(r.prob * 100).toFixed(1)}%`;
    });
  });
}

export function clearResult() {
  show($('#result-empty'), true);
  show($('#result-body'), false);
}

/* ── 품질 표시 ────────────────────────────────────────────────────────── */

/**
 * @param {'q'|'iq'} prefix 웹캠(q) / 이미지(iq) 스트립 구분
 */
export function renderQuality(prefix, m, ev, settings) {
  const set = (id, val, tone) => {
    const el = $(`#${prefix}-${id}`);
    if (!el) return;
    el.textContent = val;
    if (tone) el.dataset.tone = tone; else delete el.dataset.tone;
  };
  if (!m) {
    ['score', 'blur', 'luma', 'contrast'].forEach(k => set(k, '—'));
    return;
  }
  const score = qualityScore(m, settings);
  set('score', `${score}`, score >= 60 ? 'good' : score >= 35 ? '' : 'bad');
  set('blur', m.blur.toFixed(0), m.blur >= settings.blurMin ? 'good' : 'bad');
  set('luma', m.luma.toFixed(0),
      (m.luma >= settings.lumaMin && m.luma <= settings.lumaMax) ? 'good' : 'bad');
  set('contrast', m.contrast.toFixed(0), m.contrast >= 12 ? 'good' : 'bad');
}

export function renderCoach(ev) {
  const el = $('#cam-coach');
  const box = $('#cam-guide')?.querySelector('.guide-box');
  if (!ev || (ev.ok && !ev.coach)) {
    show(el, false);
    box?.classList.remove('bad');
    box?.classList.add('good');
    return;
  }
  el.textContent = ev.ok ? ev.coach : ev.coach;
  el.dataset.tone = ev.ok ? 'ok' : '';
  show(el, true);
  box?.classList.toggle('bad', !ev.ok);
  box?.classList.toggle('good', ev.ok);
}

/* ── 타일 ─────────────────────────────────────────────────────────────── */

/** @param {Array<{k:string,v:string,sub?:string,tone?:string}>} tiles */
export function renderTiles(container, tiles) {
  container.innerHTML = tiles.map(t => `
    <div class="tile"${t.tone ? ` data-tone="${t.tone}"` : ''}>
      <span class="tile-k">${esc(t.k)}</span>
      <span class="tile-v">${esc(t.v)}</span>
      ${t.sub ? `<span class="tile-sub">${esc(t.sub)}</span>` : ''}
    </div>`).join('');
}

/* ── 혼동행렬 ─────────────────────────────────────────────────────────── */

export function renderConfusion(table, cm) {
  const { matrix, classIds } = cm;
  const rowTotals = matrix.map(r => r.reduce((a, b) => a + b, 0));

  const head = `<thead><tr><th>실제 ＼ 예측</th>${
    classIds.map(id => `<th>${esc(classOf(id).label)}</th>`).join('')
  }<th>합계</th><th>재현율</th></tr></thead>`;

  const body = matrix.map((row, i) => {
    const total = rowTotals[i];
    const recall = total > 0 ? row[i] / total : null;
    return `<tr><th>${esc(classOf(classIds[i]).label)}</th>${
      row.map((n, j) => {
        const cls = n === 0 ? 'zero' : (i === j ? 'diag' : 'off');
        return `<td class="num ${cls}">${n}</td>`;
      }).join('')
    }<td class="num">${total}</td><td class="num">${fmtPct(recall)}</td></tr>`;
  }).join('');

  table.innerHTML =
    `<caption>대각선이 정답입니다. 붉은 칸은 오분류 — 어느 클래스끼리 헷갈리는지가 클래스 재설계의 근거가 됩니다.</caption>`
    + head + `<tbody>${body}</tbody>`;
}

export function renderPerClass(table, perClass) {
  const head = `<thead><tr><th>클래스</th><th>표본</th><th>정밀도</th><th>재현율</th><th>F1</th>
                <th>TP</th><th>FP</th><th>FN</th></tr></thead>`;
  const body = perClass.map(m => `
    <tr>
      <th>${esc(classOf(m.id).label)}</th>
      <td class="num">${m.support}</td>
      <td class="num">${fmtPct(m.precision)}</td>
      <td class="num">${fmtPct(m.recall)}</td>
      <td class="num">${fmtPct(m.f1)}</td>
      <td class="num">${m.tp}</td>
      <td class="num">${m.fp}</td>
      <td class="num">${m.fn}</td>
    </tr>`).join('');
  table.innerHTML = `<caption>클래스별 성능. 표본이 20건 미만인 클래스의 수치는 신뢰하기 어렵습니다.</caption>`
    + head + `<tbody>${body}</tbody>`;
}

/* ── 임계값 스윕 곡선 ─────────────────────────────────────────────────── */

export function renderSweep(container, sweep, current) {
  if (!sweep.length) { container.innerHTML = ''; return; }

  const W = 100, H = 62, PAD_L = 8, PAD_R = 3, PAD_T = 4, PAD_B = 9;
  const x = t => PAD_L + t * (W - PAD_L - PAD_R);
  const y = v => PAD_T + (1 - v) * (H - PAD_T - PAD_B);

  const path = (key) => {
    const pts = sweep
      .filter(m => m[key] != null)
      .map(m => `${x(m.threshold).toFixed(2)},${y(m[key]).toFixed(2)}`);
    return pts.length ? `M${pts.join(' L')}` : '';
  };

  const gridY = [0, 0.25, 0.5, 0.75, 1];
  const gridX = [0.2, 0.4, 0.6, 0.8];

  container.innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img"
         aria-label="임계값에 따른 민감도와 특이도 변화 곡선">
      ${gridY.map(v => `<line class="grid-line" x1="${PAD_L}" y1="${y(v)}" x2="${W - PAD_R}" y2="${y(v)}"/>
        <text class="axis-label" x="0" y="${y(v) + 1.6}">${(v * 100).toFixed(0)}</text>`).join('')}
      ${gridX.map(t => `<text class="axis-label" x="${x(t) - 2}" y="${H - 2}">${t.toFixed(1)}</text>`).join('')}
      <path class="line-hold" d="${path('holdRate')}"/>
      <path class="line-spec" d="${path('specificity')}"/>
      <path class="line-sens" d="${path('sensitivity')}"/>
      <line class="cursor" x1="${x(current)}" y1="${PAD_T}" x2="${x(current)}" y2="${H - PAD_B}"/>
    </svg>
    <div class="sweep-legend">
      <span><i style="background:var(--danger)"></i>민감도 (놓치지 않는 능력)</span>
      <span><i style="background:var(--accent)"></i>특이도 (정상을 정상으로)</span>
      <span><i style="background:var(--text-3)"></i>보류율</span>
      <span class="muted">가로축 = 판단 보류 임계값</span>
    </div>`;
}

/* ── 캘리브레이션 표 ──────────────────────────────────────────────────── */

export function renderCalibration(table, cal) {
  if (!cal) { table.hidden = true; return; }
  const head = `<thead><tr><th>모델 출력 구간</th><th>표본</th><th>평균 출력</th>
                <th>실측 정확도</th><th>격차</th></tr></thead>`;
  const body = cal.bins.filter(b => b.n > 0).map(b => {
    const gap = b.accuracy - b.avgConfidence;
    const tone = Math.abs(gap) < 0.05 ? '' : (gap < 0 ? 'off' : 'diag');
    return `<tr>
      <th>${Math.round(b.lo * 100)}–${Math.round(b.hi * 100)}%</th>
      <td class="num">${b.n}</td>
      <td class="num">${fmtPct(b.avgConfidence)}</td>
      <td class="num">${fmtPct(b.accuracy)}</td>
      <td class="num ${tone}">${gap >= 0 ? '+' : ''}${(gap * 100).toFixed(1)}%p</td>
    </tr>`;
  }).join('');
  table.innerHTML =
    `<caption>격차가 음수면 모델이 과신하고 있다는 뜻입니다 (말한 것보다 실제로 덜 맞음). 표본 5건 미만 구간은 판독 화면에서 인접 구간 값을 차용합니다.</caption>`
    + head + `<tbody>${body}</tbody>`;
  table.hidden = false;
}

/* ── 실패 사례 갤러리 ─────────────────────────────────────────────────── */

let galleryUrls = [];

export function renderGallery(container, failures, files) {
  galleryUrls.forEach(u => URL.revokeObjectURL(u));
  galleryUrls = [];

  if (!failures.length) {
    container.innerHTML = `<p class="hint">오분류 사례가 없습니다. 표본이 적거나 데이터가 너무 쉬운지 확인하세요.</p>`;
    return;
  }

  container.innerHTML = failures.map(f => {
    const file = files?.get(f.file);
    let src = '';
    if (file) { src = URL.createObjectURL(file); galleryUrls.push(src); }
    return `
      <div class="gitem">
        ${src ? `<img src="${src}" alt="${esc(f.file)}" loading="lazy" />` : '<div style="aspect-ratio:1;background:#1a1a1a"></div>'}
        <div class="gitem-meta">
          <div class="wrong">→ ${esc(classOf(f.topId).label)}</div>
          <div class="right">실제 ${esc(classOf(f.trueId).label)}</div>
          <div class="p">${pct(f.topProb)}</div>
        </div>
      </div>`;
  }).join('');
}

/* ── 이력 표 ──────────────────────────────────────────────────────────── */

const STATUS_LABEL = {
  ok: '판정', hold: '보류', ambiguous: '구분 어려움',
  invalid: '무효 입력', gated: '대상 없음',
};

export function renderHistory(table, entries) {
  if (!entries.length) {
    table.innerHTML = `<caption>아직 판독 이력이 없습니다.</caption>`;
    return;
  }
  const head = `<thead><tr><th></th><th>시각</th><th>입력</th><th>상태</th>
                <th>예측</th><th>유사도</th><th>보정</th><th>임계값</th></tr></thead>`;
  const body = entries.slice(0, 100).map(e => `
    <tr>
      <td>${e.thumb ? `<img src="${e.thumb}" width="32" height="32" alt=""
             style="border-radius:4px;display:block">` : ''}</td>
      <th>${new Date(e.ts).toLocaleString('ko-KR', { month: '2-digit', day: '2-digit',
            hour: '2-digit', minute: '2-digit', second: '2-digit' })}</th>
      <td>${e.source === 'live' ? '실시간' : e.source === 'webcam' ? '캡처' : '업로드'}</td>
      <td>${STATUS_LABEL[e.status] ?? e.status}</td>
      <td>${esc(classOf(e.topId).label)}</td>
      <td class="num">${fmtPct(e.topProb)}</td>
      <td class="num">${e.calibrated != null ? fmtPct(e.calibrated) : '—'}</td>
      <td class="num">${fmtPct(e.policy?.hold, 0)}</td>
    </tr>`).join('');
  table.innerHTML = `<caption>최근 100건. 원본 영상은 저장되지 않고 96px 썸네일만 남습니다.</caption>`
    + head + `<tbody>${body}</tbody>`;
}

/* ── 클래스 정합성 리포트 ─────────────────────────────────────────────── */

/**
 * TM 모델의 클래스명과 project.config.js의 클래스 id가 어긋났는지 보고합니다.
 * 실습에서 오타 하나로 몇 시간을 날리는 사고를 여기서 잡습니다.
 */
export function renderClassCheck(el, verify, ensembleCheck) {
  if (!verify.modelLabels.length) { el.hidden = true; return; }

  const lines = [];
  if (verify.ok) {
    lines.push(`<b>클래스 정합성 확인됨</b> — 모델과 설정의 클래스 ${verify.modelLabels.length}개가 정확히 일치합니다.`);
  } else {
    lines.push(`<b>클래스 불일치</b> — <code>project.config.js</code>를 수정하세요.`);
    if (verify.missing.length) {
      lines.push(`설정에는 있으나 모델에 없음: ${verify.missing.map(c => `<code>${esc(c)}</code>`).join(', ')}`);
    }
    if (verify.extra.length) {
      lines.push(`모델에는 있으나 설정에 없음: ${verify.extra.map(c => `<code>${esc(c)}</code>`).join(', ')}`);
    }
    lines.push(`모델의 실제 클래스: ${verify.modelLabels.map(c => `<code>${esc(c)}</code>`).join(', ')}`);
  }
  if (ensembleCheck && !ensembleCheck.ok) {
    lines.push(`<b>앙상블 경고</b> — 클래스 집합이 다른 모델이 있어 보팅이 왜곡됩니다: ${ensembleCheck.mismatched.join(', ')}`);
  }

  note(el, verify.ok && (!ensembleCheck || ensembleCheck.ok) ? 'ok' : 'danger',
       lines.join('<br>'));
}

/* ── 유틸 ─────────────────────────────────────────────────────────────── */

/** 모델 metadata에서 온 문자열을 그대로 innerHTML에 넣지 않도록 이스케이프합니다. */
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

export { CLASS_IDS };
