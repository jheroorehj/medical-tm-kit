/* history.js — 판독 이력과 감사 로그
 * ---------------------------------------------------------------------------
 * 의료 도메인 서비스는 "언제, 무엇을, 어떤 근거로 판정했는지"가 남아야 합니다.
 * 서버가 없으므로 브라우저에 남기고, CSV로 내보내 보고서에 붙입니다.
 *
 * 원본 영상은 저장하지 않습니다 — 96px 썸네일과 판정 메타데이터만 남깁니다.
 * (용량 문제와 개인정보 문제를 동시에 피하는 선택)
 */

import * as store from './store.js';
import { CLASS_IDS } from '../project.config.js';

const THUMB = 96;

/** 캔버스/이미지에서 작은 썸네일 dataURL을 만듭니다. */
export function makeThumb(source, size = THUMB) {
  const c = document.createElement('canvas');
  c.width = size;
  c.height = size;
  const ctx = c.getContext('2d');
  const w = source.videoWidth || source.naturalWidth || source.width;
  const h = source.videoHeight || source.naturalHeight || source.height;
  const side = Math.min(w, h);
  ctx.drawImage(source, (w - side) / 2, (h - side) / 2, side, side, 0, 0, size, size);
  return c.toDataURL('image/jpeg', 0.6);
}

/**
 * 판정 1건을 이력에 기록합니다.
 * @param {object} p
 * @param {'webcam'|'upload'|'live'} p.source
 * @param {object} p.decision infer.decide() 결과
 * @param {object|null} p.quality measureFrame 결과
 * @param {string|null} p.thumb dataURL
 * @param {object} p.settings 판정 당시 임계값 (사후 재현을 위해 함께 남깁니다)
 */
export function record({ source, decision, quality, thumb, settings }) {
  const entry = {
    ts: new Date().toISOString(),
    source,
    status: decision.status,
    topId: decision.top.id,
    topProb: decision.top.prob,
    calibrated: decision.calibrated?.value ?? null,
    margin: decision.margin,
    entropy: decision.entropy,
    probs: Object.fromEntries(
      Object.entries(decision.probs).map(([k, v]) => [k, +v.toFixed(4)])
    ),
    quality: quality ? {
      blur: +quality.blur.toFixed(1),
      luma: +quality.luma.toFixed(1),
    } : null,
    // 판정 당시 정책을 함께 남깁니다 — 나중에 임계값을 바꿔도 이 판정은 재현됩니다
    policy: {
      hold: settings.holdThreshold,
      margin: settings.marginThreshold,
    },
    thumb: thumb ?? null,
  };
  store.pushHistory(entry);
  return entry;
}

export function list() {
  return store.loadHistory();
}

export function clear() {
  store.clearHistory();
}

/** 이력을 CSV로 — 감사 로그 제출용 */
export function toCsv(entries = list()) {
  const head = [
    'timestamp', 'source', 'status', 'predicted', 'similarity',
    'calibrated', 'margin', 'entropy', 'hold_threshold',
    ...CLASS_IDS.map(c => `p_${c}`),
  ];
  const rows = entries.map(e => [
    e.ts,
    e.source,
    e.status,
    cell(e.topId),
    fixed(e.topProb),
    fixed(e.calibrated),
    fixed(e.margin),
    fixed(e.entropy),
    fixed(e.policy?.hold),
    ...CLASS_IDS.map(c => fixed(e.probs?.[c])),
  ].join(','));
  return [head.join(','), ...rows].join('\n');
}

function fixed(v, d = 4) {
  return v == null || !Number.isFinite(v) ? '' : v.toFixed(d);
}

function cell(s) {
  const v = String(s ?? '');
  return /[",\n]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v;
}

/** 브라우저에서 파일 다운로드를 트리거합니다. */
export function download(filename, text, mime = 'text/csv;charset=utf-8') {
  // Excel이 한글 CSV를 깨뜨리지 않도록 BOM을 붙입니다
  const bom = mime.startsWith('text/csv') ? '﻿' : '';
  const blob = new Blob([bom + text], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** 이력 요약 — 이력 탭 상단 통계 */
export function summarize(entries = list()) {
  const total = entries.length;
  if (!total) return { total: 0, byStatus: {}, byClass: {}, holdRate: 0 };
  const byStatus = {};
  const byClass = {};
  for (const e of entries) {
    byStatus[e.status] = (byStatus[e.status] ?? 0) + 1;
    byClass[e.topId] = (byClass[e.topId] ?? 0) + 1;
  }
  return {
    total, byStatus, byClass,
    holdRate: (byStatus.hold ?? 0) / total,
  };
}
