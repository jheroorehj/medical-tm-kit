/* infer.js — 추론 실행과 판정 결정
 * ---------------------------------------------------------------------------
 *   predictProbs()  단일 모델 추론 → {classId: prob}
 *   runGate()       게이트키퍼 1단 통과 판정
 *   runClassify()   주분류기 + 앙상블 가중 소프트보팅
 *   decide()        확률 벡터 → 최종 판정 (여기가 정책의 중심)
 *
 * decide()는 이 서비스의 "임상 안전 정책"이 코드로 표현된 지점입니다.
 * 그냥 1위를 발표하지 않고 4개의 관문을 통과시킵니다.
 */

import { SLOT } from './models.js';
import { normalizedEntropy } from './temporal.js';
import { classOf, INVALID_IDS, POSITIVE_IDS } from '../project.config.js';
import { applyCalibration } from './calibrate.js';

/** TM 예측 배열을 {className: probability} 객체로 정규화합니다. */
export async function predictProbs(model, source) {
  const preds = await model.predict(source);
  const out = {};
  for (const p of preds) out[p.className] = p.probability;
  return out;
}

/**
 * 게이트키퍼 1단. "이게 판독할 대상인가?"만 판단합니다.
 * @returns {{skipped:boolean, pass:boolean, topId:string, prob:number}}
 */
export async function runGate(registry, source, gateCfg) {
  const entry = registry.get(SLOT.GATE);
  if (!gateCfg.enabled || !entry) {
    return { skipped: true, pass: true, topId: '', prob: 0 };
  }
  const probs = await predictProbs(entry.model, source);
  const ranked = Object.keys(probs).sort((a, b) => probs[b] - probs[a]);
  const topId = ranked[0];
  const prob = probs[topId] ?? 0;
  const pass = gateCfg.passClasses.includes(topId) && prob >= gateCfg.minConfidence;
  return { skipped: false, pass, topId, prob, probs };
}

/**
 * 주분류기 + 앙상블 가중 소프트보팅.
 * 서로 다른 전처리(원본 / CLAHE / ROI 확대)로 학습한 모델들의 확률을 평균하면
 * 단일 모델보다 눈에 띄게 안정적입니다. 등록된 모델이 1개면 그냥 그 결과입니다.
 *
 * @returns {{probs:Record<string,number>, perModel:Array<{slot,label,probs}>}}
 */
export async function runClassify(registry, source) {
  const slots = registry.classifierSlots();
  if (slots.length === 0) throw new Error('분류 모델이 로드되지 않았습니다');

  const perModel = [];
  for (const slot of slots) {
    const entry = registry.get(slot);
    const probs = await predictProbs(entry.model, source);
    perModel.push({ slot, label: entry.label, weight: entry.weight, probs });
  }

  // 가중 평균 후 재정규화
  const acc = {};
  let wsum = 0;
  for (const m of perModel) {
    wsum += m.weight;
    for (const [id, p] of Object.entries(m.probs)) {
      acc[id] = (acc[id] ?? 0) + p * m.weight;
    }
  }
  const probs = {};
  let total = 0;
  for (const [id, v] of Object.entries(acc)) { probs[id] = v / wsum; total += probs[id]; }
  if (total > 0 && Math.abs(total - 1) > 1e-6) {
    for (const id of Object.keys(probs)) probs[id] /= total;
  }

  return { probs, perModel };
}

/* ── 판정 ────────────────────────────────────────────────────────────── */

/** 판정 상태 코드 */
export const STATUS = {
  GATED: 'gated',          // 게이트키퍼가 막음 — 판독 대상 없음
  INVALID: 'invalid',      // 무효 입력 클래스가 1위
  HOLD: 'hold',            // 임계값 미달 — 판단 보류
  AMBIGUOUS: 'ambiguous',  // 1·2위 마진 부족 — 구분 어려움
  OK: 'ok',                // 판정 성립
};

/**
 * 확률 벡터를 최종 판정으로 바꿉니다. 4개의 관문을 순서대로 통과시킵니다.
 *
 *   관문 0. 게이트키퍼가 막았는가?          → GATED
 *   관문 1. 무효 입력 클래스가 1위인가?      → INVALID
 *   관문 2. 1위 유사도가 임계값 미만인가?    → HOLD
 *   관문 3. 1·2위 차이가 마진 미만인가?      → AMBIGUOUS
 *   통과                                    → OK
 *
 * @param {Record<string,number>} probs
 * @param {object} settings {holdThreshold, marginThreshold, useCalibration}
 * @param {object} ctx {gate, calibration, copy}
 */
export function decide(probs, settings, ctx = {}) {
  const ranked = Object.keys(probs)
    .sort((a, b) => probs[b] - probs[a])
    .map(id => ({ id, prob: probs[id], def: classOf(id) }));

  const top = ranked[0] ?? { id: '', prob: 0, def: classOf('') };
  const second = ranked[1] ?? null;
  const margin = second ? top.prob - second.prob : top.prob;
  const entropy = normalizedEntropy(probs);
  const copy = ctx.copy ?? {};

  // 보정된 신뢰도 (캘리브레이션 테이블이 있고 사용 설정이 켜져 있을 때만)
  const calibrated = (settings.useCalibration && ctx.calibration)
    ? applyCalibration(ctx.calibration, top.prob)
    : null;

  const base = {
    ranked, top, second, margin, entropy, probs,
    calibrated,          // {value, n, binLabel} | null
  };

  // 관문 0 — 게이트키퍼
  if (ctx.gate && !ctx.gate.skipped && !ctx.gate.pass) {
    return {
      ...base, status: STATUS.GATED, tone: 'neutral', icon: '🔍',
      headline: '판독 대상 없음',
      detail: copy.gateBlockedMessage ?? '판독 대상이 확인되지 않았습니다.',
      advice: '',
      trustworthy: false,
    };
  }

  // 관문 1 — 무효 입력
  if (INVALID_IDS.includes(top.id)) {
    return {
      ...base, status: STATUS.INVALID, tone: 'neutral', icon: '🚫',
      headline: top.def.label,
      detail: `무효 입력으로 분류되었습니다 (유사도 ${pct(top.prob)})`,
      advice: top.def.advice ?? '',
      trustworthy: false,
    };
  }

  // 관문 2 — 임계값 미달
  if (top.prob < settings.holdThreshold) {
    return {
      ...base, status: STATUS.HOLD, tone: 'danger', icon: '⚠️',
      headline: '판단 보류',
      detail: `가장 높은 유사도 ${pct(top.prob)} — 기준값 ${pct(settings.holdThreshold)}에 못 미칩니다`,
      advice: copy.holdMessage ?? '',
      trustworthy: false,
    };
  }

  // 관문 3 — 마진 부족
  if (second && margin < settings.marginThreshold) {
    return {
      ...base, status: STATUS.AMBIGUOUS, tone: 'warn', icon: '🤔',
      headline: '구분 어려움',
      detail: `${top.def.label} ${pct(top.prob)} vs ${second.def.label} ${pct(second.prob)} — 차이 ${pct(margin)}`,
      advice: copy.marginMessage ?? '',
      trustworthy: false,
    };
  }

  // 통과
  const isPositive = POSITIVE_IDS.includes(top.id);
  return {
    ...base, status: STATUS.OK,
    tone: isPositive ? 'danger' : 'ok',
    icon: isPositive ? '🔴' : '🟢',
    headline: top.def.label,
    detail: `${top.def.label}와(과) 유사도 ${pct(top.prob)}`
          + (calibrated ? ` · 보정 신뢰도 ${pct(calibrated.value)}` : ''),
    advice: top.def.advice ?? '',
    trustworthy: true,
  };
}

export function pct(v, digits = 1) {
  return `${(v * 100).toFixed(digits)}%`;
}

/**
 * 이미지/비디오를 224 정사각 캔버스로 옮깁니다.
 * TM은 내부적으로 리사이즈하지만, 여기서 미리 중앙 ROI를 잡아 두면
 * 전처리 파이프라인(prep/)에서 학습 데이터를 만든 방식과 추론 방식이 일치합니다.
 * ★ 학습과 추론의 전처리를 같게 유지하는 것이 정확도에 크게 기여합니다.
 */
export function toSquareCanvas(source, roi = 1.0, size = 224, out = null) {
  const w = source.videoWidth || source.naturalWidth || source.width;
  const h = source.videoHeight || source.naturalHeight || source.height;
  const canvas = out || document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  const side = Math.min(w, h) * roi;
  ctx.drawImage(source, (w - side) / 2, (h - side) / 2, side, side, 0, 0, size, size);
  return canvas;
}
