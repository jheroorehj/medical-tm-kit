/* calibrate.js — 신뢰도 캘리브레이션
 * ---------------------------------------------------------------------------
 * softmax 확률은 "신뢰도"가 아닙니다. 심하게 과신합니다.
 * 모델이 91%라고 말한 예측들만 모아 실제 정답률을 세어 보면 70% 아래인 경우가
 * 흔합니다. 의료 도메인에서 이 격차를 그대로 사용자에게 보여주는 것은 위험합니다.
 *
 * 여기서는 홀드아웃 셋으로 "신뢰도 구간별 실측 정확도" 표를 만들고,
 * UI에는 원시 softmax 값 대신 보정된 값을 함께 표시합니다.
 *
 *   모델 출력 91%  →  이 구간의 실측 정확도 72%
 *
 * 발표에서 강한 카드가 되는 지점입니다. 대부분의 팀은 softmax를 확률로 착각합니다.
 * 주제와 무관한 순수 통계 계층입니다.
 */

const DEFAULT_BINS = 10;
const MIN_BIN_N = 5;        // 이보다 표본이 적은 구간은 신뢰하지 않습니다

/**
 * 홀드아웃 예측 기록으로 캘리브레이션 표를 만듭니다.
 * @param {Array<{prob:number, correct:boolean}>} records 1위 확률과 정답 여부
 * @param {number} nBins
 * @returns {{bins:Array, n:number, ece:number, createdAt:string}|null}
 */
export function buildCalibration(records, nBins = DEFAULT_BINS) {
  const valid = records.filter(r => Number.isFinite(r.prob));
  if (valid.length < MIN_BIN_N * 2) return null;   // 표본이 너무 적으면 만들지 않음

  const bins = Array.from({ length: nBins }, (_, i) => ({
    lo: i / nBins,
    hi: (i + 1) / nBins,
    n: 0,
    correct: 0,
    confSum: 0,
    accuracy: null,
    avgConfidence: null,
  }));

  for (const r of valid) {
    // 1.0은 마지막 구간에 포함
    let idx = Math.floor(r.prob * nBins);
    if (idx >= nBins) idx = nBins - 1;
    if (idx < 0) idx = 0;
    const b = bins[idx];
    b.n++;
    b.confSum += r.prob;
    if (r.correct) b.correct++;
  }

  for (const b of bins) {
    if (b.n > 0) {
      b.accuracy = b.correct / b.n;
      b.avgConfidence = b.confSum / b.n;
    }
  }

  // ECE (Expected Calibration Error) — 신뢰도와 실측 정확도의 가중 평균 격차
  const N = valid.length;
  let ece = 0;
  for (const b of bins) {
    if (b.n === 0) continue;
    ece += (b.n / N) * Math.abs(b.accuracy - b.avgConfidence);
  }

  return {
    bins, n: N, ece, nBins,
    createdAt: new Date().toISOString(),
  };
}

/**
 * 원시 확률을 보정된 신뢰도로 변환합니다.
 * 표본이 부족한 구간은 가장 가까운 충분한 구간의 값을 빌려 씁니다.
 * 그래도 없으면 원값을 그대로 돌려주고 n=0으로 표시합니다.
 *
 * @returns {{value:number, n:number, binLabel:string, borrowed:boolean}}
 */
export function applyCalibration(table, prob) {
  if (!table || !table.bins?.length) {
    return { value: prob, n: 0, binLabel: '보정 불가', borrowed: false };
  }
  const nBins = table.nBins ?? table.bins.length;
  let idx = Math.min(nBins - 1, Math.max(0, Math.floor(prob * nBins)));

  const usable = i => table.bins[i] && table.bins[i].n >= MIN_BIN_N;
  let borrowed = false;

  if (!usable(idx)) {
    // 좌우로 가장 가까운 사용 가능 구간 탐색
    let found = -1;
    for (let d = 1; d < nBins; d++) {
      if (usable(idx - d)) { found = idx - d; break; }
      if (usable(idx + d)) { found = idx + d; break; }
    }
    if (found === -1) {
      return { value: prob, n: 0, binLabel: '표본 부족', borrowed: false };
    }
    idx = found;
    borrowed = true;
  }

  const b = table.bins[idx];
  return {
    value: b.accuracy,
    n: b.n,
    binLabel: `${Math.round(b.lo * 100)}–${Math.round(b.hi * 100)}%`,
    borrowed,
  };
}

/** 캘리브레이션 품질을 한 문장으로 요약합니다 (UI 배지용). */
export function describeCalibration(table) {
  if (!table) return { tone: 'neutral', text: '캘리브레이션 없음 — 홀드아웃 평가를 먼저 실행하세요' };
  const ecePct = (table.ece * 100).toFixed(1);
  if (table.ece < 0.05) return { tone: 'ok',     text: `보정 오차 ${ecePct}%p — 신뢰도가 잘 맞습니다 (표본 ${table.n})` };
  if (table.ece < 0.15) return { tone: 'warn',   text: `보정 오차 ${ecePct}%p — 다소 과신 경향 (표본 ${table.n})` };
  return                       { tone: 'danger', text: `보정 오차 ${ecePct}%p — 모델이 크게 과신하고 있습니다 (표본 ${table.n})` };
}
