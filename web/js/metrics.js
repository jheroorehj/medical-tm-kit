/* metrics.js — 정직한 성능 평가
 * ---------------------------------------------------------------------------
 * TM이 화면에 보여주는 정확도를 발표에 그대로 쓰면 안 됩니다. 두 가지 이유:
 *
 *   ① TM은 학습/검증 분할을 우리가 통제할 수 없습니다 (자동 랜덤 분할).
 *      증강 이미지를 섞어 올리면 같은 원본의 회전본이 학습셋과 검증셋에
 *      동시에 들어가 정확도가 부풀려집니다. (데이터 누출)
 *   ② 의료 도메인의 언어는 정확도가 아니라 민감도와 특이도입니다.
 *
 * 그래서 prep/pipeline.py가 원본 단위로 떼어 둔 홀드아웃 셋을 TM에 넣지 않고,
 * 여기서 직접 평가합니다. 이 수치가 발표에 쓸 값입니다.
 *
 * 주제와 무관한 순수 통계 계층입니다.
 */

/**
 * 평가 기록 1건
 * @typedef {{trueId:string, probs:Record<string,number>, topId:string, topProb:number, file?:string}} EvalRecord
 */

/** argmax 기준 혼동행렬. matrix[실제][예측] = 개수 */
export function confusionMatrix(records, classIds) {
  const idx = Object.fromEntries(classIds.map((id, i) => [id, i]));
  const n = classIds.length;
  const matrix = Array.from({ length: n }, () => new Array(n).fill(0));
  let unknown = 0;

  for (const r of records) {
    const ti = idx[r.trueId];
    const pi = idx[r.topId];
    if (ti == null || pi == null) { unknown++; continue; }
    matrix[ti][pi]++;
  }
  return { matrix, classIds, unknown };
}

/** 클래스별 정밀도/재현율/F1 */
export function perClassMetrics(cm) {
  const { matrix, classIds } = cm;
  const n = classIds.length;
  return classIds.map((id, i) => {
    const tp = matrix[i][i];
    let fn = 0, fp = 0;
    for (let j = 0; j < n; j++) {
      if (j !== i) { fn += matrix[i][j]; fp += matrix[j][i]; }
    }
    const support = tp + fn;
    const precision = tp + fp > 0 ? tp / (tp + fp) : null;
    const recall = support > 0 ? tp / support : null;
    const f1 = (precision != null && recall != null && precision + recall > 0)
      ? 2 * precision * recall / (precision + recall) : null;
    return { id, tp, fp, fn, support, precision, recall, f1 };
  });
}

export function overallAccuracy(cm) {
  const { matrix } = cm;
  let correct = 0, total = 0;
  for (let i = 0; i < matrix.length; i++) {
    for (let j = 0; j < matrix.length; j++) {
      total += matrix[i][j];
      if (i === j) correct += matrix[i][j];
    }
  }
  return total > 0 ? correct / total : null;
}

/* ── 이진 임상 지표 ──────────────────────────────────────────────────── *
 * "발견해야 하는 소견(positive)" vs "그 외"의 2×2로 환원합니다.
 *
 * 보류(판단 유보) 처리 정책이 지표를 크게 바꿉니다:
 *   'escalate' — 보류를 "양성 의심으로 2차 확인에 넘김"으로 간주 (임상적으로 안전)
 *                → 민감도에 유리, 특이도에 불리
 *   'exclude'  — 보류를 분석에서 제외하고 확정 판정만 평가
 *                → 순수 모델 성능. 보류율을 함께 보고해야 정직합니다
 * ------------------------------------------------------------------------ */

/**
 * @param {EvalRecord[]} records
 * @param {string[]} positiveIds  kind==='positive' 클래스 id 목록
 * @param {number} threshold      판단 보류 임계값
 * @param {'escalate'|'exclude'} holdPolicy
 */
export function binaryMetrics(records, positiveIds, threshold, holdPolicy = 'escalate') {
  const pos = new Set(positiveIds);
  let tp = 0, fp = 0, tn = 0, fn = 0, held = 0, excluded = 0;

  for (const r of records) {
    const actualPos = pos.has(r.trueId);
    const isHeld = r.topProb < threshold;
    if (isHeld) held++;

    let predPos;
    if (isHeld) {
      if (holdPolicy === 'exclude') { excluded++; continue; }
      predPos = true;                       // escalate: 보류는 양성 의심 취급
    } else {
      predPos = pos.has(r.topId);
    }

    if (actualPos && predPos) tp++;
    else if (!actualPos && predPos) fp++;
    else if (!actualPos && !predPos) tn++;
    else fn++;
  }

  const evaluated = tp + fp + tn + fn;
  const div = (a, b) => (b > 0 ? a / b : null);

  return {
    threshold, holdPolicy,
    tp, fp, tn, fn, held, excluded, evaluated,
    holdRate: div(held, records.length),
    sensitivity: div(tp, tp + fn),     // 재현율 — 놓치지 않는 능력
    specificity: div(tn, tn + fp),     // 정상을 정상으로 보는 능력
    ppv: div(tp, tp + fp),             // 양성 예측도
    npv: div(tn, tn + fn),             // 음성 예측도
    accuracy: div(tp + tn, evaluated),
    balanced: (() => {
      const se = div(tp, tp + fn), sp = div(tn, tn + fp);
      return se != null && sp != null ? (se + sp) / 2 : null;
    })(),
  };
}

/**
 * 임계값을 0.05~0.95로 훑어 민감도-특이도 트레이드오프를 만듭니다.
 * 설정 탭의 슬라이더가 이 곡선 위를 움직입니다.
 */
export function thresholdSweep(records, positiveIds, holdPolicy = 'escalate', steps = 19) {
  const out = [];
  for (let i = 1; i <= steps; i++) {
    const t = i / (steps + 1);
    out.push(binaryMetrics(records, positiveIds, t, holdPolicy));
  }
  return out;
}

/**
 * Youden's J (민감도 + 특이도 - 1)가 최대인 임계값.
 * "균형점"의 객체적 후보입니다. 임상 우선순위에 따라 여기서 옮기면 됩니다.
 */
export function bestByYouden(sweep) {
  let best = null, bestJ = -Infinity;
  for (const m of sweep) {
    if (m.sensitivity == null || m.specificity == null) continue;
    const j = m.sensitivity + m.specificity - 1;
    if (j > bestJ) { bestJ = j; best = m; }
  }
  return best ? { ...best, youden: bestJ } : null;
}

/**
 * 목표 민감도를 만족하는 가장 높은 임계값 (= 특이도를 최대한 지키면서 놓치지 않는 점).
 * "민감도 95% 이상을 확보하되 위양성을 최소화" 같은 임상 요구를 코드로 옮긴 것입니다.
 */
export function thresholdForSensitivity(sweep, target = 0.95) {
  const ok = sweep.filter(m => m.sensitivity != null && m.sensitivity >= target);
  if (!ok.length) return null;
  return ok.reduce((a, b) => (b.threshold > a.threshold ? b : a));
}

/** 캘리브레이션 표를 만들기 위한 (확률, 정답여부) 목록으로 변환 */
export function toCalibrationRecords(records) {
  return records.map(r => ({ prob: r.topProb, correct: r.topId === r.trueId }));
}

/** 가장 크게 틀린 사례들 — "실패 사례 갤러리"용. 확신하며 틀린 것이 가장 위험합니다. */
export function worstFailures(records, limit = 24) {
  return records
    .filter(r => r.topId !== r.trueId)
    .sort((a, b) => b.topProb - a.topProb)     // 확신도가 높은데 틀린 순
    .slice(0, limit);
}

/** 혼동행렬에서 가장 자주 헷갈리는 클래스 쌍 — 클래스 재설계의 근거가 됩니다. */
export function topConfusions(cm, limit = 5) {
  const { matrix, classIds } = cm;
  const pairs = [];
  for (let i = 0; i < classIds.length; i++) {
    for (let j = 0; j < classIds.length; j++) {
      if (i !== j && matrix[i][j] > 0) {
        pairs.push({ trueId: classIds[i], predId: classIds[j], count: matrix[i][j] });
      }
    }
  }
  return pairs.sort((a, b) => b.count - a.count).slice(0, limit);
}

export function fmtPct(v, digits = 1) {
  return v == null || !Number.isFinite(v) ? '—' : `${(v * 100).toFixed(digits)}%`;
}

/** 평가 결과 전체를 CSV로 — 발표 자료와 보고서에 붙일 원본 데이터 */
export function recordsToCsv(records, classIds) {
  const head = ['file', 'true', 'pred', 'top_prob', 'correct', ...classIds.map(c => `p_${c}`)];
  const rows = records.map(r => [
    csvCell(r.file ?? ''),
    csvCell(r.trueId),
    csvCell(r.topId),
    r.topProb.toFixed(6),
    r.topId === r.trueId ? '1' : '0',
    ...classIds.map(c => (r.probs[c] ?? 0).toFixed(6)),
  ].join(','));
  return [head.join(','), ...rows].join('\n');
}

function csvCell(s) {
  const v = String(s);
  return /[",\n]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v;
}
