/* quality.js — 프레임 품질 게이트
 * ---------------------------------------------------------------------------
 * TM 모델을 전혀 건드리지 않고 실질 정확도를 올리는 가장 값싼 방법:
 * "나쁜 프레임은 애초에 추론하지 않는다".
 *
 * 흐린 프레임, 너무 어둡거나 날아간 프레임을 모델에 먹이면 모델은 그래도
 * 무언가를 답합니다(softmax는 항상 합이 1). 그 답은 노이즈입니다.
 * 여기서 미리 걸러내고, 대신 사용자에게 촬영을 코칭합니다.
 *
 * 주제와 무관한 순수 신호처리 계층입니다.
 */

const WORK = 224;               // 분석 해상도 (TM 입력과 동일하게 맞춤)

let canvas = null;
let ctx = null;

function ensureCanvas() {
  if (ctx) return;
  canvas = document.createElement('canvas');
  canvas.width = WORK;
  canvas.height = WORK;
  ctx = canvas.getContext('2d', { willReadFrequently: true });
}

/** source의 실제 픽셀 크기를 구합니다 (video/img/canvas 공통). */
export function sourceSize(source) {
  const w = source.videoWidth || source.naturalWidth || source.width || 0;
  const h = source.videoHeight || source.naturalHeight || source.height || 0;
  return { w, h };
}

/**
 * 프레임의 품질 지표를 계산합니다.
 * 중앙 정사각 ROI만 분석합니다 — 가이드 박스 안쪽이 실제 판독 대상이고,
 * 주변 배경 때문에 지표가 오염되는 것을 막습니다.
 *
 * @param {HTMLVideoElement|HTMLImageElement|HTMLCanvasElement} source
 * @param {number} roi 중앙에서 취할 비율 (0~1)
 * @returns {{blur:number, luma:number, contrast:number, saturation:number,
 *            clipHigh:number, clipLow:number}|null}
 */
export function measureFrame(source, roi = 0.75) {
  ensureCanvas();
  const { w: sw, h: sh } = sourceSize(source);
  if (!sw || !sh) return null;               // 비디오 메타데이터 도착 전

  const side = Math.min(sw, sh) * roi;
  const sx = (sw - side) / 2;
  const sy = (sh - side) / 2;

  try {
    ctx.drawImage(source, sx, sy, side, side, 0, 0, WORK, WORK);
  } catch {
    return null;                             // 아직 그릴 수 없는 상태
  }

  const data = ctx.getImageData(0, 0, WORK, WORK).data;
  const n = WORK * WORK;
  const gray = new Float32Array(n);

  let lumaSum = 0, lumaSq = 0, satSum = 0, clipHigh = 0, clipLow = 0;

  for (let i = 0, p = 0; i < n; i++, p += 4) {
    const r = data[p], g = data[p + 1], b = data[p + 2];
    // ITU-R BT.601 휘도
    const y = 0.299 * r + 0.587 * g + 0.114 * b;
    gray[i] = y;
    lumaSum += y;
    lumaSq += y * y;

    const mx = Math.max(r, g, b);
    const mn = Math.min(r, g, b);
    satSum += mx === 0 ? 0 : (mx - mn) / mx;

    if (mx >= 250) clipHigh++;
    if (mx <= 8) clipLow++;
  }

  const luma = lumaSum / n;
  const contrast = Math.sqrt(Math.max(0, lumaSq / n - luma * luma));

  // 라플라시안 분산 = 초점 선명도. 3x3 커널 [0,1,0 / 1,-4,1 / 0,1,0]
  let lapSum = 0, lapSq = 0, lapN = 0;
  for (let y = 1; y < WORK - 1; y++) {
    const row = y * WORK;
    for (let x = 1; x < WORK - 1; x++) {
      const i = row + x;
      const lap = gray[i - WORK] + gray[i + WORK] + gray[i - 1] + gray[i + 1] - 4 * gray[i];
      lapSum += lap;
      lapSq += lap * lap;
      lapN++;
    }
  }
  const lapMean = lapSum / lapN;
  const blur = lapSq / lapN - lapMean * lapMean;   // 분산이 낮으면 흐림

  return {
    blur,
    luma,
    contrast,
    saturation: satSum / n,
    clipHigh: clipHigh / n,
    clipLow: clipLow / n,
  };
}

/**
 * 지표를 정책과 비교해 통과 여부와 코칭 문구를 만듭니다.
 * 가장 심각한 문제 하나만 코칭합니다 — 동시에 여러 지시를 하면 사용자가 못 따릅니다.
 *
 * @param {object} m measureFrame 결과
 * @param {object} s 설정 {enabled, blurMin, lumaMin, lumaMax, saturationMin}
 * @returns {{ok:boolean, coach:string, issues:Array<{code,severity,message}>}}
 */
export function evaluateQuality(m, s) {
  if (!s.enabled) return { ok: true, coach: '', issues: [] };
  if (!m) return { ok: false, coach: '카메라 준비 중입니다', issues: [{ code: 'nosignal', severity: 2, message: '신호 없음' }] };

  const issues = [];

  if (m.clipHigh > 0.25) {
    issues.push({ code: 'overexposed', severity: 3, message: '빛이 너무 강합니다 — 조명이나 반사를 피해 주세요' });
  }
  if (m.luma < s.lumaMin) {
    issues.push({ code: 'dark', severity: 3, message: '너무 어둡습니다 — 조금 더 밝은 곳에서 촬영해 주세요' });
  } else if (m.luma > s.lumaMax) {
    issues.push({ code: 'bright', severity: 3, message: '너무 밝습니다 — 조명을 낮추거나 각도를 바꿔 주세요' });
  }
  if (m.blur < s.blurMin) {
    // 어두우면 노이즈가 줄어 blur 지표도 같이 떨어지므로, 어둠 문제가 있으면 우선순위를 낮춤
    const sev = m.luma < s.lumaMin ? 1 : 3;
    issues.push({ code: 'blurry', severity: sev, message: '초점이 맞지 않습니다 — 카메라를 고정하고 거리를 조절해 주세요' });
  }
  if (m.contrast < 12) {
    issues.push({ code: 'flat', severity: 2, message: '대상이 화면에 없거나 대비가 너무 낮습니다' });
  }
  if (s.saturationMin > 0 && m.saturation < s.saturationMin) {
    issues.push({ code: 'desaturated', severity: 2, message: '색이 흐릿합니다 — 조명 색과 화이트밸런스를 확인해 주세요' });
  }

  issues.sort((a, b) => b.severity - a.severity);
  const blocking = issues.filter(i => i.severity >= 3);

  return {
    ok: blocking.length === 0,
    coach: blocking.length ? blocking[0].message : (issues[0]?.message ?? ''),
    issues,
  };
}

/** 지표를 0~100 점수로 요약 (UI 게이지용). 정확한 의미는 없고 상대 비교용입니다. */
export function qualityScore(m, s) {
  if (!m) return 0;
  const focus = Math.min(1, m.blur / (s.blurMin * 2));
  const mid = (s.lumaMin + s.lumaMax) / 2;
  const expo = Math.max(0, 1 - Math.abs(m.luma - mid) / mid);
  const cont = Math.min(1, m.contrast / 45);
  const clip = Math.max(0, 1 - (m.clipHigh + m.clipLow) * 3);
  return Math.round((focus * 0.4 + expo * 0.25 + cont * 0.2 + clip * 0.15) * 100);
}
