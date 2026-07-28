/* saliency.js — 오클루전 기반 근거 히트맵 (설명가능 AI)
 * ---------------------------------------------------------------------------
 * TM은 Grad-CAM을 주지 않습니다. 하지만 오클루전 민감도 맵은 모델 내부에
 * 접근하지 않고도 만들 수 있습니다. 원리는 단순합니다.
 *
 *   이미지를 격자로 나눈다
 *   → 한 칸씩 가리고 다시 추론한다
 *   → 정답 클래스 확률이 크게 떨어진 칸 = 모델이 실제로 근거로 삼은 영역
 *
 * 8×8 = 64회 추론이면 브라우저에서 1~3초. 충분히 실용적입니다.
 *
 * 이 기능의 진짜 가치는 예쁜 히트맵이 아닙니다.
 * ★ 모델이 병변이 아니라 영상 구석의 글자·워터마크·촬영 장비 테두리를 보고
 *   있었다는 사실을 발견하는 순간입니다. (지름길 학습 / shortcut learning)
 *   그걸 찾아내 전처리로 제거하면 실제 성능이 올라가고,
 *   발표에서는 "우리는 모델이 무엇을 보는지 검증했다"는 카드가 됩니다.
 */

import { predictProbs } from './infer.js';

/**
 * @param {any} model            TM 모델 (앙상블이면 대표 1개)
 * @param {HTMLCanvasElement} src 224 정사각 캔버스 (infer.toSquareCanvas 결과)
 * @param {object} opts
 * @param {number} opts.grid      격자 한 변 (기본 8 → 64회 추론)
 * @param {string} opts.targetId  민감도를 측정할 클래스명 (보통 1위)
 * @param {number} opts.gray      가림 색 밝기 0~255
 * @param {(done:number,total:number)=>void} opts.onProgress
 * @param {{aborted:boolean}} opts.token 중간 취소용 (aborted=true로 바꾸면 중단)
 * @returns {Promise<{grid:number, values:Float32Array, base:number,
 *                    max:number, min:number, targetId:string, aborted:boolean}>}
 */
export async function occlusionMap(model, src, {
  grid = 8,
  targetId = null,
  gray = 128,
  onProgress = null,
  token = null,
} = {}) {
  const size = src.width;
  const cell = size / grid;

  // 작업 캔버스 — 매번 원본을 복원한 뒤 한 칸만 가립니다
  const work = document.createElement('canvas');
  work.width = size;
  work.height = size;
  const wctx = work.getContext('2d', { willReadFrequently: true });
  wctx.drawImage(src, 0, 0);
  const pristine = wctx.getImageData(0, 0, size, size);

  // 기준 확률 (가리지 않은 상태)
  const baseProbs = await predictProbs(model, work);
  const tid = targetId ?? Object.keys(baseProbs).sort((a, b) => baseProbs[b] - baseProbs[a])[0];
  const base = baseProbs[tid] ?? 0;

  const values = new Float32Array(grid * grid);
  const total = grid * grid;
  let done = 0;
  let aborted = false;

  for (let gy = 0; gy < grid; gy++) {
    for (let gx = 0; gx < grid; gx++) {
      if (token?.aborted) { aborted = true; break; }

      wctx.putImageData(pristine, 0, 0);
      wctx.fillStyle = `rgb(${gray},${gray},${gray})`;
      wctx.fillRect(Math.round(gx * cell), Math.round(gy * cell),
                    Math.ceil(cell), Math.ceil(cell));

      const p = await predictProbs(model, work);
      // 확률 하락폭. 음수(가렸는데 오히려 확률이 오른 경우)도 정보이므로 보존합니다.
      values[gy * grid + gx] = base - (p[tid] ?? 0);

      done++;
      onProgress?.(done, total);

      // 브라우저 프레임을 양보해 UI가 멈추지 않게 합니다
      if (done % grid === 0) await new Promise(r => requestAnimationFrame(r));
    }
    if (aborted) break;
  }

  let max = -Infinity, min = Infinity;
  for (let i = 0; i < values.length; i++) {
    if (values[i] > max) max = values[i];
    if (values[i] < min) min = values[i];
  }

  return { grid, values, base, max, min, targetId: tid, aborted };
}

/**
 * 히트맵을 원본 위에 반투명 오버레이로 그립니다.
 * 양의 하락폭(= 근거 영역)만 붉게 칠하고, 음수는 무시합니다.
 *
 * @param {HTMLCanvasElement} out 출력 캔버스 (원본 이미지를 먼저 그려 둘 것)
 * @param {object} map occlusionMap 결과
 * @param {number} alpha 최대 불투명도
 */
export function drawHeatmap(out, map, alpha = 0.62) {
  const ctx = out.getContext('2d');
  const { grid, values, max } = map;
  if (!(max > 0)) return;                    // 근거 영역이 없으면 그리지 않음

  const cw = out.width / grid;
  const ch = out.height / grid;

  // 부드럽게 보이도록 저해상도 히트맵을 만들어 확대 보간합니다
  const low = document.createElement('canvas');
  low.width = grid;
  low.height = grid;
  const lctx = low.getContext('2d');
  const img = lctx.createImageData(grid, grid);

  for (let i = 0; i < grid * grid; i++) {
    const t = Math.max(0, values[i]) / max;       // 0~1 정규화
    const [r, g, b] = heatColor(t);
    img.data[i * 4] = r;
    img.data[i * 4 + 1] = g;
    img.data[i * 4 + 2] = b;
    img.data[i * 4 + 3] = Math.round(255 * alpha * Math.pow(t, 1.4));  // 약한 신호는 더 투명
  }
  lctx.putImageData(img, 0, 0);

  ctx.save();
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(low, 0, 0, out.width, out.height);
  ctx.restore();

  // 가장 강한 칸에 테두리를 그려 "여기가 1순위 근거"를 명시합니다
  let bi = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[bi]) bi = i;
  const bx = (bi % grid) * cw;
  const by = Math.floor(bi / grid) * ch;
  ctx.save();
  ctx.strokeStyle = 'rgba(255,255,255,0.95)';
  ctx.lineWidth = 2;
  ctx.setLineDash([5, 3]);
  ctx.strokeRect(bx + 1, by + 1, cw - 2, ch - 2);
  ctx.restore();
}

/** 파랑(약) → 노랑 → 빨강(강) 컬러맵 */
function heatColor(t) {
  const c = Math.min(1, Math.max(0, t));
  if (c < 0.5) {
    const k = c / 0.5;                        // 파랑 → 노랑
    return [Math.round(40 + 215 * k), Math.round(90 + 130 * k), Math.round(200 - 160 * k)];
  }
  const k = (c - 0.5) / 0.5;                  // 노랑 → 빨강
  return [255, Math.round(220 - 160 * k), Math.round(40 - 20 * k)];
}

/**
 * 히트맵을 한 문장으로 해석합니다.
 * 근거가 중앙에 모였는지, 주변부(=지름길 학습 의심)에 있는지 판정합니다.
 */
export function interpretMap(map) {
  const { grid, values, max } = map;
  if (!(max > 0)) {
    return { tone: 'warn', text: '어느 영역을 가려도 예측이 변하지 않았습니다 — 모델이 특정 근거 없이 판정했을 가능성이 있습니다.' };
  }

  // 질량 중심과 주변부 비중 계산
  let sum = 0, cx = 0, cy = 0, edge = 0;
  for (let i = 0; i < values.length; i++) {
    const v = Math.max(0, values[i]);
    const x = i % grid, y = Math.floor(i / grid);
    sum += v; cx += v * x; cy += v * y;
    const isEdge = x === 0 || y === 0 || x === grid - 1 || y === grid - 1;
    if (isEdge) edge += v;
  }
  if (sum === 0) return { tone: 'warn', text: '유효한 근거 영역을 찾지 못했습니다.' };

  cx /= sum; cy /= sum;
  const mid = (grid - 1) / 2;
  const offset = Math.hypot(cx - mid, cy - mid) / mid;   // 0=정중앙, 1=코너
  const edgeRatio = edge / sum;

  if (edgeRatio > 0.55) {
    return {
      tone: 'danger',
      text: `근거의 ${Math.round(edgeRatio * 100)}%가 이미지 주변부에 있습니다. 모델이 병변이 아니라 배경·글자·장비 테두리를 보고 있을 가능성이 큽니다 — 전처리에서 ROI 크롭을 강화하세요.`,
    };
  }
  if (offset > 0.6) {
    return { tone: 'warn', text: '근거가 중앙에서 크게 벗어나 있습니다. 촬영 구도나 ROI 설정을 점검하세요.' };
  }
  return { tone: 'ok', text: '근거가 판독 대상 영역에 모여 있습니다. 모델이 의도한 부위를 보고 있습니다.' };
}
