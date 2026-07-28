/* holdout.js — 홀드아웃 셋 배치 평가
 * ---------------------------------------------------------------------------
 * prep/pipeline.py가 원본 단위로 떼어내 TM에 넣지 않은 홀드아웃 폴더를
 * 여기에 드래그하면, 전체를 자동 추론해 혼동행렬과 임상 지표를 계산합니다.
 *
 * 기대하는 폴더 구조 (폴더 이름 = 정답 클래스명):
 *
 *   holdout/
 *   ├── normal/     ← project.config.js의 클래스 id와 일치해야 합니다
 *   │   ├── 0001.png
 *   │   └── 0002.png
 *   └── abnormal/
 *       └── 0003.png
 *
 * 이 기능이 있으면 "TM이 말한 정확도"가 아니라 "우리가 검증한 정확도"를
 * 발표할 수 있습니다. 심사에서 이 차이는 큽니다.
 */

import { runClassify, toSquareCanvas } from './infer.js';

const IMAGE_RE = /\.(jpe?g|png|webp|bmp|gif)$/i;

/**
 * 드롭된 항목에서 (파일, 폴더명) 목록을 재귀 수집합니다.
 * 폴더 드롭은 webkitGetAsEntry가 필요하고, 이 API는 drop 이벤트 직후에만
 * 유효하므로 entry 목록을 먼저 확보한 뒤 순회합니다.
 */
export async function collectFromDrop(dataTransfer) {
  const entries = [];
  for (const item of dataTransfer.items) {
    const entry = item.webkitGetAsEntry?.();
    if (entry) entries.push(entry);
  }

  // 폴더 지원이 없는 브라우저 — 평평한 파일 목록으로 폴백
  if (!entries.length) {
    return [...dataTransfer.files]
      .filter(f => IMAGE_RE.test(f.name))
      .map(f => ({ file: f, label: '(미분류)', path: f.name }));
  }

  const out = [];
  for (const entry of entries) await walk(entry, '', out);
  return out;
}

function walk(entry, prefix, out) {
  return new Promise(resolve => {
    if (entry.isFile) {
      if (!IMAGE_RE.test(entry.name)) return resolve();
      entry.file(file => {
        // 정답 라벨 = 파일이 들어 있는 바로 위 폴더명
        const parts = prefix.split('/').filter(Boolean);
        out.push({
          file,
          label: parts.length ? parts[parts.length - 1] : '(미분류)',
          path: prefix + entry.name,
        });
        resolve();
      }, () => resolve());
      return;
    }
    if (entry.isDirectory) {
      const reader = entry.createReader();
      const all = [];
      const readMore = () => {
        // readEntries는 한 번에 최대 100개만 주므로 빌 때까지 반복해야 합니다
        reader.readEntries(async batch => {
          if (!batch.length) {
            for (const e of all) await walk(e, `${prefix}${entry.name}/`, out);
            return resolve();
          }
          all.push(...batch);
          readMore();
        }, () => resolve());
      };
      readMore();
      return;
    }
    resolve();
  });
}

/** <input type="file" webkitdirectory> 로 선택한 경우 */
export function collectFromInput(fileList) {
  return [...fileList]
    .filter(f => IMAGE_RE.test(f.name))
    .map(f => {
      const rel = f.webkitRelativePath || f.name;
      const parts = rel.split('/');
      return {
        file: f,
        label: parts.length >= 2 ? parts[parts.length - 2] : '(미분류)',
        path: rel,
      };
    });
}

/** File을 <img>로 디코드합니다. 실패해도 배치 전체가 멈추지 않게 null을 돌려줍니다. */
function loadImage(file) {
  return new Promise(resolve => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => { URL.revokeObjectURL(url); resolve(img); };
    img.onerror = () => { URL.revokeObjectURL(url); resolve(null); };
    img.src = url;
  });
}

/**
 * 배치 추론을 실행합니다.
 *
 * @param {import('./models.js').ModelRegistry} registry
 * @param {Array<{file:File,label:string,path:string}>} items
 * @param {object} opts
 * @param {number} opts.roi              추론 시 중앙 ROI 비율 (학습 전처리와 맞출 것)
 * @param {(done:number,total:number,path:string)=>void} opts.onProgress
 * @param {{aborted:boolean}} opts.token
 * @returns {Promise<{records:Array, failed:number, aborted:boolean, files:Map<string,File>}>}
 */
export async function runBatch(registry, items, { roi = 1.0, onProgress, token } = {}) {
  const records = [];
  const files = new Map();          // path → File (실패 사례 썸네일 렌더용, 저장하지 않음)
  const canvas = document.createElement('canvas');
  let failed = 0;
  let aborted = false;

  for (let i = 0; i < items.length; i++) {
    if (token?.aborted) { aborted = true; break; }
    const it = items[i];

    const img = await loadImage(it.file);
    if (!img) { failed++; onProgress?.(i + 1, items.length, it.path); continue; }

    try {
      toSquareCanvas(img, roi, 224, canvas);
      const { probs } = await runClassify(registry, canvas);
      const ids = Object.keys(probs);
      const topId = ids.reduce((a, b) => (probs[b] > probs[a] ? b : a), ids[0]);
      records.push({
        trueId: it.label,
        topId,
        topProb: probs[topId],
        probs,
        file: it.path,
      });
      files.set(it.path, it.file);
    } catch (e) {
      failed++;
      console.warn('[holdout] 추론 실패:', it.path, e);
    }

    onProgress?.(i + 1, items.length, it.path);
    // 20장마다 UI에 프레임을 양보 — 수백 장을 돌려도 화면이 얼지 않습니다
    if (i % 20 === 19) await new Promise(r => requestAnimationFrame(r));
  }

  return { records, failed, aborted, files };
}

/**
 * 폴더명이 config의 클래스 id와 맞는지 검사합니다.
 * 실습에서 가장 흔한 사고가 여기서 납니다 (폴더명 'Normal' vs 클래스 'normal').
 */
export function verifyLabels(items, classIds) {
  const found = new Set(items.map(i => i.label));
  const known = new Set(classIds);
  return {
    unknown: [...found].filter(l => !known.has(l)),   // 폴더는 있는데 config에 없음
    missing: classIds.filter(id => !found.has(id)),   // config엔 있는데 폴더가 없음
    counts: [...found].map(l => ({ label: l, n: items.filter(i => i.label === l).length })),
  };
}
