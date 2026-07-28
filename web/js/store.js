/* store.js — 설정·이력·캘리브레이션의 영속 저장 (localStorage)
 * ---------------------------------------------------------------------------
 * 주제를 몰라도 되는 순수 저장 계층. 키 접두사만 프로젝트별로 갈립니다.
 * 서버가 없으므로 모든 상태가 브라우저에 남습니다 — 영상 자체는 저장하지 않고
 * 결과 메타데이터(썸네일 dataURL 포함)만 저장합니다.
 */

const NS = 'mtk';                    // medical-tm-kit
const K = {
  settings:    `${NS}.settings`,
  modelUrls:   `${NS}.modelUrls`,
  history:     `${NS}.history`,
  calibration: `${NS}.calibration`,
  holdout:     `${NS}.holdout`,
  checklist:   `${NS}.checklist`,
};

const HISTORY_LIMIT = 300;

function read(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    return raw === null ? fallback : JSON.parse(raw);
  } catch {
    return fallback;
  }
}

function write(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
    return true;
  } catch (e) {
    // 용량 초과(QuotaExceededError)가 가장 흔한 원인 — 이력을 절반 버리고 재시도
    if (key === K.history) {
      const half = read(K.history, []).slice(0, Math.floor(HISTORY_LIMIT / 2));
      try {
        localStorage.setItem(key, JSON.stringify(half.concat(value).slice(-HISTORY_LIMIT)));
        return true;
      } catch { /* 포기 */ }
    }
    console.warn('[store] 저장 실패:', key, e);
    return false;
  }
}

/* ── 판정 정책 설정 ───────────────────────────────────────────────────── */

/** project.config의 decision/quality 초기값 위에 저장된 사용자 조정값을 덮어씁니다. */
export function loadSettings(defaults) {
  const saved = read(K.settings, {});
  return {
    ...defaults.decision,
    ...defaults.quality,
    // 저장된 값이 있으면 그것을 우선
    ...saved,
  };
}

export function saveSettings(settings) {
  return write(K.settings, settings);
}

export function resetSettings() {
  localStorage.removeItem(K.settings);
}

/* ── 모델 URL ─────────────────────────────────────────────────────────── */

/** config에 URL을 비워 뒀을 때 사용자가 입력한 값을 기억합니다. */
export function loadModelUrls() {
  return read(K.modelUrls, { gatekeeper: '', primary: '', ensemble: [] });
}

export function saveModelUrls(urls) {
  return write(K.modelUrls, urls);
}

/* ── 판독 이력 ────────────────────────────────────────────────────────── */

export function loadHistory() {
  return read(K.history, []);
}

/**
 * 이력 1건 추가. 최신이 배열 앞쪽입니다.
 * @param {object} entry {ts, source, topId, topProb, calibrated, decision, probs, thumb, quality}
 */
export function pushHistory(entry) {
  const list = read(K.history, []);
  list.unshift(entry);
  write(K.history, list.slice(0, HISTORY_LIMIT));
  return list.length;
}

export function clearHistory() {
  localStorage.removeItem(K.history);
}

/* ── 신뢰도 캘리브레이션 테이블 ───────────────────────────────────────── *
 * 홀드아웃 평가에서 산출한 "신뢰도 구간별 실측 정확도" 표입니다.
 * softmax 확률은 과신하는 경향이 있으므로, 표시용 신뢰도를 여기로 보정합니다.
 *   bins: [{ lo, hi, n, correct, accuracy }]
 * ------------------------------------------------------------------------ */

export function loadCalibration() {
  return read(K.calibration, null);
}

export function saveCalibration(table) {
  return write(K.calibration, table);
}

export function clearCalibration() {
  localStorage.removeItem(K.calibration);
}

/* ── 홀드아웃 평가 결과 ───────────────────────────────────────────────── */

export function loadHoldout() {
  return read(K.holdout, null);
}

export function saveHoldout(result) {
  return write(K.holdout, result);
}

export function clearHoldout() {
  localStorage.removeItem(K.holdout);
}

/* ── 도움말 체크리스트 ───────────────────────────────────────────────── *
 * 처음 시작하는 사용자가 어디까지 진행했는지 기억합니다.
 * 새로고침해도 남아야 의미가 있으므로 localStorage에 둡니다.
 * ------------------------------------------------------------------------ */

export function loadChecklist() {
  return read(K.checklist, {});
}

export function saveChecklist(map) {
  return write(K.checklist, map);
}

/* ── 전체 초기화 ──────────────────────────────────────────────────────── */

export function wipeAll() {
  Object.values(K).forEach(k => localStorage.removeItem(k));
}

/** 저장 용량 사용량 추정 (설정 탭에 표시) */
export function storageUsage() {
  let bytes = 0;
  for (const k of Object.values(K)) {
    const v = localStorage.getItem(k);
    if (v) bytes += k.length + v.length;
  }
  return bytes;
}
