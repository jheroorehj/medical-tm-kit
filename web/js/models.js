/* models.js — 다중 TM 모델 레지스트리
 * ---------------------------------------------------------------------------
 * TM은 모델 URL을 몇 개든 만들 수 있습니다. 대부분의 팀은 1개만 씁니다.
 * 여기서는 여러 개를 동시에 로드해 서비스 레이어에서 파이프라인을 구성합니다.
 *
 *   [프레임] → ① 게이트키퍼("판독 대상인가?") → ② 주분류기 (+앙상블 소프트보팅)
 *
 * 게이트키퍼는 softmax가 억지 답을 내는 문제(학습한 어느 클래스도 아닌 입력에
 * 대해서도 반드시 1등을 뽑는 문제)를 구조로 막는 장치입니다.
 *
 * 또 하나 중요한 역할: TM에서 만든 클래스명과 project.config.js의 클래스 id가
 * 어긋났는지 로드 시점에 검사합니다. 실습에서 오타 하나로 몇 시간 날리는 일이
 * 흔한데, 그걸 부팅 즉시 잡아 줍니다.
 */

/** TM 모델 URL을 정규화합니다. 사용자는 트레일링 슬래시를 자주 빼먹습니다. */
export function normalizeModelUrl(raw) {
  const url = (raw || '').trim();
  if (!url) return '';
  if (url.includes('xxxxxx')) return '';          // 플레이스홀더 그대로 제출 방어
  // 이미 model.json을 직접 가리키는 경우도 허용
  if (url.endsWith('model.json')) return url.slice(0, -'model.json'.length);
  return url.endsWith('/') ? url : url + '/';
}

export const SLOT = {
  GATE: 'gatekeeper',
  PRIMARY: 'primary',
  ens: i => `ensemble:${i}`,
};

export class ModelRegistry {
  constructor() {
    /** @type {Map<string, {model:any, url:string, labels:string[], label:string, weight:number}>} */
    this.entries = new Map();
    /** @type {Map<string, {state:'idle'|'loading'|'ok'|'error', message:string}>} */
    this.states = new Map();
    this.listeners = new Set();
  }

  onChange(fn) { this.listeners.add(fn); return () => this.listeners.delete(fn); }
  #emit() { this.listeners.forEach(fn => fn(this)); }

  status(slot) {
    return this.states.get(slot) || { state: 'idle', message: '미로드' };
  }

  get(slot) { return this.entries.get(slot) || null; }
  has(slot) { return this.status(slot).state === 'ok'; }

  /**
   * 모델 1개를 로드합니다.
   * @param {string} slot   SLOT 상수
   * @param {string} rawUrl TM 모델 URL
   * @param {object} meta   {label, weight}
   * @returns {Promise<boolean>} 성공 여부
   */
  async load(slot, rawUrl, meta = {}) {
    const base = normalizeModelUrl(rawUrl);
    if (!base) {
      this.states.set(slot, { state: 'error', message: '유효한 모델 URL이 아닙니다' });
      this.#emit();
      return false;
    }
    if (typeof tmImage === 'undefined') {
      this.states.set(slot, { state: 'error', message: 'Teachable Machine 라이브러리가 로드되지 않았습니다 (네트워크 확인)' });
      this.#emit();
      return false;
    }

    this.states.set(slot, { state: 'loading', message: '불러오는 중…' });
    this.#emit();

    try {
      const model = await tmImage.load(base + 'model.json', base + 'metadata.json');
      const labels = typeof model.getClassLabels === 'function'
        ? model.getClassLabels()
        : Array.from({ length: model.getTotalClasses() }, (_, i) => `class_${i}`);

      this.entries.set(slot, {
        model, url: base, labels,
        label: meta.label || slot,
        weight: meta.weight ?? 1.0,
      });
      this.states.set(slot, { state: 'ok', message: `클래스 ${labels.length}개` });
      this.#emit();
      return true;
    } catch (e) {
      this.entries.delete(slot);
      this.states.set(slot, {
        state: 'error',
        message: 'URL을 확인하세요 (CORS·오타·비공개 모델일 수 있습니다)',
      });
      console.error(`[models] ${slot} 로드 실패:`, e);
      this.#emit();
      return false;
    }
  }

  unload(slot) {
    this.entries.delete(slot);
    this.states.set(slot, { state: 'idle', message: '미로드' });
    this.#emit();
  }

  /** 주분류기 + 앙상블 슬롯 목록 (게이트키퍼 제외) */
  classifierSlots() {
    return [...this.entries.keys()].filter(k => k !== SLOT.GATE);
  }

  /**
   * config에 선언한 클래스 id와 실제 모델의 클래스명이 맞는지 검사합니다.
   * @param {string[]} configIds
   * @returns {{ok:boolean, missing:string[], extra:string[], modelLabels:string[]}}
   */
  verifyClasses(configIds, slot = SLOT.PRIMARY) {
    const entry = this.get(slot);
    if (!entry) return { ok: false, missing: configIds, extra: [], modelLabels: [] };
    const model = new Set(entry.labels);
    const cfg = new Set(configIds);
    return {
      ok: configIds.every(id => model.has(id)) && entry.labels.every(l => cfg.has(l)),
      missing: configIds.filter(id => !model.has(id)),   // config엔 있고 모델엔 없음
      extra: entry.labels.filter(l => !cfg.has(l)),      // 모델엔 있고 config엔 없음
      modelLabels: entry.labels,
    };
  }

  /** 앙상블 구성원끼리 클래스 집합이 같은지 확인 (다르면 보팅이 무의미) */
  verifyEnsembleConsistency() {
    const slots = this.classifierSlots();
    if (slots.length < 2) return { ok: true, mismatched: [] };
    const ref = new Set(this.get(slots[0]).labels);
    const mismatched = slots.slice(1).filter(s => {
      const l = this.get(s).labels;
      return l.length !== ref.size || !l.every(x => ref.has(x));
    });
    return { ok: mismatched.length === 0, mismatched };
  }
}
