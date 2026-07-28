/* temporal.js — 시간축 안정화
 * ---------------------------------------------------------------------------
 * 웹캠만 가진 자산은 "연속 프레임"입니다. 정지 이미지 1장에는 없습니다.
 *
 * 프레임 단위 예측은 심하게 흔들립니다 (같은 대상을 보고 있어도 60% ↔ 40%가
 * 왕복). 이 흔들림 자체가 정보입니다 — 모델이 확신하지 못한다는 뜻이니까요.
 *
 * 여기서 두 가지를 합니다.
 *   ① 이동평균으로 노이즈를 억제한다
 *   ② "연속 K프레임 동일 판정"을 만족할 때만 결과를 확정한다
 *
 * TM 모델을 재학습하지 않고 실질 정확도를 올리는 두 번째 레버입니다.
 * 주제와 무관한 순수 시계열 계층입니다.
 */

export class TemporalStabilizer {
  /**
   * @param {object} opts
   * @param {number} opts.window        이동평균 창 크기 (프레임 수)
   * @param {number} opts.confirmFrames 확정에 필요한 연속 동일 판정 수
   * @param {number} opts.holdThreshold 확정에 필요한 최소 유사도
   */
  constructor({ window = 8, confirmFrames = 5, holdThreshold = 0.7 } = {}) {
    this.window = Math.max(1, window);
    this.confirmFrames = Math.max(1, confirmFrames);
    this.holdThreshold = holdThreshold;
    this.reset();
  }

  reset() {
    this.buffer = [];        // 최근 확률 벡터들 [{id: p}, ...]
    this.streak = 0;         // 현재 topId가 연속 유지된 횟수
    this.streakId = null;
    this.confirmedId = null; // 확정된 클래스 (확정 후 유지)
    this.frames = 0;         // 누적 처리 프레임
  }

  /** 설정이 실행 중 바뀌면 호출합니다. 창 크기가 줄면 오래된 값을 버립니다. */
  configure({ window, confirmFrames, holdThreshold }) {
    if (window != null) this.window = Math.max(1, window);
    if (confirmFrames != null) this.confirmFrames = Math.max(1, confirmFrames);
    if (holdThreshold != null) this.holdThreshold = holdThreshold;
    if (this.buffer.length > this.window) {
      this.buffer = this.buffer.slice(-this.window);
    }
  }

  /**
   * 프레임 1개의 확률 벡터를 밀어 넣고, 안정화된 결과를 돌려줍니다.
   *
   * @param {Record<string, number>} probs {classId: probability}
   * @returns {{
   *   smoothed: Record<string, number>,  이동평균된 확률
   *   topId: string,                     평활 후 1위
   *   topProb: number,
   *   secondId: string|null,
   *   margin: number,                    1위 - 2위
   *   streak: number,                    연속 동일 판정 수
   *   progress: number,                  확정까지 진행률 0~1
   *   confirmed: boolean,                이번 프레임에 확정 조건을 만족했는가
   *   justConfirmed: boolean,            확정이 "새로" 성립한 순간인가
   *   jitter: number,                    창 내 1위 확률의 표준편차 (흔들림)
   *   filled: boolean                    창이 다 찼는가
   * }}
   */
  push(probs) {
    this.frames++;
    this.buffer.push(probs);
    if (this.buffer.length > this.window) this.buffer.shift();

    // ── ① 이동평균 ──
    const smoothed = {};
    const ids = Object.keys(probs);
    for (const id of ids) {
      let sum = 0;
      for (const f of this.buffer) sum += (f[id] ?? 0);
      smoothed[id] = sum / this.buffer.length;
    }

    const ranked = ids.slice().sort((a, b) => smoothed[b] - smoothed[a]);
    const topId = ranked[0];
    const secondId = ranked[1] ?? null;
    const topProb = smoothed[topId] ?? 0;
    const margin = secondId ? topProb - smoothed[secondId] : topProb;

    // ── ② 연속 판정 카운트 ──
    // 임계값을 넘은 상태로 같은 클래스가 유지될 때만 streak가 자랍니다.
    const passes = topProb >= this.holdThreshold;
    if (passes && topId === this.streakId) {
      this.streak++;
    } else if (passes) {
      this.streakId = topId;
      this.streak = 1;
    } else {
      this.streakId = null;
      this.streak = 0;
    }

    const confirmed = this.streak >= this.confirmFrames;
    const justConfirmed = confirmed && this.confirmedId !== topId;
    if (confirmed) this.confirmedId = topId;
    if (!passes) this.confirmedId = null;

    // ── 흔들림 지표 (창 내 1위 클래스 확률의 표준편차) ──
    let jitter = 0;
    if (this.buffer.length > 1) {
      const series = this.buffer.map(f => f[topId] ?? 0);
      const mean = series.reduce((a, b) => a + b, 0) / series.length;
      const varr = series.reduce((a, b) => a + (b - mean) ** 2, 0) / series.length;
      jitter = Math.sqrt(varr);
    }

    return {
      smoothed, topId, topProb, secondId, margin,
      streak: this.streak,
      progress: Math.min(1, this.streak / this.confirmFrames),
      confirmed, justConfirmed, jitter,
      filled: this.buffer.length >= this.window,
    };
  }
}

/**
 * 예측 벡터의 정규화된 엔트로피 (0=확신, 1=완전 무작위).
 * 임계값 하나로 잡히지 않는 "애매함"을 잡아내는 보조 지표입니다.
 * 클래스 수가 다른 프로젝트끼리도 비교 가능하도록 log(N)으로 정규화합니다.
 */
export function normalizedEntropy(probs) {
  const vals = Object.values(probs).filter(v => v > 0);
  if (vals.length <= 1) return 0;
  const h = -vals.reduce((a, p) => a + p * Math.log(p), 0);
  return h / Math.log(vals.length);
}
