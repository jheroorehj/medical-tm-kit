/* 순수 로직 검증 — 손으로 계산한 값과 대조합니다 */
const W = new URL('./web', import.meta.url).pathname;

const { PROJECT, POSITIVE_IDS, CLASS_IDS, INVALID_IDS } = await import(`${W}/project.config.js`);
const { TemporalStabilizer, normalizedEntropy } = await import(`${W}/js/temporal.js`);
const { decide, STATUS } = await import(`${W}/js/infer.js`);
const M = await import(`${W}/js/metrics.js`);
const C = await import(`${W}/js/calibrate.js`);

let pass = 0, fail = 0;
const near = (a, b, e = 1e-6) => Math.abs(a - b) < e;
function ok(name, cond, got) {
  if (cond) { pass++; console.log(`  ✓ ${name}`); }
  else { fail++; console.log(`  ✗ ${name}  → 실제값: ${JSON.stringify(got)}`); }
}

console.log('\n── project.config 파생값 ──');
ok('클래스 4개', CLASS_IDS.length === 4, CLASS_IDS);
ok('positive = abnormal, borderline', POSITIVE_IDS.join() === 'abnormal,borderline', POSITIVE_IDS);
ok('invalid = invalid', INVALID_IDS.join() === 'invalid', INVALID_IDS);
ok('inferRoi 존재', typeof PROJECT.decision.inferRoi === 'number', PROJECT.decision.inferRoi);

console.log('\n── 판정 4단 관문 (infer.decide) ──');
const S = { holdThreshold: 0.70, marginThreshold: 0.15, useCalibration: false };
const ctx = { copy: PROJECT.copy };

let d = decide({ normal: .05, borderline: .05, abnormal: .05, invalid: .85 }, S, ctx);
ok('관문1 무효 클래스 1위 → INVALID', d.status === STATUS.INVALID, d.status);
ok('  무효는 신뢰 불가 표시', d.trustworthy === false, d.trustworthy);

d = decide({ normal: .40, borderline: .30, abnormal: .20, invalid: .10 }, S, ctx);
ok('관문2 임계값 미달 → HOLD', d.status === STATUS.HOLD, d.status);

d = decide({ normal: .10, borderline: .05, abnormal: .75, invalid: .10 }, S, ctx);
ok('관문3 통과 (마진 0.65) → OK', d.status === STATUS.OK, d.status);

// ★ softmax 합=1 이므로 마진 = 1위-2위 >= 2*1위-1.
//   1위가 0.70이면 마진은 최소 0.40 → 기본 임계값에서 관문3은 도달 불가.
//   임계값을 낮춰야(민감도 우선 운영점) 활성화됩니다.
const Slow = { ...S, holdThreshold: 0.45 };
d = decide({ normal: .00, borderline: .44, abnormal: .56, invalid: .00 }, Slow, ctx);
ok('관문3 마진 0.12 < 0.15 → AMBIGUOUS (임계값 0.45)', d.status === STATUS.AMBIGUOUS, d.status);
d = decide({ normal: .00, borderline: .44, abnormal: .56, invalid: .00 }, S, ctx);
ok('  같은 입력이 임계값 0.70에서는 HOLD (관문2가 먼저)', d.status === STATUS.HOLD, d.status);
ok('  마진 관문 도달 가능 조건: 2*hold-1 < margin',
   (2 * 0.45 - 1 < 0.15) && !(2 * 0.70 - 1 < 0.15));

d = decide({ normal: .90, borderline: .05, abnormal: .03, invalid: .02 }, S, ctx);
ok('정상 판정 tone=ok', d.status === STATUS.OK && d.tone === 'ok', [d.status, d.tone]);

d = decide({ normal: .05, borderline: .03, abnormal: .90, invalid: .02 }, S, ctx);
ok('이상 판정 tone=danger', d.tone === 'danger', d.tone);

d = decide({ normal: .10, borderline: .05, abnormal: .80, invalid: .05 }, S,
           { ...ctx, gate: { skipped: false, pass: false } });
ok('관문0 게이트키퍼 차단 → GATED', d.status === STATUS.GATED, d.status);

d = decide({ normal: .10, borderline: .05, abnormal: .80, invalid: .05 }, S,
           { ...ctx, gate: { skipped: true, pass: true } });
ok('게이트 미사용 시 통과', d.status === STATUS.OK, d.status);

console.log('\n── 엔트로피 ──');
ok('균등분포 → 1.0', near(normalizedEntropy({ a: .25, b: .25, c: .25, d: .25 }), 1, 1e-9));
ok('완전 확신 → 0.0', near(normalizedEntropy({ a: 1 }), 0));

console.log('\n── 시간축 안정화 (TemporalStabilizer) ──');
const st = new TemporalStabilizer({ window: 4, confirmFrames: 3, holdThreshold: 0.7 });
const hi = { abnormal: .9, normal: .1 };
let r1 = st.push(hi), r2 = st.push(hi), r3 = st.push(hi);
ok('연속 3프레임 → 확정', r3.confirmed === true, r3.streak);
ok('  1·2프레임은 미확정', !r1.confirmed && !r2.confirmed, [r1.streak, r2.streak]);
ok('  justConfirmed는 확정 순간에만', r3.justConfirmed === true, r3.justConfirmed);
ok('  진행률 1.0', near(r3.progress, 1), r3.progress);

// 한 프레임의 이상치로는 판정이 뒤집히지 않아야 합니다 — 이동평균의 관성이 목적입니다
const r4 = st.push({ abnormal: .2, normal: .8 });
ok('단발 이상치에 흔들리지 않음 (평활 .725)', r4.confirmed === true && near(r4.topProb, .725), r4.topProb);
// 신호가 지속적으로 바뀌면 반드시 리셋되어야 합니다
const r5 = st.push({ abnormal: .1, normal: .9 });
const r6 = st.push({ abnormal: .1, normal: .9 });
ok('지속적 변화에는 streak 리셋', r6.streak === 0 || r6.topId === 'normal', [r6.streak, r6.topId]);
const r7 = st.push({ abnormal: .1, normal: .9 });
ok('  판정이 normal로 전환', r7.topId === 'normal', [r7.topId, r7.smoothed]);
ok('  jitter가 변동을 포착', r7.jitter > 0, r7.jitter);

const st2 = new TemporalStabilizer({ window: 2, confirmFrames: 1, holdThreshold: 0.5 });
st2.push({ a: 1.0, b: 0.0 });
const avg = st2.push({ a: 0.0, b: 1.0 });
ok('창 2의 이동평균 = 0.5/0.5', near(avg.smoothed.a, .5) && near(avg.smoothed.b, .5), avg.smoothed);

console.log('\n── 임상 지표 (metrics) ── [손계산 대조]');
const recs = [
  { trueId: 'normal',   topId: 'normal',   topProb: .9, probs: {} },  // TN
  { trueId: 'normal',   topId: 'abnormal', topProb: .9, probs: {} },  // FP
  { trueId: 'abnormal', topId: 'abnormal', topProb: .9, probs: {} },  // TP
  { trueId: 'abnormal', topId: 'normal',   topProb: .9, probs: {} },  // FN
  { trueId: 'borderline',  topId: 'abnormal', topProb: .9, probs: {} },  // TP (둘 다 positive)
  { trueId: 'normal',   topId: 'normal',   topProb: .3, probs: {} },  // 보류
];
const cm = M.confusionMatrix(recs, CLASS_IDS);
ok('정확도 3/6 = 0.5', near(M.overallAccuracy(cm), 0.5), M.overallAccuracy(cm));

const esc = M.binaryMetrics(recs, POSITIVE_IDS, 0.5, 'escalate');
ok('escalate: TP=2', esc.tp === 2, esc.tp);
ok('escalate: FN=1', esc.fn === 1, esc.fn);
ok('escalate: FP=2 (직접1 + 보류1)', esc.fp === 2, esc.fp);
ok('escalate: TN=1', esc.tn === 1, esc.tn);
ok('escalate: 민감도 2/3', near(esc.sensitivity, 2 / 3), esc.sensitivity);
ok('escalate: 특이도 1/3', near(esc.specificity, 1 / 3), esc.specificity);
ok('escalate: 보류율 1/6', near(esc.holdRate, 1 / 6), esc.holdRate);

const exc = M.binaryMetrics(recs, POSITIVE_IDS, 0.5, 'exclude');
ok('exclude: 보류 제외되어 특이도 1/2', near(exc.specificity, 0.5), exc.specificity);
ok('exclude: 평가 대상 5건', exc.evaluated === 5, exc.evaluated);
ok('exclude: 민감도는 동일 2/3', near(exc.sensitivity, 2 / 3), exc.sensitivity);

const sweep = M.thresholdSweep(recs, POSITIVE_IDS, 'escalate');
ok('스윕 19개 지점', sweep.length === 19, sweep.length);
ok('임계값 오름차순', sweep.every((m, i) => i === 0 || m.threshold > sweep[i - 1].threshold));
ok('임계값 최대 → 전부 보류 → 민감도 1.0',
   near(sweep[sweep.length - 1].sensitivity, 1), sweep[sweep.length - 1].sensitivity);
ok('Youden 균형점 존재', M.bestByYouden(sweep) !== null);
ok('민감도 95% 확보점 존재', M.thresholdForSensitivity(sweep, 0.95) !== null);

const pc = M.perClassMetrics(cm);
const ab = pc.find(p => p.id === 'abnormal');
ok('abnormal 표본 2 · TP1 · FN1', ab.support === 2 && ab.tp === 1 && ab.fn === 1, ab);
ok('abnormal FP=2 (normal 1 + borderline 1)', ab.fp === 2, ab.fp);
ok('abnormal 정밀도 1/3', near(ab.precision, 1 / 3), ab.precision);

const conf = M.topConfusions(cm, 5);
ok('혼동쌍 3개 검출', conf.length === 3, conf.map(c => `${c.trueId}->${c.predId}:${c.count}`));

const worst = M.worstFailures(recs, 10);
ok('오분류 3건, 확신도 내림차순', worst.length === 3 && worst[0].topProb >= worst[2].topProb, worst.length);

const csv = M.recordsToCsv(recs, CLASS_IDS);
ok('CSV 헤더 + 6행', csv.split('\n').length === 7, csv.split('\n').length);

console.log('\n── 신뢰도 캘리브레이션 ──');
// 모델이 0.95라 말하지만 실제로는 60%만 맞는 상황 (과신)
const overconf = Array.from({ length: 100 }, (_, i) => ({ prob: 0.95, correct: i < 60 }));
const tbl = C.buildCalibration(overconf);
ok('보정표 생성됨', tbl !== null);
ok('  ECE ≈ 0.35 (과신 격차)', near(tbl.ece, 0.35, 0.01), tbl.ece);
const applied = C.applyCalibration(tbl, 0.95);
ok('  0.95 → 보정 0.60', near(applied.value, 0.6), applied.value);
ok('  구간 라벨 90–100%', applied.binLabel === '90–100%', applied.binLabel);
ok('  표본 100건', applied.n === 100, applied.n);

const borrowed = C.applyCalibration(tbl, 0.15);
ok('표본 없는 구간은 인접 구간 차용', borrowed.borrowed === true, borrowed);
ok('  차용값도 0.60', near(borrowed.value, 0.6), borrowed.value);

ok('표본 부족 시 null', C.buildCalibration([{ prob: .9, correct: true }]) === null);
ok('보정표 없으면 원값 반환', near(C.applyCalibration(null, 0.77).value, 0.77));

const desc = C.describeCalibration(tbl);
ok('과신을 danger로 진단', desc.tone === 'danger', desc);

// 잘 보정된 모델
const wellCal = [
  ...Array.from({ length: 50 }, (_, i) => ({ prob: 0.9, correct: i < 45 })),
  ...Array.from({ length: 50 }, (_, i) => ({ prob: 0.6, correct: i < 30 })),
];
const good = C.buildCalibration(wellCal);
ok('잘 보정된 모델은 ECE 낮음', good.ece < 0.05, good.ece);
ok('  ok로 진단', C.describeCalibration(good).tone === 'ok');


/* ══ 대회 불변식 (CLAUDE.md 2장) ═══════════════════════════════════════════
 * 산문으로 적은 규칙은 지켜지지 않습니다. 실제로 어긋난 적이 있어서
 * (inferRoi 0.70 vs center_crop 0.85) 기계가 검사하게 만듭니다. */
const fs = await import('node:fs');
const yml = fs.readFileSync(new URL('./prep/prep.config.yaml', import.meta.url), 'utf8');
const num = (re, src) => parseFloat((src.match(re) || [])[1]);

console.log('\n── 대회 불변식 ──');

const centerCrop = num(/center_crop:\s*([\d.]+)/, yml);
ok(`추론 ROI = 학습 center_crop (${PROJECT.decision.inferRoi} / ${centerCrop})`,
   PROJECT.decision.inferRoi === centerCrop,
   { inferRoi: PROJECT.decision.inferRoi, centerCrop });

ok('판단 보류 임계값 0.70 (대회 지정 기본값)',
   PROJECT.decision.holdThreshold === 0.70, PROJECT.decision.holdThreshold);

ok('무효 입력 클래스가 정확히 1개',
   INVALID_IDS.length === 1, INVALID_IDS);

ok('발견해야 하는 소견(positive) 클래스가 1개 이상',
   POSITIVE_IDS.length >= 1, POSITIVE_IDS);

ok('클래스 id 가 전부 영문 (TM 에 그대로 노출됨)',
   CLASS_IDS.every(id => /^[a-z0-9_]+$/i.test(id)), CLASS_IDS);

ok('입력은 이미지 업로드 (웹캠 off)',
   PROJECT.ui?.enableWebcam === false, PROJECT.ui?.enableWebcam);

ok('전문 지표는 기본 접힘 (최종 사용자 관점)',
   PROJECT.ui?.showDetailByDefault === false, PROJECT.ui?.showDetailByDefault);

ok('안전 문구에 "교육용 프로토타입" 과 "진단" 포함',
   /교육용 프로토타입/.test(PROJECT.copy.safetyNote) && /진단/.test(PROJECT.copy.safetyNote));

// 화면에 나가는 카피에서 "확률" 은 부정 표현으로만 허용합니다
const visible = [PROJECT.copy.safetyNote, PROJECT.copy.safetyLong, PROJECT.copy.privacyNote,
                 PROJECT.copy.heroLead, PROJECT.copy.holdMessage, PROJECT.copy.marginMessage,
                 ...PROJECT.classes.map(c => c.advice)].join(' ');
// "확률이 아닌 / 아니라" 같은 부정 표현은 먼저 지우고 남은 것만 셉니다.
// ('아닌' 과 '아니' 는 서로 다른 음절이라 한쪽만 매칭하면 놓칩니다.)
const stripped = visible.replace(/확률\s*이?\s*아[니닌]\S*/g, '');
const badProb = (stripped.match(/확률/g) || []).length;
ok('화면 카피에서 "확률" 을 단정 표현으로 쓰지 않음', badProb === 0,
   { 남은_확률: badProb, 문맥: stripped.match(/.{0,18}확률.{0,18}/g) });

ok('한계(limitations)를 hard/soft 로 구분',
   PROJECT.dataset.limitations.length > 0
   && PROJECT.dataset.limitations.every(l => l.kind === 'hard' || l.kind === 'soft'));

console.log(`\n${'─'.repeat(50)}`);
console.log(`통과 ${pass} / 실패 ${fail}`);
process.exit(fail ? 1 : 0);
