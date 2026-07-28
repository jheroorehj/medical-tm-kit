# 프론트엔드 구현 요구사항

프론트엔드를 직접 만들 때 필요한 명세입니다. **`web/js/` 의 13개 모듈은 DOM을 거의
모릅니다** — 순수 로직이므로 그대로 재사용하고, 화면만 새로 만들면 됩니다.

- 재사용 가능(그대로 쓰세요): `store` `models` `infer` `quality` `temporal`
  `saliency` `calibrate` `metrics` `holdout` `history` `project.config`
- 교체 대상: `ui.js`(렌더 전용) · `main.js`(이벤트 배선) · `index.html` · `app.css`

검증 상태: `node test-logic.mjs` → 60/60 통과. 로직 계층은 신뢰할 수 있습니다.

---

## 0. 환경 제약 — 먼저 확인할 것

| 제약 | 내용 |
|---|---|
| **`localhost` 필수** | Chrome은 `getUserMedia`에 secure context를 요구합니다. `file://`로 열면 카메라가 **절대** 열리지 않습니다. ES 모듈 import도 CORS로 차단됩니다 |
| **CDN 2개 선행 로드** | `tfjs` → `@teachablemachine/image` 순서. `tmImage`는 전역 `tf`를 전제합니다. 앱 스크립트보다 먼저 와야 합니다 |
| **모델 URL은 공개 링크** | TM이 발급하는 URL은 접근 제어가 없습니다 |
| **비디오 메타데이터 지연** | `videoWidth`가 0인 동안 `drawImage`가 실패합니다. 항상 0 체크 후 진행 |
| **오프라인 불가** | CDN 의존. 시연 환경 네트워크를 반드시 사전 확인 |

### ★ 반드시 넣어야 하는 CSS 한 줄

```css
[hidden] { display: none !important; }
```

작성자 스타일시트의 `display` 선언이 브라우저 기본 `[hidden] { display:none }`을
덮어씁니다. `.badge { display: flex }` 가 있으면 `<div class="badge" hidden>` 이
**그대로 보입니다.** 실제로 이 버그를 겪었습니다 — 카메라가 꺼진 상태에서 LIVE
배지와 품질 스트립이 노출됐습니다. 가시성 제어를 `hidden` 속성으로 통일할 경우 필수입니다.

### ★ SVG 요소에는 `el.hidden = bool` 이 통하지 않습니다

```js
// ✗ SVG 요소에서는 아무 일도 일어나지 않습니다
svgCircle.hidden = true;

// ✓ 속성을 직접 조작하면 HTML·SVG 양쪽에서 동작합니다
on ? el.removeAttribute('hidden') : el.setAttribute('hidden', '');
```

`hidden` 은 `HTMLElement` 의 IDL 프로퍼티입니다. `SVGElement` 에는 없으므로
대입해도 **속성이 아니라 JS 확장 프로퍼티만 생기고** CSS 의 `[hidden]` 규칙이
전혀 적용되지 않습니다. 임계값 곡선의 커서선과 데이터 포인트를 숨기려다 이
버그를 겪었습니다 — 데이터가 없는데 좌표 (0,0) 에 점이 찍혀 있었습니다.
`ui.show()` 를 속성 기반으로 구현하면 한 곳에서 해결됩니다.

---

## 1. 화면 구성 (4개 뷰)

```
헤더  [로고·제목·부제·배지]        [모델 상태 pill]
      [판독] [성능 평가] [이력] [설정]     ← 탭

① 판독      좌: 입력(웹캠/이미지)   우: 결과
② 성능 평가  홀드아웃 폴더 드롭 → 지표·혼동행렬·캘리브레이션·실패사례
③ 이력      판독 로그 표 + CSV
④ 설정      모델 파이프라인 · 판정 정책 · 품질 게이트 · 저장 데이터
```

우측 결과 패널은 **고정폭(360~400px)**, 좌측이 잔여 폭. `align-items: start`를 주지
않으면 짧은 결과 카드가 입력 카드 높이만큼 늘어나 빈 공간이 생깁니다.
980px 이하에서 단일 칼럼으로 스택.

---

## 2. 판독 뷰 — 필수 요구사항

### 2-1. 입력 소스 2종

**웹캠**
- `<video autoplay muted playsinline>` — `muted` 없으면 자동재생이 차단됩니다
- 시작 / **실시간 판독 토글** / 캡처&판독 / 중지 — 4개 버튼
- 버튼 활성 상태가 짝을 이뤄 뒤집혀야 합니다 (시작 비활성 ↔ 캡처·중지 활성).
  불가능한 상태 조합을 UI가 차단
- 탭 이탈 시, `pagehide` 시 반드시 `stream.getTracks().forEach(t => t.stop())`.
  `srcObject = null` 만으로는 **카메라 LED가 꺼지지 않습니다**

**이미지 업로드**
- 드래그앤드롭 + 클릭 선택. **드롭존은 키보드로 도달·조작 가능해야 합니다**
  (`<div onclick>` 금지 → `tabindex="0"` + Enter/Space 핸들러 또는 `<button>`)
- 초기화 시 `img.src = ''` 를 쓰지 마세요. **`img.src` 는 빈 문자열을 문서 URL로
  해석해 항상 truthy가 됩니다.** `removeAttribute('src')` 를 쓰고, `blob:` URL은
  `URL.revokeObjectURL()` 로 해제하세요
- 초기화 시 판독 버튼을 **다시 비활성화**해야 합니다

### 2-2. 가이드 오버레이 — 모델이 보는 영역과 일치할 것

```js
guideBox.style.width = `${PROJECT.decision.inferRoi * 100}%`;
```

가이드 박스는 장식이 아니라 **`inferRoi`가 실제로 잘라내는 영역**을 그려야 합니다.
사용자가 "박스 안에 맞추면 판독된다"고 믿을 수 있어야 하고, 그 값은
`prep.config.yaml`의 `center_crop`과 같아야 합니다(학습·추론 전처리 일치).

품질 상태에 따라 테두리 색을 바꾸세요 — 초록(양호) / 빨강(불량).

### 2-3. 품질 게이트 + 촬영 코칭

```js
const m = measureFrame(video, settings.inferRoi);   // quality.js
const ev = evaluateQuality(m, settings);
// ev = { ok, coach, issues[] }
```

- **250ms 간격**으로 추론과 무관하게 항상 측정 (카메라가 켜져 있는 동안)
- 지표 4개 실시간 표시: 품질점수 / 초점(라플라시안 분산) / 밝기 / 대비
  → 임계값 조정 시 사용자가 실제 수치를 보며 맞춰야 하므로 노출 필수
- **코칭 문구는 한 번에 하나만.** 동시에 여러 지시를 하면 사용자가 못 따릅니다
  (`evaluateQuality`가 심각도 순으로 정렬해 하나만 돌려줍니다)
- 실시간 모드에서 `!ev.ok` 면 **추론을 건너뜁니다**(조용히). 단발 캡처는 경고만
  하고 진행 — 사용자가 의도적으로 눌렀으므로

### 2-4. 실시간 판독 루프

```js
// setInterval 금지 — 추론이 간격보다 길면 큐가 쌓입니다
async function tick() {
  if (!live) return;
  if (!busy) { busy = true; try { await runOnce('live'); } finally { busy = false; } }
  timer = setTimeout(tick, settings.inferIntervalMs);
}
```

- `busy` 플래그로 중첩 추론 차단 (필수)
- 기본 간격 200ms. 설정에서 60~1000ms 조정 가능하게
- 히트맵 분석을 시작하면 **실시간을 멈춰야 합니다** (프레임이 계속 바뀌면 무의미)

### 2-5. 추론 1회의 흐름 — 순서를 지킬 것

```
① 품질 게이트        measureFrame → evaluateQuality
② 224 정사각 변환    toSquareCanvas(source, inferRoi, 224, workCanvas)
③ 게이트키퍼         runGate(registry, canvas, gateCfg)
④ 주분류기+앙상블    runClassify(registry, canvas)  → {probs, perModel}
⑤ 시간축 안정화      stabilizer.push(probs)  ← 실시간 모드에서만
⑥ 판정               decide(effectiveProbs, settings, {gate, calibration, copy})
⑦ 렌더 + 이력
```

②의 캔버스는 **매번 새로 만들지 말고 하나를 재사용**하세요. 히트맵이 이 캔버스를
다시 씁니다.

### 2-6. 판정 결과 표시 — 5개 상태를 모두 그려야 합니다

`decide()`가 돌려주는 `status` 는 5가지입니다. **각각 다른 톤과 아이콘이 필요합니다.**

| status | 의미 | 톤 | `trustworthy` |
|---|---|---|---|
| `gated` | 게이트키퍼가 막음 — 판독 대상 없음 | neutral | false |
| `invalid` | 무효 입력 클래스가 1위 | neutral | false |
| `hold` | 1위 유사도 < 임계값 → **판단 보류** | danger | false |
| `ambiguous` | 1·2위 차이 < 마진 → 구분 어려움 | warn | false |
| `ok` | 판정 성립 (positive면 danger, negative면 ok) | ok/danger | true |

`decide()`는 `{ status, tone, icon, headline, detail, advice, trustworthy, ranked,
top, second, margin, entropy, calibrated }` 를 줍니다. **그리기만 하면 됩니다 —
판정 로직을 프론트엔드에 다시 쓰지 마세요.**

> ⚠️ `ambiguous`는 기본 설정에서 절대 발동하지 않습니다. softmax 합이 1이므로
> `마진 ≥ 2×1위−1`이고, 1위가 0.70이면 마진은 최소 0.40입니다. `2×임계값−1 < 마진기준`
> 일 때만 도달 가능 — 즉 **임계값을 낮춘 민감도 우선 운영점에서만** 활성화됩니다.
> 설정 화면에서 이 사실을 사용자에게 알려 주세요(현재 `updateMarginHint()`가 담당).

### 2-7. 불확실성 지표 5개

유사도 / 보정 신뢰도 / 1·2위 차이 / 불확실성(엔트로피) / 흔들림(jitter).

보정 신뢰도는 `calibrated.n === 0` 이면 "표본 부족"으로 표시하고, `title`에
어느 구간·몇 건 기준인지 넣으세요. 차용된 값이면(`borrowed`) 그것도 밝힐 것.

### 2-8. 클래스별 유사도 막대 — 애니메이션 함정

```js
// ✗ 이렇게 하면 트랜지션이 절대 재생되지 않습니다
container.innerHTML = bars.map(b => `<div style="width:${b.pct}%">`).join('');
```

새로 만든 노드는 초기값이 곧 최종값이므로 보간할 구간이 없습니다. **원본 `ses08`
템플릿의 실제 버그입니다.**

```js
// ✓ 노드를 재사용하고 width만 갱신 → 또는 0으로 삽입 후 rAF에서 목표값
if (classesChanged) rebuildNodes();          // 클래스 목록이 바뀔 때만
requestAnimationFrame(() => {
  nodes.forEach((el, i) => el.querySelector('.fill').style.width = `${pct}%`);
});
```

**1위만 보여주지 말고 모든 클래스를 표시하세요.** 2위와 3%p 차이인지 60%p 차이인지가
보여야 합니다. 색은 `PROJECT.classes[].color`.

### 2-9. 근거 히트맵 (오클루전)

```js
const map = await occlusionMap(model, workCanvas, {
  grid: 8, targetId: decision.top.id, onProgress, token
});
drawHeatmap(canvas, map);           // 원본을 먼저 그린 뒤 오버레이
const interp = interpretMap(map);   // { tone, text } — 자동 해석
```

- 64회 추론 = 1~3초. **진행률 표시 필수**, **중단 버튼 필수**(`token.aborted = true`)
- `saliency.js`가 프레임 양보(`requestAnimationFrame`)를 하므로 UI는 얼지 않습니다
- `interpretMap()`이 "근거의 68%가 주변부에 있습니다 → 지름길 학습 의심" 같은 해석을
  문장으로 돌려줍니다. 그대로 노출하세요 — 이게 이 기능의 실제 가치입니다

### 2-10. 항상 고정 노출

- **안전 고지문** (`PROJECT.copy.safetyNote`) — "확률"이 아니라 "유사도"
- **개인정보 고지** (`PROJECT.copy.privacyNote`) — 브라우저 내 처리, 미전송
- 판독 시각 + 판정 당시 임계값

---

## 3. 성능 평가 뷰 — 필수 요구사항

### 3-1. 폴더 드롭

```js
const items = await collectFromDrop(e.dataTransfer);   // holdout.js
// 또는 <input type="file" webkitdirectory>  → collectFromInput(files)
```

- `webkitGetAsEntry()`는 **drop 이벤트 직후에만 유효**합니다. entry 목록을 먼저
  확보한 뒤 순회하세요
- `readEntries()`는 한 번에 **최대 100개**만 줍니다. 빌 때까지 반복 호출 필수
- 폴더명 = 정답 클래스. `verifyLabels(items, CLASS_IDS)` 로 정합성 먼저 검사하고
  결과를 화면에 보여주세요. **실습에서 가장 흔한 사고 지점입니다**
  (`Normal` vs `normal`)

### 3-2. 배치 진행

수백 장을 돌리므로 **진행률 + 중단 버튼 필수**. `runBatch()`가 20장마다 프레임을
양보하므로 화면은 얼지 않습니다.

### 3-3. 지표 타일 6개

표본 / 전체 정확도 / **민감도** / **특이도** / 양성예측도 / 보류율.

`PROJECT.copy.metricPriority` 가 가리키는 지표를 시각적으로 강조하세요.
민감도·특이도가 의료 도메인의 언어입니다 — 정확도보다 크게 보여야 합니다.

### 3-4. ★ 임계값 슬라이더 → 실시간 트레이드오프

이 대회에서 가장 점수가 될 화면입니다.

```js
const sweep = thresholdSweep(records, POSITIVE_IDS, policy);  // 19개 지점
const m = binaryMetrics(records, POSITIVE_IDS, threshold, policy);
```

- 슬라이더를 움직이면 **재추론 없이** 지표가 즉시 갱신 (확률 벡터를 이미 갖고 있음)
- 민감도·특이도·보류율 곡선을 그리고 현재 임계값에 커서 표시
- **보류 처리 정책 선택** 필수: `escalate`(보류를 양성 의심으로 2차 확인 — 임상적
  안전) / `exclude`(순수 모델 성능). 이 선택이 지표를 크게 바꿉니다
- 버튼 3개: `Youden 균형점` / `민감도 95% 확보점` / `이 임계값을 판독에 적용`

### 3-5. 혼동행렬

대각선 강조(정답), 비대각선 오분류 강조, 0은 흐리게. 행별 합계와 재현율 열 추가.
`topConfusions()` 로 "가장 자주 헷갈리는 조합"을 별도 노출 — 클래스를 합칠 근거입니다.

### 3-6. 캘리브레이션

```js
const table = buildCalibration(toCalibrationRecords(records));
saveCalibration(table);   // 판독 화면이 자동으로 이 표를 씁니다
```

구간별 표: 모델 출력 / 표본 / 평균 출력 / **실측 정확도** / 격차.
격차가 음수면 과신. ECE를 한 문장으로 진단(`describeCalibration()`).

### 3-7. 실패 사례 갤러리

`worstFailures(records, 24)` — 확신하며 틀린 순. 썸네일은 `runBatch()`가 돌려주는
`files: Map<path, File>` 로 `URL.createObjectURL()`. **교체 시 이전 URL을 반드시
`revokeObjectURL()`** 하세요(메모리 누수).

---

## 4. 설정 뷰 — 필수 요구사항

### 4-1. 모델 파이프라인 3단

```
① 게이트키퍼 (on/off, URL, 통과 클래스, 최소 확신도)
② 주분류기   (URL) — 필수
③ 앙상블     (URL·라벨·가중치 여러 개, 추가/삭제)
```

- URL 정규화는 `normalizeModelUrl()` — 트레일링 슬래시 자동 보정, 플레이스홀더 거부
- 각 슬롯의 상태를 개별 표시: 미로드 / 불러오는 중 / 로드됨(클래스 N개) / 실패
- URL은 `localStorage`에 저장. 새로고침해도 유지되고 자동 재로드
- 앙상블 삭제 시 **슬롯 번호가 밀리므로** 전체를 언로드하고 다시 로드해야 합니다

### 4-2. ★ 클래스 정합성 리포트

```js
const v = registry.verifyClasses(CLASS_IDS);
// { ok, missing[], extra[], modelLabels[] }
```

TM 클래스명과 `project.config.js`의 `id`가 어긋났는지 **로드 즉시** 보고하세요.
어느 이름이 다른지, 모델의 실제 클래스가 무엇인지 다 보여줘야 합니다.
오타 하나로 몇 시간 날리는 사고를 여기서 잡습니다.
`verifyEnsembleConsistency()` 로 앙상블 구성원 간 클래스 집합 불일치도 경고.

### 4-3. 판정 정책 슬라이더

| 항목 | 범위 | 비고 |
|---|---|---|
| 판단 보류 임계값 | 0.05–0.95 | 평가 탭 슬라이더와 양방향 동기화 |
| 1·2위 최소 차이 | 0–0.6 | **도달 가능 여부 진단 필수** (2-6 참조) |
| 이동평균 창 | 1–30 프레임 | |
| 확정 조건 | 1–30 연속 프레임 | |
| 추론 간격 | 60–1000 ms | |
| 보정 신뢰도 표시 | on/off | 보정표 없으면 그 사실을 안내 |

모든 변경은 즉시 `localStorage` 저장 + `stabilizer.configure()` 반영.

### 4-4. 품질 게이트 임계값

초점 최소값(5–300) / 허용 밝기 범위(이중 슬라이더).
**웹캠을 켠 상태에서 실시간 수치를 설정 화면에도 표시**하세요 — 실제 값을 보지 않고
이 숫자를 맞출 수는 없습니다.

### 4-5. 저장 데이터 관리

이력 건수 / 보정표 유무 / 평가 결과 / 사용 용량. 개별 삭제 + 전체 초기화.
`localStorage` 용량 초과(`QuotaExceededError`)를 처리해야 합니다 —
`store.js`가 이력을 절반 버리고 재시도합니다.

---

## 5. 이력 뷰

- 원본 영상 저장 금지. **96px 썸네일 + 메타데이터만**
- 판정 당시 임계값을 함께 기록 → 정책을 바꿔도 과거 판정이 재현됩니다
- 실시간 모드는 `justConfirmed` 순간에만 기록 (초당 5건이 쌓이지 않도록)
- CSV 내보내기 — **BOM 필수**(`﻿`), 없으면 Excel에서 한글이 깨집니다
- 상단에 상태별 집계 타일(판정/보류/구분어려움/무효)

---

## 6. 접근성 — 원본 템플릿에서 빠져 있던 것들

- [ ] 탭에 `role="tablist"` / `role="tab"` / `aria-selected` / `role="tabpanel"`
- [ ] 드롭존을 **키보드로 조작 가능**하게 (`tabindex="0"` + Enter/Space)
- [ ] 결과 영역에 `aria-live="polite"` — 스크린리더가 판정 갱신을 알려야 합니다
- [ ] 실시간 판독 토글에 `aria-pressed`
- [ ] `:focus-visible` 스타일 (드롭존·버튼)
- [ ] `@media (prefers-reduced-motion: reduce)` 로 애니메이션 억제
- [ ] 색만으로 정보를 전달하지 말 것 — 판정 상태에 아이콘/텍스트 병기

---

## 7. 보안 · 데이터 취급

- [ ] **모델 클래스명을 `innerHTML`에 넣지 마세요.** `metadata.json`에서 온 외부
      문자열입니다. `textContent` 또는 이스케이프(`ui.esc()`) 사용.
      원본 템플릿의 실제 인젝션 경로였습니다
- [ ] `blob:` URL 전부 `revokeObjectURL()`
- [ ] 영상을 외부로 전송하지 않는다는 사실을 UI에 명시 (심사 항목)
- [ ] 이력에 원본 영상 저장 금지

---

## 8. 디자인 시스템 (계승 권장)

`ses08` 원본의 토큰을 그대로 물려받으면 실습 교재와 시각적 연속성이 생깁니다.

```
배경   #F7F6F3 (웜 그레이) / 카드 #FFFFFF / 함몰면 #F0EEE9
경계   #E2DED7
텍스트 #1A1916 / #6B6760 / #A8A49E   ← 3단 위계
강조   #3D5AFE  배경 #EEF1FF  배경위텍스트 #1A2EB0   ← ★ 3형제 패턴
성공   #0B6E4F / #E6F4EE     경고 #B26A00 / #FFF4E0
위험   #C0392B / #FDECEA
radius 6 / 10 / 16px         그림자 2겹 (1px 타이트 + 12px 확산)
폰트   Noto Sans KR (UI) + JetBrains Mono (모든 숫자)
```

**강조색 3형제**가 핵심입니다. 파란 배경 위에 파란 글자를 그대로 쓰면 명암비가
깨지므로 배경용·텍스트용을 따로 둡니다. 위험색도 같은 구조.

**숫자는 전부 모노스페이스 + `font-variant-numeric: tabular-nums`.** 실시간으로
갱신되는 확률·지표가 자릿수마다 흔들리면 읽을 수 없습니다.

모션은 절제하세요. 프로젝트 규칙에 따라 구조·레이아웃·색상은 `taste-skill` 계열,
애니메이션 값(easing, spring, interruptible transition)은 `apple-design` /
`emil-design-eng` 기준을 따르면 됩니다.

---

## 9. 구현 우선순위

기간이 짧으면 위에서부터.

| 순위 | 항목 | 이유 |
|---|---|---|
| 1 | 모델 로드 + 클래스 정합성 리포트 | 이게 없으면 나머지가 동작하지 않음 |
| 2 | 이미지 업로드 판독 + 5개 상태 판정 표시 | 최소 동작 데모 |
| 3 | 웹캠 캡처 판독 | 대회 요구사항 |
| 4 | 품질 게이트 + 가이드 + 코칭 | 데모에서 가장 잘 먹힘 |
| 5 | 홀드아웃 평가 + 혼동행렬 + 민감도/특이도 | 발표 수치의 근거 |
| 6 | 실시간 판독 + 시간축 안정화 | 웹캠 고유 무기 |
| 7 | 임계값 슬라이더 트레이드오프 | 심사 어필 최고 |
| 8 | 오클루전 히트맵 | 차별화 최고 |
| 9 | 캘리브레이션 | 전문성 어필 |
| 10 | 이력 · CSV · 실패 갤러리 | 보고서 자료 |

1~5까지가 "제대로 만든 서비스", 6~9가 "다른 팀에 없는 것"입니다.

---

## 10. 검증 체크리스트

프론트엔드를 만든 뒤 이것들을 확인하세요.

```
□ [hidden] { display:none !important } 넣었는가
□ 카메라 꺼진 상태에서 LIVE 배지·품질 스트립·가이드가 숨겨지는가
□ 중지/탭이탈/페이지이탈 후 카메라 LED가 꺼지는가
□ 초기화 후 판독 버튼이 비활성화되는가
□ 모델 없이 판독을 누르면 안내가 나오는가
□ 웹캠에 아무거나(손·벽) 비추면 '무효 입력' 또는 '판단 보류'가 나오는가   ★
□ 막대 그래프 애니메이션이 실제로 재생되는가
□ 판정 5개 상태가 각각 다른 톤으로 그려지는가
□ 마진 관문 도달 불가 진단이 뜨는가
□ 홀드아웃 폴더명 대소문자가 틀렸을 때 경고가 뜨는가
□ 임계값 슬라이더가 재추론 없이 지표를 갱신하는가
□ 히트맵 중단 버튼이 실제로 멈추는가
□ CSV를 Excel에서 열었을 때 한글이 깨지지 않는가
□ 새로고침 후 설정·모델URL·보정표가 유지되는가
□ 620px 폭에서 레이아웃이 깨지지 않는가
□ 키보드만으로 드롭존까지 도달·조작 가능한가
□ 콘솔에 에러가 없는가
```

★ 표시한 항목이 심사위원이 가장 먼저 시도할 것입니다.

---

## 11. 참고: 로직 모듈 API 요약

```js
// project.config.js
PROJECT, CLASS_IDS, POSITIVE_IDS, NEGATIVE_IDS, INVALID_IDS, classOf(id)

// models.js
new ModelRegistry() → .load(slot,url,meta) .get() .has() .status() .unload()
                      .classifierSlots() .verifyClasses() .verifyEnsembleConsistency()
                      .onChange(fn)
SLOT.GATE / SLOT.PRIMARY / SLOT.ens(i) · normalizeModelUrl(raw)

// quality.js
measureFrame(source, roi) → {blur, luma, contrast, saturation, clipHigh, clipLow}
evaluateQuality(m, settings) → {ok, coach, issues[]}
qualityScore(m, settings) → 0~100

// temporal.js
new TemporalStabilizer({window, confirmFrames, holdThreshold})
  .push(probs) → {smoothed, topId, topProb, margin, streak, progress,
                  confirmed, justConfirmed, jitter, filled}
  .configure() .reset()
normalizedEntropy(probs) → 0~1

// infer.js
predictProbs(model, source) → {classId: prob}
runGate(registry, source, gateCfg) → {skipped, pass, topId, prob}
runClassify(registry, source) → {probs, perModel}
decide(probs, settings, ctx) → {status, tone, icon, headline, detail, advice,
                                trustworthy, ranked, top, second, margin,
                                entropy, calibrated}
toSquareCanvas(source, roi, size, outCanvas)
STATUS.{GATED,INVALID,HOLD,AMBIGUOUS,OK} · pct(v)

// saliency.js
occlusionMap(model, canvas, {grid, targetId, onProgress, token})
  → {grid, values, base, max, min, targetId, aborted}
drawHeatmap(outCanvas, map, alpha) · interpretMap(map) → {tone, text}

// calibrate.js
buildCalibration(records, nBins) → {bins, n, ece, nBins} | null
applyCalibration(table, prob) → {value, n, binLabel, borrowed}
describeCalibration(table) → {tone, text}

// metrics.js
confusionMatrix(records, classIds) · perClassMetrics(cm) · overallAccuracy(cm)
binaryMetrics(records, positiveIds, threshold, holdPolicy)
thresholdSweep() · bestByYouden() · thresholdForSensitivity()
toCalibrationRecords() · worstFailures() · topConfusions()
recordsToCsv() · fmtPct()

// holdout.js
collectFromDrop(dataTransfer) · collectFromInput(fileList)
verifyLabels(items, classIds) → {unknown[], missing[], counts[]}
runBatch(registry, items, {roi, onProgress, token})
  → {records, failed, aborted, files: Map}

// history.js
makeThumb(source, size) · record({source,decision,quality,thumb,settings})
list() · clear() · toCsv() · summarize() · download(name, text, mime)

// store.js
loadSettings(PROJECT) · saveSettings() · resetSettings()
loadModelUrls() · saveModelUrls()
loadHistory() · pushHistory() · clearHistory()
loadCalibration() · saveCalibration() · clearCalibration()
loadHoldout() · saveHoldout() · clearHoldout()
wipeAll() · storageUsage()
```
