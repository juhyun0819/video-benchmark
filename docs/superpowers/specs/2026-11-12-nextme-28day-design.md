# NextMe 28일 타임라인: 늦은 출근일과 여행 (설계)

- 작성일: 2026-11-12
- 상태: 승인 (2026-11-12). 용량이 하루 모자라면 "그 하루만 집(아침) 비우기"로 한다(사용자 결정).
- 관련 스펙: `2026-11-12-synthetic-timeline-generation-design.md` (rev3). 이 문서는 그 스펙에 두 기능을 더한다. 늦은 출근일은 새 다이얼이고, 여행은 5.4의 `multi_day` 예외를 처음 구현하는 것이다.

## 1. 왜 필요한가

NextMe 한 사람(집·직장 고정)으로 28일 manifest를 만들려고 했더니, 필수 슬롯인 "집(아침)"이 모자랐다.

- NextMe 114일 중 아침(05~12시) 집 영상이 있는 날은 24일이다. 키워드 규칙으로 17일을 찾았고, caption을 직접 검토해 7일을 더 찾았다(`data/raw/bank/nextme800.morning_home_review.json`).
- 24일 중 15일은 영상이 11시 전에 시작한다(시간대 morning). 나머지 9일은 11시대뿐이다(시간대 midday).
- 지금 donor는 ATUS 전일제 직장인이다. 평일 집(아침) 슬롯이 6:30~7:15쯤이어서 11시대 영상은 하드 필터(시간대)를 통과하지 못한다.
- 28일 중 집 노드는 매일 쓰인다. 한 NextMe 날짜는 합성 하루에만 쓸 수 있다.

해결은 두 가지다.

- 매주 이틀을 "늦게 출근하는 날"로 둔다. 4주에 8일이고, 11시대 집 영상이 들어갈 수 있다.
- 실제 NextMe 여행 기간 하나(2026-05-07 목 ~ 05-10 일, 4일)를 여행 예외로 넣는다. 여행일에는 집 영상이 필요 없다.

## 2. 늦은 출근일 (`workday_late`)

- 다이얼: `dials.late_workdays_per_week` (정수 0~5, 기본 0). 기존 world는 영향이 없다.
- 날 유형: `workday_late`. 템플릿은 `workday`와 같다. 다른 것은 donor 시각뿐이다.
- 요일 고정: 사람마다 월~금 중 늦은 출근 요일을 seed로 한 번 고르고 매주 같은 요일에 둔다. 주 단위 패턴이라 반복 강도로 흔들지 않는다. 그래서 4주면 정확히 8일이다. 예외(연차)나 여행과 겹치면 그 날이 우선한다.
- donor: 늦은 출근일 전용 donor를 하나 더 뽑는다(`donors.weekday_late`). 실제 응답자의 하루라는 원칙(5.6)을 지키기 위해서다.
  - pool: 평일 근무자 중 일터 근무 시작이 11:30~15:00인 응답자. ATUS 2025에서 32명이다. 기상 중앙값 07:30, 퇴근 중앙값 20:30, 취침 중앙값 23:15다.
  - backoff, 가중 추출, 범위 밖 재추출(CR5), 필수 슬롯 검사는 평일 donor와 같다.
  - 효과: 집(아침) 슬롯이 약 07:30~11:10이 되어 morning과 midday 두 시간대에 걸친다. 그래서 11시대 집 영상도 들어간다.
- 기록: schedule과 manifest의 날 유형에 `workday_late`, `donors.weekday_late`가 남는다. `realized`에 `late_workday_count`를 넣는다.

## 3. 여행 (`multi_day` 예외, 스펙 5.4의 첫 구현)

### 3.1 world

- 새 노드 종류 `trip`: `{ kind: "trip", lodging_setting, outing_setting }`. 여행지 하나를 뜻한다.
- 여행 노드는 세 하위 장소로 쓴다. `<trip>:lodging`(숙소), `<trip>:outing`(관광·야외), `<trip>:meal`(여행지 식당).
- 바인딩: `node_binding.trip_1 = { dataset: ["nextme800"], person_source: "nextme800:wearer", day_group_sequence: ["2026-05-07", "2026-05-08", "2026-05-09", "2026-05-10"] }`.
  - `day_group_sequence`는 새 바인딩 필드다. 여행 k일째에는 k번째 원본 날짜의 클립만 쓴다. 기간 내내 같은 여행지(5.4)이고, 체크인부터 체크아웃까지 순서도 실제와 같아진다.
- 다이얼: `dials.trips: [{ node: "trip_1", length_days: 4 }]`. 길이는 `day_group_sequence` 길이와 같아야 한다.

### 3.2 일정 생성

- 시작일: 하루 예외와 슬롯 예외보다 먼저 정한다. seed로 고르되, 원본 여행의 요일 구성을 따른다. 원본이 목~일이면 합성에서도 목요일에 시작하는 날 중에서 고른다. 실제 영상의 요일 맥락(평일 이동, 주말 관광)이 맞기 때문이다.
- 여행 기간의 날은 `day_type: "trip_day"`, 템플릿 `trip_day`를 쓴다. 다른 예외는 여행일과 겹치지 않는다(5.3).
- 기록: `{ scope: "multi_day", type: "trip", replaces: [원래 날 유형들], day_range: [s, s+3], node: "trip_1" }`. 이 기록은 여행 기간의 모든 날에 같은 값으로 들어가고, manifest에도 남는다.
- donor: 여행일은 주말(쉬는 날) donor 시각을 쓴다.

### 3.3 `trip_day` 템플릿

| 슬롯 | 노드 | 필수 | anchor |
|---|---|---|---|
| lodging_morning | trip:lodging | 예 | wake |
| breakfast | trip:meal 또는 trip:lodging | 아니오 | breakfast |
| outing | trip:outing | 예 | sport (없으면 lunch 전) |
| lunch | trip:meal | 아니오 | lunch |
| outing_pm | trip:outing | 아니오 | — |
| dinner | trip:meal | 아니오 | dinner |
| lodging_evening | trip:lodging | 아니오 | — |

- 여행 첫날과 마지막 날은 이동이 많다(공항, 차). 이동 클립은 `transit`이고 템플릿에 이동 슬롯이 없어서 쓰지 않는다. 이동 시간은 영상 없는 시간이 된다.
- 필수 슬롯이 그 날 원본에 없으면 여행 시작일 후보를 바꿔 다시 시도한다. 끝내 안 되면 오류로 끝낸다(10장).

### 3.4 매칭

- 하드 필터 1(노드)에 `day_group_sequence`를 더한다. 여행 k일째 슬롯 후보는 k번째 원본 날짜의 클립뿐이다.
- 나머지 규칙(시간대, 재사용 금지, 날짜 하나는 하루에만)은 그대로다.

### 3.5 verifier

- 새 규칙 `trip`:
  - 여행 기간의 모든 클립이 같은 여행 노드에 속한다.
  - k일째 클립은 k번째 원본 날짜에서 나온다.
  - 여행 기간에는 집·직장 클립이 없다.

## 4. NextMe 라벨

- 클립 단위: L3 구간(2분 이상). 실험용 라벨러의 방식이다.
- 집, 직장, 식당, 산책 같은 장소는 키워드 규칙으로 정한다. 직접 검토한 7일의 오전 구간은 집으로 덮어쓴다.
- 여행 날짜(05-07~05-10)의 클립은 여행 규칙으로 라벨을 붙인다.
  - 숙소(`trip:lodging`): hotel, resort, cabin, homestay, lobby, room
  - 관광(`trip:outing`): waterfall, scenic, sightseeing, beach, trail, park, attraction
  - 식당(`trip:meal`): 여행지의 식사
- 라벨은 사람이 검수해야 한다(D9). 이번 결과는 검수 전 초안으로 표시한다.

## 5. NextMe world (`nextme-p00`)

- 속성: country HK(ATUS 대체 허용), occupation_group professional, day_shift.
- 노드와 바인딩:
  - home, workplace: NextMe 집과 직장
  - frequent_1(식당): NextMe 식당 클립
  - hobby_1(산책, 실외, 주 1회): NextMe 산책·등산 클립
  - trip_1: 위의 4일
- 끼니 장소: NextMe 실제 비율에 맞춘다.
  - 평일 점심: 학교 식당 0.36, 식당 0.55, 집 0.09
  - 평일 저녁: 식당 0.69, 학교 식당 0.19, 집 0.12
  - 쉬는 날 점심·저녁: 식당 0.85, 집 0.15
- 다이얼:
  - late_workdays_per_week 2
  - trips [{trip_1, 4}]
  - routine_strength 0.7
  - exception_rate {slot 0.1, day 0.05}
  - 28일, seed 1234

## 6. 예상 용량과 남는 위험

- 여행 4일(목~일)을 빼면 24일이 남는다. 평일 18일, 주말 6일이다.
  - 늦은 출근일 8일: 11시대 집 영상 날짜 9일과 아침 영상 날짜를 쓸 수 있다.
  - 나머지 평일 10일과 주말 6일, 합계 16일: 아침(11시 전) 집 영상이 있는 날짜가 필요하다. 그런 날은 15일이다.
- 즉 하루가 모자랄 수 있다. 생성 전 용량 검사(D7)가 정확한 수를 보고하고, 모자라면 오류로 멈춘다. 그때 두 방법 중 하나를 고른다.
  - 늦은 출근일을 하루 늘린다(주 2일 + 한 주만 3일).
  - 그 하루만 집(아침)을 비운다.
- 늦은 출근일은 퇴근이 20:30쯤이다. 필수 저녁 식사가 들어갈 시간은 G9/G11 검사가 보장하고, 안 되면 donor를 다시 뽑는다.

## 7. 범위 밖

- 렌더링(Plan 2)
- 여행 이동 슬롯
- 여행 2회 이상
- 여행지가 원본과 다른 사람인 경우
