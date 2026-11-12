# rev3 변경 전후 비교 (2026-11-12)

rev3 변경(CR1–CR8)을 적용하기 전 트리(`ea048f5`)와 적용한 뒤 트리(`8ea5cf1` 위 HEAD)를 같은 seed로 돌려 비교했다. 사용자가 요청한 비교다. 표의 `old`는 `ea048f5`, `new`는 rev3 트리다.

## 요약

| 항목 | 목표 | old | new | 판정 |
|---|---|---|---|---|
| 평일 저녁 없는 날 (스케줄) | 0 | 3.0% (323/10766) | 0.0% (0/10767) | 달성 |
| 평일 저녁 clip 없는 날 (full run) | 0 | 4.3% (47/1083) | 0.0% (0/1083) | 달성 |
| slot exception rate (공통 정의, 다이얼 0.5) | ≈ 0.5 | 48.4% (7421/15347) | 50.2% (7519/14992) | 달성 (new의 기록 E1은 50.4%, §5) |
| day exception rate (다이얼 0.1) | ≈ 0.1 | 10.3% (1234/12000) | 10.3% (1233/12000) | 변화 없음 (예상대로) |
| optional slot drop (스케줄, 하루 1개 이상) | 감소 | 10.2% (1719/16800) | 9.5% (1601/16800) | 평일은 달성, 휴일은 악화 |
| 평일 optional drop 날 | 감소 | 9.7% (1048/10766) | 6.8% (735/10767) | 달성 |
| hobby_day 저녁 drop | (CR2 범위 밖) | 4.5% (217/4800) | 12.0% (575/4800) | 악화 (알려진 문제). ablation에서 CR8 sport 채움을 되돌리면 4.0% (§4) |
| 조용한 day-level clamp | 0 | 25.6%의 날 (4302/16800) | 0 (clamp 1.8%는 모두 기록) | 달성 |
| 최종 donor 프로필의 E5 위반 (측정) | 0 | 평일 39.0% (234/600), 주말 43.0% (258/600) | 평일 0.5% (3/600, seed 149의 required_slots), 주말 0 (0/600) | 거의 달성 (§6) |
| 생성 실패 | 0 | 스케줄 0/600, full 0/60 | 스케줄 0/600, full 0/60 | 동일 |
| no_video_spans tiling | 실패 0 | 해당 없음 | manifest 60개, 1680일, 실패 0 | 달성 |

## 방법

### 트리

- old: `git worktree add <scratchpad>/rev3-baseline ea048f5`로 꺼냈다. worktree에는 `node_modules`가 없어서, 메인 checkout의 `node_modules`를 가리키는 심볼릭 링크를 worktree 루트에 두었다(`ln -s <repo>/node_modules <worktree>/node_modules`). 스크립트는 메인 checkout에서 tsx로 실행하고 `<worktree>/packages/core/src/index.ts`를 동적 import한다. core가 import하는 `zod`는 import하는 파일 위치에서 찾으므로 이 링크로 해결된다. old 결과의 `template_file`이 `templates.v1.json`이고, old 날에는 `jitter_redraws` 같은 rev3 필드가 없어서 old 코드가 실제로 실행된 것을 확인했다. 비교가 끝난 뒤 링크를 지우고 `git worktree remove`로 정리했다.
- new: 이 저장소의 HEAD.
- 각 트리는 자기 config 디렉터리를 쓴다(old는 `templates.v1.json`, new는 `templates.v2.json`. gap 표와 exception 카탈로그는 같은 파일이다).
- ATUS pool은 두 트리 모두 `data/distributions/atus2025/atus2025_stats_per_respondent.csv` 같은 파일이다. 다만 old의 주말 pool은 일한 응답자 행도 포함한다. 이것도 바뀐 점에 들어간다.

### 월드

- `config/worlds/example-nextme.json`을 바탕으로 했다.
- `frequent_1`(frequent_place, `venue_indoor`)을 추가해 `node_binding`으로 묶었다(nextme800, `nextme800:wearer`, node `frequent_1`).
- `meal_style.place_nodes.eat_out = "frequent_1"`로 두었다.
- `exception_rate = {slot: 0.5, day: 0.1}`.
- `person_id`, `seed`, `routine_strength`는 실행마다 바꿨다.
- G19에 따라 두 트리는 같은 월드 값을 쓴다. old의 `meal_style.weekend`에는 new의 `rest_day` 값을 그대로 넣었다(old 월드 파일의 원래 주말 값이 아니다).

### 규모

- 스케줄 단계: seed 1–200 × routine_strength {0, 0.5, 1} = 트리마다 600개 스케줄(16,800일).
- full run(schedule → preflight → fill → verify → manifest): seed 1–20 × routine_strength {0, 0.5, 1} = 트리마다 60개.
  - clip bank는 합성이다. `packages/mcp/test/helpers.ts`의 NextMe 10종 clip을 바탕으로 4종을 더했다.
    - `frequent_1`의 외식 clip 3종(아침, 점심, 저녁)
    - 회사 식사 clip 1종(점심과 저녁을 모두 회사에서 먹는 날용)
  - day group 32개와 클라이밍 take 20개를 메모리 catalog로 썼다.
  - 결과물(bank, 월드, timeline, manifest)은 저장소 밖 임시 디렉터리에 썼다. `data/`에는 아무것도 쓰지 않았다.

### 읽는 방법

- G7: 두 트리의 schedule과 manifest는 raw JSON으로 읽었다. 트리의 schema로 파싱하지 않았다.
- G19: jitter 키가 바뀌어 날짜별로 맞대는 비교는 의미가 없다. 그래서 분포를 비교했다.
- 단, `slot_exception`과 `day_exception`의 Bernoulli 키(`seed, "day", d, attempt, …`)와 평일 첫 donor 키는 두 트리가 같다. 그래서 이 항목들은 두 트리에서 수치가 거의 같게 나온다.

### 기준 검사

E5 경계, 평일 required slot 검사, E2 일관성은 항상 new 트리의 함수(`donorBoundViolations`, `resolveDayWindows` + `gapQuantile`, `realizedOf`)로 계산했다. 그래서 old 결과는 "rev3 기준이었다면 어땠을지"를 보여 준다.

### 실행 명령

저장소 루트에서 실행한다. `$BASE`(baseline worktree)와 `$OUT`(출력 디렉터리)은 둘 다 저장소 밖에 둔다. 이번에는 아래 scratchpad 경로를 썼다.

```bash
export PATH="/NHNHOME/langlab/zzuhyun/.node/current/bin:$PATH"
SP=/tmp/claude-3952/-NHNHOME-langlab-zzuhyun-video-benchmark/48bbfc44-42e0-4cc5-8a4f-1622eb5c7396/scratchpad
BASE="$SP/rev3-baseline"
OUT="$SP/rev3cmp"
git worktree add "$BASE" ea048f5
ln -s "$PWD/node_modules" "$BASE/node_modules"
npx tsx scripts/compare/rev3_compare.ts run --root "$BASE" --label old --data data --out "$OUT"
npx tsx scripts/compare/rev3_compare.ts run --root . --label new --data data --out "$OUT"
npx tsx scripts/compare/rev3_compare.ts report --old "$OUT/old.json" --new "$OUT/new.json"
rm "$BASE/node_modules" && git worktree remove "$BASE"
```

- 기본값은 `--seeds 200 --full-seeds 20 --rs 0,0.5,1 --slot 0.5 --day 0.1`이다.
- 실행 시간: old는 스케줄 약 1초와 full run 약 4초다. new는 ablation(§4) 4종 × 600개 스케줄이 더해져 스케줄 단계가 약 7초, full run이 약 4.5초다.
- new 트리를 두 번 돌리면 sweep, ablation, full run 결과와 manifest가 바이트까지 같다.

## 결과

### 1. 생성 실패

| 트리 | 스케줄 실패 | full run 실패 |
|---|---|---|
| old | 0/600 | 0/60 |
| new | 0/600 | 0/60 |

### 2. optional slot drop

#### 스케줄 단계 (`no_time`)

하루에 optional slot이 하나 이상 drop된 날의 비율이다.

| 트리 | 전체 | workday | hobby_day | home_day(leave_day 포함) |
|---|---|---|---|---|
| old | 10.2% (1719/16800) | 9.7% (1048/10766) | 12.1% (580/4800) | 7.4% (91/1234) |
| new | 9.5% (1601/16800) | 6.8% (735/10767) | 17.1% (821/4800) | 3.6% (45/1233) |

routine_strength별 전체 비율:

- old: 11.3%(0) / 10.2%(0.5) / 9.2%(1)
- new: 10.3% / 9.7% / 8.6%

role별 비율은 drop ÷ 시도다. 시도는 "남았거나 `no_time`으로 drop된 경우"를 센다.

| role | old | new |
|---|---|---|
| workday.lunch | 2.6% (282/10766) | 0.7% (71/10767) |
| workday.dinner | 3.0% (323/10766) | 필수 slot (drop 없음) |
| workday.home_evening | 9.8% (528/5382) | 6.9% (670/9675) |
| hobby_day.lunch | 1.9% (89/4800) | 1.2% (58/4800) |
| hobby_day.home_afternoon | 1.3% (64/4800) | 2.7% (129/4800) |
| hobby_day.dinner | 4.5% (217/4800) | 12.0% (575/4800) |
| hobby_day.home_evening | 9.4% (449/4800) | 9.4% (451/4800) |
| home_day.home_evening | 7.4% (91/1234) | 3.6% (45/1233) |
| 나머지(아침, home_day 점심/저녁 등) | 0% | 0% |

- new의 `workday.home_evening` 시도는 5382에서 9675로 늘었다. CR3에서 출현 확률을 config 값으로 바꿨기 때문이다(0.5 → `weekday_evening_home.slot_probability`).
- `shrunk`(E4)는 new에서도 0이다. 이 월드의 취미는 `weekly_freq` 2라서 토, 일에만 배정되고 평일 hobby slot이 생기지 않는다. 그래서 축소 단계가 쓰일 일이 없다. 이 설정으로는 축소 경로를 측정할 수 없다.
- 브리프는 이전 drop 비율을 "날의 약 17%"로 적었다. 이 비교 설정(외식 장소 바인딩, exception 0.5)에서 old는 10.2%였다. 17%는 다른 설정에서 잰 값으로 보인다.

#### 매칭 단계 (full run)

| 트리 | `dropped_optional` slot | 매칭 중 재생성된 날 | clip 없는 평일 저녁 |
|---|---|---|---|
| old | 0 (0/1680일) | 0/1680 | 4.3% (47/1083) |
| new | 0 (0/1680일) | 0/1680 | 0.0% (0/1083) |

합성 bank가 넉넉해서 두 트리 모두 매칭 단계 drop과 재생성이 0이다. 이 설정으로는 매칭 단계 차이를 볼 수 없다. old의 clip 없는 평일 저녁은 모두 스케줄 단계 drop에서 온 것이다.

### 3. 평일 저녁 (CR2)

| 트리 | 저녁 없는 평일 | 사유 |
|---|---|---|
| old | 3.0% (323/10766) | 모두 `no_time` drop |
| new | 0.0% (0/10767) | 없음 |

routine_strength별로는 old 3.1% / 3.0% / 2.8%, new는 모두 0이다.

### 4. 휴일 저녁 (알려진 문제)

| 트리 | hobby_day 저녁 drop | home_day 저녁 drop |
|---|---|---|
| old | 4.5% (217/4800) | 0 (0/1234) |
| new | 12.0% (575/4800). rs별 11.4% / 11.9% / 12.6% | 0 (0/1233) |

- CR2는 평일만 다룬다. hobby_day 저녁 drop은 rev3에서 오히려 늘었다(4.5% → 12.0%).
- 이 시기에 주말 donor 쪽은 두 가지가 함께 바뀌었다. 둘 다 원인 후보다.
  - CR8: sport가 없는 donor에게 같은 backoff 단계의 sport 시각을 채운다(E7).
  - CR4.5: 주말 pool에서 일한 행을 뺀다. 기상 하한도 수락 조건에 들어갔다(G6).
- 아래 표는 모두 `rev3_compare.ts`에서 나온다. 스크립트에 커밋된 집계와 ablation이다.

#### hobby slot 시작 시각별 저녁 drop

hobby_day의 hobby slot(필수, sport 앵커)이 시작하는 시각으로 날을 나눴다. 값은 저녁 drop ÷ 저녁 시도다.

| 출처 | <10:00 | 10–12 | 12–14 | 14–16 | ≥16:00 | 전체 |
|---|---|---|---|---|---|---|
| old | 0.0% (0/783) | 0.0% (0/3456) | 0.4% (1/232) | 17.6% (15/85) | 82.4% (201/244) | 4.5% (217/4800) |
| new | 0.0% (0/1642) | 0.0% (0/1638) | 4.5% (7/157) | 7.9% (22/277) | 50.3% (546/1086) | 12.0% (575/4800) |

hobby day가 시각대별로 차지하는 비율:

| 출처 | <10:00 | 10–12 | 12–14 | 14–16 | ≥16:00 |
|---|---|---|---|---|---|
| old | 16.3% | 72.0% | 4.8% | 1.8% | 5.1% |
| new | 34.2% | 34.1% | 3.3% | 5.8% | 22.6% |

저녁 drop은 거의 16시 이후에 시작하는 hobby에서만 생긴다. 이런 날의 비율이 5.1%에서 22.6%로 늘었다. old에서 10–12시가 72%를 차지한 것은 sport 기본값 10:45 때문이다.

#### 주말 donor의 sport 시각 (출처별)

schedule마다 주말 donor는 하나다(200 seed × 3 rs = 600). 출처는 세 가지다.

- own: donor 본인의 일지 값
- level: CR8 채움
- default: sport가 없어 템플릿 기본값 10:45를 쓴 경우

| 출처 | sport 출처 | schedule | p25 | p50 | p75 | hobby_day 저녁 drop |
|---|---|---|---|---|---|---|
| old | own | 28.5% (171/600) | 08:00 | 10:05 | 11:30 | 15.8% (216/1368) |
| old | default | 71.5% (429/600) | 10:45 | 10:45 | 10:45 | 0.0% (1/3432) |
| new | own | 25.0% (150/600) | 08:30 | 10:20 | 17:05 | 16.0% (192/1200) |
| new | level | 75.0% (450/600) | 09:00 | 10:15 | 14:20 | 10.6% (383/3600) |

- new에서는 채움이 모두 `level`이었다(`pool` 0건).
- 일지에 sport가 있는 donor는 old와 new 모두 약 16%가 저녁을 잃는다.
- old의 낮은 비율(4.5%)은 donor 72%가 10:45 기본값을 썼기 때문이다. 이 기본값에서는 저녁 drop이 사실상 없었다.

#### ablation (new 트리, 스케줄 단계, 200 seed × 3 rs)

스크립트 쪽에서 입력만 바꾸고 new 코드로 다시 생성했다.

- **replay**: 고른 donor 그대로 `generateDay` 루프를 다시 돌렸다(attempt 0, 실패하면 1..5). 검증용이다. 600개 schedule의 날이 `generateSchedule` 결과와 모두 같았다(불일치 0일). 그래서 아래 변형들은 같은 경로로 생성된 것이다.
- **fill_to_default**: CR8로 채운 주말 sport를 기본값 10:45로 되돌리고 날을 replay했다.
- **old_pool**: 주말 donor를 old pool(일한 행 포함)에서 뽑았다. old pool은 `ea048f5`의 `atus2025Source` 필터를 스크립트에서 재현했다.
- **old_pool+fill_to_default**: 위 둘을 함께 적용했다.

| 변형 | 실패 | hobby_day 저녁 drop | hobby_day drop 날 | workday drop 날 | home_day drop 날 |
|---|---|---|---|---|---|
| new (replay) | 0/600 | 12.0% (575/4800) | 17.1% (821/4800) | 6.8% (735/10767) | 3.6% (45/1233) |
| fill_to_default | 0/600 | 4.0% (192/4800) | 8.7% (418/4800) | 6.8% (735/10767) | 3.6% (45/1233) |
| old_pool | 0/600 | 12.6% (604/4800) | 17.4% (833/4800) | 6.8% (735/10767) | 1.4% (17/1233) |
| old_pool+fill_to_default | 0/600 | 2.9% (141/4800) | 6.8% (327/4800) | 6.8% (735/10767) | 1.4% (17/1233) |

결론:

- 늘어난 저녁 drop은 대부분 CR8의 sport 채움에서 온다. 채움만 되돌려도 12.0%가 4.0%로 내려가 old의 4.5% 수준이 된다.
- 주말 pool 변경(CR4.5)만 되돌려서는 줄지 않는다(12.6%). 둘을 함께 되돌리면 2.9%다.
- 기상 하한은 donor 수락 안에 있어 스크립트 쪽에서 끌 수 없었다. 그래서 따로 분리하지 못했다. 남은 차이(2.9% 대 4.5%)에는 이것과 수락 redraw가 섞여 있다.
- 다만 CR8이 잘못 동작한 것은 아니다. 채운 시각은 실제 응답자의 sport 시각 분포다(p75 14:20). 저녁이 사라지는 직접적인 이유는 늦게 시작하는 120분 hobby와 optional인 휴일 저녁의 조합이다.
- 진행 기록의 "취미일의 약 13–18%"와 같은 문제다. 이 설정에서는 12%였다.

### 5. 예외 (CR1, E1, E3)

| 트리 | slot rate (공통 정의) | slot rate (rev3 기록 E1) | slot share | R45 전 추첨 | R45로 제거 | no-op 비율 | day rate |
|---|---|---|---|---|---|---|---|
| old | 48.4% (7421/15347) | 측정 불가 | 측정 불가 | 50.2% (7819/15566) | 5.1% (398/7819) | 9.1% (677/7421) | 10.3% (1234/12000) |
| new | 50.2% (7519/14992) | 50.4% (7519/14917) | 29.5% (7519/25466) | 해당 없음 | 해당 없음 | 9.4% (705/7519) | 10.3% (1233/12000) |

각 열의 정의:

- **공통 정의**: 두 트리의 schedule에서 같은 방식으로 셌다. 적격일은 day exception이 없고, 카탈로그 항목 중 하나라도 맞는 날이다. 항목이 맞으려면 day type이 맞고, 교체할 role이 남아 있고, 장소를 쓸 수 있어야 한다. 비율은 slot exception이 있는 적격일 ÷ 적격일이다. old에는 재배치 가능성 검사가 없어서 new의 기록 E1보다 조금 넓은 정의다.
- **rev3 기록 E1**: new 날의 `slot_exception_eligible`로 셌다. 재배치 가능성까지 본다.
- **R45 전 추첨**: old 날을 old `decideDay`로 다시 계산한 값이다. 같은 attempt를 쓰고, old 자체의 적격 조건을 따른다. 추첨 직후에는 50.2%였는데 R45가 그중 5.1%를 지워 결과가 48.4%가 됐다.
- **no-op 비율**:
  - new: `routine_place`로 셌다.
  - old: `exception_rate.slot = 0`인 월드로 `decideDay`를 다시 돌려 routine 장소를 얻었다. rng 키가 이름별이라 나머지 결정은 같다.

rs별 수치:

- new 기록 E1: 세 rs 모두 50.4%.
- seed당 E1 평균: 0.505. 28일 월드 하나로 보면 0.185–0.76까지 퍼진다.
- old 공통 정의: 48.0% / 48.4% / 48.6%.
- day rate: 두 트리, 모든 rs에서 10.3%.

### 6. donor 단계 (CR5, E5)

| 트리 | donor | redraw 발생 | redraw 분포 0:1:2:3:4:5 | 최종 clamp/보정 | E5 위반인 채 사용 |
|---|---|---|---|---|---|
| old | 평일 | — (기능 없음) | — | — | 39.0% (234/600) |
| old | 주말 | — | — | — | 43.0% (258/600) |
| new | 평일 | 39.0% (234/600) | 366:119:41:33:12:29 | 2.5% (15/600) | 0.5% (3/600) |
| new | 주말 | 53.5% (321/600) | 279:126:105:33:24:33 | 4.5% (27/600) | 0.0% (0/600) |

"E5 위반인 채 사용"은 두 트리 모두 같은 기준 검사로 최종 donor 프로필을 측정한 값이다. old는 redraw 기능이 없다. 그래서 old에서 고른 donor 프로필에 new의 `donorBoundViolations`(lunch 45분, hobby 120분), new의 평일 required slot 검사(p90 gap, sleep − σ), 주말 기상 하한(평일 기상 − 120분)을 적용해 "rev3였다면 다시 뽑았을 donor"를 셌다.

- 위반 항목:
  - old 평일: breakfast 156, required_slots 75, lunch 24, dinner 18, work_end 15
  - old 주말: breakfast 231, dinner 24, sport 18
- new의 최종 보정 항목:
  - 평일: breakfast 12, work_end 3(G9 보정 포함)
  - 주말: breakfast 21, sport 9
- new 평일의 남은 위반 3건은 모두 seed 149다(rs 0, 0.5, 1에서 각 1건).
  - 이 donor는 redraw 5회 뒤에도 required slot 검사(p90 gap, sleep − σ)를 통과하지 못했다.
  - G9의 work_end 보정도 맞는 값을 찾지 못해 프로필이 그대로 남았다(adjustments 없음).
  - 그래도 날은 표본 gap으로 배치되어 스케줄 생성은 실패하지 않았다.
  - E5의 일반 경계(시각) 위반은 new에서 0건이다.
- 평일 첫 donor는 두 트리에서 같다(같은 pool, 같은 키). 그래서 old의 위반 비율 39.0%와 new의 redraw 비율 39.0%가 정확히 같다. 검산이 된다.
- 주말은 new pool이 일한 행을 빼고 기상 하한을 수락 조건에 넣었다. 첫 donor부터 달라서 두 수치를 바로 비교할 수 없다.
- 평일 수치는 rs별로 조금 다르다(40.5% / 38.5% / 38.0%). required slot 검사가 σ에 의존하기 때문이다.

### 7. day 단계 (CR5, E6, G12, G13)

| 트리 | redraw가 필요한 날 | redraw 수 분포 0:1:2:3+ | clamp한 날 | 기록 여부 |
|---|---|---|---|---|
| old | 25.6% (4302/16800) | — (기능 없음) | 25.6% (4302/16800) | 기록 없음 (조용한 clamp) |
| new | 14.0% (2351/16800) | 14449:1185:419:747 | 1.8% (297/16800) | `anchor_adjustments`에 모두 기록 |

routine_strength별:

| 트리 | rs 0 | rs 0.5 | rs 1 |
|---|---|---|---|
| old 위반 날 | 28.9% | 25.3% | 22.6% |
| new redraw 날 | 19.6% | 13.7% | 8.6% |
| new clamp 날 | 2.4% | 1.8% | 1.1% |

항목:

- old 위반: breakfast 2537, dinner 1013, lunch 670, work_end 254, wake 250, sport 121, sleep 2
- new clamp: breakfast 216, dinner 36, lunch 32, sport 10, wake 4, work_end 1

old 앵커를 어떻게 복원했는지:

- old의 day 앵커는 schedule에 기록되지 않는다. 그래서 old `dayAnchors`를 스크립트에 그대로 옮겨 다시 계산했다. 입력은 old의 `rngFor(seed, "day", d, attempt, "jitter", name)`, donor 프로필, 날의 attempt, old config의 σ와 sport 기본값이다.
- 복원한 앵커는 wake, sleep, work_start, work_end, breakfast, lunch, dinner, sport 전부다.
- 이 복원이 맞는지 세 가지로 확인했다.
  - 복원 후 old clamp를 적용한 wake/sleep이 기록된 `wake`/`sleep`과 16,800/16,800일 일치했다.
  - anchor가 있는 slot의 시작이 복원 앵커보다 이른 경우가 0건이었다. 61,341/69,282건은 정확히 같았고, 나머지는 앞 slot에 밀려 늦었다.
  - `work_pm`의 끝이 복원한 work_end와 10,766/10,766건 같았다.
- 위반은 G12에 따라 템플릿이 쓰는 앵커만 셌다. raw jitter 벡터 전체에 한 번에 적용했다.
- new는 의존 순서대로 하나씩 검사한다(G13). 그래서 old 수치는 "첫 추첨에서 경계를 넘은 날"이고, new의 "redraw가 필요한 날"과 같은 뜻이다.
- old에서는 위반이 곧 clamp라서 두 비율이 같다.

### 8. 구성 일관성 (E2, G15)

routine_strength별 schedule 평균이다. 칸마다 "전체 쌍 / same_reference"를 적었다.

| 트리 | 템플릿 | rs | optional_jaccard | order_similarity | start_diff(분) |
|---|---|---|---|---|---|
| new | workday | 0 | 0.861 / 0.859 | 0.954 / 0.953 | 48.0 / 47.9 |
| new | workday | 0.5 | 0.870 / 0.888 | 0.957 / 0.963 | 36.0 / 35.8 |
| new | workday | 1 | 0.886 / 0.965 | 0.962 / 0.988 | 23.4 / 23.1 |
| new | hobby_day | 0 | 0.857 / 0.857 | 0.899 / 0.899 | 48.5 / 48.5 |
| new | hobby_day | 0.5 | 0.878 / 0.878 | 0.915 / 0.915 | 36.3 / 36.3 |
| new | hobby_day | 1 | 0.952 / 0.952 | 0.971 / 0.971 | 23.0 / 23.0 |
| new | home_day | 0 | 0.914 / 0.914 | 0.938 / 0.938 | 53.7 / 53.7 |
| new | home_day | 0.5 | 0.918 / 0.918 | 0.941 / 0.941 | 41.1 / 41.1 |
| new | home_day | 1 | 0.990 / 0.990 | 0.993 / 0.993 | 23.3 / 23.3 |
| old | workday | 0 / 0.5 / 1 | 0.806→0.822→0.847 / 0.803→0.853→0.970 | 0.906→0.913→0.925 / 0.905→0.929→0.987 | 49.3→36.5→23.7 / 49.2→36.3→23.2 |

- new 평일의 same_reference는 다이얼을 따라간다. jaccard는 0.859→0.888→0.965, order는 0.953→0.963→0.988이다. 전체 쌍은 0.861→0.886으로 덜 움직인다. 평일 기준이 요일별이기 때문이다(G15).
- old 평일의 optional 집합에는 저녁이 들어간다(v1에서는 저녁이 optional). 그래서 old와 new의 평일 jaccard를 바로 비교할 수는 없다.
- hobby_day는 new가 조금 낮다. 저녁 drop이 늘었기 때문이다(§4).

### 9. no_video_spans tiling (E9)

- new full run의 manifest 60개, 1,680일 전부에서 확인했다. 확인한 조건은 clip 구간 ∪ `no_video_spans`가 [wake, sleep]을 빈틈도 겹침도 없이 덮는 것이다(허용 오차 0.0015초).
- 실패는 0건이다.
- old manifest에는 `no_video_spans`가 없어서 이 항목은 해당이 없다.

## 해석

1. **CR2(평일 저녁)는 달성했다.** 스케줄과 full run 모두 0이다. 평일 optional drop 날도 9.7%에서 6.8%로 줄었다(점심 2.6% → 0.7%, home_evening 9.8% → 6.9%).
2. **slot exception rate가 다이얼과 맞는다.** 공통 정의로 old 48.4%, new 50.2%다. new의 기록 E1은 50.4%다. old는 R45가 추첨된 예외의 5.1%를 지워 낮았다.
3. **조용한 clamp가 없어졌다.**
   - donor 단계에서는 old가 평일 39%, 주말 43%의 donor를 rev3 경계를 넘는 채로 썼다. new는 redraw 뒤 2.5%와 4.5%만 clamp하고 모두 기록한다. 최종 프로필을 측정하면 new에서 남은 위반은 평일 0.5%(seed 149의 required slot 검사)뿐이다.
   - day 단계에서는 old가 25.6%의 날을 기록 없이 clamp했다. new는 1.8%만 clamp하고 기록한다.
4. **전체 drop 비율은 거의 그대로다(10.2% → 9.5%).** 평일은 줄었지만 hobby_day 저녁 drop이 4.5%에서 12.0%로 늘었다.
   - 시각대별 집계를 보면 drop은 16시 이후에 시작하는 hobby에 몰려 있고, 그런 날이 5.1%에서 22.6%로 늘었다.
   - ablation에서 CR8 sport 채움만 기본값으로 되돌리면 4.0%가 된다. 주말 pool 변경(CR4.5)만 되돌리면 12.6%로 줄지 않는다.
   - 그래서 주된 원인은 CR8 채움이다. 정확히는 old에서 donor 72%가 쓰던 10:45 기본값이 사라져 늦은 sport 시각이 드러난 것이다.
   - 휴일 저녁을 지키려면 CR2와 같은 처리(휴일 저녁 required, 또는 hobby_day reduce_order)가 따로 필요하다. 이건 사용자가 정할 일이다.
5. **no-op exception은 이 월드에서 9.4%다.** 진행 기록의 약 16%는 다른 설정에서 잰 값으로 보인다. 분석에서 빼려면 `slot_exception_noop_share`를 쓴다.

## 측정하지 못한 것

- **old의 기록 E1 비율과 slot share**: old에는 slot exception 후보와 재배치 가능성 개념이 없다. 그래서 공통 정의와 R45 전 추첨 비율로 대신했다.
- **old의 redraw 횟수**: old에는 redraw가 없다. 대신 new 경계를 old donor와 old 앵커에 적용해 "다시 뽑았을 것"을 셌다.
- **old의 day-level 위반 판정 방식**: new는 의존 순서대로 하나씩 검사하고 old 값은 raw 벡터를 한 번에 검사했다. 그래서 dependent 항목(lunch 등)은 수가 조금 다를 수 있다. 예를 들어 old lunch는 E5 위반이 670건인데 실제 old clamp는 652건이었다.
- **E4 shrink**: 이 월드에는 평일 hobby가 없어서 두 트리 모두 0이다.
- **매칭 단계 차이**: 합성 bank가 넉넉해 `dropped_optional`과 재생성이 두 트리 모두 0이다. 실제 bank에서의 차이는 이 비교로 알 수 없다.
- **old의 tiling**: old에는 `no_video_spans`가 없다.
- **기상 하한(G6)의 영향**: donor 수락 안에 있어 ablation으로 분리하지 못했다.
- **old의 E2 평일 값**: 템플릿 v1에서는 저녁이 optional이라 jaccard 정의가 다르다. 참고용으로만 적었다.
