# NextMe 28일 실제 프레임 렌더러 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 블록 방식 28일 manifest를 실제 NextMe, Ego4D, YouTube 프레임으로 하루에 영상 하나(`Dnn.mp4`)씩 렌더링한다.

**Architecture:** Python 스크립트 하나(`scripts/render_frames.py`)다. 하루마다 정수 합성 초 `t`에 프레임 1장을 정하고, 원본 프레임을 준비한다. 준비는 두 갈래다. NextMe는 fetcher 캐시의 1024 JPEG를 쓴다. 영상은 ffmpeg가 `select`로 뽑고 회전, 자르기, 축소까지 한다. 그다음 Pillow/numpy로 원 가리기, 휘기, 타임스탬프를 하고 rawvideo로 ffmpeg(libx264)에 넘긴다. 날짜 하나가 작업 단위이고, 여러 날은 프로세스로 동시에 만든다.

**Tech Stack:** Python 3.12, Pillow 12, numpy 2, ffmpeg/ffprobe 6.1(libx264), pytest. 새 의존성은 없다.

**Spec:** `docs/superpowers/specs/2026-11-13-nextme-frame-render-design.md`(2026-11-14 승인). 옛 설계 `2026-11-07-video-render-design.md` §5(영상 시각 맞추기, 회전)와 §6(인코딩)을 참조한다.

## Global Constraints

- 코드, 메시지, 테스트는 영어로 쓴다. 테스트 fixture는 합성 데이터만 쓴다(실제 NextMe/Ego4D/YouTube 파일을 테스트에서 읽지 않는다).
- 커밋은 사용자가 요청할 때만 한다(CLAUDE.md). 아래의 Commit 단계는 사용자가 이 계획의 커밋을 허락한 경우에만 실행한다.
- 출력: 1024×1024, 1 fps, 오디오 없음, H.264. 인코딩은 정확히 `-c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p -g 60 -movflags +faststart -an`이다(crf는 옵션으로 바꿀 수 있음).
- 1일차 날짜는 `2025-04-07`(월)이다. 타임스탬프 형식은 `YYYY-MM-DD HH:MM:SS`이고 시간대는 없다.
- 타임스탬프 글꼴: `/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf`, 흰색, 숫자 높이 20px, x = 10, 아래 여백 10px, 상자 안쪽 여백 `max(3, 숫자높이/3)`을 올림한 값(20px이면 7)이다.
- 원 가리기: 중심 `(494, 507)`, 반지름 500px까지 그대로, 500→560px에서 선형으로 검게, 560px 밖은 검정이다.
- NextMe: `n = round(src)`이고 0.5는 작은 쪽이다. 없으면 ±2 안의 가장 가까운 프레임을 쓴다(같은 거리면 작은 번호). 녹화 끝(index에 `{"end": true}`가 있고 `n > 마지막 번호`)이면 5초까지 마지막 프레임을 반복한다. 넘으면 그날 오류다.
- 영상: 프레임 `k`의 원본 시각은 `pts_k − pts_0`이다. 가장 가까운 프레임을 쓰고, 같은 거리면 앞 프레임이다. 마지막 프레임을 지난 정도가 `1 / media.fps` 이하면 clamp, 넘으면 그날 오류다. `-ss`와 `fps=1`은 함께 쓰지 않는다.
- 결과 위치 기본값은 `data/renders/nextme-p00/benchmark/`이고, 날마다 `Dnn.mp4`, `Dnn.frames.jsonl`, `Dnn.render.json`을 만든다(`nn` = 두 자리 day_index).
- 원본 경로: NextMe `data/media/nextme/frames/<video_id>/*__frame_NNNNN.jpg`, index `data/media/nextme/index/<video_id>.jsonl`, Ego4D `data/media/ego4d/v2/full_scale/<video_id>.mp4`, YouTube `data/media/youtube/<video_id>.mp4`.
- 토큰이나 키는 쓰지 않는다. 렌더러는 네트워크를 쓰지 않는다.

## Review Focus

1. **NextMe 폴더가 아예 없는 날**(tar를 아직 안 받음): 그날만 건너뛰고 "NextMe `<vid>` N frames missing"으로 모아 보여 준다. 다른 날은 만든다. → Task 6 `test_day_with_missing_nextme_folder_is_skipped`
2. **인코딩 도중 끊김(Ctrl-C, ffmpeg 실패)**: 반쯤 만든 `Dnn.mp4`나 `Dnn.render.json`이 남지 않는다. 다음 실행은 그날을 다시 만든다. → Task 6 `test_failed_day_leaves_no_outputs`
3. **설정만 바꾼 재실행**(회전, `warp`, crf): 이전에 만든 날을 건너뛰지 않고 다시 만든다. → Task 6 `test_rerun_skips_same_settings_and_redoes_changed`
4. **영상 프레임 경계의 동률과 `pts_0 ≠ 0`**: 앞 프레임을 고르고 `pts_0`만큼 정확히 옮긴다. → Task 3 `test_pick_video_tie_takes_earlier`, Task 4 `test_extract_with_nonzero_start_time`
5. **같은 원본 프레임을 두 번 고르는 경우**(clamp로 마지막 프레임 반복): ffmpeg `select`는 같은 프레임을 한 번만 내보낸다. 그래도 순서대로 다시 채워 장 수가 맞는다. → Task 4 `test_extract_repeats_duplicate_indices`

---

### Task 1: 프레임 목록과 타임스탬프 글자

**Files:**
- Create: `scripts/render_frames.py`
- Create: `scripts/test_render_frames.py`

**Interfaces:**
- Produces:
  - `START_DATE = datetime.date(2025, 4, 7)`
  - `class DayError(Exception)`: 그날만 멈추는 오류
  - `@dataclass(frozen=True) class Pick: t: int; clip: int; src: float`. `clip`은 `day["clips"]`의 원래 인덱스다.
  - `day_picks(day: dict) -> list[Pick]`: 클립을 `synthetic_start_s` 순서로 놓는다. 클립마다 `ceil(a) ≤ t < b`인 정수 `t`를 만들고, `src = source_start_s + (t - a)`다. 결과는 `t` 오름차순이다.
  - `stamp_text(day_index: int, t: int) -> str`: `base = (day_index-1)*86400`이고, `t - base`가 86400 이상이면 날짜를 하루씩 넘긴다.

- [ ] **Step 1: Write the failing tests**

```python
def clip(a, b, s0, ds="nextme800", vid="v", slot="d1-s1"):
    return {"source": {"dataset": ds, "video_id": vid, "media": {"kind": "frames"}}, "video_id": vid,
            "source_start_s": s0, "source_end_s": s0 + (b - a), "synthetic_start_s": a, "synthetic_end_s": b,
            "slot_id": slot}

def test_day_picks_one_per_integer_second_and_skips_gaps():
    day = {"day_index": 1, "clips": [clip(20, 22, 100), clip(10.5, 13.2, 7)], "no_video_spans": []}
    p = day_picks(day)
    assert [x.t for x in p] == [11, 12, 13, 20, 21]
    assert [x.clip for x in p] == [1, 1, 1, 0, 0]
    assert p[0].src == pytest.approx(7.5) and p[3].src == 100

def test_stamp_text_dates_and_midnight():
    assert stamp_text(1, 7 * 3600 + 48 * 60 + 12) == "2025-04-07 07:48:12"
    assert stamp_text(28, 27 * 86400 + 59) == "2025-05-04 00:00:59"
    assert stamp_text(1, 86400 + 5) == "2025-04-08 00:00:05"
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3 -m pytest scripts/test_render_frames.py -q`
Expected: FAIL. `ImportError`가 나야 한다(`render_frames`가 아직 없음).

- [ ] **Step 3: Implement `Pick`, `DayError`, `START_DATE`, `day_picks`, `stamp_text` in `scripts/render_frames.py`**

모듈 docstring에 사용법(§8의 명령)과 spec 경로를 적는다. 테스트 파일은 fetcher 테스트처럼 `sys.path.insert(0, str(Path(__file__).parent))`로 import한다.

- [ ] **Step 4: Run to verify they pass**

Run: `python3 -m pytest scripts/test_render_frames.py -q`
Expected: 2 passed

- [ ] **Step 5: Commit** (허락된 경우)

```bash
git add scripts/render_frames.py scripts/test_render_frames.py
git commit -m "feat(render): frame list and timestamp text for real-frame renderer"
```

### Task 2: NextMe 프레임 고르기

**Files:**
- Modify: `scripts/render_frames.py`
- Modify: `scripts/test_render_frames.py`

**Interfaces:**
- Consumes: `DayError`(Task 1). fetcher의 `load_index(path) -> (known: dict[int, dict], resume: int, ended: bool)`와 `SAVED_RE`(`from fetch_nextme_frames import load_index, SAVED_RE`)
- Produces:
  - `nextme_frames(frames_dir: Path) -> dict[int, Path]`: 숨김 파일(`.`으로 시작)은 뺀다.
  - `nextme_last_frame(index_path: Path) -> int | None`: index가 끝까지 훑은 경우(`ended`)에만 가장 큰 프레임 번호를 돌려준다. 아니면 `None`이다.
  - `pick_nextme(src: float, frames: Collection[int], last: int | None) -> tuple[int, str]`: 두 번째 값은 `"exact" | "substituted" | "clamped"`다. 고를 수 없으면 `DayError`를 낸다.
    - 순서: (1) `n`이 있으면 exact다. (2) `last`가 있고 `n > last`이면, `n - last ≤ 5`이고 `last`가 디스크에 있을 때 clamped다. 아니면 오류다. (3) 그 밖에는 ±2 대체이고, 실패하면 오류다.
  - `NEXTME_SUBSTITUTE_S = 2`, `NEXTME_END_CLAMP_S = 5`

- [ ] **Step 1: Write the failing tests**

```python
def test_pick_nextme_rounds_half_down():
    assert pick_nextme(4.5, {4, 5}, None) == (4, "exact")
    assert pick_nextme(4.51, {4, 5}, None) == (5, "exact")

def test_pick_nextme_substitutes_within_two_seconds_smaller_on_tie():
    assert pick_nextme(10, {8, 12}, None) == (8, "substituted")
    with pytest.raises(DayError):
        pick_nextme(10, {7, 13}, None)

def test_pick_nextme_holds_last_frame_up_to_five_seconds_at_recording_end():
    assert pick_nextme(9840, set(range(9800, 9837)), 9836) == (9836, "clamped")
    assert pick_nextme(9841, set(range(9800, 9837)), 9836) == (9836, "clamped")
    with pytest.raises(DayError):
        pick_nextme(9842, set(range(9800, 9837)), 9836)

def test_end_clamp_needs_a_finished_walk(tmp_path):
    idx = tmp_path / "v.jsonl"
    idx.write_text('{"n": 0, "name": "a__frame_00000.jpg", "data": 0, "size": 1, "next": 512}\n'
                   '{"n": 1, "name": "a__frame_00001.jpg", "data": 1024, "size": 1, "next": 2048}\n')
    assert nextme_last_frame(idx) is None
    idx.write_text(idx.read_text() + '{"end": true}\n')
    assert nextme_last_frame(idx) == 1
    with pytest.raises(DayError):
        pick_nextme(5, {0, 1}, None)  # unknown end: only the 2 s rule

def test_nextme_frames_reads_numbers_and_ignores_temp_files(tmp_path):
    for name in ["x__frame_00003.jpg", ".x__frame_00004.jpg.tmp", "x__frame_00012.jpg"]:
        (tmp_path / name).write_bytes(b"j")
    assert sorted(nextme_frames(tmp_path)) == [3, 12]
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3 -m pytest scripts/test_render_frames.py -q -k nextme`
Expected: FAIL (`pick_nextme`이 정의되지 않음)

- [ ] **Step 3: Implement `nextme_frames`, `nextme_last_frame`, `pick_nextme`**

반올림은 `math.ceil(src - 0.5)`다(0.5는 작은 쪽으로 간다).

- [ ] **Step 4: Run to verify they pass**

Run: `python3 -m pytest scripts/test_render_frames.py -q`
Expected: 7 passed

- [ ] **Step 5: Commit** (허락된 경우)

```bash
git add scripts/render_frames.py scripts/test_render_frames.py
git commit -m "feat(render): NextMe frame picking with 2 s substitution and 5 s end hold"
```

### Task 3: 영상 정보 읽기와 프레임 고르기

**Files:**
- Modify: `scripts/render_frames.py`
- Modify: `scripts/test_render_frames.py`

**Interfaces:**
- Consumes: `DayError`
- Produces:
  - `@dataclass class VideoInfo: times: list[float]; pts0: float; width: int; height: int; rotation: int`
    - `times`: 패킷 pts를 오름차순으로 놓고 `pts0`을 뺀 원본 시각이다.
    - `rotation`: display matrix 회전(0/90/180/270)이다. 없으면 0이다.
  - `probe_video(path: Path, cache_dir: Path | None = None) -> VideoInfo`: ffprobe로 읽는다.
    - 패킷 pts는 `-select_streams v:0 -show_entries packet=pts_time -of csv=p=0`로 읽는다.
    - 스트림 정보는 `-show_entries stream=start_time,width,height:stream_side_data=rotation`로 읽는다.
    - 결과를 `cache_dir/<file name>.<size>-<mtime_ns>.json`에 캐시한다. Ego4D 한 파일에 패킷이 수백만 개라 다시 읽지 않으려는 것이다.
  - `pick_video(src: float, times: list[float], fps: float) -> tuple[int, bool]`: (프레임 인덱스, clamped)를 돌려준다. `bisect`로 가장 가까운 프레임을 찾고, 같은 거리면 앞 프레임이다. `src > times[-1]`이면 `src - times[-1] ≤ 1/fps`일 때 clamp하고, 아니면 `DayError`를 낸다.
- 테스트 helper(테스트 파일): `make_video(path, n, fps=30, size=(64, 48), rotate=None, start=0.0)`
  - `ffmpeg -f lavfi -i "nullsrc=s=WxH:r=fps,geq=r='N*2':g='128':b='0'"`로 프레임 번호를 빨강 채널에 넣는다.
  - `-frames:v n`, `-c:v libx264 -qp 0 -pix_fmt yuv444p`를 쓴다.
  - `rotate`이면 입력 옵션 `-display_rotation:v:0 <deg>`를 붙인다(ffmpeg 6.1). `start`이면 `-output_ts_offset <start>`를 붙인다.
  - `frame_number(rgb_pixel) -> int`는 `round(r / 2)`다.

- [ ] **Step 1: Write the failing tests**

```python
def test_pick_video_nearest_and_tie_takes_earlier():
    times = [i / 30 for i in range(90)]
    assert pick_video(1.0, times, 30) == (30, False)
    assert pick_video(1.0 + 1 / 60, times, 30) == (30, False)   # exact tie -> earlier

def test_pick_video_clamps_within_one_frame_and_errors_beyond():
    times = [i / 30 for i in range(90)]
    assert pick_video(times[-1] + 1 / 30, times, 30) == (89, True)
    with pytest.raises(DayError):
        pick_video(times[-1] + 2 / 30, times, 30)

def test_probe_video_reads_times_start_and_rotation(tmp_path):
    v = make_video(tmp_path / "a.mp4", 45, start=2.0, rotate=90)
    info = probe_video(v, cache_dir=tmp_path / "cache")
    assert len(info.times) == 45 and info.times[0] == 0 and info.times[1] == pytest.approx(1 / 30, abs=1e-3)
    assert info.pts0 == pytest.approx(2.0, abs=0.05)
    assert info.rotation in (90, 270)  # sign convention of the display matrix; record as read
    assert probe_video(v, cache_dir=tmp_path / "cache") == info and any((tmp_path / "cache").iterdir())
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3 -m pytest scripts/test_render_frames.py -q -k "video"`
Expected: FAIL (`pick_video`가 정의되지 않음)

- [ ] **Step 3: Implement `VideoInfo`, `probe_video`, `pick_video`**

`rotation`은 ffprobe가 준 값(예: -90)을 `% 360`으로 정규화해 기록한다.

- [ ] **Step 4: Run to verify they pass**

Run: `python3 -m pytest scripts/test_render_frames.py -q`
Expected: 10 passed

- [ ] **Step 5: Commit** (허락된 경우)

```bash
git add scripts/render_frames.py scripts/test_render_frames.py
git commit -m "feat(render): probe video packet times and pick nearest frames"
```

### Task 4: 영상 프레임 뽑기(회전, 가운데 자르기, 1024 축소)

**Files:**
- Modify: `scripts/render_frames.py`
- Modify: `scripts/test_render_frames.py`

**Interfaces:**
- Consumes: `VideoInfo`, `probe_video`(Task 3), `DayError`
- Produces:
  - `SELECT_BATCH = 120`
  - `extract_video_frames(path: Path, info: VideoInfo, indices: list[int], extra_rotation: int = 0, size: int = 1024) -> Iterator[PIL.Image.Image]`: `indices` 순서대로 RGB `size×size` 이미지를 하나씩 내보낸다. 반복된 인덱스도 그만큼 내보낸다.
    - 같은 인덱스를 빼고 오름차순으로 놓은 뒤 `SELECT_BATCH`개씩 나눈다. 묶음마다 ffmpeg를 한 번 실행한다.
      ```
      ffmpeg -nostdin -v error -ss <max(0, first_abs - 1)> -copyts -i <path>
             -vf "select='<expr>',<rot>,crop='min(iw,ih)':'min(iw,ih)',scale=<size>:<size>:flags=lanczos"
             -fps_mode passthrough -f rawvideo -pix_fmt rgb24 -
      ```
    - `first_abs`는 묶음 첫 프레임의 절대 pts(`times[i] + pts0`)다.
    - `<expr>`는 `lt(abs(t-<abs_pts>),<tol>)`를 `+`로 이은 것이다. `tol = 0.25 * min(프레임 간격)`이고, 절대 pts는 소수 6자리로 쓴다.
    - `<rot>`: 0이면 빼고, 90은 `transpose=1`, 180은 `hflip,vflip`, 270은 `transpose=2`다. 메타데이터 회전은 ffmpeg 기본 autorotate가 먼저 적용한다.
    - 묶음에서 나온 장 수가 고유 인덱스 수와 다르면 `DayError`를 낸다. ffmpeg stderr 끝 20줄을 메시지에 넣는다.

- [ ] **Step 1: Write the failing tests**

```python
def test_extract_exact_frames_in_order(tmp_path):
    v = make_video(tmp_path / "a.mp4", 300)
    info = probe_video(v)
    imgs = list(extract_video_frames(v, info, [0, 30, 31, 299], size=32))
    assert [frame_number(im.getpixel((16, 16))) for im in imgs] == [0, 30, 31, 299]
    assert all(im.size == (32, 32) for im in imgs)

def test_extract_with_nonzero_start_time(tmp_path):
    v = make_video(tmp_path / "a.mp4", 90, start=1.5)
    info = probe_video(v)
    assert [frame_number(im.getpixel((16, 16))) for im in extract_video_frames(v, info, [45], size=32)] == [45]

def test_extract_repeats_duplicate_indices(tmp_path):
    v = make_video(tmp_path / "a.mp4", 60)
    imgs = list(extract_video_frames(v, probe_video(v), [59, 59, 59], size=32))
    assert [frame_number(im.getpixel((16, 16))) for im in imgs] == [59, 59, 59]

def test_extract_center_crops_and_rotates(tmp_path):
    # 64x48 landscape -> 48x48 centre crop.
    v = make_video(tmp_path / "a.mp4", 10, size=(64, 48))
    im = next(extract_video_frames(v, probe_video(v), [0], size=48))
    assert im.size == (48, 48)
    # Portrait via display matrix: output is still square and upright per ffmpeg autorotate.
    p = make_video(tmp_path / "p.mp4", 10, size=(64, 48), rotate=90)
    assert next(extract_video_frames(p, probe_video(p), [0], size=48)).size == (48, 48)

def test_extra_rotation_turns_the_picture(tmp_path):
    v = make_marker_video(tmp_path / "m.mp4")  # 48x48, white square in the top-left corner
    up = next(extract_video_frames(v, probe_video(v), [0], size=48))
    turned = next(extract_video_frames(v, probe_video(v), [0], extra_rotation=90, size=48))
    assert up.getpixel((4, 4))[0] > 200 and turned.getpixel((43, 4))[0] > 200
```

`make_marker_video`는 `color=black:s=48x48`에 `drawbox=x=0:y=0:w=10:h=10:color=white:t=fill`을 얹어 10프레임을 만든다. `transpose=1`(시계 방향 90°)이면 왼쪽 위가 오른쪽 위로 간다.

- [ ] **Step 2: Run to verify they fail**

Run: `python3 -m pytest scripts/test_render_frames.py -q -k extract`
Expected: FAIL (`extract_video_frames`가 정의되지 않음)

- [ ] **Step 3: Implement `extract_video_frames`**

stdout을 `size*size*3` 바이트씩 읽어 `Image.frombytes("RGB", (size, size), buf)`로 만든다. 고유 인덱스의 이미지를 dict에 모았다가 `indices` 순서대로 내보낸다. 묶음 단위로 내보내야 메모리가 묶음 크기로 묶인다.

- [ ] **Step 4: Run to verify they pass**

Run: `python3 -m pytest scripts/test_render_frames.py -q`
Expected: 15 passed

- [ ] **Step 5: Commit** (허락된 경우)

```bash
git add scripts/render_frames.py scripts/test_render_frames.py
git commit -m "feat(render): extract exact video frames with select, rotate, crop and scale"
```

### Task 5: 원 가리기, 어안 휘기, 타임스탬프 그리기

**Files:**
- Modify: `scripts/render_frames.py`
- Modify: `scripts/test_render_frames.py`

**Interfaces:**
- Produces:
  - `CIRCLE = {"center": (494, 507), "r_in": 500, "r_out": 560}`
  - `FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf")`
  - `OLD_TEXT_BOX = (0, 995, 200, 1024)`
  - `circle_mask(size: int = 1024, center=(494, 507), r_in: float = 500, r_out: float = 560) -> Image.Image`: 모드 `"L"`이다. 안은 255, 띠에서는 선형, 밖은 0이다.
  - `apply_circle(img: Image.Image, mask: Image.Image) -> Image.Image`: 마스크 밖을 검게 한다(`Image.composite(img, black, mask)`).
  - `fisheye_warp(img: Image.Image, strength: float, center=(494, 507), radius: float = 500) -> Image.Image`: `strength == 0`이면 같은 객체를 그대로 돌려준다.
    - 출력 화소의 정규화 반지름이 `u = r / radius`이면, 원본에서 `u_s = u * (1 + k*u²) / (1 + k)` 위치를 bilinear로 읽는다(`k = strength`). 가운데는 커지고 테두리(`u = 1`)는 고정된다.
    - numpy로 좌표를 계산하고, 화면 밖 좌표는 검정이다.
  - `class Stamper: __init__(self, font_path: Path = FONT_PATH, digit_px: int = 20, size: int = 1024)`. 속성은 `font`, `pad`, `box`(덮을 상자), 메서드는 `draw(img, text) -> Image.Image`(복사본에 그림)다.
    - 글꼴 크기는 `"0"`의 잉크 높이가 `digit_px`에 가장 가까운 정수 크기다.
    - `pad = ceil(max(3, digit_px / 3))`다.
    - 숫자 잉크의 왼쪽은 x = 10, 아래는 `size - 10`이다.
    - `box`는 `OLD_TEXT_BOX`와 (잉크 상자 + pad)의 합집합이고, 화면 안으로 자른다. 상자를 검게 칠한 뒤 흰 글자를 쓴다.
    - 글꼴 파일이 없으면 `FileNotFoundError`를 낸다.

- [ ] **Step 1: Write the failing tests**

```python
def test_circle_mask_values():
    m = circle_mask()
    assert m.getpixel((494, 507)) == 255 and m.getpixel((494 + 499, 507)) == 255
    assert 100 < m.getpixel((494 + 375, 507 + 375)) < 155   # r = 530.3, halfway through the 500-560 band
    assert m.getpixel((0, 0)) == 0 and m.getpixel((1023, 1023)) == 0

def test_apply_circle_blackens_outside():
    red = Image.new("RGB", (1024, 1024), (255, 0, 0))
    out = apply_circle(red, circle_mask())
    assert out.getpixel((494, 507)) == (255, 0, 0) and out.getpixel((5, 5)) == (0, 0, 0)

def test_fisheye_zero_is_identity_and_positive_magnifies_centre():
    img = Image.new("RGB", (1024, 1024), (0, 0, 0))
    ImageDraw.Draw(img).ellipse((494 - 50, 507 - 50, 494 + 50, 507 + 50), fill=(255, 255, 255))
    assert fisheye_warp(img, 0) is img
    w = fisheye_warp(img, 0.3)
    assert w.getpixel((494 + 60, 507))[0] > 200   # the disc grew past r = 50
    assert w.getpixel((494 + 499, 507)) == img.getpixel((494 + 499, 507))  # rim fixed

def test_stamp_covers_old_text_and_sits_bottom_left():
    s = Stamper()
    red = Image.new("RGB", (1024, 1024), (255, 0, 0))
    out = s.draw(red, "2025-04-07 07:48:12")
    region = out.crop(OLD_TEXT_BOX)
    assert all(px != (255, 0, 0) for px in region.getdata())     # no original pixel survives
    white = [(x, y) for y in range(960, 1024) for x in range(0, 400) if out.getpixel((x, y)) == (255, 255, 255)]
    assert white and min(x for x, _ in white) >= 9 and max(y for _, y in white) <= 1014
    assert s.pad == 7 and s.box[0] == 0 and s.box[3] == 1024

def test_stamper_needs_the_font(tmp_path):
    with pytest.raises(FileNotFoundError):
        Stamper(font_path=tmp_path / "missing.ttf")
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3 -m pytest scripts/test_render_frames.py -q -k "circle or fisheye or stamp"`
Expected: FAIL (`circle_mask`가 정의되지 않음)

- [ ] **Step 3: Implement `circle_mask`, `apply_circle`, `fisheye_warp`, `Stamper`**

- [ ] **Step 4: Run to verify they pass**

Run: `python3 -m pytest scripts/test_render_frames.py -q`
Expected: 20 passed

- [ ] **Step 5: Commit** (허락된 경우)

```bash
git add scripts/render_frames.py scripts/test_render_frames.py
git commit -m "feat(render): circle mask, fisheye warp and timestamp stamp"
```

### Task 6: 하루 렌더링(원본 확인, 프레임 조립, 인코딩, 리포트, 이어 하기)

**Files:**
- Modify: `scripts/render_frames.py`
- Modify: `scripts/test_render_frames.py`

**Interfaces:**
- Consumes: Task 1–5의 모든 것
- Produces:
  - `RENDER_VERSION = 1`
  - `@dataclass(frozen=True) class RenderConfig`:
    - `media_root: Path = Path("data/media")`, `warp: float = 0.0`, `rotation: dict = {"ego4d": 0, "youtube": 0}`(`field(default_factory=…)`), `crf: int = 18`, `digit_px: int = 20`, `font_path: Path = FONT_PATH`, `size: int = 1024`
  - `ENCODE_ARGS(crf) -> list[str]`: `["-an", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p", "-g", "60", "-movflags", "+faststart"]`
  - `settings_key(manifest_info: dict, cfg: RenderConfig) -> dict`:
    - `manifest_info`는 `{"path", "id", "revision", "created_at"}`다.
    - 결과에는 `RENDER_VERSION`, `START_DATE`, `digit_px`, 글꼴 파일 이름, `CIRCLE`, `warp`, `rotation`, 인코딩 인자가 들어간다.
  - `check_day(day: dict, cfg: RenderConfig) -> list[str]`: 모자란 원본을 사람이 읽을 문장으로 돌려준다. 빈 목록이면 만들 수 있다.
    - NextMe는 클립마다 모든 pick에 `pick_nextme`을 시도해, 실패한 장 수를 `"NextMe <vid>: N frames missing"`으로 적는다.
    - 영상은 파일이 있는지만 본다: `"Ego4D <vid>: file missing"`, `"YouTube <vid>: file missing"`.
  - `render_day(manifest_info: dict, day: dict, cfg: RenderConfig, out_dir: Path) -> dict`: 성공하면 report dict를 돌려주고, 실패하면 `DayError`를 낸다.
  - `day_is_done(out_dir: Path, day_index: int, key: dict) -> bool`: `Dnn.render.json`이 있고 그 안의 `"settings"`가 `key`와 같으면 True다.
- 쓰기 순서(원자적):
  1. 시작할 때 `.Dnn.tmp-*` 남은 파일을 지운다.
  2. `.Dnn.tmp-<pid>.mp4`, `.Dnn.tmp-<pid>.frames.jsonl`에 쓴다.
  3. 성공하면 mp4 → frames.jsonl → render.json 순서로 `os.replace`한다. render.json은 tmp에 쓴 뒤 마지막에 replace하므로, render.json이 완료 표시가 된다.
  4. 실패하거나 예외(KeyboardInterrupt 포함)가 나면 tmp를 지우고, ffmpeg 프로세스를 kill한 뒤 다시 raise한다.
- 인코더: `ffmpeg -nostdin -v error -y -f rawvideo -pix_fmt rgb24 -s <size>x<size> -r 1 -i - <ENCODE_ARGS> <tmp.mp4>`, stdin으로 프레임을 넘긴다.
- 프레임 준비:
  - NextMe: 캐시 JPEG를 열어 `convert("RGB")`한다. 크기가 `size`가 아니면 `DayError`다. 원 가리기와 휘기는 하지 않는다.
  - 영상: 클립마다 `pick_video` → `extract_video_frames(..., extra_rotation=cfg.rotation[dataset])` → `fisheye_warp` → `apply_circle` 순서다.
  - 공통: 마지막에 `Stamper.draw`다.
  - 영상 원본 경로는 위 Global Constraints대로다. probe 캐시는 `out_dir/.probe/`에 둔다.
- `Dnn.frames.jsonl` 한 줄: `{"t", "stamp", "dataset", "video_id", "slot_id", "kind"}`에 더해 NextMe는 `"frame": n`, 영상은 `"pts": 절대 pts`를 넣는다.
  - `kind`는 NextMe가 `exact | substituted | clamped`, 영상이 `nearest | clamped`다.
- `Dnn.render.json` 키:
  - 장 수와 출처: `day_index`, `frames`, `clip_frames`(`[{"clip": i, "video_id", "frames"}]`), `substitutions`(`[{"video_id", "src", "frame"}]`), `clamped`(`[{"dataset", "video_id", "src", "used"}]`)
  - 영상 정보: `videos`(`{video_id: {"pts0", "rotation_metadata", "rotation_extra"}}`)
  - 설정: `circle`, `warp`, `stamp`(`{"font", "digit_px", "box"}`), `encode`, `manifest`, `settings`(= `settings_key`), `seconds`
- 테스트 helper: `make_nextme(root, vid, frames: Iterable[int], ended: bool, last=None)`
  - `root/nextme/frames/<vid>/x__frame_NNNNN.jpg`에 1024 단색 JPEG를 만든다. 색 = `(n*2 % 256, 64, 0)`이다.
  - `root/nextme/index/<vid>.jsonl`에 n 줄을 쓰고, `ended`이면 `{"end": true}`를 붙인다.
  - `mini_day(tmp_path, nextme_present=True, nextme_frames=range(0, 3), ended=False) -> (day, cfg, info)`
    - NextMe 클립 하나(`video_id` `"v"`, 원본 0–3초, 합성 0–3초)와 Ego4D 클립 하나(`video_id` `"e"`, `make_video` 120프레임, 원본 1–4초, 합성 10–13초)로 된 1일차를 만든다.
    - Ego4D 클립에는 `media: {"kind": "video", "fps": 30}`을 넣는다.
    - `cfg = RenderConfig(media_root=tmp_path / "media")`이고, `info`는 가짜 manifest 정보다.
  - `mini_manifest(tmp_path, days, day2_missing_video=False) -> Path`: `mini_day`의 하루를 `days`일만큼 `day_index`와 합성 시각을 옮겨 manifest JSON으로 쓴다. `day2_missing_video`이면 2일차 Ego4D 파일 이름을 없는 것으로 바꾼다.
  - `ffprobe_json(path) -> dict`: `ffprobe -v error -count_frames -show_streams -show_format -of json`
  - `boom(*a, **k)`: `DayError("simulated")`를 낸다.
  - 테스트는 `import render_frames`와 `from dataclasses import replace`를 쓴다.

- [ ] **Step 1: Write the failing tests**

```python
def test_render_day_outputs(tmp_path):
    day, cfg, info = mini_day(tmp_path)
    rep = render_day(info, day, cfg, tmp_path / "out")
    mp4 = tmp_path / "out" / "D01.mp4"
    meta = ffprobe_json(mp4)  # helper: streams + format
    v = [s for s in meta["streams"] if s["codec_type"] == "video"]
    assert len(meta["streams"]) == 1 and v[0]["width"] == v[0]["height"] == 1024
    assert v[0]["avg_frame_rate"] == "1/1" and int(v[0]["nb_read_frames"]) == rep["frames"] == 6
    lines = [json.loads(l) for l in (tmp_path / "out" / "D01.frames.jsonl").read_text().splitlines()]
    assert [l["dataset"] for l in lines] == ["nextme800"] * 3 + ["ego4d"] * 3
    assert lines[0]["stamp"].startswith("2025-04-07 ") and "frame" in lines[0] and "pts" in lines[3]
    assert json.loads((tmp_path / "out" / "D01.render.json").read_text())["settings"] == settings_key(info, cfg)

def test_day_with_missing_nextme_folder_is_skipped(tmp_path):
    day, cfg, info = mini_day(tmp_path, nextme_present=False)
    assert check_day(day, cfg) == ["NextMe v: 3 frames missing"]

def test_end_hold_is_reported(tmp_path):
    day, cfg, info = mini_day(tmp_path, nextme_frames=range(0, 2), ended=True)  # clip wants 0..2
    rep = render_day(info, day, cfg, tmp_path / "out")
    assert rep["clamped"] == [{"dataset": "nextme800", "video_id": "v", "src": 2, "used": 1}]

def test_failed_day_leaves_no_outputs(tmp_path, monkeypatch):
    day, cfg, info = mini_day(tmp_path)
    monkeypatch.setattr(render_frames, "extract_video_frames", boom)  # raises DayError mid-day
    with pytest.raises(DayError):
        render_day(info, day, cfg, tmp_path / "out")
    assert sorted(p.name for p in (tmp_path / "out").iterdir() if not p.name.startswith(".probe")) == []

def test_rerun_skips_same_settings_and_redoes_changed(tmp_path):
    day, cfg, info = mini_day(tmp_path)
    render_day(info, day, cfg, tmp_path / "out")
    assert day_is_done(tmp_path / "out", 1, settings_key(info, cfg))
    assert not day_is_done(tmp_path / "out", 1, settings_key(info, replace(cfg, rotation={"ego4d": 90, "youtube": 0})))
    assert not day_is_done(tmp_path / "out", 1, settings_key(info, replace(cfg, warp=0.2)))
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3 -m pytest scripts/test_render_frames.py -q -k "day"`
Expected: FAIL (`render_day`가 정의되지 않음)

- [ ] **Step 3: Implement `RenderConfig`, `ENCODE_ARGS`, `settings_key`, `check_day`, `render_day`, `day_is_done`**

`circle_mask`와 `Stamper`는 하루에 한 번만 만든다. NextMe 폴더 목록(`nextme_frames`)과 끝 번호(`nextme_last_frame`)는 video_id마다 한 번만 읽는다. 인코더 stdin에 쓰다가 `BrokenPipeError`가 나면 ffmpeg stderr 끝 20줄을 담아 `DayError`로 바꾼다.

- [ ] **Step 4: Run to verify they pass**

Run: `python3 -m pytest scripts/test_render_frames.py -q`
Expected: 25 passed

- [ ] **Step 5: Commit** (허락된 경우)

```bash
git add scripts/render_frames.py scripts/test_render_frames.py
git commit -m "feat(render): render one day to Dnn.mp4 with frames.jsonl and report"
```

### Task 7: 명령줄, 여러 날 동시 실행, warp 비교 이미지

**Files:**
- Modify: `scripts/render_frames.py`
- Modify: `scripts/test_render_frames.py`
- Modify: `CLAUDE.md`("원본 미디어"나 "진행 중" 절에 렌더 명령 한 줄)

**Interfaces:**
- Consumes: Task 6의 `RenderConfig`, `check_day`, `render_day`, `day_is_done`, `settings_key`. fetcher의 `parse_days`(`from fetch_nextme_frames import parse_days`)
- Produces:
  - `check_tools(font_path: Path) -> list[str]`: 없는 것을 돌려준다. 확인 대상은 `ffmpeg`, `ffprobe`(`shutil.which`), `ffmpeg -hide_banner -encoders` 출력의 `libx264`, 글꼴 파일이다.
  - `main(argv=None) -> int`. 인자는 다음과 같다.
    - 위치 인자 `manifest`
    - `--days`(fetcher와 같은 문법), `--jobs`(기본 4), `--out`(기본 `data/renders/nextme-p00/benchmark`), `--media-root`(기본 `data/media`)
    - `--warp`(float, 기본 0), `--rotate-ego4d`, `--rotate-youtube`(0/90/180/270, 기본 0), `--crf`(기본 18)
    - `--warp-preview`(예 `0,0.15,0.3`), `--force`
  - `warp_preview(manifest: dict, cfg: RenderConfig, strengths: list[float], out: Path) -> Path`:
    - 대상 날 가운데 첫 Ego4D 클립의 처음, 가운데, 끝 3장을 뽑는다.
    - 강도마다 한 줄씩 그리고 칸마다 강도를 글자로 적어 `out/warp-preview.jpg`에 저장한다. 원 가리기도 적용한다. 영상은 만들지 않는다.
- 동작:
  1. 도구 확인에 실패하면 무엇이 없는지 출력하고 2를 돌려준다.
  2. 대상 날마다 `day_is_done`이면 "D07 up to date"를 출력하고 건너뛴다(`--force`면 다시 만든다).
  3. 모든 대상 날을 `check_day`로 먼저 확인해, 모자란 날을 한 번에 출력하고 뺀다. 형식은 `"day 3: Ego4D 3efc…: file missing; NextMe …: 120 frames missing"`이다.
  4. 나머지 날은 `ProcessPoolExecutor(max_workers=jobs)`로 `render_day`를 실행한다. 날이 끝날 때마다 `"D01 done: 23730 frames in 812 s"` 또는 `"D05 FAILED: <message>"`를 출력한다.
  5. 모든 대상 날이 성공(또는 up to date)이면 0을 돌려준다. 건너뛰거나 실패한 날이 있으면 1이다.

- [ ] **Step 1: Write the failing tests**

```python
def test_main_renders_ready_days_and_reports_missing(tmp_path, capsys):
    manifest_path = mini_manifest(tmp_path, days=2, day2_missing_video=True)  # helper over mini_day
    rc = main([str(manifest_path), "--out", str(tmp_path / "out"), "--media-root", str(tmp_path / "media"), "--jobs", "2"])
    out = capsys.readouterr().out
    assert rc == 1 and (tmp_path / "out" / "D01.mp4").exists() and not (tmp_path / "out" / "D02.mp4").exists()
    assert "day 2: Ego4D" in out and "file missing" in out
    rc2 = main([str(manifest_path), "--out", str(tmp_path / "out"), "--media-root", str(tmp_path / "media"), "--days", "1"])
    assert rc2 == 0 and "D01 up to date" in capsys.readouterr().out

def test_main_refuses_without_tools(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(render_frames.shutil, "which", lambda name: None)
    assert main([str(mini_manifest(tmp_path, days=1)), "--out", str(tmp_path / "o")]) == 2
    assert "ffmpeg" in capsys.readouterr().out

def test_warp_preview_writes_one_image(tmp_path):
    m = json.loads(mini_manifest(tmp_path, days=1).read_text())
    p = warp_preview(m, RenderConfig(media_root=tmp_path / "media"), [0, 0.2], tmp_path / "out")
    im = Image.open(p)
    assert p.name == "warp-preview.jpg" and im.width >= 3 * 256 and im.height >= 2 * 256
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3 -m pytest scripts/test_render_frames.py -q -k "main or preview"`
Expected: FAIL (`main`이 정의되지 않음)

- [ ] **Step 3: Implement `check_tools`, `warp_preview`, `main`**

미리보기 칸은 256×256으로 줄인다. 강도 글자는 `Stamper`의 글꼴을 작게 써서 칸 왼쪽 위에 적는다.

- [ ] **Step 4: Run all tests**

Run: `python3 -m pytest scripts/test_render_frames.py scripts/test_fetch_nextme_frames.py -q`
Expected: render 28 passed + fetch 29 passed

- [ ] **Step 5: CLAUDE.md에 렌더 명령 한 줄 추가**

`python3 scripts/render_frames.py data/exports/nextme-p00-e274ee8b-r0.manifest.json --days 1`을 적고, 결과 위치(`data/renders/nextme-p00/benchmark/`)도 함께 적는다.

- [ ] **Step 6: Commit** (허락된 경우)

```bash
git add scripts/render_frames.py scripts/test_render_frames.py CLAUDE.md
git commit -m "feat(render): CLI with parallel days, missing-source summary and warp preview"
```

### Task 8: 실제 데이터 확인 (사람이 본다, 자동 테스트 아님)

**Files:** 없음(결과는 `data/renders/`, 커밋하지 않음)

- [ ] **Step 1: 원본이 다 있는지 dry 확인**

`check_day`만 돌려 28일 모두 빈 목록인지 본다.

```bash
python3 -c "import json,sys;sys.path.insert(0,'scripts');import render_frames as r;m=json.load(open('data/exports/nextme-p00-e274ee8b-r0.manifest.json'));c=r.RenderConfig();[print(d['day_index'],r.check_day(d,c)) for d in m['days'] if r.check_day(d,c)]"
```

Expected: 출력 없음. 출력이 있으면 그 날의 원본을 먼저 받는다.

- [ ] **Step 2: warp 비교 이미지**

Run: `python3 scripts/render_frames.py data/exports/nextme-p00-e274ee8b-r0.manifest.json --days 1 --warp-preview 0,0.1,0.2,0.3`
Expected: `data/renders/nextme-p00/benchmark/warp-preview.jpg`가 생긴다. 이 이미지를 사용자에게 보여 주고 `warp`와 회전 값을 정한다.

- [ ] **Step 3: 1일차 렌더**

Run: `python3 scripts/render_frames.py data/exports/nextme-p00-e274ee8b-r0.manifest.json --days 1 --warp <정한 값>`
Expected: `D01 done: N frames`. N이 manifest 1일차의 정수 `t` 개수와 같아야 한다. `D01.render.json`의 `clamped`에는 `2026-04-28_1204-1623`의 녹화 끝(2초) 항목이 있어야 한다.

- [ ] **Step 4: 사람이 확인**

- 클립 경계 앞뒤 프레임이 맞는 원본인지 본다(`D01.frames.jsonl`의 경계 `t`에서 `ffmpeg -ss <idx> -frames:v 1`로 뽑아 비교).
- 타임스탬프가 읽히는지, 원본 HKT 글자가 남지 않았는지 본다.
- 파일 크기와 화질을 보고 crf를 정한다.
- 결과를 사용자에게 보여 주고 승인을 받는다.

- [ ] **Step 5: 전체 렌더**

Run: `python3 scripts/render_frames.py data/exports/nextme-p00-e274ee8b-r0.manifest.json --jobs 8 --warp <값> 2>&1 | tee data/renders/nextme-p00/render.log`
Expected: 28일 모두 `done` 또는 `up to date`이고 종료 코드 0이다. 18일차 `D18.render.json`의 `clamped`에 `2026-07-23_1353-1637` 항목(4초)이 있어야 한다.
