import json
from pathlib import Path
import uuid
import pandas as pd
import streamlit as st
from src.common import load, save
from src.dataset_adapters import adapter
from src.clip_manager import ClipStore, create_clip, thumbnail
from src.routine_generator import build, change, TEMPLATE
from src.stream_generator import generate
from src.qa_generator import generate_qa
from src.validation import validate

ROOT = Path(__file__).resolve().parent
store = ClipStore(ROOT/"metadata/clips.json")
st.set_page_config(page_title="Video Memory Benchmark", layout="wide")
st.title("Long-term Egocentric Video Benchmark")
st.caption("Local files only · Synthetic persona · Human-verified ground truth")
browser, routine, export = st.tabs(["1. Clip Browser", "2. Persona & Routine", "3. Generate & Validate"])
with browser:
    settings = load(ROOT/"config/datasets.json", {})
    dataset = st.selectbox("Dataset", list(settings))
    folder = st.text_input("Local video folder", value=settings[dataset])
    if st.button("Scan local folder"):
        try:
            videos, errors = adapter(folder, dataset).discover()
            st.session_state.videos = videos
            settings[dataset] = folder
            save(ROOT/"config/datasets.json", settings)
            st.session_state.scan_errors = errors
        except Exception as error:
            st.error(str(error))
    videos = st.session_state.get("videos", [])
    if st.session_state.get("scan_errors"):
        st.warning(st.session_state.scan_errors)
    if videos:
        st.dataframe(pd.DataFrame(videos), use_container_width=True)
        video = videos[st.selectbox("Video index", range(len(videos)), format_func=lambda i: videos[i]["filename"])]
        start = st.number_input("Start seconds", 0.0, float(video["duration"]), 0.0, step=0.1)
        end = st.number_input("End seconds", 0.0, float(video["duration"]), float(video["duration"]), step=0.1)
        st.video(video["path"], start_time=start, end_time=end)
        activity = st.text_input("Activity label", "cooking")
        participant = st.text_input("Source participant (unknown allowed)")
        location = st.text_input("Location")
        recorded = st.text_input("Original recorded date/time (leave blank if unknown)")
        attributes = st.text_area("Verified attributes JSON", "{}")
        verified = st.checkbox("I checked this interval or its source annotation")
        evidence = st.text_input("Label/attribute provenance (reviewer or annotation reference)")
        if st.button("Save selected clip & thumbnail"):
            try:
                attrs = json.loads(attributes)
                if not isinstance(attrs, dict):
                    raise ValueError("Attributes must be a JSON object")
                if attrs and not verified:
                    raise ValueError("Attributes require verification")
                clip = create_clip(video, start, end, activity, participant, location, attrs, verified, evidence, recorded or None)
                image = ROOT/"outputs/thumbnails"/f"{clip['clip_id']}.jpg"
                thumbnail(clip, image)
                store.add(clip)
                st.image(str(image))
                st.success(f"Saved {clip['clip_id']}")
            except Exception as error:
                st.error(str(error))
    clips = store.all()
    st.dataframe(pd.DataFrame(clips), use_container_width=True)
    st.download_button("Download clip JSON", json.dumps(clips, indent=2), "clips.json")
    st.download_button("Download clip CSV", pd.DataFrame(clips).to_csv(index=False), "clips.csv")
    sequence = st.multiselect("Choose clips in playback order", [c["clip_id"] for c in clips])
    if st.button("Save playback sequence"):
        save(ROOT/"metadata/sequence.json", sequence)
    for cid in sequence:
        c = next(c for c in clips if c["clip_id"] == cid)
        st.caption(cid)
        st.video(c["source_video"], start_time=c["start_time"], end_time=c["end_time"])
with routine:
    persona_id = st.text_input("Synthetic persona ID", "remote_worker_01")
    days = st.number_input("Days D", min_value=1, max_value=365, value=7)
    repeats = st.number_input("Repetitions per slot N", min_value=1, max_value=100, value=1)
    seed = st.number_input("Random seed", min_value=0, value=42)
    st.caption("Edit template rows or delete/add activities before building. Times are virtual routine labels; stream time is compressed.")
    template = st.data_editor(pd.DataFrame(TEMPLATE), num_rows="dynamic", key="template")
    if st.button("Build routine"):
        try:
            plan = build(store.all(), persona_id, int(days), int(repeats), int(seed), template.to_dict("records"))
            save(ROOT/"metadata/plan.json", plan)
            st.success("Routine saved")
        except Exception as error:
            st.error(str(error))
    plan = load(ROOT/"metadata/plan.json")
    if plan:
        day = st.selectbox("Timeline day", range(1, plan["days"]+1))
        events = [e for e in plan["events"] if e["synthetic_day"] == day]
        st.dataframe(pd.DataFrame(events), use_container_width=True)
        if events and clips:
            eid = st.selectbox("Event to edit", [e["event_id"] for e in events])
            cid = st.selectbox("Replacement clip", [c["clip_id"] for c in clips])
            variation = st.selectbox("Variation", ["stable_routine", "episodic_deviation", "activity_replacement"])
            if st.button("Apply clip change"):
                save(ROOT/"metadata/plan.json", change(plan, eid, next(c for c in clips if c["clip_id"] == cid), variation))
                st.rerun()
            if st.button("Remove selected event"):
                plan["events"] = [e for e in plan["events"] if e["event_id"] != eid]
                save(ROOT/"metadata/plan.json", plan)
                st.rerun()
        st.download_button("Download routine plan", json.dumps(plan, indent=2), "plan.json")
with export:
    render = st.checkbox("Render MP4 (otherwise manifest only)")
    remove_audio = st.checkbox("Remove audio", value=True)
    width = st.number_input("Width", min_value=2, value=640, step=2)
    height = st.number_input("Height", min_value=2, value=360, step=2)
    fps = st.number_input("FPS", min_value=1, max_value=120, value=24)
    if st.button("Generate stream, manifest and QA"):
        try:
            plan = load(ROOT/"metadata/plan.json")
            if not plan:
                raise ValueError("Build a routine first")
            output = ROOT/"outputs"/plan["persona_id"]/uuid.uuid4().hex[:10]
            with st.spinner("Validating and generating..."):
                data = generate(plan, store.all(), output, render, int(width), int(height), int(fps), remove_audio)
                qa = generate_qa(data)
                save(output/"qa.json", qa)
                report = validate(data, qa)
                save(output/"validation.json", report)
            st.success(str(output))
            st.json(report)
            st.download_button("Manifest", json.dumps(data, indent=2), "manifest.json")
            st.download_button("QA", json.dumps(qa, indent=2), "qa.json")
            if render:
                st.video(str(output/"full_stream.mp4"))
        except Exception as error:
            st.error(str(error))

st.divider()
st.subheader("TeleEgo P4 — Synthetic Day 1")
teleego_output = ROOT/"outputs/teleego_p4_day1"
teleego_data = load(teleego_output/"source_manifest.json")
if teleego_data:
    if (teleego_output/"real_preview_30s.mp4").exists():
        st.caption("실제 원본 0–30초 미리보기 · 전체 Day 1 완성본이 아닙니다.")
        st.video(str(teleego_output/"real_preview_30s.mp4"))
    st.caption("P4의 D1·D2·D3를 재배치한 합성 하루입니다. 원본 시각 단서는 유지되며, 실제 연속 하루를 의미하지 않습니다.")
    st.write(f"Length: {teleego_data['duration']/3600:.3f} h · {teleego_data['duration_status']} · Rendered: {teleego_data['rendered']}")
    st.dataframe(pd.DataFrame(teleego_data['segments']), use_container_width=True)
    annotations = load(teleego_output/"annotations.json", [])
    segment_id = st.selectbox("TeleEgo source segment", [s["clip_id"] for s in teleego_data["segments"]])
    segment = next(s for s in teleego_data["segments"] if s["clip_id"] == segment_id)
    matching = [a for a in annotations if a["clip_id"] == segment_id]
    st.dataframe(pd.DataFrame(matching), use_container_width=True)
    if Path(segment["source_video"]).exists():
        if matching:
            index = st.number_input("Narration row preview", min_value=0, max_value=len(matching)-1, value=0)
            row = matching[index]
            st.write(row["text"])
            st.video(segment["source_video"], start_time=row["source_start"], end_time=row["source_end"])
    else:
        st.info("원본 영상 다운로드 진행 중입니다. 이 길이는 아직 SRT 기반 추정입니다.")
    st.download_button("TeleEgo source manifest", json.dumps(teleego_data, ensure_ascii=False, indent=2), "teleego_manifest.json")
    if (teleego_output/"day_01.mp4").exists() and teleego_data["rendered"]:
        st.video(str(teleego_output/"day_01.mp4"))
