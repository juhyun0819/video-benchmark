"""Explicitly synthetic fixtures: colors/test patterns are NOT egocentric footage."""
import argparse
from pathlib import Path
from src.common import run, save
from src.clip_manager import create_clip
from src.routine_generator import build, TEMPLATE
from src.stream_generator import generate
from src.qa_generator import generate_qa
from src.validation import validate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--output", default="outputs/test_demo")
    args = parser.parse_args()
    root = Path("datasets/TEST_ONLY").resolve()
    root.mkdir(parents=True, exist_ok=True)
    clips = []
    for i, slot in enumerate(TEMPLATE):
        for variant in range(2):
            path = root/f"TEST_ONLY_{slot['activity']}_{variant}.mp4"
            if not path.exists():
                run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                     f"testsrc2=size=320x180:rate=12:duration=1.5", "-vf", f"hue=h={i*50+variant*20}",
                     "-c:v", "libx264", "-threads", "2", "-pix_fmt", "yuv420p", str(path)])
            clip = create_clip({"path": str(path), "source_dataset": "TEST_ONLY"}, 0, 1,
                               slot["activity"], participant="SYNTHETIC_FIXTURE",
                               verified=True, evidence="TEST ONLY: scripted fixture label; not observed human activity")
            clip["clip_id"] = f"TEST_ONLY_{i}_{variant}"
            clips.append(clip)
    save("metadata/test_clips.json", clips)
    plan = build(clips)
    save("metadata/test_plan.json", plan)
    data = generate(plan, clips, args.output, render=args.render, width=320, height=180, fps=12)
    qa = generate_qa(data)
    save(Path(args.output)/"qa.json", qa)
    report = validate(data, qa)
    save(Path(args.output)/"validation.json", report)
    print(f"TEST ONLY: {len(data['events'])} events, {data['days']} days, {data['duration']} seconds, {len(qa)} QA; valid={report['valid']}")

if __name__ == "__main__":
    main()
