import json
import subprocess
from pathlib import Path

def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)

def load(path, default=None):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else default

def run(args):
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"Command failed: {args[0]}\n{result.stderr[-4000:]}")
    return result.stdout

def probe(path):
    data = json.loads(run(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)]))
    video = next(s for s in data["streams"] if s["codec_type"] == "video")
    n, d = map(int, video["avg_frame_rate"].split("/"))
    return {"path": str(Path(path).resolve()), "filename": Path(path).name,
            "duration": float(data["format"]["duration"]), "width": video["width"],
            "height": video["height"], "fps": n / d if d else 0,
            "has_audio": any(s["codec_type"] == "audio" for s in data["streams"])}
