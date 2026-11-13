"""Continuous wrapper over the repository's causal EgoRAG implementation.

SQLite holds the native caption/summary/embedding representation. Sampled JPEGs
are retained and charged because the existing visual answerer retrieves them.
There is no native eviction policy: exceeding B is an unsupported configuration.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import importlib
import io
import json
import math
from pathlib import Path
import sqlite3
import sys
import tempfile

from baselines.streambench.method import StreamingMemoryMethod


def native_module(name):
    root = Path(__file__).resolve().parents[2] / "egorag"
    expected = root / "videomem_adapter" / "__init__.py"
    if not expected.is_file():
        raise FileNotFoundError(f"EgoRAG source unavailable: {root}")
    present = sys.modules.get("videomem_adapter")
    if present and Path(present.__file__).resolve() != expected.resolve():
        raise RuntimeError("A different videomem_adapter package is already imported")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return importlib.import_module("videomem_adapter." + name)


def validate_frames(frames, timestamps, previous=None):
    if len(frames) != len(timestamps) or not frames:
        raise ValueError("frames/timestamps must be nonempty and have equal lengths")
    last = previous
    clean = []
    required = ("stream_id", "day", "global_frame_id", "global_timestamp", "source_timestamp", "source_id", "source_frame_index")
    for record in timestamps:
        if any(key not in record for key in required):
            raise ValueError("Incomplete common frame timestamp")
        record = {key: record[key] for key in required}
        if not isinstance(record["stream_id"], str) or not record["stream_id"]:
            raise ValueError("Invalid stream ID")
        if type(record["day"]) is not int or record["day"] < 1:
            raise ValueError("Invalid day")
        if any(type(record[key]) is not int or record[key] < 0 for key in ("global_frame_id", "source_frame_index")):
            raise ValueError("Frame indices must be nonnegative integers")
        if any(isinstance(record[key], bool) or not isinstance(record[key], (float, int)) or not math.isfinite(record[key]) or record[key] < 0
               for key in ("global_timestamp", "source_timestamp")):
            raise ValueError("Frame timestamps must be finite and nonnegative")
        if last and (record["stream_id"] != last["stream_id"] or record["global_timestamp"] <= last["global_timestamp"]
                     or record["global_frame_id"] <= last["global_frame_id"] or record["day"] < last["day"]):
            raise ValueError("Frames must remain chronological within one stream")
        clean.append(record)
        last = record
    return clean


def jpeg_frame(frame, timestamp, **extra):
    from PIL import Image
    if not isinstance(frame, Image.Image):
        raise TypeError("Text/tree encoders require PIL frames from the common sampler")
    output = io.BytesIO()
    frame.convert("RGB").save(output, format="JPEG", quality=95)
    raw = output.getvalue()
    return {"jpeg": raw, "sha256": hashlib.sha256(raw).hexdigest(),
            "file_seconds": timestamp["source_timestamp"], **extra}


def wire_frame(frame):
    return {**{key: value for key, value in frame.items() if key != "jpeg"},
            "jpeg_base64": base64.b64encode(frame["jpeg"]).decode("ascii")}


class EgoRAGMethod(StreamingMemoryMethod):
    def __init__(self, memory_budget_bytes, pipeline_config, *, frame_period_seconds=1.0,
                 models=None, embeddings=None):
        super().__init__(memory_budget_bytes)
        self.config = copy.deepcopy(pipeline_config)
        if self.config.get("index_engine", "sqlite") != "sqlite":
            raise ValueError("Streaming EgoRAG currently requires explicit SQLite exact retrieval; Chroma snapshots are unsupported")
        self.config["index_engine"] = "sqlite"
        if not math.isfinite(frame_period_seconds) or frame_period_seconds <= 0:
            raise ValueError("frame_period_seconds must be positive")
        self.frame_period_seconds = frame_period_seconds
        self._injected_models, self._injected_embeddings = models, embeddings
        self._pipe = None
        self._temporary = None
        self._frames = {}
        self._origins = {}
        self._last = None
        self._cutoff = None

    def initialize(self):
        if self._pipe is not None:
            self._pipe.store.db.close()
        if self._temporary is not None:
            self._temporary.cleanup()
        self._temporary = tempfile.TemporaryDirectory(prefix="streambench-egorag-")
        pipeline = native_module("pipeline")
        self._pipe = pipeline.Pipeline(self.config, {"clips": [], "videos": []}, self._temporary.name,
                                       models=self._injected_models, embeddings=self._injected_embeddings)
        # Move the existing adapter's database to RAM. No hidden disk cache may
        # retain evicted/uncheckpointed records. SQLite pages themselves are B.
        old = self._pipe.store.db
        memory = sqlite3.connect(":memory:")
        old.backup(memory)
        old.close()
        self._pipe.store.db = memory
        self._pipe.store.path.unlink()
        self._frames, self._origins = {}, {}
        self._last = self._cutoff = None
        self._pipe.frames = lambda clip: [wire_frame(frame) for frame in self._frames[clip["clip_id"]]]
        self.minimum_memory_bytes = self.memory_bytes()["persistent_total_bytes"]
        self.enforce_budget()

    def _ready(self):
        if self._pipe is None:
            raise RuntimeError("initialize or load_checkpoint before using EgoRAG")

    def observe(self, frames, timestamps):
        self._ready()
        timestamps = validate_frames(frames, timestamps, self._last)
        # A caption interval cannot straddle day IDs. This closes a semantic
        # interval, never resets memory, and processes every supplied sample.
        start = 0
        for stop in range(1, len(frames) + 1):
            if stop == len(frames) or timestamps[stop]["day"] != timestamps[start]["day"]:
                self._observe_clip(frames[start:stop], timestamps[start:stop])
                start = stop
        self._last = copy.deepcopy(timestamps[-1])

    def _observe_clip(self, frames, timestamps):
        pipe = self._pipe
        day = timestamps[0]["day"]
        origin = self._origins.setdefault(day, timestamps[0]["global_timestamp"])
        start = timestamps[0]["global_timestamp"] - origin
        end = timestamps[-1]["global_timestamp"] - origin + self.frame_period_seconds
        if end >= 86400:
            raise ValueError("EgoRAG's native HHMMSScc schema requires each day shorter than 24 hours")
        cid = f"{timestamps[0]['stream_id']}:{timestamps[0]['global_frame_id']}:{timestamps[-1]['global_frame_id']}"
        clip = {"clip_id": cid, "day_id": f"Day{day}", "day_order": day,
                "day_start_seconds": start, "day_end_seconds": end,
                "file_start_seconds": start, "file_end_seconds": end,
                "path": "retained-jpeg://" + cid, "timestamps": copy.deepcopy(timestamps)}
        retained = [jpeg_frame(frame, stamp, day_id=f"Day{day}", clip_id=cid,
                               day_seconds=stamp["global_timestamp"] - origin)
                    for frame, stamp in zip(frames, timestamps)]
        wire = [wire_frame(frame) for frame in retained]
        caption = pipe.call("captioner", "You are an egocentric agent. Take yourself as the main character of the video.",
                            pipe.prompts["caption"], wire)
        ids = pipe._index_caption(clip, caption)
        digest = native_module("common").digest
        pipe.store.put("captions", digest([pipe.namespace, clip]),
                       {"clip": clip, "caption": caption, "sentence_ids": ids,
                        "frames": [{k: v for k, v in f.items() if k != "jpeg_base64"} for f in wire]})
        pipe.manifest["clips"].append(clip)
        pipe.clips[cid], pipe.days[f"Day{day}"] = clip, day
        pipe.models.protocol_context["manifest_hash"] = digest(pipe.manifest)
        self._frames[cid] = retained
        self._cutoff = {"day_id": f"Day{day}", "day_order": day, "seconds": end}

    def finalize_chunk(self):
        self._ready()
        if self._cutoff:
            self._pipe.summaries(self._cutoff)
        return self.enforce_budget()

    def persistent_state(self):
        self._ready()
        return {"visual_memory_bytes": self._frames,
                "text_memory_bytes": {"sqlite_pages": self._pipe.store.db.serialize()},
                "metadata_bytes": self._metadata()}

    def _metadata(self):
        return {"config": self.config, "frame_period_seconds": self.frame_period_seconds,
                "manifest": self._pipe.manifest, "origins": self._origins, "last": self._last,
                "cutoff": self._cutoff, "namespace": self._pipe.namespace,
                "index_namespace": self._pipe.index_namespace, "provenance": self._pipe.provenance}

    def state_dict(self):
        self._ready()
        return copy.deepcopy({"schema_version": 1, **self._metadata(), "frames": self._frames,
                              "sqlite": self._pipe.store.db.serialize()})

    def load_state_dict(self, state):
        if state.get("schema_version") != 1 or state["config"] != self.config or state["frame_period_seconds"] != self.frame_period_seconds:
            raise ValueError("EgoRAG checkpoint config/schema mismatch")
        if self._pipe is None:
            self.initialize()
        if state["namespace"] != self._pipe.namespace or state["index_namespace"] != self._pipe.index_namespace:
            raise ValueError("EgoRAG checkpoint code/model provenance mismatch")
        owned = copy.deepcopy(state)
        pipe = self._pipe
        pipe.store.db.deserialize(owned["sqlite"])
        pipe.manifest = owned["manifest"]
        pipe.clips = {c["clip_id"]: c for c in pipe.manifest["clips"]}
        pipe.days = {c["day_id"]: c["day_order"] for c in pipe.manifest["clips"]}
        self._frames, self._origins = owned["frames"], owned["origins"]
        self._last, self._cutoff = owned["last"], owned["cutoff"]
        pipe.models.protocol_context["manifest_hash"] = native_module("common").digest(pipe.manifest)

    def answer(self, question, choices=None):
        self._ready()
        if not self._cutoff:
            return {"answer": "", "status": "no_observed_evidence", "query_read_bytes": 0,
                    "num_retrieved_items": 0, "num_text_items_used": 0}
        pipe = self._pipe
        baseline = pipe.store.db.serialize()
        # Record unique persistent records actually returned by native retrieval.
        cache_sizes = {(kind, key): len(payload.encode("utf-8")) for kind, key, payload in
                       pipe.store.db.execute("SELECT kind,key,payload FROM cache")}
        sentence_sizes = {row[0]: sum(len(str(v).encode("utf-8")) for v in row) for row in
                          pipe.store.db.execute("SELECT id,document,metadata,embedding FROM sentences WHERE namespace=?", (pipe.index_namespace,))}
        read = set()
        original_get, original_collection_get = pipe.store.get, pipe.index.collection.get
        def get(kind, key):
            if (kind, key) in cache_sizes:
                read.add((kind, key))
            return original_get(kind, key)
        def collection_get(*args, **kwargs):
            result = original_collection_get(*args, **kwargs)
            read.update(("sentence", key) for key in result["ids"])
            return result
        pipe.store.get, pipe.index.collection.get = get, collection_get
        try:
            query = pipe.query({"question_id": "checkpoint-query", "question": question,
                                "choices": list(choices or []), "query_time": self._cutoff})
            result = pipe.answer(query)
            cids = {hit["clip_id"] for hit in query["retrieved"]}
            reads = sum(sentence_sizes.get(key, 0) if kind == "sentence" else cache_sizes[(kind, key)]
                        for kind, key in read)
            reads += sum(len(f["jpeg"]) for cid in cids for f in self._frames[cid])
            prediction = result["prediction"]
            if prediction is None and result["status"] == "no_observed_evidence":
                prediction = ""
            return {"answer": prediction, "status": result["status"],
                    "query_read_bytes": reads, "num_retrieved_items": len(cids),
                    "num_text_items_used": len(query["retrieved"]),
                    "retrieved_items": query["retrieved"], "test_only": self.config.get("test_only", False)}
        finally:
            pipe.store.get, pipe.index.collection.get = original_get, original_collection_get
            pipe.store.db.deserialize(baseline)


def create_method(config):
    native = config.get("pipeline_config")
    if isinstance(native, str):
        native = json.loads(Path(native).read_text())
    if not isinstance(native, dict):
        raise ValueError("EgoRAG requires pipeline_config (mapping or JSON path) with real model backends")
    return EgoRAGMethod(config["memory_budget_bytes"], native,
                        frame_period_seconds=config.get("frame_period_seconds", 1.0))
