"""Tests for convert_ego4d_narrations.py. Run with: python3 -m pytest scripts/test_convert_ego4d_narrations.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from convert_ego4d_narrations import main, select_narrations

CATEGORIES = {
    "categories": [
        {"id": "walking", "label": "Walking", "dataset": "ego4d", "scenarios": ["Walking on street"], "keywords": []},
        {"id": "cooking", "label": "Cooking", "dataset": "egoexo4d", "scenarios": ["Cooking"], "keywords": []},
    ]
}


def pass1(text):
    return {
        "narrations": [{"timestamp_sec": 1.0, "narration_text": text}],
        "summaries": [{"start_sec": 0.0, "end_sec": 90.0, "summary_text": text}],
    }


def meta(*videos):
    return {"videos": [{"video_uid": uid, "scenarios": scenarios} for uid, scenarios in videos]}


def test_keeps_pass_1_of_videos_in_an_ego4d_scenario():
    narrations = {"walk": {"narration_pass_1": pass1("#C C walks")}}
    out = select_narrations(meta(("walk", ["Walking on street"])), narrations, CATEGORIES)
    assert out == {"walk": {"narration_pass_1": pass1("#C C walks")}}


def test_drops_videos_outside_the_category_scenarios():
    narrations = {"knit": {"narration_pass_1": pass1("#C C knits")}}
    out = select_narrations(meta(("knit", ["Crafting/knitting/sewing/drawing/painting"])), narrations, CATEGORIES)
    assert out == {}


def test_ignores_scenarios_of_other_datasets():
    # "Cooking" belongs to an egoexo4d category, so an Ego4D cooking video is not kept.
    narrations = {"cook": {"narration_pass_1": pass1("#C C stirs")}}
    out = select_narrations(meta(("cook", ["Cooking"])), narrations, CATEGORIES)
    assert out == {}


def test_drops_passes_other_than_pass_1():
    narrations = {"walk": {"narration_pass_1": pass1("one"), "narration_pass_2": pass1("two"), "status": "x"}}
    out = select_narrations(meta(("walk", ["Walking on street"])), narrations, CATEGORIES)
    assert out == {"walk": {"narration_pass_1": pass1("one")}}


def test_leaves_out_matching_videos_without_pass_1():
    narrations = {"walk": {"narration_pass_2": pass1("two")}}
    out = select_narrations(meta(("walk", ["Walking on street"]), ("gone", ["Walking on street"])), narrations, CATEGORIES)
    assert out == {}


def test_main_writes_the_selection(tmp_path):
    paths = {name: tmp_path / f"{name}.json" for name in ("metadata", "narrations", "categories")}
    paths["metadata"].write_text(json.dumps(meta(("walk", ["Walking on street"]), ("knit", ["Crafting"]))))
    paths["narrations"].write_text(
        json.dumps({"walk": {"narration_pass_1": pass1("a")}, "knit": {"narration_pass_1": pass1("b")}})
    )
    paths["categories"].write_text(json.dumps(CATEGORIES))
    out = tmp_path / "out" / "narration_selected.json"

    main([
        "--metadata", str(paths["metadata"]),
        "--narrations", str(paths["narrations"]),
        "--categories", str(paths["categories"]),
        "--out", str(out),
    ])

    assert json.loads(out.read_text()) == {"walk": {"narration_pass_1": pass1("a")}}
