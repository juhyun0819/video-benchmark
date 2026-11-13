"""Load unchanged pure upstream methods/prompts without importing every model SDK.

AST extraction does not rewrite their bodies. Source filenames and SHA256 are
recorded. Temporal/media/storage boundaries are implemented in the adapter.
"""
import ast
import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT = "7a97157908757cc898c26835b718653055ecc5f5"
FILES = ["main.py", "create_database.py", "egorag/agents/RagAgent.py",
         "egorag/database/Chroma.py", "egorag/utils/gen_event.py",
         "egorag/utils/prompts.py", "egorag/utils/agent_fun.py"]


def provenance():
    return {"commit": UPSTREAM_COMMIT, "files": {
        p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in FILES}}


def load_function(relative, name, namespace, class_name=None):
    tree = ast.parse((ROOT/relative).read_text())
    nodes = tree.body
    if class_name:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name).body
    node = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[node], type_ignores=[])
    env = {"re": re, **namespace}
    exec(compile(module, str(ROOT/relative), "exec"), env)
    return env[name]


def constant(relative, name):
    tree = ast.parse((ROOT/relative).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
    value = node.value
    if isinstance(value, ast.JoinedStr):
        if any(not isinstance(v, ast.Constant) for v in value.values):
            raise ValueError("Only literal upstream prompts can be loaded")
        return "".join(v.value for v in value.values)
    return ast.literal_eval(value)


def prompts():
    result = {name: constant("egorag/utils/prompts.py", name) for name in
              ["get_day_prompt", "get_hour_prompt", "query_prompt", "video_evidence_prompt",
               "with_evidence", "without_evidence"]}
    for level in ["min", "hour", "day"]:
        result["summary_"+level] = constant("egorag/utils/gen_event.py", "system_message_"+level)
    result["caption"] = ("Imagine you are the character in the video, describing from a first-person "
        "perspective what you saw, and everything that happened over time. Use I as the subject.")
    return result
