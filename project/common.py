"""Small shared IO helpers; no training-framework imports."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

DEFAULT_CONFIG = Path(__file__).parent / "configs/main.json"


def config_from(path=None):
    return json.loads(Path(path or DEFAULT_CONFIG).read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def save_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def model_inventory(path):
    """Local file provenance; metadata inventory is not a cryptographic weight hash."""
    path = Path(path)
    files = {}
    for item in sorted(path.iterdir()):
        if item.is_file() and item.suffix in (".json", ".safetensors", ".bin", ".txt"):
            files[item.name] = {"size": item.stat().st_size, "mtime_ns": item.stat().st_mtime_ns}
    return {"path": str(path), "files": files, "config_sha256": sha256(path / "config.json")}


def code_info():
    repo = Path(__file__).resolve().parent.parent
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True)
    files = {str(p.relative_to(repo)): sha256(p) for p in sorted((repo / "project").rglob("*"))
             if p.is_file() and p.suffix in (".py", ".sh", ".json")}
    return {"commit": commit.stdout.strip(), "project_code_sha256": signature(files)}


def show_summary(out, text):
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text, flush=True)


def resource_path(key):
    if key not in os.environ:
        raise RuntimeError("Launch through bash project/scripts/run.sh")
    return Path(os.environ[key])
