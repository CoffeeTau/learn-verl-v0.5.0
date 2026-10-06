"""Download pinned 2Wiki JSON files from ModelScope; verify publisher checksums."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from datetime import datetime, timezone

REPO = "voidful/2WikiMultihopQA"
REVISION = "75d023a1becf3e2e73401639536eae3d0bc82d2c"
# ModelScope repo/tree metadata checked on 2026-10-06.
FILES = {
    "train.json": (681705246, "b3fddb4d5bb42cd797919cad67616545be51b24740e0a7dabdae7bf76b8f7bfa"),
    "dev.json": (55934464, "48b9bdc69654dc580fda5f935a48b88cb89f11887587310af60d406c8d0111a6"),
}


def verify(path, size, digest):
    if not path.is_file() or path.stat().st_size != size:
        return False
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest() == digest


def main():
    root = Path(os.environ["AGENTIC_RAW_DATA_DIR"])
    out = Path(os.environ["AGENTIC_RUNS_DIR"]) / "resource_smoke"
    root.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "download_2wiki_report.json"
    report = {"status": "running", "repo_id": REPO, "revision": REVISION,
              "source": "modelscope", "started_at": datetime.now(timezone.utc).isoformat(),
              "raw_data_dir": str(root), "files": {},
              "limitation": "Mirror not verified identical to official data_ids_april7.zip; aliases absent."}
    def save():
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    save()
    try:
        for name, (size, digest) in FILES.items():
            path = root / name
            duplicates = [str(p) for p in root.rglob(name) if p != path]
            if duplicates:
                raise ValueError(f"Another dataset exists: {duplicates}. Keep different source versions separately.")
            if path.exists():
                if not verify(path, size, digest):
                    raise ValueError(f"Existing file differs from pinned mirror; not overwriting: {path}")
                print(f"Verified existing {path}", flush=True)
            else:
                partial = path.with_suffix(".json.part")
                url = f"https://modelscope.cn/api/v1/datasets/{REPO}/repo?Revision={REVISION}&FilePath={name}"
                print(f"Downloading {name} ({size / 1e6:.1f} MB) -> {path}", flush=True)
                if not verify(partial, size, digest):
                    subprocess.run(["curl", "-fL", "--retry", "3", "--connect-timeout", "30",
                                    "--speed-limit", "1024", "--speed-time", "120",
                                    "-C", "-", url, "-o", str(partial)], check=True)
                if not verify(partial, size, digest):
                    raise ValueError(f"Size/SHA256 mismatch: {partial}; send the report before retrying")
                partial.replace(path)
            report["files"][name] = {"path": str(path), "size": size, "sha256": digest, "verified": True}
            save()
        report["status"] = "passed"
    except Exception:
        report["status"] = "failed"
        report["traceback"] = traceback.format_exc()
        print(report["traceback"], file=sys.stderr)
    finally:
        save()
        print(f"DOWNLOAD: {report['status'].upper()}\nReport: {target}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
