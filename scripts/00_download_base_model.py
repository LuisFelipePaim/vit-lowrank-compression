"""Fetch and SHA-256 verify the pinned pretrained baseline; never uploads."""
import hashlib
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess


def sha256(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    m = json.loads(Path("audit/base_model_manifest.json").read_text(encoding="utf-8"))
    repo_cache = "models--" + m["repo_id"].replace("/", "--")
    target = Path.home() / ".cache/huggingface/hub" / repo_cache / "snapshots" / m["revision"] / m["path"]
    if target.exists() and sha256(target) == m["sha256"]:
        print("Pinned pretrained baseline already verified in cache.", flush=True)
        return
    part = Path("data/deit-small-base.safetensors.part")
    part.parent.mkdir(parents=True, exist_ok=True)
    plan_path = Path("data/base-model-ranges.json")
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    else:
        start = part.stat().st_size if part.exists() else 0
        plan = {"prefix_bytes": start, "ranges": [[a, min(a + 4 * 1024 * 1024, m["bytes"]) - 1]
                                                for a in range(start, m["bytes"], 4 * 1024 * 1024)]}
        plan_path.write_text(json.dumps(plan), encoding="utf-8")

    def fetch(bounds):
        a, b = bounds
        chunk = Path(f"data/base-model.range-{a}-{b}")
        for _ in range(6):
            have = chunk.stat().st_size if chunk.exists() else 0
            if have == b - a + 1:
                return chunk
            if have > b - a + 1:
                raise ValueError("Invalid range size; preserved.")
            with chunk.open("ab") as f:
                subprocess.run(["curl.exe", "--location", "--fail", "--silent", "--show-error",
                                "--connect-timeout", "30", "--speed-limit", "1024", "--speed-time", "60",
                                "--range", f"{a + have}-{b}", "--output", "-",
                                f"https://huggingface.co/{m['repo_id']}/resolve/{m['revision']}/{m['path']}"], stdout=f)
            if chunk.stat().st_size == b - a + 1:
                return chunk
        raise RuntimeError("Base model range download failed; partials preserved.")

    with ThreadPoolExecutor(max_workers=8) as executor:
        chunks = list(executor.map(fetch, plan["ranges"]))
    assembled = Path("data/deit-small-base.assembled")
    if not assembled.exists():
        with assembled.open("xb") as dst:
            for source in ([part] if plan["prefix_bytes"] else []) + chunks:
                with source.open("rb") as src:
                    shutil.copyfileobj(src, dst)
    if assembled.stat().st_size != m["bytes"] or sha256(assembled) != m["sha256"]:
        raise ValueError("Pretrained weight verification failed; file preserved.")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256(target) != m["sha256"]:
            raise ValueError("Existing cache mismatch; file preserved.")
    else:
        with assembled.open("rb") as src, target.open("xb") as dst:
            shutil.copyfileobj(src, dst)
    print("Verified pretrained baseline:", m["repo_id"], m["revision"], flush=True)


if __name__ == "__main__":
    main()
