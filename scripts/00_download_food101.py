"""Download only official training shards, with resumption and SHA-256 checks."""
from concurrent.futures import ThreadPoolExecutor, wait
import hashlib
import json
from pathlib import Path
import subprocess


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    manifest = json.loads(Path("audit/food101_train_manifest.json").read_text(encoding="utf-8"))
    out = Path("data/food101_train")
    out.mkdir(parents=True, exist_ok=True)

    # Save a stable range plan so interrupted downloads can resume without
    # discarding already downloaded prefixes or changing chunk boundaries.
    plan_path = out / "ranges.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    else:
        plan = []
        for item in manifest["files"]:
            prefix = out / (Path(item["path"]).name + ".part")
            start = prefix.stat().st_size if prefix.exists() else 0
            plan.append({**item, "prefix_bytes": start,
                         "ranges": [[a, min(a + 64 * 1024 * 1024, item["bytes"]) - 1]
                                    for a in range(start, item["bytes"], 64 * 1024 * 1024)]})
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    def download_range(item, start, end):
        name = Path(item["path"]).name
        chunk = out / f"{name}.range-{start}-{end}"
        expected = end - start + 1
        url = f"https://huggingface.co/datasets/{manifest['dataset']}/resolve/{manifest['revision']}/{item['path']}"
        for _ in range(6):
            have = chunk.stat().st_size if chunk.exists() else 0
            if have == expected:
                return
            if have > expected:
                raise ValueError(f"Range size mismatch; preserved: {chunk}")
            with chunk.open("ab") as f:
                result = subprocess.run(["curl.exe", "--location", "--fail", "--silent", "--show-error",
                                         "--connect-timeout", "30", "--speed-limit", "1024", "--speed-time", "60",
                                         "--range", f"{start + have}-{end}", "--output", "-", url], stdout=f)
            if result.returncode == 0 and chunk.stat().st_size == expected:
                return
        raise RuntimeError(f"Range transfer failed; preserved: {chunk}")

    def assemble(item):
        name = Path(item["path"]).name
        target = out / name
        if target.exists():
            if target.stat().st_size == item["bytes"] and digest(target) == item["sha256"]:
                return name
            raise ValueError(f"Existing file failed verification; preserved: {target}")
        part = out / (name + ".assembled")
        sources = ([out / (name + ".part")] if item["prefix_bytes"] else [])
        sources += [out / f"{name}.range-{a}-{b}" for a, b in item["ranges"]]
        if not part.exists():
            with part.open("xb") as dest:
                for source in sources:
                    with source.open("rb") as src:
                        for block in iter(lambda: src.read(8 * 1024 * 1024), b""):
                            dest.write(block)
        if part.stat().st_size != item["bytes"] or digest(part) != item["sha256"]:
            raise ValueError(f"Downloaded file failed SHA-256/size verification; preserved: {part}")
        part.rename(target)
        print(f"Verified {name}", flush=True)
        return name

    with ThreadPoolExecutor(max_workers=24) as executor:
        futures = {executor.submit(download_range, item, a, b) for item in plan for a, b in item["ranges"]
                   if not (out / Path(item["path"]).name).exists()}
        pending = futures
        while pending:
            _, pending = wait(pending, timeout=30)
            nbytes = sum(p.stat().st_size for p in out.iterdir() if p.is_file() and p.name != "ranges.json")
            print(f"Downloaded {nbytes / 1e9:.3f} GB; pending ranges: {len(pending)}", flush=True)
        for f in futures:
            f.result()
    for item in plan:
        assemble(item)
    print("All official training shards verified; no test shards downloaded.", flush=True)


if __name__ == "__main__":
    main()
