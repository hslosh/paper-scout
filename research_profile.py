"""
research_profile.py — load and check the reviewed research profile.

The profile (profiles/example_profile.json by default, or RESEARCH_PROFILE in
.env) records what data you have, which periods are locked, and which ideas
you've already tested. It is drafted from your own notes, which stay the
source of truth. Approval stores a hash of each source document, so editing
those notes later marks the profile stale instead of silently trusting it.

    python research_profile.py            # show status / staleness
    python research_profile.py --approve  # after reviewing the JSON by hand
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import sys
from pathlib import Path

import config

REQUIRED = ("profile_version", "review_status", "source_documents", "objectives",
            "instruments", "datasets", "costs", "sample_locks", "closed_mechanisms", "routes")


class ProfileError(Exception):
    pass


def load(path: Path | None = None) -> dict:
    path = Path(path or config.PROFILE_PATH)
    try:
        profile = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ProfileError(f"cannot read profile {path}: {type(exc).__name__}") from None
    missing = [k for k in REQUIRED if k not in profile]
    if missing:
        raise ProfileError(f"profile missing keys: {', '.join(missing)}")
    known = set(profile["instruments"])
    for d in profile["datasets"]:
        if d.get("instrument") not in known | {"context"}:
            raise ProfileError(f"dataset {d.get('id')} names unknown instrument {d.get('instrument')}")
    for m in profile["closed_mechanisms"]:
        if not m.get("keywords") or not m.get("source"):
            raise ProfileError(f"closed mechanism {m.get('id')} needs keywords and a source")
    profile["_path"] = str(path)
    return profile


def _doc_path(profile: dict, doc: str) -> Path:
    """Source documents may be absolute or relative to the profile's folder."""
    path = Path(doc).expanduser()
    if path.is_absolute() or "_path" not in profile:
        return path
    return Path(profile["_path"]).parent / path


def source_hashes(profile: dict) -> dict:
    hashes = {}
    for name, doc in profile["source_documents"].items():
        try:
            hashes[name] = hashlib.sha256(_doc_path(profile, doc).read_bytes()).hexdigest()
        except OSError:
            hashes[name] = "missing"
    return hashes


# Review metadata is excluded from the content hash; everything else (datasets,
# locks, closed mechanisms, routes, source list) is policy that approval covers.
_REVIEW_KEYS = {"review_status", "review_note", "reviewed_source_hashes", "approved_content_sha256",
                "approved_at", "_path"}


def content_hash(profile: dict) -> str:
    policy = {k: v for k, v in profile.items() if k not in _REVIEW_KEYS}
    canonical = json.dumps(policy, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def status(profile: dict) -> dict:
    """Effective review state. 'approved' only if the policy content and every
    source document are byte-identical to what was approved; otherwise 'stale'.
    Anything but 'approved' means routes are provisional triage."""
    current = source_hashes(profile)
    reviewed = profile.get("reviewed_source_hashes") or {}
    changed = sorted(k for k, v in current.items() if reviewed.get(k) != v)
    missing = sorted(k for k, v in current.items() if v == "missing")
    content_changed = profile.get("approved_content_sha256") != content_hash(profile)
    state = profile["review_status"]
    if state == "approved" and (changed or missing or content_changed):
        state = "stale"
    return {"state": state, "provisional": state != "approved",
            "changed_sources": changed if reviewed else [], "missing_sources": missing,
            "content_changed_since_approval": bool(reviewed) and content_changed,
            "profile_sha256": hashlib.sha256(Path(profile["_path"]).read_bytes()).hexdigest()}


def approve(path: Path | None = None) -> dict:
    path = Path(path or config.PROFILE_PATH)
    profile = load(path)
    hashes = source_hashes(profile)
    missing = sorted(k for k, v in hashes.items() if v == "missing")
    if missing:
        raise ProfileError(f"cannot approve: source documents missing: {', '.join(missing)}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["review_status"] = "approved"
    raw["reviewed_source_hashes"] = hashes
    raw["approved_content_sha256"] = content_hash(raw)
    raw["approved_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return raw


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--approve", action="store_true",
                        help="Mark the profile approved and stamp current source-document hashes")
    args = parser.parse_args(argv)
    try:
        if args.approve:
            approve()
        info = status(load())
    except ProfileError as exc:
        print(f"[PROFILE] {exc}")
        return 1
    print(f"[PROFILE] {info['state']}")
    if info["content_changed_since_approval"]:
        print("  profile content edited since approval; re-review and re-approve")
    for key in ("changed_sources", "missing_sources"):
        if info[key]:
            print(f"  {key}: {', '.join(info[key])}")
    return 0 if info["state"] == "approved" else 2


if __name__ == "__main__":
    sys.exit(main())
