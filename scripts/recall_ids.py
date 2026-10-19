"""Snapshot of recall-card ids, so reordering questions cannot remap progress.

Recall cards are numbered by position (``<algo>#q<n>``, see build.py) and
learners' progress is stored under those ids. Moving or rewriting a question
would silently attach someone's history to a different question. This script
keeps ``content/recall_ids.json`` (id -> hash and text of its question) and
``tests/test_recall_ids.py`` fails when the built questions no longer match it.

    python3 scripts/recall_ids.py                 # check (exit 1 on mismatch)
    python3 scripts/recall_ids.py --update        # record new cards only
    python3 scripts/recall_ids.py --update --accept gd#q3 [--accept ...]
                                                  # also accept a reworded card

New cards appended at the end are recorded by ``--update`` without question.
A changed or removed id needs an explicit ``--accept``: use it only when the
question still asks the same thing (wording fix). Otherwise keep the order:
add new questions at the end of the 面試追問 section instead of in between.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = ROOT / "content" / "recall_ids.json"

_spec = importlib.util.spec_from_file_location("_build", ROOT / "scripts" / "build.py")
_build = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_build)


def question_hash(text: str) -> str:
    """Short sha256 of a question's plain text, whitespace-normalised."""
    norm = re.sub(r"\s+", " ", text).strip()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:12]


def current_cards() -> dict[str, str]:
    """Every recall card in the content, id -> plain question text."""
    out: dict[str, str] = {}
    for md in sorted((ROOT / "content" / "algorithms").glob("*.md")):
        _, body = _build.split_frontmatter(md.read_text("utf-8"))
        _, cards = _build.extract_recall(md.stem, _build.render_markdown(body))
        out.update({c["id"]: c["q"] for c in cards})
    return out


def load_snapshot(path: Path = SNAPSHOT) -> dict[str, dict]:
    """Read the committed snapshot (empty if missing)."""
    return json.loads(path.read_text("utf-8")) if path.exists() else {}


def diff(snapshot: dict[str, dict], cards: dict[str, str]) -> dict[str, list[str]]:
    """Compare snapshot with content: ids that are new, changed or removed."""
    return {
        "new": [i for i in cards if i not in snapshot],
        "changed": [i for i in cards if i in snapshot
                    and snapshot[i]["sha"] != question_hash(cards[i])],
        "removed": [i for i in snapshot if i not in cards],
    }


def problems(d: dict[str, list[str]]) -> list[str]:
    """Human-readable messages for changed/removed ids (new ids are fine)."""
    msgs = [f"{i}: question text changed (reordered or reworded?)" for i in d["changed"]]
    msgs += [f"{i}: card no longer exists" for i in d["removed"]]
    return msgs


def updated(snapshot: dict[str, dict], cards: dict[str, str],
            accept: set[str]) -> dict[str, dict]:
    """New snapshot: add new ids, refresh/remove only the accepted ones."""
    d = diff(snapshot, cards)
    blocked = (set(d["changed"]) | set(d["removed"])) - accept
    if blocked:
        raise ValueError("not accepted: " + ", ".join(sorted(blocked)))
    out = {i: v for i, v in snapshot.items() if i in cards}
    for i in d["new"] + d["changed"]:
        out[i] = {"sha": question_hash(cards[i]), "q": cards[i]}
    return dict(sorted(out.items(), key=lambda kv: _order(kv[0])))


def _order(cid: str) -> tuple[str, int]:
    algo, _, n = cid.rpartition("#q")
    return algo, int(n)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--update", action="store_true", help="write the snapshot")
    ap.add_argument("--accept", action="append", default=[], metavar="ID",
                    help="accept a changed/removed card id (with --update)")
    args = ap.parse_args(argv)
    snap, cards = load_snapshot(), current_cards()
    if args.update:
        try:
            new = updated(snap, cards, set(args.accept))
        except ValueError as e:
            print(f"recall_ids: {e}", file=sys.stderr)
            return 1
        SNAPSHOT.write_text(json.dumps(new, ensure_ascii=False, indent=1) + "\n", "utf-8")
        print(f"recall_ids: {len(new)} cards -> {SNAPSHOT.relative_to(ROOT)}")
        return 0
    d = diff(snap, cards)
    for msg in problems(d):
        print(f"recall_ids: {msg}", file=sys.stderr)
    if d["new"]:
        print(f"recall_ids: {len(d['new'])} new card(s) not yet recorded; run --update")
    return 1 if problems(d) or d["new"] else 0


if __name__ == "__main__":
    sys.exit(main())
