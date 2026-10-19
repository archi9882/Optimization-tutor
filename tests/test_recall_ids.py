"""Recall-card ids are positional; the committed snapshot guards against remaps."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("recall_ids", ROOT / "scripts" / "recall_ids.py")
rid = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rid)

HOWTO = ("Recall-card ids are positional (algo#qN) and learners' progress is stored "
         "under them. Keep the question order (append new questions at the end), or, "
         "if a question was only reworded, run: python3 scripts/recall_ids.py --update "
         "--accept <id>. See README「回想卡 id」.")


def test_snapshot_matches_content() -> None:
    """Every built card matches its recorded question; no recorded card vanished."""
    d = rid.diff(rid.load_snapshot(), rid.current_cards())
    msgs = rid.problems(d)
    assert not msgs, "\n".join(msgs) + "\n" + HOWTO
    assert not d["new"], (f"{len(d['new'])} new card(s) not in content/recall_ids.json: "
                          f"{d['new'][:5]}; run python3 scripts/recall_ids.py --update")


def test_hash_ignores_whitespace_only() -> None:
    assert rid.question_hash("Q1. 為什麼？") == rid.question_hash("  Q1.\n為什麼？ ")
    assert rid.question_hash("Q1. 為什麼？") != rid.question_hash("Q1. 為何？")


def test_reorder_is_detected() -> None:
    """Swapping two questions flags both ids as changed."""
    cards = {"x#q1": "Q1. A", "x#q2": "Q2. B"}
    snap = rid.updated({}, cards, set())
    d = rid.diff(snap, {"x#q1": "Q1. B", "x#q2": "Q2. A"})
    assert d["changed"] == ["x#q1", "x#q2"] and not d["new"] and not d["removed"]


def test_update_requires_explicit_accept() -> None:
    snap = rid.updated({}, {"x#q1": "Q1. A", "x#q2": "Q2. B"}, set())
    reworded = {"x#q1": "Q1. A（改寫）", "x#q2": "Q2. B", "x#q3": "Q3. C"}
    with pytest.raises(ValueError, match="x#q1"):
        rid.updated(snap, reworded, set())
    with pytest.raises(ValueError, match="x#q2"):
        rid.updated(snap, {"x#q1": "Q1. A"}, set())      # removed card
    new = rid.updated(snap, reworded, {"x#q1"})
    assert list(new) == ["x#q1", "x#q2", "x#q3"]
    assert new["x#q1"]["sha"] == rid.question_hash("Q1. A（改寫）")
    assert new["x#q2"] == snap["x#q2"]


def test_order_is_numeric() -> None:
    cards = {f"x#q{n}": f"Q{n}" for n in (10, 2, 1)}
    assert list(rid.updated({}, cards, set())) == ["x#q1", "x#q2", "x#q10"]
