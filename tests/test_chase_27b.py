from pathlib import Path

from scripts.chase_27b import (
    extract_prompt_only,
    looks_eval_leak,
    require_stage_a_done,
    seed_adapter,
    seed_stage_a_jsonl,
    stage_a_steps_done,
)
from scripts.hq_distill import append_jsonl, load_prompt_pool


def test_extract_ot3_drops_qwq_answer():
    row = {
        "domain": "code",
        "conversations": [
            {"from": "human", "value": "Write a Fenwick tree."},
            {"from": "gpt", "value": "<think>plan</think>\ndef fenwick(): ..."},
        ],
    }
    messages = extract_prompt_only(row, "ot3")
    assert messages == [{"role": "user", "content": "Write a Fenwick tree."}]


def test_extract_magicoder_and_arena():
    mag = extract_prompt_only({"problem": "Implement LRU cache"}, "magicoder")
    assert mag is not None and mag[0]["content"] == "Implement LRU cache"
    arena = extract_prompt_only({"turns": [{"content": "Hard coding task"}]}, "arena_hard")
    assert arena is not None and "Hard coding" in arena[0]["content"]


def test_eval_leak_filter():
    assert looks_eval_leak("please solve", source="livecodebench")
    assert looks_eval_leak("AIME 2026 problem 3")
    assert looks_eval_leak("hard query", source="arena-hard-auto")
    assert not looks_eval_leak("Write a Fenwick tree in Rust", source="openthoughts")


def test_ocr_skips_test_split_keeps_train():
    skipped = extract_prompt_only({"input": "sort an array", "split": "test"}, "ocr")
    assert skipped is None
    mixed = extract_prompt_only({"input": "sort an array", "split": "test, train"}, "ocr")
    assert mixed is None
    kept = extract_prompt_only({"input": "Implement Dijkstra", "split": "train"}, "ocr")
    assert kept is not None
    assert "Dijkstra" in kept[0]["content"]


def test_seed_adapter_copies_once(tmp_path: Path):
    src = tmp_path / "hq_maxmix" / "adapter"
    src.mkdir(parents=True)
    (src / "adapter_config.json").write_text("{}", encoding="utf-8")
    dest = tmp_path / "hq_27b" / "adapter"
    first = seed_adapter(src, dest)
    (dest / "marker").write_text("keep", encoding="utf-8")
    seed_adapter(src, dest)
    assert first == dest
    assert (dest / "marker").read_text(encoding="utf-8") == "keep"


def test_seed_stage_a_jsonl_copies_once(tmp_path: Path):
    src = tmp_path / "hq_maxmix" / "stage_a.jsonl"
    src.parent.mkdir(parents=True)
    src.write_text('{"id":"a"}\n', encoding="utf-8")
    dest = tmp_path / "hq_27b" / "stage_a.jsonl"
    seed_stage_a_jsonl(src, dest)
    dest.write_text('{"id":"kept"}\n', encoding="utf-8")
    seed_stage_a_jsonl(src, dest)
    assert dest.read_text(encoding="utf-8") == '{"id":"kept"}\n'


def test_require_stage_a_blocks_until_target(tmp_path: Path):
    adapter = tmp_path / "hq_maxmix" / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "state.json").write_text('{"a_step": 90, "stage": "A"}\n', encoding="utf-8")
    assert stage_a_steps_done(adapter) == 90
    try:
        require_stage_a_done(adapter, min_a_step=4000)
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "90/4000" in str(exc)
    (adapter / "state.json").write_text('{"a_step": 4000, "stage": "A"}\n', encoding="utf-8")
    assert require_stage_a_done(adapter, min_a_step=4000) == 4000


def test_load_prompt_pool_prefers_chase_jsonl(tmp_path: Path):
    append_jsonl(
        tmp_path / "prompts.jsonl",
        {"id": "ot3-aa", "messages": [{"role": "user", "content": "prove it"}]},
    )
    append_jsonl(
        tmp_path / "stage_a.jsonl",
        {
            "id": "old",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
        },
    )
    pool = load_prompt_pool(tmp_path, max_traces=10)
    assert pool[0]["id"] == "ot3-aa"
    assert pool[0]["messages"][-1]["role"] == "user"
