from qwen3_8_moe.chat import Qwen38ChatFormatter
from scripts.distill_from_qwen38 import map_student_layer_to_teacher


def test_thinking_prompt_and_effort_hint():
    formatter = Qwen38ChatFormatter(enable_thinking=True, reasoning_effort="low")
    prompt = formatter.format_messages([{"role": "user", "content": "Hi"}])
    assert prompt.startswith("<|im_start|>system")
    assert "Reason briefly" in prompt
    assert prompt.endswith("<|im_start|>assistant\n<think>\n")
    assert formatter.sampling().temperature == 1.0


def test_instruct_mode_and_strip_history():
    formatter = Qwen38ChatFormatter(enable_thinking=False, preserve_thinking=False)
    prompt = formatter.format_messages(
        [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "<think>secret</think>\nHello"},
            {"role": "user", "content": "Again"},
        ]
    )
    assert "<think>secret</think>" not in prompt
    assert "Hello" in prompt
    assert not prompt.endswith("<think>\n")
    assert formatter.sampling().presence_penalty == 1.5


def test_split_thinking():
    thinking, answer = Qwen38ChatFormatter.split_thinking("<think>plan</think>\nDone")
    assert thinking == "plan"
    assert answer == "Done"


def test_distill_layer_map_preserves_mixer_type():
    for student in range(40):
        teacher = map_student_layer_to_teacher(student)
        assert teacher % 4 == student % 4
        assert 0 <= teacher < 64
    assert map_student_layer_to_teacher(0) == 0
    assert map_student_layer_to_teacher(39) == 63
