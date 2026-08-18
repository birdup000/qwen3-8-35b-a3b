from scripts.colab_pipeline import (
    detect_runtime,
    expert_module_goes_to_cpu,
    gpu_weight_budget_gib,
    is_moe_checkpoint,
    run_tiny_pipeline,
    should_load_fourbit_gpu_only,
    write_sample_data,
)


def test_detect_runtime_keys():
    info = detect_runtime()
    assert "gpu" in info and "recommend_full" in info
    assert info["vram_gb"] >= 0


def test_fourbit_gpu_only_skips_moe_experts():
    assert is_moe_checkpoint("Qwen/Qwen3.6-35B-A3B")
    assert not is_moe_checkpoint("Qwen/Qwen3.8-27B")
    assert should_load_fourbit_gpu_only(38.8, "Qwen/Qwen3.8-27B")
    assert not should_load_fourbit_gpu_only(38.8, "Qwen/Qwen3.6-35B-A3B")
    assert not should_load_fourbit_gpu_only(22.0, "Qwen/Qwen3.8-27B")
    assert gpu_weight_budget_gib(free_gb=39.1, total_gb=40.0) == 18.0
    assert gpu_weight_budget_gib(free_gb=78.0, total_gb=80.0) == 40.0
    assert expert_module_goes_to_cpu("model.layers.0.mlp.experts")
    assert expert_module_goes_to_cpu("language_model.layers.9.mlp.experts.down_proj")
    assert not expert_module_goes_to_cpu("model.layers.0.mlp.shared_expert")
    assert not expert_module_goes_to_cpu("model.layers.0.self_attn.q_proj")


def test_tiny_pipeline_and_sample_data(tmp_path):
    path = write_sample_data(tmp_path / "distill.jsonl")
    assert path.is_file()
    assert path.read_text(encoding="utf-8").count("\n") >= 8
    result = run_tiny_pipeline(steps=2, seq_len=16, device="cpu")
    assert result["generated_shape"][1] == 16
    assert result["layer_map_example"][39] == 63
