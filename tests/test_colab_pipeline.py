from scripts.colab_pipeline import detect_runtime, run_tiny_pipeline, should_load_fourbit_gpu_only, write_sample_data


def test_detect_runtime_keys():
    info = detect_runtime()
    assert "gpu" in info and "recommend_full" in info
    assert info["vram_gb"] >= 0


def test_fourbit_gpu_only_threshold():
    assert should_load_fourbit_gpu_only(38.8)
    assert should_load_fourbit_gpu_only(28.0)
    assert not should_load_fourbit_gpu_only(22.0)


def test_tiny_pipeline_and_sample_data(tmp_path):
    path = write_sample_data(tmp_path / "distill.jsonl")
    assert path.is_file()
    assert path.read_text(encoding="utf-8").count("\n") >= 8
    result = run_tiny_pipeline(steps=2, seq_len=16, device="cpu")
    assert result["generated_shape"][1] == 16
    assert result["layer_map_example"][39] == 63
