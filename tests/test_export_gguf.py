from scripts.export_gguf import (
    gguf_filename,
    is_hf_checkpoint,
    is_lora_dir,
    normalize_quant,
    quantize_flags,
    recipe_for,
    write_calib_text,
)


def test_normalize_and_filename():
    assert normalize_quant("q4_k_xl") == "Q4_K_XL"
    assert normalize_quant("UD-Q3_K_XL") == "Q3_K_XL"
    assert normalize_quant("Q4_K_L") == "Q4_K_XL"
    assert gguf_filename("q3_k_xl") == "Qwen3.8-35B-A3B-UD-Q3_K_XL.gguf"
    assert gguf_filename("Q4_K_XL") == "Qwen3.8-35B-A3B-UD-Q4_K_XL.gguf"


def test_unknown_quant_rejected():
    import pytest

    with pytest.raises(ValueError, match="Q4_K_XL"):
        recipe_for("Q3_K_S")


def test_xl_recipes_keep_embed_output_and_ssm():
    q3 = recipe_for("Q3_K_XL")
    q4 = recipe_for("Q4_K_XL")
    assert q3["base"] == "Q3_K_M"
    assert q4["base"] == "Q4_K_M"
    assert q3["token_embedding"] == q3["output"] == "q8_0"
    assert q4["token_embedding"] == q4["output"] == "q8_0"
    assert "ssm_out=q8_0" in q3["tensor_types"]
    assert "ssm_out=q8_0" in q4["tensor_types"]
    assert "ffn_down=q5_k" in q3["tensor_types"]
    assert "ffn_down=q6_k" in q4["tensor_types"]

    flags = quantize_flags("Q4_K_XL")
    assert flags[:4] == ["--token-embedding-type", "q8_0", "--output-tensor-type", "q8_0"]
    assert flags.count("--tensor-type") == len(q4["tensor_types"])


def test_write_calib_and_checkpoint_detectors(tmp_path):
    calib = write_calib_text(tmp_path / "calib.txt", texts=["hello", "world"])
    text = calib.read_text(encoding="utf-8")
    assert "hello" in text and "world" in text

    lora = tmp_path / "lora"
    lora.mkdir()
    (lora / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert is_lora_dir(lora)
    assert not is_hf_checkpoint(lora)

    hf = tmp_path / "hf"
    hf.mkdir()
    (hf / "model.safetensors").write_bytes(b"x")
    assert is_hf_checkpoint(hf)
