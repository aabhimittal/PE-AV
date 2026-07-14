import pytest

from pe_av.config import PEAVConfig


def test_presets_exist():
    for name in ["small", "base", "large"]:
        cfg = PEAVConfig.preset(name)
        assert cfg.embed_dim > 0
        # every tower shares the same hidden width in a preset
        assert cfg.frame.width == cfg.audio.width == cfg.text.width


def test_preset_unknown_raises():
    with pytest.raises(KeyError):
        PEAVConfig.preset("gigantic")


def test_yaml_roundtrip(tmp_path):
    cfg = PEAVConfig.preset("base")
    path = tmp_path / "cfg.yaml"
    cfg.to_yaml(path)
    loaded = PEAVConfig.from_yaml(path)
    assert loaded.to_dict() == cfg.to_dict()


def test_dict_roundtrip():
    cfg = PEAVConfig.preset("small")
    assert PEAVConfig.from_dict(cfg.to_dict()).to_dict() == cfg.to_dict()
