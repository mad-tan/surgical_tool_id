"""The shipped checkpoint is self-describing and loads offline."""
import torch

from conftest import ROOT
from toolid import checkpoint
from toolid.data import CLASSES

CKPT = ROOT / "checkpoints" / "model.pt"


def test_metadata_loads_with_weights_only_and_matches_training():
    ck = torch.load(CKPT, map_location="cpu", weights_only=True)
    assert ck["format"] == "toolid-v1"
    assert ck["classes"] == list(CLASSES) == ["clipper", "grasper", "hook", "scissor"]
    assert ck["arch"] == "mobilenet_v3_large"
    assert ck["pre"] == {"geometry": "native", "colour": "none"}
    assert ck["tta_hflip"] is False  # flips destroy the left/right cue
    assert len(ck["members"]) == 3
    info = ck["info"]
    assert info["experiment"] == "r3g_mbv3_ft_colour_affine" and info["n_train"] == 1402
    assert "hflip" not in info["config"]["augment"]


def test_loading_never_downloads(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("attempted to download weights")
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", boom)
    monkeypatch.setattr(torch.hub, "download_url_to_file", boom)
    ck, models = checkpoint.load(CKPT)
    x = torch.zeros(2, 3, 86, 128)
    with torch.no_grad():
        out = models[0](x)
    assert out.shape == (2, 4)
    assert all(p.dtype == torch.float32 for m in models for p in m.parameters())  # fp16 storage, fp32 compute
