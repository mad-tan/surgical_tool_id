"""Preprocessing, augmentation and the resumable trainer."""
import time

import pytest
import torch

from toolid import augment, experiments, preprocess, trainer


def test_geometry_never_stretches():
    x = torch.rand(2, 3, 86, 128)
    assert preprocess.geometry(x, "native").shape == (2, 3, 86, 128)
    assert preprocess.geometry(x, "pad128").shape == (2, 3, 128, 128)
    assert torch.equal(preprocess.geometry(x, "pad128")[:, :, 21:107], x)  # content untouched, black bars only


def test_unreadable_file_raises(tmp_path):
    p = tmp_path / "bad.png"
    p.write_bytes(b"garbage")
    with pytest.raises(preprocess.UnreadableImage):
        preprocess.load_rgb(p)


def test_no_flip_in_any_finetuning_config_and_augment_stays_in_range():
    for name, cfg in experiments.EXPERIMENTS.items():
        if name.startswith("r3"):
            assert not cfg["augment"].get("hflip"), name
    torch.manual_seed(0)
    a = augment.Augment(experiments.get("r3g_mbv3_ft_colour_affine")["augment"])
    y = a(torch.rand(4, 3, 86, 128))
    assert y.min() >= 0 and y.max() <= 1 and y.shape == (4, 3, 86, 128)


def test_paused_and_resumed_training_is_bit_identical(tmp_path):
    cfg = experiments.get("r1a_smallcnn_fixed")
    cfg.update(epochs=3, batch_size=8)
    g = torch.Generator().manual_seed(0)
    x = torch.randint(0, 256, (24, 3, 86, 128), dtype=torch.uint8, generator=g)
    y = torch.arange(24) % 4
    m1 = trainer.train(cfg, x, y, seed=3)
    state = tmp_path / "s.pt"
    m2 = None
    while m2 is None:  # one epoch per call
        st = torch.load(state, weights_only=False) if state.exists() else None
        m2 = trainer.train(cfg, x, y, seed=3, state_path=state,
                           deadline=time.time() + (st["secs_per_epoch"] * 1.5 if st else 0.01))
    assert all(torch.equal(a, b) for a, b in zip(m1.state_dict().values(), m2.state_dict().values()))
