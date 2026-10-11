"""predict.py behaves as the graders need, on CPU, with the shipped checkpoint."""
import shutil

from PIL import Image

from conftest import ROOT, read_csv, run_predict

CLASSES = {"clipper", "grasper", "hook", "scissor"}


def test_class_subfolders_one_row_per_image_even_with_duplicate_names(tmp_path, real_images):
    d = tmp_path / "data"
    for c, files in real_images.items():
        (d / c).mkdir(parents=True)
        for f in files:
            shutil.copy(f, d / c / f.name)
    # the same filename in two class folders (as in the real data: two crops of one frame)
    shutil.copy(real_images["hook"][0], d / "grasper" / "9999_100.png")
    shutil.copy(real_images["grasper"][0], d / "hook" / "9999_100.png")
    out = tmp_path / "preds.csv"
    r = run_predict(d, out)
    assert r.returncode == 0, r.stderr
    rows = read_csv(out)
    assert rows[0] == ["filename", "predicted_class"]
    body = rows[1:]
    assert len(body) == 8 + 2  # exactly one row per image file
    assert [n for n, _ in body].count("9999_100.png") == 2
    assert {p for _, p in body} <= CLASSES
    assert "occur in more than one subfolder" in r.stderr


def test_flat_folder_uppercase_extension_hidden_files(tmp_path, real_images):
    d = tmp_path / "flat"
    d.mkdir()
    src = real_images["scissor"][0]
    shutil.copy(src, d / "a_1.png")
    shutil.copy(src, d / "B_2.PNG")
    (d / ".DS_Store").write_bytes(b"\x00\x01")
    (d / ".hidden").mkdir()
    shutil.copy(src, d / ".hidden" / "c_3.png")
    (d / "notes.txt").write_text("not an image")
    out = tmp_path / "p.csv"
    r = run_predict(d, out)
    assert r.returncode == 0, r.stderr
    assert sorted(n for n, _ in read_csv(out)[1:]) == ["B_2.PNG", "a_1.png"]


def test_unreadable_image_fails_loudly_without_guessing(tmp_path, real_images):
    d = tmp_path / "data" / "hook"
    d.mkdir(parents=True)
    shutil.copy(real_images["hook"][0], d / "ok_1.png")
    (d / "broken_2.png").write_bytes(b"not a png")
    out = tmp_path / "p.csv"
    r = run_predict(tmp_path / "data", out)
    assert r.returncode == 2
    assert "broken_2.png" in r.stderr
    assert not out.exists()


def test_empty_folder_is_an_error(tmp_path):
    (tmp_path / "empty").mkdir()
    r = run_predict(tmp_path / "empty", tmp_path / "p.csv")
    assert r.returncode != 0 and "no .png images" in r.stderr


def test_other_image_sizes_are_resized_not_rejected(tmp_path, real_images):
    d = tmp_path / "flat"
    d.mkdir()
    Image.open(real_images["clipper"][0]).resize((256, 172)).save(d / "big_1.png")
    Image.open(real_images["clipper"][0]).convert("L").save(d / "gray_2.png")
    r = run_predict(d, tmp_path / "p.csv")
    assert r.returncode == 0, r.stderr
    assert len(read_csv(tmp_path / "p.csv")) == 3


def test_preflight_contract_passes():
    import subprocess
    import sys
    r = subprocess.run([sys.executable, "check_submission.py"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0 and "PREFLIGHT PASSED" in r.stdout, r.stdout + r.stderr
