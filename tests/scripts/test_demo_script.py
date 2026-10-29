import sys

import scripts.demo as demo


def test_auto_weights_and_explicit_override(monkeypatch, tmp_path):
    found = {"seg": tmp_path / "seg.pt", "pose": tmp_path / "pose.pt", "regress": None}
    monkeypatch.setattr(demo, "latest_weights", lambda name: found[name])
    launched = {}
    monkeypatch.setattr(demo, "launch", lambda a: launched.update(vars(a)))
    monkeypatch.setattr(sys, "argv", ["scripts.demo", "--pose-weights", "mine.pt", "--port", "7999", "--no-ocr"])
    demo.main()
    assert launched["seg_weights"] == str(tmp_path / "seg.pt")       # found automatically
    assert launched["pose_weights"] == "mine.pt"                     # explicit wins
    assert launched["regress_weights"] is None                       # nothing found: model off
    assert launched["port"] == 7999 and launched["no_ocr"] is True
