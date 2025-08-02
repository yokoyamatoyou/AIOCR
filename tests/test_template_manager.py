import json

from core.template_manager import TemplateManager


def test_template_manager_roundtrip(tmp_path):
    manager = TemplateManager(template_dir=str(tmp_path))
    data = {
        "name": "tmp",
        "keywords": ["a", "b"],
        "rois": {"field": {"box": [0, 0, 10, 10]}},
        "template_image_path": "templates/tmp.png",
        "corrections": [],
    }
    manager.save("sample", data)

    assert manager.list_templates() == ["sample"]
    assert manager.get_keywords("sample") == ["a", "b"]
    loaded = manager.load("sample")
    assert loaded == data


def test_detect_template(tmp_path):
    manager = TemplateManager(template_dir=str(tmp_path))
    manager.save("invoice", {"name": "invoice", "keywords": ["invoice"], "rois": {}})
    manager.save("receipt", {"name": "receipt", "keywords": ["receipt"], "rois": {}})
    detected = manager.detect_template("This is an invoice document")
    assert detected is not None
    name, _ = detected
    assert name == "invoice"


def test_load_normalises_legacy_structures(tmp_path):
    manager = TemplateManager(template_dir=str(tmp_path))
    legacy = {
        "name": "legacy",
        "rois": {},
        "keywords": "invoice",
        "corrections": {"OLD": "NEW"},
    }
    with (tmp_path / "legacy.json").open("w", encoding="utf-8") as f:
        json.dump(legacy, f, ensure_ascii=False)

    data = manager.load("legacy")
    assert data["keywords"] == []
    assert data["corrections"] == [{"wrong": "OLD", "correct": "NEW"}]
