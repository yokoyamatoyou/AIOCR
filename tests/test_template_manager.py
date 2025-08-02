from core.template_manager import TemplateManager


def test_template_manager_roundtrip(tmp_path):
    manager = TemplateManager(template_dir=str(tmp_path))
    data = {
        "name": "tmp",
        "keywords": ["a", "b"],
        "rois": {"field": {"box": [0, 0, 10, 10]}},
        "template_image_path": "templates/tmp.png",
    }
    manager.save("sample", data)

    assert manager.list_templates() == ["sample"]
    assert manager.get_keywords("sample") == ["a", "b"]
    loaded = manager.load("sample")
    assert loaded == data
