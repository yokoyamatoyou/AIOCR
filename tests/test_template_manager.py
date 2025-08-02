from core.template_manager import TemplateManager


def test_template_manager_roundtrip(tmp_path):
    manager = TemplateManager(template_dir=str(tmp_path))
    data = {"name": "tmp", "rois": {"field": {"box": [0, 0, 10, 10]}}}
    manager.save("sample", data)

    assert manager.list_templates() == ["sample"]
    loaded = manager.load("sample")
    assert loaded == data
