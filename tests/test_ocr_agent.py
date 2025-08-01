import os
from pathlib import Path
import json

import cv2
import numpy as np

from core.db_manager import DBManager
from core.template_manager import TemplateManager
from core.ocr_agent import OcrAgent
from core.ocr_bridge import DummyOCR, BaseOCR


def test_ocr_agent_process_document(tmp_path):
    # change working directory to temporary path to avoid polluting repo
    os.chdir(tmp_path)

    db_path = tmp_path / "ocr.db"
    db = DBManager(str(db_path))
    db.initialize()

    templates = TemplateManager(template_dir=str(tmp_path / "templates"))
    agent = OcrAgent(db=db, templates=templates)

    # create dummy image
    image = np.zeros((20, 20, 3), dtype=np.uint8)

    template_data = {"name": "test", "rois": {"field": {"box": [0, 0, 10, 10]}}}

    results, workspace = agent.process_document(
        image, "test.png", template_data, DummyOCR(), DummyOCR()
    )

    assert "field" in results
    assert results["field"]["result_id"] == 1
    assert Path(workspace).exists()
    db_results = db.fetch_results(1)
    assert db_results[0]["roi_name"] == "field"
    assert db_results[0]["text_mini"] == "ダミーテキスト(10x10)"
    assert db_results[0]["text_nano"] == "ダミーテキスト(10x10)"
    assert db_results[0]["confidence_score"] == 1.0
    assert db_results[0]["status"] == "high"
    with open(Path(workspace) / "extract.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["field"]["result_id"] == 1
    db.close()


class FaultyOCR(BaseOCR):
    async def run(self, image: np.ndarray) -> tuple[str, float]:
        return "m1sread", 0.95


def test_ocr_agent_corrections(tmp_path):
    os.chdir(tmp_path)

    db_path = tmp_path / "ocr.db"
    db = DBManager(str(db_path))
    db.initialize()

    templates = TemplateManager(template_dir=str(tmp_path / "templates"))
    agent = OcrAgent(db=db, templates=templates)

    image = np.zeros((20, 20, 3), dtype=np.uint8)
    template_data = {
        "name": "test",
        "rois": {"field": {"box": [0, 0, 10, 10]}},
        "corrections": [{"wrong": "1", "correct": "i"}],
    }

    results, _ = agent.process_document(image, "test.png", template_data, FaultyOCR())

    assert results["field"]["text"] == "misread"
    db.close()


def test_ocr_agent_multiple_images_single_job(tmp_path):
    os.chdir(tmp_path)

    db_path = tmp_path / "ocr.db"
    db = DBManager(str(db_path))
    db.initialize()

    templates = TemplateManager(template_dir=str(tmp_path / "templates"))
    agent = OcrAgent(db=db, templates=templates)

    image = np.zeros((20, 20, 3), dtype=np.uint8)
    template_data = {"name": "test", "rois": {"field": {"box": [0, 0, 10, 10]}}}

    job_id = db.create_job("test", "2025-01-01T00:00:00")
    for name in ["a.png", "b.png"]:
        agent.process_document(
            image,
            name,
            template_data,
            DummyOCR(),
            DummyOCR(),
            job_id=job_id,
        )

    db_results = db.fetch_results(job_id)
    assert len(db_results) == 2
    assert {r["image_name"] for r in db_results} == {"a.png", "b.png"}
    assert {r["result_id"] for r in db_results} == {1, 2}
    db.close()
