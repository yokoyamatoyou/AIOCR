import os
from pathlib import Path

import cv2
import numpy as np

from core.db_manager import DBManager
from core.template_manager import TemplateManager
from core.ocr_agent import OcrAgent
from core.ocr_bridge import DummyOCR
from core import preprocess


def test_roi_alignment_with_shift(tmp_path, monkeypatch):
    os.chdir(tmp_path)

    # avoid skew correction for this test
    monkeypatch.setattr(preprocess, "correct_skew", lambda img: img)

    # create template image with distinctive features
    template_img = np.full((200, 200, 3), 255, dtype=np.uint8)
    cv2.putText(template_img, "TEST", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.circle(template_img, (150, 150), 20, (0, 0, 0), -1)
    template_path = tmp_path / "template.png"
    cv2.imwrite(str(template_path), template_img)

    # shift image
    M = np.float32([[1, 0, 20], [0, 1, 10]])
    shifted_img = cv2.warpAffine(template_img, M, (200, 200), borderValue=(255, 255, 255))

    # prepare environment
    db = DBManager(str(tmp_path / "ocr.db"))
    db.initialize()
    templates = TemplateManager(template_dir=str(tmp_path / "templates"))
    agent = OcrAgent(db=db, templates=templates)

    template_data = {
        "name": "test",
        "template_image_path": str(template_path),
        "rois": {"field": {"box": [40, 60, 80, 40]}},
    }

    results, workspace = agent.process_document(
        shifted_img, "shifted.png", template_data, DummyOCR(), DummyOCR()
    )

    crop_path = Path(workspace) / "crops" / "P1_field.png"
    assert crop_path.exists()
    cropped = cv2.imread(str(crop_path))
    expected = shifted_img[70:110, 60:140]
    assert np.array_equal(cropped, expected)
    assert results["field"]["text"] == "ダミーテキスト(80x40)"
    db.close()
