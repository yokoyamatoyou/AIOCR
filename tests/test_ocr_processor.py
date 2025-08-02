
import asyncio
import json
import os
import shutil
import time

import cv2
import numpy as np
import pytest

# srcディレクトリをパスに追加
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from core.ocr_bridge import DummyOCR, BaseOCR
from core.ocr_processor import OCRProcessor

@pytest.fixture
def setup_workspace():
    """ テスト用のワークスペースとダミーの切り出し画像を作成 """
    workspace_dir = "test_workspace"
    crops_dir = os.path.join(workspace_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)

    # ダミーの切り出し画像を2つ作成
    img1 = np.zeros((50, 100, 3), dtype=np.uint8)
    cv2.imwrite(os.path.join(crops_dir, "P1_field_a.png"), img1)

    img2 = np.zeros((60, 120, 3), dtype=np.uint8)
    cv2.imwrite(os.path.join(crops_dir, "P2_field_b.png"), img2)

    yield workspace_dir # テスト関数にワークスペースのパスを渡す

    # テスト終了後にクリーンアップ
    shutil.rmtree(workspace_dir)

def test_process_all(setup_workspace):
    """OCRProcessorが正しくJSONファイルを生成するかテスト"""
    workspace_dir = setup_workspace

    ocr_engine = DummyOCR()
    processor = OCRProcessor(ocr_engine, workspace_dir)

    results = asyncio.run(processor.process_all())

    assert "field_a" in results
    assert "field_b" in results
    assert results["field_a"]["text"] == "ダミーテキスト(100x50)"
    assert results["field_b"]["confidence"] == 0.95

    json_path = os.path.join(workspace_dir, "extract.json")
    assert os.path.exists(json_path)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "field_a" in data
    assert data["field_b"]["source_image"] == "P2_field_b.png"


class ZeroOCR(BaseOCR):
    async def run(self, image: np.ndarray) -> tuple[str, float]:
        return "0000", 0.99


class OneOCR(BaseOCR):
    async def run(self, image: np.ndarray) -> tuple[str, float]:
        return "1111", 0.99


def test_double_check_confidence(tmp_path):
    """二重チェックによる信頼度判定をテスト"""

    workspace_dir = tmp_path / "ws"
    crops_dir = workspace_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    img = np.zeros((20, 40, 3), dtype=np.uint8)
    cv2.imwrite(str(crops_dir / "P1_field_a.png"), img)

    rois = {"field_a": {"validation_rule": "regex:\\d{4}"}}

    processor = OCRProcessor(ZeroOCR(), str(workspace_dir), validator_engine=OneOCR(), rois=rois)
    results = asyncio.run(processor.process_all())

    entry = results["field_a"]
    assert entry["text_mini"] == "0000"
    assert entry["text_nano"] == "1111"
    assert entry["confidence_level"] == "medium"
    assert entry["needs_human"] is True


class SleepOCR(BaseOCR):
    async def run(self, image: np.ndarray) -> tuple[str, float]:
        await asyncio.sleep(0.1)
        h, w = image.shape[:2]
        return f"ダミーテキスト({w}x{h})", 0.95


def test_process_all_parallel_execution(setup_workspace):
    """OCR処理が並列に実行されることを確認"""
    workspace_dir = setup_workspace
    ocr_engine = SleepOCR()
    processor = OCRProcessor(ocr_engine, workspace_dir)

    start = time.perf_counter()
    results = asyncio.run(processor.process_all())
    elapsed = time.perf_counter() - start

    assert elapsed < 0.18
    assert results["field_a"]["text"] == "ダミーテキスト(100x50)"
    assert results["field_b"]["text"] == "ダミーテキスト(120x60)"
