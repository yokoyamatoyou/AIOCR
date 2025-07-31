
import pytest
import os
import json
import shutil
import cv2
import numpy as np

# srcディレクトリをパスに追加
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from app.ocr_bridge import DummyOCR
from app.ocr_processor import OCRProcessor

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
    """ OCRProcessorが正しくJSONファイルを生成するかテスト """
    workspace_dir = setup_workspace
    
    # ダミーOCRエンジンでプロセッサを初期化
    ocr_engine = DummyOCR()
    processor = OCRProcessor(ocr_engine, workspace_dir)
    
    # 処理を実行
    results = processor.process_all()

    # 1. 返り値の検証
    assert "field_a" in results
    assert "field_b" in results
    assert results["field_a"]["text"] == "ダミーテキスト(100x50)"
    assert results["field_b"]["confidence"] == 0.95

    # 2. 生成されたJSONファイルの検証
    json_path = os.path.join(workspace_dir, "extract.json")
    assert os.path.exists(json_path)

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    assert "field_a" in data
    assert data["field_b"]["source_image"] == "P2_field_b.png"
