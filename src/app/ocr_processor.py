
import os
import cv2
import json
from typing import Optional, Dict, Any
from .ocr_bridge import BaseOCR
from . import postprocess

class OCRProcessor:
    """OCR処理全体を管理するクラス"""

    def __init__(self, ocr_engine: BaseOCR, workspace_dir: str, rois: Optional[Dict[str, Any]] = None):
        self.ocr_engine = ocr_engine
        self.workspace_dir = workspace_dir
        self.crops_dir = os.path.join(self.workspace_dir, "crops")
        self.rois = rois or {}

    def process_all(self) -> dict:
        """
        cropsディレクトリ内のすべての画像を処理し、結果をJSONにまとめる
        """
        results: Dict[str, Any] = {}
        crop_files = sorted(os.listdir(self.crops_dir))

        for filename in crop_files:
            if filename.endswith(".png"):
                # P1_field_a.png から field_a をキーとして抽出
                key = "_".join(filename.split("_")[1:]).replace(".png", "")
                
                image_path = os.path.join(self.crops_dir, filename)
                image = cv2.imread(image_path)

                text, confidence = self.ocr_engine.run(image)

                rule = None
                if key in self.rois:
                    rule = self.rois[key].get("validation_rule")

                normalized, needs_human = postprocess.postprocess_result(
                    text, confidence, rule
                )

                entry = {
                    "text": normalized,
                    "confidence": confidence,
                    "source_image": filename,
                }
                if needs_human:
                    entry["needs_human"] = True

                results[key] = entry
        
        # 結果をextract.jsonに保存
        output_path = os.path.join(self.workspace_dir, "extract.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
            
        return results
