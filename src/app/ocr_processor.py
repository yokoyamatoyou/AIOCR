
import os
import cv2
import json
from .ocr_bridge import BaseOCR

class OCRProcessor:
    """OCR処理全体を管理するクラス"""
    def __init__(self, ocr_engine: BaseOCR, workspace_dir: str):
        self.ocr_engine = ocr_engine
        self.workspace_dir = workspace_dir
        self.crops_dir = os.path.join(self.workspace_dir, "crops")

    def process_all(self) -> dict:
        """
        cropsディレクトリ内のすべての画像を処理し、結果をJSONにまとめる
        """
        results = {}
        crop_files = sorted(os.listdir(self.crops_dir))

        for filename in crop_files:
            if filename.endswith(".png"):
                # P1_field_a.png から field_a をキーとして抽出
                key = "_".join(filename.split("_")[1:]).replace(".png", "")
                
                image_path = os.path.join(self.crops_dir, filename)
                image = cv2.imread(image_path)
                
                text, confidence = self.ocr_engine.run(image)
                
                results[key] = {
                    "text": text,
                    "confidence": confidence,
                    "source_image": filename
                }
        
        # 結果をextract.jsonに保存
        output_path = os.path.join(self.workspace_dir, "extract.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
            
        return results
