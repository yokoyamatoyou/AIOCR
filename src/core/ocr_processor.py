
import os
import cv2
import json
import asyncio
from typing import Optional, Dict, Any, Tuple

from .ocr_bridge import BaseOCR
from . import postprocess

class OCRProcessor:
    """OCR処理全体を管理するクラス"""

    def __init__(
        self,
        primary_engine: BaseOCR,
        workspace_dir: str,
        validator_engine: Optional[BaseOCR] = None,
        rois: Optional[Dict[str, Any]] = None,
    ):
        self.primary_engine = primary_engine
        self.validator_engine = validator_engine
        self.workspace_dir = workspace_dir
        self.crops_dir = os.path.join(self.workspace_dir, "crops")
        self.rois = rois or {}

    async def _process_file(self, filename: str) -> Tuple[str, Dict[str, Any]]:
        key = "_".join(filename.split("_")[1:]).replace(".png", "")
        image_path = os.path.join(self.crops_dir, filename)
        image = cv2.imread(image_path)

        primary_text, primary_conf = await self.primary_engine.run(image)
        norm_primary = postprocess.normalize_text(primary_text)

        rule = None
        if key in self.rois:
            rule = self.rois[key].get("validation_rule")

        norm_secondary = None
        needs_human = False

        if self.validator_engine is not None:
            secondary_text, _ = await self.validator_engine.run(image)
            norm_secondary = postprocess.normalize_text(secondary_text)

            if norm_primary == norm_secondary:
                confidence = 1.0
                confidence_level = "high"
            else:
                valid = postprocess.check_validation(norm_primary, rule)
                if valid:
                    confidence = 0.5
                    confidence_level = "medium"
                else:
                    confidence = 0.0
                    confidence_level = "low"
                needs_human = True
        else:
            norm_primary, needs_human = postprocess.postprocess_result(
                primary_text, primary_conf, rule
            )
            confidence = primary_conf
            confidence_level = "high" if not needs_human else "low"

        entry = {
            "text": norm_primary,
            "confidence": confidence,
            "source_image": filename,
            "text_mini": norm_primary,
            "confidence_level": confidence_level,
        }
        if norm_secondary is not None:
            entry["text_nano"] = norm_secondary
        if needs_human:
            entry["needs_human"] = True

        return key, entry

    async def process_all(self) -> dict:
        """cropsディレクトリ内の画像を並行処理し、結果をJSONにまとめる"""

        crop_files = sorted(f for f in os.listdir(self.crops_dir) if f.endswith(".png"))
        tasks = [self._process_file(filename) for filename in crop_files]
        processed = await asyncio.gather(*tasks)
        results = {key: entry for key, entry in processed}

        output_path = os.path.join(self.workspace_dir, "extract.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        return results
