from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, Tuple
import asyncio

import cv2
import numpy as np

from . import preprocess
from .ocr_bridge import BaseOCR
from .ocr_processor import OCRProcessor

from .db_manager import DBManager
from .template_manager import TemplateManager


@dataclass
class OcrAgent:
    """Core class orchestrating the OCR workflow.

    The agent ties together template handling, preprocessing, OCR execution
    and database persistence into a single entry point.
    """

    db: DBManager
    templates: TemplateManager

    def process_document(
        self,
        image: np.ndarray,
        image_name: str,
        template_data: Dict[str, any],
        ocr_engine: BaseOCR,
        validator_engine: BaseOCR | None = None,
        job_id: int | None = None,
    ) -> Tuple[Dict[str, dict], str]:
        """Process a single document and persist results.

        Parameters
        ----------
        image:
            Source image as a ``numpy.ndarray``.
        image_name:
            Original filename of the uploaded image.
        template_data:
            Loaded template definition containing ROI information.
        ocr_engine:
            OCR engine implementation used for primary text extraction.
        validator_engine:
            Optional secondary OCR engine used for double-checking results.
        job_id:
            Existing database job identifier. If ``None``, a new job is created
            per document. When provided, all results are associated with the
            supplied job, enabling multiple images under a single job.

        Returns
        -------
        results: dict
            OCR results keyed by ROI name.
        workspace_dir: str
            Path to the workspace directory used for intermediate files.
        """

        now = datetime.now()
        doc_id = f"DOC_{now.strftime('%Y%m%d_%H%M%S')}"
        workspace_dir = Path("workspace") / doc_id
        crops_dir = workspace_dir / "crops"
        workspace_dir.mkdir(parents=True, exist_ok=True)
        crops_dir.mkdir(parents=True, exist_ok=True)

        # Save template for traceability
        with (workspace_dir / "template.json").open("w", encoding="utf-8") as f:
            json.dump(template_data, f, ensure_ascii=False, indent=2)

        rois = template_data.get("rois", {})

        # Preprocess image and align ROIs
        corrected_image = preprocess.correct_skew(image)

        template_path = template_data.get("template_image") or template_data.get(
            "template_image_path"
        )
        if template_path and Path(template_path).exists():
            template_img = cv2.imread(str(template_path))
            aligned_rois = preprocess.align_rois(template_img, corrected_image, rois)
        else:
            aligned_rois = rois

        for i, (key, roi_info) in enumerate(aligned_rois.items()):
            box = roi_info["box"]
            cropped = preprocess.crop_roi(corrected_image, box)
            filename = f"P{i+1}_{key}.png"
            cv2.imwrite(str(crops_dir / filename), cropped)

        # Execute OCR
        processor = OCRProcessor(
            ocr_engine,
            str(workspace_dir),
            validator_engine=validator_engine,
            rois=aligned_rois,
        )
        results = asyncio.run(processor.process_all())

        # Persist to database
        if job_id is None:
            job_id = self.db.create_job(template_data.get("name", ""), now.isoformat())
        for roi_name, info in results.items():
            result_id = self.db.add_result(
                job_id,
                image_name,
                roi_name,
                text_mini=info.get("text_mini"),
                text_nano=info.get("text_nano"),
                final_text=info["text"],
                confidence_score=info["confidence"],
                status=info.get("confidence_level"),
            )
            info["result_id"] = result_id

        # Overwrite extract.json with result IDs included
        extract_path = workspace_dir / "extract.json"
        with extract_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        return results, str(workspace_dir)
