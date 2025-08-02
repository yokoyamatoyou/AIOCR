from __future__ import annotations

from dataclasses import dataclass

from .db_manager import DBManager
from .template_manager import TemplateManager


@dataclass
class OcrAgent:
    """Placeholder for core OCR processing logic.

    This class will orchestrate the OCR workflow using the database and
    template managers implemented in Phase 1.
    """

    db: DBManager
    templates: TemplateManager
