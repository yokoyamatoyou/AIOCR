from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


class TemplateManager:
    """Manage template files stored in JSON format.

    Each template describes regions of interest (ROIs) and optional
    prompt rules or correction dictionaries for post processing.
    """

    def __init__(self, template_dir: str = "templates") -> None:
        self.template_dir = Path(template_dir)
        self.template_dir.mkdir(parents=True, exist_ok=True)

    def list_templates(self) -> List[str]:
        """Return a list of available template names."""
        return [p.stem for p in self.template_dir.glob("*.json")]

    def load(self, name: str) -> Dict[str, Any]:
        """Load a template by name.

        Parameters
        ----------
        name: str
            Template file name without extension.
        """
        path = self.template_dir / f"{name}.json"
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def save(self, name: str, data: Dict[str, Any]) -> None:
        """Save template data to a JSON file.

        Any additional fields are preserved to allow forward compatible
        extensions such as ``template_image_path``.  The ``keywords`` field is
        normalised to always be present as a list to simplify downstream
        consumption.
        """
        path = self.template_dir / f"{name}.json"
        data_to_save = dict(data)
        data_to_save.setdefault("keywords", [])
        with path.open("w", encoding="utf-8") as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)

    def get_keywords(self, name: str) -> List[str]:
        """Return the list of detection keywords for a template.

        If the template does not define any keywords an empty list is
        returned."""
        data = self.load(name)
        keywords = data.get("keywords", [])
        return keywords if isinstance(keywords, list) else []

    def append_correction(self, name: str, wrong: str, correct: str) -> None:
        """Append a correction pair to template's correction list.

        The previous implementation stored corrections in a mapping which
        overwrote existing entries when the same ``wrong`` text appeared
        multiple times.  To preserve the full history of human feedback,
        corrections are now recorded as a list of ``{"wrong": ..., "correct": ...}``
        dictionaries.
        """
        data = self.load(name)
        corrections = data.setdefault("corrections", [])
        if not isinstance(corrections, list):
            # migrate legacy dict-based structure
            corrections = [
                {"wrong": k, "correct": v} for k, v in corrections.items()
            ]
        corrections.append({"wrong": wrong, "correct": correct})
        data["corrections"] = corrections
        self.save(name, data)
