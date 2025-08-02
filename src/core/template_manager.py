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
        """Save template data to a JSON file."""
        path = self.template_dir / f"{name}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # new method to append corrections
    def append_correction(self, name: str, wrong: str, correct: str) -> None:
        """Append a correction pair to template's correction dictionary."""
        data = self.load(name)
        corrections = data.setdefault("corrections", {})
        corrections[wrong] = correct
        self.save(name, data)
