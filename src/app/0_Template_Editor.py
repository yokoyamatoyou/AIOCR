"""UI for creating and editing OCR templates."""

from __future__ import annotations

from typing import Dict, List

from pathlib import Path

from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from app.cache_utils import get_template_manager, list_templates


NEW_TEMPLATE = "新規作成"


def _load_initial_drawing(rois: Dict[str, Dict[str, List[int]]]) -> Dict[str, List[Dict[str, int]]]:
    """Convert ROI dict to drawable-canvas format."""
    objects: List[Dict[str, int]] = []
    for roi in rois.values():
        x, y, w, h = roi.get("box", [0, 0, 0, 0])
        objects.append(
            {
                "type": "rect",
                "left": x,
                "top": y,
                "width": w,
                "height": h,
                "fill": "rgba(255,0,0,0.3)",
                "stroke": "red",
            }
        )
    return {"version": "5.0.0", "objects": objects}


def main() -> None:
    st.title("Template Editor")

    manager = get_template_manager()
    templates = list_templates()

    selection = st.selectbox("テンプレートを選択", [NEW_TEMPLATE] + templates)
    template_name = st.text_input("テンプレート名", value="" if selection == NEW_TEMPLATE else selection)

    existing_rois: Dict[str, Dict[str, List[int]]] = {}
    existing_keywords: List[str] = []
    if selection != NEW_TEMPLATE:
        try:
            existing = manager.load(selection)
            existing_rois = existing.get("rois", {})
            existing_keywords = existing.get("keywords", [])
        except FileNotFoundError:
            st.warning("テンプレートが見つかりません。")

    uploaded = st.file_uploader("基準画像をアップロード", type=["png", "jpg", "jpeg"])

    keywords_text = st.text_input(
        "キーワード (カンマ区切り)",
        value=", ".join(existing_keywords),
    )

    if uploaded is None:
        st.info("画像をアップロードしてください。")
        return

    image = Image.open(uploaded)

    initial = _load_initial_drawing(existing_rois) if existing_rois else None
    canvas_result = st_canvas(
        fill_color="rgba(255,0,0,0.3)",
        stroke_width=2,
        stroke_color="red",
        background_image=image,
        height=image.height,
        width=image.width,
        drawing_mode="rect",
        initial_drawing=initial,
        key="canvas",
    )

    roi_boxes: List[List[int]] = []
    if canvas_result.json_data:
        for obj in canvas_result.json_data["objects"]:
            if obj.get("type") == "rect":
                roi_boxes.append(
                    [
                        int(obj.get("left", 0)),
                        int(obj.get("top", 0)),
                        int(obj.get("width", 0)),
                        int(obj.get("height", 0)),
                    ]
                )

    roi_definitions: Dict[str, Dict[str, object]] = {}
    for i, box in enumerate(roi_boxes):
        default_name = list(existing_rois.keys())[i] if i < len(existing_rois) else f"roi_{i + 1}"
        default_rule = (
            list(existing_rois.values())[i].get("validation_rule", "")
            if i < len(existing_rois)
            else ""
        )
        name = st.text_input(f"ROI {i + 1} 名称", value=default_name, key=f"roi_name_{i}")
        rule = st.text_input(f"ROI {i + 1} 検証ルール", value=default_rule, key=f"roi_rule_{i}")
        roi_definitions[name] = {"box": box, "validation_rule": rule}

    if st.button("保存"):
        if not template_name:
            st.error("テンプレート名を入力してください。")
        elif not roi_definitions:
            st.error("ROIを少なくとも1つ描画してください。")
        else:
            keywords = [
                kw.strip() for kw in keywords_text.split(",") if kw.strip()
            ]

            # save uploaded reference image
            suffix = Path(uploaded.name).suffix or ".png"
            image_path = manager.template_dir / f"{template_name}{suffix}"
            image.save(image_path)

            data = {
                "name": template_name,
                "keywords": keywords,
                "rois": roi_definitions,
                "template_image_path": str(image_path),
                # corrections are stored as a list for forward compatibility
                "corrections": [],
            }
            manager.save(template_name, data)
            list_templates.clear()
            st.success("テンプレートを保存しました。")


if __name__ == "__main__":
    main()
