
import streamlit as st
import os
import cv2
import json
import numpy as np
from datetime import datetime
from typing import Dict, List

import sys

# Ensure src directory is on the import path when executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import preprocess
from app.ocr_bridge import DummyOCR, GPT4oMiniVisionOCR
from app.ocr_processor import OCRProcessor
from core.template_manager import TemplateManager
from core.db_manager import DBManager

# テンプレート名と検出キーワードの対応表
TEMPLATE_KEYWORDS: Dict[str, List[str]] = {
    "invoice": ["請求", "請求書", "御中"],
}

st.title("AIOCR処理実行")

# --- サイドバー --- 
st.sidebar.title("設定")
ocr_engine_choice = st.sidebar.selectbox(
    "OCRエンジンを選択",
    ("DummyOCR", "GPT-4.1-mini")
)

# --- メイン画面 --- 

# 1. ファイルアップローダー
uploaded_image = st.file_uploader(
    "画像ファイルをアップロードしてください", type=["png", "jpg", "jpeg"]
)

# テンプレート選択肢を準備
template_manager = TemplateManager()
template_names = template_manager.list_templates()
template_option = st.selectbox(
    "帳票テンプレートを選択",
    ["自動検出"] + template_names,
)

if uploaded_image is not None and template_names:
    if st.button("OCR処理実行"):
        with st.spinner('AI-OCR処理を実行中です...'):
            # 2. ユニークな作業ディレクトリを作成
            now = datetime.now()
            doc_id = f"DOC_{now.strftime('%Y%m%d_%H%M%S')}"
            workspace_dir = os.path.join("workspace", doc_id)
            crops_dir = os.path.join(workspace_dir, "crops")
            os.makedirs(crops_dir, exist_ok=True)

            # 3. 画像を読み込み、必要ならテンプレートを自動検出
            file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)

            if template_option == "自動検出":
                st.write("テンプレートを自動検出しています...")
                text, _ = DummyOCR().run(image)
                best_template = None
                best_score = 0
                for tpl in template_names:
                    keywords = TEMPLATE_KEYWORDS.get(tpl, [])
                    score = sum(kw in text for kw in keywords)
                    if score > best_score:
                        best_score = score
                        best_template = tpl
                if best_template:
                    st.info(f"自動検出結果: {best_template}")
                    template_data = template_manager.load(best_template)
                else:
                    st.warning("テンプレートを特定できなかったため、最初のテンプレートを使用します")
                    template_data = template_manager.load(template_names[0])
            else:
                template_data = template_manager.load(template_option)

            selected_template = template_data.get("name", template_option)
            rois = template_data.get("rois", {})

            template_json_path = os.path.join(workspace_dir, "template.json")
            with open(template_json_path, "w", encoding="utf-8") as f:
                json.dump(template_data, f, ensure_ascii=False, indent=2)

            # 4. 傾き補正を行う
            
            st.write("画像の傾きを補正しています...")
            corrected_image = preprocess.correct_skew(image)
            
            # 5. テンプレートの定義に従ってROIを切り出し、保存
            st.write("ROIを切り出しています...")
            for i, (key, roi_info) in enumerate(rois.items()):
                roi_box = roi_info['box']
                cropped_img = preprocess.crop_roi(corrected_image, roi_box)
                
                filename = f"P{i+1}_{key}.png"
                save_path = os.path.join(crops_dir, filename)
                cv2.imwrite(save_path, cropped_img)

            # 6. 選択されたOCRエンジンで処理を実行
            st.write(f"{ocr_engine_choice} でOCR処理を実行しています...")
            if ocr_engine_choice == "GPT-4.1-mini":
                ocr_engine = GPT4oMiniVisionOCR()
            else:
                ocr_engine = DummyOCR()
            
            processor = OCRProcessor(ocr_engine, workspace_dir, rois=rois)
            ocr_results = processor.process_all()

            db = DBManager()
            db.initialize()
            job_id = db.create_job(selected_template, now.isoformat())
            for roi_name, info in ocr_results.items():
                status = "needs_human" if info.get("needs_human") else "ok"
                db.add_result(
                    job_id,
                    uploaded_image.name,
                    roi_name,
                    final_text=info["text"],
                    confidence_score=info["confidence"],
                    status=status,
                )
            db.close()

        # 7. 処理完了メッセージと結果を表示
        st.success("処理が完了しました！")
        st.info(f"作業ディレクトリ: {os.path.abspath(workspace_dir)}")

        st.subheader("OCR抽出結果 (extract.json)")
        st.json(ocr_results)
