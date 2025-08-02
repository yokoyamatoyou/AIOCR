
import asyncio
import streamlit as st
import os
import sys
from typing import Dict, List
from datetime import datetime

import cv2
import numpy as np

# Ensure src directory is on the import path when executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.ocr_bridge import DummyOCR, GPT4oMiniVisionOCR, GPT4oNanoVisionOCR
# Caching helpers
from app.cache_utils import get_template_manager, get_db_manager, list_templates
from core.ocr_agent import OcrAgent

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
uploaded_images = st.file_uploader(
    "画像ファイルをアップロードしてください",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

# テンプレート選択肢を準備
template_manager = get_template_manager()
template_names = list_templates()
template_option = st.selectbox(
    "帳票テンプレートを選択",
    ["自動検出"] + template_names,
)

if uploaded_images and template_names:
    if st.button("OCR処理実行"):
        db = get_db_manager()
        job_id = db.create_job(template_option, datetime.now().isoformat())
        agent = OcrAgent(db=db, templates=template_manager)

        # OCRエンジンを選択
        st.write(f"{ocr_engine_choice} でOCR処理を実行しています...")
        if ocr_engine_choice == "GPT-4.1-mini":
            ocr_engine = GPT4oMiniVisionOCR()
        else:
            ocr_engine = DummyOCR()
        nano_engine = GPT4oNanoVisionOCR()

        combined_results = {}
        workspace_dirs = {}
        progress = st.progress(0)
        total = len(uploaded_images)

        with st.spinner('AI-OCR処理を実行中です (GPT-4.1-nanoでダブルチェック)...'):
            static_template = None
            if template_option != "自動検出":
                static_template = template_manager.load(template_option)

            for idx, uploaded_image in enumerate(uploaded_images, start=1):
                file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, 1)

                if template_option == "自動検出":
                    st.write(f"{uploaded_image.name} のテンプレートを自動検出しています...")
                    text, _ = asyncio.run(GPT4oNanoVisionOCR().run(image))
                    best_template = None
                    best_score = 0
                    for tpl in template_names:
                        keywords = TEMPLATE_KEYWORDS.get(tpl, [])
                        score = sum(kw in text for kw in keywords)
                        if score > best_score:
                            best_score = score
                            best_template = tpl
                    if best_template:
                        template_data = template_manager.load(best_template)
                    else:
                        st.warning("テンプレートを特定できなかったため、最初のテンプレートを使用します")
                        template_data = template_manager.load(template_names[0])
                else:
                    template_data = static_template

                ocr_results, workspace_dir = agent.process_document(
                    image,
                    uploaded_image.name,
                    template_data,
                    ocr_engine,
                    validator_engine=nano_engine,
                    job_id=job_id,
                )
                combined_results[uploaded_image.name] = ocr_results
                workspace_dirs[uploaded_image.name] = workspace_dir
                progress.progress(idx / total)

        # 処理完了メッセージと結果を表示
        st.success("処理が完了しました！")
        st.subheader("作業ディレクトリ")
        for name, path in workspace_dirs.items():
            st.write(f"{name}: {os.path.abspath(path)}")

        st.subheader("OCR抽出結果 (extract.json)")
        st.json(combined_results)

        st.subheader("信頼度とダブルチェック結果")
        for img_name, ocr_result in combined_results.items():
            st.markdown(f"### {img_name}")
            for field, info in ocr_result.items():
                conf_score = info.get("confidence", 0.0)
                level = info.get("confidence_level", "")
                needs_human = info.get("needs_human", False)
                icon = "✅" if not needs_human else "⚠️"
                st.write(f"{icon} {field}: 信頼度 {conf_score:.2f} ({level})")
