
import streamlit as st
import os
import yaml
import cv2
import numpy as np
from datetime import datetime
from . import preprocess
from .ocr_bridge import DummyOCR, GPT4oMiniVisionOCR
from .ocr_processor import OCRProcessor

st.title("AIOCR処理実行")

# --- サイドバー --- 
st.sidebar.title("設定")
ocr_engine_choice = st.sidebar.selectbox(
    "OCRエンジンを選択",
    ("DummyOCR", "GPT-4o-mini")
)

# --- メイン画面 --- 

# 1. ファイルアップローダー
uploaded_image = st.file_uploader("画像ファイルをアップロードしてください", type=['png', 'jpg', 'jpeg'])
uploaded_yaml = st.file_uploader("ROI定義ファイル (rois.yaml) をアップロードしてください", type=['yaml', 'yml'])

if uploaded_image is not None and uploaded_yaml is not None:
    if st.button("OCR処理実行"):
        with st.spinner('AI-OCR処理を実行中です...'):
            # 2. ユニークな作業ディレクトリを作成
            now = datetime.now()
            doc_id = f"DOC_{now.strftime('%Y%m%d_%H%M%S')}"
            workspace_dir = os.path.join("workspace", doc_id)
            crops_dir = os.path.join(workspace_dir, "crops")
            os.makedirs(crops_dir, exist_ok=True)

            # 3. アップロードされたrois.yamlを読み込む
            rois = yaml.safe_load(uploaded_yaml)

            # 4. 画像を読み込み、傾き補正を行う
            file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            
            st.write("画像の傾きを補正しています...")
            corrected_image = preprocess.correct_skew(image)
            
            # 5. YAMLの定義に従ってROIを切り出し、保存
            st.write("ROIを切り出しています...")
            for i, (key, roi_info) in enumerate(rois.items()):
                roi_box = roi_info['box']
                cropped_img = preprocess.crop_roi(corrected_image, roi_box)
                
                filename = f"P{i+1}_{key}.png"
                save_path = os.path.join(crops_dir, filename)
                cv2.imwrite(save_path, cropped_img)

            # 6. 選択されたOCRエンジンで処理を実行
            st.write(f"{ocr_engine_choice} でOCR処理を実行しています...")
            if ocr_engine_choice == "GPT-4o-mini":
                ocr_engine = GPT4oMiniVisionOCR()
            else:
                ocr_engine = DummyOCR()
            
            processor = OCRProcessor(ocr_engine, workspace_dir)
            ocr_results = processor.process_all()

        # 7. 処理完了メッセージと結果を表示
        st.success("処理が完了しました！")
        st.info(f"作業ディレクトリ: {os.path.abspath(workspace_dir)}")
        
        st.subheader("OCR抽出結果 (extract.json)")
        st.json(ocr_results)
