
import pytest
import os
import cv2
import numpy as np
from unittest.mock import patch, MagicMock

# srcディレクトリをパスに追加
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from app.ocr_bridge import DummyOCR, GPT4oMiniVisionOCR
from app.core.config import settings

# OpenAI APIキーが設定されているかチェック
API_KEY_SET = settings.OPENAI_API_KEY not in [None, "", "YOUR_API_KEY_HERE", "ここにあなたのOpenAI APIキーを入力してください"]

@pytest.fixture
def sample_text_image():
    """ テスト用の日本語テキストを含む画像を生成 """
    img = np.zeros((100, 300, 3), dtype=np.uint8)
    img.fill(255)
    # 日本語フォントがない場合、cv2.putTextは日本語を描画できないため、
    # ここでは単純な英数字で代用し、ロジックのテストに集中します。
    cv2.putText(img, "Test OCR", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return img

def test_dummy_ocr(sample_text_image):
    """ DummyOCRが期待通りに動作するかテスト """
    ocr = DummyOCR()
    text, confidence = ocr.run(sample_text_image)
    assert text == "ダミーテキスト(300x100)"
    assert confidence == 0.95

@pytest.mark.skipif(not API_KEY_SET, reason="OPENAI_API_KEYが設定されていません")
def test_gpt4o_mini_vision_ocr_integration(sample_text_image):
    """ GPT4oMiniVisionOCRが実際にAPIと通信して結果を取得できるかテスト """
    ocr = GPT4oMiniVisionOCR()
    text, confidence = ocr.run(sample_text_image)
    
    assert isinstance(text, str)
    assert text != ""
    assert "Test" in text or "OCR" in text # プロンプトによっては大文字小文字が変わる可能性がある
    assert isinstance(confidence, float)
    assert confidence > 0.0

@patch('openai.OpenAI')
def test_gpt4o_mini_vision_ocr_mocked(MockOpenAI, sample_text_image):
    """ GPT4oMiniVisionOCRのAPI呼び出しをモックしてテスト """
    # モックされたクライアントとレスポンスを設定
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "モックされたOCR結果"
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response
    MockOpenAI.return_value = mock_client

    # テスト実行
    ocr = GPT4oMiniVisionOCR()
    text, confidence = ocr.run(sample_text_image)

    # アサーション
    assert text == "モックされたOCR結果"
    assert confidence == 0.99
    # APIが正しい引数で呼び出されたか確認
    mock_client.chat.completions.create.assert_called_once()
    call_args = mock_client.chat.completions.create.call_args
    assert call_args.kwargs['model'] == "gpt-4o-mini"
