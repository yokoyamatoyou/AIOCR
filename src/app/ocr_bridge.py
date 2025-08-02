
from abc import ABC, abstractmethod
import numpy as np
import base64
import cv2
import openai
from .core.config import settings

class BaseOCR(ABC):
    """すべてのOCRエンジンのための抽象基底クラス"""
    @abstractmethod
    def run(self, image: np.ndarray) -> tuple[str, float]:
        """
        画像を受け取り、(テキスト, 信頼度) のタプルを返す
        """
        pass

class DummyOCR(BaseOCR):
    """ダミーのOCRエンジン。常に固定のテキストと信頼度を返す。"""
    def run(self, image: np.ndarray) -> tuple[str, float]:
        """
        画像サイズに基づいてダミーのテキストを生成します。
        """
        h, w = image.shape[:2]
        dummy_text = f"ダミーテキスト({w}x{h})"
        return (dummy_text, 0.95)

class GPT4oMiniVisionOCR(BaseOCR):
    """GPT-4o mini を利用したOCRエンジン"""
    def __init__(self):
        self.client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)

    def run(self, image: np.ndarray) -> tuple[str, float]:
        # 画像をBase64にエンコード
        _, buffer = cv2.imencode(".png", image)
        base64_image = base64.b64encode(buffer).decode('utf-8')

        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "この画像に書かれている日本語のテキストを、改行やスペースは無視して、全ての文字を繋げて書き出してください。"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300,
            )
            text = response.choices[0].message.content.strip()
            # GPTのAPIは直接的な信頼度を返さないため、ここでは固定値を返す
            confidence = 0.99 
            return (text, confidence)
        except Exception as e:
            print(f"OpenAI API呼び出し中にエラーが発生しました: {e}")
            return ("エラー", 0.0)


class GPT4oNanoVisionOCR(BaseOCR):
    """GPT-4o nano を利用したOCRエンジン"""

    def __init__(self):
        self.client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)

    def run(self, image: np.ndarray) -> tuple[str, float]:
        """gpt-4.1-nanoモデルでOCRを実行する。"""

        _, buffer = cv2.imencode(".png", image)
        base64_image = base64.b64encode(buffer).decode("utf-8")

        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "この画像に書かれている日本語のテキストを、改行やスペースは無視して、全ての文字を繋げて書き出してください。",
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                            },
                        ],
                    }
                ],
                max_tokens=300,
            )
            text = response.choices[0].message.content.strip()
            confidence = 0.99
            return text, confidence
        except Exception as e:
            print(f"OpenAI API呼び出し中にエラーが発生しました: {e}")
            return "エラー", 0.0
