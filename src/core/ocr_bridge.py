"""OCR engine interfaces and implementations with asynchronous support."""

from __future__ import annotations

from abc import ABC, abstractmethod
import base64
from typing import Tuple

import aiohttp
import cv2
import numpy as np

from .config import settings


class BaseOCR(ABC):
    """すべてのOCRエンジンのための抽象基底クラス"""

    @abstractmethod
    async def run(self, image: np.ndarray) -> Tuple[str, float]:
        """画像を受け取り、(テキスト, 信頼度) のタプルを返す"""


class DummyOCR(BaseOCR):
    """ダミーのOCRエンジン。常に固定のテキストと信頼度を返す。"""

    async def run(self, image: np.ndarray) -> Tuple[str, float]:
        h, w = image.shape[:2]
        dummy_text = f"ダミーテキスト({w}x{h})"
        return dummy_text, 0.95


class GPT4oMiniVisionOCR(BaseOCR):
    """GPT-4o mini を利用したOCRエンジン"""

    async def run(self, image: np.ndarray) -> Tuple[str, float]:
        _, buffer = cv2.imencode(".png", image)
        base64_image = base64.b64encode(buffer).decode("utf-8")

        headers = {
            "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "gpt-4.1-mini",
            "messages": [
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
            "max_tokens": 300,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise RuntimeError(
                            f"OpenAI API request failed ({resp.status}): {error_text}"
                        )
                    data = await resp.json()
            text = data["choices"][0]["message"]["content"].strip()
            confidence = 0.99
            return text, confidence
        except Exception as e:  # pragma: no cover - network errors
            print(f"OpenAI API呼び出し中にエラーが発生しました: {e}")
            return "エラー", 0.0


class GPT4oNanoVisionOCR(BaseOCR):
    """GPT-4o nano を利用したOCRエンジン"""

    async def run(self, image: np.ndarray) -> Tuple[str, float]:
        _, buffer = cv2.imencode(".png", image)
        base64_image = base64.b64encode(buffer).decode("utf-8")

        headers = {
            "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "gpt-4.1-nano",
            "messages": [
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
            "max_tokens": 300,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise RuntimeError(
                            f"OpenAI API request failed ({resp.status}): {error_text}"
                        )
                    data = await resp.json()
            text = data["choices"][0]["message"]["content"].strip()
            confidence = 0.99
            return text, confidence
        except Exception as e:  # pragma: no cover - network errors
            print(f"OpenAI API呼び出し中にエラーが発生しました: {e}")
            return "エラー", 0.0

