"""
Token counter service — TokenCounterService, HeuristicTokenCounterService.

Mirrors Java infra token classes:
  - token.TokenCounterService
  - token.HeuristicTokenCounterService
"""

from __future__ import annotations

import unicodedata
from abc import ABC, abstractmethod


class TokenCounterService(ABC):
    """
    Token 统计服务接口。

    Mirrors Java token.TokenCounterService.
    """

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        统计文本的 Token 数。

        Args:
            text: 文本内容

        Returns:
            Token 数（无法计算时返回 0）
        """
        ...


class HeuristicTokenCounterService(TokenCounterService):
    """
    轻量 Token 估算服务 — 基于字符类型的启发式估算。

    规则：
    - ASCII 字符（英文等）：4 字符约 1 token
    - CJK 字符（中日韩）：1 字符约 1 token
    - 其他字符：2 字符约 1 token

    Mirrors Java token.HeuristicTokenCounterService.
    """

    # CJK Unicode block names for fast lookup
    _CJK_BLOCKS = frozenset({
        "CJK UNIFIED IDEOGRAPH",
        "CJK UNIFIED IDEOGRAPH EXTENSION A",
        "CJK UNIFIED IDEOGRAPH EXTENSION B",
        "CJK UNIFIED IDEOGRAPH EXTENSION C",
        "CJK UNIFIED IDEOGRAPH EXTENSION D",
        "CJK UNIFIED IDEOGRAPH EXTENSION E",
        "CJK UNIFIED IDEOGRAPH EXTENSION F",
        "CJK COMPATIBILITY IDEOGRAPH",
        "CJK COMPATIBILITY IDEOGRAPH SUPPLEMENT",
        "CJK RADICALS SUPPLEMENT",
        "CJK SYMBOLS AND PUNCTUATION",
        "HIRAGANA",
        "KATAKANA",
        "KATAKANA PHONETIC EXTENSIONS",
        "HANGUL SYLLABLES",
        "HANGUL JAMO",
        "HANGUL COMPATIBILITY JAMO",
    })

    def count_tokens(self, text: str) -> int:
        if not text or not text.strip():
            return 0

        ascii_count = 0
        cjk_count = 0
        other_count = 0

        for ch in text:
            if ch.isspace():
                continue
            if ord(ch) <= 0x7F:
                ascii_count += 1
            elif self._is_cjk(ch):
                cjk_count += 1
            else:
                other_count += 1

        ascii_tokens = (ascii_count + 3) // 4  # 英文等按 4 字符约 1 token
        other_tokens = (other_count + 1) // 2   # 其他字符按 2 字符约 1 token
        total = ascii_tokens + cjk_count + other_tokens
        return max(total, 1)

    def _is_cjk(self, ch: str) -> bool:
        """判断字符是否属于 CJK 字符集"""
        try:
            block = unicodedata.name(ch, "")
            # Check if the block name starts with any CJK prefix
            for cjk_prefix in self._CJK_BLOCKS:
                if block.startswith(cjk_prefix):
                    return True
        except ValueError:
            pass
        return False
