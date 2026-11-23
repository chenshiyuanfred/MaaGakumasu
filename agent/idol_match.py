"""偶像卡 OCR 文本匹配工具。

这里只做纯文本处理，不依赖 MaaFramework，方便离线测试与复用。
"""

import unicodedata
from typing import Tuple, Callable, Iterable, Sequence
from difflib import SequenceMatcher

# 文本相似度阈值，低于该值视为识别失败
SIMILARITY_THRESHOLD = 0.7

# 同一行文字的纵向容差（像素），用于把被切碎的文本重新拼接
LINE_TOLERANCE = 24


def normalize_text(text: str) -> str:
    """
    归一化 OCR 文本，抹平特殊符号带来的差异

    OCR 对特殊符号识别不稳定，例如「36℃ U･B･U」常被识别成「36°C U・B・U」，
    直接比较相似度只有 0.63，会导致该偶像卡无法被识别。
    NFKC 会先把全角转半角、把 ℃ 拆成 °C、把 ･ 转成 ・，
    之后只保留字母、数字、假名与汉字，符号与空格一律忽略。
    """
    return "".join(ch for ch in unicodedata.normalize("NFKC", text) if ch.isalnum())


def similarity_ratio(str1: str, str2: str) -> float:
    """返回0-1之间的相似度分数，1表示完全相同（比较前先归一化，忽略符号差异）"""
    return SequenceMatcher(None, normalize_text(str1), normalize_text(str2)).ratio()


def song_similarity_ratio(recognized_song: str, song_name: str) -> float:
    """歌曲名完整包含时忽略皮肤附加文案，否则沿用相似度比较。"""
    recognized = normalize_text(recognized_song)
    configured = normalize_text(song_name)
    if not recognized or not configured:
        return 0.0
    if configured in recognized:
        return 1.0
    return SequenceMatcher(None, recognized, configured).ratio()


def line_candidates(results: Iterable) -> list[str]:
    """
    把一次 OCR 的结果整理成候选文本：
    单条文本 + 同一行被切碎的文本按 x 坐标拼接，供后续分别比较。
    """
    items = [item for item in results or [] if getattr(item, "text", "").strip()]
    candidates = [item.text for item in items]

    rows: list[list] = []
    for item in sorted(items, key=lambda i: i.box[1]):
        if rows and abs(rows[-1][0].box[1] - item.box[1]) <= LINE_TOLERANCE:
            rows[-1].append(item)
        else:
            rows.append([item])
    for row in rows:
        if len(row) > 1:
            candidates.append("".join(item.text for item in sorted(row, key=lambda i: i.box[0])))
    return candidates


def best_match(
    candidates: Sequence[str],
    expected: str,
    scorer: Callable[[str, str], float] = similarity_ratio,
) -> Tuple[str, float]:
    """返回最相似的候选文本及其分数"""
    best_text, best_score = "", 0.0
    for text in candidates:
        score = scorer(text, expected)
        if score > best_score:
            best_text, best_score = text, score
    return best_text, best_score
