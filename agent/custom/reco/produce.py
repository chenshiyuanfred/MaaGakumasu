import os
import json
import time
from typing import Union, Optional

from utils import logger
from idol_match import (
    LINE_TOLERANCE,
    SIMILARITY_THRESHOLD,
    best_match,
    normalize_text,
    line_candidates,
    similarity_ratio,
    song_similarity_ratio,
)
from maa.define import RectType
from maa.context import Context
from maa.agent.agent_server import AgentServer
from maa.custom_recognition import CustomRecognition


@AgentServer.custom_recognition("ProduceChooseIdolAuto")
class ProduceChooseIdolAuto(CustomRecognition):
    """
    自动识别当前偶像名称和歌曲
    """

    # 文本相似度阈值，低于该值视为识别失败
    SIMILARITY_THRESHOLD = SIMILARITY_THRESHOLD

    # 偶像名称与歌曲名称所在区域的并集；勾选 True End 时整体下移，
    # 因此这里一次性覆盖两种布局，避免先探测 True End 再切换 ROI 的多余 OCR。
    PANEL_ROI = [330, 20, 390, 250]

    # 同一行文字的纵向容差（像素），用于把被切碎的文本重新拼接
    LINE_TOLERANCE = LINE_TOLERANCE

    def analyze(
        self,
        context: Context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> Union[CustomRecognition.AnalyzeResult, Optional[RectType]]:
        idol_name = json.loads(argv.custom_recognition_param)["idol_name"]
        song_name = json.loads(argv.custom_recognition_param)["song_name"]

        panel_detail = context.run_recognition(
            "ProduceChooseIdolPanel",
            argv.image,
            pipeline_override={
                "ProduceChooseIdolPanel": {
                    "recognition": "OCR",
                    "expected": [idol_name, song_name],
                    "roi": self.PANEL_ROI,
                }
            },
        )

        candidates = line_candidates(getattr(panel_detail, "all_results", None))
        name_text, name_score = best_match(candidates, idol_name, similarity_ratio)
        song_text, song_score = best_match(candidates, song_name, song_similarity_ratio)

        if name_score >= self.SIMILARITY_THRESHOLD and song_score >= self.SIMILARITY_THRESHOLD:
            logger.info(
                f"识别到偶像卡: {name_text}/{song_text}，相似度 {name_score:.2f}/{song_score:.2f}"
            )
            return CustomRecognition.AnalyzeResult(box=[0, 0, 1, 1], detail={"detail": "识别偶像卡成功"})

        logger.debug(
            f"识别偶像卡失败，偶像: {name_text}（{name_score:.2f}），歌曲: {song_text}（{song_score:.2f}）"
        )
        return CustomRecognition.AnalyzeResult(box=None, detail={"detail": "识别偶像卡失败"})

    # 兼容旧调用：把纯函数挂到类上，避免外部脚本导入时找不到这些名字
    normalize_text = staticmethod(normalize_text)
    similarity_ratio = staticmethod(similarity_ratio)
    song_similarity_ratio = staticmethod(song_similarity_ratio)
    line_candidates = staticmethod(line_candidates)
    best_match = staticmethod(best_match)


@AgentServer.custom_recognition("ProduceShowStart")
class ProduceShowStart(CustomRecognition):
    """
    检测通过屏幕是否旋转判断演出开始
    """

    def analyze(
        self,
        context: Context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> Union[CustomRecognition.AnalyzeResult, Optional[RectType]]:
        image = argv.image
        height, width = image.shape[0], image.shape[1]
        if height < width:
            return CustomRecognition.AnalyzeResult(box=[0, 0, 1, 1], detail={"detail": "屏幕旋转"})
        return CustomRecognition.AnalyzeResult(box=None, detail={"detail": "屏幕未旋转"})


@AgentServer.custom_recognition("ProduceShowEnd")
class ProduceShowEnd(CustomRecognition):
    """
    检测通过屏幕是否旋转判断演出结束
    """

    def analyze(
        self,
        context: Context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> Union[CustomRecognition.AnalyzeResult, Optional[RectType]]:
        image = argv.image
        height, width = image.shape[0], image.shape[1]
        if height > width:
            return CustomRecognition.AnalyzeResult(box=[0, 0, 1, 1], detail={"detail": "屏幕旋转"})
        return CustomRecognition.AnalyzeResult(box=None, detail={"detail": "屏幕未旋转"})


@AgentServer.custom_recognition("ProduceCardsFlagAuto")
class ProduceCardsFlagAuto(CustomRecognition):
    """
    自动识别出牌场景
    """

    def analyze(
        self,
        context: Context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> Union[CustomRecognition.AnalyzeResult, Optional[RectType]]:
        context.run_action("Click_1")
        cards_reco_detail = context.run_recognition("ProduceRecognitionCards", argv.image)
        health_reco_detail = context.run_recognition("ProduceRecognitionHealthFlag", argv.image)
        if cards_reco_detail and cards_reco_detail.hit and health_reco_detail and health_reco_detail.hit:
            logger.success("事件: 出牌场景")
            return CustomRecognition.AnalyzeResult(box=[0, 0, 1, 1], detail={"detail": "识别到出牌场景"})
        else:
            return CustomRecognition.AnalyzeResult(box=None, detail={"detail": "未识别到选择场景"})
