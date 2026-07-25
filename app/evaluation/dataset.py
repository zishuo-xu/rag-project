"""评估数据集管理 - 测试集加载"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# 默认评估数据集路径
DEFAULT_DATASET_PATH = Path("data/eval_dataset.json")


def load_eval_dataset(path: Optional[str] = None) -> dict:
    """
    加载评估数据集。

    Args:
        path: 数据集文件路径

    Returns:
        数据集字典
    """
    file_path = Path(path) if path else DEFAULT_DATASET_PATH

    if not file_path.exists():
        logger.warning(f"评估数据集不存在: {file_path}")
        return {"samples": [], "num_samples": 0}

    with open(file_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    logger.info(f"加载评估数据集: {file_path} ({dataset.get('num_samples', 0)} 条)")
    return dataset
