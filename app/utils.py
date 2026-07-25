"""跨层共享的零依赖小工具（不 import app 内任何模块，杜绝循环依赖）。"""

import json
import re


def extract_json(text: str) -> dict | None:
    """从 LLM 输出中提取首个 JSON 对象（容忍前后噪声文本与尾部逗号）。

    全管道共用的 LLM-JSON 解析（crag / faithfulness / query_transform）。
    """
    json_match = re.search(r'\{[\s\S]*\}', text)
    if not json_match:
        return None
    raw = json_match.group()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # 修复尾部逗号
        cleaned = re.sub(r',\s*([}\]])', r'\1', raw)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None
