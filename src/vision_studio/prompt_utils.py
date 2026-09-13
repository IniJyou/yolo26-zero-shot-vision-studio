from collections.abc import Iterable


def normalize_classes(
    classes: Iterable[str],
) -> list[str]:
    """清理类别名称并按照输入顺序去重。"""
    normalized: list[str] = []
    seen: set[str] = set()

    for value in classes:
        class_name = value.strip()

        if not class_name:
            continue

        deduplication_key = class_name.casefold()

        if deduplication_key in seen:
            continue

        seen.add(deduplication_key)
        normalized.append(class_name)

    if not normalized:
        raise ValueError("YOLOE提示词不能为空")

    return normalized


def parse_prompt_text(
    prompt_text: str,
) -> list[str]:
    """将英文或中文逗号分隔的文本转换为类别列表。"""
    unified_text = prompt_text.replace("，", ",")
    return normalize_classes(unified_text.split(","))