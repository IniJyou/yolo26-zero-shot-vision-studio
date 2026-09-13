import pytest

from vision_studio.prompt_utils import (
    normalize_classes,
    parse_prompt_text,
)


def test_parse_prompt_text_splits_classes() -> None:
    result = parse_prompt_text(
        "person, bus, cartoon sheep"
    )

    assert result == [
        "person",
        "bus",
        "cartoon sheep",
    ]


def test_parse_prompt_text_supports_chinese_comma() -> None:
    result = parse_prompt_text("person，bus")

    assert result == ["person", "bus"]


def test_normalize_classes_removes_duplicates() -> None:
    result = normalize_classes(
        ["person", " Person ", "bus"]
    )

    assert result == ["person", "bus"]


def test_parse_prompt_text_rejects_empty_input() -> None:
    with pytest.raises(
        ValueError,
        match="提示词不能为空",
    ):
        parse_prompt_text(" , ， ")