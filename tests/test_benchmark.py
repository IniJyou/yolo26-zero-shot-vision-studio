from pathlib import Path

import pytest

from vision_studio.benchmark import (
    label_hit,
    load_manifest,
)


HEADER = (
    "sample_id,image_path,group,target_label,"
    "yolo26_expected_label,yoloe_prompt,"
    "yoloe_expected_label,source_reference,"
    "usage_rights,notes\n"
)


def test_label_hit_matches_case_insensitively(
) -> None:
    assert label_hit(
        "Penguin",
        ["person", "penguin"],
    ) is True

    assert label_hit(
        "penguin",
        ["bird"],
    ) is False

    assert label_hit(
        None,
        ["bird"],
    ) is None


def test_load_manifest(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"test")

    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text(
        HEADER
        + (
            "001,image.jpg,common,bus,bus,"
            "bus,bus,self,self-created,test\n"
        ),
        encoding="utf-8",
    )

    cases = load_manifest(
        manifest_path=manifest_path,
        project_root=tmp_path,
    )

    assert len(cases) == 1
    assert cases[0].sample_id == "001"
    assert cases[0].image_path == (
        image_path.resolve()
    )
    assert (
        cases[0].yolo26_expected_label
        == "bus"
    )


def test_manifest_rejects_missing_image(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text(
        HEADER
        + (
            "001,missing.jpg,common,bus,bus,"
            "bus,bus,self,self-created,test\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        FileNotFoundError,
        match="评测图片不存在",
    ):
        load_manifest(
            manifest_path=manifest_path,
            project_root=tmp_path,
        )


def test_manifest_rejects_duplicate_id(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"test")

    row = (
        "001,image.jpg,common,bus,bus,"
        "bus,bus,self,self-created,test\n"
    )

    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text(
        HEADER + row + row,
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="sample_id重复",
    ):
        load_manifest(
            manifest_path=manifest_path,
            project_root=tmp_path,
        )