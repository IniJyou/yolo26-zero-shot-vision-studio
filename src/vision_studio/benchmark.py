import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from collections.abc import Iterable


BenchmarkGroup = Literal[
    "common",
    "open",
]


@dataclass(frozen=True)
class BenchmarkCase:
    sample_id: str
    image_path: Path
    group: BenchmarkGroup
    target_label: str
    yolo26_expected_label: str | None
    yoloe_prompt: str
    yoloe_expected_label: str
    source_reference: str
    usage_rights: str
    notes: str


REQUIRED_COLUMNS = {
    "sample_id",
    "image_path",
    "group",
    "target_label",
    "yolo26_expected_label",
    "yoloe_prompt",
    "yoloe_expected_label",
    "source_reference",
    "usage_rights",
    "notes",
}


def load_manifest(
    manifest_path: Path,
    project_root: Path,
) -> list[BenchmarkCase]:
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"评测清单不存在：{manifest_path}"
        )

    cases: list[BenchmarkCase] = []
    seen_ids: set[str] = set()

    with manifest_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        columns = set(reader.fieldnames or [])
        missing_columns = (
            REQUIRED_COLUMNS - columns
        )

        if missing_columns:
            missing_text = ", ".join(
                sorted(missing_columns)
            )
            raise ValueError(
                f"评测清单缺少字段：{missing_text}"
            )

        for row_number, row in enumerate(
            reader,
            start=2,
        ):
            sample_id = row[
                "sample_id"
            ].strip()

            if not sample_id:
                raise ValueError(
                    f"第{row_number}行sample_id为空"
                )

            if sample_id in seen_ids:
                raise ValueError(
                    f"sample_id重复：{sample_id}"
                )

            seen_ids.add(sample_id)

            group = row["group"].strip()

            if group not in ("common", "open"):
                raise ValueError(
                    f"不支持的分组：{group}"
                )

            relative_path = Path(
                row["image_path"].strip()
            )
            image_path = (
                project_root / relative_path
            ).resolve()

            if not image_path.is_file():
                raise FileNotFoundError(
                    f"评测图片不存在：{image_path}"
                )

            target_label = row[
                "target_label"
            ].strip()

            yoloe_prompt = row[
                "yoloe_prompt"
            ].strip()

            yoloe_expected = row[
                "yoloe_expected_label"
            ].strip()

            if (
                not target_label
                or not yoloe_prompt
                or not yoloe_expected
            ):
                raise ValueError(
                    f"第{row_number}行目标或"
                    "YOLOE字段为空"
                )

            yolo26_text = row[
                "yolo26_expected_label"
            ].strip()

            if group == "common" and not yolo26_text:
                raise ValueError(
                    f"common样本缺少YOLO26"
                    f"预期类别：{sample_id}"
                )

            cases.append(
                BenchmarkCase(
                    sample_id=sample_id,
                    image_path=image_path,
                    group=group,
                    target_label=target_label,
                    yolo26_expected_label=(
                        yolo26_text or None
                    ),
                    yoloe_prompt=yoloe_prompt,
                    yoloe_expected_label=(
                        yoloe_expected
                    ),
                    source_reference=row[
                        "source_reference"
                    ].strip(),
                    usage_rights=row[
                        "usage_rights"
                    ].strip(),
                    notes=row["notes"].strip(),
                )
            )

    if not cases:
        raise ValueError("评测清单没有样本")

    return cases


def label_hit(
    expected_label: str | None,
    detected_labels: Iterable[str],
) -> bool | None:
    if expected_label is None:
        return None

    expected_key = (
        expected_label.strip().casefold()
    )

    detected_keys = {
        label.strip().casefold()
        for label in detected_labels
    }

    return expected_key in detected_keys