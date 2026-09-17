import csv
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from statistics import mean, median


REQUIRED_COLUMNS = {
    "sample_id",
    "group",
    "target_label",
    "mode",
    "eligible",
    "expected_label",
    "prompt",
    "hit",
    "detection_count",
    "detected_labels",
    "prompt_setup_ms",
    "prediction_wall_ms",
    "inference_ms",
    "end_to_end_ms",
}


@dataclass(frozen=True)
class BenchmarkResultRow:
    sample_id: str
    group: str
    target_label: str
    mode: str
    eligible: bool
    expected_label: str
    prompt: str
    hit: bool | None
    detection_count: int
    detected_labels: str
    prompt_setup_ms: float
    prediction_wall_ms: float
    inference_ms: float
    end_to_end_ms: float


@dataclass(frozen=True)
class HitSummary:
    mode: str
    group: str
    eligible_count: int
    hit_count: int
    miss_count: int
    hit_rate: float


@dataclass(frozen=True)
class LatencySummary:
    mode: str
    sample_count: int
    prompt_setup_mean_ms: float
    prediction_wall_mean_ms: float
    inference_mean_ms: float
    inference_median_ms: float
    end_to_end_mean_ms: float
    end_to_end_median_ms: float

# 添加字段解析函数

def _parse_required_bool(
    text: str,
    field_name: str,
    row_number: int,
) -> bool:
    value = text.strip()

    if value == "1":
        return True

    if value == "0":
        return False

    raise ValueError(
        f"第{row_number}行"
        f"{field_name}必须是0或1"
    )


def _parse_optional_bool(
    text: str,
    field_name: str,
    row_number: int,
) -> bool | None:
    value = text.strip()

    if value == "":
        return None

    return _parse_required_bool(
        text=value,
        field_name=field_name,
        row_number=row_number,
    )

# 整数解析

def _parse_non_negative_int(
    text: str,
    field_name: str,
    row_number: int,
) -> int:
    try:
        value = int(text)
    except ValueError as error:
        raise ValueError(
            f"第{row_number}行"
            f"{field_name}不是整数"
        ) from error

    if value < 0:
        raise ValueError(
            f"第{row_number}行"
            f"{field_name}不能小于0"
        )

    return value

# 浮点数解析

def _parse_float(
    text: str,
    field_name: str,
    row_number: int,
) -> float:
    try:
        value = float(text)
    except ValueError as error:
        raise ValueError(
            f"第{row_number}行"
            f"{field_name}不是数字"
        ) from error

    if not isfinite(value):
        raise ValueError(
            f"第{row_number}行"
            f"{field_name}不是有限数字"
        )

    return value

# 实现CSV读取

def load_result_rows(
    result_path: Path,
) -> list[BenchmarkResultRow]:
    if not result_path.is_file():
        raise FileNotFoundError(
            f"实验结果不存在：{result_path}"
        )

    result_rows: list[
        BenchmarkResultRow
    ] = []

    with result_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        columns = set(
            reader.fieldnames or []
        )
        missing_columns = (
            REQUIRED_COLUMNS - columns
        )

        if missing_columns:
            missing_text = ", ".join(
                sorted(missing_columns)
            )
            raise ValueError(
                "实验结果缺少字段："
                f"{missing_text}"
            )

        for row_number, row in enumerate(
            reader,
            start=2,
        ):
            sample_id = (
                row["sample_id"].strip()
            )
            group = row["group"].strip()
            mode = row["mode"].strip()

            if not sample_id:
                raise ValueError(
                    f"第{row_number}行"
                    "sample_id为空"
                )

            if group not in (
                "common",
                "open",
            ):
                raise ValueError(
                    f"第{row_number}行"
                    f"分组无效：{group}"
                )

            if mode not in (
                "YOLO26",
                "YOLOE",
            ):
                raise ValueError(
                    f"第{row_number}行"
                    f"模型无效：{mode}"
                )

            eligible = (
                _parse_required_bool(
                    text=row["eligible"],
                    field_name="eligible",
                    row_number=row_number,
                )
            )

            hit = _parse_optional_bool(
                text=row["hit"],
                field_name="hit",
                row_number=row_number,
            )

            if eligible and hit is None:
                raise ValueError(
                    f"第{row_number}行"
                    "参与评测但hit为空"
                )

            if not eligible and hit is not None:
                raise ValueError(
                    f"第{row_number}行"
                    "不参与评测但hit不为空"
                )

            result_rows.append(
                BenchmarkResultRow(
                    sample_id=sample_id,
                    group=group,
                    target_label=row[
                        "target_label"
                    ].strip(),
                    mode=mode,
                    eligible=eligible,
                    expected_label=row[
                        "expected_label"
                    ].strip(),
                    prompt=row[
                        "prompt"
                    ].strip(),
                    hit=hit,
                    detection_count=(
                        _parse_non_negative_int(
                            text=row[
                                "detection_count"
                            ],
                            field_name=(
                                "detection_count"
                            ),
                            row_number=row_number,
                        )
                    ),
                    detected_labels=row[
                        "detected_labels"
                    ].strip(),
                    prompt_setup_ms=(
                        _parse_float(
                            text=row[
                                "prompt_setup_ms"
                            ],
                            field_name=(
                                "prompt_setup_ms"
                            ),
                            row_number=row_number,
                        )
                    ),
                    prediction_wall_ms=(
                        _parse_float(
                            text=row[
                                "prediction_wall_ms"
                            ],
                            field_name=(
                                "prediction_wall_ms"
                            ),
                            row_number=row_number,
                        )
                    ),
                    inference_ms=_parse_float(
                        text=row["inference_ms"],
                        field_name="inference_ms",
                        row_number=row_number,
                    ),
                    end_to_end_ms=_parse_float(
                        text=row[
                            "end_to_end_ms"
                        ],
                        field_name=(
                            "end_to_end_ms"
                        ),
                        row_number=row_number,
                    ),
                )
            )

    if not result_rows:
        raise ValueError(
            "实验结果文件没有数据"
        )

    return result_rows

# 实现命中率统计

def calculate_hit_summary(
    rows: list[BenchmarkResultRow],
    mode: str,
    group: str,
) -> HitSummary:
    eligible_rows = [
        row
        for row in rows
        if (
            row.mode == mode
            and row.group == group
            and row.eligible
        )
    ]

    if not eligible_rows:
        raise ValueError(
            "没有符合条件的可评测数据："
            f"mode={mode}, group={group}"
        )

    hit_count = sum(
        row.hit is True
        for row in eligible_rows
    )

    eligible_count = len(
        eligible_rows
    )
    miss_count = (
        eligible_count - hit_count
    )
    hit_rate = (
        hit_count / eligible_count
    )

    return HitSummary(
        mode=mode,
        group=group,
        eligible_count=eligible_count,
        hit_count=hit_count,
        miss_count=miss_count,
        hit_rate=hit_rate,
    )

# 实现延时统计

def calculate_latency_summary(
    rows: list[BenchmarkResultRow],
    mode: str,
) -> LatencySummary:
    mode_rows = [
        row
        for row in rows
        if row.mode == mode
    ]

    if not mode_rows:
        raise ValueError(
            f"没有模型数据：{mode}"
        )

    prompt_setup_values = [
        row.prompt_setup_ms
        for row in mode_rows
    ]

    prediction_wall_values = [
        row.prediction_wall_ms
        for row in mode_rows
    ]

    inference_values = [
        row.inference_ms
        for row in mode_rows
    ]

    end_to_end_values = [
        row.end_to_end_ms
        for row in mode_rows
    ]

    return LatencySummary(
        mode=mode,
        sample_count=len(mode_rows),
        prompt_setup_mean_ms=mean(
            prompt_setup_values
        ),
        prediction_wall_mean_ms=mean(
            prediction_wall_values
        ),
        inference_mean_ms=mean(
            inference_values
        ),
        inference_median_ms=median(
            inference_values
        ),
        end_to_end_mean_ms=mean(
            end_to_end_values
        ),
        end_to_end_median_ms=median(
            end_to_end_values
        ),
    )

# 实现失败样本筛选

def collect_failures(
    rows: list[BenchmarkResultRow],
) -> list[BenchmarkResultRow]:
    failures = [
        row
        for row in rows
        if (
            row.eligible
            and row.hit is False
        )
    ]

    return sorted(
        failures,
        key=lambda row: (
            row.group,
            row.sample_id,
            row.mode,
        ),
    )