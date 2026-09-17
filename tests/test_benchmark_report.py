import csv
from pathlib import Path

import pytest

from vision_studio.benchmark_report import (
    calculate_hit_summary,
    calculate_latency_summary,
    collect_failures,
    load_result_rows,
)


FIELDNAMES = [
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
]


def make_row(
    sample_id: str,
    mode: str,
    group: str = "common",
    eligible: str = "1",
    hit: str = "1",
    inference_ms: str = "10.0",
    end_to_end_ms: str = "20.0",
) -> dict[str, str]:
    return {
        "sample_id": sample_id,
        "group": group,
        "target_label": "object",
        "mode": mode,
        "eligible": eligible,
        "expected_label": (
            "object"
            if eligible == "1"
            else ""
        ),
        "prompt": (
            "object"
            if mode == "YOLOE"
            else ""
        ),
        "hit": hit,
        "detection_count": "1",
        "detected_labels": "object",
        "prompt_setup_ms": (
            "100.0"
            if mode == "YOLOE"
            else "0.0"
        ),
        "prediction_wall_ms": (
            end_to_end_ms
        ),
        "inference_ms": inference_ms,
        "end_to_end_ms": (
            end_to_end_ms
        ),
    }


def write_results(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=FIELDNAMES,
        )
        writer.writeheader()
        writer.writerows(rows)


# 添加文件不存在测试


def test_load_results_rejects_missing_file(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        FileNotFoundError,
        match="实验结果不存在",
    ):
        load_result_rows(
            tmp_path / "missing.csv"
        )

# 添加空hit测试

def test_load_results_parses_blank_hit_as_none(
    tmp_path: Path,
) -> None:
    result_path = (
        tmp_path / "results.csv"
    )

    write_results(
        result_path,
        [
            make_row(
                sample_id="open_001",
                mode="YOLO26",
                group="open",
                eligible="0",
                hit="",
            )
        ],
    )

    rows = load_result_rows(
        result_path
    )

    assert len(rows) == 1
    assert rows[0].eligible is False
    assert rows[0].hit is None


# 添加"0"解析测试

def test_load_results_parses_zero_hit_as_false(
    tmp_path: Path,
) -> None:
    result_path = (
        tmp_path / "results.csv"
    )

    write_results(
        result_path,
        [
            make_row(
                sample_id="open_001",
                mode="YOLOE",
                group="open",
                hit="0",
            )
        ],
    )

    rows = load_result_rows(
        result_path
    )

    assert rows[0].hit is False

# 添加命中率测试

def test_hit_summary_excludes_ineligible_rows(
    tmp_path: Path,
) -> None:
    result_path = (
        tmp_path / "results.csv"
    )

    write_results(
        result_path,
        [
            make_row(
                sample_id="001",
                mode="YOLO26",
                hit="1",
            ),
            make_row(
                sample_id="002",
                mode="YOLO26",
                hit="0",
            ),
            make_row(
                sample_id="003",
                mode="YOLO26",
                eligible="0",
                hit="",
            ),
        ],
    )

    rows = load_result_rows(
        result_path
    )

    summary = calculate_hit_summary(
        rows,
        mode="YOLO26",
        group="common",
    )

    assert summary.eligible_count == 2
    assert summary.hit_count == 1
    assert summary.miss_count == 1
    assert summary.hit_rate == 0.5

# 添加延迟测试

def test_latency_summary_calculates_mean_and_median(
    tmp_path: Path,
) -> None:
    result_path = (
        tmp_path / "results.csv"
    )

    write_results(
        result_path,
        [
            make_row(
                sample_id="001",
                mode="YOLO26",
                inference_ms="10.0",
                end_to_end_ms="30.0",
            ),
            make_row(
                sample_id="002",
                mode="YOLO26",
                inference_ms="20.0",
                end_to_end_ms="50.0",
            ),
        ],
    )

    rows = load_result_rows(
        result_path
    )

    summary = (
        calculate_latency_summary(
            rows,
            mode="YOLO26",
        )
    )

    assert summary.sample_count == 2
    assert (
        summary.inference_mean_ms
        == 15.0
    )
    assert (
        summary.inference_median_ms
        == 15.0
    )
    assert (
        summary.end_to_end_mean_ms
        == 40.0
    )
    assert (
        summary.end_to_end_median_ms
        == 40.0
    )

# 添加失败筛选测试

def test_collect_failures_only_returns_real_failures(
    tmp_path: Path,
) -> None:
    result_path = (
        tmp_path / "results.csv"
    )

    write_results(
        result_path,
        [
            make_row(
                sample_id="hit",
                mode="YOLOE",
                hit="1",
            ),
            make_row(
                sample_id="miss",
                mode="YOLOE",
                hit="0",
            ),
            make_row(
                sample_id="not_applicable",
                mode="YOLO26",
                eligible="0",
                hit="",
            ),
        ],
    )

    rows = load_result_rows(
        result_path
    )
    failures = collect_failures(
        rows
    )

    assert len(failures) == 1
    assert (
        failures[0].sample_id
        == "miss"
    )

