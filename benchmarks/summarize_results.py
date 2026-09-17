import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import (
    PercentFormatter,
)

from vision_studio.benchmark_report import (
    BenchmarkResultRow,
    HitSummary,
    LatencySummary,
    calculate_hit_summary,
    calculate_latency_summary,
    collect_failures,
    load_result_rows,
)
from vision_studio.config import (
    PROJECT_ROOT,
)

# 添加参数处理

def parse_arguments() -> (
    argparse.Namespace
):
    parser = argparse.ArgumentParser(
        description=(
            "汇总YOLO26与YOLOE实验结果"
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=(
            PROJECT_ROOT
            / "benchmarks"
            / "results"
            / "results.csv"
        ),
        help="原始实验结果CSV",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "benchmarks"
            / "results"
            / "report"
        ),
        help="统计报告输出目录",
    )

    return parser.parse_args()

# 路径处理

def resolve_project_path(
    path: Path,
) -> Path:
    if path.is_absolute():
        return path

    return (
        PROJECT_ROOT / path
    ).resolve()

# 保存命中率CSV

def save_hit_summary(
    output_path: Path,
    summaries: list[HitSummary],
) -> None:
    fieldnames = [
        "mode",
        "group",
        "eligible_count",
        "hit_count",
        "miss_count",
        "hit_rate",
    ]

    with output_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for summary in summaries:
            writer.writerow(
                {
                    "mode": summary.mode,
                    "group": summary.group,
                    "eligible_count": (
                        summary.eligible_count
                    ),
                    "hit_count": (
                        summary.hit_count
                    ),
                    "miss_count": (
                        summary.miss_count
                    ),
                    "hit_rate": round(
                        summary.hit_rate,
                        4,
                    ),
                }
            )

# 保存延迟CSV


def save_latency_summary(
    output_path: Path,
    summaries: list[LatencySummary],
) -> None:
    fieldnames = [
        "mode",
        "sample_count",
        "prompt_setup_mean_ms",
        "prediction_wall_mean_ms",
        "inference_mean_ms",
        "inference_median_ms",
        "end_to_end_mean_ms",
        "end_to_end_median_ms",
    ]

    with output_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for summary in summaries:
            writer.writerow(
                {
                    "mode": summary.mode,
                    "sample_count": (
                        summary.sample_count
                    ),
                    "prompt_setup_mean_ms": (
                        round(
                            summary
                            .prompt_setup_mean_ms,
                            2,
                        )
                    ),
                    "prediction_wall_mean_ms": (
                        round(
                            summary
                            .prediction_wall_mean_ms,
                            2,
                        )
                    ),
                    "inference_mean_ms": (
                        round(
                            summary
                            .inference_mean_ms,
                            2,
                        )
                    ),
                    "inference_median_ms": (
                        round(
                            summary
                            .inference_median_ms,
                            2,
                        )
                    ),
                    "end_to_end_mean_ms": (
                        round(
                            summary
                            .end_to_end_mean_ms,
                            2,
                        )
                    ),
                    "end_to_end_median_ms": (
                        round(
                            summary
                            .end_to_end_median_ms,
                            2,
                        )
                    ),
                }
            )

# 保存失败案例CSV

def save_failures(
    output_path: Path,
    failures: list[
        BenchmarkResultRow
    ],
) -> None:
    fieldnames = [
        "sample_id",
        "group",
        "target_label",
        "mode",
        "expected_label",
        "prompt",
        "detection_count",
        "detected_labels",
        "inference_ms",
        "end_to_end_ms",
    ]

    with output_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for row in failures:
            writer.writerow(
                {
                    "sample_id": row.sample_id,
                    "group": row.group,
                    "target_label": (
                        row.target_label
                    ),
                    "mode": row.mode,
                    "expected_label": (
                        row.expected_label
                    ),
                    "prompt": row.prompt,
                    "detection_count": (
                        row.detection_count
                    ),
                    "detected_labels": (
                        row.detected_labels
                    ),
                    "inference_ms": (
                        row.inference_ms
                    ),
                    "end_to_end_ms": (
                        row.end_to_end_ms
                    ),
                }
            )


# 绘制命中率图

def plot_hit_rate(
    output_path: Path,
    summaries: list[HitSummary],
) -> None:
    labels = [
        f"{item.mode}\n{item.group}"
        for item in summaries
    ]
    values = [
        item.hit_rate
        for item in summaries
    ]

    colors = [
        "#2563EB",
        "#16A34A",
        "#F59E0B",
    ]

    figure, axis = plt.subplots(
        figsize=(8, 5)
    )

    bars = axis.bar(
        labels,
        values,
        color=colors,
        width=0.6,
    )

    axis.set_title(
        "Target Hit Rate"
    )
    axis.set_ylabel(
        "Hit rate"
    )
    axis.set_ylim(
        0,
        1.1,
    )
    axis.yaxis.set_major_formatter(
        PercentFormatter(1.0)
    )
    axis.grid(
        axis="y",
        alpha=0.25,
    )

    axis.bar_label(
        bars,
        labels=[
            f"{value:.1%}"
            for value in values
        ],
        padding=3,
    )

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


# 绘制延迟图

def plot_latency(
    output_path: Path,
    summaries: list[
        LatencySummary
    ],
) -> None:
    labels = [
        item.mode
        for item in summaries
    ]

    inference_values = [
        item.inference_mean_ms
        for item in summaries
    ]

    end_to_end_values = [
        item.end_to_end_mean_ms
        for item in summaries
    ]

    positions = list(
        range(len(labels))
    )
    width = 0.34

    left_positions = [
        position - width / 2
        for position in positions
    ]
    right_positions = [
        position + width / 2
        for position in positions
    ]

    figure, axis = plt.subplots(
        figsize=(8, 5)
    )

    inference_bars = axis.bar(
        left_positions,
        inference_values,
        width=width,
        label="Model inference",
        color="#2563EB",
    )

    end_to_end_bars = axis.bar(
        right_positions,
        end_to_end_values,
        width=width,
        label="End-to-end",
        color="#F97316",
    )

    axis.set_title(
        "Average Processing Latency"
    )
    axis.set_ylabel(
        "Milliseconds"
    )
    axis.set_xticks(
        positions,
        labels,
    )
    axis.legend()
    axis.grid(
        axis="y",
        alpha=0.25,
    )

    axis.bar_label(
        inference_bars,
        fmt="%.1f",
        padding=3,
    )
    axis.bar_label(
        end_to_end_bars,
        fmt="%.1f",
        padding=3,
    )

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


# 实现main()

def main() -> None:
    arguments = parse_arguments()

    input_path = resolve_project_path(
        arguments.input
    )
    output_dir = resolve_project_path(
        arguments.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = load_result_rows(
        input_path
    )

    hit_summaries = [
        calculate_hit_summary(
            rows,
            mode="YOLO26",
            group="common",
        ),
        calculate_hit_summary(
            rows,
            mode="YOLOE",
            group="common",
        ),
        calculate_hit_summary(
            rows,
            mode="YOLOE",
            group="open",
        ),
    ]

    latency_summaries = [
        calculate_latency_summary(
            rows,
            mode="YOLO26",
        ),
        calculate_latency_summary(
            rows,
            mode="YOLOE",
        ),
    ]

    failures = collect_failures(
        rows
    )

    hit_summary_path = (
        output_dir / "hit_summary.csv"
    )
    latency_summary_path = (
        output_dir
        / "latency_summary.csv"
    )
    failures_path = (
        output_dir / "failures.csv"
    )
    hit_chart_path = (
        output_dir / "hit_rate.png"
    )
    latency_chart_path = (
        output_dir / "latency.png"
    )

    save_hit_summary(
        hit_summary_path,
        hit_summaries,
    )
    save_latency_summary(
        latency_summary_path,
        latency_summaries,
    )
    save_failures(
        failures_path,
        failures,
    )
    plot_hit_rate(
        hit_chart_path,
        hit_summaries,
    )
    plot_latency(
        latency_chart_path,
        latency_summaries,
    )

    print(
        f"读取实验记录：{len(rows)}行"
    )

    for summary in hit_summaries:
        print(
            f"{summary.mode} "
            f"{summary.group}："
            f"{summary.hit_count}/"
            f"{summary.eligible_count}，"
            f"{summary.hit_rate:.1%}"
        )

    for summary in latency_summaries:
        print(
            f"{summary.mode}："
            f"平均推理"
            f"{summary.inference_mean_ms:.2f} ms，"
            f"平均端到端"
            f"{summary.end_to_end_mean_ms:.2f} ms"
        )

    print(
        f"失败样本：{len(failures)}"
    )
    print(
        f"报告目录：{output_dir}"
    )


if __name__ == "__main__":
    main()         