import argparse
import csv
from pathlib import Path
from time import perf_counter
from typing import cast

from vision_studio.benchmark import (
    BenchmarkCase,
    label_hit,
    load_manifest,
)
from vision_studio.config import PROJECT_ROOT
from vision_studio.detectors import (
    BaseDetector,
    YOLOEDetector,
)
from vision_studio.model_cache import (
    DetectorCache,
    ModelMode,
)
from vision_studio.prompt_utils import (
    parse_prompt_text,
)


CONFIDENCE = 0.25
IOU = 0.70
IMAGE_SIZE = 640

FIELDNAMES = [
    "sample_id",
    "image_path",
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
    "preprocess_ms",
    "inference_ms",
    "postprocess_ms",
    "model_total_ms",
    "end_to_end_ms",
    "image_width",
    "image_height",
    "confidence_threshold",
    "iou_threshold",
    "input_size",
]


def prepare_yoloe(
    detector: BaseDetector,
    prompt_text: str,
) -> None:
    yoloe_detector = cast(
        YOLOEDetector,
        detector,
    )

    yoloe_detector.set_classes(
        parse_prompt_text(prompt_text)
    )


def warm_up(
    cache: DetectorCache,
    first_case: BenchmarkCase,
) -> None:
    print("正在预热 YOLO26...")
    yolo26 = cache.get("YOLO26")
    yolo26.predict(
        image_path=first_case.image_path,
        confidence=CONFIDENCE,
        iou=IOU,
        image_size=IMAGE_SIZE,
    )

    print("正在预热 YOLOE...")
    yoloe = cache.get("YOLOE")
    prepare_yoloe(
        yoloe,
        "benchmark warmup object"
    )
    yoloe.predict(
        image_path=first_case.image_path,
        confidence=CONFIDENCE,
        iou=IOU,
        image_size=IMAGE_SIZE,
    )

def evaluate_case(
    case: BenchmarkCase,
    mode: ModelMode,
    cache: DetectorCache,
) -> dict[str, object]:
    detector = cache.get(mode)

    if mode == "YOLO26":
        expected_label = (
            case.yolo26_expected_label
        )
        prompt_text = ""
    else:
        expected_label = (
            case.yoloe_expected_label
        )
        prompt_text = case.yoloe_prompt

    start_time = perf_counter()
    prompt_setup_ms = 0.0

    if mode == "YOLOE":
        prompt_start = perf_counter()

        prepare_yoloe(
            detector,
            prompt_text,
        )

        prompt_setup_ms = (
            perf_counter() - prompt_start
        ) * 1000.0

    prediction_start = perf_counter()

    result = detector.predict(
        image_path=case.image_path,
        confidence=CONFIDENCE,
        iou=IOU,
        image_size=IMAGE_SIZE,
    )

    prediction_wall_ms = (
        perf_counter() - prediction_start
    ) * 1000.0

    end_to_end_ms = (
        perf_counter() - start_time
    ) * 1000.0

    detected_labels = [
        detection.label
        for detection in result.detections
    ]

    unique_labels = list(
        dict.fromkeys(detected_labels)
    )

    hit = label_hit(
        expected_label,
        detected_labels,
    )

    relative_path = (
        case.image_path
        .relative_to(PROJECT_ROOT)
        .as_posix()
    )

    return {
        "sample_id": case.sample_id,
        "image_path": relative_path,
        "group": case.group,
        "target_label": case.target_label,
        "mode": mode,
        "eligible": (
            1 if expected_label is not None else 0
        ),
        "expected_label": (
            expected_label or ""
        ),
        "prompt": prompt_text,
        "hit": (
            ""
            if hit is None
            else int(hit)
        ),
        "detection_count": (
            result.detection_count
        ),
        "detected_labels": "|".join(
            unique_labels
        ),
        "prompt_setup_ms": round(
            prompt_setup_ms,
            2,
        ),
        "prediction_wall_ms": round(
            prediction_wall_ms,
            2,
        ),
        "preprocess_ms": (
            result.preprocess_ms
        ),
        "inference_ms": result.inference_ms,
        "postprocess_ms": (
            result.postprocess_ms
        ),
        "model_total_ms": result.total_ms,
        "end_to_end_ms": round(
            end_to_end_ms,
            2,
        ),
        "image_width": result.image_width,
        "image_height": result.image_height,
        "confidence_threshold": CONFIDENCE,
        "iou_threshold": IOU,
        "input_size": IMAGE_SIZE,
    }

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="运行图片检测对比实验"
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            PROJECT_ROOT
            / "benchmarks"
            / "manifest.csv"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "benchmarks"
            / "results"
            / "results.csv"
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    cases = load_manifest(
        manifest_path=arguments.manifest,
        project_root=PROJECT_ROOT,
    )

    if arguments.limit is not None:
        if arguments.limit <= 0:
            raise ValueError(
                "limit必须大于0"
            )

        cases = cases[:arguments.limit]

    cache = DetectorCache()
    warm_up(cache, cases[0])

    rows: list[dict[str, object]] = []

    total_tasks = len(cases) * 2
    completed_tasks = 0

    for case in cases:
        for mode in ("YOLO26", "YOLOE"):
            row = evaluate_case(
                case=case,
                mode=mode,
                cache=cache,
            )
            rows.append(row)

            completed_tasks += 1
            print(
                f"[{completed_tasks}/{total_tasks}] "
                f"{case.sample_id} {mode} "
                f"hit={row['hit']}"
            )

    arguments.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with arguments.output.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=FIELDNAMES,
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"实验结果已保存：{arguments.output}")


if __name__ == "__main__":
    main()