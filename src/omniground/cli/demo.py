"""TiPToP-faithful grounding demo used by ``pixi run demo``.

Pipeline:

task_instruction -> detect_and_translate template -> complete prompt ->
RGB/PIL image resized to 800 px wide -> PNG multipart POST -> GroundingResult.
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from PIL import Image, ImageDraw, ImageFont

from ..core.config import PROJECT_ROOT, load_config
from ..core.contracts import GroundingResult


EXAMPLES_DIR = PROJECT_ROOT / "examples"
DEFAULT_IMAGE_PATH = EXAMPLES_DIR / "input" / "demo.png"
RESULTS_DIR = EXAMPLES_DIR / "results"
BEIJING_TIMEZONE = ZoneInfo("Asia/Shanghai")
PROMPT_TEMPLATE_PATH = EXAMPLES_DIR / "prompts" / "detect_and_translate.txt"
PROMPT_TEMPLATE_URL = (
    "https://raw.githubusercontent.com/Str0keOOOO/tiptop/main/"
    "tiptop/perception/prompts/detect_and_translate.txt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one TiPToP-compatible OmniGround grounding request."
    )
    parser.add_argument(
        "--task-instruction",
        default=None,
        help="Natural-language task instruction supplied by the user",
    )
    parser.add_argument("--model-id", default=None, help="Model used by the temporary demo server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--config", default=None)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE_PATH)
    parser.add_argument(
        "--result-image",
        type=Path,
        default=None,
        help="Optional output path. Defaults to examples/results/<model-id>/.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    return parser.parse_args()


def default_result_image_path(
    model_id: str,
    request_elapsed_seconds: float,
) -> Path:
    """Return the annotated image path for one model-specific demo run."""
    generated_at = datetime.now(BEIJING_TIMEZONE)
    timestamp = generated_at.strftime("%Y%m%d-%H%M%S-%f")
    run_name = f"{timestamp}_BJT_gen-{request_elapsed_seconds:.3f}s"
    return RESULTS_DIR / model_id / run_name / f"{run_name}.png"


def result_json_path(image_path: Path) -> Path:
    """Store the API response alongside its annotated image."""
    return image_path.with_suffix(".json")


def result_log_path(image_path: Path) -> Path:
    """Store detailed client-side phase timings alongside one result."""
    return image_path.with_suffix(".log")


def save_grounding_result_json(result: GroundingResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        result.model_dump_json(indent=2),
        encoding="utf-8",
    )
    print(f"Grounding JSON 结果已保存：{output_path}")


def save_run_log(output_path: Path, lines: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"运行日志已保存：{output_path}")


def load_prompt_template() -> str:
    """Load a cached TiPToP template, or download it once when absent."""
    if PROMPT_TEMPLATE_PATH.is_file():
        print(
            f"检测到已有 prompt 模板：{PROMPT_TEMPLATE_PATH}。将使用旧模板；"
            "提示：上游模板可能已经更新，如需最新版本请删除该文件后重新运行。"
        )
        return PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")

    print(f"未检测到 prompt 模板，正在下载：{PROMPT_TEMPLATE_URL}")
    try:
        response = requests.get(PROMPT_TEMPLATE_URL, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise SystemExit(f"无法下载 TiPToP prompt 模板：{exc}") from exc

    template = response.text
    if "{task_instruction}" not in template:
        raise SystemExit(
            "下载的 prompt 模板缺少 {task_instruction} 占位符，已拒绝写入本地文件。"
        )

    PROMPT_TEMPLATE_PATH.write_text(template, encoding="utf-8")
    print(f"已下载并保存 prompt 模板：{PROMPT_TEMPLATE_PATH}")
    return template


def render_prompt(template: str, task_instruction: str) -> str:
    if not task_instruction.strip():
        raise ValueError("task_instruction 不能为空")

    try:
        return template.format(task_instruction=task_instruction.strip())
    except (KeyError, ValueError) as exc:
        raise ValueError(f"TiPToP prompt 模板格式不正确：{exc}") from exc


def prepare_tiptop_image(image_path: Path) -> tuple[Image.Image, bytes]:
    """Mirror TiPToP preprocessing: PIL RGB, resize to 800-wide, PNG encode."""
    if not image_path.is_file():
        raise FileNotFoundError(f"找不到测试图片：{image_path}")

    with Image.open(image_path) as source:
        image = source.convert("RGB")

    try:
        target_width = 800
        if image.width != target_width:
            target_height = round(image.height * target_width / image.width)
            image = image.resize(
                (target_width, target_height),
                Image.Resampling.LANCZOS,
            )

        encoded = io.BytesIO()
        image.save(encoded, format="PNG")
        print(
            f"图像预处理完成：PIL RGB，{image.width}x{image.height}，PNG 编码。"
        )
        return image, encoded.getvalue()
    except Exception:
        image.close()
        raise


def draw_grounding_result(
    image: Image.Image,
    result: GroundingResult,
    task_instruction: str,
    model_id: str,
    request_elapsed_seconds: float,
    output_path: Path,
) -> None:
    """Draw normalized boxes, model name, and request timing on the image."""
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    width, height = canvas.size

    title = f"Task: {task_instruction}"
    title_box = draw.textbbox((8, 8), title, font=font)
    draw.rectangle(
        (4, 4, title_box[2] + 8, title_box[3] + 8),
        fill=(0, 0, 0),
    )
    draw.text((8, 8), title, fill=(255, 255, 255), font=font)

    info_text = f"Model: {model_id} | Request time: {request_elapsed_seconds:.3f} s"
    info_box = draw.textbbox((8, 0), info_text, font=font)
    info_height = info_box[3] - info_box[1]
    info_top = height - info_height - 12

    draw.rectangle(
        (
            4,
            info_top - 4,
            min(width - 4, info_box[2] + 8),
            height - 4,
        ),
        fill=(0, 0, 0),
    )
    draw.text(
        (8, info_top),
        info_text,
        fill=(255, 255, 255),
        font=font,
    )

    for bbox in result.bboxes:
        ymin, xmin, ymax, xmax = bbox.box_2d

        left = round(xmin * width / 1000)
        top = round(ymin * height / 1000)
        right = round(xmax * width / 1000)
        bottom = round(ymax * height / 1000)

        draw.rectangle(
            (left, top, right, bottom),
            outline=(255, 64, 64),
            width=4,
        )

        label_box = draw.textbbox((left, top), bbox.label, font=font)
        label_top = max(
            0,
            top - (label_box[3] - label_box[1]) - 6,
        )

        draw.rectangle(
            (left, label_top, label_box[2] + 6, top),
            fill=(255, 64, 64),
        )
        draw.text(
            (left + 3, label_top + 2),
            bbox.label,
            fill=(255, 255, 255),
            font=font,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG")
    canvas.close()

    print(f"检测框结果图已保存：{output_path}")


def wait_until_ready(
    session: requests.Session,
    base_url: str,
    timeout_seconds: float,
) -> float:
    started_at = time.perf_counter()
    deadline = time.monotonic() + timeout_seconds

    while time.monotonic() < deadline:
        try:
            response = session.get(
                base_url + "/ready",
                timeout=1,
            )
            if response.status_code == 200:
                return time.perf_counter() - started_at
        except requests.RequestException:
            pass

        time.sleep(0.2)

    raise RuntimeError("服务未能在超时时间内通过 /ready 检查")


def run_grounding(
    args: argparse.Namespace,
    task_instruction: str,
) -> GroundingResult:
    run_started_at = time.perf_counter()
    phase_started_at = run_started_at
    config = load_config(args.config)
    config_elapsed_seconds = time.perf_counter() - phase_started_at
    model_id = args.model_id or config.default_model
    phase_started_at = time.perf_counter()
    template = load_prompt_template()
    prompt = render_prompt(template, task_instruction)
    prompt_elapsed_seconds = time.perf_counter() - phase_started_at
    phase_started_at = time.perf_counter()
    image, image_bytes = prepare_tiptop_image(args.image)
    image_preprocess_elapsed_seconds = time.perf_counter() - phase_started_at

    print("\n===== task_instruction =====")
    print(task_instruction)

    print("\n===== TiPToP 完整 prompt =====")
    print(prompt)

    command = [
        sys.executable,
        "-m",
        "omniground.cli.server",
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    if args.model_id:
        command.extend(["--model-id", args.model_id])

    if args.config:
        command.extend(["--config", args.config])

    process = subprocess.Popen(command, cwd=PROJECT_ROOT)
    base_url = f"http://{args.host}:{args.port}"

    session = requests.Session()

    # 子服务器位于本地，忽略系统代理配置。
    session.trust_env = False

    try:
        server_ready_elapsed_seconds = wait_until_ready(
            session,
            base_url,
            args.timeout_seconds,
        )

        print(
            "\n服务已就绪，"
            "按 TiPToP multipart/form-data 格式请求 /generate。"
        )

        # 计时开始：只统计 /generate 请求及服务端处理耗时。
        request_started_at = time.perf_counter()

        response = session.post(
            base_url + "/generate",
            files={
                "image": (
                    "image.png",
                    image_bytes,
                    "image/png",
                )
            },
            data={
                "prompt": prompt,
                "temperature": str(args.temperature),
            },
            timeout=args.timeout_seconds,
        )

        # session.post 返回时，响应体已经接收完成。
        request_elapsed_seconds = (
            time.perf_counter() - request_started_at
        )
        backend_inference_header = response.headers.get("X-Backend-Inference-Ms")
        backend_inference_seconds = (
            float(backend_inference_header) / 1000
            if backend_inference_header is not None
            else None
        )
        backend_phase_timings = {
            key.lower().removeprefix("x-backend-timing-").replace("-", "_"): float(value) / 1000
            for key, value in response.headers.items()
            if key.lower().startswith("x-backend-timing-")
        }

        print(
            "Grounding 请求耗时："
            f"{request_elapsed_seconds:.3f} 秒"
        )

        if not response.ok:
            request_id = response.headers.get("X-Request-ID")
            print(
                "Grounding 服务返回错误："
                f"HTTP {response.status_code}"
                + (f"，request_id={request_id}" if request_id else "")
            )
            try:
                print(json.dumps(response.json(), ensure_ascii=False, indent=2))
            except ValueError:
                print(response.text[:4096])

        response.raise_for_status()

        response_parse_started_at = time.perf_counter()
        result = GroundingResult.model_validate(response.json())
        response_parse_elapsed_seconds = time.perf_counter() - response_parse_started_at

        output_path = args.result_image or default_result_image_path(
            model_id=model_id,
            request_elapsed_seconds=request_elapsed_seconds,
        )

        render_started_at = time.perf_counter()
        draw_grounding_result(
            image=image,
            result=result,
            task_instruction=task_instruction,
            model_id=model_id,
            request_elapsed_seconds=request_elapsed_seconds,
            output_path=output_path,
        )
        render_elapsed_seconds = time.perf_counter() - render_started_at

        save_json_started_at = time.perf_counter()
        save_grounding_result_json(
            result,
            result_json_path(output_path),
        )
        save_json_elapsed_seconds = time.perf_counter() - save_json_started_at
        total_elapsed_seconds = time.perf_counter() - run_started_at
        save_run_log(
            result_log_path(output_path),
            [
                f"timestamp_bjt={datetime.now(BEIJING_TIMEZONE).isoformat()}",
                f"model_id={model_id}",
                f"task_instruction={task_instruction}",
                f"config_load_seconds={config_elapsed_seconds:.6f}",
                f"prompt_load_and_render_seconds={prompt_elapsed_seconds:.6f}",
                f"image_preprocess_seconds={image_preprocess_elapsed_seconds:.6f}",
                f"server_startup_and_model_load_seconds={server_ready_elapsed_seconds:.6f}",
                f"generate_http_request_seconds={request_elapsed_seconds:.6f}",
                f"backend_inference_seconds={backend_inference_seconds:.6f}"
                if backend_inference_seconds is not None
                else "backend_inference_seconds=unavailable",
                *[
                    f"backend_{phase_name}_seconds={phase_seconds:.6f}"
                    for phase_name, phase_seconds in sorted(backend_phase_timings.items())
                ],
                f"response_parse_seconds={response_parse_elapsed_seconds:.6f}",
                f"render_output_image_seconds={render_elapsed_seconds:.6f}",
                f"save_output_json_seconds={save_json_elapsed_seconds:.6f}",
                f"total_demo_seconds={total_elapsed_seconds:.6f}",
                "note=generate_http_request includes image upload, server inference, and response transfer;",
                "note=server_startup_and_model_load ends when /ready returns 200.",
            ],
        )

        return result
    finally:
        image.close()
        session.close()

        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def main() -> None:
    args = parse_args()
    task_instruction = args.task_instruction

    if task_instruction is None:
        try:
            task_instruction = input(
                "请输入 task_instruction："
            ).strip()
        except EOFError as exc:
            raise SystemExit(
                "请通过 --task-instruction 提供任务指令。"
            ) from exc

    try:
        result = run_grounding(
            args,
            task_instruction,
        )
    except (
        FileNotFoundError,
        RuntimeError,
        ValueError,
        requests.RequestException,
    ) as exc:
        raise SystemExit(
            f"Grounding 测试失败：{exc}"
        ) from exc

    print("\n===== OmniGround 返回结果 =====")
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
