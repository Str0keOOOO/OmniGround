"""TiPToP-faithful grounding demo used by ``pixi run demo``.

Pipeline:

task_instruction -> detect_and_translate template -> complete prompt ->
RGB/PIL image resized to 800 px wide -> PNG multipart POST -> GroundingResult.
"""

from __future__ import annotations

import argparse
import io
import subprocess
import sys
import time
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

from omniground.config import PROJECT_ROOT
from omniground.schemas import GroundingResult


EXAMPLES_DIR = PROJECT_ROOT / "examples"
DEFAULT_IMAGE_PATH = EXAMPLES_DIR / "demo.png"
DEFAULT_RESULT_IMAGE_PATH = EXAMPLES_DIR / "result.png"
PROMPT_TEMPLATE_PATH = EXAMPLES_DIR / "detect_and_translate.txt"
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
        default=DEFAULT_RESULT_IMAGE_PATH,
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    return parser.parse_args()


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
    request_elapsed_seconds: float,
    output_path: Path,
) -> None:
    """Draw normalized boxes and request timing on the preprocessed image."""
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

    info_text = f"Request time: {request_elapsed_seconds:.3f} s"
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
) -> None:
    deadline = time.monotonic() + timeout_seconds

    while time.monotonic() < deadline:
        try:
            response = session.get(
                base_url + "/ready",
                timeout=1,
            )
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass

        time.sleep(0.2)

    raise RuntimeError("服务未能在超时时间内通过 /ready 检查")


def run_grounding(
    args: argparse.Namespace,
    task_instruction: str,
) -> GroundingResult:
    template = load_prompt_template()
    prompt = render_prompt(template, task_instruction)
    image, image_bytes = prepare_tiptop_image(args.image)

    print("\n===== task_instruction =====")
    print(task_instruction)

    print("\n===== TiPToP 完整 prompt =====")
    print(prompt)

    command = [
        sys.executable,
        "-m",
        "omniground.cli.run_server",
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
        wait_until_ready(
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

        print(
            "Grounding 请求耗时："
            f"{request_elapsed_seconds:.3f} 秒"
        )

        response.raise_for_status()

        result = GroundingResult.model_validate(
            response.json()
        )

        draw_grounding_result(
            image=image,
            result=result,
            task_instruction=task_instruction,
            request_elapsed_seconds=request_elapsed_seconds,
            output_path=args.result_image,
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


def test_render_prompt_inserts_task_instruction() -> None:
    assert render_prompt(
        'Task: "{task_instruction}". Return {{}}.',
        "pick up the ball",
    ) == 'Task: "pick up the ball". Return {}.'


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
