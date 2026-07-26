from omniground.cli.demo import (
    RESULTS_DIR,
    default_result_image_path,
    render_prompt,
    result_json_path,
    result_log_path,
)


def test_render_prompt_inserts_task_instruction() -> None:
    assert render_prompt(
        'Task: "{task_instruction}". Return {{}}.',
        "pick up the ball",
    ) == 'Task: "pick up the ball". Return {}.'


def test_default_result_image_path_uses_model_directory_and_metadata() -> None:
    output_path = default_result_image_path(
        model_id="rynnbrain1.1-2b",
        request_elapsed_seconds=11.684,
    )

    assert output_path.parent.parent == RESULTS_DIR / "rynnbrain1.1-2b"
    assert output_path.parent.name.endswith("_BJT_gen-11.684s")
    assert output_path.name == f"{output_path.parent.name}.png"
    assert result_json_path(output_path) == output_path.with_suffix(".json")
    assert result_log_path(output_path) == output_path.with_suffix(".log")
