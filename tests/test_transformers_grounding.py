from pathlib import Path

from omniground.backends.robobrain25 import RoboBrainBackend, parse_robobrain_grounding
from omniground.backends.transformers_grounding import TransformersGroundingBackend
from omniground.backends.rynnbrain11 import RynnBrainBackend, parse_rynnbrain_grounding
from omniground.config import AppConfig, ModelConfig
from omniground.registry import ModelRegistry


def test_transformers_grounding_prompt_preserves_request_and_contract() -> None:
    prompt = TransformersGroundingBackend._render_prompt("locate the yellow ball")

    assert prompt.startswith("locate the yellow ball\n\n")
    assert '"bboxes"' in prompt
    assert '"predicates"' in prompt
    assert "[ymin,xmin,ymax,xmax]" in prompt


def test_registry_creates_one_shared_adapter_for_both_embodied_families() -> None:
    config = AppConfig(
        default_model="rynn",
        models={
            "rynn": ModelConfig(backend="rynnbrain11", mode="local"),
            "robo": ModelConfig(backend="robobrain25", mode="local"),
        },
        path=Path("models.yaml"),
    )
    registry = ModelRegistry(config)

    rynn = registry._create_backend("rynn", config.models["rynn"])
    robo = registry._create_backend("robo", config.models["robo"])

    assert isinstance(rynn, RynnBrainBackend)
    assert isinstance(robo, RoboBrainBackend)


def test_rynnbrain_native_output_converts_to_grounding_contract() -> None:
    result = parse_rynnbrain_grounding(
        "<object> (153, 249), (412, 537) </object>",
        "pick up the yellow ball",
    )

    assert result.model_dump() == {
        "bboxes": [{"box_2d": (249, 153, 537, 412), "label": "yellow_ball"}],
        "predicates": [{"name": "holding", "args": ["yellow_ball"]}],
    }


def test_rynnbrain_uses_native_prompt_instead_of_json_contract() -> None:
    prompt = RynnBrainBackend._render_prompt(
        'Perform two tasks on this image based on the task instruction: "pick up the yellow ball".'
    )

    assert "pick up the yellow ball" in prompt
    assert "<object> (x1, y1), (x2, y2) </object>" in prompt
    assert '"bboxes"' not in prompt


def test_robobrain_native_output_converts_to_grounding_contract() -> None:
    result = parse_robobrain_grounding(
        "The target is [153, 249, 412, 537].",
        "pick up the yellow ball",
    )

    assert result.model_dump() == {
        "bboxes": [{"box_2d": (249, 153, 537, 412), "label": "yellow_ball"}],
        "predicates": [{"name": "holding", "args": ["yellow_ball"]}],
    }


def test_robobrain_uses_official_grounding_prompt() -> None:
    prompt = RoboBrainBackend._render_prompt(
        'Perform two tasks on this image based on the task instruction: "pick up the yellow ball".'
    )

    assert "pick up the yellow ball" in prompt
    assert "[x1, y1, x2, y2]" in prompt
    assert '"bboxes"' not in prompt
