from pathlib import Path

from omniground.backends.transformers_grounding import TransformersGroundingBackend
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

    assert isinstance(rynn, TransformersGroundingBackend)
    assert isinstance(robo, TransformersGroundingBackend)
