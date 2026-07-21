# OmniGround

OmniGround is a unified HTTP service for robot visual grounding and task semantics. It presents one strict TiPToP-compatible contract above multiple vision-language models (VLMs), including local Molmo2, isolated HTTP services, and OpenAI-compatible APIs.

The first release prioritizes a stable protocol, validation, adapter boundary, and test harness over loading many models in one process. Configure either the OpenAI-compatible API or a local model for inference.

# Input and Output Contract

This is the public API. A client does not need to inspect source code to use OmniGround correctly.

## Request

| Item | Value |
|---|---|
| URLs | `POST /generate` and the fully identical alias `POST /v1/generate` |
| Content-Type | `multipart/form-data` |
| Default server address | `http://127.0.0.1:8011` |

| Field | Required | Type | Meaning |
|---|---:|---|---|
| `image` | Yes | PNG or JPEG file | One non-empty image. Unsupported files receive `415`. |
| `prompt` | Yes | UTF-8 string | The complete task prompt. OmniGround passes it to the model unchanged. |
| `model_id` | Yes | string | Configured model key, e.g. `openai-compatible` or `molmo2-er`. Unknown keys receive `404`. |
| `temperature` | No | number >= 0 | Sampling temperature; the model config supplies the default when omitted. |

The request body limit is 16 MiB by default and can be changed with `--max-request-bytes`.

```bash
curl -X POST http://127.0.0.1:8011/generate \
  -F "image=@examples/demo_image.png" \
  -F "prompt=<完整的任务 prompt>" \
  -F "model_id=molmo2-er" \
  -F "temperature=0"
```

Python `requests` example:

```python
from pathlib import Path
import requests

with Path("examples/demo_image.png").open("rb") as image:
    response = requests.post(
        "http://127.0.0.1:8011/generate",
        files={"image": ("demo_image.png", image, "image/png")},
        data={
            "prompt": "Return exactly one JSON object with bboxes and predicates.",
            "model_id": "openai-compatible",
            "temperature": "0",
        },
        timeout=120,
    )
response.raise_for_status()
result = response.json()
```

## Successful response

Every successful `/generate` response has `application/json` content type and its body directly is this object—there is no vendor wrapper:

```json
{
  "bboxes": [
    {
      "box_2d": [240, 310, 670, 720],
      "label": "yellow_ball"
    },
    {
      "box_2d": [500, 100, 900, 850],
      "label": "black_stool_seat"
    }
  ],
  "predicates": [
    {
      "name": "holding",
      "args": ["yellow_ball"]
    }
  ]
}
```

Rules enforced for all backends:

1. `box_2d` is strictly `[ymin, xmin, ymax, xmax]`, never `[xmin, ymin, xmax, ymax]`.
2. Each coordinate is an integer normalized to `0`–`1000`; pixels and `0`–`1` floats are invalid unless a future, explicitly configured adapter converts them.
3. A box satisfies `0 <= ymin < ymax <= 1000` and `0 <= xmin < xmax <= 1000`.
4. Each `label` is a non-empty trimmed string and is unique. Lowercase underscore labels such as `yellow_ball` and `red_can_left` are recommended.
5. A predicate is `{ "name": string, "args": [string, ...] }`. The initial protocol supports `holding(movable)` and `on(movable, surface)`.
6. Every predicate argument must exactly match an emitted bbox label.
7. Both top-level fields are always present. An empty result is:

```json
{"bboxes": [], "predicates": []}
```

Never return an envelope such as:

```json
{"result": {"bboxes": [], "predicates": []}}
```

or:

```json
{"text": "{\"bboxes\": [], \"predicates\": []}"}
```

The response must not contain Markdown fences, explanations, natural language, or model reasoning. TiPToP parses this body directly.

## Error response

Errors are JSON with a stable code and a safe message. Python tracebacks, complete prompts, and raw model output are never returned to clients.

```json
{
  "error": {
    "code": "INVALID_MODEL_OUTPUT",
    "message": "The selected model did not return valid OmniGround JSON."
  }
}
```

Typical statuses are `400` invalid input, `404` unknown model, `413` too-large request, `415` unsupported image, `502` inference/model-output failure, and `503` unavailable model or optional dependency.

## Architecture

```text
TiPToP / curl / test client
            |
            v
 FastAPI: /generate, /v1/generate
            |
            v
  ModelRegistry (models.yaml, lazy cache)
       |                              |
       v                              v
 Molmo2Backend              OpenAICompatibleBackend
       |                              |
 local PyTorch process               external API
                    \              /
                     v            v
       parser.py -> Pydantic validation -> direct GroundingResult JSON
```

The HTTP server has no model-specific inference code. Every model implements `BaseBackend`, returns a validated `GroundingResult`, and is selected through the registry. Torch, Transformers, and network-client imports occur only in the relevant adapter's lazy path.

## Layout

```text
OmniGround/
├── main.py                       # thin python entry point
├── configs/models.yaml           # model registry configuration
├── examples/                     # PNG and complete demo prompt
├── third_party/molmo2/           # optional Git submodule checkout
├── src/omniground/
│   ├── server.py                 # model-independent FastAPI contract
│   ├── schemas.py                # public Pydantic response schema
│   ├── parser.py                 # conservative raw-text parsing
│   ├── validation.py             # validation helpers
│   ├── registry.py               # configuration and backend cache
│   ├── backends/                 # local and API adapters
│   └── cli/                      # setup, download, server, and demo tasks
└── tests/                        # GPU-free contract and unit tests
```

## Quick start

Install [Pixi](https://pixi.sh/), then initialize the base environment:

```bash
cd OmniGround
pixi install
pixi run setup
```

`setup` initializes checked-out submodules and checks base dependencies. It does not install every optional VLM dependency. For a clone that needs Molmo2 source, use either:

```bash
git clone --recursive https://github.com/Str0keOOOO/OmniGround.git
```

or:

```bash
git submodule update --init --recursive
```

### API protocol demo

Set the API credential and start the configured Qwen-compatible backend:

```bash
export OPENAI_API_KEY='replace-with-your-key'
pixi run server -- --model-id openai-compatible
```

Then use the curl example with `model_id=openai-compatible`, or run the self-cleaning demo:

```bash
pixi run demo -- --task-instruction "pick up the ipad"
pixi run demo -- --model-id openai-compatible --task-instruction "pick up the ipad"
```

`pixi run demo` executes `tests/test_ground.py`. It asks for `task_instruction` (or accepts `--task-instruction`), fills TiPToP's `detect_and_translate.txt`, opens `examples/demo.png` as RGB, resizes it to 800 pixels wide, PNG-encodes it, sends the exact TiPToP-style multipart request, validates the direct JSON using Pydantic, and always stops the child process. A schema failure exits non-zero. The template is cached at `examples/detect_and_translate.txt`: an existing file is used with a warning that it may be outdated; a missing file is downloaded from the TiPToP repository.

The server listens on `0.0.0.0:8011` and never launches a demo itself:

```bash
pixi run server -- --host 0.0.0.0 --port 8011 --model-id openai-compatible
python main.py --model-id openai-compatible
python -m omniground --model-id openai-compatible
```

The last two commands work after project installation, for example `pixi install`.

## Models and backends

`configs/models.yaml` is the only mapping from `model_id` to an adapter; it contains no credentials.

| `model_id` | Backend | Mode | Use |
|---|---|---|---|
| `molmo2-er` | `molmo2` | local | Direct Molmo2 checkpoint in this process |
| `openai-compatible` | `openai_compatible` | api | OpenAI-style multimodal `chat/completions` endpoint |

Only the requested backend is created. It loads lazily on its first generation request; OmniGround never loads all configured VLMs at startup. `/v1/models` lists id, backend, mode, and load status without API keys. `/health` means the process is alive; `/ready?model_id=<id>` checks basic availability without forcing a large model into GPU memory.

### Molmo2 local backend

`backends/molmo2.py` verifies `third_party/molmo2` and `models/Molmo2-ER`, imports Torch/Transformers only at first use, supports the configured CPU/GPU device setting, and serializes generation for single-GPU safety. It keeps raw text for protected debug logging and sends that text through the common strict parser.

Download only when ready to use it:

```bash
pixi run download-checkpoints
pixi run download-checkpoints -- molmo2-er
pixi run server -- --model-id molmo2-er
pixi run demo -- --model-id molmo2-er --task-instruction "pick up the yellow ball"
```

The downloader skips a non-empty target, accepts `--output-dir`, imports `huggingface_hub` only when called, and weights are ignored by Git. Install optional local dependencies in a compatible environment:

```bash
pip install -e '.[molmo2,download]'
```

Molmo2 may emit points more naturally than boxes. OmniGround asks it for boxes; a point-only response produces `POINT_OUTPUT_NOT_SUPPORTED`. It never invents a box around a point or guesses coordinate order.

### Local and API modes

| Mode | Model location | Best when |
|---|---|---|
| `local` | OmniGround process | Dependencies are compatible with this environment. |
| `api` | External provider | No third-party checkout is necessary. |

VLMs may conflict on Python, PyTorch, CUDA, Transformers, FlashAttention, xformers, vLLM, torchcodec, or tokenizers. Do not assume they can all coexist in one Pixi environment. Use the API backend when a local model's dependency stack is unsuitable.

The OpenAI-compatible adapter uses the lazy-loaded official `OpenAI` SDK client for multimodal chat-completions data. Its `openai-compatible` YAML entry is configured for `qwen3.7-plus`, the compatible endpoint, and `extra_body.enable_thinking: true`.

Set the credential in the shell before starting the API backend; never commit a real key to YAML or source code:

```bash
export OPENAI_API_KEY='replace-with-your-rotated-key'
pixi install
pixi run server -- --model-id openai-compatible
```

The SDK call is non-streaming (`stream=False`) because OmniGround must return one validated JSON body. `enable_thinking` is forwarded to the provider, but `reasoning_content` is deliberately ignored: `/generate` returns only the final JSON from `message.content`, never a thinking trace.

## TiPToP configuration

```yaml
perception:
  vlm:
    url: "http://127.0.0.1:8011"
    endpoint: "/generate"
    timeout_seconds: 120
```

TiPToP supplies the full prompt and expects direct `bboxes` and `predicates`. An OpenAI, Gemini, `result`, or `text` envelope is not compatible.

## Adding a new model

Adding a model does not require modifying `server.py`:

1. Decide whether source checkout is required; add a submodule at `third_party/<model-name>` only if it is.
2. Create `src/omniground/backends/<model_name>.py`.
3. Inherit `BaseBackend`; keep optional imports in `load()`.
4. Implement `load()`, `generate()`, and optionally `unload()`.
5. Preserve raw model output and return a validated `GroundingResult`.
6. Register the id and backend settings in `configs/models.yaml`.
7. Add the adapter to the small `registry.py` factory; do not change the HTTP request handler.
8. Add parser/contract tests and a downloader when relevant.
9. Update the models table and document dependencies.

```python
class ExampleBackend(BaseBackend):
    def load(self) -> None:
        ...

    def generate(self, request: GenerationRequest) -> GroundingResult:
        raw_text = self.model.generate(image=request.image, prompt=request.prompt)
        self.last_raw_text = raw_text
        return parse_and_validate(raw_text)
```

The parser safely removes one JSON Markdown fence and a deterministic trailing comma before `]` or `}`. It accepts one JSON object surrounded by prose for interoperability, but rejects multiple objects, envelopes, ambiguous coordinate scales, invalid boxes, duplicate labels, missing predicate references, and point-only output. It does not silently swap x/y or create fake boxes.

## Operations and GPU notes

Standard logging records request id, model id, backend, elapsed time, first-load flag, parser/validation outcome, error type, and prompt length. It never records whole images by default. Prompts and raw model output can be sensitive, so normal logs contain only lengths; detailed content requires explicitly enabled debug logging.

Local Molmo2 serializes requests to protect a single GPU. Size models and GPU memory before deployment. If its CUDA stack is incompatible with the installation, use the OpenAI-compatible API backend instead.

## Tests and troubleshooting

The remaining lightweight test verifies TiPToP template interpolation and needs no GPU, checkpoint, or API key:

```bash
pixi run test
```

For the complete visual-grounding flow, use `pixi run demo -- --model-id openai-compatible --task-instruction "..."`. It runs `tests/test_ground.py`, including image preprocessing, TiPToP prompt rendering, a real multipart request, response validation, child-server cleanup, and bbox rendering. `pixi run test` uses `openai-compatible` with the task instruction `pick up the ipad` and writes the boxed image to `examples/result.png`; it requires `OPENAI_API_KEY`.

| Symptom | Resolution |
|---|---|
| `UNKNOWN_MODEL` | Use an id from `GET /v1/models` or add it to `configs/models.yaml`. |
| `UNSUPPORTED_IMAGE_TYPE` | Upload a real PNG/JPEG rather than a renamed text file. |
| `INVALID_MODEL_OUTPUT` | Require one direct JSON object in the model prompt; inspect protected debug logs. |
| `POINT_OUTPUT_NOT_SUPPORTED` | Ask for reliable boxes; do not expand a point into a guessed box. |
| `MODEL_NOT_READY` | Initialize submodules, download the checkpoint, install compatible optional dependencies, or use `openai-compatible`. |
| GPU OOM | Reduce concurrency/model size or use `openai-compatible`. |

## License and third-party notices

OmniGround source is Apache-2.0. Model weights and repositories under `third_party/` retain their own licenses, terms, and download requirements. Review the Molmo2 repository/license and API provider terms before deployment. Weights, credentials, and local environments must not be committed.
