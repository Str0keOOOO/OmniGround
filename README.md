# OmniGround

## 请求

| Item | Value |
|---|---|
| URLs | `POST /generate` 及其完全一致的别名 `POST /v1/generate` |
| Content-Type | `multipart/form-data` |
| 默认服务器地址 | `http://127.0.0.1:8011` |

| 字段 | 必填 | 类型 | 说明 |
|---|---:|---|---|
| image | 是 | PNG 或 JPEG 文件 | 一张非空图片；不支持的格式返回 415 |
| prompt | 是 | UTF-8 字符串 | 完整任务 prompt，OmniGround 原样传给模型 |
| temperature | 否 | number >= 0 | 采样温度，省略时使用模型配置中的默认值 |

模型在 OmniGround 服务启动时选择，`/generate` 请求不再需要 `model_id`；未指定启动参数时使用配置文件中的 `default_model`。

## 成功响应

每个成功的 /generate 响应均为 application/json，响应体直接为该对象：

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

## 快速开始

```bash
git clone https://tiptop-robot.readthedocs.io/en/latest/installation/
cd OmniGround

# Install dependencies
pixi run setup

# 下载需要的本地检查点（只选一个；笔记本优先 rynnbrain1.1-2b）
pixi run download-checkpoints -- rynnbrain1.1-2b
```

setup 会初始化已检出的子模块并检查基础依赖，不会安装所有可选的 VLM 依赖。运行 RynnBrain 1.1 或 RoboBrain 2.5 前，请额外安装一次本地推理依赖：

```bash
pip install -e .[embodied,download]
```

如果克隆需要 Molmo2 或 RoboBrain 2.5 源码，请使用：

```bash
git clone --recursive https://github.com/Str0keOOOO/OmniGround.git
```

或：

```bash
git submodule update --init --recursive
```

其余模块类比即可

可以运行demo，最终结果在`examples/`

```bash
pixi run demo -- --model-id rynnbrain1.1-2b --task-instruction "pick up the yellow ball"
```

运行后端（模型在服务启动时选择，未指定时使用 `configs/models.yaml` 的 `default_model`）

```bash
pixi run server -- --host 0.0.0.0 --port 8011 --model-id qwen3.7-plus
pixi run server -- --model-id molmo2-er
pixi run server -- --model-id rynnbrain1.1-2b
pixi run server -- --model-id robobrain2.5-4b
```

## 模型与后端

configs/models.yaml是model_id到适配器的唯一映射，不含任何凭证信息，包含api密钥，需要自己根据models_example.yaml复制一份。

| model_id | Backend | Mode | 用途 |
|---|---|---|---|
| molmo2-er | molmo2 | local | 本地 Molmo2 检查点 |
| rynnbrain1.1-2b | rynnbrain11 | local | RynnBrain 1.1 2B；笔记本优先选择 |
| rynnbrain1.1-9b | rynnbrain11 | local | RynnBrain 1.1 9B |
| robobrain2.5-4b | robobrain25 | local | RoboBrain 2.5 4B |
| robobrain2.5-8b-nv | robobrain25 | local | RoboBrain 2.5 8B NVIDIA 变体 |
| qwen3.7-plus | qwen3.7-plus | api | 通过 OpenAI API 协议调用的 Qwen3.7-Plus VLM |
|（通用 API） | openai_backend | api | 其他兼容 OpenAI API 的多模态 chat/completions 端点 |

RynnBrain 1.1 与 RoboBrain 2.5 都只在首次 `/generate` 请求时加载；HTTP 输入和输出与其他后端完全一致。适配器会在保留用户任务 prompt 的同时补充统一 JSON 输出约束，保证响应始终是 `bboxes` 与 `predicates`。RoboBrain 2.5 的官方推理仓库作为子模块固定在 `af98c932aac9ff715d70da177088d7bb95573ff7`。

本地内存有限时，先使用 `rynnbrain1.1-2b`。9B 与 8B 模型需要明显更多的显存/内存；没有合适 GPU 时可在 `configs/models.yaml` 把对应 `device` 设为 `cpu`，但推理会很慢。模型权重被下载到 `models/` 并保持 Git 忽略。
