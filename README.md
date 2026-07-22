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

# Download pretrained weights,such as molmo2-er which can skip
pixi run download-checkpoints -- molmo2-er
```

setup 会初始化已检出的子模块并检查基础依赖，不会安装所有可选的 VLM 依赖。如果克隆需要Molmo2源码，请使用：

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
pixi run demo -- --model-id molmo2-er --task-instruction "pick up the yellow ball"
```

运行后端（模型在服务启动时选择，未指定时使用 `configs/models.yaml` 的 `default_model`）

```bash
pixi run server -- --host 0.0.0.0 --port 8011 --model-id qwen3.7-plus
pixi run server -- --model-id molmo2-er
```

## 模型与后端

configs/models.yaml是model_id到适配器的唯一映射，不含任何凭证信息，包含api密钥，需要自己根据models_example.yaml复制一份。

| model_id | Backend | Mode | 用途 |
|---|---|---|---|
| molmo2-er | molmo2 | local | 本地 Molmo2 检查点 |
| qwen3.7-plus | qwen3.7-plus | api | 通过 OpenAI API 协议调用的 Qwen3.7-Plus VLM |
|（通用 API） | openai_backend | api | 其他兼容 OpenAI API 的多模态 chat/completions 端点 |
