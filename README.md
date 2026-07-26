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
pixi run pip install -e ".[embodied,download]"
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

可以运行 demo。每次运行会按模型保存在独立目录 `examples/results/<model-id>/<北京时间_耗时>/`，目录内包含标注图片、接口返回的 JSON 和详细耗时日志，例如
`20260726-145812-123456_BJT_gen-11.684s/20260726-145812-123456_BJT_gen-11.684s.{png,json,log}`。日志会区分服务启动/模型加载、`/generate` 请求、响应解析和结果保存等阶段。这些生成结果已被 Git 忽略。
运行本地模型前，请将 `<gpu-id>` 替换为要使用的物理 GPU 编号；该 GPU 会在进程内显示为 `cuda:0`。

```bash
CUDA_VISIBLE_DEVICES=<gpu-id> pixi run demo -- --model-id rynnbrain1.1-2b --task-instruction "pick up the ipad"
```

运行后端（模型在服务启动时选择，未指定时使用 `configs/models.yaml` 的 `default_model`）

```bash
pixi run server -- --host 0.0.0.0 --port 8011 --model-id qwen3.7-plus
CUDA_VISIBLE_DEVICES=<gpu-id> pixi run server -- --model-id molmo2-er
CUDA_VISIBLE_DEVICES=<gpu-id> pixi run server -- --model-id rynnbrain1.1-2b
CUDA_VISIBLE_DEVICES=<gpu-id> pixi run server -- --model-id robobrain2.5-4b
CUDA_VISIBLE_DEVICES=<gpu-id> pixi run server -- --model-id qwen3.5-9b
CUDA_VISIBLE_DEVICES=<gpu-id> pixi run server -- --model-id qwen3.5-4b
CUDA_VISIBLE_DEVICES=<gpu-id> pixi run server -- --model-id qwen3.5-2b
CUDA_VISIBLE_DEVICES=<gpu-id> pixi run server -- --model-id qwen3.5-0.8b
```

## 模型与后端

`configs/models.yaml` 是 `model_id` 到适配器的唯一映射，不应包含凭证；请从 `configs/models.example.yaml` 复制后按部署环境填写。

| model_id | Backend | Mode | 用途 |
|---|---|---|---|
| molmo2-er（无法满足需求） | molmo2 | local | 本地 Molmo2 检查点 |
| rynnbrain1.1-2b（无法满足需求） | rynnbrain | local | RynnBrain 1.1 2B |
| rynnbrain1.1-9b（无法满足需求） | rynnbrain | local | RynnBrain 1.1 9B |
| robobrain2.5-4b（无法满足需求） | robobrain | local | RoboBrain 2.5 4B |
| robobrain2.5-8b-nv（受限暂时没有下载） | robobrain | local | RoboBrain 2.5 8B NVIDIA 变体 |
| qwen3.5-9b（推荐） | qwen35 | local | Qwen/Qwen3.5-9B 本地多模态模型 |
| qwen3.5-4b | qwen35 | local | Qwen/Qwen3.5-4B 本地多模态模型 |
| qwen3.5-2b | qwen35 | local | Qwen/Qwen3.5-2B 本地多模态模型 |
| qwen3.5-0.8b | qwen35 | local | Qwen/Qwen3.5-0.8B 本地多模态模型 |
| qwen3.7-plus | openai | api | 通过 OpenAI 兼容协议调用的 Qwen3.7-Plus VLM |
|（通用 API） | openai | api | 其他兼容 OpenAI API 的多模态 chat/completions 端点 |
