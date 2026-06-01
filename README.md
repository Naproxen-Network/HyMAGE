# HyMAGE

HyMAGE 是一个以 LLM 驱动的 Semantic Preferential Attachment (SPA) 为核心的合成超图生成项目。当前仓库聚焦 HyMAGE 生成流程，并保留结构指标分析与标注工具。

## Features

- 实体生成：支持人口画像、药物、生物反应物、电商实体和自定义实体。
- SPA 超图生成：结合偏好连接和 LLM 语义决策生成超边。
- 可配置候选池：`--candidate_pool_size` 控制每次边生成时展示给 LLM 的候选节点数量。
- 结构分析与标注：`HyMAGE-Benchmark/` 结构指标分析和人工/AI 辅助标注功能。

## Directory Layout

```text
HyMAGE/
├── HyMAGE-Main/              # SPA 超图生成核心
├── HyMAGE-EntityGen/         # 实体/节点生成工具
├── HyMAGE-Benchmark/         # 结构分析与标注 Web 工具
├── docs/                     # 项目说明与阅读笔记
├── requirements.txt          # Python 依赖
├── LICENSE                   # MIT License
├── .gitignore                # Git 忽略规则
└── README.md
```

## Installation

建议使用虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Common Workflows

### 1. 生成实体

```bash
cd HyMAGE-EntityGen
python generate_personas_standalone.py --n 1000 --output personas_1000.json
```

### 2. 生成超图

运行 LLM 相关脚本前，请在当前工作目录准备 `api-key.txt`，或按脚本参数传入 API key 文件。

```bash
cd HyMAGE-Main
python run.py \
  --entities ../HyMAGE-EntityGen/personas_1000.json \
  --config path/to/reference_hypergraph.txt \
  --output ./output \
  --model gpt-3.5-turbo \
  --pa_prob 0.7 \
  --candidate_pool_size 15 \
  --max_iterations 100
```

关键参数：

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--entities` | 实体 JSON 文件路径 | 必填 |
| `--config` | 参考超图文件，用于提取超边大小分布 | 必填 |
| `--output` | 输出目录 | `./output` |
| `--model` | LLM 模型名 | `gpt-3.5-turbo` |
| `--pa_prob` | 偏好连接采样概率 | `0.7` |
| `--candidate_pool_size` | 每次生成超边时展示给 LLM 的候选节点数量 | `15` |
| `--max_iterations` | 最大演化轮数 | `100` |
| `--semantic` | 语义预设或自定义网络描述 | `None` |
| `--base_url` | OpenAI-compatible API 地址 | `https://api.openai.com/v1` |

### 3. 运行结构分析与标注平台

```bash
cd HyMAGE-Benchmark
python app.py
```

默认访问地址：

```text
http://localhost:5001
```

## Notes

- LLM 相关脚本会读取 `api-key.txt` 或 `OPENAI_API_KEY`，真实密钥不应提交到 Git。
- Benchmark 平台会在运行时自动创建 `uploads/` 和 `annotations/`，这些目录已在 `.gitignore` 中忽略。
- 更多整理说明见 `docs/PROJECT_NOTES.md`。
- License: MIT.
