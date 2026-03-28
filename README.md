# VLLM 的部署流程

## 1. 配置虚拟环境
```bash
conda create -n vllm python=3.9 -y
```

## 2. 安装 VLLM
```bash
pip install vllm
pip show vllm
```

## 3. 下载大模型文件
通过 Modelscope 来下载大模型：
```bash
pip install modelscope
```
下载到本地路径：
```bash
modelscope download --model Qwen/Qwen3-8B --local_dir .\ChinaTravel-main\chinatravel\agent\tpc_agent\model
```

## 4. 部署 OpenAI API 服务
用 VLLM 拉取服务：
```bash
vllm serve
--/root/ ChinaTravel-main / chinatravel / agent / tpc_agent / model
--api-key abc123
--serverd-model-name Qwen/Qwen3-8B (模型的名称)
--max_model_len 4096（张量并行的数量）--port 7890
```

---

# 本项目的调用过程

**项目目标：** 本项目目标是利用中国旅游行程规划的智能代理类（ TPCAgent ），通过对酒店、景点、餐厅、交通等资源的智能排序和选择，以及对用户查询约束的验证，自动生成符合需求的旅游行程计划。

## 1) 定义 TPCLLM 类
定义一个 TPCLLM 类，负责初始化并调用本地 OpenAI 兼容大模型接口，为上层旅行规划代理提供稳定的文本生成能力 (`tpc_llm.py`)。

以下是关键参数的设置：
```python
class TPCLLM(AbstractLLM):
    def __init__(self, api_key=None, base_url=None):
        super().__init__()
        self.path = os.path.join(
            project_root_path, "local_llm", "deepseek_v3_tokenizer"
        )
        self.name = "tpc_llm"                          # 模型名称
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") # 调用的APIKEY
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "http://localhost:7890/v1")       # 模型的请求地址，其中7890是部署模型时的监听端口号
        self.llm = None
