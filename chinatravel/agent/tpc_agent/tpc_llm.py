
import os
import sys

from transformers import AutoTokenizer

project_root_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root_path not in sys.path:
    sys.path.append(project_root_path)
if os.path.dirname(project_root_path) not in sys.path:
    sys.path.append(os.path.dirname(project_root_path))

from chinatravel.agent.llms import AbstractLLM
import time
from datetime import datetime
from transformers import AutoTokenizer
from openai import OpenAI
from json_repair import repair_json
class TPCLLM(AbstractLLM):
    def __init__(self, api_key=None, base_url=None):
        super().__init__()
        self.path = os.path.join(
            project_root_path, "local_llm", "deepseek_v3_tokenizer"
        )
        self.name = "tpc_llm"
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "http://localhost:7890/v1")
        self.llm = None
        self.tokenizer = AutoTokenizer.from_pretrained(self.path)
        self._initialize_llm()
        self.last_failure_time = None
        self.failure_count = 0

    def _initialize_llm(self):
        """初始化LLM客户端"""
        if not self.api_key:
            print(
                "Failed to initialize LLM client: Missing API key. "
                "Set the OPENAI_API_KEY environment variable or pass api_key explicitly."
            )
            self.llm = None
            return
        try:
            self.llm = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key
            )
        except Exception as e:
            print(f"Failed to initialize LLM client: {str(e)}")
            self.llm = None

    def _should_retry(self):
        """判断是否应该重试"""
        if self.last_failure_time is None:
            return True
        elapsed = (datetime.now() - self.last_failure_time).total_seconds()
        if self.failure_count > 3 and elapsed < 60:  # 如果失败超过3次且在1分钟内
            return False
        return True

    def _send_request(self, messages, kwargs, max_retries=3):
        """发送请求到Deepseek API"""
        if not self._should_retry():
            return '{"error": "Service temporarily unavailable due to frequent failures"}'

        if self.llm is None:
            self._initialize_llm()
            if self.llm is None:
                return '{"error": "LLM client not initialized"}'

        for attempt in range(max_retries):
            try:
                # Token计算
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                input_tokens = self.tokenizer(text)["input_ids"]
                self.input_token_count += len(input_tokens)
                self.input_token_maxx = max(self.input_token_maxx, len(input_tokens))

                # 请求设置
                request_kwargs = {
                    "model": "Qwen3-8B",
                    "max_tokens": 9216,
                    "temperature": 0,
                    "top_p": 0.00000001,
                    "timeout": 30,
                    **kwargs
                }

                response = self.llm.chat.completions.create(
                    messages=messages,
                    **request_kwargs
                )

                # 成功时重置失败计数器
                self.failure_count = 0

                res_str = response.choices[0].message.content
                output_tokens = self.tokenizer(res_str)["input_ids"]
                self.output_token_count += len(output_tokens)

                return res_str.strip()

            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                self.last_failure_time = datetime.now()
                self.failure_count += 1

                if attempt == max_retries - 1:
                    return '{"error": "Request failed after multiple attempts"}'
                time.sleep(1 + attempt)  # 指数退避

    def _get_response(self, messages, one_line=False, json_mode=False):
        """获取LLM响应"""
        kwargs = {}

        if one_line:
            kwargs["stop"] = ["\n"]
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            res_str = self._send_request(messages, kwargs)
            if json_mode:
                try:
                    res_str = repair_json(res_str, ensure_ascii=False)
                except Exception as e:
                    print(f"JSON repair failed: {str(e)}")
                    res_str = '{"error": "Invalid JSON response"}'
            return res_str
        except Exception as e:
            print(f"Error in _get_response: {str(e)}")
            return '{"error": "Request failed, please try again."}'

    def check_health(self):
        """检查服务健康状态"""
        test_msg = [{"role": "user", "content": "ping"}]
        response = self._send_request(test_msg, {"max_tokens": 5})
        return not response.startswith('{"error":')



if __name__ == "__main__":
    # model = Mistral()
    model = TPCLLM("abc123")
    print(model([{"role": "user", "content": "我们3人，从成都出发，到上海旅行3天，要求如下：不希望游览淀山湖风景区 和 田子坊石库门"}], one_line=False))
