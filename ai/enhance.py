import os
import json
import sys
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
# INSERT_YOUR_CODE
import requests

import dotenv
import argparse
from tqdm import tqdm

import langchain_core.exceptions
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from structure import Structure

def load_env():
    env_candidates = [
        os.path.join(os.getcwd(), ".env"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"),
    ]
    loaded = set()
    for env_path in env_candidates:
        if env_path in loaded:
            continue
        if os.path.exists(env_path):
            dotenv.load_dotenv(env_path)
            loaded.add(env_path)


load_env()
template = open("template.txt", "r").read()
system = open("system.txt", "r").read()

REQUIRED_AI_FIELDS = ("tldr", "motivation", "method", "result", "conclusion")
DEFAULT_AI_FIELDS = {
    "tldr": "Summary generation failed",
    "motivation": "Motivation analysis unavailable",
    "method": "Method extraction failed",
    "result": "Result analysis unavailable",
    "conclusion": "Conclusion extraction failed",
}

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="jsonline data file")
    parser.add_argument("--max_workers", type=int, default=1, help="Maximum number of parallel workers")
    return parser.parse_args()


def _normalize_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for part in content:
            if isinstance(part, str):
                chunks.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return "\n".join(chunks)
    return str(content)


def _extract_json_candidate(text: str) -> Optional[str]:
    if not text:
        return None

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        return fenced.group(1)

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return None


def _fallback_request(content: str, language: str, model_name: str) -> Optional[str]:
    """Bypass langchain and call API directly to avoid proxy tool injection."""
    prompt = (
        system.replace("{language}", language)
        + "\n\n"
        + template.replace("{content}", content).replace("{language}", language)
        + "\n\nReturn ONLY a valid JSON object with keys: "
        + ", ".join(REQUIRED_AI_FIELDS)
        + "."
    )
    resp = requests.post(
        os.environ["OPENAI_BASE_URL"] + "/chat/completions",
        headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"], "Content-Type": "application/json"},
        json={
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "tools": [],
            "tool_choice": "none",
        },
        timeout=60,
    )
    if resp.status_code == 200:
        return resp.json()["choices"][0]["message"].get("content", "")
    return None


def _coerce_ai_fields(raw_data: Dict[str, Any]) -> Dict[str, str]:
    cleaned: Dict[str, str] = {}
    for field in REQUIRED_AI_FIELDS:
        if field in raw_data:
            value = raw_data[field]
            cleaned[field] = value if isinstance(value, str) else str(value)
    return cleaned

def process_single_item(structured_chain, model_name: str, item: Dict, language: str) -> Dict:
    def is_sensitive(content: str) -> bool:
        """
        调用 spam.dw-dengwei.workers.dev 接口检测内容是否包含敏感词。
        返回 True 表示触发敏感词，False 表示未触发。
        """
        if not content:
            return False
        try:
            resp = requests.post(
                "https://spam.dw-dengwei.workers.dev",
                json={"text": content},
                timeout=5
            )
            if resp.status_code == 200:
                result = resp.json()
                # 约定接口返回 {"sensitive": true/false, ...}
                return result.get("sensitive", True)
            else:
                # 接口异常时默认放行，避免整批数据被误杀
                print(f"Sensitive check failed with status {resp.status_code}", file=sys.stderr)
                return False
        except Exception as e:
            print(f"Sensitive check error: {e}", file=sys.stderr)
            # 网络/服务异常时默认放行，避免整批数据被误杀
            return False

    def check_github_code(content: str) -> Dict:
        """提取并验证 GitHub 链接"""
        code_info = {}

        # 1. 优先匹配 github.com/owner/repo 格式
        github_pattern = r"https?://github\.com/([a-zA-Z0-9-_]+)/([a-zA-Z0-9-_\.]+)"
        match = re.search(github_pattern, content)
        
        if match:
            owner, repo = match.groups()
            # 清理 repo 名称，去掉可能的 .git 后缀或末尾的标点
            repo = repo.rstrip(".git").rstrip(".,)")
            
            full_url = f"https://github.com/{owner}/{repo}"
            code_info["code_url"] = full_url
            
            # 尝试调用 GitHub API 获取信息
            github_token = os.environ.get("TOKEN_GITHUB")
            headers = {"Accept": "application/vnd.github.v3+json"}
            if github_token:
                headers["Authorization"] = f"token {github_token}"
            
            try:
                api_url = f"https://api.github.com/repos/{owner}/{repo}"
                resp = requests.get(api_url, headers=headers, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    code_info["code_stars"] = data.get("stargazers_count", 0)
                    code_info["code_last_update"] = data.get("pushed_at", "")[:10]
            except Exception:
                # API 调用失败不影响主流程
                pass
            return code_info

        # 2. 如果没有 github.com，尝试匹配 github.io
        github_io_pattern = r"https?://[a-zA-Z0-9-_]+\.github\.io(?:/[a-zA-Z0-9-_\.]+)*"
        match_io = re.search(github_io_pattern, content)
        
        if match_io:
            url = match_io.group(0)
            # 清理末尾标点
            url = url.rstrip(".,)")
            code_info["code_url"] = url
            # github.io 不进行 star 和 update 判断
                
        return code_info

    # 检查 summary 字段
    if is_sensitive(item.get("summary", "")):
        return None

    # 检测代码可用性
    code_info = check_github_code(item.get("summary", ""))
    if code_info:
        item.update(code_info)

    """处理单个数据项"""
    ai_data: Dict[str, str] = {}

    try:
        if structured_chain is not None:
            response: Structure = structured_chain.invoke({
                "language": language,
                "content": item['summary']
            })
            if hasattr(response, "model_dump"):
                ai_data = _coerce_ai_fields(response.model_dump())
            elif isinstance(response, dict):
                ai_data = _coerce_ai_fields(response)
    except langchain_core.exceptions.OutputParserException as e:
        # 尝试从错误信息中提取 JSON 字符串并修复
        error_msg = str(e)
        partial_data: Dict[str, Any] = {}
        
        if "Function Structure arguments:" in error_msg:
            try:
                # 提取 JSON 字符串
                json_str = error_msg.split("Function Structure arguments:", 1)[1].strip().split('are not valid JSON')[0].strip()
                # 预处理 LaTeX 数学符号 - 使用四个反斜杠来确保正确转义
                json_str = json_str.replace('\\', '\\\\')
                # 尝试解析修复后的 JSON
                partial_data = json.loads(json_str)
            except Exception as json_e:
                print(f"Failed to parse JSON for {item.get('id', 'unknown')}: {json_e}", file=sys.stderr)
        
        ai_data = _coerce_ai_fields(partial_data)
        print(f"Using partial AI data for {item.get('id', 'unknown')}: {list(partial_data.keys())}", file=sys.stderr)
    except Exception as e:
        print(f"Structured generation failed for {item.get('id', 'unknown')}: {e}", file=sys.stderr)

    if not ai_data:
        try:
            raw_text = _fallback_request(item['summary'], language, model_name)
            if raw_text:
                candidate = _extract_json_candidate(raw_text)
                if candidate:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        ai_data = _coerce_ai_fields(parsed)
            if not ai_data:
                print(f"Fallback JSON parse failed for {item.get('id', 'unknown')}", file=sys.stderr)
        except Exception as e:
            print(f"Fallback generation failed for {item.get('id', 'unknown')}: {e}", file=sys.stderr)

    item['AI'] = {**DEFAULT_AI_FIELDS, **ai_data}

    # 检查 AI 生成的所有字段
    for v in item.get("AI", {}).values():
        if is_sensitive(str(v)):
            return None
    return item

def process_all_items(data: List[Dict], model_name: str, language: str, max_workers: int) -> List[Dict]:
    """并行处理所有数据项"""
    llm = ChatOpenAI(model=model_name)
    structured_method = os.environ.get("STRUCTURED_OUTPUT_METHOD", "auto").strip()
    structured_llm = None
    try:
        if structured_method and structured_method.lower() != "auto":
            structured_llm = llm.with_structured_output(Structure, method=structured_method)
        else:
            structured_llm = llm.with_structured_output(Structure)
    except Exception as e:
        print(f"Failed to initialize structured output ({structured_method or 'auto'}): {e}", file=sys.stderr)

    print('Connect to:', model_name, file=sys.stderr)
    print(f"Structured method: {structured_method}", file=sys.stderr)

    # Merge system prompt into human message to avoid proxy tool injection on SystemMessage
    combined_template = system + "\n\n" + template

    prompt_template = ChatPromptTemplate.from_messages([
        HumanMessagePromptTemplate.from_template(template=combined_template)
    ])

    structured_chain = prompt_template | structured_llm if structured_llm is not None else None
    
    # 使用线程池并行处理
    processed_data = [None] * len(data)  # 预分配结果列表
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_idx = {
            executor.submit(process_single_item, structured_chain, model_name, item, language): idx
            for idx, item in enumerate(data)
        }
        
        # 使用tqdm显示进度
        for future in tqdm(
            as_completed(future_to_idx),
            total=len(data),
            desc="Processing items"
        ):
            idx = future_to_idx[future]
            try:
                result = future.result()
                processed_data[idx] = result
            except Exception as e:
                print(f"Item at index {idx} generated an exception: {e}", file=sys.stderr)
                # Add default AI fields to ensure consistency
                processed_data[idx] = data[idx]
                processed_data[idx]['AI'] = DEFAULT_AI_FIELDS.copy()
    
    return processed_data

def main():
    args = parse_args()
    model_name = os.environ.get("MODEL_NAME", 'deepseek-chat')
    language = os.environ.get("LANGUAGE", 'Chinese')

    # 检查并删除目标文件
    target_file = args.data.replace('.jsonl', f'_AI_enhanced_{language}.jsonl')
    if os.path.exists(target_file):
        os.remove(target_file)
        print(f'Removed existing file: {target_file}', file=sys.stderr)

    # 读取数据
    data = []
    with open(args.data, "r") as f:
        for line in f:
            data.append(json.loads(line))

    # 去重
    seen_ids = set()
    unique_data = []
    for item in data:
        if item['id'] not in seen_ids:
            seen_ids.add(item['id'])
            unique_data.append(item)

    data = unique_data
    print('Open:', args.data, file=sys.stderr)
    
    # 并行处理所有数据
    processed_data = process_all_items(
        data,
        model_name,
        language,
        args.max_workers
    )

    kept_items = [item for item in processed_data if item is not None]
    if kept_items and all(item.get("AI") == DEFAULT_AI_FIELDS for item in kept_items):
        raise RuntimeError(
            "AI enhancement failed for all items. "
            "Please verify MODEL_NAME / OPENAI_BASE_URL / API compatibility."
        )
    
    # 保存结果
    with open(target_file, "w") as f:
        for item in processed_data:
            if item is not None:
                f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    main()
