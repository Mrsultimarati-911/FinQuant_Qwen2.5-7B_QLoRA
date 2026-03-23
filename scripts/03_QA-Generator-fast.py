import os
import json
import asyncio
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

# 初始化异步 API 客户端
API_KEY = "sk-f904f7212faf4b88a3b83799797b84fa"
client = AsyncOpenAI(
    api_key=API_KEY,
    base_url="https://api.deepseek.com"
)

# 【核心架构参数】：并发控制器
# 设置为 30，意味着同时有 30 个请求在路上。这能在不触发 API 频率限制的情况下达到极速。
CONCURRENCY_LIMIT = 30
semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

system_prompt = """
你是一位顶级的 A 股量化金融研究员与 AI 数据合成专家。
你的任务是阅读一段来自券商金工系列的研报切片文本，并将其转化为用于大模型指令微调（Instruction Tuning）的高质量问答对。

请遵循以下极度严苛的标准：
1. 寻找核心量化逻辑：提取单因子测试方法、收益预测模型、风险模型优化、特征工程或机器学习算法的具体应用。
2. 坚决过滤废话（宁缺毋滥）：如果文本主要是“分析师声明”、“投资评级说明”、“免责条款”、“机构联系方式”、“纯目录”或没有实质性实证推导的过渡段落，请务必返回空列表 []。绝对不要给废话生成问答！
3. 深度生成：生成 1 到 2 个极具专业深度的问答对。输出的解答需要严谨、详细，尽可能还原研报中的数学推导或因子构建思路。

输出必须是严格的 JSON 数组格式：
[
  {
    "instruction": "作为一个量化研究员，请解释...", 
    "input": "", 
    "output": "基于研报内容的详细、严谨的解答。"
  }
]
不要输出任何 markdown 标记（如 ```json），直接输出纯 JSON 数组。
"""


async def process_chunk(chunk, max_retries=3):
    """
    带重试机制的异步网络请求节点
    """
    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"请基于以下量化研报文本生成微调问答对：\n\n{chunk['text']}"}
                    ],
                    temperature=0.3,
                    response_format={"type": "json_object"}
                )

                result_str = response.choices[0].message.content.strip()
                result_json = json.loads(result_str)

                if isinstance(result_json, list):
                    return result_json
                elif isinstance(result_json, dict):
                    for val in result_json.values():
                        if isinstance(val, list):
                            return val
                    return [result_json]
                return []

            except Exception as e:
                # 遇到错误（如并发限流 HTTP 429），进行指数退避重试 (等待 1s, 2s, 4s...)
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    # 重试 3 次仍然失败，打印日志并跳过，防止阻塞全局进度
                    print(f"\n[警告] Chunk {chunk['chunk_id']} 处理失败: {e}")
                    return []


async def main():
    input_jsonl = r'D:\Python_Project_of_Study\Ai_Study\data\processed\corpus_chunks_ocr.jsonl'
    output_jsonl = r'D:\Python_Project_of_Study\Ai_Study\data\outputs\instruction_tuning_quant.jsonl'

    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)

    # 1. 一次性加载全部 Chunk 数据到内存
    chunks_data = []
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                chunks_data.append(json.loads(line))

    print(f"成功加载 {len(chunks_data)} 个文本块。准备启动异步高并发引擎...")

    # 2. 创建所有异步任务
    tasks = [process_chunk(chunk) for chunk in chunks_data]
    total_qa_pairs = 0

    # 3. 动态接收结果并实时写入硬盘
    # 使用 'w' 模式会覆盖之前的测试数据，保证最终文件的纯净度
    with open(output_jsonl, 'w', encoding='utf-8') as f_out:
        # as_completed 谁先完成就先处理谁，绝不排队死等
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="⚡ 极速提炼 LoRA 语料"):
            qa_list = await coro

            if qa_list:
                for qa in qa_list:
                    f_out.write(json.dumps(qa, ensure_ascii=False) + '\n')
                    total_qa_pairs += 1

            # 实时将内存缓冲区的数据刷入硬盘，防止意外断电丢失
            f_out.flush()

    print(f"\n🎉 全量跑通！成功提炼出 {total_qa_pairs} 条量化指令微调数据。")
    print(f"你的 LoRA 黄金数据集已妥善保存在：{output_jsonl}")


if __name__ == "__main__":
    # Windows 环境下运行 asyncio 常见标准写法
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())