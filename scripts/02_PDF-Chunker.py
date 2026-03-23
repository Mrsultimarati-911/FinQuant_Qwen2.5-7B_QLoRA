import os
import json
import pandas as pd
from rapidocr_pdf import RapidOCRPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re

# 初始化 RapidOCR PDF 提取引擎
print("正在初始化本地 OCR 引擎...")
ocr_engine = RapidOCRPDF()


def extract_text_with_fallback_ocr(pdf_path):
    """
    智能复合解析引擎：
    基于 RapidOCR 强行读取加密或扫描版的券商研报 PDF。
    """
    text = ""
    print(f"  -> 尝试使用 OCR 解析: {pdf_path}")

    try:
        # RapidOCRPDF 默认提取所有页面
        # 常见返回格式为二维列表：[['0', '第一页内容', '0.9'], ['1', '第二页内容', '0.9']]
        res = ocr_engine(pdf_path)

        # 兼容处理：有些版本会返回包含解析时间的元组 (result_list, elapse_time)
        extract_data = res[0] if isinstance(res, tuple) else res

        pages_text = []
        if extract_data:
            for item in extract_data:
                # 提取标准返回格式中的第二项（文本内容）
                if isinstance(item, list) and len(item) >= 2:
                    pages_text.append(str(item[1]))
                elif isinstance(item, str):
                    pages_text.append(item)
                else:
                    pages_text.append(str(item))

        full_raw_text = "\n".join(pages_text)

        # 敏捷清洗：针对量化金工研报的特定脏数据进行清洗
        cleaned_text = re.sub(r'\n+', '\n', full_raw_text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # 压缩多余空格
        cleaned_text = re.sub(r'(\d+)\s*\n+\s*(\d+)', r'\1\2', cleaned_text)  # 修复排版导致的数字截断

        # 去掉常见无用后缀 (可以根据实际情况在数组里继续补充)
        for disclaimer in ["请务必阅读正文最后免责声明", "免责声明", "本报告仅供参考"]:
            cleaned_text = cleaned_text.replace(disclaimer, "")

        if cleaned_text:
            text = cleaned_text.strip()

    except Exception as e:
        print(f"  -> !!! OCR 解析彻底失败: {e}")

    return text


def chunk_with_metadata(index_csv_path, output_jsonl_path):
    # 1. 读取上一步生成的索引
    df = pd.read_csv(index_csv_path)

    # 2. 初始化 LangChain 的文本切分器
    # 针对因子推导逻辑较长，chunk_size 设为 1200
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n", "。", "；", " ", ""]
    )

    total_chunks = 0

    # 3. 追加写入模式 ('a')，方便随时中断和恢复
    with open(output_jsonl_path, 'a', encoding='utf-8') as f:

        # 提取PDF
        for index, row in df.iterrows():
            print(f"\n正在处理 [{index + 1}/{len(df)}]: {row['file_name']}")

            # 提取文本
            full_text = extract_text_with_fallback_ocr(row['file_path'])

            print(f"  -> 成功提取文本长度: {len(full_text)} 字符")

            if len(full_text) < 200:
                print("  -> 警告：文字极少 (<200字)，可能是纯图片且分辨率极低，跳过。")
                continue

            # 切分文本
            chunks = text_splitter.split_text(full_text)
            print(f"  -> 切分为 {len(chunks)} 个 Chunk")

            # 注入 Metadata 并写入 JSONL
            for i, chunk_text in enumerate(chunks):
                chunk_data = {
                    "chunk_id": f"{row['series_name']}_{row['series_number']}_chunk{i}",
                    "series_name": row['series_name'],
                    "series_number": row['series_number'],
                    "source_file": row['file_name'],
                    "text": chunk_text
                }
                f.write(json.dumps(chunk_data, ensure_ascii=False) + '\n')
                total_chunks += 1

    print(f"\nOCR 切块完成！目前共生成 {total_chunks} 个带有 Metadata 的文本块。")
    print(f"中间文件已保存至: {output_jsonl_path}")


if __name__ == "__main__":
    index_csv = r'保存文件索引的csv文件地址'
    output_chunks = r'分块后文件保存地址'

    chunk_with_metadata(index_csv, output_chunks)
