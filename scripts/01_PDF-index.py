import os
import re
import pandas as pd


def build_corpus_index(base_path):
    data = []
    # 遍历你的券商研报文件夹
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)

        if not os.path.isdir(folder_path):
            continue

        for file_name in os.listdir(folder_path):
            if file_name.endswith('.pdf'):
                # 用正则提取文件名中的数字序号（如 "华泰人工智能系列1.pdf" 提取出 1）
                # 假设你的文件名里包含数字序号，如果没有，可以设为默认值 0
                match = re.search(r'\d+', file_name)
                series_num = int(match.group()) if match else 0

                file_path = os.path.join(folder_path, file_name)

                data.append({
                    "series_name": folder_name,  # 系列名称 (如 huatai_Ai)
                    "series_number": series_num,  # 系列序号 (用于排序和追踪演进)
                    "file_name": file_name,
                    "file_path": file_path
                })

    df = pd.DataFrame(data)
    # 按照系列名称和序号进行排序，确保逻辑连贯性
    df = df.sort_values(by=["series_name", "series_number"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    # 请替换为你本地的真实路径
    pdf_base_path = r'D:\Python_Project_of_Study\Ai_Study\data\raw_pdf'
    output_csv = r'D:\Python_Project_of_Study\Ai_Study\data\processed\corpus_index.csv'

    print("正在扫描并构建券商金工系列研报索引...")
    index_df = build_corpus_index(pdf_base_path)

    # 保存为 CSV
    index_df.to_csv(output_csv, index=False, encoding='utf-8-sig')

    print(f"索引构建完成！共收录 {len(index_df)} 份 PDF。")
    print("前 5 条索引预览：")
    print(index_df.head())