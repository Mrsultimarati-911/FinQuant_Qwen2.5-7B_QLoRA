import json
import os


def clean_lora_data(input_path, output_path):
    cleaned_count = 0
    with open(input_path, 'r', encoding='utf-8') as f_in, \
            open(output_path, 'w', encoding='utf-8') as f_out:

        for line in f_in:
            if not line.strip():
                continue

            data = json.loads(line)

            # 清洗大模型套话口癖
            original_output = data.get("output", "")

            # 清除各种可能的开头套话
            prefixes_to_remove = [
                "基于研报内容，", "基于研报内容 ", "根据研报内容，", "根据提供的研报内容，",
                "研报指出，", "在实证分析中，","总而言之","本文","本报告","本研报","综上所述"
            ]

            for prefix in prefixes_to_remove:
                if original_output.startswith(prefix):
                    # 截断套话，并确保首字母大写（如果是英文）或格式整洁
                    original_output = original_output[len(prefix):].lstrip()
                    break  # 去掉一个就行了

            # 更新清洗后的文本
            data["output"] = original_output

            # 写入新文件
            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
            cleaned_count += 1

    print(f"清洗完成！共处理了 {cleaned_count} 条数据，去除了'基于研报内容'等套话。")
    print(f"最终版训练集已保存至：{output_path}")


if __name__ == "__main__":
    input_file = r'原始数据存放位置'
    # 生成最终的纯净版数据
    final_output = r'清洗后数据保存位置'

    clean_lora_data(input_file, final_output)
