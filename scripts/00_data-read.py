import pandas as pd
import os

path = f'D:\\Python_Project_of_Study\\Ai_Study\\data\\raw_pdf'

total_file_list = os.listdir(path)

print(total_file_list)

i = 0

for file in total_file_list:
    file_path = os.path.join(path, file)
    file_list = os.listdir(file_path)
    print(len(file_list))
    i = i + len(file_list)

print('pdf文件数量为：',i)