build_chinese_char.py

运行前修改参数：
raw_data_path：txt文件存放位置
tokenized_data_path: 可训练格式文本所在位置

read.py:

运行前修改第20行为需要验证的.npy文件


运行：
python build_chinese_char.py #每次运行记录运行时输出
python read.py #验证处理正确性