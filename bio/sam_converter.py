import csv
import sys
import os

# 定义一个函数来尝试读取文件
def try_read_file(file_path, encoding):
    try:
        with open(file_path, 'r', encoding=encoding) as file:
            # 尝试读取前几行
            for _ in range(5):
                pass
                # print(file.readline())
        print(f"使用 {encoding} 编码读取成功。\n")
        return True
    except Exception as e:
        # print(f"尝试使用 {encoding} 编码读取时发生错误：{e}\n")
        return False

def detect_file_encoding(file_path):
    # 尝试使用不同的编码读取文件的前几行，常见的除了utf-8外，还有latin1, gb2312等
    encodings = ['utf-8', 'latin1', 'gb2312', 'ascii', 'gbk']
    for encoding in encodings:
        if try_read_file(file_path, encoding):
            return encoding
    return 'utf-8'

def convert_sam_to_csv(input_path):
    # 构建输出文件路径：更改文件扩展名为.csv
    base_name = os.path.splitext(input_path)[0]
    output_path = f"{base_name}.csv"
    input_encoding = detect_file_encoding(input_path)


    with open(input_path, 'r', encoding=input_encoding) as sam_file, open(output_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # 写入CSV的表头
        csv_writer.writerow(['QNAME', 'FLAG', 'RNAME', 'POS', 'MAPQ', 'CIGAR', 'RNEXT', 'PNEXT', 'TLEN', 'SEQ', 'QUAL'])
        
        for line in sam_file:
            if line.startswith('@'):
                continue  # 跳过头部信息行
            fields = line.split('\t')
            # 只提取必需的11个字段
            essential_fields = fields[:11]
            csv_writer.writerow(essential_fields)

    print("转换完成，输出文件：", output_path)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("使用方法: python convert.py <input_path>")
    else:
        input_path = sys.argv[1]
        convert_sam_to_csv(input_path)