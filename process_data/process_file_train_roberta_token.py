# -*- coding: utf-8 -*-
# @Time    : 2021/9/14 下午3:49
# @Author  : WuDiDaBinGe
# @FileName: process_file_train_roberta_token.py
# @Software: PyCharm
from tqdm import tqdm


def process_all_data_to_many_file(input_file_path, out_dir):
    total_rows = 0
    file_count = 0
    text_data = []
    with open(input_file_path, 'r', encoding='utf-8') as f:
        # TODO:修改数据总数
        for index, doc in enumerate(tqdm(f, total=1500001)):
            doc_dict = eval(doc)
            total_rows += 1
            doc_vob = doc_dict['title'] + " " + doc_dict['content']
            text_data.append(doc_vob)
            if len(text_data) == 10000:
                f_out = open(out_dir + f'/text_{file_count}.txt', 'w')
                f_out.write('\n'.join(text_data))
                text_data = []
                file_count += 1
    f_out = open(out_dir + f'/text_{file_count}.txt', 'w')
    f_out.write('\n'.join(text_data))
    f_out.close()
    print(file_count)
    return total_rows


if __name__ == '__main__':
    process_all_data_to_many_file('/dataset/daguan_data_class/newdata.json', out_dir='/home/yxb/Documents')
