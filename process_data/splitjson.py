import json
import random


# 数据样式
# {
#   "title": "5-12",
#   "content": "2281 21552 4873 3042 274 9601 19121 5540 19934 15318 24445 19226 ， 10935 16542 15785 8147 9963 26765 20458 22241 28099 17281 21771 3644 12520 27063 1451 24053 8808 12790",
# }

# 数据内存太大，不行
def split_json0(filename):
    json_string = []
    # 读取大规模json文件，按照行读
    with open(filename, 'r', encoding='utf-8') as f:
        try:
            while True:
                line_data = f.readline()
                if line_data:
                    jsonfile = json.loads(line_data)  # 把读取的一行的json数据加载出来
                    json_string.append(jsonfile)
                else:
                    break
        except Exception as e:
            print(e)
            f.close()

    print(type(json_string))
    print(len(json_string))
    random.shuffle(json_string)

    val_list = []
    for i in range(10000000):
        val_list.append(json_string[i])
        if 1 % 10000 == 0:
            print("已处理 %s 条数据" % i)

    with open('new.json', 'w') as f:
        json.dump(val_list, f)

    print("split done!")


def split_json(inputfile):
    totalnum = 0
    num = 0
    f_out = open('./newdata.json', 'w')
    with open(inputfile, 'r', encoding='utf-8') as f:
        for doc in f:
            num = num + 1
            if num < 500000:
                doc_dict = eval(doc)
                # son.dumps()把dict降级为字符串
                str = json.dumps(doc_dict) + '\n'
                f_out.write(str)
                totalnum = totalnum + 1
            if 2000000 < num < 2500000:
                doc_dict = eval(doc)
                # son.dumps()把dict降级为字符串
                str = json.dumps(doc_dict) + '\n'
                f_out.write(str)
                totalnum = totalnum + 1
            if 4000000 < num < 4500000:
                doc_dict = eval(doc)
                # son.dumps()把dict降级为字符串
                str = json.dumps(doc_dict) + '\n'
                f_out.write(str)
                totalnum = totalnum + 1
            if num > 6000000:
                doc_dict = eval(doc)
                # son.dumps()把dict降级为字符串
                str = json.dumps(doc_dict) + '\n'
                f_out.write(str)
                totalnum = totalnum + 1

            if totalnum % 10000 == 0:
                print("已处理 %s 条数据" % (num))
            if totalnum > 1500000:
                break

    f_out.close()


def split_4_dataset(input_file, out_file, sub_nums=4):
    total_num = 0
    num = 0
    file_list = []
    for i in range(sub_nums):
        file_list.append(open(out_file + f'/subset_{i}.json'))
    with open(input_file, 'r', encoding='utf-8'  ) as f:
        for doc in f:
            total_num += 1
            doc_dict = eval(doc)
            doc_vob = doc_dict['title'].split(" ") + doc_dict['content'].split(" ")
            if len(doc_vob) > 15:
                flag = num % sub_nums
                if flag == 0:
                    pass
                elif flag == 1:
                    pass
                elif flag == 2:
                    pass
                elif flag == 3:
                    pass
                num += 1


if __name__ == '__main__':
    file = '/home/wsj/dataset/2021达观杯/datagrand_2021_unlabeled_data.json'
    file_test = 'process_data/test.json'
    split_json(file)
