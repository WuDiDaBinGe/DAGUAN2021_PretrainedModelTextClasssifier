# -*- coding: utf-8 -*-
# @Time    : 2021/9/14 下午4:56
# @Author  : WuDiDaBinGe
# @FileName: train_roberta_tokenizer.py
# @Software: PyCharm
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer

paths = [str(x) for x in Path("/dataset/daguan_data_class").glob("**/*.txt")]

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()
# Customize training
tokenizer.train(files=paths, vocab_size=30_400, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])
tokenizer.save_model("../dataset/roberta_pretrained", "roberta")
