# -*- coding: utf-8 -*-
# @Time    : 2021/9/12 下午8:57
# @Author  : WuDiDaBinGe
# @FileName: layer_different_lr.py
# @Software: PyCharm
def get_parameters_layers(model, init_lr, multiplier, classifier_lr, bert_layer=6):
    """
    Bert 微调的 层之间使用不同的学习率
    :param model:bert+分类器模型
    :param init_lr: bert初始学习率
    :param multiplier: 学习率衰减参数
    :param classifier_lr:分类器学习率
    :param bert_layer:bert encoder的层数
    :return: 不同层之间的参数以及对应的学习率字典
    """
    parameters = []
    lr = init_lr
    for layer in range(bert_layer, -1, -1):
        layer_params = {
            'params': [p for n, p in model.named_parameters() if f'encoder.layer.{layer}.' in n],
            'lr': lr
        }
        # 层数越低 学习率越小
        lr *= multiplier
        parameters.append(layer_params)
    # TODO: 需要模型名称修改
    classifier_params = {
        'params': [p for n, p in model.named_parameters() if 'convs.' in n or 'fc_cnn' in n or 'linear' in n],
        'lr': classifier_lr
    }
    parameters.append(classifier_params)
    return parameters