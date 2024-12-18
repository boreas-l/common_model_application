# ------ coding : utf-8 ------
# @FileName     : build_metric.py
# @Author       : lxc
# @Time         : 2024/10/11 15:42

class TopK:
    def __init__(self, topk):
        self.topk = topk
        self.maxk = max(self.topk)

    def __call__(self, x, label):
        batch_size = x.size(0)
        values, indices = x.topk(self.maxk, 1, True, True)
        indices = indices.t()
        correct = indices.eq(label.view(1, -1).expand_as(indices))
        res = []
        for k in self.topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res


metric_dict = {
    "TopK": TopK,
}
def build_metric(config):
    """
    构建指标计算函数
    :param config:
    :return:
    """
    metric_name = config.pop("name")
    metric_func = metric_dict[metric_name]
    metric = metric_func(**config)

    return metric
