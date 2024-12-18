# ------ coding : utf-8 ------
# @FileName     : data_loader.py
# @Author       : lxc
# @Time         : 2024/8/19 9:39

"""创建数据集生成对象"""
from utils.dataset_utils import create_operators
import cv2
from torch.utils.data import Dataset, DataLoader
import os
import traceback
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print("BASE_DIR: ", BASE_DIR)


def build_dataloader(config, mode, task_cls='cls'):
    """
    构建dataloader
    :param config: 配置信息
    :param mode: 模式【Train/Eval/Test】
    :param task_cls: 任务类别-cls[分类]、seg[分割]等
    :return:
    """
    assert mode in ['Train', 'Eval', 'Test'], "数据集模式必须在 Train/Eval/Test 之间"
    config_dataset = config[mode]
    anno_txt_path = config_dataset['dataset']['cls_label_path']
    transforms = config_dataset['dataset']['transform_ops']
    dataset = ClsDataset(anno_txt_path, transforms)
    print("build dataset success...")
    # build dataloader
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config_dataset['sampler']['batch_size'],
        shuffle=config_dataset['sampler']['shuffle'],
        num_workers=config_dataset['loader']['num_workers'],
        drop_last=config_dataset['sampler']['drop_last'],
    )
    print("build data_loader success...")

    return data_loader


class ClsDataset(Dataset):
    def __init__(self, anno_txt_path, transforms=None):
        self.anno_txt_path = os.path.join(BASE_DIR, anno_txt_path[2:])
        print("anno txt path: ", self.anno_txt_path)
        if transforms is not None:
            self.transforms = create_operators(transforms)
        self.images = []
        self.labels = []
        # 读取数据集内容
        self._read_data()

    def _read_data(self):
        anno_content = open(self.anno_txt_path).read().strip().split('\n')
        for line in anno_content:
            img_path, label = line.split('\t')
            self.images.append(img_path)
            self.labels.append(int(label))

    def __getitem__(self, index):
        try:
            print("images: ", self.images[index], type(self.images[index]))
            img = cv2.imread(self.images[index])
            img = Image.fromarray(img)
            if self.transforms is not None:
                for op in self.transforms:
                    img = op(img)
            return img, self.labels[index]
        except Exception as e:
            print(f"Exception occured when parse line: {self.images[index]} with msg: {traceback.format_exc()}")
            return None, None

    def __len__(self):
        return len(self.images)

    @property
    def class_num(self):
        return len(set(self.labels))


if __name__ == '__main__':
    from config import parse_args, get_config

    args = parse_args()
    args.config = "../configs/ResNet18.yaml"
    config_content = get_config(args.config, args.override)
    print(f">>>config: {config_content}")
    data_loader = build_dataloader(config_content['DataLoader'], mode='Train')
    print(">>>data loader: ", data_loader)
    print("done")
