# ------ coding : utf-8 ------
# @FileName     : temp.py
# @Author       : lxc
# @Time         : 2024/10/8 16:38

import os

txt_file_path = "./datasets/cat_dog_cls/valid.txt"
txt_content = open(txt_file_path).read().strip().split('\n')
print(txt_content)
with open(txt_file_path, 'w') as f:
    for line in txt_content:
        img_path, label = line.split(' ')
        write_line = f"{img_path}\t{label}"
        f.write(write_line+'\n')
print("done")
