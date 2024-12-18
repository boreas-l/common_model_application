# ------ coding : utf-8 ------
# @FileName     : train_main.py
# @Author       : lxc
# @Time         : 2024/10/11 11:07

import time

def train_epoch(model, train_dataloader, train_config, criterion, logger):
    start = time.time()
    for iter_id, data_batch in enumerate(train_dataloader):
        if iter_id >= train_config['max_iter']:
            break
        print(data_batch[0].shape, data_batch[1].shape)
        batch_size = data_batch[0].shape[0]
        train_config['global_step'] += 1
        input_image, input_label = data_batch
        input_image = input_image.to(train_config['device'])
        input_label = input_label.to(train_config['device'])
        model_output = model(input_image)
        loss = criterion(model_output, input_label)
        loss = loss / train_config['update_freq']
        loss.backward()
        if (iter_id + 1) % train_config['update_freq'] == 0:
            train_config['optimizer'].step()
            train_config['optimizer'].zero_grad()
            train_config['scheduler'].step()
        metric_value = train_config['metric'](model_output, input_label)
        print("metric value: ", metric_value)

