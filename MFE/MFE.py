import json
import os
import zipfile
import numpy as np
from datetime import datetime

from tqdm import tqdm
from transformers import TFBertModel, BertTokenizer, ViTFeatureExtractor, TFAutoModel
from PIL import Image

# 加载 BERT 模型和分词器
model_name = "bert-base-uncased"  # 或其他预训练模型，如 "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = TFBertModel.from_pretrained(model_name)

# 模型名称
model_name = "google/vit-base-patch16-224"  # 也可以选择其他预训练模型
# 加载特征提取器和模型
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
vit_model = TFAutoModel.from_pretrained(model_name)

def time_convert(time_str):
    time_format = "%a %b %d %H:%M:%S %z %Y"
    dt = datetime.strptime(time_str, time_format)
    time_stamp = dt.timestamp()
    return time_stamp


def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def batch_spilt(data_list, batch_size):
    result_li, temp_li = [], []
    for data in data_list:
        if len(temp_li) < batch_size:
            temp_li.append(data)
        else:
            result_li.append(temp_li)
            temp_li = []
    if len(temp_li) > 0:
        result_li.append(temp_li)
    return result_li

def text_feature(text_li):
    # Tokenizer 对文本进行编码
    inputs = tokenizer(
        text_li,
        max_length=128,  # 最大序列长度
        padding="max_length",  # 填充到最大长度
        truncation=True,  # 截断超长序列
        return_tensors="tf"  # 返回 TensorFlow 格式的张量
    )
    # 使用模型提取特征
    outputs = bert_model(inputs)
    # 获取最后一层隐藏状态 (batch_size, seq_len, hidden_size)
    last_hidden_states = outputs.last_hidden_state[:, 0, :].numpy()
    return last_hidden_states

def load_images(image_path_li):
    return_mat = None
    for image_path in image_path_li:
        if image_path is not None:
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                image = Image.new("RGB", [224, 224])
        else:
            image = Image.new("RGB", [224, 224])
        try:
            inputs = feature_extractor(images=image, return_tensors="tf")
        except Exception as e:
            print(image_path)
            print(e)
            exit(1)
        if return_mat is None:
            return_mat = inputs["pixel_values"]
        else:
            return_mat = np.concatenate([return_mat, inputs["pixel_values"]], axis=0)
    return return_mat

def image_feature(image_li):
    outputs = vit_model(image_li)
    # 输出特征，通常是 CLS token
    cls_token = outputs.last_hidden_state[:, 0, :].numpy()
    return cls_token

def save_feature(text_li, image_li, dir_name):
    write_path = f'{save_path}\\{dir_name}.txt'
    if not os.path.exists(write_path):
        with open(write_path, 'w+', encoding='utf-8') as f:
            pass
    with open(write_path, 'a+', encoding='utf-8') as f:
        for index in range(len(text_li)):
            text = str(list(text_li[index]))[1:-1]
            image = str(list(image_li[index]))[1:-1]
            f.write(f"{text}#{image}\n")

def unzip_file(zip_path, extract_to=None):
    # 如果未指定解压目录，默认为 ZIP 文件所在的目录
    if extract_to is None:
        extract_to = os.path.dirname(zip_path)

    # 创建解压目录（如果不存在的话）
    os.makedirs(extract_to, exist_ok=True)

    # 打开 ZIP 文件并解压
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)


# 数据集地址
zip_path = 'C:\\Users\\21103\\Desktop\\dataset\\SD_MVAE_dataset\\up\\normal'
dataset_path = 'C:\\Users\\21103\\Desktop\\dataset\\SD_MVAE_dataset\\temp1'
save_path = 'C:\\Users\\21103\\Desktop\\dataset\\SD_MVAE_dataset\\dataset1\\non'
batch_size = 128


# 获取多模态用户行为序列
for dir_index, dir_name in enumerate(os.listdir(zip_path)):
    if '.DS_Store' in dir_name:
        continue
    print(dir_index, dir_name)
    zip_file_path = os.path.join(zip_path, dir_name)
    dir_name = dir_name.split('.zip')[0]
    if os.path.exists(f'{save_path}\\{dir_name}.txt'):
        print(dir_index, dir_name, 'exist')
        continue
    dir_path = os.path.join(dataset_path, dir_name)
    print(f'unzip: {dir_index, dir_name}')
    unzip_file(zip_file_path, dir_path)
    files_path = os.path.join(dir_path, 'events')
    behavior_li = []
    pqbr = tqdm(total=len(os.listdir(files_path)))
    for file_index, file_name in enumerate(os.listdir(files_path)):
        if '.DS_Store' in file_name:
            continue
        pqbr.desc = f'load behavior {dir_index}-->{file_index}'
        pqbr.update(1)

        file_path = os.path.join(files_path, file_name)
        file_json = read_json(file_path)
        time_stamp = time_convert(file_json['created_at'])
        content = file_json['text_raw']
        file_id = file_json['id']
        image_path = f'{dir_path}\\images\\{file_id}.jpg'
        if not os.path.exists(image_path):
            image_path = f'{dir_path}\\images\\{file_id}.png'
            if not os.path.exists(image_path):
                image_path = None
        behavior_li.append([time_stamp, content, image_path])
    behavior_li.sort(key=lambda x: -x[0])
    # behavior_li = behavior_li[:1200]

    # 历史行为数据序列切分
    seq_li = batch_spilt(behavior_li, batch_size)
    pqbr = tqdm(total=len(seq_li))
    for behavior_id, behavior_li in enumerate(seq_li):
        pqbr.desc = f'process {dir_index}-->{behavior_id}'
        pqbr.update(1)

        try:
            text_li = [i[1] for i in behavior_li]
            tf = text_feature(text_li)

            image_path_li = [i[2] for i in behavior_li]
            image_li = load_images(image_path_li)
            image_f = image_feature(image_li)

            save_feature(tf, image_f, dir_name)
        except:
            continue