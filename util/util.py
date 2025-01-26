import json
import os

from config import Config


class Util(object):
    text_CLS = None
    image_CLS = None

    @staticmethod
    def CLS_init():
        Util.text_CLS = [0 for i in range(Config.dim)]
        Util.image_CLS = [0 for i in range(Config.dim)]

    @staticmethod
    def load_embedding(user_path):
        Util.CLS_init()
        text_li, image_li, mask_li = [Util.text_CLS], [Util.image_CLS], [0]
        with open(user_path, encoding='utf-8') as f:
            line = f.readline()
            while line:
                line = line.replace('\r\n', '')
                text, image = line.split('#')
                text = [float(i) for i in text.split(',')]
                image = [float(i) for i in image.split(',')]
                mask_li.append(0)
                text_li.append(text)
                image_li.append(image)
                if len(text_li) >= Config.behavior_size:
                    break
                line = f.readline()
        while len(text_li) < Config.behavior_size:
            mask_li.append(-100)
            text_li.append(Config.empty_text)
            image_li.append(Config.empty_image)
        return text_li, image_li, mask_li

    @staticmethod
    def save_data(train_li, test_li, save_path):
        print(save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        data_dict = {'train': train_li, 'test': test_li}
        start_index = 0
        while os.path.exists(f'{save_path}\\data_{start_index}.json'):
            start_index += 1
        dir_path = f'{save_path}\\data_{start_index}.json'
        with open(dir_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(data_dict, ensure_ascii=False))

    @staticmethod
    def save_model(model):
        save_path = f'{Config.save_path}\\pre_trained_model'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        model.save(f'{save_path}\\model_param')


    @staticmethod
    def save_err(err_li):
        save_path = f'{Config.save_path}err.json'
        if not os.path.exists(save_path):
            err_dict = dict()
        else:
            err_dict = dict(json.load(open(save_path)))
        for err_user in err_li:
            if err_user not in err_dict.keys():
                err_dict[err_user] = 1
            else:
                err_dict[err_user] += 1
        if 'err_num' in err_dict.keys():
            err_dict['err_num'] = err_dict['err_num'] + 1
            err_num = int(err_dict['err_num'])
        else:
            err_dict['err_num'] = 1
        with open(save_path, 'w+', encoding='utf-8') as f:
            f.write(json.dumps(err_dict, ensure_ascii=False))
        return err_dict['err_num']