# MUDformer

## Dataset

Due to the dataset magnitude exceeding the GitHub maximum limit, we uploaded the constructed spammer detection benchmark dataset to other professional websites.

**Weibo-25:** [GitHub](https://github.com/yzhouli/Spammer-Detection-Dataset), [Baidu](https://aistudio.baidu.com/datasetdetail/312909), and [Kaggle](https://www.kaggle.com/datasets/yangzhou32/spammer-detection-v1)

**MisDerdect:** The original dataset is [here](https://github.com/yzhouli/SocialNet/tree/master/Weibo), and the dataset supplemented via the official API is [here](https://www.kaggle.com/datasets/yangzhou32/spammer-detection-v2).

## Model and Train

This site maintains the source code and training code for the MUDformer model. Among other things, the dataset processing code for model training is found in the MFE folder. The standards for the dataset are shown in the Datasets/train.json file. Run.py is the training how to and Config.py is the configuration centre for training.
