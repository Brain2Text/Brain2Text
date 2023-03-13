# Brain2Word: Text Generation from Brain Signals Based on Subject-Independent Brain-Computer Interface
This repository is the official implementation of Brain2Word: Text Generation from Brain Signals Based on Subject-Independent Brain-Computer Interface

## Requirements
All algorithm are developed in Python 3.8.

To install requirements:

```setup
pip install -r requirements.txt
```

## Training
To train the model for EEG under the subject-independent condition, run this command:
```train
python train_txt_SI.py --dataLoc './sampledata/' --logDir './logs/' --gpuNum 0 1 --batch_size 130
```
To fine-tune the model for EEG, run this command:
```train
python tuning_txt_SI.py --dataLoc './sampledata/' --logDir './logs/' --gpuNum 0 1 --batch_size 130 --pretrain_model './pretrain' --fewshot 5 --subNum 0
```
>📋 [the arguments of models](https://github.com/Brain2Word/Brain2Word/blob/main/models/readme.md)

## Evaluation
To evaluate the trained model for EEG under the subject-independent condition, run:
```eval
python eval_txt_SI.py --dataLoc './sampledata/' --logDir './logs/' --gpuNum 0 1 --batch_size 52 --fewshot 5 --subNum 0 --evalmodel '/tunemodel'
```


