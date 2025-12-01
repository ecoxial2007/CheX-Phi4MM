
需要预先准备的数据：
分别放置在
data\MIMIC-VQA-CoT\annotations\eval , hard_uniform, uniform ……

https://drive.google.com/drive/folders/1p-GlXypee4UvnwMmfjn3KVuAh9SBuKz_


0. SFT 预训练权重
预先放置在 ./checkpoints/restore_MIMIC_Think_phi4_ep1_lr6e-5

1 ./utils/test_REXKG/BiomedNLP
从HF下载并重命名
（https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext）

2. ReXKG 
./utils/test_REXKG/result1/run_entity
./utils/test_REXKG/result1/run_relation
从Google Drive 下载并分别放置在
https://drive.google.com/drive/folders/1sNZDT8bI97AJwBGq_O5Y8oMrTjmA0Fpe

一些可能会出现的问题
chat_template.json 删掉即可
