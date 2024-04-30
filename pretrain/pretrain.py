from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.nn as nn
import torch
from loguru import logger
from dataclasses import dataclass
from transformers import Qwen2ForCausalLM, Qwen2Config
from transformers import Trainer, TrainingArguments


@dataclass
class Args:
    max_length: int


class MyDataset(Dataset):
    """加载数据"""

    def __init__(self, filenames, args):
        self.data = []
        self.index_map = {}
        self.token_size, self.smp_size = 0, 0
        data_lst = []
        for _, filename in enumerate(filenames):
            data = np.load(filename)
            data_lst.append(data)
        data = np.concatenate(data_lst)
        data = data[: args.max_length * int(len(data) / args.max_length)]
        # np.random.shuffle(data)
        self.data = data.reshape(-1, args.max_length)

        self.token_size = self.data.shape[0] * args.max_length
        self.sample_size = self.data.shape[0]
        logger.info(f"token_size: {self.token_size}, smp_size: {self.sample_size}")

    def __len__(self):
        return self.sample_size

    def __getitem__(self, index: int):
        sample = self.data[index]
        X = np.array(sample[:-1]).astype(np.int64)
        Y = np.array(sample[1:]).astype(np.int64)
        # return torch.from_numpy(X), torch.from_numpy(Y)
        return {
            "input_ids": torch.from_numpy(X),
            "labels": torch.from_numpy(Y),
        }


args = Args(max_length=1024)

# 先用 24M 的模型测试一下
config = Qwen2Config.from_json_file("../config/qwen_0.12B.config")
model = Qwen2ForCausalLM(config)
print(model.num_parameters())

train_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=1,
    fp16=True,
    save_strategy="epoch",
)

trainer = Trainer(
    model,
    train_args,
    train_dataset=MyDataset(["../input/pretrain_data/wiki_pretrain.npy"], args=args),
)

trainer.train()
