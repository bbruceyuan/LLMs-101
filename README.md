# LLMs-101
计划自己动手训练一个 miniLLM，模型参数暂定 0.2B 。打算把 `data process`，`tokenizer`，`pretrain`，`sft`，`rlhf`都自己做一遍。深入了解预训练的细节。

我们都知道掌握 LLM 的最好的办法是动手实践，但是囿于业务压力，我们很多时候没有精力和机会真正自己训练一个 LLM。因此 LLM 算法工程师可能就变成了「prompt 调优工程师」，工作内容只有三件事：看业务数据，写 Prompt 和 ICL 样例，利用少量数据 SFT。这些事情做久了就会觉得无聊，那么我们一起动手从头开始训练一个 miniLLM 吧，欢迎围观！

## 数据集
|数据集|介绍|数据量（DiskSize）|
|--|--|--|
|[维基百科](https://huggingface.co/datasets/pleisto/wikipedia-cn-20230720-filtered)|huggingface 中的开源数据集|~1GB|
|[百度百科](https://pan.baidu.com/s/1jIpCHnWLTNYabftavo3DVw?pwd=bwvb), 提取码: bwvb|百科百科数据|16GB|
|[悟道 200GB](https://data.baai.ac.cn/details/WuDaoCorporaText)|悟道开源 200G（完整 2T）|206GB|
|[天工150B](https://huggingface.co/datasets/Skywork/SkyPile-150B)|天工高质量 150B |620GB|


## DataProcess
第一期采用的成熟的开源数据集，维基百科，百度百科，悟道和天工数据集。因此数据预处理部分会相关比较简单，会持续更新。

## Tokenizer
由于 BPE 算法实在是太占用内存了，第一期采用 `SentencePiece` Random Select 一些数据跑通 Train Tokenizer 的过程。 

## Pretrain

## SFT

## RLHF

## 其他
1. 受限于算力问题，本项目仅仅用于学习作用，可能最终 0.2B 效果并不令人满意。
2. 我会在[MEMOS](https://memos.bbruceyuan.com)记录平常想到的一些东西, 如果国内想访问更快，可以点击[这里](http://43.153.192.214/)。Cloudflare 速度真的是负优化。


## Reference
1. https://github.com/Tongjilibo/build_MiniLLM_from_scratch/tree/master
2. https://github.com/DLLXW/baby-llama2-chinese