# 参考 https://github.com/DLLXW/baby-llama2-chinese/blob/main/dataset.py
import pathlib
import json
import numpy as np
from loguru import logger
from transformers import Qwen2Tokenizer

tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen-tokenizer")


BASE_PATH = pathlib.Path("../input")

# 用于保存 tokenizer 训练的数据
BASE_PRETRAIN_DATA_DIR = BASE_PATH / "pretrain_data"

if not BASE_PRETRAIN_DATA_DIR.exists():
    BASE_PRETRAIN_DATA_DIR.mkdir(parents=True)


def process_wiki():
    """
    JSONL 格式
       {'completion': '昭通机场（ZPZT）是位于中国云南昭通的民用机场，始建于1935年，1960年3月开通往返航班“昆明－昭通”，原来属军民合用机场。1986年机场停止使用。1991年11月扩建，于1994年2月恢复通航。是西南地区「文明机场」，通航城市昆明。 机场占地1957亩，飞行区等级为4C，有一条跑道，长2720米，宽48米，可供波音737及以下机型起降。机坪面积6600平方米，停机位2个，航站楼面积1900平方米。位于城东6公里处，民航路与金鹰大道交叉处。\n航点\n客服电话\n昭通机场客服电话：0870-2830004',
    'source': 'wikipedia.zh2307'}
    """
    wiki_path = (
        BASE_PATH / "wikipedia-cn-20230720-filtered/wikipedia-cn-20230720-filtered.json"
    )
    wiki_pretrain_path = BASE_PRETRAIN_DATA_DIR / "wiki_pretrain.npy"

    with wiki_path.open("r") as f:
        wiki_data = json.load(f)
    logger.info("load wiki data 成功")
    # 把每一行都转换成 tokenize 之后的形式，存放在 numpy 中；

    doc_ids = []
    for item in wiki_data:
        content = item.get("completion", "") + "<|endoftext|>"
        text_ids = tokenizer.encode(content)
        if len(text_ids) > 5:
            # 这样是把所有的 token 都方案 id 的形式放到一个 array 中, 用 special_token 进行分割
            doc_ids += text_ids

    # 把所有的 doc_ids 转换成 numpy 的形式
    # 因为 qwen的词表比 chatGLM 大，所以没法用一个 unit16 表示
    doc_ids_arr = np.array(doc_ids, np.uint32)
    logger.info(f"wiki token 数量 is: {len(doc_ids_arr)}")

    # 把 token 数量保存到文件中
    with (BASE_PRETRAIN_DATA_DIR / "wiki_token_count.txt").open("w") as f:
        f.write(str(len(doc_ids_arr)))
    # 保存 numpy 的训练数据
    np.save(wiki_pretrain_path, doc_ids_arr)
    logger.info("Wiki 处理完成")


def main():
    process_wiki()


if __name__ == "__main__":
    main()
