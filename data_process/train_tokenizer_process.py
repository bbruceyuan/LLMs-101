import pathlib
import json
import random
from loguru import logger


BASE_PATH = pathlib.Path("../input")

# 用于保存 tokenizer 训练的数据
BASE_TOKEN_DATA_PATH = BASE_PATH / "tokenizer2"

# 设置数据集已经它对应的 path和用于 train tokenizer 的路径
# 百度百科数据集
baidu_wiki_path = BASE_PATH / "baidubake/563w_baidubaike.json"
baidu_wiki_token_path = BASE_TOKEN_DATA_PATH / "baidu_wiki_token.txt"
tiangong_dir = BASE_PATH / "SkyPile-150B/data"
tiangong_token_path = BASE_TOKEN_DATA_PATH / "tiangong_token.txt"

wiki_path = (
    BASE_PATH / "wikipedia-cn-20230720-filtered/wikipedia-cn-20230720-filtered.json"
)
wiki_token_path = BASE_TOKEN_DATA_PATH / "wiki_token.txt"

wudao_path_dir = BASE_PATH / "WuDaoCorpus"
wudao_token_path = BASE_TOKEN_DATA_PATH / "wudao_token.txt"

"""
每一行一个 sentence
"""


# 设置 train 的比例
def use_this_line_or_file(prob=0.05) -> bool:
    rand = random.random()
    if rand < prob:
        return True
    else:
        return False


def baidu_wiki_process(
    item,
    baidu_wiki_token_path_writer,
):
    content_list = []
    summary = item.get("summary", "")
    if summary and len(summary) > 50:
        content_list.append(summary.strip())
    sections = item.get("sections", [])
    for sec in sections:
        content = sec.get("content", "")
        if content and len(content.split("\n")) == 1 and len(content) > 50:
            # 只用只有一行的内容
            content_list.append(content.strip())
    if len(content_list) > 0:
        baidu_wiki_token_path_writer.write("\n".join(content_list))
        baidu_wiki_token_path_writer.write("\n")


def process_baidu_wiki():
    """
    百度百科的数据格式是 JSONL
    """
    writer = baidu_wiki_token_path.open("a")
    with baidu_wiki_path.open("r") as f:
        for line in f.readlines():
            data = line
            if not use_this_line_or_file():
                continue
            try:
                line_data = json.loads(data)
                baidu_wiki_process(
                    line_data,
                    writer,
                )
            except Exception:
                continue
    logger.info("百度 Wiki 处理完成")


def process_tiangong():
    # 先随机选一个文件，然后再用其中的几行
    wf = tiangong_token_path.open("a")
    for path in tiangong_dir.iterdir():
        if not use_this_line_or_file(0.2):
            continue
        with path.open("r") as f:
            for line in f.readlines():
                if not use_this_line_or_file():
                    continue
                try:
                    line_data = json.loads(line)

                    content = line_data.get("text", "")
                    if len(content) < 50:
                        continue
                    for true_line in content.split("\n"):
                        if len(true_line) < 50:
                            continue
                        wf.write(true_line + "\n")
                except Exception:
                    continue
    wf.close()
    logger.info("tiangong 处理完成")


def process_wiki():
    """
       {'completion': '昭通机场（ZPZT）是位于中国云南昭通的民用机场，始建于1935年，1960年3月开通往返航班“昆明－昭通”，原来属军民合用机场。1986年机场停止使用。1991年11月扩建，于1994年2月恢复通航。是西南地区「文明机场」，通航城市昆明。 机场占地1957亩，飞行区等级为4C，有一条跑道，长2720米，宽48米，可供波音737及以下机型起降。机坪面积6600平方米，停机位2个，航站楼面积1900平方米。位于城东6公里处，民航路与金鹰大道交叉处。\n航点\n客服电话\n昭通机场客服电话：0870-2830004',
    'source': 'wikipedia.zh2307'}
    """
    with wiki_path.open("r") as f:
        wiki_data = json.load(f)

    with wiki_token_path.open("a") as f:
        for item in wiki_data:
            # 使用其中的某一部分
            if not use_this_line_or_file():
                continue
            try:
                content_arr = item.get("completion", "").split("\n")
                for cur_content in content_arr:
                    if len(cur_content) > 50:
                        f.write(cur_content + "\n")
            except Exception:
                continue
    logger.info("Wiki 处理完成")


def process_wudao():
    wf = wudao_token_path.open("a")
    for path in wudao_path_dir.iterdir():
        if not use_this_line_or_file(0.2):
            # 80% 丢弃
            continue
        with path.open("r") as f:
            try:
                data = json.load(f)
            except Exception:
                continue
        for item in data:
            if not use_this_line_or_file():
                continue
            title = item.get("title", "")
            if len(title) > 50:
                wf.write(title + "\n")
            content = item.get("content", "")
            for line in content.split("\n"):
                if len(line) > 50:
                    wf.write(line + "\n")
    wf.close()
    logger.info("wudao 处理完成")


def main():
    # process_baidu_wiki()
    # process_tiangong()
    process_wiki()
    process_wudao()


if __name__ == "__main__":
    main()
