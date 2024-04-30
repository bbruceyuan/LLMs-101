import pathlib
import sentencepiece as spm
from tokenizers import SentencePieceBPETokenizer
from transformers import PreTrainedTokenizerFast


def main():
    corpus_file = "../input/tokenizer/baidu_wiki_token.txt"
    # 这里学习 qwen，最终的
    special_tokens = ["<|endoftext|>"]
    tokenizer = SentencePieceBPETokenizer(
        add_prefix_space=False,
    )
    """
    files: Union[str, List[str]],
        vocab_size: int = 30000,
        min_frequency: int = 2,
        special_tokens: List[Union[str, AddedToken]] = ["<unk>"],
        limit_alphabet: int = 1000,
        initial_alphabet: List[str] = [],
        show_progress: bool = True,
    """
    tokenizer.train(
        corpus_file,
        vocab_size=30000,
        min_frequency=3,
        special_tokens=special_tokens,
    )
    tokenizer.save("mytoken")
    # tokenizer.save_pretrained("mytoken")
    # AttributeError: 'SentencePieceBPETokenizer' object has no attribute 'save_pretrained'

    # 将训练的tokenizer转换为PreTrainedTokenizerFast并保存
    # 转换是为了方便作为`AutoTokenizer`传到其他`huggingface`组件使用。

    slow_tokenizer = tokenizer
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=slow_tokenizer,
    )

    fast_tokenizer.save_pretrained("fast-mytoken")


if __name__ == "__main__":
    main()
