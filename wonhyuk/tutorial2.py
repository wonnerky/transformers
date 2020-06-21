from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from torch.utils.data.dataset import Dataset
from pathlib import Path

class EsperantoDataset(Dataset):
    def __init__(self, evaluate: bool = false):
        tokenizer = ByteLevelBPETokenizer(
            "./esperberto-vocab.json",
            './esperberto-merges.txt',
        )
        tokenizer._tokenizer.post_processor = BertProcessing(
            ("</s>", tokenizer.token_to_id("</s>")),
            ("<s>", tokenizer.token_to_id("<s>")),
        )
        tokenizer.enable_truncation(max_length=512)

        self.examples = []

        src_files = Path("./")