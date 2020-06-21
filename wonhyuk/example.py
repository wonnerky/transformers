from pathlib import Path
from tokenizers import ByteLevelBPETokenizer

paths = [str(x) for x in Path("./data/eo/data/").glob("**/*.txt")]

tokenizer = ByteLevelBPETokenizer()

tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

tokenizer.save(".", "esperberto")

