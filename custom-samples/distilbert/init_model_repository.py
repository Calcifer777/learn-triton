# %%
from pathlib import Path
from typing import Optional

import numpy as np
from transformers import DistilBertTokenizer
import torch

# %%
PATH_BASE = Path(__file__).parent
PATH_OUT_TOKENIZER = PATH_BASE / "models" / "preprocess" / "1"


# %%
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# %%
tokenizer.save_pretrained(PATH_OUT_TOKENIZER / "tokenizer")

# %%
inputs = np.array(['Classify query: test']).tolist()
features = tokenizer.batch_encode_plus(inputs)
np.array(features["input_ids"])
np.array(features["attention_mask"])

# %%

[x for x in dir(tokenizer) if not x.startswith("_")]