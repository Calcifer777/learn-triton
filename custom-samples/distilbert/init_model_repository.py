# %%
from pathlib import Path
from typing import Optional

import numpy as np
from transformers import (
  DistilBertTokenizer,
  DistilBertForTokenClassification
)
import torch

# %%
PATH_BASE = Path(__file__).parent
PATH_OUT_TOKENIZER = PATH_BASE / "models" / "preprocess" / "1"
PATH_OUT_NER = PATH_BASE / "models" / "ner" / "1"
MODEL_ID = "distilbert-base-uncased"

# %%
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_ID)

# %%
tokenizer.save_pretrained(PATH_OUT_TOKENIZER / "tokenizer")

# %%
inputs = np.array(['Classify query: test']).tolist()
features = tokenizer.batch_encode_plus(inputs, return_tensors="pt")
features

# %%
model = DistilBertForTokenClassification.from_pretrained(MODEL_ID, torchscript=True, num_labels=13)

model.eval()
# %%

traced_model = torch.jit.trace(model, [features["input_ids"], features["attention_mask"]])
torch.jit.save(traced_model, PATH_OUT_NER / "model.pt")

# %%
