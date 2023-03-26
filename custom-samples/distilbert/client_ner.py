# %%
from pprint import pprint
import json
import requests
import tritonclient.http as httpclient
import numpy as np
from tritonclient.utils import np_to_triton_dtype

# %%
client = httpclient.InferenceServerClient(url="localhost:8001")

# %%
input_ids = np.array([[101, 2023, 2003, 1037, 5164, 102]])
attention_mask = np.array([[101, 2023, 2003, 1037, 5164, 102]])

input_tensors = [
  httpclient.InferInput("input_ids", input_ids.shape, np_to_triton_dtype(input_ids.dtype)),
  httpclient.InferInput("slice", attention_mask.shape, np_to_triton_dtype(attention_mask.dtype)),

]
input_tensors[0].set_data_from_numpy(input_ids)
input_tensors[1].set_data_from_numpy(attention_mask)

output_tensors = [ httpclient.InferRequestedOutput(name="502"), ]

results = client.infer(
  model_name="ner", 
  inputs=input_tensors,
  # outputs=output_tensors,
)
print(f"{results.as_numpy('502')=}")
print(f"{results.as_numpy('502').shape=}")
# %%
