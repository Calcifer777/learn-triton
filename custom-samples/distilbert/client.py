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
text = np.array([["This is a string"]], dtype="object")

input_tensor = httpclient.InferInput("text", text.shape, "BYTES")
input_tensor.set_data_from_numpy(text)
output_tensors = [
  httpclient.InferRequestedOutput(name="input_ids"),
  httpclient.InferRequestedOutput(name="attention_mask"),
]
results = client.infer(
  model_name="preprocess", 
  inputs=[input_tensor],
  outputs=output_tensors,
)
print(f"{results.as_numpy('input_ids')=}")
print(f"{results.as_numpy('attention_mask')=}")

# %%

input_json = {
   "inputs":[
      {	
      "name": "text",
      "shape": [1,1],
      "datatype": "BYTES",
      "data": [["This is a string"]]
      }
   ]
}
res = requests.post(
    'http://localhost:8000/v2/models/preprocess/versions/1/infer', 
    json=input_json
).json()
pprint(res)

"""
{'model_name': 'preprocess',
 'model_version': '1',
 'outputs': [{'data': [101, 2023, 2003, 1037, 5164, 102],
              'datatype': 'INT64',
              'name': 'input_ids',
              'shape': [1, 6]},
             {'data': [1, 1, 1, 1, 1, 1],
              'datatype': 'INT64',
              'name': 'attention_mask',
              'shape': [1, 6]}]}
"""