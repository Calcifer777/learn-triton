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
text_obj = np.array(["This is a string"], dtype=np.object_)

input_tensors = [
  httpclient.InferInput("text", text_obj.shape, "BYTES"),
]
input_tensors[0].set_data_from_numpy(np.array(["sad"], dtype=np.object_))
results = client.infer(model_name="preprocess", inputs=input_tensors)

# %%

input_json = {
   "inputs":[
      {	
      "name": "text",
      "shape": [1,1],
      "datatype": "BYTES",
      "data": ["This is a string"]
      }
   ]
}
res = requests.post(
    'http://localhost:8000/v2/models/preprocess/versions/1/infer', 
    json=input_json
).json()
pprint(res)