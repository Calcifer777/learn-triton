# %%
import tritonclient.http as httpclient
import numpy as np

# %%
client = httpclient.InferenceServerClient(url="localhost:8000")

# %%
text = np.array([["This is a string"]], dtype="object")

input_tensor = httpclient.InferInput("text", text.shape, "BYTES")
input_tensor.set_data_from_numpy(text)
output_tensors = [
  httpclient.InferRequestedOutput(name="502"),
]
results = client.infer(
  model_name="e2e", 
  inputs=[input_tensor],
  outputs=output_tensors,
)
print(f"{results.as_numpy('502')=}")
print(f"{results.as_numpy('502').shape=}")
