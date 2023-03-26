
import json
from pathlib import Path
import numpy as np
from transformers import DistilBertTokenizer
try:
  import triton_python_backend_utils as pb_utils
except ImportError:
  import sys
  sys.path.append(Path(__file__))
  import triton_python_backend_utils as pb_utils



class TritonPythonModel:

  def initialize(self, args):
    cfg = json.loads(args["model_config"])
    self.feature_extractor_path = (
      Path(args["model_repository"]) / 
      args["model_version"] / 
      "tokenizer"
    )
    self.feature_extractor = DistilBertTokenizer.from_pretrained(
      self.feature_extractor_path
    )
    # Get outputs configuration and convert to Triton dtypes
    self.out_input_ids_cfg = pb_utils.get_output_config_by_name(cfg, "input_ids")
    self.out_input_ids_dtype = pb_utils.triton_string_to_numpy(
      self.out_input_ids_cfg['data_type']
    )
    self.out_att_mask_cfg = pb_utils.get_output_config_by_name(cfg, "attention_mask")
    self.out_att_mask_dtype = pb_utils.triton_string_to_numpy(
      self.out_att_mask_cfg['data_type']
    )

  def execute(self, requests):
    """
    c_python_backend_utils.Tensor
      as_numpy
      from_dlpack
      is_cpu
      name
      shape
      to_dlpack
      triton_dtype
    """
    responses = []
    for request in requests:
      inputs_raw = pb_utils.get_input_tensor_by_name(request, "text")
      inputs = inputs_raw.as_numpy().astype(str).squeeze(0).tolist()
      features = self.feature_extractor.batch_encode_plus(inputs)
      inference_response = pb_utils.InferenceResponse(output_tensors=[
          pb_utils.Tensor("input_ids", np.array(features['input_ids'])),
          pb_utils.Tensor("attention_mask", np.array(features['attention_mask'])),
      ])
      responses.append(inference_response)
    return responses