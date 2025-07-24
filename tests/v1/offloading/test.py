from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
prompts = ["Once upon a time there was a cat. The cat was big. It was blue. And then suddenly it"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, seed=42)
ktc = KVTransferConfig(kv_connector="OffloadingConnector", kv_role="kv_both", kv_connector_extra_config={"spec_name": "SharedStorageOffloadingSpec"})
llm = LLM(model="facebook/opt-125m", enforce_eager=True, kv_transfer_config=ktc, enable_prefix_caching=False)
for _ in range(2):
    outputs = llm.generate(prompts, sampling_params)
    print(outputs[0].outputs[0].text)