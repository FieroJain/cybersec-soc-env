from huggingface_hub import HfApi
import os
api = HfApi(token=os.environ.get('HF_TOKEN'))
models = list(api.list_models(author='Fieerawe'))
if models:
    print("Models found:")
    for m in models:
        print(" -", m.id)
else:
    print("No models on Hub yet")