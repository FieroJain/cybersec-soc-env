from huggingface_hub import HfApi
import os
api = HfApi(token=os.environ.get('HF_TOKEN'))
api.upload_file(
    path_or_fileobj='c:/Users/HP/Documents/openenv/cybersec_soc_env/server/app.py',
    path_in_repo='server/app.py',
    repo_id='Fieerawe/cybersec-soc-env',
    repo_type='space',
    commit_message='add missing endpoints'
)
print('Done!')