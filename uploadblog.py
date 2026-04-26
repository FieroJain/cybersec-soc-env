from huggingface_hub import HfApi
import os
api = HfApi(token=os.environ.get('HF_TOKEN'))
api.upload_file(
    path_or_fileobj='c:/Users/HP/Documents/openenv/BLOG.md',
    path_in_repo='Blog.md',
    repo_id='Fieerawe/cybersec-soc-env',
    repo_type='space',
    commit_message='add blog post'
)
print('Done!')
print('https://huggingface.co/spaces/Fieerawe/cybersec-soc-env/blob/main/Blog.md')