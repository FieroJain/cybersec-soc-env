from huggingface_hub import HfApi
import os
api = HfApi(token=os.environ.get('HF_TOKEN'))
for png in ['loss_curve.png', 'topology_finding.png', 'training_curve.png']:
    api.upload_file(
        path_or_fileobj=f'c:/Users/HP/Documents/openenv/{png}',
        path_in_repo=png,
        repo_id='Fieerawe/cybersec-soc-env',
        repo_type='space',
        commit_message=f'add {png}'
    )
    print(f'Uploaded {png}')
print('All PNGs uploaded!')