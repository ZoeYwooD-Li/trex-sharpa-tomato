"""CPU smoke test of prepared data using the official collate, no VLM weights.

The test processor emits placeholder image tokens; this is not a model test.
"""
import argparse, json, sys, types
from pathlib import Path
import torch

class Batch(dict):
    __getattr__=dict.__getitem__

class TestProcessor:
    tokenizer=types.SimpleNamespace(pad_token_id=0)
    def apply_chat_template(self,*args,**kwargs):return 'test'
    def __call__(self,**kwargs):
        return Batch(input_ids=torch.tensor([[1,2,3]]),pixel_values=torch.zeros((len(kwargs['images']),4)),
                     image_grid_thw=torch.ones((len(kwargs['images']),3),dtype=torch.long))
    def image_processor(self,images,**kwargs):
        return Batch(pixel_values=torch.zeros((len(images),4)),image_grid_thw=torch.ones((len(images),3),dtype=torch.long))

def validate(upstream,prepared,output):
    for package in ('qwen_vla','utils'):
        module=types.ModuleType(package);module.__path__=[str(Path(upstream).resolve()/package)]
        sys.modules[package]=module
    from sharpa_dataset import SharpaDataset
    config=types.SimpleNamespace(lerobot_root=str(Path(prepared).resolve()),action_dim=62,action_chunk=16,
        use_robot_state=0,image_size=[384,288],use_flare=1,n_flare_steps=8,flare_frame_stride=4,
        use_tactile_vec=1,use_tactile_deform=1,use_tactile_vqvae=1,vqvae_window=16)
    dataset=SharpaDataset(config,TestProcessor(),types.SimpleNamespace(print=print))
    val=dataset.create_val_split(.05,42)
    first=dataset[0];last=val[len(val)-1];batch=dataset.collate_fn([first,last])
    expected={'noisy_actions':(2,16,62),'norm_actions':(2,16,62),'target':(2,16,62),
      'tactile_f6_history':(2,16,10,6),'tactile_f6s':(2,10,6),'tactile_deforms':(2,10,1,240,240)}
    for key,shape in expected.items():
        assert tuple(batch[key].shape)==shape
        assert torch.isfinite(batch[key]).all()
    assert last['observation.images.head_is_pad'][1:].all()
    # DataLoader workers must not receive open VideoCapture / Parquet handles.
    state=dataset.__getstate__()
    assert all(not state[key] for key in ('_numeric','_parquets','_groups','_videos'))
    result={'train_frames':len(dataset),'val_frames':len(val),'train_episodes':len(dataset.episodes),
      'val_episodes':len(val.episodes),'batch_shapes':{key:list(shape) for key,shape in expected.items()},
      'all_checked_tensors_finite':True,'future_tail_padding':True,'worker_handles_cleared':True,
      'geometry':'collection-project nominal FR3 URDF, link8 flange',
      'processor':'test placeholder; real Qwen processor/model forward not run'}
    Path(output).write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--upstream',required=True);p.add_argument('--prepared',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();validate(a.upstream,a.prepared,a.output)
