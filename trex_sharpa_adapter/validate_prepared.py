"""CPU smoke test of prepared data using the official collate.

Pass --processor-path to test the real Qwen processor without model weights.
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

def validate(upstream,prepared,output,processor_path=None):
    if processor_path:
        sys.path.insert(0,str(Path(upstream).resolve()))
        from transformers import AutoProcessor
        from qwen_vla.sharpa_dataset import SharpaDataset
        processor=AutoProcessor.from_pretrained(processor_path,trust_remote_code=True,
                                                local_files_only=True)
    else:
        for package in ('qwen_vla','utils'):
            module=types.ModuleType(package);module.__path__=[str(Path(upstream).resolve()/package)]
            sys.modules[package]=module
        from sharpa_dataset import SharpaDataset
        processor=TestProcessor()
    config=types.SimpleNamespace(lerobot_root=str(Path(prepared).resolve()),action_dim=62,action_chunk=16,
        use_robot_state=0,image_size=[384,288],use_flare=1,n_flare_steps=8,flare_frame_stride=4,
        use_tactile_vec=1,use_tactile_deform=1,use_tactile_vqvae=1,vqvae_window=16)
    dataset=SharpaDataset(config,processor,types.SimpleNamespace(print=print))
    val=dataset.create_val_split(.05,42)
    first=dataset[0];last=val[len(val)-1];batch=dataset.collate_fn([first,last])
    expected={'noisy_actions':(2,16,62),'norm_actions':(2,16,62),'target':(2,16,62),
      'tactile_f6_history':(2,16,10,6),'tactile_f6s':(2,10,6),'tactile_deforms':(2,10,1,240,240)}
    for key,shape in expected.items():
        assert tuple(batch[key].shape)==shape
        assert torch.isfinite(batch[key]).all()
    if processor_path:
        assert batch['pixel_values'].ndim==2 and batch['pixel_values'].shape[0]>0
        assert batch['flare_pixel_values'].ndim==2 and batch['flare_pixel_values'].shape[0]>0
        assert tuple(batch['image_grid_thw'].shape)==(6,3)
        assert tuple(batch['flare_grid_thw'].shape)==(16,3)
    assert last['observation.images.head_is_pad'][1:].all()
    # DataLoader workers must not receive open VideoCapture / Parquet handles.
    state=dataset.__getstate__()
    assert all(not state[key] for key in ('_numeric','_parquets','_groups','_videos'))
    result={'train_frames':len(dataset),'val_frames':len(val),'train_episodes':len(dataset.episodes),
      'val_episodes':len(val.episodes),'batch_shapes':{key:list(shape) for key,shape in expected.items()},
      'all_checked_tensors_finite':True,'future_tail_padding':True,'worker_handles_cleared':True,
      'geometry':'collection-project nominal FR3 URDF, link8 flange',
      'processor':('real Qwen processor; model forward not run' if processor_path else
                   'test placeholder; real Qwen processor/model forward not run')}
    if processor_path:
        result['processor_shapes']={key:list(batch[key].shape) for key in
          ('input_ids','pixel_values','image_grid_thw','flare_pixel_values','flare_grid_thw')}
    Path(output).write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--upstream',required=True);p.add_argument('--prepared',required=True);p.add_argument('--output',required=True)
    p.add_argument('--processor-path',default=None,help='Local Qwen processor directory; optional')
    a=p.parse_args();validate(a.upstream,a.prepared,a.output,a.processor_path)
