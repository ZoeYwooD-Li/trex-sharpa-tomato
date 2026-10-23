"""Numerical regression + optional real-data loader smoke test (no model/GPU).

The smoke test uses explicitly synthetic arm geometry to exercise I/O only.
It does NOT validate the real FR3 tool frame or action kinematics.
"""
from __future__ import annotations
import argparse, ast, json, sys, tempfile, types
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sharpa_core import *

def tests(upstream):
    # Execute the official math as written, without importing its cv2 dependency.
    tree=ast.parse((Path(upstream)/'utils/lerobot_common.py').read_text(encoding='utf-8'))
    names={'compute_chunk_delta_pose','build_action_chunk'}
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    ns={'np':np,'ACTION_CHUNK':16,'ACTION_DIM':62,'FRAME_STRIDE':1}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'upstream_math','exec'),ns)
    rng=np.random.default_rng(21)
    data={}
    for side in ('left','right'):
        for kind in ('state','target'):
            T=np.broadcast_to(np.eye(4),(25,4,4)).copy()
            for i in range(len(T)):
                T[i,:3,:3]=rotation(rng.normal(size=3),rng.normal())
                T[i,:3,3]=rng.normal(size=3)
            data[side+'_'+kind+'_pose']=T
        data[side+'_hand_target']=rng.normal(size=(25,22))
    for i in (0,5,12,24):
        ours=action_chunk(data,i,25)
        theirs=ns['build_action_chunk'](data['left_state_pose'],data['left_target_pose'],data['left_hand_target'],
                    data['right_state_pose'],data['right_target_pose'],data['right_hand_target'],i,25)
        np.testing.assert_allclose(ours,theirs,atol=1e-6)
    # Clip at a cut: later actions MUST NOT leak into the preceding segment.
    chunk=action_chunk(data,4,6)
    np.testing.assert_allclose(chunk[1:],np.broadcast_to(chunk[1],chunk[1:].shape))
    assert continuous_segments(np.array([0,33333333,66666666,500000000,533333333]),30)==[(0,3),(3,5)]
    arr=np.arange(48,dtype=np.uint8).reshape(3,4,4)
    flat=pa.array(arr.ravel())
    nested=pa.FixedSizeListArray.from_arrays(pa.FixedSizeListArray.from_arrays(flat,4),4)
    np.testing.assert_array_equal(dense_column(pa.chunked_array([nested.slice(0,1),nested.slice(1)])),arr)
    scratch=Path.cwd()/'work';scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as tmp:
        file=Path(tmp)/'test.urdf'
        file.write_text('<robot name="test"><link name="base"/><link name="a"/><link name="tip"/>'
          '<joint name="q" type="revolute"><parent link="base"/><child link="a"/><axis xyz="0 0 1"/></joint>'
          '<joint name="tool" type="fixed"><parent link="a"/><child link="tip"/><origin xyz="1 0 0"/></joint></robot>')
        fk=URDFFK(file,'base','tip',['q'])
        T=fk(np.array([[0],[np.pi/2]]))
        np.testing.assert_allclose(T[:,:3,3],[[1,0,0],[0,1,0]],atol=1e-7)
    return {'official_delta_math':'passed (4 random chunks including episode tail)',
      'cut_boundary_padding':'passed','cut_detection':'passed','nested_parquet_arrays':'passed','urdf_transform_order':'passed'}

def synthetic_urdf(path):
    text=['<robot name="synthetic_test_only"><link name="base"/>']
    for i in range(7):
        text.append(f'<link name="link{i}"/><joint name="j{i}" type="revolute"><parent link="'+('base' if i==0 else f'link{i-1}')+f'"/><child link="link{i}"/><origin xyz="0 0 0.1"/><axis xyz="0 0 1"/></joint>')
    path.write_text(''.join(text)+'</robot>')

def prepare_test():
    from prepare_sharpa import prepare
    scratch=Path.cwd()/'work';scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as tmp:
        tmp=Path(tmp);synthetic_urdf(tmp/'synthetic.urdf')
        roots=[]
        state_names=[]
        for side in ('left','right'):
            state_names.extend(f'{side}_fr3_joint{i}.position' for i in range(1,8))
            state_names.extend(f'{side}_fr3_joint{i}.velocity' for i in range(1,8))
        for side in ('left','right'):
            state_names.extend(f'{side}_{x}.position' for x in HAND)
            state_names.extend(f'{side}_{x}.velocity' for x in HAND)
        action_names=[f'{s}_fr3_joint{i}.position' for s in ('left','right') for i in range(1,8)]
        action_names += [f'{s}_{x}.command' for s in ('left','right') for x in HAND]
        for gi in range(2):
            root=tmp/f'source{gi}';roots.append(str(root))
            (root/'meta').mkdir(parents=True);(root/'data').mkdir();(root/'conversion/records').mkdir(parents=True)
            info={'codebase_version':'v2.1','fps':30,'chunks_size':1000,'data_path':'data/episode_{episode_index:06d}.parquet',
              'video_path':'{video_key}.mp4','features':{'observation.state':{'names':state_names},'action':{'names':action_names}}}
            (root/'meta/info.json').write_text(json.dumps(info))
            (root/'meta/tasks.jsonl').write_text(json.dumps({'task_index':0,'task':'test'}))
            (root/'meta/episodes.jsonl').write_text(json.dumps({'episode_index':0,'length':20,'source_recording_id':f'g{gi}'}))
            (root/'conversion/records/000000.json').write_text(json.dumps({'source_snapshot':[{'name':f'bag{gi}.db3','size':1,'mtime_ns':1}]}))
            for cam in ('cam0','cam1','cam2'):(root/f'observation.images.{cam}.mp4').write_bytes(b'test-placeholder')
            columns={'observation.state':np.full((20,116),gi+.1).tolist(),'action':np.full((20,58),gi+.2).tolist(),
              'timestamp':(np.arange(20)/30).tolist(),'frame_index':list(range(20)),'episode_index':[0]*20,
              'audit.original_timestamp_ns':(np.arange(20)*33333333).tolist(),'task_index':[0]*20}
            columns.update({key:np.full((20,6),gi+1.).tolist() for key in WRENCH_SOURCE})
            pq.write_table(pa.table(columns),root/'data/episode_000000.parquet')
        arms={s:{'urdf':str(tmp/'synthetic.urdf'),'base_link':'base','tip_link':'link6','joint_names':[f'j{i}' for i in range(7)]} for s in ('left','right')}
        config={'dataset_roots':roots,'cameras':{'head':'cam0','wrist_left':'cam1','wrist_right':'cam2'},
                'arms':arms,'seed':42,'val_ratio':.5,'stats_max_frames':20}
        (tmp/'config.json').write_text(json.dumps(config));prepare(tmp/'config.json',tmp/'prepared')
        manifest=read_json(tmp/'prepared/manifest.json');stats=read_json(tmp/'prepared/stats_data.json')
        train=[e for e in manifest['episodes'] if e['split']=='train'];val=[e for e in manifest['episodes'] if e['split']=='val']
        assert len(train)==len(val)==1 and train[0]['group']!=val[0]['group']
        expected=1. if train[0]['group']=='g0' else 2.
        np.testing.assert_allclose(stats['rlbench']['tactile_f6']['mean'],expected)
        assert np.asarray(stats['rlbench']['action']['q01']).shape==(16,62)
    return 'passed (synthetic source fixtures; grouped split, training-only stats, [16,62] quantiles)'

def smoke(upstream, root, out):
    import cv2, torch
    from PIL import Image, ImageDraw
    upstream=Path(upstream).resolve()
    # Namespace shells avoid loading qwen_vla.__init__, which imports the VLM.
    for package in ('qwen_vla','utils'):
        mod=types.ModuleType(package);mod.__path__=[str(upstream/package)];sys.modules[package]=mod
    from sharpa_dataset import SharpaDataset
    base=Path(root);info=read_json(base/'meta/info.json')
    path=base/info['data_path'].format(episode_chunk=0,episode_index=0)
    raw=raw_episode(path,info)
    scratch=Path.cwd()/'work';scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as temp:
        temp=Path(temp);synthetic_urdf(temp/'synthetic.urdf')
        fk=URDFFK(temp/'synthetic.urdf','base','link6',[f'j{i}' for i in range(7)])
        data=convert_numeric(raw,info,{'left':fk,'right':fk})
        np.savez_compressed(temp/'numeric.npz',**data)
        segments=continuous_segments(raw['audit.original_timestamp_ns'],30)
        videos={role:str(base/info['video_path'].format(episode_chunk=0,episode_index=0,video_key='observation.images.'+cam))
          for role,cam in [('head','cam0'),('wrist_right','cam1'),('wrist_left','cam2')]}
        # Source collection_state.json device_identities records this camera mapping.
        ep={'parquet':str(path),'numeric':'numeric.npz','length':len(raw['action']),'segments':segments,
            'task':'Pick a tomato.','videos':videos,'split':'train','group':'test','fps':30}
        (temp/'manifest.json').write_text(json.dumps({'config':{'val_ratio':.05},'seed':42,'episodes':[ep]}))
        aa=np.array([action_chunk(data,i,next(b for a,b in segments if a<=i<b)) for i in range(min(64,len(raw['action'])))])
        (temp/'stats_data.json').write_text(json.dumps({'rlbench':{'action':stats_block(aa),'state':stats_block(data['state']),
            'tactile_f6':stats_block(data['f6'].reshape(-1,60))}}))
        class Batch(dict):
            __getattr__=dict.__getitem__
        class Processor:
            tokenizer=types.SimpleNamespace(pad_token_id=0)
            def apply_chat_template(self,*a,**k):return 'test'
            def __call__(self,**kwargs):
                return Batch(input_ids=torch.tensor([[1,2,3]]),
                    pixel_values=torch.zeros((len(kwargs['images']),4)),image_grid_thw=torch.ones((len(kwargs['images']),3),dtype=torch.long))
            def image_processor(self,images,**kwargs):
                return Batch(pixel_values=torch.zeros((len(images),4)),image_grid_thw=torch.ones((len(images),3),dtype=torch.long))
        cfg=types.SimpleNamespace(lerobot_root=str(temp),action_dim=62,action_chunk=16,use_robot_state=0,
          image_size=[384,288],use_flare=1,n_flare_steps=8,flare_frame_stride=4,
          use_tactile_vec=1,use_tactile_deform=1,use_tactile_vqvae=1,vqvae_window=16)
        ds=SharpaDataset(cfg,Processor(),types.SimpleNamespace(print=print))
        item=ds[0];tail=ds[len(ds)-1];batch=ds.collate_fn([item,tail])
        assert batch['noisy_actions'].shape==(2,16,62)
        assert batch['tactile_f6_history'].shape==(2,16,10,6)
        assert batch['tactile_deforms'].shape==(2,10,1,240,240)
        np.testing.assert_allclose(item['observation.tactile_f6'].numpy(),np.broadcast_to(data['f6'][0],(16,10,6)))
        assert tail['observation.images.head_is_pad'][1:].all()
        for key in ('noisy_actions','target','norm_actions','tactile_f6s','tactile_f6_history','tactile_deforms'):
            assert torch.isfinite(batch[key]).all(),key
        montage=Image.new('RGB',(960,280),'white');draw=ImageDraw.Draw(montage)
        for i,cam in enumerate(('cam0','cam1','cam2')):
            role=('head','wrist_right','wrist_left')[i]
            arr=(item['observation.images.head'][0] if i==0 else item['observation.images.'+role])
            im=Image.fromarray((arr.permute(1,2,0).numpy()*255).astype(np.uint8));im.thumbnail((320,240))
            montage.paste(im,(i*320,30));draw.text((i*320+10,8),cam+' = '+role,fill='black')
        montage.save(out/'camera_preview.jpg')
        return {'source':str(base),'frames_loaded':len(raw['action']),'continuous_segments':len(segments),
          'batch_shapes':{k:list(v.shape) for k,v in batch.items() if isinstance(v,torch.Tensor)},
          'real_video_decode':'passed (first and last frame with future windows)',
          'real_deformation_decode':'passed (10 fingers, first and last frame)',
          'official_collate':'passed with a small test processor',
          'robot_geometry':'SYNTHETIC TEST ONLY; actual FR3 calibration remains required',
          'camera_roles':'cam0=head; cam1=wrist_right; cam2=wrist_left (collection_state.json)', 'full_model_forward':'not run'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--upstream',required=True);p.add_argument('--root');p.add_argument('--output',required=True)
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    result={'numerical_tests':tests(args.upstream),'prepare_integration':prepare_test()}
    if args.root:result['real_data_smoke']=smoke(args.upstream,args.root,out)
    (out/'validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
