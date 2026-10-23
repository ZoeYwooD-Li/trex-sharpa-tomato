"""Prepare SMALL numeric caches/manifest/stats, without copying RGB or tactile maps."""
from __future__ import annotations
import argparse, hashlib, json, shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from sharpa_core import *

def prepare_episode(job):
    root,info,ep,tasks,config,fks,output,cache_index=job
    ei=ep['episode_index'];fmt={'episode_index':ei,'episode_chunk':ei//info['chunks_size']}
    path=root/info['data_path'].format(**fmt)
    raw=raw_episode(path,info)
    if len(raw['action'])!=ep['length']:raise ValueError(f'Length mismatch {path}')
    numeric=convert_numeric(raw,info,fks,config.get('hand_order',HAND),config.get('hand_scale',1.),config.get('wrench_scale'))
    segments=continuous_segments(raw['audit.original_timestamp_ns'],info['fps'],config.get('gap_factor',1.5))
    task_id=raw['task_index']
    if len(set(task_id.tolist()))!=1:raise ValueError(f'Mixed tasks: {path}')
    record_path=root/'conversion/records'/f'{ei:06d}.json'
    if not record_path.exists():raise ValueError(f'Missing provenance {record_path}; cannot safely group splits')
    record=read_json(record_path)
    bags=[s for s in record.get('source_snapshot',[]) if s['name'].endswith(('.db3','.mcap'))]
    if not bags:raise ValueError(f'No source recording identity: {record_path}')
    group=ep.get('source_recording_id') or hashlib.sha256(json.dumps(bags,sort_keys=True).encode()).hexdigest()
    cache=f'numeric/{cache_index:06d}.npz'
    np.savez_compressed(output/cache,**numeric)
    videos={role:str(root/info['video_path'].format(video_key='observation.images.'+camera,**fmt)) for role,camera in config['cameras'].items()}
    if any(not Path(v).is_file() for v in videos.values()):raise FileNotFoundError(videos)
    return {'parquet':str(path),'numeric':cache,'length':len(raw['action']),
      'segments':segments,'group':group,'task':tasks[int(task_id[0])],'videos':videos,
      'source_root':str(root),'episode_index':ei,'fps':info['fps']}

def write_adapter_metadata(config,output):
    output=Path(output);models={};(output/'robot').mkdir(exist_ok=True)
    for side,arm in config['arms'].items():
        source=Path(arm['urdf']);destination=output/'robot'/f'{side}.urdf'
        shutil.copy2(source,destination)
        models[side]={**arm,'urdf':str(destination.relative_to(output)).replace('\\','/'),
                      'sha256':hashlib.sha256(destination.read_bytes()).hexdigest()}
        for notice in ('FRANKA_DESCRIPTION_LICENSE','FRANKA_DESCRIPTION_NOTICE'):
            if (source.parent/notice).exists():shutil.copy2(source.parent/notice,output/'robot'/notice)
    metadata={'version':1,'action_dim':62,'action_chunk':16,'action_mode':'delta-base from chunk-start measured FK; absolute hand targets',
      'robot_models':models,'hand_order':list(config.get('hand_order',HAND)),
      'hand_scale':config.get('hand_scale',1.),'wrench_scale':config.get('wrench_scale',[1]*6),'cameras':config['cameras']}
    (output/'adapter_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return models

def prepare(config_path, output):
    config=load_config(config_path); fks=validate_config(config)
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    (output/'numeric').mkdir()
    manifest={'version':1,'config':config,'episodes':[], 'seed':config.get('seed',42)}
    manifest['robot_models']=write_adapter_metadata(config,output)
    for root_string in config['dataset_roots']:
        root=Path(root_string).absolute(); info=read_json(root/'meta/info.json')
        if info['codebase_version'] != 'v2.1' or info['fps'] != 30:
            raise ValueError(f'Expected v2.1 at 30Hz: {root}')
        tasks={x['task_index']:x['task'] for x in map(json.loads,(root/'meta/tasks.jsonl').read_text(encoding='utf-8').splitlines())}
        epmeta=list(map(json.loads,(root/'meta/episodes.jsonl').read_text(encoding='utf-8').splitlines()))
        start_index=len(manifest['episodes'])
        jobs=[(root,info,ep,tasks,config,fks,output,start_index+i) for i,ep in enumerate(epmeta)]
        with ThreadPoolExecutor(max_workers=int(config.get('prepare_workers',4))) as pool:
            for prepared in pool.map(prepare_episode,jobs):
                prepared['dataset_id']='/'.join(str(root).replace('\\','/').rstrip('/').split('/')[-2:])
                manifest['episodes'].append(prepared)
                print(f'Prepared {root.name}/{prepared["episode_index"]}: {prepared["length"]} frames, {len(prepared["segments"])} segments',flush=True)
    groups=sorted({ep['group'] for ep in manifest['episodes']})
    rng=np.random.default_rng(manifest['seed']); rng.shuffle(groups)
    ratio=config.get('val_ratio',.05)
    if not 0 <= ratio < 1: raise ValueError('val_ratio must be in [0,1)')
    if ratio and len(groups)<2:raise ValueError('Need at least 2 source recordings to hold out validation')
    nv=min(len(groups)-1,max(1,int(round(len(groups)*ratio)))) if ratio else 0
    val=set(groups[:nv])
    for ep in manifest['episodes']:ep['split']='val' if ep['group'] in val else 'train'
    # Uniform sampling over training frames only. No validation leakage; bounded RAM.
    train=[ep for ep in manifest['episodes'] if ep['split']=='train']
    total=sum(ep['length'] for ep in train)
    if not total: raise ValueError('Empty train split')
    count=min(total,config.get('stats_max_frames',20000))
    positions=np.sort(rng.choice(total,size=count,replace=False)); acc=0; aa=[]; ss=[]; ff=[]
    for ep in train:
        selected=positions[(positions>=acc)&(positions<acc+ep['length'])]-acc; acc+=ep['length']
        if not len(selected):continue
        with np.load(output/ep['numeric']) as blob:
            data={key:blob[key] for key in blob.files}
            for i in selected:
                end=next(b for a,b in ep['segments'] if a<=i<b)
                aa.append(action_chunk(data,int(i),end));ss.append(data['state'][i]);ff.append(data['f6'][i].reshape(-1))
    stats={'rlbench':{'action':stats_block(np.array(aa)),'state':stats_block(np.array(ss)),
      'tactile_f6':stats_block(np.array(ff)),'num_transitions':total,'num_trajectories':len(train)}}
    manifest['stats_sampling']={'frames':count,'total_training_frames':total,'exact':count==total,'seed':manifest['seed']}
    # No guessed tracking-error noise. The proposed launch uses use_robot_state=0.
    (output/'stats_data.json').write_text(json.dumps(stats),encoding='utf-8')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(f'Done: {len(train)} train / {len(manifest["episodes"])-len(train)} val episodes; stats={count} training frames')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();prepare(a.config,a.output)
