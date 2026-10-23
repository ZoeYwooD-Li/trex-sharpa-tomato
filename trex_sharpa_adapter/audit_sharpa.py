"""Read every numerical frame; sample dense maps at episode ends.

--resume reuses datasets already completed in the report; only use it if sources
have not changed since that report was started.
"""
import argparse, json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from sharpa_core import read_json, raw_episode, dense_column, continuous_segments, DEFORM_SOURCE

def inspect_episode(args):
    path,info=args;raw=raw_episode(path,info)
    dif=np.diff(raw['audit.original_timestamp_ns'])
    gaps=int(np.sum((dif<=0)|(dif>1.5e9/info['fps'])))
    item={'episode':int(path.stem.split('_')[-1]),'frames':len(raw['action']),
          'bytes':path.stat().st_size,'gaps':gaps,'max_gap_ms':float(dif.max()/1e6),
          'segments':len(continuous_segments(raw['audit.original_timestamp_ns'],info['fps']))}
    return item

def audit(root,output,resume=False,workers=4):
    root=Path(root);output=Path(output)
    report=read_json(output) if resume and output.exists() else {'root':str(root),'datasets':[]}
    done={x['root'] for x in report['datasets']}
    for meta in sorted(root.glob('*/*/meta/info.json')):
        base=meta.parent.parent
        if str(base) in done:continue
        info=read_json(meta)
        item={'root':str(base),'version':info['codebase_version'],'robot':info['robot_type'],
          'episodes':info['total_episodes'],'frames':info['total_frames'],'fps':info['fps'],
          'features':{k:{'shape':v['shape'],'dtype':v['dtype']} for k,v in info['features'].items()},
          'actual_frames':0,'parquet_bytes':0,'video_bytes':0,'errors':[],
          'episodes_with_original_time_gaps':[],'numeric_episodes_checked':0}
        paths=sorted((base/'data').glob('*/*.parquet'))
        if len(paths)!=info['total_episodes']:raise ValueError(f'Episode count mismatch {base}')
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for j,result in enumerate(pool.map(inspect_episode,[(p,info) for p in paths])):
                item['actual_frames']+=result['frames'];item['parquet_bytes']+=result['bytes']
                item['numeric_episodes_checked']+=1
                if result['gaps']:item['episodes_with_original_time_gaps'].append(result)
                if j%10==0:print(f'{base.name}: {j+1}/{len(paths)}',flush=True)
        item['missing_videos']=sum(not (base/info['video_path'].format(episode_chunk=ep//info['chunks_size'],episode_index=ep,video_key=k)).is_file()
          for ep in range(info['total_episodes']) for k,v in info['features'].items() if v['dtype']=='video')
        item['video_bytes']=sum(p.stat().st_size for p in (base/'videos').glob('*/*/*.mp4'))
        for path in (paths[0],paths[-1]):
            pf=pq.ParquetFile(path)
            table=pf.read_row_group(0,columns=DEFORM_SOURCE)
            maps={key:dense_column(table[key])[:1] for key in DEFORM_SOURCE}
            item.setdefault('deform_samples',[]).append({k:{'shape':list(a.shape[1:]),'min':int(a.min()),'max':int(a.max())} for k,a in maps.items()})
        if item['actual_frames']!=info['total_frames']:item['errors'].append('Frame count mismatch')
        report['datasets'].append(item)
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(f'{base.name}: complete, frames={item["actual_frames"]}, missing_video={item["missing_videos"]}',flush=True)
    report['complete']=len(report['datasets'])==len(list(root.glob('*/*/meta/info.json')))
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--resume',action='store_true');p.add_argument('--workers',type=int,default=4)
    args=p.parse_args();audit(args.root,args.output,args.resume,args.workers)
