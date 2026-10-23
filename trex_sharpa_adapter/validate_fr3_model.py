"""Validate the supplied nominal FR3 model and real recorded joint conversion.

No robot commands. SciPy is only used for an independent math cross-check.
"""
import argparse, json, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from sharpa_core import *

def reference_fk(path,base,tip,joint_names,q):
    tree=ET.parse(path).getroot()
    parents={j.find('child').get('link'):j for j in tree.findall('joint')}
    chain=[]
    while tip!=base:
        joint=parents[tip];chain.insert(0,joint);tip=joint.find('parent').get('link')
    result=[]
    for row in q:
        T=np.eye(4)
        for joint in chain:
            origin=joint.find('origin');A=np.eye(4)
            if origin is not None:
                A[:3,3]=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
                A[:3,:3]=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix()
            T=T@A
            if joint.get('type')!='fixed':
                axis=joint.find('axis');axis=np.fromstring(axis.get('xyz') if axis is not None else '1 0 0',sep=' ')
                M=np.eye(4);M[:3,:3]=Rotation.from_rotvec(axis/np.linalg.norm(axis)*row[joint_names.index(joint.get('name'))]).as_matrix()
                T=T@M
        result.append(T)
    return np.array(result)

def validate(config_path,output,prepared=None):
    config=load_config(config_path);fks=validate_config(config);rng=np.random.default_rng(42)
    result={'geometry':'nominal URDF supplied in collection project; no hardware Cartesian ground-truth comparison',
            'chosen_reference':'per-arm base link0 -> link8 flange','sources':[],'random_fk_max_error':0.,
            'link7_to_link8_max_error':0.,'mount_invariance_max_error':0.}
    for side in ('left','right'):
        arm=config['arms'][side];tree=ET.parse(arm['urdf']).getroot();by_name={j.get('name'):j for j in tree.findall('joint')}
        limits=[by_name[n].find('limit') for n in arm['joint_names']]
        q=rng.uniform([float(l.get('lower')) for l in limits],[float(l.get('upper')) for l in limits],size=(64,7))
        got=fks[side](q);ref=reference_fk(arm['urdf'],arm['base_link'],arm['tip_link'],arm['joint_names'],q)
        error=float(np.max(np.abs(got-ref)));assert error<1e-10
        result['random_fk_max_error']=max(result['random_fk_max_error'],error)
        fk7=URDFFK(arm['urdf'],arm['base_link'],f'{side}_fr3v2_link7',arm['joint_names'])
        offset=np.eye(4);offset[2,3]=.107
        error=float(np.max(np.abs(fk7(q)@offset-got)));assert error<1e-10
        result['link7_to_link8_max_error']=max(result['link7_to_link8_max_error'],error)
    for dataset_root in config['dataset_roots']:
        root=Path(dataset_root);info=read_json(root/'meta/info.json')
        raw=raw_episode(root/info['data_path'].format(episode_chunk=0,episode_index=0),info)
        data=convert_numeric(raw,info,fks,config.get('hand_order',HAND),config.get('hand_scale',1.),config.get('wrench_scale'))
        global_fks={side:URDFFK(config['arms'][side]['urdf'],'lychee_root',config['arms'][side]['tip_link'],config['arms'][side]['joint_names']) for side in ('left','right')}
        world=convert_numeric(raw,info,global_fks,config.get('hand_order',HAND),config.get('hand_scale',1.),config.get('wrench_scale'))
        sample_error=0.;recover_error=0.;rotation_error=0.
        segments=continuous_segments(raw['audit.original_timestamp_ns'],info['fps'])
        for i in sorted(set([0,len(raw['action'])-1]+[a for a,b in segments]+[b-1 for a,b in segments])):
            end=next(b for a,b in segments if a<=i<b)
            local_chunk=action_chunk(data,i,end);world_chunk=action_chunk(world,i,end)
            e=float(np.max(np.abs(local_chunk-world_chunk)));assert e<1e-6
            result['mount_invariance_max_error']=max(result['mount_invariance_max_error'],e)
            for side,off in [('left',0),('right',31)]:
                base=data[side+'_state_pose'][i]
                for k in range(16):
                    delta=local_chunk[k,off:off+9];D=np.eye(4)
                    D[:3,3]=delta[:3];D[:3,:3]=np.column_stack([delta[3:6],delta[6:9],np.cross(delta[3:6],delta[6:9])])
                    expected=data[side+'_target_pose'][min(i+k,end-1)]
                    error=float(np.max(np.abs(base@D-expected)));assert error<1e-6
                    recover_error=max(recover_error,error)
                R=data[side+'_state_pose'][:,:3,:3]
                rotation_error=max(rotation_error,float(np.max(np.abs(np.swapaxes(R,1,2)@R-np.eye(3)))))
                expected_hand=raw['action'][:,named_indices(info,'action',[f'{side}_{n}.command' for n in config.get('hand_order',HAND)])]
                np.testing.assert_allclose(data[side+'_hand_target'],expected_hand*config.get('hand_scale',1.),atol=1e-7)
        assert rotation_error<1e-10
        result['sources'].append({'root':str(root),'real_frames_converted':len(raw['action']),
            'segments':len(segments),'absolute_pose_roundtrip_max_error':recover_error,
            'rotation_orthogonality_max_error':rotation_error,'hand_targets_preserved':True})
    if prepared:
        root=Path(prepared);manifest=read_json(root/'manifest.json');frames=0
        for ep in manifest['episodes']:
            with np.load(root/ep['numeric']) as blob:
                assert blob['state'].shape==(ep['length'],62)
                assert blob['f6'].shape==(ep['length'],10,6)
                assert all(np.isfinite(blob[k]).all() for k in blob.files)
            frames+=ep['length']
        train={ep['group'] for ep in manifest['episodes'] if ep['split']=='train'}
        val={ep['group'] for ep in manifest['episodes'] if ep['split']=='val'}
        assert not (train&val)
        result['prepared_dataset']={'episodes':len(manifest['episodes']),'frames':frames,'train_sources':len(train),
          'val_sources':len(val),'normalization':manifest['stats_sampling'],'finite':True,'split_disjoint':True}
    Path(output).write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--prepared')
    a=p.parse_args();validate(a.config,a.output,a.prepared)
