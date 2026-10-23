"""Relink a prepared numeric cache to the training machine's raw-data mount."""
import argparse, json
from pathlib import Path, PurePosixPath

def relocate(prepared,data_root):
    prepared=Path(prepared);path=prepared/'manifest.json'
    manifest=json.loads(path.read_text(encoding='utf-8'))
    roots={}
    for ep in manifest['episodes']:
        dataset_id=PurePosixPath(ep['dataset_id'])
        if dataset_id.is_absolute() or '..' in dataset_id.parts or len(dataset_id.parts)!=2:
            raise ValueError(f'Unexpected dataset_id {dataset_id}')
        old=ep['source_root'].replace('\\','/').rstrip('/')
        new=Path(data_root).absolute()/Path(*dataset_id.parts)
        def map_path(value):
            value=value.replace('\\','/')
            if not value.startswith(old+'/'):raise ValueError('Source path is outside its dataset root')
            rel=PurePosixPath(value[len(old)+1:])
            if rel.is_absolute() or '..' in rel.parts:raise ValueError('Invalid relative data path')
            out=new/Path(*rel.parts)
            if not out.is_file():raise FileNotFoundError(out)
            return str(out)
        ep['parquet']=map_path(ep['parquet'])
        ep['videos']={key:map_path(value) for key,value in ep['videos'].items()}
        roots[old]=str(new);ep['source_root']=str(new)
    manifest['config']['dataset_roots']=list(roots.values())
    backup=prepared/'manifest.before_relocation.json'
    if backup.exists():raise FileExistsError(f'Backup already exists: {backup}')
    backup.write_bytes(path.read_bytes())
    path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(f'Relinked {len(manifest["episodes"])} episodes; original manifest saved in {backup}')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared',required=True);p.add_argument('--data-root',required=True)
    a=p.parse_args();relocate(a.prepared,a.data_root)
