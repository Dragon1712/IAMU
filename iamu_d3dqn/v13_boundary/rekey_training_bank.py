"""Re-key external training probe metadata without changing saved simulator worlds."""
import argparse,copy
from pathlib import Path
from v13_diagnostics.common import load_trusted,save_trusted,atomic_json,digest
from .utils import boundary_provenance

def rekey_training_bank(bank,episode_offset=10000000):
    if not bank.get('probes') or episode_offset<=0:raise ValueError('Invalid bank/offset')
    new=copy.deepcopy(bank);mapping=[];seen=set()
    for p in new['probes']:
        old=int(p['episode_id']);step=int(p['raw_step_index'])
        if p['state_id']!=f'{old}:{step}':raise ValueError('Unknown state_id format')
        p['episode_id']=old+episode_offset;p['state_id']=f'{old+episode_offset}:{step}'
        if p['state_id'] in seen:raise ValueError('State collision')
        seen.add(p['state_id']);mapping.append((f'{old}:{step}',p['state_id']))
    new['label_bank_namespace_offset']=episode_offset
    new['original_state_id_mapping']=mapping
    return new

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--episode-offset',type=int,default=10000000)
    a=p.parse_args(argv);dest=Path(a.output)
    if dest.exists():raise FileExistsError(dest)
    bank=rekey_training_bank(load_trusted(a.input),a.episode_offset)
    save_trusted(dest,bank)
    atomic_json(dest.parent/'rekey_manifest.json',boundary_provenance(a,inputs=[a.input])|
       dict(output_sha256=digest(dest),n_states=len(bank['probes'])))
if __name__=='__main__':main()
