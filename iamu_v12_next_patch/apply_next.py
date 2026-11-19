#!/usr/bin/env python3
"""Offline, fail-closed application of IAMU V1.2 P0-P4 diagnostics.

Usage: python apply_next.py --repo /path/to/iamu_d3dqn --check|--apply

Always patch a SEPARATE working copy of V1.2. Do not update a workspace in
which an existing V1.2 online run may still need exact-code resume.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime,timezone

EXPECTED_BLOBS={
 'v12_protocol/replay.py':'90e1c5c0f533fd75ca9c60b1c803e3d85c600bce',
 'v12_protocol/agent.py':'306c01dfe93ccc7a334b9b4bdcc414183276b599',
 'v12_protocol/fixed_data.py':'ec14c3f9fdebb99d64370c0a2ae8f71a0ab274df',
 'v12_protocol/validation.py':'f8c014b07acec08b2513a2e16c45491049b9e179',
}
UNMODIFIED_BASE_BLOBS={
 'iamu_dynamic/core.py':'4826023d2524275fa508af8bcc84740da84a725c',
 'iamu_dynamic/rl_env_v11.py':'08bd876da3ae0fb6f0753a2169a7d7d541576050',
 'iamu_dynamic/rl_ddqn_v11.py':'990a795cde401dfc136ef70d9e070e4b0b3ca173',
 'iamu_dynamic/rl_experiment_v11.py':'29bd4e78e91225b4037b6c45d59cd2cd21fe57ac',
 'iamu_dynamic/risk_selection.py':'af8743c1c6fe6de82b2846077c929bd7b6d4229a',
 'v11_protocol/train_v11.py':'50338334289ef7cc8c6f7583eba419f295eb80ea',
 'v11_protocol/v11_actors.py':'53105787a4936132b43975aa8b08a3b927b4bd5e',
 'v11_protocol/rounds.py':'3c26e8cf36b7b1bd5234637ab28ebd44217e1760',
 'v12_protocol/compress.py':'63c00f82620253c763efb496dad0b55980dc4e36',
 'v12_protocol/transitions.py':'875c1393a38f0324a42532cce2999f578211f748',
}
NEW_NAMES=['baseline_cache.py','probe_bank.py','probe_eval.py',
 'counterfactual_probe.py','diagnostics.py','fixed_f1.py','analyze_diagnostics.py']
NEW_TESTS=['test_diagnostic_protocol_next.py','test_probe_restore_next.py']


def git_blob_sha(contents):
    return hashlib.sha1(f'blob {len(contents)}\0'.encode()+contents).hexdigest()


def replace_once(content,old,new,name):
    count=content.count(old)
    if count!=1:raise ValueError(f'{name}: anchor found {count} times, expected 1')
    return content.replace(old,new)


def transformed(path,content,payload):
    if path=='v12_protocol/replay.py':
        return replace_once(content,
           '        self.sample_counts={k:v for k,v in self.sample_counts.items() if k in self._indices}',
           '        valid_keys=set(self._indices)\n        self.sample_counts={k:v for k,v in self.sample_counts.items() if k in valid_keys}',path)
    if path=='v12_protocol/agent.py':
        content=replace_once(content,
            '        nn.utils.clip_grad_norm_(self.online.parameters(),10.)',
            '        grad_norm_preclip=nn.utils.clip_grad_norm_(self.online.parameters(),10.)',path)
        return replace_once(content,
            '        return float(loss.detach())',
            '''        # Opt-in only: detached observations from THIS SAME TD update.
        # No extra forward/backward, RNG or optimizer calls.
        diagnostic_callback=getattr(self,'diagnostic_callback',None)
        if diagnostic_callback is not None and self.gradient_steps%diagnostic_callback.sample_every==0:
            diagnostic_callback(step=self.gradient_steps,transitions=transitions,
                q_taken=chosen.detach().cpu().numpy().copy(),
                target=y.detach().cpu().numpy().copy(),
                td_error=(y-chosen).detach().cpu().numpy().copy(),
                grad_norm_preclip=float(grad_norm_preclip.detach().cpu()),
                target_sync=(self.gradient_steps%self.target_update_steps==0))
        return float(loss.detach())''',path)
    if path=='v12_protocol/fixed_data.py':
        content=replace_once(content,
            '''def train(a):
    source=torch.load(a.data,map_location='cpu',weights_only=False)''',
            '''def train(a):
    if getattr(a,'probe_bank',None):
        from .fixed_f1 import train_f1
        return train_f1(a)
    if getattr(a,'diagnostics',False) or getattr(a,'checkpoint_every',0):
        raise ValueError('F1 diagnostics and snapshots require --probe-bank')
    source=torch.load(a.data,map_location='cpu',weights_only=False)''',path)
        content=replace_once(content,
            "    t.add_argument('--out',required=True)\n    a=p.parse_args(argv)",
            """    t.add_argument('--probe-bank',help='Required common raw probe for F1; keeps legacy behavior when omitted')
    t.add_argument('--diagnostics',action='store_true',help='Record true online/target TD statistics, opt-in')
    t.add_argument('--checkpoint-every',type=int,default=0,help='F1 model-only snapshot interval, 250 recommended')
    t.add_argument('--probe-every',type=int,default=250,help='Common-state Q-gap evaluation interval')
    t.add_argument('--baseline-cache-dir',help='Optional immutable Keep/Greedy/Periodic cache')
    t.add_argument('--out',required=True)
    a=p.parse_args(argv)""",path)
        return content
    if path=='v12_protocol/validation.py':
        return (payload/'v12_protocol/validation_next.py.template').read_text(encoding='utf-8')
    raise ValueError('Unknown changed file '+path)


def plan(root,payload):
    missing=[];inconsistent=[]
    for name,sha in {**EXPECTED_BLOBS,**UNMODIFIED_BASE_BLOBS}.items():
        p=root/name
        if not p.is_file():missing.append(str(p));continue
        got=git_blob_sha(p.read_bytes())
        if got!=sha:inconsistent.append(dict(path=name,expected=sha,observed=got))
    for name in NEW_NAMES:
        p=root/'v12_protocol'/name
        if p.exists():inconsistent.append(dict(path=str(p),reason='new-file-already-exists'))
        if not (payload/'v12_protocol'/name).is_file():missing.append('patch payload '+name)
    if not (root/'v12_protocol/train.py').is_file():
        missing.append(str(root/'v12_protocol/train.py'))
    for name in NEW_TESTS:
        testfile=root/'tests_v12'/name
        if testfile.exists():
            inconsistent.append(dict(path=str(testfile),reason='test-already-exists'))
        if not (payload/'tests_v12'/name).is_file():missing.append('patch test '+name)
    return missing,inconsistent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',required=True,help='Absolute iamu_d3dqn directory, NOT project root')
    grp=p.add_mutually_exclusive_group(required=True)
    grp.add_argument('--check',action='store_true');grp.add_argument('--apply',action='store_true')
    a=p.parse_args()
    root=Path(a.repo).expanduser().resolve()
    payload=Path(__file__).resolve().parent/'payload'
    if not root.is_dir():p.error(f'Code directory does not exist: {root}')
    missing,inconsistent=plan(root,payload)
    if missing or inconsistent:
        print(json.dumps(dict(status='REFUSED',missing=missing,drift=inconsistent),indent=2,ensure_ascii=False))
        raise SystemExit(2)
    if a.check:
        print('[PASS] Critical source and dependency Git blob SHAs match 70606c7b; no diagnostic files exist.')
        return
    # Backups are outside the code root and unrelated to historical results.
    backup=root.parent/('p0p4_backup_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S'))
    if backup.exists():raise FileExistsError(backup)
    backup.mkdir()
    originals={}
    for path in EXPECTED_BLOBS:
        dest=backup/path;dest.parent.mkdir(parents=True,exist_ok=True)
        src=root/path;shutil.copy2(src,dest);originals[path]=src.read_text(encoding='utf-8')
    changed={p:transformed(p,source,payload) for p,source in originals.items()}
    for name,text in changed.items():
        path=root/name
        path.write_text(text,encoding='utf-8')
    for name in NEW_NAMES:
        shutil.copy2(payload/'v12_protocol'/name,root/'v12_protocol'/name)
    for name in NEW_TESTS:
        shutil.copy2(payload/'tests_v12'/name,root/'tests_v12'/name)
    (backup/'application_manifest.json').write_text(json.dumps(dict(
        base_commit='70606c7b7ac4c6fc9063e25c92f56f58e1061c76',
        created_utc=datetime.now(timezone.utc).isoformat(),root=str(root),
        original_blobs=EXPECTED_BLOBS,validated_dependency_blobs=UNMODIFIED_BASE_BLOBS,
        new_files=NEW_NAMES,new_tests=NEW_TESTS,backup=str(backup)),indent=2))
    paths=list(EXPECTED_BLOBS)+['v12_protocol/'+x for x in NEW_NAMES]+['tests_v12/'+x for x in NEW_TESTS]
    for name in paths:
        code=root/name
        cp=subprocess.run([sys.executable,'-m','py_compile',str(code)],capture_output=True,text=True)
        if cp.returncode:
            print(f'[ERROR] {name}: {cp.stderr}');raise SystemExit(3)
    print('[APPLIED] '+str(root))
    print('[BACKUP] '+str(backup))
    print('[WARN] This source has a NEW code fingerprint. Do NOT resume historical V1.2 online states from it.')

if __name__=='__main__':main()
