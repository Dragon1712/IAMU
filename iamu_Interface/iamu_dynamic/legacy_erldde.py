"""Optional frozen original ERL-DDE proposer (NO retraining).

This module compiles a separate *instrumented build* of the user's original
BACAP_env.cpp. It does not edit or overwrite any ERL-DDE source or .so.
The frozen trained ES model is replayed on each ETA snapshot, and its original
C++ action selection order and berth coordinate provide HINTS to the new
commitment scheduler; it is not a replacement for the published simulation.
"""
from __future__ import annotations
import ctypes
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Mapping

from .core import Hint,Vessel,Port

class _State(ctypes.Structure):
    _fields_=[('data',(ctypes.c_double*8)*30)]
class _Result(ctypes.Structure):
    _fields_=[('state',_State),('reward',ctypes.c_double),('done',ctypes.c_bool)]

def build_probe(repo_root:str|Path, output_dir:str|Path)->Path:
    """Compile original Linux C++ simulator with extra read-only selection accessor.

    Note: this source is included at build-time from user's LOCAL original repo;
    its contents are not bundled or relicensed here.
    """
    repo_root=Path(repo_root).resolve()
    original=repo_root/'BACAP_env.cpp'
    if not original.is_file():raise FileNotFoundError(original)
    dest=Path(output_dir).resolve();dest.mkdir(parents=True,exist_ok=True)
    src=dest/'bacap_probe_compile.cpp';so=dest/'libBACAP_env_probe.so'
    include=str(original).replace('\\','\\\\').replace('"','\\"')
    code=f'''#include "{include}"
extern "C" {{
  int probe_selected_vessel(void* p, int action_index) {{
    auto* env = static_cast<BerthOptimizationEnvironment*>(p);
    if (!env || action_index < 0 ||
        action_index >= static_cast<int>(env->arrival_vessel.size())) return -1;
    return std::get<0>(env->arrival_vessel[action_index]);
  }}
}}
'''
    src.write_text(code,encoding='utf-8')
    subprocess.run(['g++','-std=c++17','-O2','-fPIC','-shared',str(src),'-o',str(so)],check=True)
    return so

class _ProbeEnv:
    def __init__(self,library:Path,case_file:Path):
        self.lib=ctypes.CDLL(str(library))
        self.lib.create_env.argtypes=[ctypes.c_char_p];self.lib.create_env.restype=ctypes.c_void_p
        self.lib.delete_env.argtypes=[ctypes.c_void_p];self.lib.delete_env.restype=None
        self.lib.reset.argtypes=[ctypes.c_void_p,ctypes.POINTER(_State)];self.lib.reset.restype=None
        self.lib.step.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_double),ctypes.POINTER(_Result)];self.lib.step.restype=None
        self.lib.probe_selected_vessel.argtypes=[ctypes.c_void_p,ctypes.c_int]
        self.lib.probe_selected_vessel.restype=ctypes.c_int
        self.ptr=self.lib.create_env(str(case_file).encode('utf-8'))
        if not self.ptr: raise RuntimeError('Could not construct probe environment')
    def reset(self):
        import numpy as np
        state=_State();self.lib.reset(self.ptr,ctypes.byref(state))
        return np.ctypeslib.as_array(state.data).copy().reshape(30,8)
    def action_ship(self,i):return int(self.lib.probe_selected_vessel(self.ptr,int(i)))
    def step(self,values):
        import numpy as np
        values=np.asarray(values,dtype='float64').reshape(-1)
        if len(values)!=5:raise ValueError('ERL-DDE must output 5 action components')
        arr=(ctypes.c_double*5)(*map(float,values));result=_Result()
        self.lib.step(self.ptr,arr,ctypes.byref(result))
        return np.ctypeslib.as_array(result.state.data).copy().reshape(30,8),bool(result.done)
    def close(self):
        if self.ptr:self.lib.delete_env(self.ptr);self.ptr=None
    def __enter__(self):return self
    def __exit__(self,*_):self.close()

class FrozenERLDDEHints:
    def __init__(self,repo_root:str|Path,checkpoint:str|Path,probe_so:str|Path,
                 max_steps=1000):
        import torch
        self.repo_root=Path(repo_root).resolve()
        self.checkpoint=Path(checkpoint).resolve()
        self.probe_so=Path(probe_so).resolve()
        for p in (self.checkpoint,self.probe_so):
            if not p.is_file():raise FileNotFoundError(p)
        spec=importlib.util.spec_from_file_location('iamu_original_erldde_model',
                                                  self.repo_root/'model.py')
        if spec is None or spec.loader is None:raise ImportError('Original model.py unavailable')
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        self.model=mod.ES().cpu().eval()
        # Only load local, user-trusted checkpoint as a tensor state_dict.
        try:
            weights=torch.load(self.checkpoint,map_location='cpu',weights_only=True)
        except TypeError:
            weights=torch.load(self.checkpoint,map_location='cpu')
        self.model.load_state_dict(weights)
        self.torch=torch;self.max_steps=max_steps
        self.cache={}
    def __call__(self,vessels:list[Vessel],port:Port,etas:Mapping[int,float])->Hint:
        key=tuple((i,round(etas[i],8)) for i in sorted(etas))
        if key in self.cache:return self.cache[key]
        # The original C++ simulator assumes arrivals appear in ascending order.
        order=sorted(vessels,key=lambda v:(etas[v.id],v.id))
        ranks={};positions={}
        with tempfile.TemporaryDirectory(prefix='iamu_erldde_snapshot_') as temp:
            case=Path(temp)/'snapshot.txt'
            with case.open('w',encoding='utf-8') as f:
                for i,v in enumerate(order):
                    vals=[f'{etas[v.id]:.10g}',f'{v.workload:.10g}',
                          f'{v.length:.10g}',str(v.max_cranes)]
                    if i==0:vals.append(str(port.cranes))
                    f.write('\t'.join(vals)+'\n')
            with _ProbeEnv(self.probe_so,case) as sim:
                state=sim.reset();done=False
                for step in range(self.max_steps):
                    with self.torch.no_grad():
                        action=self.model(self.torch.from_numpy(state).float().unsqueeze(0))
                    # Selection index refers to the simulator's *current*
                    # waiting-vessel list, NOT an original dataset row.
                    idx=int(action[0]);selected=sim.action_ship(idx)
                    if selected>=0 and selected<len(order):
                        source_id=order[selected].id
                        if source_id not in ranks:
                            ranks[source_id]=len(ranks)
                            positions[source_id]=max(0.,min(port.quay_length-order[selected].length,
                                                            float(action[1])*port.quay_length))
                    state,done=sim.step(action)
                    if done:break
                if not done:
                    # Partial hints are safe; missing ships are handled by
                    # earliest-ETA order in the new planner.
                    pass
        result=Hint(rank=ranks,position=positions,backend='frozen_erldde')
        self.cache[key]=result
        return result
