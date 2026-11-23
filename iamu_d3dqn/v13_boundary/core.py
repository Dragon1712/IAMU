"""SMDP value decomposition and legal event-to-macro advancement."""
from dataclasses import dataclass
import math
import statistics

MODES=('primitive','budget_tail','decision_epoch')

@dataclass(frozen=True)
class BranchValue:
    interval_reward:float
    discount:float
    bootstrap_value:float
    suffix_return:float
    done:bool
    n_raw_steps:int
    def __post_init__(self):
        if self.n_raw_steps<1 or not 0<=self.discount<=1:raise ValueError('Invalid macro')
        if self.done and self.discount!=0:raise ValueError('Terminal must have zero discount')
        if not all(math.isfinite(v) for v in (self.interval_reward,self.discount,self.bootstrap_value,self.suffix_return)):
            raise ValueError('Nonfinite value')
    @property
    def td_target(self):return self.interval_reward+self.discount*self.bootstrap_value
    @property
    def full_return(self):return self.interval_reward+self.discount*self.suffix_return
    @property
    def bootstrap_error(self):return self.discount*(self.bootstrap_value-self.suffix_return)

def paired_identity(keep,edit,tol=1e-7):
    td=edit.td_target-keep.td_target
    ret=edit.full_return-keep.full_return
    boot=edit.bootstrap_error-keep.bootstrap_error
    residual=td-ret-boot
    if not math.isclose(residual,0,abs_tol=tol,rel_tol=tol):raise AssertionError(f'Identity residual {residual}')
    return dict(target_gap=td,return_gap=ret,bootstrap_gap=boot,identity_residual=residual,
                keep_bootstrap_error=keep.bootstrap_error,edit_bootstrap_error=edit.bootstrap_error)

def macro_steps(env,first_action,representation,gamma):
    """Execute first action then forced Keep until next representation anchor."""
    if representation not in MODES or not 0<gamma<=1:raise ValueError('mode/gamma')
    if env.done:raise ValueError('terminal snapshot')
    if not env._observation().mask[first_action]:raise ValueError('illegal action')
    reward=0.;power=1.;n=0
    while True:
        nxt,r,done,info=env.step(first_action if n==0 else 0)
        reward+=power*float(r);power*=gamma;n+=1
        if done:break
        if representation=='primitive':break
        if representation=='budget_tail' and int(nxt.remaining_budget)>0:break
        if representation=='decision_epoch' and bool(nxt.mask[1:].any()):break
        if bool(nxt.mask[1:].any()):raise AssertionError('Skipped actionable macro state')
    return dict(reward=reward,discount=0. if done else power,n=n,done=bool(done),
                next_obs=nxt,last_info=info)

def grouped_state_mean(rows,field,group_fields=('model','checkpoint','continuation','source_budget','scenario')):
    groups={}
    for row in rows:
        key=tuple(row.get(k) for k in group_fields)+(row['state_id'],)
        groups.setdefault(key,[]).append(float(row[field]))
    grouped={}
    for key,vals in groups.items():grouped.setdefault(key[:-1],[]).append(statistics.mean(vals))
    return [dict(zip(group_fields,key))|dict(n_states=len(vals),mean=statistics.mean(vals),
                 median=statistics.median(vals),minimum=min(vals),maximum=max(vals))
            for key,vals in sorted(grouped.items(),key=lambda x:str(x[0]))]
