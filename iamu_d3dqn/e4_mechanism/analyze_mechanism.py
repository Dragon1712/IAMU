#!/usr/bin/env python3
"""E4-E readable diagnostic comparing new online and frozen replay branches."""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
from statistics import mean


def read(path):
    with Path(path).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))

def f(r,key):
    x=r.get(key)
    return float(x) if x not in (None,'','None') else None

def pretty(v):return '—' if v is None or not math.isfinite(v) else f'{v:.3f}'

def match(rows,ep):
    z=[x for x in rows if int(x['episode'])==ep]
    return z[0] if z else None

def tderror(rows,ep,subset,field='mean_delta',probe=None):
    z=[r for r in rows if int(r['episode'])==ep and r['subset']==subset
       and (probe is None or r['probe']==probe)]
    return f(z[0],field) if z else None

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--online',required=True)
    p.add_argument('--offline',nargs='+',required=True)
    p.add_argument('--eval',default=None)
    p.add_argument('--episodes',default='200,250,300,350,400')
    p.add_argument('--out',default='results/v10_e5_mechanism_report')
    a=p.parse_args()
    online=Path(a.online);off=[Path(x) for x in a.offline]
    out=Path(a.out)
    if out.exists() and any(out.iterdir()):p.error('Use a NEW --out')
    out.mkdir(parents=True,exist_ok=True)
    checkpoints=list(map(int,a.episodes.split(',')))
    sources={'online':dict(q=read(online/'e5_mechanism/qgap_trace.csv'),
        replay=read(online/'e5_mechanism/replay_trace.csv'),
        td=read(online/'e5_mechanism/td_trace.csv'))}
    hashes=set()
    for folder in off:
        meta=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        mode=meta['args']['mode']
        if mode in sources:raise ValueError('Duplicate mode: '+mode)
        hashes.add(meta['source_fork_sha256'])
        sources[mode]=dict(q=read(folder/'qgap_probe.csv'),
            replay=read(folder/'replay_check.csv'),td=read(folder/'td_probe.csv'))
    if len(hashes)!=1:raise ValueError('Not all offline branches originate from SAME Episode200 fork')
    text=['# E4-E 固定 Replay 训练机制实验分析','',
          '> 在线与离线曲线必须都从相同 Episode 200 **完整**训练状态分叉。'
          ' 离线训练不新增数据，按在线真实梯度更新数量推进。','',
          f'- 在线实验：`{online}`','- 分叉 SHA256：`'+next(iter(hashes))+'`','',
          '## 1. 固定状态 Q-gap 与 Keep 倾向','',
          '| 模式 | Episode | 平均最佳 Q-gap | 有合法 Edit 时预测 Keep 比例 | Mean Q_keep |',
          '|---|---:|---:|---:|---:|']
    for mode,files in sources.items():
        for ep in checkpoints:
            r=match(files['q'],ep)
            if r:text.append(f'| {mode} | {ep} | {pretty(f(r,"mean_best_q_gap"))} | '
                     f'{pretty(f(r,"fraction_keep_with_legal"))} | {pretty(f(r,"mean_q_keep"))} |')
    text+=['','## 2. Replay 数据组成与动作条件分布','',
       '| 模式 | Episode | Replay N | 有合法 Edit 时的 Keep 比例 | 被迫 Keep 比例 | 最近 20% Keep 比例 |',
       '|---|---:|---:|---:|---:|---:|']
    for mode,files in sources.items():
        for ep in checkpoints:
            r=match(files['replay'],ep)
            if r:text.append(f'| {mode} | {ep} | {r["n"]} | '
               f'{pretty(f(r,"keep_fraction_actionable"))} | '
               f'{pretty(f(r,"forced_keep_fraction"))} | '
               f'{pretty(f(r,"newest20_keep_actionable_fraction"))} |')
    text+=['','## 3. 固定 EP200 Replay TD Probe：分动作 TD 残差','',
       'Delta = reward + gamma * target_Q(next, online_argmax_with_margin) - online_Q(current, action)',
       '', '| 模式 | Episode | Keep(可选择) mean Delta | Edit mean Delta | Edit MAE |',
       '|---|---:|---:|---:|---:|']
    for mode,files in sources.items():
        for ep in checkpoints:
            probe='fork_fixed' if mode=='online' else 'ep200_fixed'
            ds=files['td']
            text.append(f'| {mode} | {ep} | '
                f'{pretty(tderror(ds,ep,"choice_keep",probe=probe))} | '
                f'{pretty(tderror(ds,ep,"edit",probe=probe))} | '
                f'{pretty(tderror(ds,ep,"edit",field="mae",probe=probe))} |')
    if a.eval:
        rows=read(Path(a.eval)/'runs.csv')
        text+=['','## 4. 原 Case 8/9 同条件闭环评价','',
              '| 模型 | Episode | 平均等待 ↓ | 平均改约 | 相对 Keep 改善 |',
              '|---|---:|---:|---:|---:|']
        keys=sorted(set((x['model'],x['episode']) for x in rows))
        for model,ep in keys:
            z=[x for x in rows if x['model']==model and x['episode']==ep]
            text.append(f'| {model} | {ep} | '
              f'{pretty(mean(f(x,"weighted_wait") for x in z))} | '
              f'{pretty(mean(f(x,"spent") for x in z))} | '
              f'{pretty(mean(f(x,"gain_vs_keep") for x in z))} |')
    text+=['','## 5. 结论应遵循的判断规则','',
       '- 若冻结 Replay 的 **uniform** 分支仍然出现大幅 Q-gap 下移，说明在**不增加训练转移数据**条件下，优化/自举过程足以产生该现象；不代表证明了唯一原因。',
       '- 若 **uniform** 塌缩而 **fixed_target** 稳定，Target Network 的更新是有力的候选机制；仍需检查目标同步频率的独立消融。',
       '- 若 **balanced** 明显优于 **uniform**，动作采样比例是可改变结果的因素；采样带替换和经验复用的副作用也必须报告。',
       '- 若只有 online 出现塌缩、offline 保持稳定，应优先排查新经验分布变化、actor 训练策略漂移和 off-policy 采样偏差。',
       '- TD 残差的大小不等于 Bellman 目标的无偏误差；应分别报告 selected action=Edit、可选择 Keep、forced Keep 的样本数量。',
       '- 不应依据 Case 8/9 的机制诊断反复修改超参数后还声称其是独立泛化测试。','']
    (out/'report.md').write_text('\n'.join(text),encoding='utf-8')
    (out/'manifest.json').write_text(json.dumps(dict(args=vars(a),fork_sha256=next(iter(hashes)),
        notes='Conditional diagnostics and exploratory validation, not a proof of a unique root cause'),indent=2))
    print('[DONE]',out/'report.md')

if __name__=='__main__':main()
