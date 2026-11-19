#!/usr/bin/env python3
"""Human-readable E4-D diagnosis: mechanism evidence, not causal proof.

Works on a standalone --out folder or a parallel merged/ folder.
Returns a Markdown report and a plain stdout summary. No pandas required.
"""
from __future__ import annotations
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean


def read(path):
    if not path.is_file():
        return []
    with path.open('r',encoding='utf-8',newline='') as f:
        return list(csv.DictReader(f))


def value(x):
    try:return float(x)
    except (ValueError,TypeError):return float('nan')


def format_num(x,n=3):
    return '-' if x is None else f'{x:.{n}f}'


def summarized_mean(a,field):
    x=[value(r.get(field)) for r in a]
    x=[v for v in x if v==v]
    return mean(x) if x else None


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True,help='E4-D root or parallel merged/ folder')
    p.add_argument('--out',default=None,help='Report markdown output path')
    a=p.parse_args(argv)
    root=Path(a.input)
    eps=read(root/'trajectory_episodes.csv')
    summaries=read(root/'trajectory_summary.csv')
    cal=read(root/'calibration_summary.csv')
    decision=read(root/'decision_summary.csv')
    shifts=read(root/'matched_q_shift.csv')
    events=read(root/'trajectory_events.csv')
    cf=read(root/'counterfactual.csv')
    if not eps or not summaries:
        raise SystemExit('Missing E4-D output CSVs: '+str(root))
    modes=sorted(set(r['model'] for r in eps))
    lines=['# E4-D Keep 策略塌缩与 Q-gap 校准诊断结果','',
        '本报告反事实以**同一实现未来**、固定后续策略配对回放为参考；不能视为 Q(s,a) 的精确条件期望真值。',
        '', '## 1. 闭环策略与 Keep 倾向','',
        '| 检查点 | 平均等待 ↓ | 平均改约次数 | 有合法候选时 Keep 占比 | 仅因 Keep margin 而 Keep/轨迹 |',
        '|---|---:|---:|---:|---:|']
    for model in modes:
        r=next((x for x in summaries if x['model']==model and x['scenario']=='ALL' and x['budget']=='ALL'),None)
        if not r:continue
        lines.append('| '+model+' | '+ ' | '.join([
            format_num(value(r['mean_wait'])),
            format_num(value(r['mean_spent'])),
            format_num(value(r['actionable_keep_rate'])),
            format_num(value(r['mean_margin_only_keeps']))])+' |')
    lines+=['','**判断：** 若晚期模型的合法状态 Keep 比例上升、且 Q-gap 低于 Keep margin 的比例明显提高，优先检查值函数塌缩；若主要缺候选，则检查候选掩码及 positive screen。','',
             '## 2. 实现未来上的 Q-gap 误差（同一状态）','',
             '| 检查点 | 状态锚点 | 后续策略 | Edit 配对数 | 归一化 Q-gap MAE ↓ | 方向正确率 ↑ | 预测偏差 |',
             '|---|---|---|---:|---:|---:|---:|']
    for model in modes:
        for anchor in ['early','keep']:
            for cont in ['anchor','greedy','keep']:
                r=next((x for x in cal if x['model']==model and x['anchor']==anchor and
                        x['continuation']==cont and x['scenario']=='ALL' and x['budget']=='ALL'),None)
                if not r:continue
                lines.append('| '+ ' | '.join([model,anchor,cont,
                    str(r['n_pairs']),format_num(value(r['q_gap_mae'])),
                    format_num(value(r['direction_accuracy'])),
                    format_num(value(r['q_gap_signed_bias']))])+' |')
    lines+=['','MAE 单位是模型训练的归一化奖励，若 `reward_scale=100`，乘以 100 才是原始加权等待单位。','',
             '## 3. 固定状态反事实首步后悔值','',
             '| 模型 | 状态锚点 | 后续策略 | 决策状态 | 平均首步后悔 ↓ | CVaR90 后悔 ↓ | 遗漏有利 Edit | 有害 Edit |',
             '|---|---|---|---:|---:|---:|---:|---:|']
    for model in modes:
        for anchor in ['early','keep']:
            for cont in ['anchor','greedy','keep']:
                r=next((x for x in decision if x['model']==model and x['anchor']==anchor and
                        x['continuation']==cont and x['scenario']=='ALL' and x['budget']=='ALL'),None)
                if not r:continue
                lines.append('| '+' | '.join([model,anchor,cont,str(r['n_states']),
                    format_num(value(r['mean_first_action_regret'])),
                    format_num(value(r['cvar90_first_action_regret'])),
                    str(r['missed_beneficial_keep']),str(r['harmful_edit'])])+' |')
    lines+=['','## 4. 相同状态下的 Q-gap 漂移','',
            '| 早期模型 → 对照模型 | 编辑候选数 | 平均 Q-gap 变化 | Edit 由过阈值变为不过阈值次数 | 动作翻转次数（按唯一状态） |',
            '|---|---:|---:|---:|---:|']
    groups=defaultdict(list)
    for r in shifts:
        if int(r['action_id'])==0:continue
        groups[(r['reference_model'],r['comparison_model'])].append(r)
    for (early,late),rr in sorted(groups.items()):
        changed_state=set()
        crossing=0
        for r in rr:
            if r['reference_pass_margin']=='True' and r['comparison_pass_margin']=='False':crossing+=1
            if r['action_flipped']=='True':
                changed_state.add((r['case'],r['scenario'],r['eta_seed'],r['budget'],r['anchor'],r['event_index']))
        lines.append('| '+f'{early} → {late}'+' | '+str(len(rr))+' | '+
          format_num(summarized_mean(rr,'q_gap_shift'))+' | '+str(crossing)+' | '+str(len(changed_state))+' |')
    lines+=['','## 5. 当前数据有效性与科学限制','',
            f'- 轨迹记录：{len(eps)}；有动作的事件记录：{len(events)}；实现未来的配对动作记录：{len(cf)}。',
            '- 仅用已实现的未来计算事后成本：属于固定场景反事实诊断，不是可部署的 oracle 策略。',
            '- 状态银行沿早期策略与 Keep 生成。不同轨迹访问的状态分布不同，跨检查点的行为比较应优先使用相同状态银行。',
            '- 两种后续策略（anchor/greedy）的结果必须分别报告，不可将其混成唯一的 Q 真值。',
            '- 案例 8/9 是已被多次用于选模的验证集；最终机制仍需冻结后在 Case 10–14 检验。',
            '- 若 `matched_q_shift` 的过阈值比例下降同时真实正收益候选仍普遍存在，支持“Keep 价值偏差/动作价值塌缩”解释；否则不要单凭低改约次数认定塌缩。']
    text='\n'.join(lines)+'\n'
    dest=Path(a.out) if a.out else root/'diagnosis_report.md'
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(text,encoding='utf-8')
    print(text)
    print('Saved:',dest)


if __name__=='__main__':main()
