"""Pure deterministic validation-aligned collection schedule."""
def protocol_rounds(jobs,round_size,eval_every):
    if round_size<1 or eval_every<1:raise ValueError('Invalid round boundaries')
    out=[];start=0
    while start<len(jobs):
        boundary=min(len(jobs),((start//eval_every)+1)*eval_every)
        end=min(start+round_size,boundary)
        out.append(jobs[start:end]);start=end
    return out
