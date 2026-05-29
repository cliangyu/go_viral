#!/usr/bin/env python
"""REASONING test: does letting ckpt-225 GENERATE its CoT (then the head reads the post-CoT hidden)
beat the BYPASS (empty <cot></cot>) val SRCC of 0.514? The SFT assistant is a pure <cot>...</cot>
reasoning trace (gain/hold/lose per segment); the bypass amputates it. If reasoned >> bypass, the
bypass-feature 'ceiling' is an artifact and RL should optimize the GENERATIVE/reasoning path.

For each val ad: (1) generate the assistant (CoT) from video+prompt; (2) re-forward with the
GENERATED cot as the assistant (train-mode encode, like the bypass but real CoT) and read the head's
r_pred; (3) cross-ad SRCC. Compare to bypass (assistant='<cot></cot>')."""
from __future__ import annotations
import argparse, importlib.util, json, os, re
import numpy as np, torch

def import_plugin(p):
    spec=importlib.util.spec_from_file_location('retention_plugin',p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def spearman(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float)
    if len(x)<3: return np.nan
    def rk(a):
        u,inv,c=np.unique(a,return_inverse=True,return_counts=True); cs=np.cumsum(c); st=cs-c; return ((st+cs-1)/2.0)[inv]
    rx,ry=rk(x),rk(y); rx-=rx.mean(); ry-=ry.mean()
    d=np.sqrt((rx*rx).sum()*(ry*ry).sum()); return float((rx*ry).sum()/d) if d>0 else np.nan

def head_curve(model, template, TI, row, assistant):
    """encode row with the given assistant content (train mode), forward, return head r_pred (R(0..Tmax))."""
    r=dict(row); r['messages']=list(row['messages']); r['messages'][-1]={'role':'assistant','content':assistant}
    enc=template.encode(TI.from_dict(r)); batch=template.data_collator([enc])
    batch={k:(v.cuda() if isinstance(v,torch.Tensor) else v) for k,v in batch.items()}
    with torch.no_grad(): out=model(**batch)
    rp=getattr(out,'r_pred',None)
    if rp is None: rp=model._retention_h_holder.r_pred
    return np.concatenate([[1.0], rp[0].float().cpu().numpy()])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',required=True); ap.add_argument('--val-jsonl',required=True); ap.add_argument('--plugin',required=True)
    ap.add_argument('--attn-impl',default='flash_attn'); ap.add_argument('--head-type',default='hazard')
    ap.add_argument('--max-length',type=int,default=32768); ap.add_argument('--max-new',type=int,default=600)
    ap.add_argument('--limit',type=int,default=None); ap.add_argument('--t-lo',type=int,default=1); ap.add_argument('--t-hi',type=int,default=30)
    ap.add_argument('--output',default=None)
    args=ap.parse_args()
    os.environ['RETENTION_HEAD_TYPE']=args.head_type; import_plugin(args.plugin)
    from swift.model import get_model_processor
    from swift.template import get_template
    from swift.template.template_inputs import TemplateInputs
    from swift.template.base import MaxLengthError
    model,proc=get_model_processor(args.checkpoint,torch_dtype=torch.bfloat16,attn_impl=args.attn_impl,
                                   model_kwargs={'device_map':'cuda'},model_type='qwen2_5_omni_retention')
    model.eval()
    tmpl_train=get_template(proc,max_length=args.max_length,template_type='qwen2_5_omni_retention',remove_unused_columns=False)
    tmpl_train.set_mode('train')
    tmpl_gen=get_template(proc,max_length=args.max_length,template_type='qwen2_5_omni_retention',remove_unused_columns=False)
    tmpl_gen.set_mode('pii')   # inference/generation mode (assistant generated)
    print(f'[gen] loaded {args.checkpoint}')
    rows=[json.loads(l) for l in open(args.val_jsonl) if (json.loads(l).get('R') or json.loads(l).get('R_true'))]
    if args.limit: rows=rows[:args.limit]
    preds_gen,preds_byp,trues,Ts,skipped=[],[],[],[],0
    for i,r in enumerate(rows):
        R_true=np.array(r.get('R') or r.get('R_true'),float); T=len(R_true)-1
        if T<5: skipped+=1; continue
        try:
            # 1) generate the CoT
            rg=dict(r); rg['messages']=list(r['messages']); rg['messages'][-1]={'role':'assistant','content':''}
            enc=tmpl_gen.encode(TemplateInputs.from_dict(rg)); batch=tmpl_gen.data_collator([enc])
            batch={k:(v.cuda() if isinstance(v,torch.Tensor) else v) for k,v in batch.items()}
            with torch.no_grad():
                gen=model.generate(**batch,max_new_tokens=args.max_new,do_sample=False,num_beams=1)
            in_len=batch['input_ids'].shape[1]; new=gen[0][in_len:]
            cot=proc.tokenizer.decode(new,skip_special_tokens=True)
            if '<cot>' not in cot: cot='<cot>'+cot
            if '</cot>' not in cot: cot=cot.split('</cot>')[0]+'</cot>' if '</cot>' in cot else cot+'</cot>'
            # 2) head curve from the GENERATED cot, and from the bypass (empty)
            R_gen=head_curve(model,tmpl_train,TemplateInputs,r,cot)
            R_byp=head_curve(model,tmpl_train,TemplateInputs,r,'<cot></cot>')
        except (MaxLengthError,Exception) as e:                                  # noqa
            skipped+=1
            if i<3: print(f'[gen] ad{i} skipped: {type(e).__name__}: {str(e)[:160]}')
            continue
        preds_gen.append(R_gen); preds_byp.append(R_byp); trues.append(R_true); Ts.append(T)
        if i<3: print(f'[gen] ad{i} cot[:120]={cot[:120]!r}')
        if (i+1)%20==0: print(f'[gen] {len(preds_gen)} done, {skipped} skipped')
    def crossad(preds):
        per=[]
        for t in range(args.t_lo,args.t_hi+1):
            pv=[P[t] for P,T in zip(preds,Ts) if T>=t]; tv=[Tr[t] for Tr,T in zip(trues,Ts) if T>=t]
            rho=spearman(pv,tv)
            if not np.isnan(rho): per.append(rho)
        return float(np.mean(per)) if per else float('nan')
    srcc_gen=crossad(preds_gen); srcc_byp=crossad(preds_byp)
    print(f'\n[gen] n={len(preds_gen)} skipped={skipped}')
    print(f'[gen] BYPASS  (empty cot) cross-ad SRCC = {srcc_byp:.4f}')
    print(f'[gen] REASONED(gen cot)   cross-ad SRCC = {srcc_gen:.4f}')
    print(f'[gen] delta (reasoned - bypass) = {srcc_gen-srcc_byp:+.4f}   vs baseline 0.5142')
    if args.output:
        json.dump({'srcc_bypass':srcc_byp,'srcc_reasoned':srcc_gen,'n':len(preds_gen),'skipped':skipped},open(args.output,'w'),indent=2)

if __name__=='__main__': main()
