#!/usr/bin/env python3
"""Label Studio JSON inter-annotator agreement, layered and auditable.

Default: submitted annotations only. Predictions NEVER count as independent work.
Token-level kappa compares all lexical tokens, including UNLABELLED, by nesting depth.
Tags [ground]/[air] are ignored; [noise]/[unintelligible] are scored as atomic tokens.
Partial-word selections use lexical-character majority; ties/ambiguous overlaps are
reported as AMBIGUOUS, not silently dropped. Such tokens are NOT scored.
"""
import argparse, csv, json, math, re, sys
from collections import Counter, defaultdict
from pathlib import Path

NONE = 'UNLABELLED'
TAGS = {'ground', 'air'}
TOKEN_RE = re.compile(r'\[[^\]\n]+\]|[^\W_]+(?:[\'’\-][^\W_]+)*', re.UNICODE)

class DataError(Exception): pass

def load(path, source):
    with open(path, encoding='utf-8') as f: tasks=json.load(f)
    if isinstance(tasks, dict) and isinstance(tasks.get('tasks'),list): tasks=tasks['tasks']
    if not isinstance(tasks,list): raise DataError(f'{path}: expected JSON task list')
    out={}
    for task in tasks:
        data=task.get('data') or {}
        text=data.get('text')
        if not isinstance(text,str): raise DataError(f'{path}: task missing data.text')
        key = (('sample_id',str(data['sample_id'])) if data.get('sample_id') else
               ('source_turn',str(data['source_file']),str(data['turn_id']))
               if data.get('source_file') and data.get('turn_id') else ('text',text))
        if key in out: raise DataError(f'{path}: duplicate task identity {key}')
        if source=='submitted':
            candidates=[a for a in task.get('annotations',[]) if not a.get('was_cancelled')]
        else:
            candidates=task.get('drafts',[])
        if len(candidates)!=1:
            raise DataError(f'{path}: {key} has {len(candidates)} {source} entries; require exactly one (predictions are never used)')
        results=candidates[0].get('result',[])
        regions={}
        relations=[]
        for r in results:
            typ=r.get('type')
            if typ=='relation':
                relations.append(r); continue
            if typ!='labels' or r.get('from_name')!='intent': continue
            v=r.get('value') or {}
            labels=v.get('labels')
            if not isinstance(labels,list) or len(labels)!=1 or not isinstance(labels[0],str):
                raise DataError(f'{path}: {key} invalid intent labels {labels}')
            start,end=v.get('start'),v.get('end')
            if not isinstance(start,int) or not isinstance(end,int) or not 0<=start<end<=len(text):
                raise DataError(f'{path}: {key} invalid intent offsets {start}:{end}')
            rid=r.get('id')
            if not isinstance(rid,str) or rid in regions:
                raise DataError(f'{path}: {key} missing/duplicate intent region id {rid}')
            regions[rid]={'id':rid,'start':start,'end':end,'label':labels[0],
                          'parent':r.get('parentID'),'raw_text':text[start:end]}
        # Explicit parentID takes precedence; relation parent_of supplements.
        for rel in relations:
            if 'parent_of' not in rel.get('labels',[]): continue
            parent,child=rel.get('from_id'),rel.get('to_id')
            if child in regions and parent in regions:
                if regions[child]['parent'] not in (None,parent):
                    raise DataError(f'{path}: {key} conflicting parents for {child}')
                regions[child]['parent']=parent
        def depth(rid,seen=None):
            seen=set() if seen is None else seen
            if rid in seen: raise DataError(f'{path}: {key} parent cycle')
            parent=regions[rid]['parent']
            if not parent: return 0
            if parent not in regions: raise DataError(f'{path}: {key} unknown parent {parent}')
            if not (regions[parent]['start']<=regions[rid]['start'] and
                    regions[rid]['end']<=regions[parent]['end']):
                raise DataError(f'{path}: {key} child {rid} not within parent {parent}')
            return 1+depth(parent,seen|{rid})
        for rid,reg in regions.items(): reg['depth']=depth(rid)
        out[key]={'text':text,'regions':regions,'relations':relations}
    return out

def tokens(text):
    for m in TOKEN_RE.finditer(text):
        value=m.group()
        if value.startswith('[') and value.endswith(']') and value[1:-1].lower() in TAGS:
            continue
        yield (m.start(),m.end(),value)

def lexical_positions(text,start,end):
    # Tags like [unintelligible] retain lexical core, ignoring brackets for coverage.
    return {i for i in range(start,end) if text[i].isalnum()}

def assign(tok,regs,text):
    s,e,value=tok
    chars=lexical_positions(text,s,e)
    if not chars: return NONE,''
    coverage=[]
    for r in regs:
        hit=len(chars & set(range(r['start'],r['end'])))
        if hit: coverage.append((hit,r))
    if not coverage: return NONE,''
    top=max(n for n,_ in coverage)
    winners=[r for n,r in coverage if n==top]
    if len(winners)>1: return None,'AMBIGUOUS_MULTIPLE'
    if top*2<=len(chars): return None,'AMBIGUOUS_PARTIAL'
    return winners[0]['label'],('PARTIAL_TOKEN_MAJORITY' if top<len(chars) else '')

def kappa(pairs):
    n=len(pairs)
    if not n:return ('N/A','N/A','N/A',0)
    agree=sum(a==b for a,b in pairs)
    po=agree/n
    ca=Counter(a for a,b in pairs);cb=Counter(b for a,b in pairs)
    pe=sum(ca[k]*cb[k] for k in set(ca)|set(cb))/(n*n)
    k='undefined' if math.isclose(pe,1,abs_tol=1e-12) else f'{(po-pe)/(1-pe):.6f}'
    return f'{po:.6f}',f'{pe:.6f}',k,agree

def span_key(r,trim=False):
    return ((r['start'],r['end']) if not trim else
            (r['start']+len(r['raw_text'])-len(r['raw_text'].lstrip()),
             r['end']-len(r['raw_text'])+len(r['raw_text'].rstrip())),
            r['label'],r['depth'])

def relation_set(task,trim=False):
    regs=task['regions']; result=set()
    for rel in task['relations']:
        u,v=regs.get(rel.get('from_id')),regs.get(rel.get('to_id'))
        if not u or not v: continue
        for label in rel.get('labels',[]):
            result.add((label,span_key(u,trim),span_key(v,trim)))
    return result

def write_csv(path,rows,columns):
    with open(path,'w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore')
        w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser(description='Auditable layered Label Studio JSON IAA')
    ap.add_argument('annotator_a');ap.add_argument('annotator_b')
    ap.add_argument('--output-dir',default='.')
    ap.add_argument('--source',choices=['submitted','draft'],default='submitted',
                    help='Use draft ONLY for explicitly reviewed diagnostic comparison; never independent IAA')
    args=ap.parse_args()
    a=load(args.annotator_a,args.source);b=load(args.annotator_b,args.source)
    if set(a)!=set(b):
        raise DataError(f'Task sets differ: only A={len(set(a)-set(b))}, only B={len(set(b)-set(a))}. No scores produced.')
    reports=[];dis=[];pairs=defaultdict(list);issues=Counter()
    raw_a=set();raw_b=set();trim_a=set();trim_b=set()
    rel_a=set();rel_b=set()
    for key in sorted(a,key=str):
        ta,tb=a[key],b[key]
        if ta['text']!=tb['text']:raise DataError(f'{key}: source transcripts differ')
        text=ta['text']; task_id=':'.join(map(str,key))
        ar=list(ta['regions'].values());br=list(tb['regions'].values())
        for r in ar:
            raw_a.add((task_id,span_key(r)));trim_a.add((task_id,span_key(r,True)))
        for r in br:
            raw_b.add((task_id,span_key(r)));trim_b.add((task_id,span_key(r,True)))
        rel_a.update((task_id,r) for r in relation_set(ta))
        rel_b.update((task_id,r) for r in relation_set(tb))
        for depth in range(max([r['depth'] for r in ar+br],default=0)+1):
            ra=[r for r in ar if r['depth']==depth]
            rb=[r for r in br if r['depth']==depth]
            for start,end,word in tokens(text):
                la,wa=assign((start,end,word),ra,text)
                lb,wb=assign((start,end,word),rb,text)
                layer=f'depth_{depth}'
                if la is None or lb is None:
                    issues['ambiguous_tokens']+=1
                    status='EXCLUDED_AMBIGUOUS'
                else:
                    pairs[layer].append((la,lb))
                    pairs['all_layers'].append((la,lb))
                    status='AGREE' if la==lb else 'DISAGREE'
                if wa or wb: issues['partial_token_flags']+=1
                if status!='AGREE' or wa or wb:
                    dis.append(dict(task=task_id,layer=layer,start=start,end=end,text=word,
                                    annotator_a=la if la is not None else 'AMBIGUOUS',
                                    annotator_b=lb if lb is not None else 'AMBIGUOUS',
                                    status=status,flag_a=wa,flag_b=wb))
    def metric(name,value):reports.append({'metric':name,'value':value})
    def jaccard(x,y):return f'{len(x&y)/len(x|y):.6f}' if x|y else 'N/A'
    metric('comparison_source',args.source)
    metric('independence_status','NOT_VERIFIED' if args.source=='submitted' else 'NOT_INDEPENDENT_DRAFT_DIAGNOSTIC')
    metric('shared_tasks',len(a))
    metric('raw_exact_span_label_jaccard',jaccard(raw_a,raw_b))
    metric('whitespace_normalized_span_label_jaccard',jaccard(trim_a,trim_b))
    metric('annotator_a_intent_regions',len(raw_a))
    metric('annotator_b_intent_regions',len(raw_b))
    metric('ambiguous_excluded_token_layer_units',issues['ambiguous_tokens'])
    metric('partial_token_flags',issues['partial_token_flags'])
    for layer in ['all_layers']+sorted(k for k in pairs if k!='all_layers'):
        po,pe,k,agree=kappa(pairs[layer])
        metric(f'{layer}_scored_units',len(pairs[layer]))
        metric(f'{layer}_agree_units',agree)
        metric(f'{layer}_observed_agreement',po)
        metric(f'{layer}_chance_agreement',pe)
        metric(f'{layer}_cohens_kappa',k)
    # Relations are compared separately, by endpoint span/label/layer rather than local region IDs.
    metric('relations_a',len(rel_a));metric('relations_b',len(rel_b))
    metric('relation_exact_jaccard',jaccard(rel_a,rel_b))
    metric('relation_precision_b_vs_a',f'{len(rel_a&rel_b)/len(rel_b):.6f}' if rel_b else 'N/A')
    metric('relation_recall_b_vs_a',f'{len(rel_a&rel_b)/len(rel_a):.6f}' if rel_a else 'N/A')
    for side,items,other in [('A',rel_a,rel_b),('B',rel_b,rel_a)]:
        for task_id,(name,src,dst) in sorted(items-other,key=str):
            dis.append(dict(task=task_id,layer='relations',text=name,
                            annotator_a=str(src)+' -> '+str(dst) if side=='A' else '',
                            annotator_b=str(src)+' -> '+str(dst) if side=='B' else '',
                            status=f'RELATION_ONLY_{side}'))
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'agreement_report.csv',reports,['metric','value'])
    write_csv(out/'disagreements.csv',dis,['task','layer','start','end','text','annotator_a','annotator_b','status','flag_a','flag_b'])
    for r in reports:print(f"{r['metric']}: {r['value']}")
    print(f'Files: {out/"agreement_report.csv"}, {out/"disagreements.csv"}')
    if issues['ambiguous_tokens']:
        print('WARNING: ambiguous token-layer units excluded; interpret scores with exclusion counts.',file=sys.stderr)
    if args.source=='draft':
        print('WARNING: draft comparison is diagnostic only, NOT independent IAA.',file=sys.stderr)
    return 0

if __name__=='__main__':
    try:sys.exit(main())
    except (DataError,OSError,ValueError,json.JSONDecodeError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr);sys.exit(1)
