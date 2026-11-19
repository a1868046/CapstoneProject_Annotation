#!/usr/bin/env python3
"""ATC Label Studio independent intent IAA, version 3 (standard library only).
Offsets remain untouched. Default uses one submitted annotation per sample.
Kappa is per layer, on the same full set of lexical tokens in every shared task.
"""
import argparse, csv, json, math, re, sys
from collections import Counter, defaultdict
from pathlib import Path

NONE="UNLABELLED"
# A paired wrapper keeps its contents; a comment removes all enclosed contents.
COMMENT=re.compile(r'\[comment_[^\]]*\].*?\[\|_comment\]',re.I|re.S)
WRAPPER=re.compile(r'\[(?:air|ground|noise)[_|][^\]]*\]|\[\|(?:air|ground|noise)\]',re.I)
EVENT=re.compile(r'\[(?!\|)[^\]\n]+\]')
WORD=re.compile(r'[^\W_]+(?:[\'’\-][^\W_]+)*',re.U)

class DataError(Exception): pass

def masked_tokens(text):
    mask=[True]*len(text)
    for m in COMMENT.finditer(text):
        for i in range(*m.span()):mask[i]=False
    for m in WRAPPER.finditer(text):
        for i in range(*m.span()):mask[i]=False
    # Malformed comment delimiters must not accidentally score comment text.
    if re.search(r'\[comment_',text,re.I) and not COMMENT.search(text):
        raise DataError("Unclosed or malformed comment wrapper")
    out=[];i=0
    while i<len(text):
        if not mask[i]:i+=1;continue
        ev=EVENT.match(text,i)
        if ev and all(mask[j] for j in range(i,ev.end())):
            out.append((i,ev.end(),ev.group(),'event'));i=ev.end();continue
        # (Praha(Prág)) and (9(najn)): score only the outer written form.
        if text[i]=='(':
            m=re.match(r'\(([^()\s]+)\([^()]*\)\)',text[i:])
            if m:
                base=m.group(1); start=i+1
                out.append((start,start+len(base),base,'word'))
                i+=m.end();continue
        m=WORD.match(text,i)
        if m and all(mask[j] for j in range(i,m.end())):
            out.append((i,m.end(),m.group(),'word'));i=m.end();continue
        i+=1
    return out

def stable_key(data):
    if data.get('sample_id'):return str(data['sample_id'])
    if data.get('source_file') and data.get('turn_id'):
        return str(data['source_file'])+'::'+str(data['turn_id'])
    if data.get('turn_id'):return str(data['turn_id'])
    raise DataError('Task missing stable data.sample_id or data.turn_id (text fallback disabled)')

def utf16_slice(text,s,e):
    # Source can explicitly state UTF-16 offsets; Python strings use code points.
    raw=text.encode('utf-16-le')
    return raw[2*s:2*e].decode('utf-16-le')

def load(path,mode):
    tasks=json.loads(Path(path).read_text(encoding='utf-8'))
    if isinstance(tasks,dict):tasks=tasks.get('tasks')
    if not isinstance(tasks,list):raise DataError(f'{path}: expected task list')
    output={}
    for task in tasks:
        data=task.get('data') or {};text=data.get('text')
        if not isinstance(text,str):raise DataError(f'{path}: missing data.text')
        key=stable_key(data)
        if key in output:raise DataError(f'{path}: duplicate key {key}')
        entries=[x for x in task.get('annotations' if mode=='submitted' else 'drafts',[])
                 if mode=='draft' or not x.get('was_cancelled')]
        if len(entries)!=1:raise DataError(f'{path}: {key}: expected exactly one {mode} entry, got {len(entries)}')
        regs={};rawrels=[];problems=[]
        for r in entries[0].get('result',[]):
            typ=r.get('type')
            if typ=='relation':rawrels.append(r);continue
            if typ!='labels' or r.get('from_name')!='intent':continue
            v=r.get('value') or {};s=v.get('start');e=v.get('end');labs=v.get('labels')
            if not isinstance(s,int) or not isinstance(e,int) or not 0<=s<e:
                raise DataError(f'{key}: invalid offsets {s},{e}')
            if not isinstance(labs,list) or len(labs)!=1:
                raise DataError(f'{key}: intent must have one label')
            # Detect UTF-16 source offsets, and convert only in-memory coordinates.
            offset_unit=data.get('offset_unit','')
            if 'UTF-16' in offset_unit:
                try:
                    prefix=utf16_slice(text,0,s);selected=utf16_slice(text,s,e)
                    ps=len(prefix);pe=ps+len(selected)
                except UnicodeError:raise DataError(f'{key}: UTF-16 offset splits surrogate')
            else:
                ps,pe=s,e
                if e>len(text):raise DataError(f'{key}: offset out of bounds')
                selected=text[s:e]
            saved=v.get('text')
            if isinstance(saved,str) and selected.strip()!=saved.strip():
                raise DataError(f'{key}: selection text mismatch at {s}:{e}: {saved!r} vs {selected!r}')
            rid=r.get('id')
            if not rid or rid in regs:raise DataError(f'{key}: missing/duplicate intent ID {rid}')
            regs[rid]=dict(id=rid,start=ps,end=pe,label=labs[0],parent=r.get('parentID'),
                           original_start=s,original_end=e,selected=selected)
        # parentID and explicit parent_of are equivalent, but conflicting claims flagged.
        claims=defaultdict(set)
        for rid,r in regs.items():
            if r['parent']:claims[rid].add(r['parent'])
        edges=[]
        for rel in rawrels:
            src,dst=rel.get('from_id'),rel.get('to_id')
            labs=rel.get('labels') or []
            direction=rel.get('direction')
            if src not in regs or dst not in regs:
                problems.append(('INVALID_RELATION_ENDPOINT',str(src),str(dst),str(labs)));continue
            if direction not in ('right','left'):
                problems.append(('INVALID_RELATION_DIRECTION',str(src),str(dst),str(direction)));continue
            # Relation semantics: arrow direction determines from/to, even if fields are reversed.
            if direction=='left':src,dst=dst,src
            for lab in labs:
                if lab=='parent_of':claims[dst].add(src)
                else:edges.append((lab,src,dst))
        for child,parents in claims.items():
            if len(parents)>1:
                problems.append(('CONFLICTING_PARENTS',child,'|'.join(sorted(parents)),''))
            if any(parent not in regs for parent in parents):
                problems.append(('MISSING_PARENT_ENDPOINT',child,'|'.join(sorted(parents)),''))
            if len(parents)==1 and next(iter(parents)) in regs:
                regs[child]['parent']=next(iter(parents))
                edges.append(('parent_of',regs[child]['parent'],child))
        def depth(rid,seen=frozenset()):
            if rid in seen:raise DataError(f'{key}: parent cycle at {rid}')
            parent=regs[rid]['parent']
            if not parent or parent not in regs:return 0
            if not (regs[parent]['start']<=regs[rid]['start'] and
                    regs[rid]['end']<=regs[parent]['end']):
                problems.append(('CHILD_OUTSIDE_PARENT',parent,rid,''))
            return 1+depth(parent,seen|{rid})
        for rid,r in regs.items():r['depth']=depth(rid)
        output[key]=dict(text=text,regions=regs,edges=set(edges),problems=problems)
    return output

def normalize_span(r,whitespace=False):
    s,e=r['start'],r['end']
    if whitespace:
        raw=r['selected'];s+=len(raw)-len(raw.lstrip());e-=len(raw)-len(raw.rstrip())
    return (s,e,r['label'],r['depth'])

def labels_for_token(token,regs,text):
    s,e,word,kind=token
    # Event markers are atomic; selected inner text counts as covering event.
    indices={i for i in range(s,e) if text[i].isalnum()}
    if not indices:indices=set(range(s,e))
    matches=[]
    for r in regs:
        n=sum(r['start']<=i<r['end'] for i in indices)
        if n:matches.append((n,r))
    if not matches:return NONE,[],False
    # Never silently resolve overlapping different labels by majority coverage.
    conflicting=len({r['label'] for _,r in matches})>1
    best=max(n for n,_ in matches)
    winners=[r for n,r in matches if n==best]
    if conflicting or len(winners)!=1 or 2*best<=len(indices):
        return None,[r['label'] for _,r in matches],conflicting
    return winners[0]['label'],[r['label'] for _,r in matches],False

def kap(pairs):
    n=len(pairs)
    if not n:return ('N/A','N/A','undefined',0)
    agree=sum(a==b for a,b in pairs);po=agree/n
    a=Counter(x for x,y in pairs);b=Counter(y for x,y in pairs)
    pe=sum(a[k]*b[k] for k in a.keys()|b.keys())/n**2
    return f'{po:.6f}',f'{pe:.6f}',('undefined' if math.isclose(pe,1,abs_tol=1e-12)
                                     else f'{(po-pe)/(1-pe):.6f}'),agree

def jacc(a,b):
    return f'{len(a&b)/len(a|b):.6f}' if a|b else 'N/A'

def csvwrite(path,rows,fields):
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('annotator_a');parser.add_argument('annotator_b')
    parser.add_argument('--output-dir',default='.')
    parser.add_argument('--source',choices=['submitted','draft'],default='submitted')
    args=parser.parse_args()
    a=load(args.annotator_a,args.source);b=load(args.annotator_b,args.source)
    if set(a)!=set(b):raise DataError(f'Nonmatching task IDs: A only {sorted(set(a)-set(b))[:5]}; B only {sorted(set(b)-set(a))[:5]}')
    if any(a[k]['text']!=b[k]['text'] for k in a):raise DataError('Matched task has different transcript text')
    report=[];dis=[];checks=[];scores=defaultdict(list);exclusions=Counter()
    sa=set();sb=set();sna=set();snb=set();ea=set();eb=set()
    maxdepth=max((r['depth'] for task in list(a.values())+list(b.values())
                  for r in task['regions'].values()),default=0)
    def edgekeys(task,key):
        regs=task['regions'];out=set()
        for lab,src,dst in task['edges']:
            if src in regs and dst in regs:
                out.add((key,lab,normalize_span(regs[src]),normalize_span(regs[dst])))
        return out
    for key in sorted(a):
        ta,tb=a[key],b[key];text=ta['text']
        for side,task in [('A',ta),('B',tb)]:
            for code,x,y,extra in task['problems']:
                checks.append(dict(task=key,annotator=side,issue=code,source=x,target=y,detail=extra))
        for r in ta['regions'].values():
            sa.add((key,normalize_span(r)));sna.add((key,normalize_span(r,True)))
        for r in tb['regions'].values():
            sb.add((key,normalize_span(r)));snb.add((key,normalize_span(r,True)))
        ea.update(edgekeys(ta,key));eb.update(edgekeys(tb,key))
        for token in masked_tokens(text):
            s,e,word,kind=token
            for depth in range(maxdepth+1):
                ra=[r for r in ta['regions'].values() if r['depth']==depth]
                rb=[r for r in tb['regions'].values() if r['depth']==depth]
                la,all_a,conf_a=labels_for_token(token,ra,text)
                lb,all_b,conf_b=labels_for_token(token,rb,text)
                if conf_a or conf_b:
                    checks.append(dict(task=key,annotator='A' if conf_a else 'B',issue='SAME_LAYER_CONFLICT',
                                       source=str(s),target=str(e),detail=f'{word}: A={all_a}; B={all_b}'))
                for mode in ('main','supplementary'):
                    if mode=='main' and kind=='event':continue
                    status='AGREE' if la==lb else 'DISAGREE'
                    if la is None or lb is None:
                        status='EXCLUDED_AMBIGUOUS';exclusions[(mode,depth)]+=1
                    else:scores[(mode,depth)].append((la,lb))
                    if status!='AGREE':
                        dis.append(dict(task=key,score=mode,layer=depth,start=s,end=e,
                                        text=word,annotator_a=la,annotator_b=lb,status=status))
    def add(k,v):report.append(dict(metric=k,value=v))
    add('source',args.source)
    add('independence','NOT_VERIFIED' if args.source=='submitted' else 'DRAFT_DIAGNOSTIC_NOT_INDEPENDENT')
    add('shared_tasks',len(a))
    add('raw_span_jaccard',jacc(sa,sb));add('whitespace_normalized_span_jaccard',jacc(sna,snb))
    for mode in ('main','supplementary'):
        for depth in range(maxdepth+1):
            pairs=scores[(mode,depth)];po,pe,k,agree=kap(pairs)
            pre=f'{mode}_layer_{depth}'
            for field,value in [('scored_tokens',len(pairs)),('agreed_tokens',agree),
                                ('excluded_ambiguous',exclusions[(mode,depth)]),
                                ('observed_agreement',po),('chance_agreement',pe),('cohens_kappa',k)]:
                add(f'{pre}_{field}',value)
    add('relation_links_a',len(ea));add('relation_links_b',len(eb))
    add('relation_jaccard',jacc(ea,eb))
    for side,missing in [('A',ea-eb),('B',eb-ea)]:
        for key,lab,src,dst in sorted(missing,key=str):
            checks.append(dict(task=key,annotator=side,issue='RELATION_ONLY_'+side,
                               source=str(src),target=str(dst),detail=lab))
    add('validation_flags',len(checks))
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    csvwrite(out/'agreement_report.csv',report,['metric','value'])
    csvwrite(out/'disagreements.csv',dis,['task','score','layer','start','end','text','annotator_a','annotator_b','status'])
    csvwrite(out/'validation_flags.csv',checks,['task','annotator','issue','source','target','detail'])
    for r in report:print(f"{r['metric']}: {r['value']}")
    print(f'Wrote reports to {out}')
    if checks:print('WARNING: review validation_flags.csv before interpreting agreement.',file=sys.stderr)
    if args.source=='draft':print('WARNING: drafts are not independent IAA.',file=sys.stderr)

if __name__=='__main__':
    try:main()
    except (DataError,ValueError,OSError,json.JSONDecodeError) as exc:
        print('ERROR:',exc,file=sys.stderr);sys.exit(1)
