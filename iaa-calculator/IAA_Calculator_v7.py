#!/usr/bin/env python3
"""ATC Label Studio independent intent IAA, version 7 (standard library only).
Offsets remain untouched. Default uses one submitted annotation per sample.
One kappa over matched label pairs for lexical tokens and event markers.
"""
import argparse, csv, json, math, re, sys
from collections import Counter, defaultdict
from pathlib import Path

NONE="UNLABELLED"
# A paired wrapper keeps its contents; a comment removes all enclosed contents.
COMMENT=re.compile(r'\[comment_[^\]]*\].*?\[\|_comment\]',re.I|re.S)
# Ignore speaker/format markers and wrapper delimiters, not wrapped speech.
WRAPPER=re.compile(r'\[(?:air|ground|speaker|no_eng)\]|\[[^\]\n]+_\|?\]|\[\|?_[^\]\n]+\]',re.I)
EVENT=re.compile(r'\[(?!\|)[^\]\n]+\]')
WORD=re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*",re.U)
PRONUNCIATION=re.compile(r'\(([^()]+?)\s*(\([^()]*\))\s*\)')

class DataError(Exception): pass

def masked_tokens(text):
    mask=[True]*len(text)
    def hide(start,end):
        mask[start:end]=[False]*(end-start)
    for m in COMMENT.finditer(text):hide(*m.span())
    for m in WRAPPER.finditer(text):hide(*m.span())
    for m in PRONUNCIATION.finditer(text):hide(*m.span(2))
    if re.search(r'\[comment_',text,re.I) and not COMMENT.search(text):
        raise DataError("Unclosed or malformed comment wrapper")
    out=[];i=0
    while i<len(text):
        if not mask[i]:i+=1;continue
        ev=EVENT.match(text,i)
        if ev and all(mask[i:ev.end()]):
            out.append((i,ev.end(),ev.group(),'event'));i=ev.end();continue
        m=WORD.match(text,i)
        if m and all(mask[i:m.end()]):
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

def selection_text_matches(selected, saved):
    # Exported selection text may contain literal backslash-n sequences.
    # Never modify transcript text, annotation offsets, or scored spans.
    if not isinstance(saved, str):
        return True
    def canonical(value):
        value = value.replace(chr(92) + 'r' + chr(92) + 'n', '\n')
        value = value.replace(chr(92) + 'n', '\n')
        value = value.replace(chr(92) + 't', '\t')
        return ' '.join(value.split())
    return canonical(selected) == canonical(saved)

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
            if not selection_text_matches(selected, saved):
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

def labels_for_token(token, regs, text):
    # Preserve all distinct labels touching the token, regardless of depth.
    start,end,_,_=token
    indices={i for i in range(start,end) if text[i].isalnum()}
    if not indices: indices=set(range(start,end))
    return {r['label'] for r in regs if any(r['start']<=i<r['end'] for i in indices)}

def pair_labels(a,b):
    shared=sorted(a & b)
    left,right=sorted(a-b),sorted(b-a)
    n=max(len(left),len(right))
    pairs=[(x,x) for x in shared]+list(zip(left+[NONE]*(n-len(left)),right+[NONE]*(n-len(right))))
    return pairs or [(NONE,NONE)]

def kap(pairs):
    n=len(pairs)
    if not n:return ('N/A','N/A','undefined',0)
    agree=sum(a==b for a,b in pairs);po=agree/n
    a=Counter(x for x,y in pairs);b=Counter(y for x,y in pairs)
    pe=sum(a[k]*b[k] for k in a.keys()|b.keys())/n**2
    return f'{po:.6f}',f'{pe:.6f}',('undefined' if math.isclose(pe,1,abs_tol=1e-12)
                                     else f'{(po-pe)/(1-pe):.6f}'),agree

def csvwrite(path,rows,fields):
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('annotator_a');parser.add_argument('annotator_b')
    parser.add_argument('--output-dir',default='.')
    parser.add_argument('--source',choices=['submitted','draft'],default='submitted')
    args=parser.parse_args()
    a=load(args.annotator_a,args.source)
    b=load(args.annotator_b,args.source)

    shared = set(a) & set(b)
    print(f"Annotator A transcripts: {len(a)}")
    print(f"Annotator B transcripts: {len(b)}")
    print(f"Matching transcripts: {len(shared)}")

    if not shared:
        raise DataError("No matching transcripts found")

    a = {k: a[k] for k in shared}
    b = {k: b[k] for k in shared}

    if any(a[k]['text'] != b[k]['text'] for k in a):
        raise DataError("Matched task has different transcript text")

    report=[];dis=[];checks=[];pairs=[];boundaries=[];relations=[]
    token_count=event_count=0
    def span(r):
        return normalize_span(r,True)[:3]
    def edgekeys(task):
        regs=task['regions']
        return {(lab,span(regs[src]),span(regs[dst])) for lab,src,dst in task['edges']}
    for key in sorted(a):
        ta,tb=a[key],b[key];text=ta['text']
        for side,task in [('A',ta),('B',tb)]:
            for code,x,y,extra in task['problems']:
                checks.append(dict(task=key,annotator=side,issue=code,source=x,target=y,detail=extra))
        ra=list(ta['regions'].values());rb=list(tb['regions'].values())
        sa={span(r) for r in ra};sb={span(r) for r in rb}
        for side,missing in [('A',sa-sb),('B',sb-sa)]:
            for start,end,label in sorted(missing):
                boundaries.append(dict(task=key,annotator=side,start=start,end=end,label=label,text=text[start:end],status='SPAN_ONLY_'+side))
        ea,eb=edgekeys(ta),edgekeys(tb)
        for side,missing in [('A',ea-eb),('B',eb-ea)]:
            for lab,src,dst in sorted(missing):
                relations.append(dict(task=key,annotator=side,relation=lab,source=str(src),target=str(dst),status='RELATION_ONLY_'+side))
        for token in masked_tokens(text):
            start,end,word,kind=token
            token_count+=1;event_count+=kind=='event'
            la=labels_for_token(token,ra,text);lb=labels_for_token(token,rb,text)
            for x,y in pair_labels(la,lb):
                pairs.append((x,y))
                if x!=y:
                    dis.append(dict(task=key,start=start,end=end,text=word,kind=kind,annotator_a=x,annotator_b=y,status='DISAGREE'))
    def add(k,v):report.append(dict(metric=k,value=v))
    po,pe,k,agree=kap(pairs)
    add('source',args.source)
    add('independence','NOT_VERIFIED' if args.source=='submitted' else 'DRAFT_DIAGNOSTIC_NOT_INDEPENDENT')
    add('shared_tasks',len(a))
    add('scored_tokens',token_count);add('event_tokens',event_count)
    add('scored_label_pairs',len(pairs));add('agreed_label_pairs',agree)
    add('observed_agreement',po);add('chance_agreement',pe);add('cohens_kappa',k)
    add('validation_flags',len(checks))
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    csvwrite(out/'agreement_report.csv',report,['metric','value'])
    csvwrite(out/'disagreements.csv',dis,['task','start','end','text','kind','annotator_a','annotator_b','status'])
    csvwrite(out/'validation_flags.csv',checks,['task','annotator','issue','source','target','detail'])
    csvwrite(out/'boundary_differences.csv',boundaries,['task','annotator','start','end','label','text','status'])
    csvwrite(out/'relationship_differences.csv',relations,['task','annotator','relation','source','target','status'])
    for r in report:print(f"{r['metric']}: {r['value']}")
    print(f'Wrote reports to {out}')
    if checks:print('WARNING: review validation_flags.csv before interpreting agreement.',file=sys.stderr)
    if args.source=='draft':print('WARNING: drafts are not independent IAA.',file=sys.stderr)

if __name__=='__main__':
    try:main()
    except (DataError,ValueError,OSError,json.JSONDecodeError) as exc:
        print('ERROR:',exc,file=sys.stderr);sys.exit(1)
