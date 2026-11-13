import hashlib
import json
import re
import time
import os
import copy
from collections import defaultdict
from pathlib import Path
from .common import digest, public_question, visible, timecode, seconds_from_code, write
from .upstream import prompts, provenance, load_function
from .store import Store, Index
from .media import extract_frames
from .backends import Models, Embeddings

TIME_NOTE = ("Times are Day-relative MP4 playback positions, HHMMSScc (centiseconds), "
             "not recording timestamps or a virtual clock. Do not invent a wall-clock mapping.\n")


class MemoryNotReady(RuntimeError):
    """Eligible videos have not been successfully indexed under this protocol."""


def code_hashes():
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob('*.py'))}


class Pipeline:
    def __init__(self, config, manifest, work, models=None, embeddings=None):
        self.config, self.manifest = copy.deepcopy(config), copy.deepcopy(manifest)
        if type(config.get('top_k',3)) is not int or config.get('top_k',3)<=0:
            raise ValueError('top_k must be a positive integer')
        self.work = Path(work)
        self.work.mkdir(parents=True,exist_ok=True)
        self.prompts = prompts()
        self.models = models or Models(config,self.work/'model_calls.jsonl')
        endpoint_identity = {r:digest(os.environ.get(s.get('base_url_env','EGORAG_BASE_URL'),''))
            for r,s in config.get('models',{}).items() if s.get('backend')=='openai_compatible'}
        self.provenance = {'adapter_code':code_hashes(),'upstream':provenance(),
            'config':config,'prompt_hash':digest(self.prompts),'endpoint_identity':endpoint_identity,
            'local_checkpoint_identity':getattr(self.models,'cache_identity',{})}
        self.namespace = digest(self.provenance)
        self.models.protocol_context = {'namespace':self.namespace,'manifest_hash':digest(manifest)}
        self.store = Store(self.work/'records.sqlite')
        self.embeddings = embeddings or Embeddings(config)
        # Dataset-specific search index; caption/embedding caches remain reusable
        # when a new Day is appended. Old sentence tails cannot leak via expansion.
        self.index_namespace = digest([self.namespace,manifest])
        self.index = Index(self.store,self.index_namespace,self.embeddings.embed,config.get('index_engine','chroma'))
        self.hits = 0
        self.clips = {c['clip_id']:c for c in manifest['clips']}
        if len(self.clips) != len(manifest['clips']):
            raise ValueError('Duplicate clip IDs')
        self.days = {c['day_id']:c['day_order'] for c in manifest['clips']}
        from .media import _state
        for video in manifest.get('videos',[]):
            state = video.get('provenance',{}).get('source_state')
            if state and _state(Path(video['path'])) != state:
                raise ValueError('Source video changed; regenerate manifest')
        write(self.work/'provenance'/f'{self.namespace}.json',self.provenance)

    def call(self, role, system, prompt, frames=None):
        self._check_local_checkpoints()
        frames = frames or []
        identities = [{k:v for k,v in f.items() if k!='jpeg_base64'} for f in frames]
        key = digest([self.namespace,role,system,prompt,identities])
        old = self.store.get('calls',key)
        if old is not None:
            self.hits += 1
            return old['text']
        text = self.models.generate(role,system,prompt,frames=frames)
        if not isinstance(text,str) or not text.strip():
            raise RuntimeError('Empty model response')
        self.store.put('calls',key,{'text':text})
        return text

    def _check_local_checkpoints(self):
        check = getattr(self.models,'assert_checkpoints_unchanged',None)
        if check is not None:
            check()

    def frames(self, clip):
        args = self.config.get('frames',{})
        frames = extract_frames(clip,fps=args.get('fps',1),max_frames=args.get('max_frames',30),
                                max_side=args.get('max_side',512))
        return [{**f,'clip_id':clip['clip_id'],'day_id':clip['day_id']} for f in frames]

    def create(self, cutoff=None, limit=None):
        self._check_local_checkpoints()
        done, reused, errors = [], [], []
        candidates = [c for c in self.manifest['clips'] if cutoff is None or visible(c,cutoff)]
        if limit is not None: candidates = candidates[:limit]
        for clip in candidates:
            self._check_local_checkpoints()
            key = digest([self.namespace,clip])
            cached = self.store.get('captions',key)
            if cached:
                self._index_caption(clip,cached['caption'])
                reused.append(clip['clip_id']); continue
            try:
                frames = self.frames(clip)
                caption = self.call('captioner','You are an egocentric agent. Take yourself as the main character of the video.',
                                    self.prompts['caption'],frames)
                ids = self._index_caption(clip,caption)
                self.store.put('captions',key,{'clip':clip,'caption':caption,'sentence_ids':ids,
                                              'frames':[{k:v for k,v in f.items() if k!='jpeg_base64'} for f in frames]})
                done.append(clip['clip_id'])
            except Exception as exc:
                # Preserve failures separately; no fake caption or negative answer.
                errors.append({'clip_id':clip['clip_id'],'error_type':type(exc).__name__,
                               'error_message':self.models._sanitize(str(exc))[:500]})
            write(self.work/'create_progress.json',{'new':done,'reused':reused,'errors':errors,'total_selected':len(candidates)})
        return {'new':done,'reused':reused,'errors':errors,'total_selected':len(candidates)}

    def _index_caption(self,clip,caption):
        self._check_local_checkpoints()
        # Same sentence-level text representation as upstream video-create.
        sentences = [s.strip()+'.' for s in caption.split('.') if s.strip()]
        if not sentences: raise RuntimeError('No caption sentences')
        ekey = digest([self.namespace,sentences])
        emb = self.store.get('embeddings',ekey)
        if emb is None:
            emb = self.embeddings.embed(sentences)
            self.store.put('embeddings',ekey,emb)
        ids = [f"DAY{clip['day_order']}-{timecode(clip['day_start_seconds'])}-{timecode(clip['day_end_seconds'])}_{i}"
               for i in range(len(sentences))]
        meta = {'date':clip['day_order'],'start_time':int(timecode(clip['day_start_seconds'])),
                'end_time':int(timecode(clip['day_end_seconds'])), 'clip_id':clip['clip_id'],
                'video_path':clip['path'],'start_seconds':clip['day_start_seconds'],
                'end_seconds':clip['day_end_seconds']}
        self.index.collection.upsert(ids=ids,documents=sentences,embeddings=emb,metadatas=[meta]*len(ids))
        self.store.put('indexed',digest([self.index_namespace,clip['clip_id']]),{'ids':ids})
        return ids

    def captions(self, cutoff):
        self._check_local_checkpoints()
        rows = []
        for clip in self.manifest['clips']:
            if visible(clip,cutoff):
                row = self.store.get('captions',digest([self.namespace,clip]))
                if row: rows.append(row)
        return rows

    def summaries(self, cutoff):
        rows = self.captions(cutoff)
        # Every level is rebuilt/cached from an exact causal child set. A full-day
        # summary cache can never stand in for the prefix of that same day.
        source = [{'day_id':r['clip']['day_id'],'day_order':r['clip']['day_order'],
                   'day_start_seconds':r['clip']['day_start_seconds'],'day_end_seconds':r['clip']['day_end_seconds'],
                   'text':r['caption'],'children':[r['clip']['clip_id']]} for r in rows]
        levels = {}
        for name,span,role,prompt_name in [('ten_minute',600,'summary_10min','summary_min'),
                 ('hour',3600,'summary_hour','summary_hour'),('day',None,'summary_day','summary_day')]:
            groups = defaultdict(list)
            for r in source:
                groups[(r['day_order'],int(r['day_start_seconds']//span) if span else 0)].append(r)
            result = []
            for _,children in sorted(groups.items()):
                children.sort(key=lambda x:x['day_start_seconds'])
                text = self.call(role,self.prompts[prompt_name],TIME_NOTE+'All descriptions: '+json.dumps(children))
                record = {'level':name,'day_id':children[0]['day_id'],'day_order':children[0]['day_order'],
                    'day_start_seconds':min(c['day_start_seconds'] for c in children),
                    'day_end_seconds':max(c['day_end_seconds'] for c in children),
                    'text':text,'children':[i for c in children for i in c['children']]}
                if not visible(record,cutoff): raise RuntimeError('Noncausal summary')
                record['summary_id'] = digest([self.namespace,record])
                self.store.put('summaries',record['summary_id'],record)
                result.append(record)
            levels[name] = result
            source = result
        return levels

    def checked_question(self,row):
        q = public_question(row)
        t = q['query_time']
        if self.days.get(t['day_id']) != t['day_order']:
            raise ValueError('query Day ID/order does not match manifest')
        return q

    def query(self, row):
        self._check_local_checkpoints()
        start = time.monotonic()
        q = self.checked_question(row)
        cutoff = q['query_time']
        caps = self.captions(cutoff)
        eligible = [c for c in self.manifest['clips'] if visible(c,cutoff)]
        if len(caps)!=len(eligible) or any(not self.store.get('indexed',digest([self.index_namespace,c['clip_id']])) for c in eligible):
            raise MemoryNotReady('Run create for the eligible videos under the current configuration')
        base = {'question':q,'namespace':self.namespace,'manifest_hash':digest(self.manifest),
                'cutoff':cutoff,'crossing_clip_policy':'exclude_entire_clip',
                'eligible_clip_ids':[r['clip']['clip_id'] for r in caps],
                'test_only':self.config.get('test_only',False)}
        if not caps:
            return {**base,'status':'no_observed_evidence','retrieved':[],'summary_ids':[],
                    'elapsed_seconds':time.monotonic()-start}
        # Supplied annotation keywords are deliberately never read.
        rewrite = self.call('query_rewriter',self.prompts['query_prompt'],TIME_NOTE+
            f"Question: {q['question']}\nQuestion Time: DAY{cutoff['day_order']} {timecode(cutoff['seconds'])}\nOutput:")
        match = re.search(r'Keywords:\s*\[([^\]]+)\]',rewrite)
        if not match: raise ValueError('Invalid keyword output')
        keywords = match.group(1).strip()
        where_causal = {'$or':[{'date':{'$lt':cutoff['day_order']}}, {'$and':[
            {'date':{'$eq':cutoff['day_order']}},{'end_seconds':{'$lte':cutoff['seconds']}}]}]}
        summary_ids = []
        if 'last time' in q['question'].lower() or 'the last' in q['question'].lower():
            # Preserve upstream recency path: keyword/time rewrite -> top50 ->
            # chronological sentence context -> LLM selects most recent evidence.
            match = re.search(r'Search Time Range:\s*\[DAY(\d+)\s+(\d{8})\]',rewrite)
            if not match: raise ValueError('Invalid recency range output')
            proposed = (int(match.group(1)),seconds_from_code(match.group(2)))
            search = min(proposed,(cutoff['day_order'],cutoff['seconds']))
            where = {'$and':[where_causal,{'$or':[{'date':{'$lt':search[0]}},{'$and':[
                {'date':{'$eq':search[0]}},{'end_seconds':{'$lte':search[1]}}]}]}]}
            raw = self.index.custom_query([keywords],n_results=50,where=where,filter_first=False)
            hits = [{k:raw[k][0][i] for k in ['ids','documents','metadatas','distances']}
                    for i in range(len(raw['ids'][0]))]
            hits = self._safe_hits(hits,caps)
            hits.sort(key=lambda h:(h['metadatas']['date'],h['metadatas']['end_time']))
            if hits:
                results = [{'id':h['ids'],'document':self.index.get_caption(h['ids'],1)['documents'],
                            'date':h['metadatas']['date'],'end_time':h['metadatas']['end_time'],
                            'distance':h['distances']} for h in hits]
                selection_wire = []
                def select_call(prompt,system_message,**kw):
                    text = self.call('recency_selector',system_message,prompt)
                    selection_wire.append(text)
                    return text
                selector = load_function('egorag/utils/agent_fun.py','get_ids',{'call_gpt4':select_call})
                selector(q['question'],results)
                # General parser repair: upstream regex captures one suffix digit.
                match = re.fullmatch(r'\s*ID:\s*(DAY\d+-\d{8}-\d{8}_\d+)\s*',selection_wire[0])
                selected = match.group(1) if match else None
                if selected not in {h['ids'] for h in hits}: raise ValueError('Recency selector returned absent ID')
                hits = [h for h in hits if h['ids']==selected]
            selected_range = [search[0],0,search[1]]
        else:
            levels = self.summaries(cutoff)
            def event(r):
                return {'date':r['day_order'],'start_time':int(timecode(r['day_start_seconds'])),
                        'end_time':int(timecode(r['day_end_seconds'])),'generated_text':r['text']}
            hours = [event(r) for r in levels['hour']]
            days = [event(r) for r in levels['day'] if r['day_order']<cutoff['day_order']]
            summary_ids = [r['summary_id'] for level in levels.values() for r in level]
            outer = self
            class SafeRange:
                def filter_event_data(self,date,t,hour_docs,date_docs):
                    current = [r for r in hour_docs if r['date']==cutoff['day_order']]
                    return date_docs+current,current
            def range_call(prompt,system_message,**kwargs):
                response = outer.call('range_selector',system_message,TIME_NOTE+prompt)
                if system_message == outer.prompts['get_day_prompt'].format(question=q['question'],date=f"DAY{cutoff['day_order']}",time=timecode(cutoff['seconds'])):
                    found = re.fullmatch(r'\s*\[(\d+)\]\s*',response)
                    if not found or int(found.group(1)) not in {r['date'] for r in hours}:
                        raise ValueError('Day selector returned unavailable or future day')
                elif not re.fullmatch(r'\s*\[\d{8}\]\s*\n\s*\[\d{8}\]\s*',response):
                    raise ValueError('Invalid hour range output')
                return response
            fn = load_function('egorag/agents/RagAgent.py','get_range',{
                    'call_gpt4':range_call,'get_day_prompt':self.prompts['get_day_prompt'],
                    'get_hour_prompt':self.prompts['get_hour_prompt']},class_name='RagAgent')
            day,begin,end = fn(SafeRange(),q['question'],f"DAY{cutoff['day_order']}",timecode(cutoff['seconds']),hours,days)
            if day > cutoff['day_order']: raise RuntimeError('Future retrieval day')
            end_seconds = seconds_from_code(end)
            if day == cutoff['day_order']: end_seconds = min(end_seconds,cutoff['seconds'])
            begin_seconds = seconds_from_code(begin)
            if begin_seconds > end_seconds: raise ValueError('Reversed retrieval range')
            where = {'$and':[where_causal,{'date':{'$eq':day}},
                {'start_seconds':{'$gte':begin_seconds}},{'end_seconds':{'$lte':end_seconds}}]}
            raw = self.index.custom_query([keywords],n_results=self.config.get('top_k',3),where=where,filter_first=True)
            hits = [{k:raw[k][i] for k in ['ids','documents','metadatas','distances']}
                    for i in range(len(raw['ids']))]
            hits = self._safe_hits(hits,caps)
            selected_range = [day,begin_seconds,end_seconds]
        retrieved = []
        for h in hits:
            c = self.clips[h['metadatas']['clip_id']]
            retrieved.append({'sentence_id':h['ids'],'clip_id':c['clip_id'],'clip':copy.deepcopy(c),
                'document':h['documents'],'distance':float(h['distances']),
                'score_kind':'squared_l2' if 'last time' in q['question'].lower() or 'the last' in q['question'].lower() else 'cosine_distance'})
        return {**base,'status':'complete','keywords':keywords,'rewrite':rewrite,
                'selected_range':selected_range,'summary_ids':summary_ids,'retrieved':retrieved,
                'elapsed_seconds':time.monotonic()-start}

    def _safe_hits(self,hits,caps):
        allowed = {r['clip']['clip_id'] for r in caps}
        for h in hits:
            if h['metadatas']['clip_id'] not in allowed:
                raise RuntimeError('Retrieved stale/future/non-manifest clip')
        return hits

    def answer(self,result):
        self._check_local_checkpoints()
        started = time.monotonic()
        if result.get('status')=='execution_error':
            return {**result,'prediction':None,'used_video_segments':[],
                    'elapsed_seconds':time.monotonic()-started}
        if result['namespace'] != self.namespace or result['manifest_hash'] != digest(self.manifest):
            raise ValueError('Retrieval result belongs to different config/data/code')
        q = self.checked_question(result['question'])
        evidence,used,all_frames,captions = [],[],[],{}
        seen = set()
        for hit in result['retrieved']:
            cid = hit['clip_id']
            if cid in seen: continue
            seen.add(cid)
            clip = self.clips[cid]
            if hit['clip'] != clip or not visible(clip,q['query_time']):
                raise RuntimeError('Noncausal or changed answer video interval')
            frames = self.frames(clip)
            response = self.call('evidence', 'You are an AI assistant that answers questions based on video content.',
                self.prompts['video_evidence_prompt'].format(question=q['question']),frames)
            found = re.search(r'I can provide evidence\. Evidence:\s*(.+)',response,re.S)
            if found: evidence.append({'time':hit['sentence_id'],'evidence_info':found.group(1).strip()})
            elif "I can't provide evidence." not in response:
                raise ValueError('Invalid evidence output')
            captions[cid] = self.index.get_caption(hit['sentence_id'],10)['documents']
            used.append({'clip_id':cid,'path':clip['path'],'file_start_seconds':clip['file_start_seconds'],
                'file_end_seconds':clip['file_end_seconds'],'day_id':clip['day_id'],
                'frames':[{k:v for k,v in f.items() if k!='jpeg_base64'} for f in frames]})
            all_frames.extend(frames)
        if not used:
            prediction,status = None,'no_observed_evidence'
        else:
            options = '\n'.join(f'{chr(65+i)}. {s}' for i,s in enumerate(q['choices']))
            values = {'question':q['question'],'question_time':f"DAY{q['query_time']['day_order']} {timecode(q['query_time']['seconds'])}",
                      'formatted_options':options,'caption_results':captions,
                      'combined_string':'\n\n'.join(f"time: {e['time']}\nevidence_info: {e['evidence_info']}" for e in evidence)}
            template = self.prompts['with_evidence' if evidence else 'without_evidence']
            prompt = TIME_NOTE+template.format(**values)
            if not q['choices']: prompt += '\nNo choices were supplied; answer concisely in natural language instead of a letter.'
            # Explicit visual answer variant: final answerer also receives actual
            # sampled retrieved video frames, not only evidence-card text.
            prediction = self.call('answerer','You are an AI assistant that answers questions based on video content.',prompt,all_frames)
            status = 'complete'
        return {'question':q,'status':status,'prediction':prediction,'evidence_cards':evidence,
                'used_video_segments':used,'namespace':self.namespace,'test_only':self.config.get('test_only',False),
                'retrieval':result,'elapsed_seconds':time.monotonic()-started}

    def storage(self):
        sources = {c['path'] for c in self.manifest['clips']}
        physical = sum(Path(p).stat().st_size for p in sources)
        logical = dict(self.store.db.execute('SELECT kind,SUM(length(payload)) FROM cache GROUP BY kind'))
        vector_bytes = self.store.db.execute('SELECT SUM(length(embedding)) FROM sentences').fetchone()[0] or 0
        return {'source_video_bytes':physical,'source_video_files':len(sources),
                'cache_logical_json_bytes_by_kind':logical,'sqlite_embedding_json_bytes':vector_bytes,
                'database_physical_bytes':self.store.path.stat().st_size,
                'chroma_physical_bytes':sum(p.stat().st_size for p in (self.work/'chroma').rglob('*') if p.is_file()),
                'memory_eviction':False,'source_video_reencoded':False}
