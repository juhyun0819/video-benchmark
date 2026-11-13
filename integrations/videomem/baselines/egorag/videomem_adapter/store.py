"""Append-only cached records + the upstream Chroma cosine search implementation.

SQLite stores sentences/embeddings portably; no memory eviction. The search
method is the actual unmodified upstream Chroma.custom_query(filter_first=True).
This storage substitution is explicitly reported; it is not Chroma ANN.
"""
import json
import sqlite3
from pathlib import Path
from typing import List, Optional, Dict
import numpy as np
from .common import digest
from .upstream import load_function


def matches(meta, where):
    if not where:
        return True
    if "$and" in where:
        return all(matches(meta, x) for x in where["$and"])
    if "$or" in where:
        return any(matches(meta, x) for x in where["$or"])
    for k, expr in where.items():
        v = meta[k]
        for op, target in expr.items():
            if not {"$eq": v == target, "$lt": v < target, "$lte": v <= target,
                    "$gt": v > target, "$gte": v >= target}[op]:
                return False
    return True


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS cache(kind TEXT,key TEXT,payload TEXT,PRIMARY KEY(kind,key))")
        self.db.execute("CREATE TABLE IF NOT EXISTS sentences(namespace TEXT,id TEXT,document TEXT,metadata TEXT,embedding TEXT,PRIMARY KEY(namespace,id))")
        self.db.commit()

    def get(self, kind, key):
        row = self.db.execute("SELECT payload FROM cache WHERE kind=? AND key=?", (kind,key)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, kind, key, payload):
        # Atomic success only. Technical failures are never cached as predictions.
        self.db.execute("INSERT OR IGNORE INTO cache VALUES(?,?,?)", (kind,key,json.dumps(payload)))
        self.db.commit()


class Collection:
    def __init__(self, store, namespace, embed):
        self.store, self.namespace, self.embed = store, namespace, embed

    def upsert(self, ids, documents, metadatas, embeddings):
        rows = [(self.namespace,i,d,json.dumps(m),json.dumps(e)) for i,d,m,e in
                zip(ids,documents,metadatas,embeddings)]
        self.store.db.executemany("INSERT OR REPLACE INTO sentences VALUES(?,?,?,?,?)", rows)
        self.store.db.commit()

    def get(self, ids=None, where=None, include=None):
        rows = self.store.db.execute("SELECT id,document,metadata,embedding FROM sentences WHERE namespace=? ORDER BY id",(self.namespace,))
        result = {k: [] for k in ["ids","documents","metadatas","embeddings"]}
        allowed = set(ids) if ids is not None else None
        for i,d,m,e in rows:
            m = json.loads(m)
            if (allowed is not None and i not in allowed) or not matches(m,where):
                continue
            for key,val in zip(result,[i,d,m,json.loads(e)]): result[key].append(val)
        return result

    def query(self, query_texts, n_results, where=None):
        # Chroma's default L2 ranking, used by upstream's last-time branch.
        data = self.get(where=where)
        if not data['ids']:
            return {k:[[]] for k in ['ids','documents','metadatas','distances']}
        query = np.asarray(self.embed(query_texts))[0]
        distance = np.sum((np.asarray(data['embeddings'])-query)**2,axis=1)
        inds = np.argsort(distance,kind='stable')[:n_results]
        return {**{k:[[data[k][i] for i in inds]] for k in ['ids','documents','metadatas']},
                'distances':[[float(distance[i]) for i in inds]]}


class Index:
    def __init__(self, store, namespace, embed, engine="sqlite"):
        self.embedding_function = embed
        if engine == "chroma":
            import chromadb
            from chromadb.config import Settings
            client = chromadb.PersistentClient(path=str(store.path.parent/'chroma'),
                                               settings=Settings(anonymized_telemetry=False))
            coll = client.get_or_create_collection('vm-'+namespace[:48], embedding_function=None)
            self._collection = self.collection = ChromaCollection(coll,embed)
        elif engine == "sqlite":
            self._collection = self.collection = Collection(store,namespace,embed)
        else:
            raise ValueError("Unknown index engine")


class ChromaCollection:
    def __init__(self, collection, embed):
        self.inner, self.embed = collection, embed

    def get(self, **kwargs):
        kwargs.setdefault('include',['documents','metadatas','embeddings'])
        return self.inner.get(**kwargs)

    def upsert(self, **kwargs):
        return self.inner.upsert(**kwargs)

    def query(self, query_texts, n_results, where=None):
        # Explicit embedding injection keeps roles/settings in our provenance.
        return self.inner.query(query_embeddings=self.embed(query_texts),
                                n_results=min(n_results,self.inner.count()),where=where)


_ns = {"np":np,"List":List,"Optional":Optional,"Dict":Dict,"ChromaError":RuntimeError,
       "DocumentNotFoundError":LookupError}
for _name in ['custom_query','_generate_id_range','get_caption']:
    setattr(Index,_name,load_function('egorag/database/Chroma.py',_name,_ns,class_name='Chroma'))
