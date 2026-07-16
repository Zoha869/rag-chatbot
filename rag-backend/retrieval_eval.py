
import os, re, math, json, time, random
from collections import defaultdict
from dotenv import load_dotenv, find_dotenv
import requests
from qdrant_client import QdrantClient, models
from groq import Groq

load_dotenv(find_dotenv())                 
for name in ("JINA_API_KEY", "GROQ_API_KEY"):
    if not os.getenv(name):
        raise RuntimeError(f"{name} is missing — add it to your .env file.")



class _Obj:
    def __init__(self, **kw): self.__dict__.update(kw)

class JinaClient:
    EMBED_URL  = "https://api.jina.ai/v1/embeddings"
    RERANK_URL = "https://api.jina.ai/v1/rerank"

    def __init__(self, api_key):
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })

    def _post(self, url, payload):
        r = self.session.post(url, json=payload, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"Jina API {r.status_code}: {r.text}")
        return r.json()

    def embed(self, texts, model, input_type="document"):
        task = "retrieval.query" if input_type == "query" else "retrieval.passage"
        data = self._post(self.EMBED_URL,
                          {"model": model, "task": task, "input": list(texts)})["data"]
        embeddings = [d["embedding"] for d in sorted(data, key=lambda x: x["index"])]
        return _Obj(embeddings=embeddings)

    def rerank(self, query, documents, model, top_k=5):
        results = self._post(self.RERANK_URL,
                             {"model": model, "query": query, "documents": documents,
                              "top_n": top_k})["results"]
        objs = [_Obj(index=r["index"], relevance_score=r["relevance_score"]) for r in results]
        return _Obj(results=objs)


EMBED_MODEL  = "jina-embeddings-v3"
RERANK_MODEL = "jina-reranker-v2-base-multilingual"
GEN_MODEL    = "llama-3.3-70b-versatile"
COLLECTION   = "medical_rag_eval"

jina  = JinaClient(os.getenv("JINA_API_KEY"))
groq  = Groq(api_key=os.getenv("GROQ_API_KEY"))
qdrant = QdrantClient(":memory:")

print("clients ready")


def with_backoff(fn, *args, tries=6, base=8, **kwargs):
    """Call a rate-limited API with exponential backoff. Good practice, not decoration."""

    for attempt in range(tries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            msg = str(e).lower()
            if attempt == tries - 1 or not any(s in msg for s in ("rate", "429", "quota", "limit")):
                raise
            wait = base * (attempt + 1)
            print(f"  rate limited — waiting {wait}s ...")
            time.sleep(wait)

print("backoff helper ready")


DOCS = {
"hypertension.md": """
Essential Hypertension. Definition: persistently elevated blood pressure with
no identifiable secondary cause, accounting for roughly 90-95% of hypertension
cases. Classification (ACC/AHA): normal <120/80 mmHg, elevated 120-129/<80,
stage 1 hypertension 130-139/80-89, stage 2 ≥140/90 mmHg, confirmed on
repeated measurements. Risk factors: high dietary sodium, obesity, sedentary
lifestyle, excess alcohol, chronic stress, and family history. Often
asymptomatic until target-organ damage occurs (left ventricular hypertrophy,
retinopathy, chronic kidney disease, stroke). Workup includes basic metabolic
panel, lipid profile, urinalysis, and ECG to assess end-organ effects and rule
out secondary causes when clinically indicated. Management: lifestyle measures
(sodium restriction, weight loss, physical activity, reduced alcohol) for all
patients. Pharmacologic first-line classes include thiazide diuretics, ACE
inhibitors or ARBs, and calcium channel blockers, often combined for stage 2
disease. Target is typically <130/80 mmHg, individualized by age and
comorbidities.
""",
"migraine.md": """
Migraine. Pathophysiology: believed to involve trigeminovascular activation
and cortical spreading depression, leading to release of vasoactive peptides
(e.g., CGRP) and neurogenic inflammation around cranial vessels and meninges.
Clinical features: recurrent moderate-to-severe headache, typically unilateral
and pulsating, lasting 4-72 hours, associated with nausea, vomiting, and
photophobia or phonophobia; about a third of patients experience an aura
(visual disturbances, sensory changes) preceding the headache. Diagnosis is
clinical, based on headache characteristics and associated symptoms, after
excluding secondary causes when red-flag features are present (sudden
"thunderclap" onset, new neurological deficit, onset after age 50). Acute
management: NSAIDs or triptans for moderate-to-severe attacks, taken early in
the attack for best effect; antiemetics for associated nausea. Preventive
therapy is considered for frequent or disabling attacks and includes
beta-blockers, certain anticonvulsants (e.g., topiramate), tricyclic
antidepressants, or CGRP-targeted monoclonal antibodies.
""",
"type2_diabetes.md": """
Type 2 Diabetes Mellitus. Pathophysiology: progressive insulin resistance in
peripheral tissues (muscle, liver, adipose) combined with relative insulin
deficiency from beta-cell dysfunction, leading to chronic hyperglycemia.
Clinical features: often asymptomatic early; may present with polyuria,
polydipsia, unexplained weight loss, blurred vision, or recurrent infections.
Diagnosis: fasting plasma glucose ≥126 mg/dL, HbA1c ≥6.5%, random glucose
≥200 mg/dL with symptoms, or abnormal 2-hour OGTT, confirmed on repeat testing.
Management: first-line is lifestyle modification (diet, weight loss, exercise)
plus metformin unless contraindicated (e.g., significant renal impairment).
Second-line agents include SGLT2 inhibitors, GLP-1 receptor agonists,
sulfonylureas, and DPP-4 inhibitors, chosen based on comorbidities such as
cardiovascular or renal disease. Insulin is added when oral agents fail to
achieve glycemic targets. Regular monitoring for microvascular (retinopathy,
nephropathy, neuropathy) and macrovascular complications is essential.
""",
"anemia.md": """
Iron-Deficiency Anemia. Etiology: most commonly caused by chronic blood loss
(menstrual, gastrointestinal), inadequate dietary intake, malabsorption
(e.g., celiac disease), or increased demand during pregnancy and growth.
Clinical features: fatigue, pallor, dyspnea on exertion, and in more severe or
chronic cases, koilonychia (spoon nails), glossitis, and pica. Laboratory
findings: low hemoglobin and hematocrit, microcytic hypochromic red cells
(low MCV), low serum ferritin (most specific marker), low serum iron, and
elevated total iron-binding capacity (TIBC). In adults, especially older
patients, unexplained iron-deficiency anemia warrants investigation for an
occult gastrointestinal source such as colorectal malignancy. Management:
identify and treat the underlying cause; oral ferrous sulfate or similar iron
salts are first-line, with hemoglobin expected to rise over 4-6 weeks;
intravenous iron or transfusion is reserved for malabsorption, intolerance to
oral iron, or severe/symptomatic anemia.
""",
"asthma.md": """
Bronchial Asthma. Pathophysiology: chronic airway inflammation causing
reversible bronchoconstriction, airway hyperresponsiveness, and variable
airflow obstruction, often triggered by allergens, exercise, cold air, or
respiratory infections. Clinical features: episodic wheezing, shortness of
breath, chest tightness, and cough, often worse at night or early morning.
Diagnosis: spirometry showing reversible obstruction (FEV1/FVC reduced,
improving ≥12% and 200 mL post-bronchodilator); peak flow variability supports
diagnosis. Classification ranges from intermittent to severe persistent based
on symptom frequency and lung function. Management follows a stepwise
approach: short-acting beta-agonists (SABA) for quick relief at all steps;
low-dose inhaled corticosteroids (ICS) introduced early even in mild persistent
disease; combination ICS-long-acting beta-agonist (LABA) therapy for
moderate-to-severe disease; leukotriene receptor antagonists or biologics
(e.g., anti-IgE, anti-IL5) considered for severe or refractory cases. A
written action plan and trigger avoidance are core to long-term control.
""",
"pneumonia.md": """
Community-Acquired Pneumonia (CAP). Definition: acute lower respiratory tract
infection acquired outside a healthcare setting. Common pathogens: Streptococcus
pneumoniae (most common bacterial cause), Haemophilus influenzae, Mycoplasma
pneumoniae, Chlamydophila pneumoniae, and respiratory viruses such as
influenza. Clinical presentation: fever, productive cough, pleuritic chest
pain, dyspnea, tachypnea; elderly patients may present atypically with
confusion or general decline rather than classic symptoms. Diagnosis: chest
X-ray showing a new infiltrate, supported by clinical findings; sputum culture
and blood cultures in more severe cases. Severity is assessed with scoring
tools such as CURB-65 (confusion, urea, respiratory rate, blood pressure, age
≥65) to guide inpatient versus outpatient management. Treatment: outpatient
CAP is typically treated with a macrolide or doxycycline in healthy adults, or
a beta-lactam plus macrolide in those with comorbidities; inpatient cases
often require a beta-lactam plus macrolide or a respiratory fluoroquinolone,
guided by local resistance patterns and severity.
""",
}

print(f"{len(DOCS)} documents loaded: {list(DOCS.keys())}")

def chunk_text(text, source, size=45, overlap=10):
    words = text.split()
    chunks, start, step = [], 0, max(1, size - overlap)
    while start < len(words):
        piece = " ".join(words[start:start + size]).strip()
        if piece:
            chunks.append({"text": piece, "source": source})
        start += step

    return chunks

def build_chunks(docs, size=45, overlap=10):
    out = []
    for source, text in docs.items():
        out.extend(chunk_text(text, source, size, overlap))
    for i, c in enumerate(out):
        c["id"] = i
    return out

chunks = build_chunks(DOCS)
CHUNK_BY_ID = {c["id"]: c for c in chunks}
print(f"{len(chunks)} chunks from {len(DOCS)} documents")


emb = with_backoff(jina.embed, [c["text"] for c in chunks],
                   model=EMBED_MODEL, input_type="document")
dense_vectors = emb.embeddings
DIM = len(dense_vectors[0])

if qdrant.collection_exists(COLLECTION):
    qdrant.delete_collection(COLLECTION)
qdrant.create_collection(
    collection_name=COLLECTION,
    vectors_config=models.VectorParams(size=DIM, distance=models.Distance.COSINE),
)
qdrant.upsert(collection_name=COLLECTION, points=[
    models.PointStruct(id=c["id"], vector=v,
                       payload={"text": c["text"], "source": c["source"]})
    for c, v in zip(chunks, dense_vectors)
])
print(f"Indexed {len(chunks)} chunks -> {DIM}-dim vectors")


GOLDEN_RAW = {
    # --- narrow / factoid: one exact passage answers it ---
    "What blood sugar number confirms someone has type 2 diabetes?":     ["126 mg/dL"],
    "What's usually the first medicine started for type 2 diabetes?":   ["metformin"],
    "How many weeks before iron pills fix low hemoglobin?":             ["4-6 weeks"],
    "Which single blood test best confirms iron-deficiency anemia?":    ["ferritin"],
    "What blood pressure reading counts as stage 2 hypertension?":      ["140/90"],
    "What score do doctors use to decide if pneumonia needs a hospital bed?": ["CURB-65"],
    "What's the most common germ behind community-acquired pneumonia?": ["Streptococcus pneumoniae"],
    "How long can one migraine attack last?":                           ["4-72 hours"],
    "Which inhaler gives quick relief during an asthma attack?":        ["SABA"],
    "How much lung-function improvement on a breathing test suggests asthma?": ["12%"],
    "What headache warning sign means it might not just be a migraine?": ["thunderclap"],
    "What should older adults with unexplained anemia be checked for?": ["colorectal malignancy"],
    "Which drug helps prevent frequent migraines?":                     ["topiramate"],
    "What test result pattern is typical for iron-deficiency anemia's red cells?": ["microcytic hypochromic"],
    "What triggers commonly set off an asthma flare-up?":               ["allergens, exercise, cold air"],
    # --- broad: a whole document answers it ---
    "Tell me everything about how high blood pressure is managed.":     ["src:hypertension.md"],
    "Give me the full picture on type 2 diabetes.":                     ["src:type2_diabetes.md"],
    "Explain asthma to me — causes, symptoms, and treatment.":          ["src:asthma.md"],
    "What should I know about community-acquired pneumonia overall?":   ["src:pneumonia.md"],
    # --- unanswerable, on purpose (not in this 6-condition corpus) ---
    "What's the treatment protocol for tuberculosis?":                  [],
    "How is chronic kidney disease staged and treated?":                [],
    "What's the recommended childhood vaccination schedule?":           [],
}

def resolve_golden(golden_raw, chunks):
    """Turn markers into concrete relevant chunk-id sets for THIS chunking.
    'src:file.md' -> every chunk of that document. Anything else -> case-insensitive substring match.
    """
    resolved = {}
    for q, markers in golden_raw.items():
        rel = set()
        for m in markers:
            if m.startswith("src:"):
                fname = m.split("src:", 1)[1]
                rel |= {c["id"] for c in chunks if c["source"] == fname}
            else:
                rel |= {c["id"] for c in chunks if m.lower() in c["text"].lower()}
        resolved[q] = rel
    return resolved

GOLDEN = resolve_golden(GOLDEN_RAW, chunks)
print(f"{len(GOLDEN)} golden queries resolved")


broken = [q for q, rel in GOLDEN.items() if not rel and GOLDEN_RAW[q]]
print("Labels that resolved to nothing (should be empty):", broken or "none — labels look sane")


SCORABLE = {q: rel for q, rel in GOLDEN.items() if rel}
print(f"{len(SCORABLE)} scorable queries, {len(GOLDEN) - len(SCORABLE)} unanswerable")


def precision_at_k(retrieved, relevant, k):
    top = retrieved[:k]
    if not top:
        return 0.0
    return sum(1 for cid in top if cid in relevant) / len(top)

def recall_at_k(retrieved, relevant, k):
    if not relevant:
        return 0.0
    top = retrieved[:k]
    return sum(1 for cid in top if cid in relevant) / len(relevant)

def reciprocal_rank(retrieved, relevant):
    for rank, cid in enumerate(retrieved, start=1):
        if cid in relevant:
            return 1.0 / rank
    return 0.0

def _dcg(gains):
    return sum(g / math.log2(idx + 2) for idx, g in enumerate(gains))

def ndcg_at_k(retrieved, relevant, k):
    gains = [1.0 if cid in relevant else 0.0 for cid in retrieved[:k]]
    ideal = [1.0] * min(len(relevant), k)
    idcg = _dcg(ideal)
    return (_dcg(gains) / idcg) if idcg > 0 else 0.0


perfect, relevant = [7, 9, 3, 1, 2], {7, 9, 3}
assert precision_at_k(perfect, relevant, 3) == 1.0
assert recall_at_k(perfect, relevant, 3) == 1.0
assert reciprocal_rank(perfect, relevant) == 1.0
assert abs(ndcg_at_k(perfect, relevant, 5) - 1.0) < 1e-9
assert abs(reciprocal_rank([1, 2, 7], {7}) - 1/3) < 1e-9
assert ndcg_at_k([1, 2, 7], {7}, 5) < ndcg_at_k([7, 1, 2], {7}, 5)
print("metrics implemented and self-tested \u2713")


def evaluate(retriever, golden=None, k=5, label="", verbose=True):
    """retriever(query, k) -> list of chunk ids, best first."""
    golden = golden if golden is not None else SCORABLE
    p = r = rr = nd = 0.0
    for q, relevant in golden.items():
        got = retriever(q, k)
        p  += precision_at_k(got, relevant, k)
        r  += recall_at_k(got, relevant, k)
        rr += reciprocal_rank(got, relevant)
        nd += ndcg_at_k(got, relevant, k)
    n = len(golden)
    out = {"config": label, f"P@{k}": p/n, f"R@{k}": r/n, "MRR": rr/n, f"nDCG@{k}": nd/n}
    if verbose:
        print(f"{label:34s} " + "  ".join(f"{m}={v:.3f}" for m, v in list(out.items())[1:]))
    return out

RESULTS = []
print("harness ready")

QUERIES = list(GOLDEN.keys())
qemb = with_backoff(jina.embed, QUERIES, model=EMBED_MODEL, input_type="query")
QVEC = dict(zip(QUERIES, qemb.embeddings))
print(f"cached {len(QVEC)} query vectors in 1 API call")

def embed_query(q):
    if q in QVEC:
        return QVEC[q]
    v = with_backoff(jina.embed, [q], model=EMBED_MODEL, input_type="query").embeddings[0]
    QVEC[q] = v
    return v

def dense_search(query, k=5):
    hits = qdrant.query_points(collection_name=COLLECTION, query=embed_query(query),
                               limit=k, with_payload=False).points
    return [h.id for h in hits]

RESULTS.append(evaluate(dense_search, k=5, label="1 \u00b7 dense (baseline)"))
def worst_queries(retriever, k=5, n=4):
    rows = []
    for q, rel in SCORABLE.items():
        got = retriever(q, k)
        rows.append((recall_at_k(got, rel, k), reciprocal_rank(got, rel), q, got, rel))
    rows.sort(key=lambda r: (r[0], r[1]))
    for rec, rr, q, got, rel in rows[:n]:
        got_src = [CHUNK_BY_ID[i]["source"] for i in got[:3]]
        rel_src = sorted({CHUNK_BY_ID[i]["source"] for i in rel})
        print(f"R@{k}={rec:.2f} MRR={rr:.2f}  {q}")
        print(f"    wanted: {rel_src}")
        print(f"    got:    {got_src}\n")

worst_queries(dense_search)


def tokenize(text):
    # keep alphanumerics and hyphenated codes/scores ("curb-65") as single tokens
    return re.findall(r"[a-z0-9][a-z0-9\-]*", text.lower())

class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [tokenize(d) for d in docs]
        self.N = len(self.docs)
        self.avgdl = sum(len(d) for d in self.docs) / self.N
        self.tf = [dict((t, d.count(t)) for t in set(d)) for d in self.docs]
        df = defaultdict(int)
        for d in self.docs:
            for t in set(d):
                df[t] += 1
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def score(self, query, idx):
        s = 0.0
        dl = len(self.docs[idx])
        for t in tokenize(query):
            if t not in self.idf:
                continue
            f = self.tf[idx].get(t, 0)
            if not f:
                continue
            s += self.idf[t] * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s

    def search(self, query, k=5):
        scored = [(self.score(query, i), i) for i in range(self.N)]
        scored = [(s, i) for s, i in scored if s > 0]
        scored.sort(reverse=True)
        return [i for _, i in scored[:k]]

bm25 = BM25([c["text"] for c in chunks])

print("BM25 for 'What score do doctors use to decide if pneumonia needs a hospital bed?':")
for cid in bm25.search("What score do doctors use to decide if pneumonia needs a hospital bed?", k=3):
    print(f"  [{cid}] ({CHUNK_BY_ID[cid]['source']}) {CHUNK_BY_ID[cid]['text'][:60]}...")

RESULTS.append(evaluate(lambda q, k: bm25.search(q, k), k=5, label="2 \u00b7 BM25 keyword only"))



def rrf_fuse(ranked_lists, k_const=60, limit=5):
    scores = defaultdict(float)
    for lst in ranked_lists:
        for rank, cid in enumerate(lst, start=1):
            scores[cid] += 1.0 / (k_const + rank)
    return [cid for cid, _ in sorted(scores.items(), key=lambda x: -x[1])][:limit]

assert rrf_fuse([[99, 1, 2, 3, 4], [7, 8, 9, 10, 99]], limit=1) == [99]

def hybrid_search(query, k=5, pool=20):
    dense_ids = dense_search(query, k=pool)
    bm25_ids  = bm25.search(query, k=pool)
    return rrf_fuse([dense_ids, bm25_ids], limit=k)

RESULTS.append(evaluate(hybrid_search, k=5, label="3 \u00b7 hybrid (dense + BM25, RRF)"))


q = "What score do doctors use to decide if pneumonia needs a hospital bed?"
for name, fn in [("dense ", dense_search), ("hybrid", hybrid_search)]:
    got = fn(q, 5)
    marks = ["*" if cid in GOLDEN[q] else " " for cid in got]
    srcs  = [f"{m}{CHUNK_BY_ID[c]['source']}" for m, c in zip(marks, got)]
    print(f"{name}: {srcs}")
print("\n(* = a chunk labelled relevant in the golden set)")



def llm(prompt, temperature=0):
    r = groq.chat.completions.create(
        model=GEN_MODEL, temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return r.choices[0].message.content.strip()

REWRITE_PROMPT = """Rewrite this question as a short, keyword-rich search query for a medical reference
document search engine. Keep any exact clinical values, scores, drug names or acronyms EXACTLY as written
(e.g. CURB-65, HbA1c, SABA). Return ONLY the rewritten query, nothing else.

Question: {q}"""

def rewrite(q):
    return llm(REWRITE_PROMPT.format(q=q))

for q in ["What blood sugar number confirms someone has type 2 diabetes?",
          "Which inhaler gives quick relief during an asthma attack?"]:
    print(f"{q}\n  -> {rewrite(q)}\n")


MULTI_PROMPT = """Generate {n} different search queries that would find medical-reference passages
answering this question. Vary the vocabulary (use clinical synonyms), but keep exact values/scores/drug names
unchanged. Return ONLY the queries, one per line, no numbering.

Question: {q}"""

def multi_query(q, n=3):
    lines = [l.strip(" -\u2022\t") for l in llm(MULTI_PROMPT.format(n=n, q=q)).splitlines()]
    return [l for l in lines if l][:n]

def multiquery_search(q, k=5, pool=20):
    variants = [q] + multi_query(q, n=2)
    lists = []
    for v in variants:
        lists.append(dense_search(v, k=pool))
        lists.append(bm25.search(v, k=pool))
    return rrf_fuse(lists, limit=k)

print("variants for 'What triggers commonly set off an asthma flare-up?':")
for v in multi_query("What triggers commonly set off an asthma flare-up?", n=3):
    print("  -", v)


HYDE_PROMPT = """Write a short factual paragraph (2 sentences), in the style of a clinical reference
textbook, that would plausibly answer this question. Do not say you are unsure. This text is for search indexing
only and will never be shown to a user as medical advice.

Question: {q}"""

def hyde_search(q, k=5):
    hypothetical = llm(HYDE_PROMPT.format(q=q))
    vec = with_backoff(jina.embed, [hypothetical],
                       model=EMBED_MODEL, input_type="document").embeddings[0]
    hits = qdrant.query_points(collection_name=COLLECTION, query=vec,
                               limit=k, with_payload=False).points
    return [h.id for h in hits]

print("HyDE draft for 'Which drug helps prevent frequent migraines?':\n ",
      llm(HYDE_PROMPT.format(q="Which drug helps prevent frequent migraines?")).replace("\n", " ")[:220], "...")


CONDENSE_PROMPT = """Given the conversation, rewrite the follow-up into a STANDALONE question
that makes sense with no history. Resolve all pronouns. Return ONLY the question.

Conversation:
{history}

Follow-up: {q}"""

history = ("User: What's usually the first medicine started for type 2 diabetes?\n"
           "Assistant: Metformin, alongside lifestyle changes.")
followup = "what if their kidneys are bad?"
standalone = llm(CONDENSE_PROMPT.format(history=history, q=followup))

print(f"follow-up  : {followup}")
print(f"standalone : {standalone}\n")
print("retrieved with the RAW follow-up:")
for cid in hybrid_search(followup, 3):
    print(f"   ({CHUNK_BY_ID[cid]['source']}) {CHUNK_BY_ID[cid]['text'][:60]}...")
print("\nretrieved with the REWRITTEN query:")
for cid in hybrid_search(standalone, 3):
    print(f"   ({CHUNK_BY_ID[cid]['source']}) {CHUNK_BY_ID[cid]['text'][:60]}...")



def rewrite_hybrid_search(q, k=5, pool=20):
    rq = rewrite(q)
    return rrf_fuse([dense_search(rq, k=pool), bm25.search(rq, k=pool)], limit=k)

RESULTS.append(evaluate(rewrite_hybrid_search, k=5, label="4 \u00b7 hybrid + rewrite"))
RESULTS.append(evaluate(multiquery_search,     k=5, label="5 \u00b7 hybrid + multi-query"))
RESULTS.append(evaluate(hyde_search,           k=5, label="6 \u00b7 HyDE (dense)"))



def rerank_search(query, k=5, pool=25, base=hybrid_search):
    cand_ids = base(query, k=pool)
    if not cand_ids:
        return []
    docs_ = [CHUNK_BY_ID[cid]["text"] for cid in cand_ids]
    res = with_backoff(jina.rerank, query, docs_, model=RERANK_MODEL, top_k=k)
    return [cand_ids[r.index] for r in res.results]

q = "What score do doctors use to decide if pneumonia needs a hospital bed?"
before = hybrid_search(q, 8)
after  = rerank_search(q, 8, pool=8)
mark = lambda ids: [("*" if c in GOLDEN[q] else " ") + CHUNK_BY_ID[c]["source"] for c in ids]
print("query:", q)
print("  before rerank:", mark(before))
print("  after  rerank:", mark(after))
print("\n(* = relevant. Watch the starred chunks climb toward rank 1.)")


RESULTS.append(evaluate(lambda q, k: rerank_search(q, k, pool=25, base=hybrid_search),
                        k=5, label="7 \u00b7 hybrid + rerank (25\u219205)"))


def full_stack(q, k=5, pool=25):
    return rerank_search(q, k=k, pool=pool, base=rewrite_hybrid_search)

RESULTS.append(evaluate(full_stack, k=5, label="8 \u00b7 rewrite + hybrid + rerank"))



def show(results):
    cols = [c for c in results[0].keys() if c != "config"]
    head = f"{'config':34s}" + "".join(f"{c:>10s}" for c in cols)
    print(head); print("-" * len(head))
    best = {c: max(r[c] for r in results) for c in cols}
    for r in results:
        line = f"{r['config']:34s}"
        for c in cols:
            star = "*" if abs(r[c] - best[c]) < 1e-9 else " "
            line += f"{r[c]:>9.3f}{star}"
        print(line)
    print("\n* = best in column")

show(RESULTS)


def rebuild_and_eval(size, overlap, label):
    ch = build_chunks(DOCS, size=size, overlap=overlap)
    vecs = with_backoff(jina.embed, [c["text"] for c in ch],
                        model=EMBED_MODEL, input_type="document").embeddings
    name = f"tmp_{size}_{overlap}"
    if qdrant.collection_exists(name):
        qdrant.delete_collection(name)
    qdrant.create_collection(name, vectors_config=models.VectorParams(
        size=len(vecs[0]), distance=models.Distance.COSINE))
    qdrant.upsert(name, points=[models.PointStruct(id=c["id"], vector=v, payload={})
                                for c, v in zip(ch, vecs)])
    golden = resolve_golden(GOLDEN_RAW, ch)
    scorable = {q: r for q, r in golden.items() if r}
    local_bm25 = BM25([c["text"] for c in ch])

    def retriever(q, k):
        hits = qdrant.query_points(name, query=embed_query(q), limit=20, with_payload=False).points
        return rrf_fuse([[h.id for h in hits], local_bm25.search(q, k=20)], limit=k)

    row = evaluate(retriever, golden=scorable, k=5, label=label)
    print(f"    ({len(ch)} chunks)")
    return row

chunk_rows = [
    rebuild_and_eval(25, 6,  "chunk 25w / overlap 6"),
    rebuild_and_eval(70, 15, "chunk 70w / overlap 15"),
]
print()
show([RESULTS[2]] + chunk_rows)   



def worst_rows(retriever, k=5, n=3):
    rows = []
    for q, rel in SCORABLE.items():
        got = retriever(q, k)
        rows.append((recall_at_k(got, rel, k), reciprocal_rank(got, rel), q,
                     [CHUNK_BY_ID[i]["source"] for i in got[:3]],
                     sorted({CHUNK_BY_ID[i]["source"] for i in rel})))
    rows.sort(key=lambda r: (r[0], r[1]))
    return rows[:n]

worst = worst_rows(dense_search)

lines = []
lines.append("# EVAL.md — Retrieval Evaluation, Medical Reference RAG (Ledger)\n")
lines.append("Domain: clinical/textbook reference material for 6 conditions. Evaluation-only notebook — not a diagnostic tool.\n")
lines.append("## Golden set\n")
lines.append(f"- {len(GOLDEN)} queries total \u2014 {len(SCORABLE)} scorable, {len(GOLDEN) - len(SCORABLE)} unanswerable-by-design.\n")
lines.append("- Labelled by **content marker** (exact phrase/value, or `src:file.md` for whole-document queries), not chunk id \u2014 survives re-chunking (see Part 9).\n")
lines.append("- Full query -> marker mapping: see `GOLDEN_RAW` in the notebook.\n")
lines.append("\n## Results table\n")
lines.append("```\n")
cols = [c for c in RESULTS[0].keys() if c != "config"]
head = f"{'config':34s}" + "".join(f"{c:>10s}" for c in cols)
lines.append(head + "\n")
lines.append("-" * len(head) + "\n")
best = {c: max(r[c] for r in RESULTS) for c in cols}
for r in RESULTS:
    line = f"{r['config']:34s}"
    for c in cols:
        star = "*" if abs(r[c] - best[c]) < 1e-9 else " "
        line += f"{r[c]:>9.3f}{star}"
    lines.append(line + "\n")
lines.append("```\n")
lines.append("\n## Three worst baseline (dense) queries\n")
for rec, rr, q, got, rel in worst:
    lines.append(f"- **{q}**  (R@5={rec:.2f}, MRR={rr:.2f})\n")
    lines.append(f"  - wanted: {rel}\n")
    lines.append(f"  - got: {got}\n")
    lines.append("  - _diagnosis: TODO \u2014 was this a synonym gap, an exact-value/acronym miss, a distractor chunk, or genuinely unanswerable?_\n")
lines.append("\n## Shipping decision\n")
lines.append("_TODO \u2014 what would you ship into Ledger's `rag.py`, and what wasn't worth its latency? Weigh: hybrid is one extra in-process BM25 pass (cheap); rewrite/rerank are +1 network call per query (real latency in a chat product)._\n")
lines.append("\n## One honest limitation of this evaluation\n")
lines.append("_TODO \u2014 e.g. queries were written by the same person who built the retriever; only 6 conditions / ~19 scorable queries, so small score differences are noise; Jina embeddings here vs Voyage in production means absolute numbers don't transfer 1:1, only the qualitative pattern does._\n")

with open("EVAL.md", "w", encoding="utf-8") as f:
    f.writelines(lines)

print("EVAL.md written \u2014 open it and fill in the three TODOs (diagnosis x3, shipping decision, limitation).")
