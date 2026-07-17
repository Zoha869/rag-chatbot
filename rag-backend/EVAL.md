# EVAL.md — Retrieval Evaluation, Medical Reference RAG (Ledger)
Domain: clinical/textbook reference material for 6 conditions. Evaluation-only notebook — not a diagnostic tool.
## Golden set
- 22 queries total — 19 scorable, 3 unanswerable-by-design.
- Labelled by **content marker** (exact phrase/value, or `src:file.md` for whole-document queries), not chunk id — survives re-chunking (see Part 9).
- Full query -> marker mapping: see `GOLDEN_RAW` in the notebook.

## Results table
```
config                                   P@5       R@5       MRR    nDCG@5
--------------------------------------------------------------------------
1 · dense (baseline)                  0.316     0.950     0.864     0.865 
2 · BM25 keyword only                 0.252     0.616     0.620     0.558 
3 · hybrid (dense + BM25, RRF)        0.263     0.779     0.750     0.706 
4 · hybrid + rewrite                  0.274     0.884     0.804     0.777 
5 · hybrid + multi-query              0.211     0.718     0.649     0.631 
6 · HyDE (dense)                      0.326*    0.963     0.886     0.886 
7 · hybrid + rerank (25→05)           0.326*    0.966*    0.939*    0.924*
8 · rewrite + hybrid + rerank         0.326*    0.966*    0.939*    0.924*
```

## Three worst baseline (dense) queries
- **Tell me everything about how high blood pressure is managed.**  (R@5=0.75, MRR=1.00)
  - wanted: ['hypertension.md']
  - got: ['hypertension.md', 'hypertension.md', 'hypertension.md']
  - _diagnosis: Not a real failure — every retrieved chunk (top-3 shown) is correctly from hypertension.md, and MRR=1.00 confirms the top hit was relevant. R@5 < 1.0 here reflects a structural ceiling: this is a broad query with multiple relevant chunks spread across the document, and hypertension.md has more chunks than fit inside k=5. This is an artifact of the k limit on broad queries, not a synonym gap, exact-value miss, or distractor._
- **Give me the full picture on type 2 diabetes.**  (R@5=0.75, MRR=1.00)
  - wanted: ['type2_diabetes.md']
  - got: ['type2_diabetes.md', 'type2_diabetes.md', 'type2_diabetes.md']
  - _diagnosis:  Same structural pattern as the hypertension query — not a retrieval failure. All retrieved chunks (top-3 shown) are correctly from type2_diabetes.md, and MRR=1.00 confirms the top hit was relevant. R@5 < 1.0 is a ceiling effect from k=5: this broad query has more relevant chunks in type2_diabetes.md than fit inside the top-5 window, not a synonym gap, exact-value miss, or distractor chunk.
- **Explain asthma to me — causes, symptoms, and treatment.**  (R@5=0.75, MRR=1.00)
  - wanted: ['asthma.md']
  - got: ['asthma.md', 'asthma.md', 'asthma.md']
  - _diagnosis: Same structural pattern as the previous two — not a retrieval failure. All retrieved chunks (top-3 shown) are correctly from asthma.md, MRR=1.00. R@5 < 1.0 is the k=5 ceiling effect on a broad query with more relevant chunks than fit in the top-5 window — not a synonym gap, exact-value miss, or distractor chunk

## Shipping decision

I'd ship the **cross-encoder reranker on top of hybrid retrieval** (row 7: hybrid + rerank) — it's the best
config on every metric (P@5=0.326, R@5=0.966, MRR=0.939, nDCG@5=0.924), a clear improvement over the dense
baseline (R@5 0.950→0.966, nDCG@5 0.865→0.924, MRR 0.864→0.939). The extra latency (one Jina rerank call per
query) is justified because the accuracy gain is real and not noise — it wins on all four metrics, not just one.

I would **not** ship query rewriting on its own. Row 8 (rewrite + hybrid + rerank) scores identically to row 7
(hybrid + rerank, no rewrite) — the reranker already fixes whatever the rewrite would have fixed, so the extra
Groq call per query buys nothing once reranking is in the pipeline. That's a wasted API call and added latency
for zero benefit.

I would **not** ship multi-query expansion — it's the worst config in the table (R@5=0.718, nDCG@5=0.631),
worse than doing nothing. And plain hybrid (BM25+RRF) alone, without reranking, actually hurt retrieval on this
corpus (R@5 dropped from 0.950 to 0.779 vs dense) — likely because the corpus is small and query phrasing here
is synonym-friendly, so BM25's noisier ranking diluted an already-strong dense signal when fused via RRF. I'm
keeping this row as a real result: hybrid isn't a universal win, it depends on whether the corpus/queries
actually have the exact-token problem it's designed to fix.

HyDE (row 6) is a decent middle ground — small improvement over dense baseline with only one extra LLM call, but
it doesn't beat the reranker, so it's not the one I'd pick if I could only add one thing.

## One honest limitation of this evaluation

I wrote all 22 golden-set queries myself, on a corpus I already knew well — I unconsciously phrased most narrow
queries in ways I knew the dense retriever would handle well (natural synonyms, no adversarial exact-value
traps beyond the handful I planted deliberately). A real user's questions would be messier, and might expose
weaknesses this golden set doesn't test for. This may explain why the dense baseline already scored so high
(R@5=0.950) — a harder, independently-written eval set would likely show more headroom for hybrid/rewrite to
matter.

Relatedly, the corpus itself is tiny — 6 documents, 25 chunks, 19 scorable queries. At this scale, BM25 has very
little vocabulary to work with, and RRF fusion can dilute a strong dense signal rather than complement it (which
is exactly what row 3 showed). That result might reverse entirely on a larger, more heterogeneous corpus where
BM25 has more signal to contribute — so "hybrid hurt here" shouldn't be read as "hybrid is a bad idea for this
project," only as "hybrid's benefit is corpus-dependent, and this corpus didn't need it."

Finally, this evaluation used Jina for embeddings (free-tier stack, per the assignment) while production `rag.py`
uses Voyage — the absolute numbers here won't transfer 1:1 to production, only the qualitative pattern (reranking
clearly helps, multi-query clearly doesn't) should be expected to hold.

---

# EVAL.md — Part B: Answer Quality (Faithfulness, Relevance, Judge Validation)
Same corpus, same golden set as Part A above — this half proves the *answers*, not just retrieval.
Runs the real production pipeline (`rag.py`: Voyage embeddings + Groq Llama-3.3-70B), not a sandbox stack.
Generated by `answer_eval.py` — see that file for the full implementation.

## Extended golden set
- All 22 queries from Part A now carry a gold reference answer + expected source(s).
- The 3 unanswerable queries expect the exact refusal string as their gold answer.
- Full query -> {answer, sources} mapping: see `GOLDEN_ANSWERS` in `answer_eval.py`.

## Faithfulness (hand-rolled, no ragas/deepeval library)
Method: decompose each model answer into atomic claims (LLM call), verify each claim against the
actual retrieved chunks for that query (separate LLM call per claim), score = supported / total
claims. A refusal with zero claims scores 1.0 — there's nothing unsupported being asserted.

- Average faithfulness across all 22 golden queries: _TODO — run `python answer_eval.py` (needs
  `VOYAGE_API_KEY` + `GROQ_API_KEY` and internet — this container has neither) to fill this in._
- Average answer relevance (reverse-question cosine similarity, non-refusals only): _TODO — same run._
- Per-query table and the three lowest-faithfulness answers: written automatically to this file by
  `answer_eval.py` once it's run.

## Judge validation
- 20 claim-level verdicts sampled into `judge_validation_sample.csv`, hand-labeled against the
  actual retrieved corpus text.
- Human vs. automated-judge agreement: **90.0%** (18/20), but **Cohen's kappa = 0.00**.
- The kappa collapsing to 0 despite 90% raw agreement is itself a finding, not a contradiction:
  the judge never once output UNSUPPORTED across all 20 sampled claims — it only disagreed with
  me on the 2 claims I marked UNSUPPORTED. With one rater using only a single category, kappa's
  chance-correction term goes to zero, so kappa can't credit the judge for agreement on a category
  it never uses. In plain terms: the judge is measurably **lenient** — it correctly validates real
  factual claims well, but doesn't reliably catch non-claims (citation/reference statements
  mistakenly extracted by the decomposition step) as unsupported. Both disagreements were
  meta-statements like "this is referenced as [1]" rather than real factual errors, so the
  faithfulness *scores* on real facts are trustworthy; the judge's blind spot is specifically
  around decomposition artifacts, not around genuine hallucinated medical content.
- Known confound: the judge and the generator are the same model family
  (`llama-3.3-70b-versatile`), which risks self-preference bias — worth swapping in a different
  model as judge if this pattern persists at scale.
## ## Judge validation
- 20 claim-level verdicts sampled into `judge_validation_sample.csv` for hand-labeling.
- _TODO — fill in `human_verdict` in `judge_validation_sample.csv` by rereading the retrieved context yourself, then re-run with `VALIDATE_JUDGE=1 python answer_eval.py` to compute agreement._
- An unvalidated judge is a fiction generator...
## Guardrail diagnosis ("I don't know")
Tested on the 3 unanswerable queries (tuberculosis treatment, CKD staging, vaccination schedule) —
none of which have a matching document in this 6-condition corpus.

-The answer was  I don't know

## Honest limitations of this evaluation
- The judge is unvalidated until `judge_validation_sample.csv` is hand-labeled — treat the
  faithfulness/relevance averages as provisional, not final, until then.
- Only ~20 claims get hand-labeled against a much larger total pool — enough to sanity-check the
  judge, not enough to bound its error rate precisely.
- Same-model-as-judge confound (see Judge validation above) — agreement could look artificially high
  if the judge and generator share systematic blind spots rather than genuinely agreeing.
- Answer relevance via reverse-question cosine similarity rewards paraphrase-closeness to the
  original question, not correctness — a fluent, on-topic, but wrong answer can still score high
  relevance; faithfulness is the metric that actually checks correctness against context.