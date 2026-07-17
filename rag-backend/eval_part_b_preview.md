
---

# EVAL.md — Part B: Answer Quality (Faithfulness, Relevance, Judge Validation)
Same corpus, same golden set as Part A above — this half proves the *answers*, not just retrieval. Runs the real production pipeline (`rag.py`: Voyage embeddings + Groq Llama-3.3-70B), not a sandbox stack.

## Extended golden set
- All 22 queries from Part A now carry a gold reference answer + expected source(s).
- The 3 unanswerable queries expect the exact refusal string as their gold answer.
- Full query -> {answer, sources} mapping: see `GOLDEN_ANSWERS` in `answer_eval.py`.

## Faithfulness (hand-rolled, no ragas/deepeval library)
Method: decompose each model answer into atomic claims (LLM call), verify each claim against the actual retrieved chunks for that query (separate LLM call per claim), score = supported / total claims. A refusal with zero claims scores 1.0.

- Average faithfulness across all 22 golden queries: **0.964**
- Average answer relevance (reverse-question cosine similarity, non-refusals only, n=19): **0.807**

### Per-query faithfulness / relevance
```
query                                                         faithfulness   relevance
--------------------------------------------------------------------------------------
What blood sugar number confirms someone has type 2 diabet           0.600       0.818
What's usually the first medicine started for type 2 diabe           1.000       0.857
How many weeks before iron pills fix low hemoglobin?                 0.667       0.761
Which single blood test best confirms iron-deficiency anem           1.000       0.887
What blood pressure reading counts as stage 2 hypertension           1.000       0.905
What score do doctors use to decide if pneumonia needs a h           1.000       0.621
What's the most common germ behind community-acquired pneu           1.000       0.906
How long can one migraine attack last?                               1.000       0.931
Which inhaler gives quick relief during an asthma attack?            1.000       0.766
How much lung-function improvement on a breathing test sug           1.000       0.675
What headache warning sign means it might not just be a mi           1.000       0.723
What should older adults with unexplained anemia be checke           1.000       0.786
Which drug helps prevent frequent migraines?                         1.000       0.878
What test result pattern is typical for iron-deficiency an           1.000       0.829
What triggers commonly set off an asthma flare-up?                   1.000       0.900
Tell me everything about how high blood pressure is manage           0.938       0.821
Give me the full picture on type 2 diabetes.                         1.000       0.786
Explain asthma to me — causes, symptoms, and treatment.              1.000       0.760
What should I know about community-acquired pneumonia over           1.000       0.721
What's the treatment protocol for tuberculosis?                      1.000   (refusal)
How is chronic kidney disease staged and treated?                    1.000   (refusal)
What's the recommended childhood vaccination schedule?               1.000   (refusal)
```


### Three lowest-faithfulness answers
- **What blood sugar number confirms someone has type 2 diabetes?**  (faithfulness=0.60)
  - unsupported claims: ['A fasting plasma glucose ≥126 mg/dL confirms the diagnosis of type 2 diabetes.', 'HbA1c ≥6.5% confirms the diagnosis of type 2 diabetes.']
  - _diagnosis: Claim-decomposition artifact, not a hallucination. Both values (126 mg/dL, HbA1c ≥6.5%) are real, correctly-cited criteria straight from type2_diabetes.md — they appear in my own gold answer for this exact query. The source text lists them as one disjunctive criterion set ("≥126 mg/dL, OR HbA1c ≥6.5%, OR random glucose ≥200 mg/dL with symptoms, OR abnormal OGTT — confirmed on repeat testing"), but the decomposition step split that into two standalone absolute claims ("X confirms the diagnosis"), dropping the "OR" and the repeat-testing qualifier. The verifier then correctly marked each isolated claim as not literally supported on its own, even though the underlying fact is accurate. Fix: the decomposition prompt should be told to preserve disjunctive ("or") criteria as a single claim instead of splitting each condition into an independent absolute statement.
- **How many weeks before iron pills fix low hemoglobin?**  (faithfulness=0.67)
  - unsupported claims: ['The source anemia.md is referenced as [1] and [2].']
  - _diagnosis: Pure claim-decomposition artifact — this "claim" isn't a factual statement about hemoglobin or iron therapy at all, it's a meta-statement about citation numbering that the decomposer mistakenly extracted as if it were an independent checkable fact. There is nothing here for the verifier to support or refute against the corpus, so marking it UNSUPPORTED is the decomposer's error, not the model's or the retriever's. The actual factual claim (hemoglobin rises over 4-6 weeks) was correctly extracted separately and scored SUPPORTED.
- **Tell me everything about how high blood pressure is managed.**  (faithfulness=0.94)
  - unsupported claims: ['A comprehensive approach to hypertension management incorporates ongoing monitoring.', 'Ongoing monitoring in hypertension management aims to improve patient outcomes.']
  - _diagnosis: This one is the model adding outside/generic knowledge, not a decomposition artifact. hypertension.md does mention diagnostic tests (metabolic panel, lipid profile, urinalysis, ECG) to assess end-organ effects, but it never frames this as "ongoing monitoring" or states that monitoring "aims to improve patient outcomes" — that's a generic clinical-summary flourish the model added on top of the retrieved chunks, plausible-sounding but not literally grounded in this corpus. Lowest-impact of the three (faithfulness still 0.94) since it's editorializing around a real fact rather than inventing a wrong one.
## Judge validation
- 20 claim-level verdicts sampled into `judge_validation_sample.csv` for hand-labeling.
- _TODO — fill in `human_verdict` in `judge_validation_sample.csv` by rereading the retrieved context yourself, then re-run with `VALIDATE_JUDGE=1 python answer_eval.py` to compute agreement._
- An unvalidated judge is a fiction generator — the faithfulness numbers above are provisional until this step is run.
- Known confound: the judge and the generator are the same model family (`llama-3.3-70b-versatile`), which risks self-preference bias — the judge may be more lenient toward this model's phrasing than an independent judge would be. Worth swapping in a different model as judge if the agreement score comes back weak.

## Guardrail diagnosis ("I don't know")
- Guardrail held on 3/3 unanswerable queries.
- **What's the treatment protocol for tuberculosis?** — guardrail held. Retrieval still returned ['asthma.md', 'pneumonia.md'] (Qdrant always returns its k nearest neighbours, relevant or not — it never returns empty-handed).
- **How is chronic kidney disease staged and treated?** — guardrail held. Retrieval still returned ['hypertension.md', 'pneumonia.md', 'type2_diabetes.md'] (Qdrant always returns its k nearest neighbours, relevant or not — it never returns empty-handed).
- **What's the recommended childhood vaccination schedule?** — guardrail held. Retrieval still returned ['anemia.md', 'asthma.md', 'pneumonia.md'] (Qdrant always returns its k nearest neighbours, relevant or not — it never returns empty-handed).
- Cross-check vs. Part A: the retrieval metrics there (P@5/R@5/MRR/nDCG@5) were computed only over the 19 scorable queries — they say nothing about retrieval's behavior on out-of-corpus queries, because there's no relevant chunk to recall. This section fills that gap: retrieval doesn't return nothing on those 3 queries, it returns its nearest neighbours regardless of relevance, so the guardrail's correctness is entirely on generation, not retrieval.

## Honest limitations of this evaluation
- The judge is unvalidated until the `judge_validation_sample.csv` hand-labels are filled in — treat the faithfulness/relevance averages above as provisional, not final.
- Only ~20 claims get hand-labeled against a total pool of 153 — enough to sanity-check the judge, not enough to bound its error rate precisely.
- Same-model-as-judge confound (see Judge validation section) — agreement could look artificially high if the judge and generator share systematic blind spots.
- Answer relevance via reverse-question cosine similarity rewards paraphrase-closeness to the original question, not correctness — a fluent, on-topic, but wrong answer can still score high relevance; faithfulness is the metric that actually checks correctness against context.
