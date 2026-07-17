import os
import csv
import time
import random
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
for name in ("VOYAGE_API_KEY", "GROQ_API_KEY"):
    if not os.getenv(name):
        raise RuntimeError(f"{name} is missing — add it to your .env file.")

import rag  

REFUSAL = "I don't know based on the provided documents."
JUDGE_MODEL = rag.GEN_MODEL  
VOYAGE_PACE_SECONDS = 21  

def with_backoff(fn, *args, tries=6, base=25, **kwargs):
    """Call a rate-limited API with exponential backoff."""
    for attempt in range(tries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            msg = str(e).lower()
            is_rate_limit = any(s in msg for s in ("rate", "429", "quota", "limit"))
            if attempt == tries - 1 or not is_rate_limit:
                raise
            wait = base * (attempt + 1)
            print(f"  rate limited — waiting {wait}s before retry {attempt + 2}/{tries} ...")
            time.sleep(wait)

def paced(fn, *args, pace=VOYAGE_PACE_SECONDS, **kwargs):
    """with_backoff + a fixed pause afterward, so we never burst past 3 RPM in the first place."""
    result = with_backoff(fn, *args, **kwargs)
    time.sleep(pace)
    return result

print("building the real production index (rag.py), same as `uvicorn main:app` does on startup...")
n_chunks = rag.build_index(size=45, overlap=10)
print(f"indexed {n_chunks} chunks\n")



GOLDEN_ANSWERS = {
   
    "What blood sugar number confirms someone has type 2 diabetes?": {
        "answer": "A fasting plasma glucose of 126 mg/dL or higher confirms type 2 diabetes "
                  "(also HbA1c ≥6.5%, random glucose ≥200 mg/dL with symptoms, or an "
                  "abnormal 2-hour OGTT, confirmed on repeat testing).",
        "sources": ["type2_diabetes.md"],
    },
    "What's usually the first medicine started for type 2 diabetes?": {
        "answer": "Metformin, alongside first-line lifestyle modification (diet, weight loss, "
                  "exercise), unless contraindicated such as by significant renal impairment.",
        "sources": ["type2_diabetes.md"],
    },
    "How many weeks before iron pills fix low hemoglobin?": {
        "answer": "Hemoglobin is expected to rise over 4-6 weeks of oral iron therapy.",
        "sources": ["anemia.md"],
    },
    "Which single blood test best confirms iron-deficiency anemia?": {
        "answer": "Serum ferritin — the most specific marker for iron-deficiency anemia.",
        "sources": ["anemia.md"],
    },
    "What blood pressure reading counts as stage 2 hypertension?": {
        "answer": "≥140/90 mmHg, confirmed on repeated measurements.",
        "sources": ["hypertension.md"],
    },
    "What score do doctors use to decide if pneumonia needs a hospital bed?": {
        "answer": "CURB-65 (confusion, urea, respiratory rate, blood pressure, age ≥65), "
                  "used to guide inpatient vs. outpatient management.",
        "sources": ["pneumonia.md"],
    },
    "What's the most common germ behind community-acquired pneumonia?": {
        "answer": "Streptococcus pneumoniae — the most common bacterial cause of CAP.",
        "sources": ["pneumonia.md"],
    },
    "How long can one migraine attack last?": {
        "answer": "4-72 hours.",
        "sources": ["migraine.md"],
    },
    "Which inhaler gives quick relief during an asthma attack?": {
        "answer": "A short-acting beta-agonist (SABA), used for quick relief at all steps of "
                  "the stepwise management approach.",
        "sources": ["asthma.md"],
    },
    "How much lung-function improvement on a breathing test suggests asthma?": {
        "answer": "An improvement of ≥12% and 200 mL in FEV1 after a bronchodilator supports "
                  "the diagnosis (reversible obstruction on spirometry).",
        "sources": ["asthma.md"],
    },
    "What headache warning sign means it might not just be a migraine?": {
        "answer": "A sudden \"thunderclap\" onset (also: new neurological deficit, or onset "
                  "after age 50) — a red flag prompting work-up for a secondary cause.",
        "sources": ["migraine.md"],
    },
    "What should older adults with unexplained anemia be checked for?": {
        "answer": "An occult gastrointestinal source such as colorectal malignancy.",
        "sources": ["anemia.md"],
    },
    "Which drug helps prevent frequent migraines?": {
        "answer": "Topiramate (an anticonvulsant) is one preventive option; beta-blockers, "
                  "tricyclic antidepressants, and CGRP-targeted monoclonal antibodies are others.",
        "sources": ["migraine.md"],
    },
    "What test result pattern is typical for iron-deficiency anemia's red cells?": {
        "answer": "Microcytic hypochromic red cells (low MCV).",
        "sources": ["anemia.md"],
    },
    "What triggers commonly set off an asthma flare-up?": {
        "answer": "Allergens, exercise, cold air, or respiratory infections.",
        "sources": ["asthma.md"],
    },
   
    "Tell me everything about how high blood pressure is managed.": {
        "answer": "Lifestyle measures (sodium restriction, weight loss, physical activity, "
                  "reduced alcohol) for everyone, plus pharmacologic first-line classes "
                  "(thiazide diuretics, ACE inhibitors or ARBs, calcium channel blockers), "
                  "often combined for stage 2 disease, targeting <130/80 mmHg individualized "
                  "by age and comorbidities.",
        "sources": ["hypertension.md"],
    },
    "Give me the full picture on type 2 diabetes.": {
        "answer": "Progressive insulin resistance plus relative insulin deficiency causing "
                  "chronic hyperglycemia; often asymptomatic early; diagnosed by fasting "
                  "glucose, HbA1c, random glucose, or OGTT; managed with lifestyle change "
                  "plus metformin first-line, then SGLT2 inhibitors/GLP-1 agonists/"
                  "sulfonylureas/DPP-4 inhibitors, then insulin; monitored for microvascular "
                  "and macrovascular complications.",
        "sources": ["type2_diabetes.md"],
    },
    "Explain asthma to me — causes, symptoms, and treatment.": {
        "answer": "Chronic airway inflammation causing reversible bronchoconstriction, "
                  "triggered by allergens, exercise, cold air, or infections; episodic "
                  "wheezing, breathlessness, chest tightness, cough; diagnosed by spirometry; "
                  "treated stepwise with SABA for relief, low-dose ICS early, ICS-LABA "
                  "combinations for worse disease, and biologics for severe/refractory cases.",
        "sources": ["asthma.md"],
    },
    "What should I know about community-acquired pneumonia overall?": {
        "answer": "An acute lower respiratory infection acquired outside a healthcare "
                  "setting, most commonly caused by Streptococcus pneumoniae; presents with "
                  "fever, productive cough, pleuritic pain, dyspnea; diagnosed by chest X-ray; "
                  "severity assessed with CURB-65; treated with a macrolide/doxycycline "
                  "outpatient or beta-lactam plus macrolide/fluoroquinolone inpatient.",
        "sources": ["pneumonia.md"],
    },
   
    "What's the treatment protocol for tuberculosis?": {"answer": REFUSAL, "sources": []},
    "How is chronic kidney disease staged and treated?": {"answer": REFUSAL, "sources": []},
    "What's the recommended childhood vaccination schedule?": {"answer": REFUSAL, "sources": []},
}
print(f"golden set extended: {len(GOLDEN_ANSWERS)} queries now carry gold answer + expected sources\n")



def run_pipeline(golden_answers, k=5):
    records = []
    total = len(golden_answers)
    for i, (q, gold) in enumerate(golden_answers.items(), start=1):
        print(f"  [{i}/{total}] {q[:70]}")
        result = paced(rag.rag_answer, q, k=k)  # 1 Voyage embed call inside, paced to stay under 3 RPM
        records.append({
            "query": q,
            "gold_answer": gold["answer"],
            "gold_sources": gold["sources"],
            "model_answer": result["answer"],
            "model_sources": result["sources"],
            "chunks_used": result["chunks_used"],
        })
    return records

print("running every golden query through rag.rag_answer() — real Voyage + Groq calls.")
print(f"paced at ~{VOYAGE_PACE_SECONDS}s/query to respect Voyage's free-tier 3 RPM cap "
      f"(~{len(GOLDEN_ANSWERS) * VOYAGE_PACE_SECONDS // 60} min for this part)...")
RECORDS = run_pipeline(GOLDEN_ANSWERS, k=5)
print(f"got {len(RECORDS)} model answers\n")



def llm(prompt, temperature=0):
    r = with_backoff(
        rag.groq.chat.completions.create,
        model=JUDGE_MODEL, temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return r.choices[0].message.content.strip()

DECOMPOSE_PROMPT = """Break the following answer into a list of atomic factual claims — one
independent, checkable statement per line. If the answer is a refusal (it says it doesn't know),
return exactly one line: NO_CLAIMS. Return ONLY the claims, one per line, no numbering.

Answer: {a}"""

def decompose_claims(answer):
    text = llm(DECOMPOSE_PROMPT.format(a=answer))
    if text.strip() == "NO_CLAIMS":
        return []
    return [l.strip(" -•\t") for l in text.splitlines() if l.strip()]

VERIFY_PROMPT = """Context passages:
{ctx}

Claim: "{claim}"

Is this claim directly supported by the context passages above? Answer with exactly one word:
SUPPORTED or UNSUPPORTED."""

def verify_claim(claim, context_text):
    verdict = llm(VERIFY_PROMPT.format(ctx=context_text, claim=claim)).upper()
    return "UNSUPPORTED" if "UNSUPPORTED" in verdict else "SUPPORTED"

def faithfulness(record):
    """supported_claims / total_claims. A refusal with zero claims scores 1.0 — there's
    nothing unsupported being asserted, so it can't be unfaithful."""
    claims = decompose_claims(record["model_answer"])
    if not claims:
        return 1.0, []
    ctx_text = rag.format_context(record["chunks_used"])
    verdicts = [(c, verify_claim(c, ctx_text)) for c in claims]
    supported = sum(1 for _, v in verdicts if v == "SUPPORTED")
    return supported / len(claims), verdicts



REVERSE_Q_PROMPT = """Generate {n} different questions that this answer could plausibly be
responding to. Return ONLY the questions, one per line, no numbering.

Answer: {a}"""

def generate_reverse_questions(answer, n=3):
    lines_ = [l.strip(" -•\t") for l in llm(REVERSE_Q_PROMPT.format(n=n, a=answer)).splitlines()]
    return [l for l in lines_ if l][:n]

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0

def embed_queries_batch(texts):
    """One paced Voyage call for a whole list of texts, instead of one call per text —
    cuts 4 calls/record (query + 3 reverse-questions) down to 1."""
    resp = paced(rag.voyage.embed, texts, model=rag.EMBED_MODEL, input_type="query")
    return resp.embeddings

def answer_relevance(record):
    """Mean cosine similarity between the original question and n questions the LLM
    reverse-generates from the answer. Refusals are scored separately as a guardrail
    correctness check (Part 4) instead — a refusal isn't 'about' anything, so
    reverse-generation doesn't meaningfully apply to it."""
    reverse_qs = generate_reverse_questions(record["model_answer"], n=3)
    if not reverse_qs:
        return 0.0
    vecs = embed_queries_batch([record["query"]] + reverse_qs)
    qvec, rvecs = vecs[0], vecs[1:]
    sims = [cosine(qvec, rv) for rv in rvecs]
    return sum(sims) / len(sims)

print("scoring faithfulness + answer relevance for every record (this makes a lot of Groq/Voyage calls)...")
ALL_CLAIM_ROWS = [] 
for rec in RECORDS:
    score, verdicts = faithfulness(rec)
    rec["faithfulness"] = score
    rec["claim_verdicts"] = verdicts
    rec["answer_relevance"] = None if rec["model_answer"].strip() == REFUSAL else answer_relevance(rec)
    for claim, verdict in verdicts:
        ALL_CLAIM_ROWS.append({"query": rec["query"], "claim": claim, "judge_verdict": verdict})

avg_faith = sum(r["faithfulness"] for r in RECORDS) / len(RECORDS)
scored_rel = [r["answer_relevance"] for r in RECORDS if r["answer_relevance"] is not None]
avg_rel = sum(scored_rel) / len(scored_rel) if scored_rel else 0.0
print(f"avg faithfulness: {avg_faith:.3f}  |  avg answer relevance (non-refusals, n={len(scored_rel)}): {avg_rel:.3f}")
print(f"{len(ALL_CLAIM_ROWS)} total claim-level verdicts collected\n")



random.seed(7)
SAMPLE_SIZE = min(20, len(ALL_CLAIM_ROWS))
sample = random.sample(ALL_CLAIM_ROWS, SAMPLE_SIZE) if ALL_CLAIM_ROWS else []

with open("judge_validation_sample.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["query", "claim", "judge_verdict", "human_verdict"])
    writer.writeheader()
    for row in sample:
        writer.writerow({**row, "human_verdict": ""})

print(f"wrote {SAMPLE_SIZE} claims to judge_validation_sample.csv")
print("STOP: open that file, reread the actual retrieved context for each query, and fill in")
print("human_verdict (SUPPORTED / UNSUPPORTED) yourself. Then re-run with:")
print("    VALIDATE_JUDGE=1 python answer_eval.py")
print("to compute judge-vs-human agreement.\n")

def score_judge_agreement(path="judge_validation_sample.csv"):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    labeled = [r for r in rows if r["human_verdict"].strip()]
    if not labeled:
        print("no human_verdict values filled in yet — nothing to score.")
        return None
    agree = sum(1 for r in labeled if r["judge_verdict"].strip().upper() == r["human_verdict"].strip().upper())
    n = len(labeled)
    pct = agree / n
    cats = ["SUPPORTED", "UNSUPPORTED"]
    pj = {c: sum(1 for r in labeled if r["judge_verdict"].strip().upper() == c) / n for c in cats}
    ph = {c: sum(1 for r in labeled if r["human_verdict"].strip().upper() == c) / n for c in cats}
    pe = sum(pj[c] * ph[c] for c in cats)
    kappa = (pct - pe) / (1 - pe) if pe < 1 else 1.0
    print(f"judge vs human agreement: {agree}/{n} = {pct:.1%}  (Cohen's kappa = {kappa:.2f})")
    return pct, kappa

AGREEMENT = score_judge_agreement() if os.getenv("VALIDATE_JUDGE") == "1" else None



UNANSWERABLE = [q for q, g in GOLDEN_ANSWERS.items() if g["sources"] == []]

print("=== Guardrail diagnosis: unanswerable queries ===\n")
guardrail_rows = []
for q in UNANSWERABLE:
    rec = next(r for r in RECORDS if r["query"] == q)
    refused = rec["model_answer"].strip() == REFUSAL
    top_sources = sorted({c["source"] for c in rec["chunks_used"]})
    guardrail_rows.append((q, refused, top_sources, rec["model_answer"]))
    status = "PASS (refused)" if refused else "FAIL (answered anyway)"
    print(f"- {q}\n  retrieval still returned: {top_sources}\n  guardrail: {status}")
    if not refused:
        print(f"  model said: {rec['model_answer'][:200]}")
    print()

n_pass = sum(1 for _, refused, _, _ in guardrail_rows if refused)
print(f"guardrail held on {n_pass}/{len(guardrail_rows)} unanswerable queries\n")



scored = sorted(RECORDS, key=lambda r: r["faithfulness"])[:3]
print("=== Three lowest-faithfulness answers ===")
for r in scored:
    print(f"faithfulness={r['faithfulness']:.2f}  {r['query']}")
print()



lines = []
lines.append("\n---\n")
lines.append("\n# EVAL.md — Part B: Answer Quality (Faithfulness, Relevance, Judge Validation)\n")
lines.append("Same corpus, same golden set as Part A above — this half proves the *answers*, not just "
              "retrieval. Runs the real production pipeline (`rag.py`: Voyage embeddings + Groq "
              "Llama-3.3-70B), not a sandbox stack.\n")

lines.append("\n## Extended golden set\n")
lines.append(f"- All {len(GOLDEN_ANSWERS)} queries from Part A now carry a gold reference answer + "
              "expected source(s).\n")
lines.append("- The 3 unanswerable queries expect the exact refusal string as their gold answer.\n")
lines.append("- Full query -> {answer, sources} mapping: see `GOLDEN_ANSWERS` in `answer_eval.py`.\n")

lines.append("\n## Faithfulness (hand-rolled, no ragas/deepeval library)\n")
lines.append("Method: decompose each model answer into atomic claims (LLM call), verify each claim "
              "against the actual retrieved chunks for that query (separate LLM call per claim), "
              "score = supported / total claims. A refusal with zero claims scores 1.0.\n\n")
lines.append(f"- Average faithfulness across all {len(RECORDS)} golden queries: **{avg_faith:.3f}**\n")
lines.append(f"- Average answer relevance (reverse-question cosine similarity, non-refusals only, "
              f"n={len(scored_rel)}): **{avg_rel:.3f}**\n")

lines.append("\n### Per-query faithfulness / relevance\n")
lines.append("```\n")
head = f"{'query':60s}{'faithfulness':>14s}{'relevance':>12s}"
lines.append(head + "\n")
lines.append("-" * len(head) + "\n")
for r in RECORDS:
    rel = f"{r['answer_relevance']:.3f}" if r["answer_relevance"] is not None else "  (refusal)"
    lines.append(f"{r['query'][:58]:60s}{r['faithfulness']:>14.3f}{rel:>12s}\n")
lines.append("```\n")

lines.append("\n### Three lowest-faithfulness answers\n")
for r in scored:
    unsupported = [c for c, v in r["claim_verdicts"] if v == "UNSUPPORTED"]
    lines.append(f"- **{r['query']}**  (faithfulness={r['faithfulness']:.2f})\n")
    lines.append(f"  - unsupported claims: {unsupported if unsupported else '(none — low score is from a small claim count)'}\n")
    lines.append("  - _diagnosis: TODO — was the unsupported claim from the model adding outside "
                  "knowledge, misreading a retrieved chunk, or a claim-decomposition artifact "
                  "(over-splitting one supported idea into a falsely-independent claim)?_\n")

lines.append("\n## Judge validation\n")
lines.append(f"- {SAMPLE_SIZE} claim-level verdicts sampled into `judge_validation_sample.csv` for hand-labeling.\n")
if AGREEMENT:
    pct, kappa = AGREEMENT
    lines.append(f"- Human vs. automated-judge agreement: **{pct:.1%}** (Cohen's kappa = {kappa:.2f}) "
                  f"on {SAMPLE_SIZE} hand-labeled claims.\n")
else:
    lines.append("- _TODO — fill in `human_verdict` in `judge_validation_sample.csv` by rereading the "
                  "retrieved context yourself, then re-run with `VALIDATE_JUDGE=1 python answer_eval.py` "
                  "to compute agreement._\n")
lines.append("- An unvalidated judge is a fiction generator — the faithfulness numbers above are "
              "provisional until this step is run.\n")
lines.append("- Known confound: the judge and the generator are the same model family "
              f"(`{JUDGE_MODEL}`), which risks self-preference bias — the judge may be more lenient "
              "toward this model's phrasing than an independent judge would be. Worth swapping in a "
              "different model as judge if the agreement score comes back weak.\n")

lines.append("\n## Guardrail diagnosis (\"I don't know\")\n")
lines.append(f"- Guardrail held on {n_pass}/{len(guardrail_rows)} unanswerable queries.\n")
for q, refused, top_sources, ans in guardrail_rows:
    verdict = "held" if refused else "FAILED"
    lines.append(f"- **{q}** — guardrail {verdict}. Retrieval still returned {top_sources} "
                  "(Qdrant always returns its k nearest neighbours, relevant or not — it never "
                  "returns empty-handed).\n")
    if not refused:
        lines.append(f"  - model said: \"{ans[:150]}\"\n")
        lines.append("  - _localized failure: this is a **generation**-guardrail failure, not a "
                      "retrieval failure — retrieval was never going to have a real match for a "
                      "genuinely out-of-corpus question, so the refusal has to come from generation "
                      "reading the context critically and noticing nothing answers the question._\n")
lines.append("- Cross-check vs. Part A: the retrieval metrics there (P@5/R@5/MRR/nDCG@5) were computed "
              "only over the 19 scorable queries — they say nothing about retrieval's behavior on "
              "out-of-corpus queries, because there's no relevant chunk to recall. This section fills "
              "that gap: retrieval doesn't return nothing on those 3 queries, it returns its nearest "
              "neighbours regardless of relevance, so the guardrail's correctness is entirely on "
              "generation, not retrieval.\n")

lines.append("\n## Honest limitations of this evaluation\n")
lines.append("- The judge is unvalidated until the `judge_validation_sample.csv` hand-labels are filled "
              "in — treat the faithfulness/relevance averages above as provisional, not final.\n")
lines.append("- Only ~20 claims get hand-labeled against a total pool of "
              f"{len(ALL_CLAIM_ROWS)} — enough to sanity-check the judge, not enough to bound its "
              "error rate precisely.\n")
lines.append("- Same-model-as-judge confound (see Judge validation section) — agreement could look "
              "artificially high if the judge and generator share systematic blind spots.\n")
lines.append("- Answer relevance via reverse-question cosine similarity rewards paraphrase-closeness "
              "to the original question, not correctness — a fluent, on-topic, but wrong answer can "
              "still score high relevance; faithfulness is the metric that actually checks correctness "
              "against context.\n")

with open("EVAL.md", "r", encoding="utf-8") as f:
    existing = f.read()

if "Part B: Answer Quality" in existing:
    with open("eval_part_b_preview.md", "w", encoding="utf-8") as f:
        f.writelines(lines)
    print("EVAL.md already contains a Part B — not overwriting it. New Part B written to "
          "eval_part_b_preview.md instead; merge by hand if you re-ran this after edits.")
else:
    with open("EVAL.md", "w", encoding="utf-8") as f:
        f.write(existing.rstrip("\n") + "\n" + "".join(lines))
    print("EVAL.md updated with Part B (answer quality).")

print("\nRemember: judge validation still needs your hand labels — see judge_validation_sample.csv, "
      "then re-run with VALIDATE_JUDGE=1.")
