
import csv

def score_judge_agreement(path="judge_validation_sample.csv"):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    labeled = [r for r in rows if r["human_verdict"].strip()]
    if not labeled:
        print("no human_verdict values filled in yet — open the CSV and fill that column in first.")
        return None

    agree = sum(
        1 for r in labeled
        if r["judge_verdict"].strip().upper() == r["human_verdict"].strip().upper()
    )
    n = len(labeled)
    pct = agree / n

    cats = ["SUPPORTED", "UNSUPPORTED"]
    pj = {c: sum(1 for r in labeled if r["judge_verdict"].strip().upper() == c) / n for c in cats}
    ph = {c: sum(1 for r in labeled if r["human_verdict"].strip().upper() == c) / n for c in cats}
    pe = sum(pj[c] * ph[c] for c in cats)
    kappa = (pct - pe) / (1 - pe) if pe < 1 else 1.0

    print(f"labeled: {n}/{len(rows)} rows")
    print(f"judge vs human agreement: {agree}/{n} = {pct:.1%}  (Cohen's kappa = {kappa:.2f})")

    disagreements = [r for r in labeled if r["judge_verdict"].strip().upper() != r["human_verdict"].strip().upper()]
    if disagreements:
        print(f"\n{len(disagreements)} disagreement(s):")
        for r in disagreements:
            print(f"  - [{r['query'][:50]}] \"{r['claim'][:70]}\"")
            print(f"      judge={r['judge_verdict']}  human={r['human_verdict']}")

    return pct, kappa

if __name__ == "__main__":
    score_judge_agreement()