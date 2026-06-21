"""Deployable latency: distilled student (autoregressive, on GPU) vs bear (non-autoregressive).

Reviewer #4: the existing latency figure compares the DeepSeek *teacher* (API) to bear, not
the artifact we'd actually ship. This times the distilled 1.5B student's per-instance
generation on an H100 (model load excluded via @enter) against bear-1.1 on the SAME HotpotQA
instances. bear is non-autoregressive so it should be faster per call; the honest story is
"the student trades some latency for an 8.5x token reduction and query-awareness."

Run from repo root:  modal run recompress/distill/latency_student_vs_bear.py --n 30
"""
from __future__ import annotations
import os, sys, json, time

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from recompress.distill.infer import app, Compressor


@app.local_entrypoint()
def main(n: int = 30, ratio: float = 0.3, out: str = "results/latency_student_vs_bear.json"):
    from recompress.act1.data import load_hotpotqa, context_to_text
    from recompress.act1.bear import compress_bear
    from recompress.act1.tokens import count_tokens
    import statistics as st

    insts = load_hotpotqa(n=n)
    items = [{"text": context_to_text(i), "question": i["question"]} for i in insts]
    print(f"timing {n} HotpotQA instances")

    # --- student: timed inside the container (pure generation, load excluded) ---
    print("timing distilled student on H100...")
    timed = Compressor().compress_batch_timed.remote(items, ratio)
    student_lat = [t["latency_s"] for t in timed]
    student_out = [t["n_out"] for t in timed]

    # --- bear: timed locally (SDK / blind deletion, non-autoregressive) ---
    print("timing bear-1.1 locally...")
    bear_lat, bear_out = [], []
    for it in items:
        t0 = time.perf_counter()
        c = compress_bear(it["text"], ratio)
        bear_lat.append(time.perf_counter() - t0)
        bear_out.append(count_tokens(c))

    def summ(lat, outs):
        return {"mean_latency_s": st.mean(lat), "median_latency_s": st.median(lat),
                "p90_latency_s": sorted(lat)[int(0.9 * len(lat))],
                "mean_out_tokens": st.mean(outs)}

    res = {"n": n, "ratio": ratio,
           "student_distilled": summ(student_lat, student_out),
           "bear": summ(bear_lat, bear_out)}
    os.makedirs(os.path.join(_REPO, "results"), exist_ok=True)
    with open(os.path.join(_REPO, out), "w") as f:
        json.dump(res, f, indent=2)

    print("\n" + "=" * 60)
    print("  DEPLOYABLE LATENCY: student vs bear (HotpotQA, n=%d)" % n)
    s, b = res["student_distilled"], res["bear"]
    print(f"  student: mean {s['mean_latency_s']:.3f}s  median {s['median_latency_s']:.3f}s  "
          f"-> {s['mean_out_tokens']:.0f} tok")
    print(f"  bear:    mean {b['mean_latency_s']:.3f}s  median {b['median_latency_s']:.3f}s  "
          f"-> {b['mean_out_tokens']:.0f} tok")
    ratio_lat = s['mean_latency_s'] / b['mean_latency_s'] if b['mean_latency_s'] else 0
    print(f"\n  student is {ratio_lat:.1f}x bear's latency, for {b['mean_out_tokens']/max(1,s['mean_out_tokens']):.1f}x fewer output tokens")
    print(f"  saved {out}")
