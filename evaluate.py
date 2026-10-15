"""Aggregate-only binary evaluation. BLOCK/abstention is never a true positive."""

import argparse
import collections
import hashlib
import json
import math
import platform
import time
from pathlib import Path

from models import ARTIFACTS, OnnxNer, digest
from policy import Classifier


def wilson(success, count, z=1.95996398454):
    if not count:
        return [None, None]
    p = success / count
    center = (p + z * z / (2 * count)) / (1 + z * z / count)
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / (1 + z * z / count)
    return [center - half, center + half]


def percentile(values, p):
    return sorted(values)[max(0, math.ceil(len(values) * p) - 1)] if values else None


def evaluate(classifier, rows):
    counts = collections.Counter()
    groups = collections.defaultdict(collections.Counter)
    failed_groups = collections.Counter()
    fp_reasons = collections.Counter()
    latency = []
    for row in rows:
        positive = row.get("private", row.get("expected", {}).get("disposition") != "ALLOW")
        source = row.get("source", "text")
        unsupported = source in {"attachment_binary", "image", "audio", "video", "pdf"}
        answer = classifier.classify(row["input"], source)
        verdict = answer["verdict"]
        latency.append(answer["elapsed_ms"])
        group = groups[row.get("group", "other")]
        group["cases"] += 1
        group[verdict] += 1
        if unsupported:
            counts["unsupported"] += 1
            counts["unsupported_blocked"] += verdict == "BLOCK"
            continue
        if positive:
            counts["positive"] += 1
            counts["true_positive"] += verdict == "PRIVATE"
            counts["false_negative"] += verdict == "PUBLIC"
            counts["abstain_positive"] += verdict == "BLOCK"
            if verdict != "PRIVATE":
                failed_groups[row.get("group", "other")] += 1
        else:
            counts["negative"] += 1
            counts["true_negative"] += verdict == "PUBLIC"
            counts["false_positive"] += verdict == "PRIVATE"
            counts["abstain_negative"] += verdict == "BLOCK"
            if verdict == "PRIVATE":
                fp_reasons.update(answer["reasons"])
    tp, pos = counts["true_positive"], counts["positive"]
    fp, neg = counts["false_positive"], counts["negative"]
    recall = tp / pos if pos else None
    fpr = fp / neg if neg else None
    return {
        "counts": dict(counts),
        "recall_supported": recall,
        "recall_wilson95": wilson(tp, pos),
        "false_positive_rate": fpr,
        "precision": tp / (tp + fp) if tp + fp else None,
        "accuracy_supported": (tp + counts["true_negative"]) / (pos + neg) if pos + neg else None,
        "by_group": {g: dict(c) for g, c in sorted(groups.items())},
        "miss_groups": dict(failed_groups),
        "false_positive_reasons": dict(fp_reasons),
        "quality_pass_latency_ms": {"p50": percentile(latency, .5), "p95": percentile(latency, .95), "max": max(latency)},
        "target_pass": recall is not None and recall >= .97 and fpr is not None and fpr <= .05
                       and counts["abstain_positive"] == 0 and counts["abstain_negative"] == 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--rules-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warm-calls", type=int, default=0)
    args = parser.parse_args()
    started = time.perf_counter()
    if args.rules_only:
        classifier = Classifier(rules_only=True)
    else:
        classifier = Classifier(OnnxNer(args.models, "a1", 1), OnnxNer(args.models, "nym", 4))
    load = time.perf_counter() - started
    rows = [json.loads(line) for line in args.corpus.read_text().splitlines() if line]
    result = {
        "privacy": "aggregates only; no input, values or per-case records",
        "mode": "rules-only" if args.rules_only else "rules-a1-nym",
        "corpus_sha256": digest(args.corpus), "cases": len(rows),
        "synthetic": True, "independent_validation": False,
        "versions": ARTIFACTS, "load_seconds": load,
        "platform": platform.platform(), "python": platform.python_version(),
        "source_sha256": {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
        "evaluation": evaluate(classifier, rows),
    }
    if args.warm_calls:
        short = [row for row in rows if len(row["input"]) <= 2000]
        for row in short[:20]:
            classifier.classify(row["input"], row.get("source", "text"))
        durations = []
        for i in range(args.warm_calls):
            row = short[i % len(short)]
            durations.append(classifier.classify(row["input"], row.get("source", "text"))["elapsed_ms"])
        result["warm_latency_ms"] = {"calls": len(durations), "p50": percentile(durations, .5),
                                     "p95": percentile(durations, .95), "max": max(durations)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=True, indent=2) + "\n")
    print(json.dumps({"mode": result["mode"], "cases": len(rows), **result["evaluation"]}, ensure_ascii=True))


if __name__ == "__main__":
    main()
