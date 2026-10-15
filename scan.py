"""Resident JSONL classifier: {text, source} in; aggregate decision only out."""

import argparse
import json
import sys
from pathlib import Path

from models import OnnxNer
from policy import Classifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, required=True)
    args = parser.parse_args()
    try:
        classifier = Classifier(OnnxNer(args.models, "a1", 1), OnnxNer(args.models, "nym", 4))
    except Exception:
        classifier = Classifier()
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if not isinstance(request, dict) or not isinstance(request.get("text"), str):
                raise ValueError("invalid_request")
            answer = classifier.classify(request["text"], request.get("source", "text"))
        except Exception:
            answer = {"verdict": "BLOCK", "contains_private": None, "reasons": ["INVALID_REQUEST"]}
        print(json.dumps(answer, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
