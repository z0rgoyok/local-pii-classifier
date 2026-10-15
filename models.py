"""Pinned ONNX models. No network or remote Python code during inference."""

import hashlib
import json
from pathlib import Path
import urllib.request

ARTIFACTS = {
    "a1": {
        "repo": "Powpowpow23/ru-pii-ner-a1",
        "revision": "67a6270bf24661bf62ea0e93582d02edcc8ee517",
        "graph": "onnx/model.int8.onnx",
        "sha256": "0884223eaec3ceb4be4b196df0aafd641e23a728996c885f45820ff799262185",
        "config": "inference_config.json", "tokenizer": "tokenizer.json",
        "config_sha256": "a87282d358803255e9019ac5b9b8fd4a04f8bc09cf44dc56a57a32e93976f3d0",
        "tokenizer_sha256": "faf2155893824596549e4081c7c287e5b7df23479778548ccf9333c6a97ea36a",
    },
    "nym": {
        "repo": "Wismut/nym-pii-multilingual-small",
        "revision": "4348999cd3c2e20c49615e9af7c6bbb45b64cd85",
        "graph": "int8/model_int8.onnx",
        "sha256": "139006aea2cbd8e709d322f056232570de54661f624143be4893aaa387190286",
        "config": "int8/config.json", "tokenizer": "int8/tokenizer.json",
        "config_sha256": "3f07065571e22bb73eba28ddb1fae4509c0cf703762e3bfa7a6ab2111cf7cb88",
        "tokenizer_sha256": "c299144e68dfec1dc536204a7ae3712710c5c6cade9269a83f5042250d47d8de",
    },
}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def download(root):
    """Only public assets from explicit commit URLs; no account/token discovery."""
    manifest = {}
    for name, spec in ARTIFACTS.items():
        dest = Path(root) / name
        dest.mkdir(parents=True, exist_ok=True)
        hashes = {}
        for key, filename in (("graph", "model.onnx"), ("config", "config.json"),
                              ("tokenizer", "tokenizer.json")):
            url = f"https://huggingface.co/{spec['repo']}/resolve/{spec['revision']}/{spec[key]}"
            path = dest / filename
            with urllib.request.urlopen(url, timeout=120) as source, path.open("wb") as target:
                while block := source.read(1024 * 1024):
                    target.write(block)
            hashes[filename] = digest(path)
            expected = spec["sha256"] if key == "graph" else spec[key + "_sha256"]
            if hashes[filename] != expected:
                raise ValueError("model_digest_mismatch")
        manifest[name] = {"revision": spec["revision"], "files": hashes}
    (Path(root) / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


class OnnxNer:
    def __init__(self, root, name, threads=1):
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.np = np
        self.name = name
        path = Path(root) / name
        manifest = json.loads((Path(root) / "manifest.json").read_text())[name]
        if manifest["revision"] != ARTIFACTS[name]["revision"]:
            raise ValueError("revision_mismatch")
        expected_files = {"model.onnx": ARTIFACTS[name]["sha256"],
                          "config.json": ARTIFACTS[name]["config_sha256"],
                          "tokenizer.json": ARTIFACTS[name]["tokenizer_sha256"]}
        if manifest["files"] != expected_files:
            raise ValueError("manifest_mismatch")
        for filename, expected in manifest["files"].items():
            if digest(path / filename) != expected:
                raise ValueError("asset_digest_mismatch")
        if digest(path / "model.onnx") != ARTIFACTS[name]["sha256"]:
            raise ValueError("model_digest_mismatch")
        config = json.loads((path / "config.json").read_text())
        self.labels = config["types"] if name == "a1" else [
            config["id2label"][str(i)] for i in range(len(config["id2label"]))]
        self.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        self.tokenizer.no_padding()
        self.tokenizer.enable_truncation(max_length=256 if name == "a1" else 480, stride=64)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        self.session = ort.InferenceSession(str(path / "model.onnx"), options,
                                           providers=["CPUExecutionProvider"])
        self.inputs = {x.name for x in self.session.get_inputs()}

    def predict(self, text, min_entity_mass=0.25):
        np = self.np
        encoded = self.tokenizer.encode(text)
        windows = [encoded] + encoded.overflowing
        if len(windows) > 128:
            raise ValueError("model_window_limit")
        found = set()
        for part in windows:
            values = {"input_ids": part.ids, "attention_mask": part.attention_mask,
                      "token_type_ids": part.type_ids}
            arrays = {k: np.asarray([v], dtype=np.int64) for k, v in values.items() if k in self.inputs}
            output = self.session.run(["ner_logits" if self.name == "a1" else "logits"], arrays)[0][0]
            if not np.isfinite(output).all():
                raise ValueError("nonfinite_model_output")
            if self.name == "a1":
                if output.shape != (len(part.ids), len(self.labels), 3):
                    raise ValueError("invalid_shape")
                tags = output.argmax(-1)
                for col, label in enumerate(self.labels):
                    for idx, (start, end) in enumerate(part.offsets):
                        if not part.special_tokens_mask[idx] and end > start and tags[idx, col] != 0:
                            found.add(label)
            else:
                if output.shape != (len(part.ids), len(self.labels)):
                    raise ValueError("invalid_shape")
                probabilities = np.exp(output - output.max(-1, keepdims=True))
                probabilities /= probabilities.sum(-1, keepdims=True)
                entity_mass = 1 - probabilities[:, 0]
                best = probabilities[:, 1:].argmax(-1) + 1
                for idx, (start, end) in enumerate(part.offsets):
                    if not part.special_tokens_mask[idx] and end > start and entity_mass[idx] >= min_entity_mass:
                        found.add(self.labels[best[idx]][2:])
        return found


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(download(args.destination), indent=2))
