# Local PII classifier

Local, fail-closed classification of Russian, English and mixed issue-tracker text. The process returns only `PRIVATE`, `PUBLIC` or `BLOCK`, detector names and timing. It never returns matched values or the input text.

The pipeline is: Unicode/HTML normalization and recursive JSON decoding, deterministic rules, `ru-pii-ner-a1`, then `nym-pii-multilingual-small`. The first signal returns `PRIVATE`. Unsupported content, limits and runtime errors return `BLOCK`.

## Reproduce the benchmark

Requirements: macOS or Linux, [`uv`](https://docs.astral.sh/uv/), Python 3.12 and about 500 MB of temporary disk space.

```sh
git clone https://github.com/z0rgoyok/local-pii-classifier.git
cd local-pii-classifier
./run.sh
```

The script creates an isolated environment, installs pinned dependencies, verifies model artifacts by SHA-256, runs unit tests and both synthetic corpora, then removes the environment, model files and package cache. Aggregate results remain under `build/results/run.*/` and are ignored by Git.

The downloaded models are pinned to immutable revisions:

- [`Powpowpow23/ru-pii-ner-a1`](https://huggingface.co/Powpowpow23/ru-pii-ner-a1), revision `67a6270bf24661bf62ea0e93582d02edcc8ee517`, INT8 ONNX, Apache-2.0. It uses 256-token windows, 64-token overlap and one CPU thread.
- [`Wismut/nym-pii-multilingual-small`](https://huggingface.co/Wismut/nym-pii-multilingual-small), revision `4348999cd3c2e20c49615e9af7c6bbb45b64cd85`, INT8 ONNX, MIT. It uses 480-token windows, 64-token overlap, four CPU threads and entity mass `1 - P(O) >= 0.80`.

Inference uses ONNX Runtime's CPU provider. No model-repository Python code is downloaded or executed.

## Run as a resident JSONL process

Prepare the pinned assets with `python models.py <model-directory>`, then start:

```sh
python scan.py --models <model-directory>
```

Each stdin line is an object such as `{"text":"Contact Alice at test@example.invalid","source":"comment"}`. The output is an aggregate decision:

```json
{"verdict":"PRIVATE","contains_private":true,"reasons":["EMAIL"],"policy":"pii-cascade-v2","elapsed_ms":0.2}
```

Keep one process alive so both model sessions are reused across requests. Pass content through stdin rather than process arguments.

## Policy

Rules cover contacts, identity and test-record fields, long identifiers, credentials, private keys, connection strings, internal hosts and IP addresses, sensitive health/HR context and prompt-injection phrases. A narrow allowlist permits documentation URLs without credentials, query, fragment or UUID path. Binary attachments require a separate extraction or OCR stage and receive `BLOCK`.

The included data is synthetic. The 52-case development corpus and 600-case challenge corpus contain invented names, reserved domains and fixture credentials. See [RESULTS.md](RESULTS.md) for measured results and limitations.

## License

This repository's code and synthetic data are available under the MIT License. Downloaded model artifacts remain under their respective licenses shown above and are not included in this repository.
