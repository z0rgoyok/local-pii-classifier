# Experiment results

The final `pii-cascade-v2` policy detected 40 of 41 supported positive documents in the 52-case development corpus: 97.6% recall, 0 of 10 false positives, and one unsupported binary document correctly returned `BLOCK`.

On the 600-case generated regression corpus it detected 290 of 290 supported positive documents, produced 3 false positives among 300 negative documents (1%), and returned `BLOCK` for all 10 unsupported attachments. The generated corpus was used while tuning the policy, so this is regression evidence rather than independent validation. Its 600 cases are transformations of 60 base templates.

On a MacBook Pro with M4 Max and 48 GB RAM, the final clean reproduction run measured 1.03 ms p50, 43.82 ms p95 and 66.83 ms maximum across 600 warmed classifications up to 2,000 characters. Creating both ONNX sessions took 2.23 seconds with warm filesystem caches and 11.85 seconds on the preceding cold run. Results will vary by machine.

These measurements establish reproducibility on synthetic issue-tracker text. They do not establish recall on a new production distribution. A separate, anonymized and independently labeled corpus is required before using the measured percentage as a production claim.
