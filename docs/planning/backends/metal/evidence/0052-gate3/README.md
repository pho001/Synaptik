# Task 0052 Gate-3 custom-route evidence

This directory retains Task 0052's auditable custom-only structural facts and local timing capture.
It does not change or rerun the immutable Gate-1/Gate-2 numerical evidence in
[`../0052/`](../0052/README.md). Every opaque direct or MPSGraph candidate remains
`DOMAIN-BLOCKED` and was absent. The raw device timings are historical diagnostics only: they are
not qualification, route-selection, or tuning-identity authority.

## Fixed protocol

The exact retained proof kernel source was compiled with `MTLMathModeSafe`. One process on the
system-default Apple M3 Max prepared all 16 custom survivors before measurement, exact-checked the
fixed cost-workload results, verified input preservation, ran four alternating-order warmup rounds,
and retained eight alternating-order rounds. Each candidate batch ran to at least 25 ms with a
1,000,000-execution ceiling. Preparation, upload, validation, download, and reporting were outside
the synchronous execution-plus-completion timing boundary. Nothing was discarded or retried, and
there was no device, size, axis, mode, optimization, fallback, or numerical-oracle matrix.

The harness owns every command encoder dispatch and Metal resource. Each comparison, tensor/scalar
extrema, fused CLAMP, reduction, and scan route has one dispatch and zero route-owned temporary
bytes above steady input/output and immutable metadata resources. Composed custom CLAMP has two
dispatches and one 1,048,576-element FLOAT32 intermediate: 4,194,304 route-owned temporary bytes.

## Identities and controls

| Artifact | SHA-256 |
|---|---|
| retained proof kernel [`../0052/oracle.metal`](../0052/oracle.metal) | `ff15f63c9d2e54d62fa2157e1c424cc00acb90d5355101b18a07175e789a40c0` |
| [`gate3.mm`](gate3.mm) | `42bf2a7c1f8f665c3e4fb3fa9ef1771afbe4e423ea1385de7bd1fa5585e423d0` |
| [`audit.py`](audit.py) | `e6743e9e9c21374c93398d3e042bab747dab3151afbcf53e9460d2c3ecc4a284` |
| [`run-once.py`](run-once.py) | `5f4277d38b5088f560da95ee8bf3d070a3a2b522108f1aa8255d2389bcfaffc2` |
| compiled executable, identity only; removed | `befe573660ef30395d5c0c667689c0bb4978de99bbdb14fe1e3b4937599e39d5` |
| [`raw.txt`](raw.txt) | `4ec49dac639f27389bf276886b9f82eac28fbcf85ea0d1bf73ddbda6d3e92f64` |

The source audit in [`audit.txt`](audit.txt) passed before the device invocation. The one process
exited zero on `Apple M3 Max`, wrote no standard error, reported safe math, validated all candidates,
and preserved every input. `raw.txt` contains all 128 retained duration/iteration samples.

## Historical evidence and disposition

| Candidate | Diagnostic median ns | Dispatches | Route temporary bytes | Historical disposition |
|---|---:|---:|---:|---|
| `custom_cmp_gt` | 285,213.036 | 1 | 0 | selected for `GREATER_THAN` |
| `custom_cmp_ge` | 285,153.648 | 1 | 0 | selected for `GREATER_OR_EQUAL` |
| `custom_cmp_lt` | 284,228.488 | 1 | 0 | selected for `LESS_THAN` |
| `custom_cmp_le` | 284,868.847 | 1 | 0 | selected for `LESS_OR_EQUAL` |
| `custom_cmp_eq` | 285,735.318 | 1 | 0 | selected for `EQUAL` |
| `custom_cmp_ne` | 283,842.348 | 1 | 0 | selected for `NOT_EQUAL` |
| `custom_tensor_min` | 288,421.133 | 1 | 0 | selected for tensor `MIN` |
| `custom_tensor_max` | 292,235.948 | 1 | 0 | selected for tensor `MAX` |
| `custom_scalar_min` | 289,512.448 | 1 | 0 | selected for scalar `MIN` |
| `custom_scalar_max` | 288,510.293 | 1 | 0 | selected for scalar `MAX` |
| `custom_clamp_fused` | 292,241.523 | 1 | 0 | selected for `CLAMP` |
| `composed_custom_clamp` | 377,525.038 | 2 | 4,194,304 | rejected for `CLAMP` |
| `custom_reduction_min` | 11,158,687.500 | 1 | 0 | selected for reduction `MIN` |
| `custom_reduction_max` | 11,028,132.000 | 1 | 0 | selected for reduction `MAX` |
| `custom_scan_sum` | 366,007.848 | 1 | 0 | selected for `CUM_SUM` |
| `custom_scan_prod` | 368,565.257 | 1 | 0 | selected for `CUM_PROD` |

Fourteen operations had exactly one surviving candidate, so no comparative cost gate applied.
Fused custom CLAMP is fixed solely because its exact one dispatch and zero route-owned temporary
bytes strictly dominate composed custom CLAMP's two dispatches and 4,194,304 bytes under identical
proven semantics/domain. The measured medians remain unedited history and did not authorize that
choice. Runtime performs no selection, fallback, retry, or measurement.
