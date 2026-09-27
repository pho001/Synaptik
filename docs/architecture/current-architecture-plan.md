# Current architecture documentation

This document is the navigation page for explanations of the current Synaptik architecture.

The authoritative architecture root and sole authority index is:

- [`../../ARCHITECTURE.md`](../../ARCHITECTURE.md)

The root explicitly incorporates exactly six scoped normative contracts:

- [Foundational modules](contracts/foundational-modules.md)
- [Fixed recurrent scan](contracts/recurrent-scan.md)
- [Compiler and automatic differentiation](contracts/compiler-autograd.md)
- [Runtime, Prepare, and Engine](contracts/runtime-prepare-engine.md)
- [Backend execution](contracts/backend-execution.md)
- [Extensions and training](contracts/extensions-training.md)

Those files are normative only within their stated scopes. This file and all other architecture
documents outside `contracts/` are explanatory. They do not replace the contract or report
implementation completion.

Focused architecture documentation:

- [Overview](overview.md)
- [Lifecycle](lifecycle.md)
- [Module boundaries](module-boundaries.md)
- [Dependency rules](dependency-rules.md)
- [Partition scoring](partition-scoring.md)
- [Performance evidence and model autotuning](performance-evidence-and-tuning.md)
- [Training graph](training-graph.md)
- [ADR 0009: Compiler-owned pre-capture Tensor-expression autograd](../design/decisions/0009-compiler-owned-pre-capture-tensor-expression-autograd.md)
- [ADR 0007: Neural-network module and training boundary](../design/decisions/0007-neural-network-module-and-training-boundary.md)
- [Tracing](tracing.md)
- [Runtime / Prepare / Backend boundary](runtime-prepare-backend-boundary.md)
- [ADR 0010: Staged backend preparation](../design/decisions/0010-staged-backend-preparation.md)
- [ADR 0011: Per-run Runtime resource ownership and cold binding](../design/decisions/0011-per-run-runtime-resource-ownership.md)
- [ADR 0012: Fixed recurrent scan without graph regions](../design/decisions/0012-fixed-recurrent-scan-without-regions.md)
- [ADR 0013: Prepared-execution persistent-resource lifecycle](../design/decisions/0013-prepared-execution-persistent-resource-lifecycle.md)
- [ADR 0014: Scope-indexed normative architecture contracts](../design/decisions/0014-scope-indexed-normative-architecture-contracts.md)
- [ADR 0015: Explicit Engine backend composition](../design/decisions/0015-explicit-engine-backend-composition.md)
- [ADR 0016: CPU/Metal mixed-owner prepared schedule](../design/decisions/0016-cpu-metal-mixed-owner-schedule.md)
- [ADR 0017: Reusable inference session facade](../design/decisions/0017-reusable-inference-session-facade.md)
- [ADR 0018: Public Training Session and SGD lifecycle](../design/decisions/0018-public-training-session-and-sgd-lifecycle.md)
- [ADR 0019: Explicit numerical profiles](../design/decisions/0019-explicit-numerical-profiles.md)
- [ADR 0020: Synchronous single-default-device Metal execution](../design/decisions/0020-synchronous-single-default-device-metal-execution.md)
- [ADR 0021: Total recursive ACCELERATOR numerical floor](../design/decisions/0021-total-recursive-accelerator-numerical-floor.md)
- [ADR 0022: Auditable custom Metal route cost evidence](../design/decisions/0022-auditable-custom-metal-route-cost-evidence.md)

## Status

This index is current. The architecture describes the intended complete system. The repository
now has substantive Model, Backend Contract, Planning, Compiler, Runtime, Prepare, Engine, CPU,
and Metal implementations, plus partial Config and Trace contracts. The public lifecycle is
runnable through fixed CPU `Engine.standard()` or explicit `Engine.builder()` registration of
opened CPU and Metal integrations. `Engine.session(...)` prepares one compiled graph once and
exposes repeated or concurrent runs through the same prepared execution and existing result
lifecycle. Ordinary preparation composes non-empty plans across registered owners, with
deterministic owner-indexed representations and explicit direct CPU-to-Metal and Metal-to-CPU
transfers for all six current data types over fully static rank-0..16 canonical or positive-stride
non-overlapping layouts with checked physical spans.
CPU realizes both numerical profiles through identical exact behavior and routes. Model defines
`STRICT_IEEE` as the unchanged current contract and `ACCELERATOR` as its total recursive
`FLOAT32` superset; that semantic reach does not imply backend support. Metal execution remains
occurrence- and profile-qualified. Its common domain includes exact unary operations; all 36 ordered
casts; FLOAT64/FLOAT32/BFLOAT16 classification; scalar and right-aligned BOOL logic; the nine
promoted floating `WHERE` signatures; and exact six-carrier affine, indexing, replacement, and
movement occurrences. Exact index roles accept INT32 or INT64. `FOLD_AXIS` admits the five numeric
carriers but not BOOL; `UNFOLD2D`/`FOLD2D` and `UNFOLD3D`/`FOLD3D` admit only
FLOAT64/FLOAT32/BFLOAT16. Replacement scatter remains `NONE` only, and window folds remain
non-overlapping. The bounded generated-gradient closure covers floating casts, inverse affine
movement, exact replacement, saved condition/index roles, and non-overlapping window adjoints
without inferring arithmetic reductions, additive scatter, overlap accumulation, dynamic or empty
geometry, signed-stride transfer, or any of the 32 production-false semantic families.
Both profiles additionally admit no-gradient INT32/INT64 MATMUL with INT64-dominant promotion and
exact modular result arithmetic. Accelerator additionally admits the documented FLOAT32
arithmetic, extrema, scalar, reduction, scan, MSE, average-pooling, convolution, and explicit-state
dropout rows; every positive-static FLOAT32 MATMUL vector, matrix, batched, and right-aligned
broadcast geometry; and no-gradient BFLOAT16/FLOAT32 mixed MATMUL with FLOAT32 result. Both
profiles admit exact maximum pooling and raw INT64 initial state. Strict floating MATMUL and every
other unlisted occurrence fail closed before route selection.
Canonical typed host ingress/publication and direct CPU/Metal transfer support all six data types at
ranks `0..16`; transfer also accepts resolved positive-stride non-overlapping physical storage
layouts and rejects unresolved, zero-stride, negative-stride, or overlapping geometry. BOOL
validation visits logical elements only. Selected affine view results retain authenticated logical
Shape, stride, and offset while Java preparation and native preflight independently derive and
validate the physical storage span before any write; selected materializing results are canonical.
An eligible singleton NEG retains its dedicated custom route. Every occurrence selected by Task
0066 wires `6..11,16..19,39..45,51,69,71..84` and the exact Task-0069 rank-one FLOAT32 L1
occurrence use the fixed shared `CUSTOM_PROGRAM` route with declared run-owned value buffers and
one Java/native invocation; there is no selected-node MPSGraph fallback. Current Metal uses ABI 5
with the same thirteen exports and one bounded schema-16 route-bearing program image over type
wires `1..6`, operation wires `1..115`, attribute wires `0..41`, and route wires `1..3`. Structural
coverage is `101 / 14`; production capability is exactly `84 / 31`; route catalogs are
`75 / 35 / 5` MPSGraph and `71 / 44 / 0` custom. Backend-local identities are version
twenty-three, and identity twenty-two and every older identity fail closed.

The Training extension now owns a public reusable
Engine-backed scalar session with persistent SGD, accumulation, and detached in-memory state over
its bounded native-storage domain. A standard-Metal convenience, generic plugin
registration/discovery, broader Metal and transfer coverage, CUDA, generic graph/plan tuning,
broader optimizers, and durable persistence remain planned. The
[implementation roadmap](../planning/roadmap.md) records the exact delivery frontier.

## Decisions and strategies

- [Architecture decision records and design notes](../design/README.md)
- [Backend integration guides](../index.md#documentation-areas)
- [Developer guides](../index.md#documentation-areas)
