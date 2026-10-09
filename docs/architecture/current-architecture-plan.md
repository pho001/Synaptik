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
- [ADR 0019: Explicit numerical profiles (historical)](../design/decisions/0019-explicit-numerical-profiles.md)
- [ADR 0020: Synchronous single-default-device Metal execution](../design/decisions/0020-synchronous-single-default-device-metal-execution.md)
- [ADR 0021: Total recursive ACCELERATOR numerical floor (superseded)](../design/decisions/0021-total-recursive-accelerator-numerical-floor.md)
- [ADR 0022: Auditable custom Metal route cost evidence](../design/decisions/0022-auditable-custom-metal-route-cost-evidence.md)
- [ADR 0023: Low-precision ACCELERATOR parity and P0 evidence (profile rule superseded)](../design/decisions/0023-low-precision-accelerator-parity.md)
- [ADR 0024: Environment-bound Metal raw-route certificates (superseded)](../design/decisions/0024-environment-bound-metal-raw-certificates.md)
- [ADR 0025: Custom-only Metal low precision](../design/decisions/0025-custom-only-metal-low-precision.md)
- [ADR 0026: Profile-free numerical semantics (Accepted)](../design/decisions/0026-profile-free-numerical-semantics.md)

## Status

This index is current. The architecture describes the intended complete system. The repository
now has substantive Model, Backend Contract, Planning, Compiler, Runtime, Prepare, Engine, CPU,
and Metal implementations, plus partial Config and Trace contracts. The public lifecycle is
runnable through fixed CPU `Engine.standard()` or explicit `Engine.builder()` registration of
opened CPU and Metal integrations. `Engine.session(...)` prepares one compiled graph once and
exposes repeated or concurrent runs through the same prepared execution and existing result
lifecycle. Ordinary preparation composes non-empty plans across registered owners, with
deterministic owner-indexed representations and explicit direct CPU-to-Metal and Metal-to-CPU
transfers for all seven current data types over fully static rank-0..16 canonical or positive-stride
non-overlapping layouts with checked physical spans.
Completed Task 0032 defines one Model family/dtype numerical semantics without a graph-wide
numerical selector. Exact stored/discrete/cast/raw boundaries remain separate from named floating
arithmetic sites; FLOAT32/BFLOAT16/FLOAT16 arithmetic may use site-local DAZ/FTZ, whereas FLOAT64
may not. The main-worktree implementation removes the selector and cold transport; provider
answers and route qualification remain separate. Integrated validation and independent Class C
review passed. This describes validated implementation; Git history records publication.
Metal's supported exact domain includes unary operations; all
49 ordered casts; FLOAT64/FLOAT32/BFLOAT16/FLOAT16 classification; scalar and right-aligned BOOL
logic; all 14 floating `WHERE` signatures other than the two direct mixed-low signatures; and
exact seven-carrier affine, indexing, replacement, ordering, and movement occurrences. Exact index
roles accept INT32 or INT64. `FOLD_AXIS` admits the six numeric carriers but not BOOL;
`UNFOLD2D`/`FOLD2D` and `UNFOLD3D`/`FOLD3D` admit all four floating carriers. Replacement scatter
remains `NONE` only, and window folds remain non-overlapping. The bounded generated-gradient
closure covers floating casts, inverse affine movement, exact replacement, saved condition/index
roles, and non-overlapping window adjoints without inferring arithmetic reductions, additive
scatter, overlap accumulation, or dynamic or empty geometry.
The current provider additionally admits no-gradient INT32/INT64 MATMUL with INT64-dominant promotion and
exact modular result arithmetic, exact maximum pooling over all four floating carriers, and raw
INT64 initial state. In the frozen ACCELERATOR provider baseline, supported homogeneous FLOAT32 occurrences have
BFLOAT16 and FLOAT16 counterparts. This includes the admitted arithmetic, extrema, scalar,
reduction, scan, MATMUL, MSE, convolution, average-pooling, dropout, rank-one L1 and ScatterAdd,
and singleton-variance domains. Direct BFLOAT16/FLOAT16 mixed-low execution remains unsupported;
callers establish an explicit FLOAT32 boundary. Every low arithmetic occurrence selects the fixed
custom program. Exact homogeneous no-gradient low raw-preserving `RESHAPE`, simple `PERMUTE`,
materializing `CONTIGUOUS`, `SLICE`, `CONCAT`, and `TILE` use the same custom-only route. Classic
MPS, MPP, MPSGraph, CPU, retry, and fallback are not low-precision partition-route alternatives.
The custom program executes BFLOAT16/FLOAT16-valued operations with custom kernels; any
FLOAT32-only node, whether independent or after an explicit cast, may use an internal MPSGraph
boundary step without changing the partition route. Neither generated-pointwise nor
anchor-epilogue fusion occurs anywhere in that partition, and BFLOAT16 is never silently
substituted with FLOAT16. Every other
unlisted occurrence fails closed before route selection.
The post-cutover `EXP` additions are independently bounded: canonical no-gradient FLOAT32 uses
direct MPSGraph, while canonical homogeneous BFLOAT16 and FLOAT16 at positive-static rank `1..16`
use distinct typed custom steps with FLOAT32 working evaluation and one final low narrowing.
An explicit low-to-FLOAT32 cast followed by FLOAT32 `EXP` is a separate internal composition;
it does not turn a low-valued `EXP` into an MPSGraph node.
Canonical typed host ingress/publication and direct CPU/Metal transfer support all seven data types at
ranks `0..16`; transfer also accepts resolved positive-stride non-overlapping physical storage
layouts and rejects unresolved, zero-stride, negative-stride, or overlapping geometry. BOOL
validation visits logical elements only. Selected affine view results retain authenticated logical
Shape, stride, and offset while Java preparation and native preflight independently derive and
validate the physical storage span before any write; selected materializing results are canonical.
An eligible singleton NEG retains its dedicated custom route. Every partition containing
BFLOAT16/FLOAT16 values and the exact Task-0069 rank-one FLOAT32 L1/ScatterAdd/VARIANCE occurrences
at wires `114`, `70`, and `112` use the fixed shared `CUSTOM_PROGRAM` route with compact run-owned
materialized slots and one Java/native invocation; there is no selected-node execution fallback.
This includes exact no-gradient homogeneous low raw-preserving movement and layout operations.
ScatterAdd completes its INT32/INT64 index scan before any
encoding or mutation, keeps duplicates in source order, raw-copies unaddressed cells, and closes
the existing rank-one Gather data cotangent. Singleton VARIANCE dispatches one writer through
exactly `DIV`, `SUB`, `MUL`, `DIV`, yielding positive zero for every finite input and NaN class for
NaN or infinity. Eligible linear canonical FLOAT32 `FLOOR`/`CEIL`/`SIGN`/`RELU` chains use bounded
deterministic generated units without intermediate materialization. Eligible qualified
MATMUL/Conv2d suffixes use one ordered typed anchor step, one dispatch, and one final store without
a materialized suffix intermediate.

Current Metal uses ABI 7 with thirteen exports and one bounded schema-20 route-bearing program
image over type wires `1..7`, operation wires `1..115`, attribute wires `0..41`, and route wires
`1..3`. Structural coverage is `102 / 13`; production capability is exactly `87 / 28` over
operation kinds. Backend-local tuning identities are version 32. Schema 19 and identity 29 are
historical pre-cutover allocations; identity 30 is the historical cutover value and 31 the
superseded FLOAT32 `EXP` value. All are rejected as stale.

The historical 508-row v1 capability ledger recorded both numerical profiles. The current
254-row v2 ledger snapshots actual current profile-free provider answers, including separately
qualified FLOAT32, BFLOAT16, and FLOAT16 `EXP` occurrences, without profile or target/exclusion
columns. The frozen v1 `ACCELERATOR` `EXP` answer remains false. Neither ledger asserts route,
runtime/device, or generated-backward
ownership. The separate append-only identity-allocation
ledger records the pre-cutover Model ordinal 6, Metal type wire 7, schema 19, identities 29, CPU
generator schema 68, and native ABI 7. Current CPU generator schema is 69; current Metal program
schema is 20 with a 124-byte header; current tuning identity is 32; native ABI remains 7.

Low-precision trace identifies the selected custom route and reports its ordered logical boundary
dtype tuple, with no profile. It has no accumulator or working-type field: FLOAT32 working values and
accumulators remain a separate Model arithmetic guarantee, not a trace fact. The trace carries no
hypothetical candidate or removed certificate/environment state. Metal has no
compilation/executable/preparation cache. Its tuning values are session-scoped and bind the dtype,
ABI, schema/policy, program semantics, route candidate, and random context nonce. CPU schema-69
artifact identity also distinguishes BFLOAT16 from FLOAT16 despite their shared short carrier.

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
