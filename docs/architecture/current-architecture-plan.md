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
- [ADR 0023: Low-precision ACCELERATOR parity and P0 evidence](../design/decisions/0023-low-precision-accelerator-parity.md)
- [ADR 0024: Environment-bound Metal raw-route certificates](../design/decisions/0024-environment-bound-metal-raw-certificates.md)

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
CPU realizes both numerical profiles through identical exact behavior and routes. Model defines
`STRICT_IEEE` as its exact per-operation contract and `ACCELERATOR` as the recursive FLOAT32 and
homogeneous low-precision superset; semantic reach does not imply backend support. Metal execution
remains occurrence- and profile-qualified. Its common domain includes exact unary operations; all
49 ordered casts; FLOAT64/FLOAT32/BFLOAT16/FLOAT16 classification; scalar and right-aligned BOOL
logic; all 14 floating `WHERE` signatures other than the two direct mixed-low signatures; and
exact seven-carrier affine, indexing, replacement, ordering, and movement occurrences. Exact index
roles accept INT32 or INT64. `FOLD_AXIS` admits the six numeric carriers but not BOOL;
`UNFOLD2D`/`FOLD2D` and `UNFOLD3D`/`FOLD3D` admit all four floating carriers. Replacement scatter
remains `NONE` only, and window folds remain non-overlapping. The bounded generated-gradient
closure covers floating casts, inverse affine movement, exact replacement, saved condition/index
roles, and non-overlapping window adjoints without inferring arithmetic reductions, additive
scatter, overlap accumulation, or dynamic or empty geometry.
Both profiles additionally admit no-gradient INT32/INT64 MATMUL with INT64-dominant promotion and
exact modular result arithmetic, exact maximum pooling over all four floating carriers, and raw
INT64 initial state. Under `ACCELERATOR`, every supported homogeneous FLOAT32 occurrence has
BFLOAT16 and FLOAT16 counterparts. This includes the admitted arithmetic, extrema, scalar,
reduction, scan, MATMUL, MSE, convolution, average-pooling, dropout, rank-one L1 and ScatterAdd,
and singleton-variance domains. Direct BFLOAT16/FLOAT16 mixed-low execution remains unsupported;
callers establish an explicit FLOAT32 boundary. Every low arithmetic occurrence selects the fixed
custom program. Exact homogeneous no-gradient low raw-preserving `RESHAPE`, simple `PERMUTE`,
materializing `CONTIGUOUS`, `SLICE`, `CONCAT`, and `TILE` images may additionally expose MPSGraph
only after a complete schema-1 environment/program certificate match. Classic MPS and MPP have
qualified-negative evidence only and own no candidate, route wire, certificate, or fallback.
Every other unlisted occurrence fails closed before route selection.
Canonical typed host ingress/publication and direct CPU/Metal transfer support all seven data types at
ranks `0..16`; transfer also accepts resolved positive-stride non-overlapping physical storage
layouts and rejects unresolved, zero-stride, negative-stride, or overlapping geometry. BOOL
validation visits logical elements only. Selected affine view results retain authenticated logical
Shape, stride, and offset while Java preparation and native preflight independently derive and
validate the physical storage span before any write; selected materializing results are canonical.
An eligible singleton NEG retains its dedicated custom route. Every low arithmetic occurrence and
the exact Task-0069 rank-one FLOAT32 L1/ScatterAdd/VARIANCE occurrences at wires `114`, `70`, and
`112` use the fixed shared `CUSTOM_PROGRAM` route with compact run-owned materialized slots and one
Java/native invocation; there is no selected-node execution fallback. Exact no-gradient
homogeneous BFLOAT16/FLOAT16 `RESHAPE`, simple `PERMUTE`, materializing `CONTIGUOUS`, `SLICE`,
`CONCAT`, and `TILE` images retain custom first and may add MPSGraph only after an exact schema-1
environment/program certificate lookup. ScatterAdd completes its INT32/INT64 index scan before any
encoding or mutation, keeps duplicates in source order, raw-copies unaddressed cells, and closes
the existing rank-one Gather data cotangent. Singleton VARIANCE dispatches one writer through
exactly `DIV`, `SUB`, `MUL`, `DIV`, yielding positive zero for every finite input and NaN class for
NaN or infinity. Eligible linear canonical FLOAT32 `FLOOR`/`CEIL`/`SIGN`/`RELU` chains use bounded
deterministic generated units without intermediate materialization. Eligible ACCELERATOR
MATMUL/Conv2d suffixes use one ordered typed anchor step, one dispatch, and one final store without
a materialized suffix intermediate.

Current Metal uses ABI 6 with fourteen exports and one bounded schema-19 route-bearing program
image over type wires `1..7`, operation wires `1..115`, attribute wires `0..41`, and route wires
`1..3`. Structural coverage is `101 / 14`; production capability is exactly `86 / 29`; route
catalogs are `75 / 35 / 5` MPSGraph and `73 / 42 / 0` custom. Backend-local identities are version
twenty-eight, and every other identity fails closed. ABI 6 reports the immutable native context
environment. Java adds the exact loaded-dylib and capability-ledger hashes, and the schema-1 store
matches the complete key or qualifies nothing.

One canonical 508-row representative capability ledger is generated from the actual CPU and Metal
providers under both profiles. The ledger retains supported and explicit unsupported FLOAT32
answers plus independently queried corresponding BFLOAT16 and FLOAT16 answers. Target/exclusion
columns are architecture mapping, not inferred provider facts, and no row asserts route,
runtime/device, certificate, or generated-backward ownership. The separate append-only
identity-allocation ledger records active Model ordinal 6, Metal type wire 7, schema 19, identities
28, certificate schema 1, CPU generator schema 68, and native ABI 6. The frozen certificate field
schema permits canonical `NONE` for a non-arithmetic accumulator. Metal's active store contains
exactly 24 positive raw-preserving MPSGraph certificates. Its environment-bound classic-MPS/MPP
qualification matrix is negative-only; those families own no candidate, wire, certificate, or
fallback. Both evidence sets remain backend-owned rather than provider facts.

Low-precision trace now identifies the selected custom or certified-MPSGraph route rather than the
candidate set. Only selected MPSGraph carries its complete certificate key plus distinct accuracy
and determinism evidence; custom carries no borrowed evidence. Metal has no
compilation/executable/preparation cache. Its tuning values are session-scoped and bind the dtype,
ABI, schema/policy, program semantics, route candidate, and random context nonce; certificate and
environment facts are safely outside the workload digest because they are immutable within that
context and a reopened context cannot reuse the session value. CPU schema-68 artifact identity
also distinguishes BFLOAT16 from FLOAT16 despite their shared short carrier.

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
