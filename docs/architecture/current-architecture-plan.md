# Current architecture documentation

This document is the navigation page for explanations of the current Synaptik architecture.

The authoritative architecture root and sole authority index is:

- [`../../ARCHITECTURE.md`](../../ARCHITECTURE.md)

The root explicitly incorporates exactly six scoped [normative](../glossary.md#term-normative) contracts:

- [Foundational modules](contracts/foundational-modules.md)<span data-glossary-navigation="true"> ([modules glossary](../glossary.md#term-module))</span>
- [Fixed and general differentiable scan](contracts/recurrent-scan.md)<span data-glossary-navigation="true"> ([general differentiable scan glossary](../glossary.md#general-differentiable-scan))</span>
- [Compiler and automatic differentiation](contracts/compiler-autograd.md)<span data-glossary-navigation="true"> ([Compiler glossary](../glossary.md#term-compiler); [automatic differentiation glossary](../glossary.md#autograd))</span>
- [Runtime, Prepare, and Engine](contracts/runtime-prepare-engine.md)<span data-glossary-navigation="true"> ([Runtime glossary](../glossary.md#term-runtime); [Prepare glossary](../glossary.md#prepare); [Engine glossary](../glossary.md#term-engine))</span>
- [Backend execution](contracts/backend-execution.md)<span data-glossary-navigation="true"> ([Backend glossary](../glossary.md#backend))</span>
- [Extensions and training](contracts/extensions-training.md)

Those files are normative only within their stated scopes. This file and all other architecture
documents outside `contracts/` are explanatory. They do not replace the contract or report
implementation completion.

Focused architecture documentation:

- [Overview](overview.md)
- [Lifecycle](lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../glossary.md#lifecycle))</span>
- [Module boundaries](module-boundaries.md)
- [Dependency rules](dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../glossary.md#term-dependency))</span>
- [Partition scoring](partition-scoring.md)<span data-glossary-navigation="true"> ([Partition glossary](../glossary.md#partition))</span>
- [Performance evidence and model autotuning](performance-evidence-and-tuning.md)<span data-glossary-navigation="true"> ([model autotuning glossary](../glossary.md#autotuning--model-autotuning))</span>
- [Stable finite softmax and special-value boundary](softmax-numerical-semantics.md)<span data-glossary-navigation="true"> ([finite glossary](../glossary.md#term-finite); [softmax glossary](../glossary.md#softmax--log-softmax))</span>
- [Training graph](training-graph.md)<span data-glossary-navigation="true"> ([Training graph glossary](../glossary.md#training-graph))</span>
- [ADR 0009: Compiler-owned pre-capture Tensor-expression autograd](../design/decisions/0009-compiler-owned-pre-capture-tensor-expression-autograd.md)<span data-glossary-navigation="true"> ([ADR glossary](../glossary.md#term-architecture-decision-record-adr); [pre-capture Tensor-expression autograd glossary](../glossary.md#pre-capture-tensor-expression-autograd))</span>
- [ADR 0007: Neural-network module and training boundary](../design/decisions/0007-neural-network-module-and-training-boundary.md)
- [Tracing](tracing.md)
- [Runtime / Prepare / Backend boundary](runtime-prepare-backend-boundary.md)
- [ADR 0010: Staged backend preparation](../design/decisions/0010-staged-backend-preparation.md)
- [ADR 0011: Per-run Runtime resource ownership and cold binding](../design/decisions/0011-per-run-runtime-resource-ownership.md)<span data-glossary-navigation="true"> ([resource ownership glossary](../glossary.md#term-resource-ownership))</span><span data-glossary-navigation="true"> ([run glossary](../glossary.md#term-run); [ownership glossary](../glossary.md#term-ownership); [binding glossary](../glossary.md#term-binding))</span>
- [ADR 0012: Fixed recurrent scan without graph regions](../design/decisions/0012-fixed-recurrent-scan-without-regions.md)<span data-glossary-navigation="true"> ([scan glossary](../glossary.md#term-cumulative-scan); [graph glossary](../glossary.md#graph))</span>
- [General differentiable scan: ordinary PURE CPU composition and selected first-order AD](general-differentiable-scan.md)<span data-glossary-navigation="true"> ([PURE glossary](../glossary.md#term-pure); [CPU glossary](../glossary.md#term-cpu); [AD glossary](../glossary.md#autograd))</span>
- [ADR 0045: Bounded general differentiable scan (historical rationale)](../design/decisions/0045-general-differentiable-scan.md)
- [ADR 0013: Prepared-execution persistent-resource lifecycle](../design/decisions/0013-prepared-execution-persistent-resource-lifecycle.md)
- [ADR 0014: Scope-indexed normative architecture contracts](../design/decisions/0014-scope-indexed-normative-architecture-contracts.md)<span data-glossary-navigation="true"> ([Scope glossary](../glossary.md#term-scope))</span>
- [ADR 0015: Explicit Engine backend composition](../design/decisions/0015-explicit-engine-backend-composition.md)<span data-glossary-navigation="true"> ([backend composition glossary](../glossary.md#backend-composition))</span>
- [ADR 0016: CPU/Metal mixed-owner prepared schedule](../design/decisions/0016-cpu-metal-mixed-owner-schedule.md)<span data-glossary-navigation="true"> ([Metal glossary](../glossary.md#term-metal); [owner glossary](../glossary.md#term-owner); [prepared schedule glossary](../glossary.md#prepared-schedule--preparedschedule))</span>
- [ADR 0017: Reusable inference session facade](../design/decisions/0017-reusable-inference-session-facade.md)<span data-glossary-navigation="true"> ([inference glossary](../glossary.md#term-inference); [facade glossary](../glossary.md#term-facade))</span>
- [ADR 0018: Public Training Session and SGD lifecycle](../design/decisions/0018-public-training-session-and-sgd-lifecycle.md)<span data-glossary-navigation="true"> ([Training Session glossary](../glossary.md#training-session--trainingsession); [SGD glossary](../glossary.md#term-stochastic-gradient-descent-sgd))</span>
- [ADR 0019: Explicit numerical profiles (historical)](../design/decisions/0019-explicit-numerical-profiles.md)
- [ADR 0020: Synchronous single-default-device Metal execution](../design/decisions/0020-synchronous-single-default-device-metal-execution.md)<span data-glossary-navigation="true"> ([Synchronous glossary](../glossary.md#term-synchronous); [device glossary](../glossary.md#term-device))</span>
- [ADR 0021: Total recursive ACCELERATOR numerical floor (superseded)](../design/decisions/0021-total-recursive-accelerator-numerical-floor.md)<span data-glossary-navigation="true"> ([ACCELERATOR glossary](../glossary.md#term-accelerator))</span>
- [ADR 0022: Auditable custom Metal route cost evidence](../design/decisions/0022-auditable-custom-metal-route-cost-evidence.md)
- [ADR 0023: Low-precision ACCELERATOR parity and P0 evidence (profile rule superseded)](../design/decisions/0023-low-precision-accelerator-parity.md)<span data-glossary-navigation="true"> ([precision glossary](../glossary.md#term-precision))</span>
- [ADR 0024: Environment-bound Metal raw-route certificates (superseded)](../design/decisions/0024-environment-bound-metal-raw-certificates.md)
- [ADR 0025: Custom-only Metal low precision](../design/decisions/0025-custom-only-metal-low-precision.md)
- [ADR 0026: Profile-free numerical semantics (Accepted)](../design/decisions/0026-profile-free-numerical-semantics.md)<span data-glossary-navigation="true"> ([Profile-free numerical semantics glossary](../glossary.md#profile-free-numerical-semantics))</span>
- [ADR 0029: Stable finite softmax (Accepted)](../design/decisions/0029-stable-finite-softmax.md)

## Status

This index is current. The architecture describes the intended complete system. The repository
now has substantive [Model](../glossary.md#term-model), [Backend](../glossary.md#backend) Contract, [Planning](../glossary.md#term-planning), [Compiler](../glossary.md#term-compiler), [Runtime](../glossary.md#term-runtime), [Prepare](../glossary.md#prepare), [Engine](../glossary.md#term-engine), [CPU](../glossary.md#term-cpu),
and [Metal](../glossary.md#term-metal) implementations, plus partial Config and [Trace](../glossary.md#trace) contracts. The public [lifecycle](../glossary.md#lifecycle) is
runnable through fixed CPU `Engine.standard()` or explicit `Engine.builder()` registration of
opened CPU and Metal integrations. `Engine.prepare(...)` publishes one reusable prepared [owner](../glossary.md#term-owner);
`Engine.run` binds exact IDs in any order and [`Engine.runValues`](../glossary.md#fresh-value-binding--enginerunvalues) binds fresh fully static
exact-[descriptor](../glossary.md#term-descriptor) values in compiled order. Both reuse Runtime's unique [lease](../glossary.md#term-lease) and existing result
lifecycle. The former [InferenceSession](../glossary.md#former-inference-session--inferencesession) [facade](../glossary.md#term-facade) is removed; see
[ADR 0039](../design/decisions/0039-portable-graph-model-persistence.md)<span data-glossary-navigation="true"> ([ADR glossary](../glossary.md#term-architecture-decision-record-adr))</span>. Ordinary preparation composes non-empty plans across registered owners, with
deterministic owner-indexed representations and explicit direct CPU-to-Metal and Metal-to-CPU
transfers for all seven current data types over fully static [rank](../glossary.md#term-rank)-0..16 [canonical](../glossary.md#term-canonical-form) or positive-[stride](../glossary.md#term-stride)
non-overlapping [layouts](../glossary.md#layout) with checked physical spans.
Completed Task 0032 defines one Model family/[dtype](../glossary.md#data-type--datatype) numerical semantics without a [graph](../glossary.md#graph)-wide
numerical selector. Exact stored/discrete/cast/raw boundaries remain separate from named floating
arithmetic sites; [FLOAT32](../glossary.md#term-float32)/[BFLOAT16](../glossary.md#term-bfloat16)/[FLOAT16](../glossary.md#term-float16) arithmetic may use site-local [DAZ](../glossary.md#term-daz)/[FTZ](../glossary.md#term-ftz), whereas [FLOAT64](../glossary.md#term-float64)
may not. The main-worktree implementation removes the selector and cold transport; provider
answers and route [qualification](../glossary.md#term-qualification) remain separate. Integrated [validation](../glossary.md#term-validation) and independent Class C
review passed. This describes validated implementation; Git history records [publication](../glossary.md#term-publication).
Metal's supported exact domain includes [unary](../glossary.md#term-unary) operations; all
49 ordered casts; FLOAT64/FLOAT32/BFLOAT16/FLOAT16 classification; [scalar](../glossary.md#term-scalar) and right-aligned [BOOL](../glossary.md#data-type--datatype)
logic; all 14 floating `WHERE` [signatures](../glossary.md#term-signature) other than the two direct mixed-low signatures; and
exact seven-[carrier](../glossary.md#term-storage-carrier) affine, indexing, replacement, ordering, and movement occurrences. Exact index
roles accept [INT32](../glossary.md#data-type--datatype) or [INT64](../glossary.md#data-type--datatype). `FOLD_AXIS` admits the six numeric carriers but not BOOL;
`UNFOLD2D`/`FOLD2D` and `UNFOLD3D`/`FOLD3D` admit all four floating carriers. Replacement scatter
remains `NONE` only. `FOLD_AXIS` and FLOAT64/low image folds retain non-overlap placement; FLOAT32
image folds additionally accumulate overlap with one output-owned deterministic traversal and no
atomics, scratch or [host](../glossary.md#term-host) fallback. Shared Compiler [convolution](../glossary.md#term-channels-first-convolution-layers) adjoints now qualify [Conv3d](../glossary.md#conv3d--ncdhw)
training on CPU FLOAT32/FLOAT64 and Metal FLOAT32; support requires the entire generated [topology](../glossary.md#term-topology),
not merely the forward convolution. CPU raw `UNFOLD3D` also preserves BFLOAT16/FLOAT16 bits,
without unlocking low fold/convolution arithmetic. CPU [Pool3d](../glossary.md#pool3d--ncdhw-pooling)'s existing no-[gradient](../glossary.md#term-gradient) [admission](../glossary.md#term-admission)
is unchanged, so this is not a general trainable Conv3d-to-Pool3d CPU chain.
The current provider additionally admits no-gradient INT32/INT64 [MATMUL](../glossary.md#matrix-multiplication--matmul) with INT64-dominant promotion and
exact modular result arithmetic, exact maximum [pooling](../glossary.md#term-pooling) over all four floating carriers, and raw
INT64 initial state. In the frozen [ACCELERATOR](../glossary.md#term-accelerator) provider baseline, supported [homogeneous](../glossary.md#term-homogeneous) FLOAT32 occurrences have
BFLOAT16 and FLOAT16 counterparts. This includes the admitted arithmetic, [extrema](../glossary.md#saturation-pole-and-extremum), scalar,
reduction, [scan](../glossary.md#term-cumulative-scan), MATMUL, [MSE](../glossary.md#mean-squared-error--mse), convolution, average-pooling, [dropout](../glossary.md#dropout--inverted-dropout), [rank-one](../glossary.md#axis-rank-scalar-vector-and-matrix) [L1](../glossary.md#term-l1) and ScatterAdd,
and [singleton](../glossary.md#term-singleton)-[variance](../glossary.md#term-variance) domains. Direct BFLOAT16/FLOAT16 mixed-low execution remains unsupported;
callers establish an explicit FLOAT32 boundary. Every low arithmetic [occurrence](../glossary.md#term-occurrence) selects the fixed
custom program. Exact homogeneous no-gradient low raw-preserving `RESHAPE`, simple `PERMUTE`,
materializing [`CONTIGUOUS`](../glossary.md#term-contiguous), [`SLICE`](../glossary.md#slice), `CONCAT`, and `TILE` use the same custom-only route. Classic
[MPS](../glossary.md#term-mps), [MPP](../glossary.md#term-mpp), [MPSGraph](../glossary.md#term-mpsgraph), CPU, retry, and fallback are not low-[precision](../glossary.md#term-precision) [partition](../glossary.md#partition)-route alternatives.
The custom program executes BFLOAT16/FLOAT16-valued operations with custom [kernels](../glossary.md#kernel); any
FLOAT32-only [node](../glossary.md#node), whether independent or after an explicit cast, may use an internal MPSGraph
boundary step without changing the partition route. Neither generated-[pointwise](../glossary.md#term-pointwise) nor
anchor-[epilogue](../glossary.md#term-epilogue) [fusion](../glossary.md#term-fusion) occurs anywhere in that partition, and BFLOAT16 is never silently
substituted with FLOAT16. Every other
unlisted occurrence fails closed before route selection.
The post-cutover [`EXP`](../glossary.md#term-expx) additions are independently bounded: canonical no-gradient FLOAT32 uses
direct MPSGraph, while canonical homogeneous BFLOAT16 and FLOAT16 at positive-static rank `1..16`
use distinct typed custom steps with FLOAT32 working [evaluation](../glossary.md#term-evaluation) and one final low [narrowing](../glossary.md#term-narrowing).
An explicit low-to-FLOAT32 cast followed by FLOAT32 `EXP` is a separate internal composition;
it does not turn a low-valued `EXP` into an MPSGraph node.
Task 0076 adds a bounded canonical no-gradient FLOAT32 [`SIGMOID`](../glossary.md#term-sigmoid) occurrence through an exact
stored-bit sign [guard](../glossary.md#term-guard) and the stable Model formula in one composed MPSGraph step.
Task 0077 separately admits homogeneous BFLOAT16 and FLOAT16 `SIGMOID` through typed custom steps
with one FLOAT32 [exponent](../glossary.md#term-exponent) and one final low narrowing. An explicit low-to-FLOAT32 cast may place
the FLOAT32 step alongside a low custom step inside the same custom partition.
Canonical typed host ingress/publication and direct CPU/Metal transfer support all seven data types at
ranks `0..16`; transfer also accepts resolved positive-stride non-overlapping physical storage
layouts and rejects unresolved, zero-stride, negative-stride, or overlapping geometry. BOOL
validation visits logical [elements](../glossary.md#term-element) only. Selected affine [view](../glossary.md#term-view-meanings) results retain authenticated logical
[Shape](../glossary.md#shape), stride, and offset while Java preparation and native [preflight](../glossary.md#term-preflight) independently derive and
validate the [physical storage span](../glossary.md#referenced-element-span) before any write; selected materializing results are canonical.
An eligible singleton NEG retains its dedicated custom route. Every partition containing
BFLOAT16/FLOAT16 values and the exact Task-0069 rank-one FLOAT32 L1/ScatterAdd/VARIANCE occurrences
at wires `114`, `70`, and `112` use the fixed shared `CUSTOM_PROGRAM` route with compact [run](../glossary.md#term-run)-owned
materialized slots and one Java/native invocation; there is no selected-node execution fallback.
This includes exact no-gradient homogeneous low raw-preserving movement and layout operations.
ScatterAdd completes its INT32/INT64 index scan before any
encoding or mutation, keeps duplicates in source order, raw-copies unaddressed cells, and closes
the existing rank-one [Gather](../glossary.md#gather) data [cotangent](../glossary.md#cotangent). Singleton VARIANCE dispatches one writer through
exactly `DIV`, `SUB`, `MUL`, `DIV`, yielding [positive zero](../glossary.md#nan-infinity-and-signed-zero) for every [finite](../glossary.md#term-finite) input and [NaN](../glossary.md#term-nan) class for
NaN or [infinity](../glossary.md#term-infinity). Eligible [linear](../glossary.md#term-linear-meanings) canonical FLOAT32 `FLOOR`/`CEIL`/`SIGN`/`RELU` chains use bounded
deterministic generated units without intermediate [materialization](../glossary.md#materialization). Eligible qualified
MATMUL/[Conv2d](../glossary.md#convolution--conv2d) suffixes use one ordered typed anchor step, one [dispatch](../glossary.md#term-dispatch), and one final store without
a materialized suffix intermediate.

Current Metal uses [ABI](../glossary.md#term-application-binary-interface-abi) 7 with thirteen exports and one bounded [schema](../glossary.md#term-schema)-20 route-bearing program
image over type wires `1..7`, [operation](../glossary.md#operation) wires `1..115`, attribute wires `0..41`, and route wires
`1..3`. Structural coverage is `103 / 12`; production capability is exactly `88 / 27` over
operation kinds. Backend-local tuning [identities](../glossary.md#term-identity-meanings) are version 36. Schema 19 and [identity](../glossary.md#term-identity-meanings) 29 are
historical pre-cutover allocations; identity 30 is the historical cutover value and 31 the
superseded FLOAT32 `EXP` value; 32 is the superseded low-`EXP` value and 33 the superseded
FLOAT32 `SIGMOID` value; 34 is the superseded low-`SIGMOID` value and 35 is the superseded
native-boundary and exact low-value repair value. All are rejected as stale. Identity 36 corrects
standalone scalar admission without expanding the provider domain.

The historical 508-row v1 [capability ledger](../glossary.md#capability-ledger) recorded both numerical profiles. The current
254-row v2 ledger snapshots actual current profile-free provider answers, including separately
qualified FLOAT32, BFLOAT16, and FLOAT16 `EXP` and `SIGMOID` occurrences, without
profile or [target](../glossary.md#term-target)/exclusion columns. The frozen v1 `ACCELERATOR` `EXP` and `SIGMOID` answers remain
false. Neither ledger asserts route,
runtime/[device](../glossary.md#term-device), or generated-backward
[ownership](../glossary.md#term-ownership-meanings). The separate append-only identity-allocation
ledger records the pre-cutover Model ordinal 6, Metal type wire 7, schema 19, identities 29, CPU
generator schema 68, and native ABI 7. Current CPU generator schema is 70; current Metal program
schema is 20 with a 124-byte header; current tuning identity is 36; native ABI remains 7.
CPU schema 70 rejects stale schema-69 [artifacts](../glossary.md#term-artifact) after [activation](../glossary.md#term-activation)-extreme and low-working-site
repairs; it does not change Model's numerical contract.

Low-precision trace identifies the selected custom route and reports its ordered logical boundary
dtype tuple, with no profile. It has no [accumulator](../glossary.md#accumulator-type) or working-type field: FLOAT32 working values and
accumulators remain a separate Model arithmetic guarantee, not a trace fact. The trace carries no
hypothetical candidate or removed [certificate](../glossary.md#formal-proof-certificate-and-trusted-assumptions)/environment state. Metal has no
[compilation](../glossary.md#compiler-engine-and-execution-stage-names)/executable/preparation [cache](../glossary.md#term-cache). Its tuning values are session-scoped and bind the dtype,
ABI, schema/policy, program semantics, route candidate, and random context nonce. CPU schema-70
artifact identity also distinguishes BFLOAT16 from FLOAT16 despite their shared short carrier.

The Training extension now owns a public reusable
Engine-backed scalar session with persistent [SGD](../glossary.md#term-stochastic-gradient-descent-sgd), accumulation, and [detached](../glossary.md#detached-model-results) in-memory state over
its bounded native-storage domain. A standard-Metal convenience, generic [plugin](../glossary.md#term-plugin)
registration/discovery, broader Metal and transfer coverage, [CUDA](../glossary.md#term-cuda), generic graph/plan tuning,
broader [optimizers](../glossary.md#term-optimizer), and durable [persistence](../glossary.md#term-persistence) remain planned. The
[implementation roadmap](../planning/roadmap.md) records the exact delivery frontier.

## Decisions and strategies

- [Architecture decision records and design notes](../design/README.md)
- [Backend integration guides](../index.md#documentation-areas)<span data-glossary-navigation="true"> ([Backend glossary](../glossary.md#backend))</span>
- [Developer guides](../index.md#documentation-areas)
