# Implementation Roadmap

## Authority

This roadmap coordinates implementation order. It is not an architecture contract. The authoritative contract is [`ARCHITECTURE.md`](../../ARCHITECTURE.md), and it wins if this roadmap conflicts with it.

## Execution policy

Implementation advances through one active frontier at a time. Complete the current area's tasks in master-plan order before moving to the next area. Create a detailed task specification only for the next unfinished task.

Parallel work is not the default. It requires an explicit roadmap or master-plan note confirming that dependencies and affected files do not overlap.

## Ordered project areas

| Order | Project area | Status | Entry condition | Exit condition |
|---|---|---|---|---|
| 1 | [`modules/model`](modules/model/master-plan.md) | Complete | Repository and planning infrastructure are ready. | The historical selected capability milestone remains closed and focused compiler prerequisites through task 0025D are complete. |
| 2 | [`modules/trace`](modules/trace/master-plan.md) | In progress (interleaved) | Required model contracts are stable or confirmed unnecessary. | Typed trace DTO contracts and validation are complete. |
| 3 | [`modules/backend-contract`](modules/backend-contract/master-plan.md) | Complete | Foundational value-model conventions and the stable trace foundation are complete. | Backend identity and declarative requirement contracts are complete. |
| 4 | [`modules/config`](modules/config/master-plan.md) | In progress (interleaved) | Model and backend identity contracts required by configuration are stable. | Compile, prepare, run, planning-cost, and model-autotuning request contracts are complete where stable consumers justify them. |
| 5 | [`modules/planning`](modules/planning/master-plan.md) | Complete | Stable model/backend identity contracts permit the explicitly bounded capability-query interleave before config scoring is complete. | Ownership, partitioning, scoring, logical memory planning, and the selected contract-closure audit are complete. |
| 6 | [`modules/runtime`](modules/runtime/master-plan.md) | Complete | Compiler/planning handoff, backend identities, the trace foundation, and ADR 0011's per-run resource ownership/cold-binding decision are stable. | Runtime 0012, 0013, and 0014 resolved the selected cleanup, status, and architecture-enforcement findings; the Runtime closure milestone is complete. |
| 7 | [`modules/compiler`](modules/compiler/master-plan.md) | Complete | Model, config, planning, backend-contract, and trace contracts are ready for the complete compiler lifecycle; bounded task 0001 may start from the closed model graph/provenance contracts alone. | Compile artifacts, graph transformations, and autograd compilation are complete. |
| 8 | [`modules/prepare`](modules/prepare/master-plan.md) | Complete | Compiler/planning artifacts and Runtime recipe/runner contracts are stable; ADR 0010 authorizes the analysis-first staged handoff. | Shared prepare contracts and validation are complete. |
| 9 | [`backends/openblas-provider`](backends/openblas-provider/master-plan.md) | Complete | Native interop conventions needed by the provider are decided. | The low-level provider contract and validation are complete. |
| 10 | [`backends/cpu`](backends/cpu/master-plan.md) | In progress | Model, config, planning, runtime, prepare, backend-contract, trace, and OpenBLAS contracts are ready. | CPU is a conforming reference backend for the selected capability set. |
| 11 | [`modules/engine`](modules/engine/master-plan.md) | Draft | Compiler, runtime, prepare, and the CPU backend can be composed. | The public compile, prepare, and run lifecycle works end to end on CPU. |
| 12 | [`backends/metal`](backends/metal/master-plan.md) | Draft | Shared backend contracts and CPU reference behavior are stable. | Metal passes the applicable backend-conformance suite. |
| 13 | [`backends/cuda`](backends/cuda/master-plan.md) | Draft | Shared backend contracts and CPU reference behavior are stable. | CUDA passes the applicable backend-conformance suite. |
| 14 | [`extensions/onnx`](extensions/onnx/master-plan.md) | Draft | The model representation and public tensor semantics are stable. | Selected import/export mappings and compatibility validation are complete. |
| 15 | [`extensions/nn`](extensions/nn/master-plan.md) | Draft | Model semantics, compiler capture, and execution foundations are stable. | Module, parameter, buffer, train/eval, and selected layer contracts are complete. |
| 16 | [`extensions/training`](extensions/training/master-plan.md) | Draft | NN parameter contracts, config, compiler autograd, and runtime publication contracts are stable. | Backend-independent optimizer and training-session capabilities are complete. |
| 17 | [`tools/benchmarks`](tools/benchmarks/master-plan.md) | Draft | Engine and selected execution paths are operational. | Fixed reproducible workload suites and observational reporting are complete. |
| 18 | [`tools/tuning`](tools/tuning/master-plan.md) | Draft | Compiler/planning/prepare candidate boundaries, concrete backend route generators, operational engine paths, and artifact consumers are stable. | One model-autotuning workflow reuses compatible local workload results, selects a bounded complete plan, and writes explicit validated caches or prepared-plan records before runtime. |
| 19 | [`tools/cli`](tools/cli/master-plan.md) | Draft | Engine and diagnostic contracts are stable. | Selected diagnostic and execution commands are complete. |

The order above is the default delivery sequence, not a new dependency rule. Allowed and forbidden dependencies remain defined only by `ARCHITECTURE.md`.

## Current frontier

The compiler project area is Complete through
[Compiler 0006 Explicit functional gradient requests and higher-order differentiation](modules/compiler/tasks/0006-explicit-functional-gradient-requests-and-higher-order-differentiation.md)
with a bounded functional one/two-stage request, ordered
`GradientPublicationBinding` values, derivative-order metadata, and the final
`ForwardPublicationBinding` terminology correction. Its focused correction command passed 35
tests, and the single final affected-module command passed Model 127 suites/1,031 tests plus
Compiler 31 suites/208 tests with no skips, failures, or errors. No later compiler task has a
detailed specification.

The completed Runtime implementation frontier is
[Runtime 0010 Prepared runner and dynamic execution](modules/runtime/tasks/0010-prepared-runner-and-dynamic-execution.md).
It preserves `PreparedExecution` exactly as memory plan plus schedule and adds one
narrow `runtime.run` runner plus explicit executable read/write declarations. One run creates an
isolated state from dense caller inputs, cold-binds every executable, transfer, and publication
occurrence before the first action, traverses direct bound references in schedule order, applies
conservative output-validity transitions, and either constructs the whole-state `RunResult` lease
or closes the state after failure. Empty-plan/schedule/result runs, read/write overlap, repeated
runs, and cleanup failure suppression are explicit. Trace 0001–0002 are preserved but not
consumed because no Trace-owned run-payload family is current; Runtime 0010 adds no trace payload
or emission. Its focused command passed 26 tests, and the final Runtime command passed 17 suites
and 143 tests without failures, errors, or skips. Clean documentation context
`/root/runtime0010_docs` passed Javadoc, generated-page, eight-file Markdown, exact 14-path,
status, and whitespace gates. Complete
[Runtime 0011 Runtime contract closure audit](modules/runtime/tasks/0011-runtime-contract-closure-audit.md)
records [`BLOCKING_GAP`](modules/runtime/runtime-contract-closure-audit.md). The audit found a
shared-throwable cleanup defect, stale general architecture implementation-status prose, and no
Runtime-focused architecture dependency/hot-path enforcement. The combined capability checkpoint
still passed 205 suites and 1,530 tests with zero failures, errors, or skips, including Runtime's
17 suites/143 tests and architecture's 3 suites/3 tests; Runtime Javadoc and final documentation/
scope/status checks passed. The Runtime milestone remains open. Bounded
[Runtime 0012 Run-state shared-throwable cleanup](modules/runtime/tasks/0012-run-state-shared-throwable-cleanup.md)
is Complete and resolves only `RUNTIME-CLEANUP-001`. Its primary-identity guard prevents Java
self-suppression from aborting reverse attempt-all cleanup when distinct resources throw the same
exact `Throwable`; the original primary remains the thrown object and later distinct failures
remain suppressed in encounter order. The focused `RunStateTest` run passed 16 tests, and the one
final Runtime run passed 17 suites/144 tests with zero failures, errors, or skips. Clean
documentation context `019fbefd-f12e-7450-b554-81a816c3e6b8` reused that evidence, finalized
Javadoc/API/glossary/planning text, and passed Runtime Javadoc, five-file Markdown, exact scope/
status/history/later-spec, and whitespace validation. Complete
[Runtime 0013 General architecture status correction](modules/runtime/tasks/0013-general-architecture-status-correction.md)
resolves only the stale general architecture status finding through five exact replacements in
three architecture pages and the architecture-test guide. Clean documentation context
`019fc161-1298-72e1-a2bb-82ac8cbfb672` passed seven-file Markdown, preserved-history, exact
replacement, fourteen-path scope, status, later-file-absence, and whitespace gates without
running Java or test tasks. Detailed
[Runtime 0014 Runtime architecture enforcement](modules/runtime/tasks/0014-runtime-architecture-enforcement.md)
is Complete. Its dependency-free architecture suite locks the exact Runtime project edges,
requires exhaustive hot/non-hot production-source classification, and rejects Model `Operation`
or `CompiledNode` in the explicit direct-execution subset. The focused suite and final combined
checkpoint passed, resolving `ARCHITECTURE-ENFORCEMENT-001`; the Runtime milestone is Complete.
[Prepare 0003 Prepare orchestration and validation](modules/prepare/tasks/0003-prepare-orchestration-and-validation.md)
is Complete. It closes the shared layer by
projecting exact partition contexts, coordinating typed backend analysis and finalization,
assembling one complete Runtime schedule, validating source/execution/publication coverage, and
returning one exact `PreparedExecution`. Its only Runtime extension is a generic backend-created
initially-valid buffer origin for non-bindable logical splats; Runtime receives no graph or scalar
fact. It adds no concrete backend, discovery, Engine facade, physical work in Prepare, execution,
tuning, or dynamic binding.

The Prepare milestone is therefore closed. The OpenBLAS provider project area is also Complete.
CPU 0005A is now Complete and atomically supersedes the provisional CPU 0001–0005 implementation;
those five task records remain preserved as Superseded historical evidence. The current completed
CPU implementation frontier is detailed
[CPU 0006 Portable static affine views and boundary materialization](backends/cpu/tasks/0006-portable-static-affine-views-and-boundary-materialization.md).
It preserves CPU 0005H's nineteen same-typed FLOAT32/FLOAT64 unary kinds, three separate floating
classifications, and exact forty-eight-opcode inventory; CPU 0005G's extrema, clamp, Tensor power,
and canonical-BOOL logic; CPU 0005F's division and scalar-power realizations; CPU 0005E's five
types and typed carriers; and CPU 0005D's one-copy materialization and bounded optional
persistence. CPU 0005I adds preferred-species FLOAT32 parity for the exact existing eligible set,
separates vector instruction emission from pure vector math, makes ERF/GELU coefficient provenance
auditable, and advances generated compatibility from schema 8 to schema 9. CPU 0005J then adds
preferred-species floating extrema/clamp/ReLU/sign/cast, signed-integral arithmetic/extrema/cast,
canonical-BOOL logic/cast, and virtual internal-mask-to-WHERE rows while advancing compatibility
to schema 10. Every unsafe or unavailable row retains deterministic scalar fallback. CPU 0006
bounds the first layout/indexing slice to resolved-layout static affine view folding plus one
exact boundary copy for all six Model data types. It adds
`SHORT_ARRAY` to the six current generated carrier forms, yielding seven forms, solely to preserve
raw BFLOAT16 bits through existing `short[]` storage; BFLOAT16 pointwise arithmetic, conversion,
numerical semantics, and vector support remain unadvertised. Internal affine values keep their
graph/logical-memory identity without CPU declarations or Runtime slots; final resolved layouts,
including zero strides, are materialized through disjoint distinct-address writes. Schema 11 and
the cold binding/finalization path retain exact compatibility and overlap checks. The focused
implementation command and sole final CPU suite passed; the final suite recorded 29 suites and
140 tests with zero failures, zero errors, and one skipped opt-in persistence-evidence test on
OpenJDK 26.0.1+8-34. Clean documentation context `/root/cpu_0006_docs` reused that stabilized Java
evidence and finalized the CPU Javadocs, guide, glossary, and planning/status records. The former
broad 0006A row is now dependency-split: detailed
[CPU 0006A Portable pad, tile, and tensor-composition movement](backends/cpu/tasks/0006a-portable-pad-tile-and-tensor-composition-movement.md)
is `Complete`; Draft 0006A1 retains static window extraction; and Draft 0006A2 retains Gather and
one-hot with complete pre-write execution-time index validation. Functional scatter/fold,
ordering, and explicit-state random work remain Draft 0006B–0006D without detailed specs.
Historical
[CPU 0001 Capability, representation, binding, and parallel foundation](backends/cpu/tasks/0001-cpu-capability-representation-binding-and-parallel-foundation.md)
is `Superseded`; historical
[CPU 0002 Portable Class-File API generator foundation](backends/cpu/tasks/0002-portable-class-file-api-generator-foundation.md)
is `Superseded` after delivering deterministic typed specializations and fingerprints, the exact
four structural scalar/Vector and single/parallel modes, heap-array/`MemorySegment`/mixed direct signatures,
family-owned and low-level emitter seams, verified fresh hidden-class artifacts, and no generated-
artifact cache. Implementation context `019fc815-42aa-7de2-8970-a2fcab3a390e` recorded a focused
3-suite/18-test pass and the sole final CPU 9-suite/34-test pass, both with zero failures, errors,
or skips; the clean documentation pass reused that evidence because it changed no executable Java
behavior. At the CPU-0002 checkpoint, CPU advertised and executed no Model operation. Historical
[CPU 0003 Durable generated-kernel artifact store and cold loading](backends/cpu/tasks/0003-bounded-generated-artifact-cache-and-cold-finalization.md)
is `Superseded`. Historical
[CPU 0004 Typed portable analysis, specialization, and finalization](backends/cpu/tasks/0004-typed-portable-analysis-specialization-and-finalization.md)
is `Superseded`. Historical
[CPU 0005 Dense ADD and partition-sequence execution](backends/cpu/tasks/0005-dense-add-and-partition-sequence-execution.md)
is `Superseded`; it added the first narrow truthful operation route and solved maximal same-owner
partition execution as an ordered CPU-private node-kernel sequence. Implementation context
`019fd19a-4262-7580-9674-226595356fbc` recorded the sole final CPU 31-test pass, which the clean
documentation pass reused because it changed no executable Java behavior. Detailed
[CPU 0005A Atomic partition-kernel architecture reset](backends/cpu/tasks/0005a-atomic-partition-kernel-architecture-reset.md)
is Complete. Detailed CPU 0005C vector and parallel strategies are Complete. Detailed CPU 0005D
materialization/specialization/persistence evidence is also `Complete`; detailed 0005E is
`Complete` for its bounded first five-type core pointwise increment. Detailed CPU 0005F floating
division and exact scalar-power realization is `Complete`; detailed CPU 0005G and CPU 0005H are
`Complete`; detailed CPU 0005I, CPU 0005J, and CPU 0006 are `Complete`; and CPU
0006A is `Complete`; 0006A1–0017 remain `Draft` without
detailed specifications.
CPU 0003
implements a
model-independent filesystem store beneath an explicit
caller/composition-supplied trusted local root. It uses deterministic content-addressed paths,
exact structural compatibility metadata, checksummed and fully verified class bytes, and forced
temporary-file plus atomic-move publication so separate store instances and later JVM processes
can reuse the same bytes safely. Equal process-local requests single-flight; loaded artifacts are
interned only weakly with stale-key cleanup, and current prepared ownership supplies strong
lifetime. There is no strong completed LRU, age/expiry correctness, automatic disk eviction,
operation coverage, public composition, Config surface, tuning policy, or Runtime hot-path work.
Implementation context `019fc96e-494b-74f2-b6e9-5b55d649cd6c` recorded the focused
3-suite/30-test pass and the sole final CPU 10-suite/48-test pass, both with zero failures, errors,
or skips. The clean documentation pass reused that executable evidence because it changed no
executable Java behavior.

CPU 0004 connects the existing foundations through one package-private typed candidate source,
deterministic exact validation and selection, pre-assignment declarations, post-assignment
artifact finalization, immutable generated-kernel/direct-handle retention, and direct cold
binding through `CpuBorrowedBuffer`/`CpuNativeBuffer`. Backend-declared byte geometry remains
opaque during CPU analysis; shared Prepare validates it against assigned plan geometry, and no
dense-layout or materialization policy changed. Its final focused command passed three suites/24
tests, and the sole stabilized CPU module command passed 13 suites/72 tests with no failures,
errors, or skips. Clean documentation context `/root/cpu_0004_docs` reused that evidence and
passed CPU Javadoc, generated-page, five-file Markdown, exact 18-path, surface/status/excluded-
path, and whitespace validation without changing executable Java. CPU remains fail-closed and
the proof remains synthetic.

Future Model task 0026 records the prerequisite for true IEEE-754 binary16 `FLOAT16` and explicit
mixed-precision operation contracts. No backend may advertise FLOAT16 before that Model task is
complete. This does not change the active CPU frontier: CPU generated-artifact storage,
current-type analysis/finalization, and current-type family coverage may proceed independently
while remaining fail-closed for FLOAT16. Metal later prioritizes FLOAT16 and gates BFLOAT16 by
exact device and operation capability; Intel and AMD native CPU routes prioritize BFLOAT16 and
admit FLOAT16 only with exact ABI, ISA/hardware, operation, and measured-benefit evidence.

The
provider's completed foundation,
[OpenBLAS provider 0001 Library loading and required symbol binding](backends/openblas-provider/tasks/0001-library-loading-and-required-symbol-binding.md),
is Complete. That atomic foundation loads one caller-specified name or absolute path, binds the
exact FLOAT32/
FLOAT64 GEMM and get/set thread-count symbols under the standard 32-bit-`blasint` C ABI, and owns
their closeable JDK Foreign Function and Memory lookup lifetime. It invokes no native function and
adds no platform discovery, config interpretation, fallback, CPU route choice, tuning, cache,
residency, prepared execution, or backend orchestration. Its implementation context's single
final provider command passed 3 suites and 21 tests without skips, failures, or errors. Clean
documentation context `/root/openblas_0001_docs` reused that evidence and passed its recorded
Javadoc, Markdown, generated-page, exact-scope, surface/ABI/dependency/status/history/later-
specification, and whitespace checks without changing executable Java or rerunning Java tests.
Detailed
[OpenBLAS provider 0002 FLOAT32/FLOAT64 row-major GEMM invocation](backends/openblas-provider/tasks/0002-float32-float64-row-major-gemm-invocation.md)
is Complete. It adds exactly two low-level dense row-major no-transpose calls over caller-owned
native segments and the task-0001 lifetime, with fixed CBLAS constants, derived leading
dimensions, checked spans, exact overlap/lifecycle validation, raw scalar forwarding, and
deterministic fake-handle tests. CPU retains batching, broadcasting, transpose/layout conversion,
packing, storage, execution, loops, route choice, and fallback. The implementation context's one
final provider command passed 4 suites and 39 tests without skips, failures, or errors. Clean
documentation context `/root/openblas_0002_docs` reused that evidence, finalized production and
package Javadocs, the CPU guide, glossary, and planning records, and passed provider Javadoc,
five-file Markdown, generated-page, exact 11-path, surface/ABI/dependency/status/history/later-
specification, and whitespace checks without changing executable Java or rerunning Java tests.
Detailed
[OpenBLAS provider 0003 Thread control and native provider checkpoint](backends/openblas-provider/tasks/0003-thread-control-and-native-provider-checkpoint.md)
is Complete. It adds only direct positive thread-count query/control over the already-bound handles,
conservatively treats owners of one loaded binary as potentially sharing library/process state
with caller-owned coordination/restoration, and makes no guarantee across independently loaded
copies or arbitrary native consumers. It closes the provider milestone through an explicit
isolated checkpoint using one caller-supplied absolute compatible-library path. Production and
ordinary tests add no discovery or native prerequisite, completed tasks 0001–0002 remain
unchanged, and no CPU task was Ready at that provider checkpoint. Its final ordinary provider
command passed 5 suites and 50
tests without skips, failures, or errors. Clean documentation context
`/root/task_0003_impl/openblas_0003_docs` reused that evidence, finalized the affected Javadocs,
CPU guide, glossary, and planning records, and passed provider Javadoc, five-file Markdown,
generated-page, exact twelve-path, surface/package/dependency/status/history/later-specification,
and whitespace checks without changing executable Java or rerunning Java tests. Clean validation
context `/root/task_0003_native_resume` then supplied the real arm64 OpenBLAS 0.3.33 library to
the exact native command. It passed with shared observation through two owners, the fixed
SGEMM/DGEMM cases, and restoration of the original thread count of 16. Only afterward, the exact
repository/architecture capability checkpoint passed with 54 actionable tasks (2 executed, 52
up-to-date). Clean completion-documentation context
`/root/task_0003_native_resume/openblas_0003_completion_docs` reused all successful executable
and Javadoc evidence, changed only the three planning status/evidence paths, and passed the stale
Markdown, exact twelve-path/surface/status/history/later-specification, completed-task, and
whitespace checks without rerunning Java. Task 0003, the provider milestone, and the provider
project area are Complete. The later CPU planning pass selected only task 0001 as Ready.

The selected CPU target strategy remains Draft planning only. Java 26
`java.lang.classfile.CodeBuilder` is the selected current primary generator for all portable CPU
computation kernels; it is an implementation direction, not an architecture invariant. The
Class-File API plus Vector API portable route is the production baseline and always-available
semantic fallback wherever it truthfully supports the occurrence. Portable execution has exactly
scalar single-thread, scalar parallel, Vector API single-thread, and Vector API parallel modes.
Every currently selected
executable Model semantic must gain truthful portable generated coverage before the portable CPU
milestone closes; metadata-only or zero-work view occurrences need no generated computation, and
unsupported executable work fails closed and is not advertised. OpenBLAS is an optional portable
native route only for supported BLAS-compatible linear algebra, never a universal or identity-
preferred fallback.
Intel work keeps oneMKL BLAS/VML standalone routes distinct from oneDNN DNN/ML partition and
fusion routes. Apple CPU work uses Accelerate BLAS, vDSP, and vForce; MPSGraph and custom Metal
kernels remain exclusive to the separate Metal backend after Planning selects Metal ownership,
and BNNS waits for a concrete integration spike or use case. AMD work keeps AOCL-BLAS and
AOCL-LibM distinct from a later optional ZenDNN DNN-partition route, which also waits for a
concrete use case. Portable scalar and Vector API routes remain available alongside AOCL-BLAS;
AOCL-LibM is considered only for eligible sufficiently large vector-math workloads.

ARM is capability-first rather than assigned one provider. Apple Silicon may admit the Accelerate
routes above; another ARM target uses portable code generation unless a later task supplies an
explicitly verified provider. No library or route receives another `BackendId`: Planning chooses
the single CPU owner and CPU Prepare compares eligible peer routes by exact capability and whole-
plan cost.

Concrete backend analysis, not Planning, generates complete typed candidates for each operation
occurrence or partition and workload. It filters platform and provider availability, operation
attributes, data type, `Shape`, layout, exact numerical/determinism compatibility, and resource
validity before comparing native-call overhead, safe heuristics, or compatible tuning-cache
evidence. There is no global vendor-priority list, and small workloads may remain on Vector API
or scalar. Low-level provider layers own ABI and lifetime mechanics only; CPU owns capability
truth, coordination, route choice, fallback, and tuning. Current exact/default semantics exclude
vendor relaxed/fast-math candidates until a future explicit numerical policy permits them.

The selected numerical direction preserves completed Model 0018P: public `EXP` and `TANH` remain
stable mathematical requests, and no public fast variants return. The separately named
`GELU_TANH_APPROXIMATION` remains one fixed function rather than permission to choose an arbitrary
faster algorithm. Internal portable or vendor implementations may use fast algorithms only when
they meet the ordinary operation contract; genuinely relaxed behavior requires explicit caller
permission from Draft Config 0006. Hardware, providers, workload size, objectives, tuning, and
benchmarks never grant that permission and may compare only already eligible candidates.

Completed Compiler 0003A continues to provide only its currently guarded exact typed scalar
`POW(+1) -> input` bypass. Draft Compiler 0007 records future graph-level exact identities and
permission-aware algebra without rewriting completed 0001–0006 history. Exact `POW(0)` requires
the complete exceptional-value, constant-sidecar, typed shape-correct one-splat, output/publication,
phase/autograd, and descriptor proof. Tensor exponents require compiler-owned immutable uniform-
constant facts, never storage reads or factory-history inference; fractional identities such as
`POW(0.5) -> SQRT` are not assumed.

Complete CPU 0005F separately owns exact/default floating division and scalar-power realization after
the reset and pointwise foundation. It adds same-typed FLOAT32/FLOAT64 binary DIV with ordinary
right-aligned broadcast and exact-result type, same-typed scalar DIV with exact
`ScalarValueAttrs`, and direct same-typed scalar power for every exponent. Common route-independent
CPU analysis may select positive one for either zero exponent, identity for positive one, one typed
multiply for positive two, or one typed division for negative one. Reciprocal power remains
semantic `SCALAR_POW`, not public or scalar DIV. Other integral exponents remain direct because
multiply chains and exponentiation by squaring lack a universal intermediate-rounding, overflow,
and underflow proof. It does not depend on relaxed Config. Draft CPU 0017 later consumes Config
0006 for genuinely relaxed route candidates. Common analysis owns eligibility and the plan;
emitters and vendor adapters consume it. Numerical mode and realization-changing power plans
participate in specialization/cache compatibility and cold manifests, never hot-path policy
lookup. Forward and compiler-generated gradient operations share the same policy.

Complete CPU 0005G follows 0005F and owns the remaining exact algebraic/logical pointwise rows whose
current Model contracts are already sufficient: same-typed binary/scalar MIN/MAX, first-class
floating CLAMP, direct floating Tensor/Tensor POW, and canonical-BOOL AND/OR/NOT. It preserves one
first-class CLAMP instruction and deliberately keeps every new row scalar or parallel-scalar while
retaining all existing vector coverage. It also records why cross-type CAST remains fail-closed:
the Model currently permits construction but intentionally defines no numerical conversion.
The implementation expands the closed vocabulary from 22 to 31 opcodes and generator schema 6
to 7 without another route, type, public API, or shared-module change. Implementation context
`/root/cpu_0005g_impl` passed the focused 9-suite/41-test command and exactly one final
25-suite/106-test CPU command with zero failures/errors and one existing opt-in persistence-timing
skip. Clean documentation context `/root/cpu_0005g_docs` reused that Java evidence, finalized the
affected Javadocs, CPU guide, glossary, and planning records, and passed CPU Javadoc, Markdown,
exact authorized-scope, semantic/status, and whitespace gates without changing executable Java.

Complete CPU 0005H owns the distinct unary/transcendental/activation closure through an explicit
algorithm, special-value, accuracy, and vector-eligibility matrix. It keeps cross-type CAST
fail-closed because Model defines no numerical conversion contract. The task expanded the closed
vocabulary from 31 to 48 opcodes and schema 7 to 8 without another route, type, public API, shared-
module, dependency, build, or architecture change. Implementation context `/root/cpu_0005h_impl`
passed compilation, test compilation, the required focused nine-class/43-test command, and exactly
one new final 25-suite/108-test CPU command with zero failures/errors and one expected opt-in timing
skip after correcting FLOAT32 RSQRT to keep square root and reciprocal in FLOAT64 before one final
narrowing. Those counts supersede the earlier 42-test/107-test evidence. Clean documentation
context `/root/cpu_0005h_docs` reopened, reused the restabilized evidence, and finalized the
affected Javadocs/package summaries, CPU guide, glossary, and planning records. Complete CPU 0005I
now closes the intentional FLOAT64-only vector boundary without expanding the opcode inventory.
Complete CPU 0005J follows it for bounded pointwise parity before family expansion. CPU 0006 depends
on 0005J, and CPU 0009 closure explicitly includes CPU 0005A–0005J so neither the pointwise
precision nor the new eligibility matrix can be omitted silently.

CPU 0005I gives the exact closed 21-opcode eligible subset preferred-species FLOAT32/FLOAT64
parity, keeps scalar fallback and scalar tails, separates family vector instruction emission from
pure typed vector math, and records the retained Cephes/binary32 ERF/GELU provenance and bounds.
Schema 9 invalidates older generated envelopes without a migration reader. The implementation
pass's revised focused matrix passed 11 suites/62 tests; the final CPU suite passed 26 suites/117
tests with zero failures/errors and one expected opt-in persistence-evidence skip. Clean
documentation context `/root` reused that evidence and passed CPU Javadoc, Markdown, official
Oracle/Netlib links, exact 28-changed-path membership inside the 29-path map, inventory/status,
and whitespace checks without changing executable behavior or rerunning Java tests.

CPU 0005J preserves the forty-eight-opcode semantic inventory and adds the exact schema-10 typed
value-vector and virtual-mask matrix. FLOAT32/FLOAT64 extrema, clamp, ReLU, sign, and same-type
cast; INT32/INT64 modular arithmetic, signed extrema, and same-type cast; canonical BOOL logic and
same-type cast; and virtual floating predicate masks into logical masks or floating WHERE now use
their matching Java 26 preferred species when access and topology gates pass. Materialized masks,
non-scalar external WHERE conditions, general odometers, short ranges, direct scalar power, and
unsupported opcode/type pairs retain deterministic scalar or parallel-scalar fallback. The
implementation pass compiled production and tests, passed its exact twelve-class matrix as 12
suites/76 tests, passed additional focused generated-kernel/vector-math/specialization/preparation
runs, and passed its sole final CPU suite as 26 suites/125 tests with zero failures or errors and
one skipped opt-in persistence-evidence test. The clean documentation pass reused that stable Java
evidence and finalized CPU Javadocs, guide, glossary, task, master plan, and roadmap without
changing executable behavior or rerunning Java tests.

Whole-partition lowering, fusion legality/profitability, canonical CPU IR, access planning,
materialization accounting, numerical/determinism filtering, and representation planning remain
common across routes. Provider adapters do not reinterpret graphs, duplicate broadcast/fusion
planners, or take shared resource-lifetime ownership.

Draft CPU 0007 already owns reduction, scan, stable multi-pass softmax/log-softmax, statistics,
and normalization families. Draft CPU 0008 already owns heavy portable linear algebra,
convolution, pooling, attention, and loss families; its initial bounded epilogue direction is
MATMUL or convolution followed by an optional compatible bias ADD and at most one existing exact
pointwise activation or clamp, with a safe split whenever semantic, Shape/layout, publication,
resource, or numerical-order conditions do not hold. These rows remain the family implementation
owners rather than being replaced by generic fusion planning.

Before portable closure, Draft CPU 0008A adds general partition-DAG decomposition into computation
units plus bounded vertical and horizontal fusion with deterministic materialized split fallback.
Its budgets cover fan-out, indexing complexity, generated-code size, simultaneously live values,
and candidate/unit count. Draft CPU 0008B then adds a closed typed CPU-private recognizer for
selected MATMUL, convolution, and reduction epilogues plus explicit semantic kernels. It creates
no public pattern registry or domain-specific language, adds no Model kind, and never silently
recognizes decomposed softmax as stable `SOFTMAX`. Draft CPU 0008C ranks only complete legal fused
and split candidates and records typed cold accepted, rejected, and selected facts. Legality is a
fail-closed semantic/resource gate; profitability may still select a split candidate because of
code size, live-value pressure, materialization, route eligibility, or estimated complete-plan
cost.

Ordinary CPU preparation uses bounded deterministic no-measurement profitability heuristics.
The distinct future Config 0006A request inputs, Prepare 0004 opaque candidate handoff, CPU 0016
compatible workload-cache selection, and Tuning 0001–0002 measurement/selection workflow may later
measure eligible complete candidates only when model autotuning is explicitly requested before
Runtime. Autotuning is not the default fusion-profitability mechanism, and Runtime never searches,
reads or mutates tuning caches, or revises a prepared selection. Later Trace backend payloads and
tuning inspection may consume translated typed cold decision facts without taking ownership of
CPU legality, recognition, or selection. CPU 0009 depends on 0008A–0008C and closes these
foundations before CPU 0010–0015 add portable/native peer-route choices; those later routes consume
the common decomposition, recognition, and profitability decisions instead of creating competing
graph interpreters or fusion planners.

Portable generation uses family-specific typed lowerers plus shared scalar, vector, heap,
segment, range, tile, partial-reduction, and combine emitters rather than one god generator.
Operation semantics are lowered once across heap/native and scalar/vector modes. Each generated
class is specialized for one exact operation or fused-partition fingerprint and all
bytecode-relevant type, storage, layout, selected shape, execution-mode, Vector API species,
unroll/tile/tail, numerical/determinism, and target facts. Runtime identities and addresses are
excluded, and dimensions remain typed invocation parameters unless baking exact values has a
selected specialization benefit. Generated hot code performs no heap-base discovery, generic
type check, route choice, cache lookup, or storage/type/layout/vector/parallel/broadcast/operation
switch. Shared CPU parallel infrastructure, not generated classes, owns workers and coordination.

CPU storage and execution routes are orthogonal. Complete task 0001 implements aligned native
off-heap `MemorySegment` storage as the canonical interoperable run-owned representation, with
exact shared-arena lifetime, zero-size, alignment, allocation, cleanup, borrowed-storage, and
cold-binding rules. Cold binding uses direct typed arrays when an exact matching heap carrier is
observable; otherwise `CpuBufferArgument.Segment` retains the exact segment or slice without
asserting native provenance. This includes genuine native segments and JDK 26 read-only heap
segments whose `heapBase()` is empty. Later scalar Java, Vector API,
and FFM native-provider calls can use the same compatible native segment without a route-
transition copy. The Model `HostTensorStorage` contract remains unchanged and continues to admit
borrowed heap-backed and native-backed segments.

Borrowed caller heap inputs may remain heap-backed when profitable and compatible. When an exact
selected downstream native route requires native memory, CPU preparation plans at most one
necessary materialization for that representation and reuses it across compatible consumers; a
preceding Java kernel may instead write directly into the native output. Specialized opaque or
prepacked layouts, CPU-to-device transfer, incompatible layout or alignment, and explicit heap
export may still require distinct representations or materialization. This is per-value and use-
aware planning, not an all-Java versus all-native model mode or a claim that all external data is
native.

Concrete CPU preparation selects routes and representations jointly over relevant CPU dataflow
and partition uses. Exact semantics, determinism, data type, `Shape`, layout, alignment, lifetime,
and provider eligibility are hard filters. Complete valid plan cost then includes kernel time,
Java/native call overhead, allocation, copies or materialization, packing/reorder work, and
resource requirements, preventing a locally fastest kernel from forcing a worse transition plan.
Compile and Planning remain logical and backend-neutral; shared Prepare assigns and reconciles
slots and explicit materializations, while Runtime executes the prepared schedule and tracks
validity and residency.

Current CPU analysis selects the first complete valid synthetic candidate, representation,
specialization, and prepared parallel configuration supplied by its direct candidate source.
Later operation families own real candidate generation. The CPU-private finalizer added by CPU
0004, only after shared slot assignment, consults the
explicit-root generated-kernel artifact store. A valid hit supplies exact verified class bytes; a
miss emits deterministic verified bytes, publishes a complete immutable envelope atomically, then
re-reads, verifies, defines, and resolves the exact static `MethodHandle`. Runtime invokes only
that cold-resolved typed entry and performs no disk, cache, hash, validation, lookup, route, or
argument work.

The deterministic artifact key includes the full canonical specialization and every generator,
Java/Class-File, generated-class, entry-name, and descriptor compatibility fact, but excludes
model, Tensor, graph, value, slot, storage, address, run, emitter, handle, class-loader, and store
identity. A digest/path or checksum never replaces exact metadata comparison and class-shape
verification. Compatible age never invalidates an entry.

The store permits separate instances and later JVM processes sharing one trusted local root to
reuse the same class bytes. Cross-process writers may redundantly generate, but forced temporary-
file plus atomic-move publication and final revalidation prevent partial artifacts from being
defined. Checksums detect accidental corruption; they do not authenticate attacker-controlled
bytecode, so the caller owns root write isolation and administration.

Process-local equal requests use single-flight across store instances. Loaded artifacts are held
only by weak interning with stale-key cleanup; there is no strong global completed LRU, expiry,
background service, or automatic disk eviction. The current CPU portable prepared executable
strongly retains the loaded class, lookup, and handle while the recipe remains reachable.

The generated-class artifact store remains separate from the persistent tuning cache: tuning
selects compatible route/configuration evidence, while the artifact store reuses exact executable
class bytes after selection. Neither performs Runtime hot-path work.

Planning still chooses backend ownership, including CPU versus Metal; concrete backend prepare
still owns implementation routes; and OpenBLAS remains a leaf. The separately authorized
architecture synchronization now permits implementation-neutral generated JVM-bytecode CPU
computation kernels. It preserves CPU analysis ownership of lowering, specialization, route
choice, and resource declarations; CPU finalization ownership of generation and compatible
artifact reuse after slot assignment; and Runtime's prepared-execution-only role. The existing
backend-owned-lowering decision record now states this mechanism-neutral result. No dependency or
module-boundary rule changed, so architecture tests required no update.

CPU task 0002 assigns the exact package-private
`CpuPortableExecutionMode.emit(CodeBuilder, CpuKernelSpecialization,
CpuFamilyKernelEmitter)` method sole ownership of structural scalar-versus-Vector emitter
construction and dispatch. `CpuClassFileKernelGenerator` delegates to it without that switch.
The Class-File API and `CodeBuilder` remain current non-authoritative implementation selections
rather than permanent architecture dependencies.

CPU tasks 0001–0005 are Superseded by Complete CPU 0005A. Historically, CPU 0004 connected one already CPU-owned partition to the
existing typed Prepare analysis/finalization and
Runtime cold-binding boundaries while keeping every real Model operation fail-closed. Its direct
typed candidate source selects the existing carrier/access specialization, one of the exact four
portable modes, and a prepared parallel recipe; exact shared requirements precede assignment;
only the post-assignment CPU finalizer consults CPU 0003's explicit-root store; and the immutable executable
strongly retains the generated artifact and direct handle. `CpuBorrowedBuffer` remains the
non-owning `HostTensorStorage` representation/lifetime boundary for caller inputs and cold
argument classification, not an artifact cache. Bounded synthetic tests prove the lifecycle but
justify no computation or capability claim. CPU 0005 adds the exact static canonical dense
FLOAT64/FLOAT32/INT32/INT64 ADD route and ordered maximal-partition execution. CPU 0005A
replaced that provisional per-node path atomically with structured internal packages, complete-
partition units, legal/profitable fusion before exact declarations, route-independent IR,
universal start/end Class-File generation, optional persistence, and one partition executable.
CPU 0005B is Complete. It makes each ordered heap/segment carrier pattern a
code-shaping specialization with one direct entry per generated class, rather than emitting every
possible carrier combination into each artifact. After CPU 0005E, `DEFAULT` leaves its explicit
carrier list empty to select one exact segment form for each lowering-derived boundary; explicit
inputs may select another type-compatible ordered pattern. Its five generated state machines
cover dense linear, all-zero/scalar, last-axis bias, block/outer, and general odometer access.
Exact cold range spans support constant-time alias checks, and a complete bounded static
injectivity decision accepts valid interleaved positive strides while rejecting repeated writes.
Detailed CPU 0005C through CPU 0005J are Complete.
Detailed CPU 0006 and detailed CPU 0006A are Complete after splitting static movement,
window extraction, and value-dependent indexing by dependency. CPU 0006A1–0017 and the refined
Config, Prepare, Metal, and tuning rows remain `Draft` without new detailed specifications.
Completed OpenBLAS history and every completed project area remain unchanged.

CPU remains the active global project area. CPU 0005A through CPU 0006 are `Complete`, detailed
CPU 0006A is `Complete`, and CPU 0006A1 and later work remain `Draft` without detailed
specifications.
CPU 0006A delivers one exact static resolved-layout PAD, TILE, CONCAT, or STACK occurrence with
ordered one-through-sixteen composition inputs, first-occurrence unique declarations, compact
cold movement geometry, all-six-type represented-bit scalar/parallel-scalar generation, one
distinct injective output, and schema 12. Its focused command passed 8 suites/43 tests; the sole
final CPU command passed 32 suites/153 tests with zero failures or errors and one skipped opt-in
persistence test on Java 26.0.2, HotSpot 26.0.2+10-55. Clean documentation context `/root` reused
that stabilized evidence and passed CPU Javadoc plus documentation, exact-scope/status/inventory,
forbidden-change, and whitespace validation without rerunning Java tests.
CPU 0001–0005 are
historical Superseded records whose Git history
and validation evidence remain intact. Complete CPU 0004 added no public facade, registry, service locator,
default artifact-root Config API, operation family, vendor route, tuning/benchmark work, broad
cost model, shared-module/build/architecture change, or later specification. Prepare 0004 is only
a deferred bounded interleave after a concrete CPU typed-
candidate producer and tuning-artifact consumer exist; it does not reopen or reorder the
completed Prepare project area.

CPU 0005A completed the intentional architecture/capability reset before family expansion. CPU
0005B extends its exact vertical slice to fully static resolved right-broadcastable FLOAT64
ADD -> exact GELU -> MUL access while preserving one chain lowered
to one CPU-private computation-oriented execution unit, one canonical loop IR, one generated
artifact/class, one partition-level `BoundInvocation`, and one partition-level
`PreparedExecutable`. The ADD and GELU results remain graph and IR values but have no physical
slots. `LogicalMemoryPlan` continues to describe them logically; only backend buffer declarations
create assigned slots. Focused tests prove that the same class bytes and loaded compatibility identity serve two
compatible extents through primitive start/end binding; compatible extents are not default class
identity.

The permanent CPU model makes bytecode-first Class-File/Vector generation the portable
production baseline and supported semantic fallback. Metadata-only work may emit no class, native
libraries are exact-capability peer routes, and scalar reference is conformance/fail-closed
checking rather than Runtime interpretation. One route-independent IR records typed
boundary/virtual values, ordered semantics, structural access form, loop model, and stores while
excluding selected route/configuration, graph/Runtime identity, generator versions, and instance
bindings. Disk class-byte persistence is optional, disabled and unclaimed by default, and never
correctness-critical; no-root realization emits, verifies, and defines in memory and no migration
reader is planned.

The reset replaced the flat execution package with unsupported internal `memory`,
`prepare`, `lowering`, `ir`, portable `codegen.emit`, `route.portable`, `cache`, `executable`, and
`reference` packages. Java subpackages are not friends, so only minimal internal collaboration
contracts may be technically public; `CpuCapabilityProvider` remains the sole supported public CPU
API. CPU 0005A creates no native placeholder package. Later concrete Draft tasks own
`route.nativeblas` provider leaves for OpenBLAS/Accelerate/oneMKL/AOCL and `route.nativeops` leaves
for vDSP/vForce/VML/oneDNN/AOCL-LibM/ZenDNN over the shared analysis.

Completed CPU 0005B supplies one right-aligned access system over current Model
`ShapeBroadcast`/`LayoutDescriptor` contracts. Completed CPU 0005C selects all four portable
strategies cold, uses the Java 26 preferred FLOAT64 species for direct contiguous runs and scalar
broadcasts, preserves scalar tails and general-odometer fallback, and borrows an explicit
caller-owned worker group for deterministic disjoint chunks. Its final corrected CPU module run
passed 18 suites and 49 tests with zero failures, errors, or skips; clean documentation context
`/root/cpu_0005c_docs` reused that evidence because no executable Java or tests changed afterward.
At the completed CPU 0005C frontier, the next work began with direct-versus-materialized analysis
and persistence/specialization evidence; later tasks then covered broader pointwise families. They retain no hot cursor,
universal vectorization, gather, masked tail, tuning, or performance claim. Broadcast gradients
remain Compiler/Model `SUM_TO_SHAPE` plus later CPU reduction coverage.

The same task includes only the narrow shared Prepare hardening proved necessary by the audit.
For a value crossing planned partitions, the producer partition, when present, and every distinct
external consumer partition must each declare the value. Values confined to one planned
partition remain optional declarations so CPU-private same-unit virtual intermediates are legal.
Existing bindable-input, constant, and publication completeness checks are unchanged, and shared
Prepare gains no CPU fusion or materialization policy.

Runtime 0009's focused command passed 4 suites and 32 tests, and its single final Runtime command
passed 16 suites and 130 tests, with no failures, errors, or skips. Clean documentation context
`019fbe69-07e8-7a20-b132-c3b70c663d4d` finalized the affected Javadocs, Runtime/Public APIs,
focused architecture status, backend guide, glossary, and planning records without changing
executable behavior or repeating the successful tests. Runtime Javadoc, generated-page
inspection, eight-file Markdown validation, exact 18-path scope/status checks, and whitespace
validation passed.

The completed Runtime 0007 foundation adds one immutable prepared representation description
with dense caller-input occurrences and typed backend creators, one optional first-only creation
schedule occurrence, package-private
all-or-cleaned cold state creation, structural residency, and explicit independent validity for
each buffer copy. Caller inputs are borrowed and initially valid; created buffers are run-owned
and initially invalid; workspaces are run-owned scratch outside logical validity.

Its focused command passed 3 suites and 37 tests, and the single final Runtime command passed 11
suites and 94 tests, with no skips, failures, or errors. The separate clean documentation pass
finalized seven production/package Javadocs, Runtime/Public APIs, focused architecture status,
backend guide, glossary, and planning records without changing executable Java or repeating the
successful Java tests. Runtime Javadoc and generated-page inspection, the focused Runtime API
example, eight-file Markdown validation, exact surface/order/rollback/validity/hot-path/import/
mechanism/build, exact 18-path scope, synchronized status/later-specification absence, and
whitespace gates passed.

The completed Prepare implementation frontier is
[Prepare 0001 Backend partition analysis and resource declaration](modules/prepare/tasks/0001-backend-partition-analysis-and-resource-declaration.md).
It replaces the placeholder with the exact six-declaration public
`io.github.pho001.synaptik.prepare.analysis` surface plus package documentation. The immutable
`PrepareContext` preserves exact partition-node order, resolves every node input/output through a
unique projected value, requires one descriptor-matching logical-memory requirement per projected
value, accepts only fully static Shapes, and limits exact-typed logical splats to projected graph
inputs. The intentionally asymmetric projection does not add a separate requirement that every
otherwise valid projected value occur in a node input or output.

The typed backend input/plan marker roles remain opaque to shared Prepare.
`BackendPartitionPreparer` deterministically returns the exact context partition, one selected
plan, and exact `Buffer`/`Workspace` declarations. Sizes are non-negative bytes, alignments are
positive powers of two, buffer IDs and analysis-local workspace IDs are unique in their separate
domains, and neither declaration is a Runtime slot or physical resource. Analysis performs no
measurement, cache mutation, allocation, finalization, executable construction, or Runtime work.

The implementation-focused four-suite command and the single final
`./gradlew :modules:prepare:test` command each passed 11 tests with no failures, errors, or skips.
The separate clean documentation pass finalized all affected production/package Javadocs, the
backend guide, focused architecture status, glossary, and planning records without changing
executable Java or repeating the successful test suite. Prepare Javadoc, Markdown, exact public
surface, import, static-shape, scope, status, later-specification-absence, and whitespace gates
passed.

Prepare 0001 does not consume the deferred run/prepare configuration rows and changes no Gradle
edge or architecture rule. It implements the ADR 0010 analysis producer without exposing
`CompileArtifacts`, Compiler diagnostics, publication/constant plans, dynamic binding, route
vocabulary, or another Compiler-owned type to a backend.

ADR 0010 resolves the blocker with one staged handoff:

```text
Prepare 0001 analysis request/result and exact resource declarations
  -> Runtime 0002 workspace slots and final prepared-memory geometry
  -> Prepare 0002 assignment and source-to-slot association
  -> Runtime 0003 run-state/resource foundation (Complete)
  -> Runtime 0004 executable and cold-binding contracts (Complete)
  -> Prepare 0002 backend finalization against assigned slots
  -> Runtime schedule/execution work
  -> first concrete backend implementation
```

Prepare owns orchestration and the analysis boundary. A concrete backend deterministically
selects its lowering/route from an explicit Prepare projection of stable semantic and Planning
facts, fully resolved bindings, target capabilities, configuration, and compatible cached
decisions. It returns an opaque selected plan plus exact buffer/workspace byte-size and alignment
requirements. Shared preparation then assigns Runtime-owned slots, initially one distinct slot
per workspace declaration. The backend finalizes only afterward and cannot change its route or add
undeclared shared requirements.

Prepare 0001 and
[Prepare 0002 Backend partition finalization handoff](modules/prepare/tasks/0002-backend-partition-finalization-handoff.md)
are Complete. Prepare 0002 adds deterministic complete-set source validation, conservative slot
assignment, exact declaration-to-slot associations, typed backend finalization, and the minimal
`PreparedPartition(partition, executable)` result. Detailed
[Runtime 0002 Prepared memory and workspace contracts](modules/runtime/tasks/0002-prepared-memory-and-workspace-contracts.md)
is Complete.

Prepare 0002's single final module command passed 7 suites and 22 tests with no skips, failures,
or errors. Its clean documentation pass finalized all six production/package Javadocs, five
explanatory documents, and synchronized planning records without changing executable Java or
repeating the successful tests. Prepare Javadoc, the Java 26 finalizer example, nine-file
Markdown validation, exact API shape/mechanisms, exact 18-path scope, unchanged boundaries,
status, and whitespace gates passed.

Detailed
[Prepare 0003 Prepare orchestration and validation](modules/prepare/tasks/0003-prepare-orchestration-and-validation.md)
is Complete. It retains Compiler aggregates only inside shared
Prepare, keeps concrete backend-facing contexts Compiler-free, reuses task 0002 assignment and
finalization, and introduces one explicitly supplied Prepare-owned schedule assembler after the
complete prepared-partition set exists. Prepare then validates bindable-input order, exact
executable coverage, representation coordinates, and ordered forward/gradient publication before
constructing `PreparedExecution`. A narrow `InitializedBuffer(BufferCreator)` Runtime variant
lets a backend materialize a compiler logical splat into fresh run-owned initially-valid storage
without exposing `ScalarValue` to Runtime. The selected boundary fails closed for dynamic Shapes
and zero-node requested values without declared buffer geometry rather than inventing binding or
allocation policy.

Runtime 0002 keeps Runtime independent of Prepare and Model. It adds the nominally distinct
plan-local `WorkspaceSlot` plus immutable final per-`BufferSlot` and per-`WorkspaceSlot` byte-size
and alignment entries in `PreparedMemoryPlan`. The plan preserves supplied deterministic order
and requires unique slots in separate buffer/workspace domains. It imports or retains no
`PreparationResourceRequirement`, `BackendPartitionAnalysis`, `ValueId`,
`LogicalMemoryRequirement`, or `PlannedPartition`.

Complete Prepare 0002 traverses the ordered analyses and requirements, retains exact
source-to-slot associations, and constructs this Runtime geometry. Its policy is conservative:
one distinct buffer slot per distinct declared buffer value, maximum geometry for repeated value
declarations, and one distinct workspace slot per workspace declaration, with no reuse before a
liveness/interference proof exists. Physical storage, allocation, bytes ownership, pooling,
aliasing, device/residency, run-state binding/access, schedule/execution, publication, and transfer
remain outside Runtime 0002 and Prepare 0002.

ADR 0011 resolves the Runtime 0003 resource blocker. Prepared recipes are immutable/reusable;
every active complete logical run has exactly one isolated `RunState`; Runtime owns logical state,
ownership, cleanup, and later validity/residency; and concrete backends implement physical
representations and mechanics. Checked heterogeneous cold binding now creates backend-owned
typed direct-reference invocation objects before the hot path.

Detailed
[Runtime 0003 Run-state and runtime resource foundation](modules/runtime/tasks/0003-run-state-and-runtime-resource-foundation.md)
is Complete. Its bounded surface adds only nominal closeable buffer/workspace representation roles,
borrowed/run-owned buffer bindings, and array-indexed `RunState` lifecycle/access ordered like
`PreparedMemoryPlan`. It may carry multiple explicit buffer representations without promising
validity/coherence; workspace remains one run-owned backend-local representation per slot.
At that Runtime 0003 frontier, Prepare finalization, allocation, full residency, transfer,
publication/results, scheduling, and runner work remained later.

Detailed
[Runtime 0004 Prepared executable and bound invocation](modules/runtime/tasks/0004-prepared-executable-and-bound-invocation.md)
is Complete. It defines one immutable reusable `PreparedExecutable` recipe with ordered dense buffer/
representation and workspace selections, final common plan/run validation, explicit concrete-
backend compatibility hooks, and one per-run backend-owned `BoundInvocation`. The invocation
retains the exact `RunState`, rejects execution after that state closes, and stores direct concrete
typed references so its hot call performs no slot lookup, compatibility cast, graph work, backend
discovery, route/configuration search, allocation, transfer, residency decision, or publication.

Runtime 0004 adds no auxiliary binding-resource lifecycle and deliberately omits `PreparedUnit`.
Complete Prepare 0002 selects only `PreparedPartition(partition, executable)` for its current
consumer, so no distinct Runtime unit invariant exists. Detailed
[Runtime 0005 Prepared schedule contract](modules/runtime/tasks/0005-prepared-schedule-contract.md)
is Complete and resolves that question by using the exact `PreparedExecutable` as each ordered
execution occurrence. Its smallest stable schedule retains one exact `PreparedMemoryPlan`, an
immutable ordered snapshot, and a sealed plan-associated step contract with only the current
execution variant.

Transfer, materialization, and publication are not forced into Runtime 0005. Current implemented
Runtime lacks the stable representation/residency and delivery/result facts those variants
require, while Compiler publication and Planning logical-memory identities cannot become Runtime
payloads. Complete Runtime 0007 fixes only the immutable representation-description,
schedule-reachability, cold creation/rollback, structural residency, and explicit per-copy
validity foundation. Detailed
[Runtime 0008 Prepared buffer transfer and materialization schedule](modules/runtime/tasks/0008-prepared-buffer-transfer-and-materialization-schedule.md)
is Complete; Runtime 0009 is now Complete before the Draft runner.
Complete
[Runtime 0006 Prepared execution aggregate](modules/runtime/tasks/0006-prepared-execution-aggregate.md)
selects only an exact `PreparedMemoryPlan` and exact
`PreparedSchedule`, validates that the schedule reports the same plan reference, and retains both
as immutable reusable Runtime-ready state. It adds no close/run lifecycle, persistent resource,
configuration, allocation, binding, execution, publication, Prepare orchestration, or Engine
behavior. Config 0007 is therefore later runner/publication input rather than a dependency.

Complete Runtime 0007 preserves `PreparedExecution` exactly as memory plan plus schedule. Its
compatible schedule prefix carries one immutable `PreparedRepresentationPlan` that distinguishes
dense borrowed caller-input positions from concrete-backend creation callbacks. A package-private
cold Runtime operation validates all caller inputs before creation, creates buffers then
workspaces, and cleans only successfully created run-owned results in reverse order on partial
failure. `RunState` gains one explicit validity bit per resident buffer representation: borrowed
inputs start valid, created run-owned buffers start invalid, multiple or zero valid copies are
permitted, and workspaces remain scratch rather than coherent logical-value copies. It adds no
transfer route, copy, materialization, kernel, runner, publication, Config, Prepare, concrete
backend, dependency, or architecture behavior.

Complete Runtime 0008 selects one public immutable prepared transfer recipe, one per-run bound action,
and one exact-plan buffer-transfer schedule occurrence. It addresses distinct already-created
representation positions of one logical buffer. Materialization is the same transfer when it
produces an equivalent destination representation; no second kind, allocation, or route search is
introduced. Cold binding uses one exact `RunState` and direct physical references. The bound
action makes an already-valid destination a no-op; otherwise it requires the source valid,
invokes backend work exactly once, and marks only the destination valid after success. A thrown
backend failure leaves all Runtime validity unchanged.

Runtime 0008 keeps `PreparedExecution` unchanged and adds no runner/traversal, executable-output
invalidation, publication/result, Prepare orchestration, concrete backend, config/tuning/tracing,
or coherence policy. Runtime 0009–0014 are Complete. Runtime 0011 remains its unchanged historical
`BLOCKING_GAP` verdict, while Runtime 0012–0014 resolved its three selected findings and closed the
Runtime milestone. Prepare 0003 is Complete and closes the Prepare milestone. Backend
Contract remains Complete and closed. Module dependency directions are unchanged; the Runtime
architecture tests enforce existing rules rather than changing them.

Runtime 0008's focused command passed three suites and 31 tests, and its single final Runtime
module command passed 13 suites and 113 tests, with no failures, errors, or skips. The separate
clean documentation pass finalized the five affected production/package Javadocs, Runtime/Public
APIs, focused architecture status, backend guide, glossary, and planning records without changing
executable Java or repeating the successful tests. Runtime Javadoc/generated-page inspection,
current Java extension examples, eight-file Markdown validation, exact surface/direct-hot-path/
mechanism/build/scope/status checks, and whitespace validation passed.

The completed Runtime foundation still adds no Runtime physical allocation/access implementation,
graph-value conversion or source association, `PreparedUnit`, schedule consumption, runner,
public output-value access, transfer-route selection, executable-output invalidation, dynamic
eviction, concrete backend implementation, tracing emission, or Engine facade. Complete Runtime
0009 adds only Runtime-owned publication/result lifetime contracts.
The preceding
[Compiler 0005E First-order gradient coverage closure checkpoint](modules/compiler/tasks/0005e-first-order-gradient-coverage-closure-checkpoint.md)
closed the compiled-production 37-family, 107-kind, 128-signature role inventory with exact
`D`/`ND`/`FC`, family-owner, ranged-cardinality, fail-closed, and Tensor-ID evidence.
The final focused seven-suite 0005D command passed. After the MSE negative-coefficient expression
order was corrected, the replacement full compiler-module run passed 28 suites and 189 tests with
no failures, errors, or skips. No executable Java changed afterward. The separate clean
documentation pass finalized Javadocs and six affected documentation/planning paths under the
authorized 20-path ceiling.
The preceding artifact task is
[Compiler 0005 Publication, planning orchestration, and compile artifacts](modules/compiler/tasks/0005-publication-planning-orchestration-and-compile-artifacts.md).
Its completed compiler prerequisites include
[Compiler 0004 Compiler-owned pre-capture autograd and graph compilation](modules/compiler/tasks/0004-compiler-owned-pre-capture-autograd-and-combined-graph-compilation.md)
and
[Compiler 0004A Exact-composition gradient-rule extensions](modules/compiler/tasks/0004a-exact-composition-gradient-rule-extensions.md).
The immediately preceding compiler task is
[Compiler 0004B Shared-algebra cotangent normalization and local derivative rules](modules/compiler/tasks/0004b-shared-algebra-cotangent-normalization-and-local-derivative-rules.md).
The earlier completed Model producer prerequisite is
[Model 0025 Canonical TensorProducer outputs](modules/model/tasks/0025-canonical-tensor-producer-outputs.md).
The completed floating-semantics prerequisite is
[Model 0025A Portable floating comparison, extrema, and clamp semantics](modules/model/tasks/0025a-portable-floating-comparison-extrema-and-clamp-semantics.md).
Compiler 0005A is Complete with the exact 48-kind source-backed elementwise/activation inventory,
fixed derivative policies, exact coefficient bits, and request-local typed splats. Complete
[Model 0025B Binding-aware expansion](modules/model/tasks/0025b-binding-aware-expansion.md)
supplied the binding-aware prerequisite for
[Compiler 0005B](modules/compiler/tasks/0005b-reduction-scan-softmax-statistics-and-normalization-gradient-completion.md).
Compiler 0005B is Complete with binding-aware EXPAND inference/preflight, output-slot-aware
accumulation, and its exact reduction/scan/softmax/statistics/norm/normalization matrix. The exact
eighteen-path change passed 22 compiler suites/170 tests with no skips, failures, or errors plus
its independent Javadoc, Markdown, surface, scope, and status gates. Compiler 0005E is Complete;
detailed Compiler 0006 is Complete with distinct `ForwardPublicationBinding` and
`GradientPublicationBinding` terminology and unchanged publication semantics.
The exact fifteen-path 0005A change passed 22 compiler suites/159 tests with no skips, failures,
or errors plus its independent Javadoc and documentation gates.
Compiler 0004B's closed matrix covers mixed-floating cotangent Shape/DataType
normalization through ordinary `sumToShape` and `cast`, binary/scalar DIV formulas, direct-zero
FLOOR/CEIL/SIGN conventions, and ordinary/masked MEAN formulas with logical-one denominators for
static, dynamic, and expression Shapes. Forward and generated expressions retain one shared
algebra, validation contract, numerical contract, and exact optimization pipeline. Floating-
comparison-dependent and every later cohesive gradient-family task remain blocked or Draft as
recorded in the task. The final exact 16-path change passed its 136-test compiler module suite,
independent documentation review, and 1,275-test compiler transformation/autograd capability
checkpoint. Compiler 0005 keeps the graph-wide loop in Compiler, invokes Planning
through one colocated owner-selection collaboration and the existing package-owned partition and
logical-memory operations, and returns immutable compile artifacts with output-only publication,
constant, and diagnostic plans. Its final affected-module validation passed 9 Planning suites with
68 tests and 22 Compiler suites with 150 tests; the repository and architecture checkpoint passed
172 suites with 1,294 tests, including 3 architecture suites with 3 tests, with no skipped tests,
failures, or errors.

The first clean Compiler 0005 implementation context stopped before edits because the originally
planned fourth-package facade could not invoke package-private top-level operations in its three
sibling Java packages. The corrected task kept eligibility and its baseline selector
package-private, added one public owner-selection collaboration in `planning.capability`, and
widened only the audited partition and logical-memory operations in their owning packages. The
completed 37-path change includes the affected package Javadocs and visibility-locking tests;
architecture, completed Planning semantics, Compiler artifact design, and failure ordering remain
unchanged.

After 0005 and focused Model 0025A, one explicit first-order gradient-completion milestone covers
the complete current model operation inventory before higher-order work:

| Compiler task | Status | Depends on | Milestone responsibility |
|---|---|---|---|
| [0005A Derivative policy and elementwise/activation gradient completion](modules/compiler/tasks/0005a-derivative-policy-and-elementwise-activation-gradient-completion.md) | Complete | Model 0025A; Compiler 0005 | Completed the exact 48-kind binary/scalar arithmetic, selection/cast, unary, activation, comparison/logical/classification inventory with fixed tie, endpoint, discontinuity, domain, NaN, infinity, normalization, and non-differentiable-role policy. |
| [0005B Reduction, scan, softmax, statistics, and normalization gradient completion](modules/compiler/tasks/0005b-reduction-scan-softmax-statistics-and-normalization-gradient-completion.md) | Complete | Model 0025B; 0005A | Adopted binding-aware EXPAND and completed binding-dependent sum-to-Shape, products, extrema, scans, softmax/log-softmax, statistics, norms, and layer/RMS/batch normalization, including saved batch-statistic outputs and non-differentiable mask/index roles. |
| [0005C Layout, window, indexing, scatter, ordering, and stochastic gradient completion](modules/compiler/tasks/0005c-layout-window-indexing-scatter-ordering-and-stochastic-gradient-completion.md) | Complete | Models 0025C–0025D; 0005B | Completed dynamic layout/slice/composition/window rules, including retained dynamic extraction and target-relative placement obligations, Gather/scatter variants with fixed duplicate/zero/tie policies, exact sort/top-K routing, and dropout through canonical auxiliaries while preserving non-differentiable coordinates, indices, one-hot/BOOL outputs, RNG state, masks, and configuration roles. |
| [0005D Attention, convolution, pooling, and loss gradient completion](modules/compiler/tasks/0005d-attention-convolution-pooling-and-loss-gradient-completion.md) | Complete | 0005B, 0005C | Verified the implemented MATMUL/linear chain and completed two-output attention, grouped convolution, pooling, and every representable current loss role/reduction mode; one-output attention and dynamic/zero-depth index loss fail closed. |
| [0005E First-order gradient coverage closure checkpoint](modules/compiler/tasks/0005e-first-order-gradient-coverage-closure-checkpoint.md) | Complete | 0005A, 0005B, 0005C, 0005D | Added one package-private checker; closed all current 37 kind families, 107 constants, 128 signature variants, legal output slots, and ordered input roles; and proved fail-closed/Tensor-ID plus bounded transitive and connected nested-pass formula closure at the first-order checkpoint. |
| [0006 Explicit functional gradient requests and higher-order differentiation](modules/compiler/tasks/0006-explicit-functional-gradient-requests-and-higher-order-differentiation.md) | Complete | 0005E and the stable public compile/artifact boundary from 0005 | Added one immutable one/two-stage functional request, explicit/default seeds, ERROR/ZERO disconnected behavior, ordered `GradientPublicationBinding` values, and compiler-owned derivative-order metadata over the closed formula matrix without Tensor gradient state or another compile facade. |

Compiler 0005A–0005E, Model prerequisites 0025C–0025D, and detailed Compiler 0006 are Complete.
No later compiler task has a detailed specification. Family tasks
must not claim that every operation role has a gradient: BOOL, index, random-number-generator
(RNG) state, mask, and configuration roles remain intentionally non-differentiable where
applicable. Each compiler task must explicitly choose its required tie, subgradient,
discontinuity, empty-domain, or exceptional-value policy before claiming coverage; completed
Model 0025A chooses only forward semantics and no derivative policy.
Complete Model 0025B similarly broadens only existing EXPAND construction: unresolved aligned pairs
retain exact target Shape and the later source-one-or-source-equal obligation with unresolved
layout. It adds no compiler predicate or binding behavior. Complete Compiler 0005B owns that
adoption through its detailed task file.
Complete Model 0025C similarly fixes only the forward represented-value grouping and MUL/MIN/MAX
meaning for configurable Scatter Elements and Scatter-ND. It does not choose a derivative.
Complete Model 0025D similarly adds only finite length-defined extraction across unresolved selected
extents and target-relative symbolic slice placement through existing slice kinds and attributes.
Complete Compiler 0005C owns retained-constraint inference/proof, preflight, and gradients.
Compiler 0005C constructs one separate stable ARGSORT Tensor occurrence with the exact
same input, normalized axis, and direction as a one-output SORT occurrence, then use that ordinary
expression in the backward formula. This is not a hidden SORT output or public API change.
Preflight must fail closed before partial backward construction if the exact SORT/ARGSORT
signatures or attributes cannot be matched.
Completed Compiler 0004–0004B matrices remain the implemented baseline and must be preserved,
not replanned as missing work.
Dynamic and binding-dependent rules remain logical compile work, and canonical same-occurrence
auxiliary outputs remain Tensor-expression inputs rather than physical saved buffers or a runtime
tape. Task 0005E must also verify that operations used inside generated formulas are themselves
differentiable before 0006 can request differentiation through them.
The historical selected model capability milestone remains closed; this focused interleave reopens
the model plan only to supply the smallest missing prerequisite for compiler-owned pre-capture
automatic differentiation. Each `TensorProducer` now retains one canonical `Tensor` wrapper for
every output slot and exposes exact indexed lookup. `TensorFactory` constructs the producer and all
wrappers atomically, so ordinary and hidden multi-output values have stable object identities
without reconstructing wrappers from captured graph values.

Task 0025 does not add gradient or backward lifecycle state to `Tensor`, a result carrier, mutable
gradient storage, derivative rules, graph capture changes, or compiler behavior. The intentional
`Tensor -> provenance -> TensorProducer -> outputs -> Tensor` cycle is safely published through
final state and is ordinarily garbage-collectable when unreachable. This is a model contract and
Javadoc/test task, not a new dependency direction.

The post-Compiler-0005 reassessment reopens the model plan once more for task 0025A's smaller
documentation/Javadoc-only prerequisite. Floating comparisons must use ordinary ordered numeric
relations and numeric equality; pairwise/scalar MIN/MAX must propagate NaN and select the
directional signed zero; and first-class CLAMP must be exactly ordered
`MIN(MAX(input, minValue), maxValue)`. FLOAT64, FLOAT32, and BFLOAT16 use their represented values
and existing exact `ScalarValue` bits. The task adds no evaluator, operation, Tensor method, data
type, backend behavior, or derivative convention. Compiler 0005A remained Draft without a task
file at the historical completion point of Model 0025A. The later focused planning decision now
promoted only
[Compiler 0005A](modules/compiler/tasks/0005a-derivative-policy-and-elementwise-activation-gradient-completion.md)
with its detailed derivative-policy specification; that compiler task is now Complete.
The latest reassessment selected and completed only
[Model 0025B Binding-aware expansion](modules/model/tasks/0025b-binding-aware-expansion.md)
as the binding-aware Model prerequisite. It broadens the existing EXPAND expression contract only
far enough to represent compatibility that depends on later dimension bindings. Compiler-owned
deferred constraints, preflight proof, gradient construction, lowering, and execution remain
outside the model task. The prerequisite and Compiler 0005B are now Complete.

Compiler 0004 follows completed
[Compiler 0003B Compile-time constants and constant folding](modules/compiler/tasks/0003b-compile-time-constants-and-constant-folding.md),
[Compiler 0003A Exact arithmetic rewriting](modules/compiler/tasks/0003a-exact-arithmetic-rewriting.md),
[Compiler 0003 Canonicalization and forward optimization](modules/compiler/tasks/0003-canonicalization-and-forward-optimization.md),
[Compiler 0002 Captured-graph inference and validation](modules/compiler/tasks/0002-captured-graph-inference-and-validation.md),
and [Compiler 0001 Tensor expression graph capture](modules/compiler/tasks/0001-tensor-expression-graph-capture.md).
Their completed contracts remain unchanged.

Compiler 0004 is Complete with its detailed task specification, implementation, focused and full
compiler validation, and independent documentation pass. Its Model 0025 and Compiler 0001–0003B
prerequisites are complete. Its implemented architecture is
compiler-owned pre-capture Tensor-expression autograd:

```text
original forward Tensor expression DAG
  -> fail-closed derivative preflight
  -> compiler reverse traversal using public Tensor operations
  -> combined forward-and-gradient Tensor DAG
  -> one phase-aware capture
  -> inference, canonicalization, exact combined optimization, and validation
```

Compiler-owned named rule families build ordinary Tensor expressions, accumulate
contributions by original Tensor object identity, and keep all identity maps local to one compile.
Generated zeros and ones are explicit storage-free logical-splat constants. Capture receives
forward outputs, target-specific gradient result roles, the original forward producer identity
set, and explicit constants; it assigns graph IDs once and classifies original
occurrences as `FORWARD` and generated derivative occurrences as `BACKWARD`. Distinct requested
gradient targets may intentionally share one captured `ValueId`, while the final graph-output
boundary lists each distinct output value once.

The completed 0006 frontier accepts one public immutable one/two-stage functional request with
exact forward/stage-one references, aligned explicit or scalar-default cotangent seeds,
identity-unique targets from the original forward inventory, and ERROR/ZERO disconnected policy.
`FORWARD_ONLY` requires no gradient request. `FORWARD_AND_BACKWARD` and the initial
`TRAINING_STEP` require the same functional request and perform identical combined construction;
`TRAINING_STEP` adds no optimizer update.

The current general package-private entry owner is `GraphCompiler`; its exact parameter list
remains direct and has no request aggregate. It returns mode-neutral `GraphCompilation`.
`FORWARD_ONLY` has no BACKWARD nodes and empty gradient results; backward-capable modes may carry
the combined forward/backward graph. This graph-stage result is not the later-lifecycle
`CompileArtifacts`
aggregate.

The closed implemented 0004/0004A rule matrix contains same-floating-type binary ADD/SUB/MUL,
scalar ADD/SUB/MUL,
same-type WHERE branches, same-type floating CAST, NEG/EXP/EXPM1/SIGMOID/TANH/ERF, ordinary and
masked SUM, locally invertible SUM_TO_SHAPE, CUM_SUM, every floating MATMUL vector/matrix rank
pairing, CONTIGUOUS, RESHAPE, EXPAND, EXPAND_DIMS, SQUEEZE, PERMUTE, normalized SLICE and both
SLICE_UPDATE data roles, SELECT, PAD, TILE, CONCAT, and STACK under their recorded guards. Every
operation outside that implemented matrix fails preflight before any derivative Tensor identity is
allocated.

Before implementation, the task verified that six exact obsolete Java prototype paths under the
compiler production source root were present and untracked, then deleted them as its first cleanup
step without reading or adapting their contents. Their absence passed before Gradle and at final
status. They remain six removal-only paths in addition to the 32 tracked create/modify paths, for
the specified 38 touched-path ceiling.

For backward-capable modes, the combined graph, not an already optimized captured forward graph,
is the optimization unit. Only exact rules already proved by Compiler 0003, 0003A, and 0003B may
be applied; CSE remains phase-local and every changed graph is revalidated through Compiler 0002.
Compiler 0004 also owns the bounded first fail-closed rule matrix and this exact combined
optimization, so no separate cleanup task is planned. Complete Compiler 0004A adds only the
audited policy-free matrix for typed ERF, masked and locally invertible shape-target SUM,
role-aware floating MATMUL, and selected exact slice/select/pad/tile/composition adjoints. It
preserves the request and one-capture pipeline.
Complete Compiler 0004B adds the bounded shared-algebra matrix described above. It introduces no
gradient-only arithmetic, comparison, cast, exceptional-value, validation, rewrite, fold, or pass
contract. Only direct-zero FLOOR/CEIL/SIGN and all-false masked MEAN require explicit local
first-order conventions; mixed floating, DIV, and ordinary MEAN use ordinary Tensor operations
and their shared semantics.
Complete Compiler 0005 retains compile artifacts and Compiler-owned publication/planning
orchestration through its implementation. Complete Model 0025A fixes the shared floating
comparison/extrema/clamp forward contract consumed unchanged by complete Compiler 0005A.
Complete Compiler 0005B–0005E close the current-inventory first-order milestone in dependency
order, with Complete Model 0025B supplying the 0005B prerequisite and Complete Models
0025C–0025D supplying the forward-scatter and dynamic-slice prerequisites for 0005C.
Detailed Compiler 0006 now fixes the explicit one/two-stage functional request,
create-graph/order, seeds, disconnected policy, results, and derivative-order artifact contract.
It is Complete after the 0005E closure checkpoint and 0005 artifact boundary. The approved final
terminology correction renamed the Model forward binding to `ForwardPublicationBinding`, named
the Compiler result association `GradientPublicationBinding`, and left behavior unchanged. Its
focused correction command passed 35 tests; the single final affected-module command passed Model
127 suites/1,031 tests and Compiler 31 suites/208 tests without skips, failures, or errors.
Compiler 0004B's module validation, independent documentation pass, and compiler
transformation/autograd capability checkpoint all passed; the checkpoint covered 167 suites and
1,275 tests with no skipped tests, failures, or errors.

This interleave changes neither allowed dependencies nor downstream lifecycle readiness. Config
0004, Trace 0003 and later, Runtime, Prepare, backends, Engine, and training extensions remain
Draft at their existing frontiers. Compiler 0005 is the concrete orchestrator that justifies
one narrow package-cohesive Planning callable seam: owner selection composes the two internal
capability stages, while the existing partition and logical-memory operations become directly
callable in their owning packages. Its implementation preserves the audited evaluator/generator
semantics.

The preceding completed planning frontier is
[Planning 0005 Logical materialization and memory requirements](modules/planning/tasks/0005-logical-materialization-and-memory-requirements.md).
Its implementation and independent documentation pass add one immutable requirement for every
graph value plus the ordered `LogicalMemoryPlan` aggregate. Derivation consumes
`CompiledGraphModel` and ordered complete `PlannedPartition` recipes, validates exact graph-order
coverage and maximal owner runs, then retains the exact `ValueId`, `TensorDescriptor`, optional
producing partition, distinct consuming partitions, and graph-output obligation. These facts
express partition inputs/outputs, same-owner and cross-owner boundaries, graph-output
preservation, and partition-internal values without physical memory or transfer decisions.

Planning 0005 adds no public orchestration, `ForwardPublicationBinding` or
`GradientPublicationBinding` input, cost quantity, physical size, lifetime, slot, allocation,
transfer, copy, device, route, schedule, or runtime state.
[Planning 0006](modules/planning/tasks/0006-planning-contract-closure-audit.md) is now Complete.
Its clean documentation-focused
[planning contract closure audit](modules/planning/planning-contract-closure-audit.md) records a
`CLOSED` verdict: the five public declarations are sufficient and minimal, and the four current
evaluator/generator operations may remain package-private until a concrete compiler-owned
orchestrator establishes one narrow collaboration. The audit changed no Java or executable
behavior. Compiler 0005 is that later concrete consumer: it adds the narrow owner-selection
collaboration and widens only the existing partition and logical-memory operations.

Planning 0003 consumes the package-private hard-eligibility result through one colocated
package-private stateless selector. The provider-ordered eligible `BackendId` list is already the
complete candidate set. Current baseline comparison uses only the existing optional preferred
`DeviceClass`: the first matching eligible backend wins, otherwise the first eligible backend
wins, and provider order resolves ties. Empty eligibility fails terminally before scoring. No
public facade, candidate record, numeric score, shared production `OperationFamily`, workload
bucket, or cost-profile classification is needed.

Config 0004 remains Draft without a detailed specification. This cost-free baseline does not
justify profile data; the first later concrete cost-bearing planning consumer must establish the
exact backend-neutral classification and units before Config 0004 can become Ready.

Planning 0004 consumes one complete `Map<NodeId, BackendId>` assembled from per-occurrence owner
results and groups `CompiledGraphModel.nodes()` by consecutive equal owners. The current graph
already provides immutable validated topological node order, structural closure, graph boundaries,
phase classification, and multi-output nodes, so the task adds no compiler capture or
orchestration.

For this bounded frontier, adjacency means consecutive positions in the stored topological node
list. Equal owners form one maximal run; an owner transition splits it; nonconsecutive equal owners
remain separate. Graph inputs/outputs are values, and a multi-output producer remains one
indivisible node. Planning 0005 completed the next derived boundary, materialization, and
logical-memory step. At that completed Planning frontier both generators were package-private,
while the immutable partition and logical-memory recipes were public for later cross-package
lifecycle consumers. Compiler 0005 is the first concrete consumer and widens those exact two
operations without changing their semantics. No ownership row, phase split,
graph-edge component search, cost/workload classification, device/route/kernel choice, lowering,
or executable state is added.

Planning 0005 uses only the closed graph and ordered partition recipes. It validates exact
graph-order coverage and maximal owner runs before deriving values. It keeps dynamic and
expression Shapes representable by retaining `TensorDescriptor`; it adds no eager element or
byte count, lifetime, slot, allocation, transfer, copy, device, route, schedule, or residency.
`ForwardPublicationBinding` and `GradientPublicationBinding` are not Planning inputs. Compiler
places them in the compiler-owned `PublicationPlan`; `graph.outputs()` supplied only logical
preservation at the earlier Planning frontier. Planning 0006 is Complete with the selected
planning milestone, and Compiler 0005 now supplies the concrete orchestration consumer.

The performance follow-up remains Draft-only at its actual future owners. Compiler and planning
own complete valid graph and ownership candidates; shared prepare owns a future opaque
orchestration and artifact-lifecycle boundary; CPU, Metal, and CUDA own typed route candidate
generators; and `tools/tuning` owns the single two-phase model-autotuning workflow. A
representative model corpus may pre-seed the same workload cache, but no separate platform-
calibration workflow or profile remains planned. These later rows do not change the current
frontier. At Planning 0006 closure, Config 0004 and later work remained Draft without another
detailed specification. Subsequent reassessments completed Compiler 0001 capture, Compiler 0002
validation, Compiler 0003 transformation, and Compiler 0003A exact arithmetic rewriting in order.
The subsequent reassessment selected only Compiler 0003B compile-time constants/folding before
autograd, and that task is now Complete. The subsequent reassessment selected only focused Model
0025 before compiler work resumed, and that task is Complete. Compiler 0004, 0004A, and 0004B are
now Complete; none advances cost, tuning, or downstream lifecycle work. Compiler 0005 is Complete.
The preceding reassessment selected and completed only Model 0025A before compiler gradient
planning resumed, and Compiler 0005A is now Complete. The next reassessment selected and completed
Model 0025B, and Compiler 0005B is now Complete. The latest reassessment selected and completed
only Model 0025C as the forward-scatter prerequisite, then selected and completed only Model 0025D
as the remaining dynamic-slice prerequisite. Compiler 0005C and detailed Compiler 0005D are
Complete; detailed Compiler 0005E is also Complete. Detailed Compiler 0006 is Complete, and no later
compiler task has a detailed specification.
Compiler 0004 owns combined exact cleanup before 0005 partitioning/orchestration; the new
first-order milestone follows 0005, and 0006 waits for both the stable public compile/artifact
boundary and completion of the 0005E closure checkpoint.

The preceding completed planning step is
[Planning 0002 Per-query backend hard eligibility](modules/planning/tasks/0002-per-query-backend-hard-eligibility.md).
It validates complete provider/snapshot associations by equal `BackendId`, queries every
provider that survives availability and exact hard intent once, and combines its backend-level
support into one internal provider-ordered immutable `BackendId` list. Its focused and final
planning tests plus independent documentation pass are complete. It adds no public matrix,
score, profile, ownership choice, selected device, device-level capability, route, kernel,
preparation, runtime state, or execution.

[Config 0001 Backend intent foundation](modules/config/tasks/0001-backend-intent-foundation.md) and
[Config 0002 Compile modes and graph optimization configuration](modules/config/tasks/0002-compile-modes-and-graph-optimization-configuration.md)
remain Complete. Config task 0003 is also Complete; Config 0004–0008 remain ordered Draft work
without detailed specifications. Planning task 0001 remains Complete after its focused
suites, independent documentation pass, and single final 1,079-test repository suite passed.
Planning task 0002 is Complete; Planning task 0003 is Complete with its detailed specification;
Planning task 0004 is Complete with its detailed specification; Planning task 0005 is Complete
with its detailed specification; and Planning task 0006 is Complete with its detailed
documentation-only closure audit and `CLOSED` verdict. Config 0004 remains Draft because the current
baseline, same-owner grouping, and descriptor-retaining logical requirements consume no cost
classification or profile. Subsequent frontier reassessments selected bounded Compiler 0001
capture, Compiler 0002 validation, and Compiler 0003 transformation in order; all are Complete.
Compiler 0003A, Compiler 0003B, Compiler 0004, and Compiler 0004A are Complete. Focused
[Model 0025](modules/model/tasks/0025-canonical-tensor-producer-outputs.md) is Complete and
supplies Compiler 0004's canonical-output prerequisite. Compiler 0004B is also Complete after its
module tests, independent documentation pass, and capability checkpoint; Compiler 0005 is also
Complete after its module tests, independent documentation pass, and repository/architecture
checkpoint. Completed
[Model 0025A](modules/model/tasks/0025a-portable-floating-comparison-extrema-and-clamp-semantics.md)
is the latest completed interleave.
[Compiler 0005A](modules/compiler/tasks/0005a-derivative-policy-and-elementwise-activation-gradient-completion.md)
is Complete. Complete
[Model 0025B](modules/model/tasks/0025b-binding-aware-expansion.md) supplies the model prerequisite
for
[Compiler 0005B](modules/compiler/tasks/0005b-reduction-scan-softmax-statistics-and-normalization-gradient-completion.md),
which is Complete. Complete
[Model 0025C](modules/model/tasks/0025c-portable-functional-scatter-reduction-semantics.md)
supplies the forward-scatter prerequisite for Complete Compiler 0005C. Complete
[Model 0025D](modules/model/tasks/0025d-dynamic-extent-slice-extraction-and-symbolic-slice-placement.md)
supplies its remaining dynamic-slice construction prerequisite. Detailed
[Compiler 0005C](modules/compiler/tasks/0005c-layout-window-indexing-scatter-ordering-and-stochastic-gradient-completion.md)
is Complete. Detailed
[Compiler 0005D](modules/compiler/tasks/0005d-attention-convolution-pooling-and-loss-gradient-completion.md)
is Complete. Detailed
[Compiler 0005E](modules/compiler/tasks/0005e-first-order-gradient-coverage-closure-checkpoint.md)
is also Complete. Detailed
[Compiler 0006](modules/compiler/tasks/0006-explicit-functional-gradient-requests-and-higher-order-differentiation.md)
is Complete; no later compiler task has a detailed specification. Compiler
0004 owns combined exact
cleanup before 0005 partitioning/orchestration; 0005A follows completed Model 0025A and Compiler
0005, 0005B follows completed Model 0025B and Compiler 0005A, 0005C follows completed Model 0025C,
completed Model 0025D, and Compiler 0005B, 0005D–0005E follow in order, and 0006 follows the 0005E
first-order closure checkpoint plus the stable public compile/artifact boundary.
No compiler task consumes or advances Config 0004.

Trace tasks
[0001 Core trace event envelope](modules/trace/tasks/0001-core-trace-event-envelope.md) and
[0002 Model correlation identifiers](modules/trace/tasks/0002-model-correlation-identifiers.md)
remain Complete. Trace tasks 0003–0008 remain Draft without detailed specifications; the boolean
capability and hard-eligibility work does not stabilize structured rejection diagnostics or trace
payload schemas.

[Backend-contract 0001 Backend and device identifiers](modules/backend-contract/tasks/0001-backend-and-device-identifiers.md)
is Complete. It replaces only the backend-contract placeholder with open backend identity and
backend-scoped device identity values.
[Backend-contract 0002 Device classification](modules/backend-contract/tasks/0002-device-classification.md)
is Complete. It adds only the coarse `CPU`/`ACCELERATOR` category needed by later availability and
requirements.
[Backend-contract 0003 Backend availability snapshot](modules/backend-contract/tasks/0003-backend-availability-snapshot.md)
is Complete. It adds only a caller-supplied immutable association from one backend's currently
reported device identities to their classes.
[Backend-contract 0004 Declarative backend requirements](modules/backend-contract/tasks/0004-declarative-backend-requirements.md)
is Complete. Its sealed exact-backend, exact-device, and device-class hard eligibility targets,
final backend-contract module suite, independent documentation stabilization, and single final
repository capability checkpoint all passed. The selected backend-contract milestone and project
area are closed. Config task 0001 consumes but does not modify its requirement vocabulary.
Registration, discovery, refresh, capability providers, planning interpretation, preparation,
execution, and concrete backend behavior remain planned.

Trace remains In progress rather than Complete. Its tasks 0003–0008 remain ordered Draft work
without detailed specifications. The completed backend identities make only the first
producer-owned vocabulary concrete; typed backend attributes, trace-local backend/device
correlations, and lifecycle payload schemas still wait for their complete backend, config,
planning, runtime, compiler, and prepare producer contracts. Returning to those trace rows will
not make trace depend on the producer modules; producers will still translate their facts into
the trace-owned DTO leaf.

The completed model frontier is recorded below:

- [0017A Contiguous semantic kind](modules/model/tasks/0017a-contiguous-semantic-kind.md) — Complete.
- [0017B Contiguous Tensor expression](modules/model/tasks/0017b-contiguous-tensor-expression.md)
  — Complete.
- [0017C Reshape and expand semantics](modules/model/tasks/0017c-reshape-and-expand-semantics.md)
  — Complete.
- [0017D Reshape Tensor expressions](modules/model/tasks/0017d-reshape-tensor-expressions.md)
  — Complete.
- [0017D1 Expand Tensor expressions](modules/model/tasks/0017d1-expand-tensor-expressions.md)
  — Complete.
- [0017E Axis-transform semantics](modules/model/tasks/0017e-axis-transform-semantics.md)
  — Complete.
- [0017F Permute and transpose Tensor expressions](modules/model/tasks/0017f-permute-and-transpose-tensor-expressions.md)
  — Complete.
- [0017F1 Expand-dimensions and squeeze Tensor expressions](modules/model/tasks/0017f1-expand-dimensions-and-squeeze-tensor-expressions.md)
  — Complete.
- [0017G Slice semantics](modules/model/tasks/0017g-slice-semantics.md) — Complete.
- [0017H Slice Tensor expressions](modules/model/tasks/0017h-slice-tensor-expressions.md)
  — Complete.
- [0017I Pad and tile semantics](modules/model/tasks/0017i-pad-and-tile-semantics.md) — Complete.
- [0017J Pad and tile Tensor expressions](modules/model/tasks/0017j-pad-and-tile-tensor-expressions.md)
  — Complete.
- [0017K Tensor composition semantics](modules/model/tasks/0017k-tensor-composition-semantics.md)
  — Complete.
- [0017L Tensor composition expressions](modules/model/tasks/0017l-tensor-composition-expressions.md)
  — Complete.
- [0017M Unfold and fold semantics](modules/model/tasks/0017m-unfold-and-fold-semantics.md)
  — Complete.
- [0017N Unfold and fold Tensor expressions](modules/model/tasks/0017n-unfold-and-fold-tensor-expressions.md)
  — Complete; it historically included public `foldAxis`, which completed task 0018R later
  removed.
- [0018A Scalar select semantics](modules/model/tasks/0018a-scalar-select-semantics.md) — Complete.
- [0018B Scalar select Tensor expression](modules/model/tasks/0018b-scalar-select-tensor-expression.md)
  — Complete.
- [0018C Axis gather semantics](modules/model/tasks/0018c-axis-gather-semantics.md) — Complete.
- [0018D Axis gather Tensor expressions](modules/model/tasks/0018d-axis-gather-tensor-expressions.md)
  — Complete.
- [0018D1 Primitive take convenience](modules/model/tasks/0018d1-primitive-take-convenience.md)
  — Complete.
- [0018E Gather-ND semantics](modules/model/tasks/0018e-gather-nd-semantics.md) — Complete.
- [0018F Gather-ND Tensor expressions](modules/model/tasks/0018f-gather-nd-tensor-expressions.md)
  — Complete.
- [0018G Axis scatter semantics](modules/model/tasks/0018g-axis-scatter-semantics.md) — Complete.
- [0018H Axis scatter Tensor expressions](modules/model/tasks/0018h-axis-scatter-tensor-expressions.md)
  — Complete.
- [0018I Scatter-ND semantics](modules/model/tasks/0018i-scatter-nd-semantics.md) — Complete.
- [0018J Scatter-ND Tensor expression](modules/model/tasks/0018j-scatter-nd-tensor-expression.md)
  — Complete.

The latest completed implementation frontier is:

- [0018N Typed scalar value contract](modules/model/tasks/0018n-typed-scalar-value-contract.md) —
  Complete.

The latest completed implementation frontier also includes:

- [0018O Indexing taxonomy and unstack normalization](modules/model/tasks/0018o-indexing-taxonomy-and-unstack-normalization.md)
  — Complete.

The latest completed implementation frontier now also includes:

- [0018P Elementwise semantic cleanup](modules/model/tasks/0018p-elementwise-semantic-cleanup.md)
  — Complete.

Task 0018P completed one atomic migration to the exact thirteen-kind unary vocabulary with
`RECIPROCAL`/`reciprocal`, no `INV` or fast variants, portable `EXP`/`TANH` meanings, and no
aliases. It preserves typed scalar semantics. Completed task 0018T owns complete scalar arithmetic
normalization. Completed task 0018T1 separately owns floating-preserving `rsqrt`, `log1p`, and
`expm1` plus fixed-BOOL floating classifications.

Tasks [0018Q](modules/model/tasks/0018q-masked-reduction-redesign.md) and
[0018R](modules/model/tasks/0018r-slice-and-window-public-contract-cleanup.md) are complete. Task
[0018S](modules/model/tasks/0018s-tensor-factory-surface-cleanup.md) is also complete. Task
[0018T](modules/model/tasks/0018t-scalar-arithmetic-family-normalization.md) is Complete. Task
[0018T1](modules/model/tasks/0018t1-unary-numeric-gaps-and-floating-diagnostics.md) is Complete.
Task [0018U](modules/model/tasks/0018u-integral-elementwise-arithmetic-and-comparisons.md) is
Complete. Task [0018U1](modules/model/tasks/0018u1-integral-reductions-and-arg-min-normalization.md)
is also Complete. Task
Completed task [0018V](modules/model/tasks/0018v-multi-axis-and-statistical-reductions.md) closes
ordered multi-axis and statistical reduction semantics and the capability-reset checkpoint in a
cohesive 17-path change. The former broad task 0019 is decomposed. Focused
[task 0019](modules/model/tasks/0019-matmul-semantics-and-tensor-expression.md) is Complete for
MATMUL. Completed
[task 0019A](modules/model/tasks/0019a-modern-activation-semantics-and-tensor-expressions.md) added
exact GELU, fixed tanh-approximation GELU, and SiLU. Completed
[task 0019A1](modules/model/tasks/0019a1-embedding-convenience.md) adds embedding. Completed
[task 0019A2](modules/model/tasks/0019a2-one-hot-encoding.md) adds first-class one-hot encoding
after that activation task. ReLU remains current from completed tasks 0014C–0014D. The former
broad RNG/dropout frontier is split: completed
[task 0019B](modules/model/tasks/0019b-explicit-graph-rng-state-foundation.md) owns the explicit
graph RNG state foundation, and completed
[task 0019B1](modules/model/tasks/0019b1-explicit-graph-dropout-construction.md) owns dropout with
explicit state. The former sorting/top-K row is split: completed
[task 0019C](modules/model/tasks/0019c-sort-and-argsort.md) owns full stable sort/argsort, and
completed [task 0019C1](modules/model/tasks/0019c1-top-k-values-and-indices.md) owns genuine
multi-output top-K. Its final model suite, independent documentation review, Javadoc, runnable example,
Markdown, exact 18-path audit, repository checkpoint, status, and whitespace validation passed.
Completed [task 0019D](modules/model/tasks/0019d-linear-convenience.md) adds explicit linear
composition. Completed [task 0019E](modules/model/tasks/0019e-scaled-dot-product-attention.md)
adds first-class attention semantics, immutable attrs, four receiver overloads, API-locking tests,
and documentation in one cohesive 17-path change. The former broad convolution/pooling frontier
is split without renumbering later work: focused
[task 0020](modules/model/tasks/0020-nchw-conv2d-semantics-and-tensor-expressions.md) is Complete,
while the former pooling follow-up is split into completed
[task 0020A](modules/model/tasks/0020a-nchw-max-pool2d-semantics-and-tensor-expression.md) for max
pooling and completed
[task 0020A1](modules/model/tasks/0020a1-nchw-average-pool2d-semantics-and-tensor-expression.md)
for average pooling. The former broad 0021 normalization row is now split: focused
[task 0021](modules/model/tasks/0021-layer-normalization-semantics-and-tensor-expressions.md) is
Complete for layer normalization. Completed
[task 0021A](modules/model/tasks/0021a-rms-normalization-semantics-and-tensor-expressions.md) adds
RMS normalization. Focused
[task 0021B](modules/model/tasks/0021b-batch-normalization-inference.md) is Complete for stateless
five-input inference. Task 0021C is Complete. The former broad loss row is now split into completed
[task 0022](modules/model/tasks/0022-mean-squared-error-loss.md), completed
[task 0022A](modules/model/tasks/0022a-dense-target-categorical-cross-entropy-with-logits.md) for
dense-target categorical cross-entropy with logits, and completed task 0022B for index-target
categorical cross-entropy with logits. Task 0022A is Complete;
task 0023 is Complete with its detailed specification and final matrix. Tasks 0023A–0023E are
Complete with their detailed specifications. Task
[0023F](modules/model/tasks/0023f-scaled-dot-product-attention-weights-output.md) is Complete with
its detailed same-occurrence attention-output specification before established task 0024. The
model capability and contract closure audit is Complete with a `BLOCKING_GAP` verdict; focused
[task 0024A](modules/model/tasks/0024a-graph-value-tensor-status-javadoc-correction.md), the
bounded `GraphValue` Tensor-status Javadoc correction, is Complete. It resolved the audit's sole
blocker without changing Java behavior and closed the selected model capability milestone.
Focused task 0025 is a later compiler-foundation interleave and does not reopen that historical
capability audit or its verdict. Complete task 0025A is a second bounded interleave that clarifies
only the already-selected portable comparison/extrema/clamp forward contract before Compiler
0005A; it does not change the historical capability verdict or choose derivative behavior.
Complete task 0025B is a third bounded interleave that broadens only existing EXPAND construction
for binding-dependent compatibility before Compiler 0005B; it did not itself add compiler
constraints, gradient behavior, lowering, or execution and does not change the historical
capability verdict. Compiler 0005B subsequently adopted that prerequisite.
Complete task 0025C is a fourth bounded interleave that completes only configurable Scatter Elements
and Scatter-ND MUL/MIN/MAX forward represented-value semantics before Compiler 0005C. It does not
change the historical capability verdict, execute values, or select a derivative.
Complete task 0025D is a fifth bounded interleave that adds only finite length-defined slice
extraction across unresolved selected extents and target-relative symbolic slice placement before
Compiler 0005C. It does not change the historical capability verdict or add binding, compiler,
lowering, execution, or derivative behavior.
Task 0023B's focused 15-suite run passed 124 tests, its single final model suite passed 981 tests
across 125 suites, and the separate documentation pass validated model Javadoc, the executable
example, Markdown, exact 26-path scope, the 190-method public Tensor surface, and synchronized
status.
Task 0023C's focused 15-suite run passed 139 tests, its single final model suite passed 996 tests
across 126 suites, and the separate documentation pass validated model Javadoc, the runnable Java
26 update/crop metadata example, Markdown and official references, exact 27-path scope, the
192-method public Tensor surface, and synchronized Complete/Draft status.

Completed [task 0023D](modules/model/tasks/0023d-public-fold-axis-and-dynamic-window-transforms.md)
preserves canonical rank-three im2col/col2im geometry and selects one canonical symbolic
Dimension-product form rather than a second non-flattened window representation. It restores
public `foldAxis`, generalizes unfold2d/fold2d to exact dynamic channel and spatial formulas, and
adds one exact typed-padding UNFOLD2D variant while preserving direct conceptual-zero padding and
all architecture boundaries.

Task 0023D's focused 17-suite run passed 175 tests, and its single final model suite passed 1,008
tests across 126 suites with no failures, errors, or skips. Independent documentation review
finalized all nine affected production Javadocs, Tensor/Compile APIs, glossary and planning
records, then validated model Javadoc, a runnable Java 26 metadata example, generated API pages,
the 194-method public Tensor surface, Markdown, exact 33-path scope, status, and whitespace.

Completed [task 0023E](modules/model/tasks/0023e-cumulative-scan-normalization-and-product.md)
atomically replaces the sum-only scan type/helper names with one `CUM_SUM`/`CUM_PROD` family,
preserves the two public `cumSum` forms, and adds two public `cumProd` forms. Its focused run passed
44 tests across five suites, and its single final model suite passed 1,008 tests across 126 suites
with no failures, errors, or skips. Independent documentation review finalized the affected
Javadocs, Tensor/Compile APIs, glossary and planning records, then validated model Javadoc, Java 26
API reflection, generated API pages, the 196-method public Tensor surface, Markdown, exact 33-path
scope, synchronized Complete/Draft status, and whitespace without rerunning executable Java tests.

Completed [task 0023F](modules/model/tasks/0023f-scaled-dot-product-attention-weights-output.md)
preserves the four one-output attention methods and adds four explicit output-plus-normalized-
weights methods, one two-component public result, and one shared two-output producer form under
the existing kind. Its focused run passed 40 tests across five suites, and its single final model
suite passed 1,016 tests across 127 suites with no failures, errors, or skips. Independent
documentation review finalized the four affected production Javadocs, Tensor/Compile APIs,
glossary and planning records, then validated model Javadoc, a runnable Java 26 metadata example,
reflection/generated API shape, the 200-method public Tensor surface, Markdown, exact 27-path
scope, synchronized Complete/Draft status, and whitespace without rerunning executable Java tests.

[Task 0023](modules/model/tasks/0023-adjoint-expressibility-audit.md) is the planning-only
adjoint-expressibility frontier after the completed post-0022B checkpoint. Its
[final matrix](modules/model/adjoint-expressibility-audit.md) selects six reusable public
capability gaps and no compiler-only semantic gap: 0023A binding-aware sum-to-Shape for deferred
MATMUL/attention batch binding, 0023B Gather-compatible axis scatter-add for unresolved gathered
extents, 0023C signed slice placement plus target-relative dynamic crop, 0023D public foldAxis plus
redesigned dynamic/configurable 2D windows, 0023E cumulative product,
and 0023F same-occurrence attention weights. Current Scatter Elements and Scatter-ND serve Gather
Elements and Gather-ND exactly; typed scalar expansion supplies dynamic constants; max-pool
selection requires no separate indices output. Positive-static-depth Gather also composes through
one-hot selection and reduction. Dynamic 2D windows must not assume the current flattened
`outputHeight*outputWidth` Shape can multiply two unresolved extents. The checkpoint evidence
remains 966 root tests
across 124 suites, model Javadoc, 188 public Tensor methods, and 657-link/176-anchor documentation
validation. The audit implements no gradient, compiler, execution, backend/runtime, Gradle,
dependency, or architecture change. Completed task 0023A adds the existing SUM kind's exact
`SumToShapeAttrs` variant and one public `sumToShape(Shape)` metadata expression. Its focused
14-suite run passed 131 tests, its replacement final model suite passed 977 tests, and the separate
documentation pass validated model Javadoc, examples, Markdown, exact 25-path scope, the 189-method
surface, and synchronized status. It adds no compiler adoption, binding implementation, gradient,
execution, backend/runtime, dependency, Gradle, or architecture change.

Task 0021A adds one distinct RMS-normalization kind and typed attributes, exact no-scale and
scale-only receiver methods, uncentered mean-square semantics, and one-output provenance. Its
implementation context passed the exact focused command and final 908-test model suite.
Independent documentation review finalized Javadocs, Tensor/Compile APIs, glossary and planning
records after model Javadoc, 607-link/165-anchor Markdown, official-reference, exact public-
surface, exact 19-path, status, formatting, and whitespace validation passed.

Task 0019D adds conventional `[outFeatures, inFeatures]` weight-transposed MATMUL plus optional
exact rank-one bias as visible PERMUTE -> MATMUL -> optional ADD composition. Complete local
validation precedes intermediate IDs; no-bias creates two wrappers and returns MATMUL, while bias
creates three and returns ADD. Its implementation context passed 60 focused tests and one final
836-test/105-suite model run. Independent documentation review finalized the two production
Javadocs, Tensor/Compile APIs, glossary, planning records, runnable producer-chain example, and
generated-Javadoc, public-surface, Markdown, exact-scope, status, terminology, and whitespace
validation.

Task 0018U added same-category numeric promotion, selected modular INT32/INT64 ADD, SUB, MUL, MIN,
and MAX Tensor and exact-scalar construction, and all six signed-integral comparisons without a
new public Tensor method or operation kind. At that historical frontier, integral DIV, POW, range
CLAMP, reductions, and arg-min remained deferred; task 0018U1 has since completed the selected
reduction and arg-min work. The final task-0018U model suite passed 734 tests across 90 suites; its
separate documentation pass finalized Javadocs and the Tensor/Compile API, glossary, capability
baseline, task, master plan, and roadmap without repeating the successful Java suite.

Task 0018T1 added first-class `rsqrt`, `log1p`, and `expm1` metadata plus separately typed
`isFinite`, `isNaN`, and `isInf` BOOL classifications. Its implementation context passed the
exact focused command and final model suite. Independent documentation review finalized the five
affected production Javadocs and seven documentation/planning files after model Javadoc, the
runnable transform/classification metadata example, generated-page and exact-surface checks,
493 local links including 139 anchors, exact 18-path/status checks, formatting, and
`git diff --check` passed.

Task 0018S narrowed TensorFactory to construction, import, constants, and integer ranges; made
stateless `TensorRandoms` the sole public explicit-source random owner; and moved prefix fixture
preparation out of production. Its implementation context passed 58 focused tests and the
715-test root checkpoint. Independent documentation review finalized affected Javadocs, Tensor
API, glossary, planning records, a runnable public example, and generated-Javadoc, Markdown,
surface, exact-scope, status, terminology, and whitespace validation.

Task 0018T completed the parallel seven-operation Tensor/binary and Tensor/scalar arithmetic
vocabulary, pairwise `minimum`/`maximum` naming, first-class range CLAMP, and scalar MAX/MIN
one-bound conveniences. The implementation context passed the six-suite focused command and all
715 model tests across 88 suites. Independent documentation review finalized five Javadocs,
Tensor/Compile APIs, glossary, capability/task/master/roadmap records, generated Javadoc, a
compiled Java 26 surface example, Markdown, removed-vocabulary, exact 18-path, status, formatting,
and whitespace validation. Explicit authorization added only `TensorNumericReductionTest` to the
original 17-path scope after its stale pairwise calls caused the initial focused compilation to
fail.

Task 0018Q removed heuristic mask-axis placement, retained first-class two-input masked SUM/MEAN,
and requires ordinary right-aligned broadcasting to produce exactly the input Shape. Callers make
other axis intent visible with Shape/rank edits. The implementation context passed the exact
focused command and all 720 model tests across 88 suites; independent documentation review
completed the four Javadocs, Tensor/Compile APIs, glossary and planning synchronization, runnable
explicit-alignment example, generated-Javadoc, Markdown, official-link, exact-scope, status,
terminology, and whitespace validation.

Task 0018O finalized canonical Gather, Gather Elements, Scatter Elements, and repeated-SELECT
unstack without compatibility aliases or first-class unstack/fixed-add kinds. Its implementation
context passed the exact focused command and the 725-test/88-suite model suite; independent
documentation review completed Javadocs, APIs, glossary, planning synchronization, and final
surface, Markdown, exact 29-path, status, and whitespace validation.

The capability-reset audit found that operation validity, shared multi-output provenance,
symbolic extent arithmetic, typed scalar values, and several provisional legacy-derived APIs must
be hardened before linear algebra. Tasks 0018K–0018V now form that ordered reset. Tasks 0018K
through 0018T1, task 0018U, task 0018U1, linked task 0018V, and focused MATMUL task 0019 are
complete. Tasks 0019A, 0019A1, and
[task 0019A2](modules/model/tasks/0019a2-one-hot-encoding.md) and task 0019B are also complete. Task
0019B owns only explicit graph RNG state, with dropout separated into completed task 0019B1.
Task 0019C is now complete for stable sort/argsort. Its completed
[0019C1 follow-up](modules/model/tasks/0019c1-top-k-values-and-indices.md) owns top-K. Completed
[task 0019D](modules/model/tasks/0019d-linear-convenience.md) owns linear as explicit
PERMUTE/MATMUL/optional-ADD composition. Completed
[task 0019E](modules/model/tasks/0019e-scaled-dot-product-attention.md) owns first-class attention;
tasks 0020, 0020A, 0020A1, 0021, and
[0021A](modules/model/tasks/0021a-rms-normalization-semantics-and-tensor-expressions.md) are
Complete.

The former broad task 0017 is decomposed into tasks 0017A–0017N so parameterless contiguous
meaning, public expression construction, shape/view transformations, slicing, pad/tile,
composition, and unfold/fold contracts can be implemented and validated independently. Tasks
0017A–0017N have detailed specifications and are complete. The former broad task 0018 is now
decomposed into focused tasks 0018A–0018J for select, gather, and functional-scatter semantics and
expressions. Tasks 0018A through 0018J, tasks 0018K through 0018T, and task 0018T1 are complete;
task 0018U, task 0018U1, task 0018V, focused MATMUL task 0019, task 0019A, and task 0019A1 are
Complete. Tasks 0019A2, 0019B, 0019B1, 0019C, 0019C1, 0019D, and 0019E are also Complete. Task
0020, 0020A, 0020A1, 0021, and 0021A are Complete. Task
[0021B](modules/model/tasks/0021b-batch-normalization-inference.md) is Complete. Focused
[0021C](modules/model/tasks/0021c-batch-normalization-training-and-statistic-transition.md) is
Complete. Tasks 0022, 0022A, and 0022B are Complete. Task 0023 is Complete with its detailed
audit specification and result artifact. Task 0023A is Complete with its detailed specification.
Tasks 0023B, 0023C, 0023D, 0023E, and 0023F are Complete with their detailed specifications,
while task 0024 is Complete with its closure artifact. Task 0024A is Complete after correcting the
sole stale `GraphValue` current-versus-planned Javadoc sentence. The selected model capability
milestone is closed.

Task 0018R selects normalized start/length/signed-step slice attributes rather than a negative-end
sentinel, retains the general array primitive, adds explicit-step `sliceAxis` and one-occurrence
`flip(int... axes)`, and leaves every negative-step result layout-unresolved under the current
non-negative-stride descriptor. It removes public `Tensor.foldAxis` without an alias while
retaining `WindowTransformKind.FOLD_AXIS` and `FoldAxisAttrs` as public Java semantic contracts
without a public Tensor receiver/construction method.
[Task 0023](modules/model/tasks/0023-adjoint-expressibility-audit.md) selected completed task 0023D
to restore the generally useful public overlap-add primitive and separately generalize 2D windows.
Public `unfold` remains unchanged; public `foldAxis`, dynamic `unfold2d`/`fold2d`, and exact typed
unfold padding are now current.

The task-0018R implementation context passed the exact 78-test focused contract command and all
715 model tests across 88 suites. Independent documentation review finalized all seven affected
production Javadocs, Tensor and Compile API references, glossary, capability baseline, task,
master plan, and roadmap. Model Javadoc, the Java 26 slice/flip metadata example, generated pages,
Markdown links and anchors, the two official URLs, exact eighteen-path scope, public-surface and
removed-vocabulary checks, synchronized status, fences, newlines, terminology, whitespace, and
`git diff --check` passed. Public unfold/unfold2d/fold2d behavior, architecture, dependencies,
build configuration, and every other module remain unchanged.

Task [0018M](modules/model/tasks/0018m-symbolic-extent-expressions.md) is complete with canonical
checked symbolic extent arithmetic, explicit floor/ceiling division, identity-based bounded
unknowns, non-static Shape inspection, readable diagnostics, and conservative structural
broadcasting. Independent documentation review finalized the affected Javadocs, Tensor API,
glossary, capability baseline, task evidence, model master plan, and roadmap after the reused
765-test model result, model Javadoc, runnable example, public-surface, Markdown, exact 17-path,
status, and whitespace checks passed. No Tensor operation adoption, binding/evaluation,
compiler/prepare/runtime/backend behavior, dependency, build, or architecture change was added.

Task [0018M1](modules/model/tasks/0018m1-dynamic-extent-adoption.md) is complete. Pad now applies
canonical before-then-after symbolic addition, tile applies one canonical multiplication per
axis, and concat encounter-order folds every selected extent through symbolic addition. Neutral
operations preserve exact Dimension references, checked static/coefficient/offset overflow still
precedes Tensor allocation, and layouts remain unresolved. Independent documentation review
finalized helper and public Tensor Javadocs, Tensor API, glossary, capability baseline, task
evidence, model master plan, and roadmap after the reused 766-test/88-suite model result, model
Javadoc, Markdown, exact eleven-path, status, and whitespace checks passed. The original ten-path
cap was explicitly expanded only to correct stale public Tensor dynamic-rejection Javadocs;
signatures and executable behavior did not change.

Task [0018N](modules/model/tasks/0018n-typed-scalar-value-contract.md) is complete. One immutable
`ScalarValue` now preserves exact typed primitive bits for all six current data types. Scalar,
clamp, and padding attributes retain it, and public Tensor expression construction requires exact
receiver/value type equality while retained `double` overloads mean exact FLOAT64. The existing
TensorFactory and BFLOAT16 conversion surfaces remain unchanged. The implementation context passed
57 focused tests and the final 770-test model suite; independent documentation review finalized
Javadocs and the Tensor/Compile API, glossary, capability, task, master-plan, and roadmap records.

Task [0018K](modules/model/tasks/0018k-operation-signature-and-construction-hardening.md) is
complete with exact family-owned attribute variants and inclusive local input/output occurrence
cardinality. `Operation` now rejects incompatible kind/attributes pairs and derives its signature;
`CompiledNode` validates final local counts after its existing list checks. Independent
documentation review finalized affected Javadocs, Tensor API, Compile API, glossary, task
evidence, model master plan, and roadmap after the final 743-test/86-suite model run, model
Javadoc, 413-link/110-anchor Markdown, fence/final-newline, scope, and whitespace checks passed.
Operand-aware and graph-wide validation, compiler/backend/runtime behavior, and shared
multi-output Tensor provenance remained deferred to task 0018L.

Task [0018L](modules/model/tasks/0018l-shared-multi-output-tensor-provenance.md) is complete with
one identity-based immutable `TensorProducer`, indexed `TensorProvenance`, exact descriptor-slot
agreement, package-private single/multi-output construction, and atomic migration of all current
single-output helpers. Independent documentation review finalized affected Javadocs, Tensor API,
Compile API, glossary, task evidence, model master plan, and roadmap after the final
749-test/87-suite model run, model Javadoc, compiled example, Markdown, scope, status, and
whitespace checks passed. Current unstack remains independent one-output producers; production
multi-output operations, compiler capture, graph-local identity, gradients, backend behavior, and
execution remain deferred.

Task [0018A](modules/model/tasks/0018a-scalar-select-semantics.md) is complete with the exact
`SELECT` identity and normalized scalar axis/index attributes. Its independent documentation
review passed focused 9-test, all 638-model-test/75-suite, model-Javadoc, root-test,
javap/reflection/import/generated-page, Markdown, exact eight-path, synchronized-status, and
no-0018B-spec checks. Public Tensor construction and every cross-layer behavior remain deferred.

Task [0018B](modules/model/tasks/0018b-scalar-select-tensor-expression.md) is complete with exact
public scalar-coordinate normalization, axis removal, conditional logical-view geometry, and
fresh one-input provenance. Its independent documentation review finalized Tensor/helper
Javadocs, Tensor and Compile API status, glossary terminology, task evidence, model master plan,
and roadmap. Value selection, physical aliasing, gradients, compiler capture/canonicalization,
materialization, backend behavior, and execution remain deferred to their owning layers.

Task [0018C](modules/model/tasks/0018c-axis-gather-semantics.md) is complete with exact
`GATHER`, `GATHER_AXIS`, and `TAKE_ALONG_AXIS` meanings plus one shared normalized non-negative
axis attribute. Its independent documentation review finalized both production Javadocs, Tensor
API, glossary, task evidence, model master plan, and roadmap after focused 9-test, all
657-model-test/77-suite, model-Javadoc, root-test, bytecode/reflection/import/generated-page,
Markdown, exact eight-path, synchronized-status, and no-0018D-spec checks passed. Task 0018D now
owns public Tensor construction, index-type/Shape validation, result metadata, and provenance.
Index-value bounds, gradients, compiler behavior, lowering, and execution remain deferred.

Task [0018D](modules/model/tasks/0018d-axis-gather-tensor-expressions.md) is complete with exact
public `gather`, `gatherAxis`, tensor-index `take`, and `takeAlongAxis` expressions. Its independent
documentation review finalized Tensor/helper Javadocs, two explicitly authorized semantic
Javadoc timing/bounds corrections, Tensor and Compile API references, glossary, task evidence,
master plan, and roadmap. Construction validates INT32/INT64 index metadata and the distinct
structural Shape rules, preserves data metadata with unresolved layout, and records fresh ordered
provenance without reading values or defining bounds, gradients, compiler, backend, or execution
behavior.

Task [0018D1](modules/model/tasks/0018d1-primitive-take-convenience.md) is complete with exact
public `take(int, int[])` adaptation. It copies one non-empty caller array into an independent
dense rank-one INT32 index Tensor before delegating once to tensor-index take, retaining
GATHER_AXIS semantics and exact `[data, generatedIndices]` provenance. Its independent
documentation review finalized Tensor/helper Javadocs, Tensor and Compile API status, glossary,
task evidence, model master plan, and roadmap after all required validation passed. Index bounds,
gradients, compiler capture, backend behavior, and execution remain outside this task.

Task [0018E](modules/model/tasks/0018e-gather-nd-semantics.md) is complete with exact
`GATHER_ND` meaning and normalized non-negative batch-dimension attributes. Its independent
documentation review finalized both production Javadocs, Tensor API, glossary, task evidence,
model master plan, and roadmap after focused 9-test, all 684-model-test/80-suite, model-Javadoc,
root-test, bytecode/reflection/import/source/generated-page, Markdown, exact eight-path,
synchronized-status, and no-0018F-spec checks passed. Task 0018F now completes the public Tensor,
rank/batch/tuple-depth/index-type/result-Shape, and provenance work; gradients, compiler behavior,
lowering, bounds, and execution remain deferred to their owning layers.

Task [0018F](modules/model/tasks/0018f-gather-nd-tensor-expressions.md) is complete with exact
zero-batch and explicit-batch public Gather-ND expressions. Its independent documentation review
finalized Tensor/helper and the two authorized semantic temporal Javadocs, Tensor and Compile API
references, glossary, task evidence, model master plan, and roadmap after focused 10-test and
14-test suites, all 694 model tests across 81 suites, model Javadoc, root tests, executable
example, bytecode/reflection/import/source/generated-page, 417-link/121-anchor,
fence/whitespace/newline, exact twelve-path, synchronized-status, semantic-bytecode-equivalence,
and no-0018G-spec checks passed. Construction validates exact integral index metadata, ranks,
structural batch prefixes, and static positive tuple depth, derives exact prefix-plus-suffix Shape
including canonical scalar, and records fresh ordered provenance without reading values. Index
bounds, gradients, compiler behavior, materialization, lowering, backend behavior, and execution
remain deferred.

Task [0018G](modules/model/tasks/0018g-axis-scatter-semantics.md) is complete with exact
`SCATTER_ADD`, `SCATTER_AXIS_ADD`, and `SCATTER_ELEMENTS` meanings plus reusable
`NONE`/`ADD`/`MUL`/`MAX`/`MIN` reduction vocabulary and explicit scatter-elements attributes. The
two fixed-add kinds reuse unchanged `IndexAxisAttrs`. Its independent documentation review
finalized production Javadocs, Tensor API, glossary, task evidence, model master plan, and roadmap
after focused 12-test, all 706-model-test/82-suite, model-Javadoc, root-test,
javap/reflection/import/source/generated-page, Markdown, exact ten-path, synchronized-status,
semantic-bytecode-equivalence, and no-0018H-spec checks passed. Public Tensor construction,
input-aware type/Shape/axis validation are now complete in task 0018H. Index bounds and duplicate
detection, gradients, compiler behavior, lowering, and execution remain deferred.

Task [0018H](modules/model/tasks/0018h-axis-scatter-tensor-expressions.md) is complete with four
public Tensor methods and one field-free eleven-method helper for reduced-rank fixed-add scatter,
rank-changing fixed-add axis scatter, and configurable same-rank scatter-elements. Its independent
documentation review finalized Tensor/helper and three authorized semantic Javadocs, Tensor and
Compile API references, glossary, task evidence, model master plan, and roadmap after focused
10-test and 14-test suites, all 716 model tests across 83 suites, model Javadoc, root tests,
executable example, bytecode/reflection/import/source/generated-page checks, 425-link/127-anchor
Markdown validation, exact thirteen-path and synchronized-status checks, semantic-bytecode
equivalence, and no-0018I-spec checks passed. Index bounds, duplicate detection, writes,
reductions, gradients, compiler behavior, lowering, backend behavior, and execution remain
deferred.

Task [0018I](modules/model/tasks/0018i-scatter-nd-semantics.md) is complete with exact
`SCATTER_ND` semantics and immutable normalized batch-count plus shared-reduction attributes.
Tuple depth remains the final indices Dimension, updates follow the Gather-ND result Shape, and
the functional result keeps data Shape. Its independent documentation review retained both
production Javadocs and finalized Tensor API, glossary, task evidence, model master plan, and
roadmap after focused 9-test, all 725-model-test/84-suite, model-Javadoc, root-test,
javap/reflection/import/source/generated-page, 425-link/131-anchor, fence/whitespace/newline,
exact eight-path, synchronized-status, and no-0018J-spec checks passed. Public Tensor
construction, input-aware validation, values, gradients, lowering, and execution remain outside
this semantic task.

Task [0018J](modules/model/tasks/0018j-scatter-nd-tensor-expression.md) is complete with three
public overloads and one field-free eleven-method helper for zero-batch replacement, zero-batch
explicit reduction, and explicit reduction plus batch count. The shared path validates exact
types, reduction eligibility, ranks, batch prefix, tuple depth, and updates Shape before creating
one fresh unresolved-layout result with exact data metadata and ordered provenance. Independent
documentation review finalized semantic and public Javadocs, Tensor and Compile APIs, glossary,
task evidence, model master plan, and roadmap after focused 10-test, Tensor API 14-test, all
735-model-test/85-suite, model-Javadoc, root-test, javap/reflection/import/source/generated-page,
Java 26 example, 442-link/134-anchor Markdown, fence/whitespace/newline, exact twelve-path,
semantic-bytecode-equivalence, and no-task-0019-spec checks passed. Index/update values, bounds,
duplicates, writes/reductions, gradients, compiler behavior, lowering, backend behavior, and
execution remain separately owned.

The post-0018 capability-reset audit found no architecture conflict, but it rejected blanket
legacy parity as the next-step rule. It initially inserted Draft tasks 0018K–0018V for operation
validity,
multi-output provenance, symbolic extents, typed scalars, public-taxonomy cleanup, and missing core
numeric/reduction semantics. Linear algebra moves behind those dependencies. Completed task
history remains unchanged; the cleanup rows explicitly own any future replacement of provisional
APIs implemented by completed tasks.

Task [0014B Binary arithmetic Tensor expressions](modules/model/tasks/0014b-binary-arithmetic-tensor-expressions.md)
is complete. Its explicitly authorized tenth path corrected the Compile API status without adding
compiler behavior. The post-0014B reassessment kept the ordered model frontier because downstream
prerequisite modules remain placeholders. Task
[0014C](modules/model/tasks/0014c-unary-elementwise-semantic-kinds.md) is complete. Task
[0014D](modules/model/tasks/0014d-unary-elementwise-tensor-expressions.md) is complete. Task
[0014E](modules/model/tasks/0014e-scalar-arithmetic-and-clamp-semantics.md) is complete. Task
[0014F](modules/model/tasks/0014f-scalar-arithmetic-and-clamp-tensor-expressions.md) is complete.
Task [0015A](modules/model/tasks/0015a-binary-comparison-semantic-kinds.md) is complete. It adds the
six typed parameterless ordered binary comparison meanings without public Tensor expressions,
inference, provenance, or execution. Task
[0015B](modules/model/tasks/0015b-binary-comparison-tensor-expressions.md) is complete. It adds six
floating-only broadcast-aware Tensor comparison methods that create storage-free BOOL results with
false gradient eligibility and exact ordered provenance, without numerical execution. Task
[0015C](modules/model/tasks/0015c-boolean-logical-semantic-kinds.md) is complete. It adds one
parameterless boolean-logical semantic enum with exact AND, OR, and NOT identities while leaving
BOOL descriptors and public Tensor expressions to task 0015D. Task
[0015D](modules/model/tasks/0015d-boolean-logical-tensor-expressions.md) is complete. It adds exact
BOOL-only AND/OR broadcasting and shape-preserving NOT expression construction with fixed
non-differentiable BOOL results and provenance, without truth-value execution. Task
[0015E](modules/model/tasks/0015e-where-selection-semantic-kind.md) is complete. It adds the sole
parameterless `WHERE` semantic identity and documents its ordered condition, true-branch, and
false-branch roles without adding public Tensor construction or indexing behavior. Task
[0015F](modules/model/tasks/0015f-where-selection-tensor-expression.md) is complete. It adds exact
BOOL/floating validation, ordered pairwise broadcasting, branch-only gradient eligibility, and
three-input provenance without value selection or execution. Task
[0015G](modules/model/tasks/0015g-cast-semantic-kind-and-attributes.md) is complete. It adds the
exact `CAST` semantic identity and immutable target-data-type attributes without public Tensor
construction, inference, conversion policy, gradients, or execution. Task
[0015H](modules/model/tasks/0015h-cast-tensor-expression.md) is complete. It adds a fresh explicit
storage-free expression for every current source/target pair, including same-type requests, while
leaving conversion, canonicalization, gradient rules, and execution to their owning layers.
The broad former task 0016 is decomposed into focused aggregate, scan, and softmax semantic/
expression tasks. [0016A](modules/model/tasks/0016a-reduction-semantic-kinds-and-attributes.md) is
complete; it defines aggregate semantic kinds, normalized single-axis/full parameters, and
arg-max tie policy without Tensor behavior or execution. Task
[0016B](modules/model/tasks/0016b-sum-mean-and-product-tensor-expressions.md) is complete. It adds
floating full and one-axis sum/mean/product expressions, rank-zero full results, local Shape
derivation, and provenance without value aggregation. Task
[0016C](modules/model/tasks/0016c-min-and-max-tensor-reduction-expressions.md) is complete. It extends
the same bounded helper and focused test with full and one-axis floating min/max expressions while
keeping reduction identities distinct from binary min/max and deferring numerical comparison,
empty-domain, tie-gradient, compiler, and execution behavior. Task
[0016D](modules/model/tasks/0016d-boolean-all-and-any-tensor-expressions.md) is complete. It
generalizes the same six-method helper with kind-aware numeric/BOOL validation and adds full and
one-axis all/any expressions while deferring truth evaluation and empty-domain identity.
Task [0016E](modules/model/tasks/0016e-arg-max-tensor-expressions.md) is complete. It adds axis-only
numeric arg-max construction with explicit tie semantics, fixed INT64 results, and a dedicated
helper while leaving value comparison and execution deferred at that historical frontier.
Completed task 0018U1 later replaces the arg-max-only types/helper with shared arg-extrema
contracts, adds arg-min, fixes ordering and static-empty-selected-axis semantics, and broadens
ordinary SUM/PROD/MIN/MAX to exact signed-integral input.
Task [0016F](modules/model/tasks/0016f-masked-reduction-semantics-and-axis-mapping.md) is complete.
It adds the typed semantic contract and explicit ordered mask-dimension-to-input-axis mapping
needed to preserve legacy-compatible masks that ordinary right-aligned broadcasting cannot
represent. Task
[0016F1](modules/model/tasks/0016f1-masked-sum-and-mean-tensor-expressions.md) is complete. It adds
deterministic local Shape-based mapping resolution and public axis-removing masked sum/mean
expressions without value, storage, gradient, compiler, or backend behavior. Task
[0016G](modules/model/tasks/0016g-cumulative-sum-semantic-kind-and-attributes.md) is complete. It
defines only the cumulative-sum kind and immutable normalized-axis, exclusive, and reverse
attributes. Task
[0016H](modules/model/tasks/0016h-cumulative-sum-tensor-expressions.md) is complete. It adds local
numeric validation, axis normalization, exact shape/type/eligibility retention with unresolved
layout, and one-input provenance without value accumulation, gradient rules, compiler capture,
backend behavior, or execution. Task
[0016I](modules/model/tasks/0016i-softmax-semantic-kinds-and-attributes.md) is complete. It adds
typed SOFTMAX and LOG_SOFTMAX identities plus their shared normalized-axis attributes and documents
ideal probability/log-probability slice semantics without Tensor construction, numerical policy,
gradients, compiler behavior, backend behavior, or execution. Task
[0016J](modules/model/tasks/0016j-softmax-tensor-expressions.md) is complete. It adds public
floating softmax/log-softmax expressions with axis normalization, shape-preserving descriptor
construction, and one-input provenance without numerical evaluation or decomposition.
Task [0017A](modules/model/tasks/0017a-contiguous-semantic-kind.md) is complete. It defines only the
parameterless contiguous-layout request and its distinction from resolved layout classification
and later materialization. Task
[0017B](modules/model/tasks/0017b-contiguous-tensor-expression.md) is complete. It adds the public
storage-free expression with static-resolved and dynamic-unresolved result layout rules while
leaving copy choice and materialization to later compiler/planning/prepare/backend work.
Task [0017C](modules/model/tasks/0017c-reshape-and-expand-semantics.md) is complete. It defines only
the two target-shape semantic identities and shared immutable Shape attributes; public request
normalization, compatibility validation, layout derivation, and provenance remain in expression
tasks. Task [0017D](modules/model/tasks/0017d-reshape-tensor-expressions.md) is complete. It adds
raw-inferred and exact-Shape reshape expressions with conditional contiguous-input/static-target
view geometry. Task
[0017D1](modules/model/tasks/0017d1-expand-tensor-expressions.md) is complete with directional
right-aligned singleton/leading-axis validation and resolved zero-stride view geometry; storage
aliasing, materialization, gradients, compiler behavior, lowering, and execution remain deferred.
Task [0017E](modules/model/tasks/0017e-axis-transform-semantics.md) is complete with exact PERMUTE,
EXPAND_DIMS, and SQUEEZE meanings plus immutable normalized permutation/single-axis attributes.
Task [0017F](modules/model/tasks/0017f-permute-and-transpose-tensor-expressions.md) is complete with
arbitrary complete permutation and rank-two transpose over PERMUTE `[1, 0]`. The former combined
expression row is split. Task
[0017F1](modules/model/tasks/0017f1-expand-dimensions-and-squeeze-tensor-expressions.md) is complete
with expand-dimensions and squeeze construction whose insertion/existing-axis normalization,
singleton proof, Shape construction, and stride algebra remain distinct from permutation.
Task [0017G](modules/model/tasks/0017g-slice-semantics.md) is complete. It defines
one `SLICE` identity and immutable normalized parallel half-open bounds, distinct axes, and
positive steps. Single-axis convenience is the same operation with one step-one entry. Task
[0017H](modules/model/tasks/0017h-slice-tensor-expressions.md) is complete with public
long-bound/step requests, static-axis normalization/clamping, zero-extent results, local
Shape/view geometry, and fresh provenance. Task
[0017I](modules/model/tasks/0017i-pad-and-tile-semantics.md) is complete with separate typed
constant-padding and positive complete-pattern per-axis tiling semantics, immutable ordered
attributes, scalar identity parameters, and uninterpreted raw padding constants. Task
[0017J](modules/model/tasks/0017j-pad-and-tile-tensor-expressions.md) is complete with public Tensor
construction, its original checked static and identity-only dynamic Shape arithmetic, unresolved
result layout, and fresh provenance. Completed task 0018M1 later replaces only those conservative
dynamic derivation rules with canonical symbolic formulas. Task
[0017K](modules/model/tasks/0017k-tensor-composition-semantics.md) is complete with CONCAT, STACK,
and individually indexed UNSTACK-output semantics without provenance or graph changes. Task
[0017L](modules/model/tasks/0017l-tensor-composition-expressions.md) is complete with ordered public
concat/stack, immutable-list unstack expression construction, unresolved result layouts, and exact
ordered or individually indexed provenance without producer grouping or cross-layer behavior.
Task [0017M](modules/model/tasks/0017m-unfold-and-fold-semantics.md) is complete. It defines
general-axis sliding windows and the overlap-add fold semantics that task 0017N historically
exposed publicly, plus NCHW
im2col columns, and overlap-accumulating col2im through typed immutable semantic parameters.
Task 0017N completed all four then-public Tensor expressions; task 0018R later removed public
`foldAxis` while preserving the historical completion record and retained public Java semantic
contracts. [Task 0023](modules/model/tasks/0023-adjoint-expressibility-audit.md) selected completed
task 0023D for a public generally useful overlap-add capability before compiler
backward construction; none of these planning statements claims gradient implementation.
Task [0017N](modules/model/tasks/0017n-unfold-and-fold-tensor-expressions.md) is complete with the
exact signatures present at its historical completion, locally provable static/dynamic Shape
rules, checked window arithmetic,
unresolved layouts, and one-input provenance without values, gradients, compiler behavior, or
execution. Its independent documentation review passed focused 16-test, all 629 model-test across
74 suites, model-Javadoc, root-test, executable-example, bytecode/reflection, generated-page,
370-link/108-anchor, exact fifteen-path, synchronized-status, and no-0018-spec checks.

Package migrations `0003A` through `0003C` and tasks `0004`–`0012` are complete. Task `0012`
implemented only descriptor-based construction, optional borrowed storage attachment, and
JVM-wide tensor-ID allocation. Task [`0012A`](modules/model/tasks/0012a-host-storage-allocation.md)
is complete. It adds exact-span typed primitive-array allocation through the existing borrowed
heap-segment storage contract without arena ownership or close behavior. Task
[`0012B`](modules/model/tasks/0012b-flat-typed-tensor-import.md) is complete. It imports copied
flat primitive arrays into resolved dense-contiguous tensors with exact carrier/count validation
and canonical BOOL normalization. Task
[`0012C`](modules/model/tasks/0012c-nested-typed-tensor-import.md) is complete. It validates
rectangular multidimensional primitive arrays, infers exact carrier type and static dense shape,
flattens row-major, and delegates final creation to flat import. Task
[`0012D`](modules/model/tasks/0012d-constant-tensor-creation.md) is complete. It adds exact typed
rank-zero scalars plus independent dense zeros, ones, zeros-like, and ones-like tensors. Task
[`0012E`](modules/model/tasks/0012e-range-and-prefix-population.md), range and prefix population,
is complete. It adds eager non-empty typed integer ranges and copied strict/cyclic flat-prefix
population under canonical dense descriptors. Task
[`0012F`](modules/model/tasks/0012f-random-tensor-creation.md) is complete. It adds eager normal
population for three floating types from an explicit transient caller-owned source with bounded
reproducibility. [`0012G`](modules/model/tasks/0012g-uniform-random-tensor-creation.md) is complete;
it adds bounded continuous-uniform floating samples with explicit binary64 half-open bounds and the
same transient source policy. [`0012H`](modules/model/tasks/0012h-integral-random-tensor-creation.md)
is complete; it adds typed bounded integral sampling with primitive-bound type inference and direct
JDK bounded calls. [`0012I`](modules/model/tasks/0012i-bernoulli-random-tensor-creation.md) is
complete; it adds canonical BOOL Bernoulli samples from a finite scalar probability using one
unbounded source call per element, including at probability endpoints. Task
[`0013`](modules/model/tasks/0013-tensor-provenance-skeleton.md) is complete. It adds immutable
operation-and-ordered-input origin metadata without turning Tensor into graph IR or implementing
compiler capture. Task
[`0013A`](modules/model/tasks/0013a-full-value-and-identity-matrix-tensor-creation.md) is complete;
it adds canonical type-safe `full`, rectangular `identityMatrix`, and the exact convenience alias
`eye`. The completed post-foundation checkpoint selected continued sequential model operation-
family work. Task
[`0014A`](modules/model/tasks/0014a-binary-arithmetic-semantic-kinds.md) is complete and provides
the first production concrete OperationKind family. Task
[`0014B`](modules/model/tasks/0014b-binary-arithmetic-tensor-expressions.md) has implemented the
first public binary arithmetic expression surface and is complete after full validation and the
authorized Compile API status correction.

## Model task sequence

| Order | Task | Status |
|---|---|---|
| 1 | 0001 DataType model | Complete |
| 2 | 0002 Shape and dimension model | Complete |
| 3 | 0003 Layout descriptor model | Complete |
| 4 | 0003A Data type package migration | Complete |
| 5 | 0003B Shape package migration | Complete |
| 6 | 0003C Layout package migration | Complete |
| 7 | 0004 Typed identifiers | Complete |
| 8 | 0005 Operation semantic foundation | Complete |
| 9 | 0006 Operation model | Complete |
| 10 | [0007 Tensor descriptor model](modules/model/tasks/0007-tensor-descriptor-model.md) | Complete |
| 11 | [0008 Graph value and node model](modules/model/tasks/0008-graph-value-and-node-model.md) | Complete |
| 12 | [0009 Compiled graph model](modules/model/tasks/0009-compiled-graph-model.md) | Complete |
| 13 | [0010 Host storage abstraction](modules/model/tasks/0010-host-storage-abstraction.md) | Complete |
| 14 | [0011 Public Tensor skeleton](modules/model/tasks/0011-public-tensor-skeleton.md) | Complete |
| 15 | [0012 Tensor factory foundation](modules/model/tasks/0012-tensor-factory.md) | Complete |
| 16 | [0012A JVM-managed heap host storage allocation](modules/model/tasks/0012a-host-storage-allocation.md) | Complete |
| 17 | [0012B Flat typed tensor import](modules/model/tasks/0012b-flat-typed-tensor-import.md) | Complete |
| 18 | [0012C Nested typed tensor import](modules/model/tasks/0012c-nested-typed-tensor-import.md) | Complete |
| 19 | [0012D Constant tensor creation](modules/model/tasks/0012d-constant-tensor-creation.md) | Complete |
| 20 | [0012E Range and prefix population](modules/model/tasks/0012e-range-and-prefix-population.md) | Complete |
| 21 | [0012F Random tensor creation](modules/model/tasks/0012f-random-tensor-creation.md) | Complete |
| 22 | [0012G Uniform random tensor creation](modules/model/tasks/0012g-uniform-random-tensor-creation.md) | Complete |
| 23 | [0012H Integral random tensor creation](modules/model/tasks/0012h-integral-random-tensor-creation.md) | Complete |
| 24 | [0012I Bernoulli random tensor creation](modules/model/tasks/0012i-bernoulli-random-tensor-creation.md) | Complete |
| 25 | [0013 Tensor provenance skeleton](modules/model/tasks/0013-tensor-provenance-skeleton.md) | Complete |
| 26 | [0013A Full-value and identity-matrix tensor creation](modules/model/tasks/0013a-full-value-and-identity-matrix-tensor-creation.md) | Complete |
| 27 | [0014A Binary arithmetic semantic kinds](modules/model/tasks/0014a-binary-arithmetic-semantic-kinds.md) | Complete |
| 28 | [0014B Binary arithmetic Tensor expressions](modules/model/tasks/0014b-binary-arithmetic-tensor-expressions.md) | Complete |
| 29 | [0014C Unary elementwise semantic kinds](modules/model/tasks/0014c-unary-elementwise-semantic-kinds.md) | Complete |
| 30 | [0014D Unary elementwise Tensor expressions](modules/model/tasks/0014d-unary-elementwise-tensor-expressions.md) | Complete |
| 31 | [0014E Scalar arithmetic and clamp semantics](modules/model/tasks/0014e-scalar-arithmetic-and-clamp-semantics.md) | Complete |
| 32 | [0014F Scalar arithmetic and clamp Tensor expressions](modules/model/tasks/0014f-scalar-arithmetic-and-clamp-tensor-expressions.md) | Complete |
| 33 | [0015A Binary comparison semantic kinds](modules/model/tasks/0015a-binary-comparison-semantic-kinds.md) | Complete |
| 34 | [0015B Binary comparison Tensor expressions](modules/model/tasks/0015b-binary-comparison-tensor-expressions.md) | Complete |
| 35 | [0015C Boolean logical semantic kinds](modules/model/tasks/0015c-boolean-logical-semantic-kinds.md) | Complete |
| 36 | [0015D Boolean logical Tensor expressions](modules/model/tasks/0015d-boolean-logical-tensor-expressions.md) | Complete |
| 37 | [0015E Where selection semantic kind](modules/model/tasks/0015e-where-selection-semantic-kind.md) | Complete |
| 38 | [0015F Where selection Tensor expression](modules/model/tasks/0015f-where-selection-tensor-expression.md) | Complete |
| 39 | [0015G Cast semantic kind and attributes](modules/model/tasks/0015g-cast-semantic-kind-and-attributes.md) | Complete |
| 40 | [0015H Cast Tensor expression](modules/model/tasks/0015h-cast-tensor-expression.md) | Complete |
| 41 | [0016A Reduction semantic kinds and attributes](modules/model/tasks/0016a-reduction-semantic-kinds-and-attributes.md) | Complete |
| 42 | [0016B Sum, mean, and product Tensor expressions](modules/model/tasks/0016b-sum-mean-and-product-tensor-expressions.md) | Complete |
| 43 | [0016C Min and max Tensor reduction expressions](modules/model/tasks/0016c-min-and-max-tensor-reduction-expressions.md) | Complete |
| 44 | [0016D Boolean all and any Tensor expressions](modules/model/tasks/0016d-boolean-all-and-any-tensor-expressions.md) | Complete |
| 45 | [0016E Arg-max Tensor expressions](modules/model/tasks/0016e-arg-max-tensor-expressions.md) | Complete |
| 46 | [0016F Masked reduction semantics and axis mapping](modules/model/tasks/0016f-masked-reduction-semantics-and-axis-mapping.md) | Complete |
| 47 | [0016F1 Masked sum and mean Tensor expressions](modules/model/tasks/0016f1-masked-sum-and-mean-tensor-expressions.md) | Complete |
| 48 | [0016G Cumulative-sum semantic kind and attributes](modules/model/tasks/0016g-cumulative-sum-semantic-kind-and-attributes.md) | Complete |
| 49 | [0016H Cumulative-sum Tensor expressions](modules/model/tasks/0016h-cumulative-sum-tensor-expressions.md) | Complete |
| 50 | [0016I Softmax semantic kinds and attributes](modules/model/tasks/0016i-softmax-semantic-kinds-and-attributes.md) | Complete |
| 51 | [0016J Softmax Tensor expressions](modules/model/tasks/0016j-softmax-tensor-expressions.md) | Complete |
| 52 | [0017A Contiguous semantic kind](modules/model/tasks/0017a-contiguous-semantic-kind.md) | Complete |
| 53 | [0017B Contiguous Tensor expression](modules/model/tasks/0017b-contiguous-tensor-expression.md) | Complete |
| 54 | [0017C Reshape and expand semantics](modules/model/tasks/0017c-reshape-and-expand-semantics.md) | Complete |
| 55 | [0017D Reshape Tensor expressions](modules/model/tasks/0017d-reshape-tensor-expressions.md) | Complete |
| 56 | [0017D1 Expand Tensor expressions](modules/model/tasks/0017d1-expand-tensor-expressions.md) | Complete |
| 57 | [0017E Axis-transform semantics](modules/model/tasks/0017e-axis-transform-semantics.md) | Complete |
| 58 | [0017F Permute and transpose Tensor expressions](modules/model/tasks/0017f-permute-and-transpose-tensor-expressions.md) | Complete |
| 59 | [0017F1 Expand-dimensions and squeeze Tensor expressions](modules/model/tasks/0017f1-expand-dimensions-and-squeeze-tensor-expressions.md) | Complete |
| 60 | [0017G Slice semantics](modules/model/tasks/0017g-slice-semantics.md) | Complete |
| 61 | [0017H Slice Tensor expressions](modules/model/tasks/0017h-slice-tensor-expressions.md) | Complete |
| 62 | [0017I Pad and tile semantics](modules/model/tasks/0017i-pad-and-tile-semantics.md) | Complete |
| 63 | [0017J Pad and tile Tensor expressions](modules/model/tasks/0017j-pad-and-tile-tensor-expressions.md) | Complete |
| 64 | [0017K Tensor composition semantics](modules/model/tasks/0017k-tensor-composition-semantics.md) | Complete |
| 65 | [0017L Tensor composition expressions](modules/model/tasks/0017l-tensor-composition-expressions.md) | Complete |
| 66 | [0017M Unfold and fold semantics](modules/model/tasks/0017m-unfold-and-fold-semantics.md) | Complete |
| 67 | [0017N Unfold and fold Tensor expressions (historically including public foldAxis)](modules/model/tasks/0017n-unfold-and-fold-tensor-expressions.md) | Complete |
| 68 | [0018A Scalar select semantics](modules/model/tasks/0018a-scalar-select-semantics.md) | Complete |
| 69 | [0018B Scalar select Tensor expression](modules/model/tasks/0018b-scalar-select-tensor-expression.md) | Complete |
| 70 | [0018C Axis gather semantics](modules/model/tasks/0018c-axis-gather-semantics.md) | Complete |
| 71 | [0018D Axis gather Tensor expressions](modules/model/tasks/0018d-axis-gather-tensor-expressions.md) | Complete |
| 72 | [0018D1 Primitive take convenience](modules/model/tasks/0018d1-primitive-take-convenience.md) | Complete |
| 73 | [0018E Gather-ND semantics](modules/model/tasks/0018e-gather-nd-semantics.md) | Complete |
| 74 | [0018F Gather-ND Tensor expressions](modules/model/tasks/0018f-gather-nd-tensor-expressions.md) | Complete |
| 75 | [0018G Axis scatter semantics](modules/model/tasks/0018g-axis-scatter-semantics.md) | Complete |
| 76 | [0018H Axis scatter Tensor expressions](modules/model/tasks/0018h-axis-scatter-tensor-expressions.md) | Complete |
| 77 | [0018I Scatter-ND semantics](modules/model/tasks/0018i-scatter-nd-semantics.md) | Complete |
| 78 | [0018J Scatter-ND Tensor expression](modules/model/tasks/0018j-scatter-nd-tensor-expression.md) | Complete |
| 79 | [0018K Operation signature and construction hardening](modules/model/tasks/0018k-operation-signature-and-construction-hardening.md) | Complete |
| 80 | [0018L Shared multi-output Tensor provenance](modules/model/tasks/0018l-shared-multi-output-tensor-provenance.md) | Complete |
| 81 | [0018M Symbolic extent expressions](modules/model/tasks/0018m-symbolic-extent-expressions.md) | Complete |
| 82 | [0018M1 Dynamic extent adoption in pad, tile, and concat](modules/model/tasks/0018m1-dynamic-extent-adoption.md) | Complete |
| 83 | [0018N Typed scalar value contract](modules/model/tasks/0018n-typed-scalar-value-contract.md) | Complete |
| 84 | [0018O Indexing taxonomy and unstack normalization](modules/model/tasks/0018o-indexing-taxonomy-and-unstack-normalization.md) | Complete |
| 85 | [0018P Elementwise semantic cleanup](modules/model/tasks/0018p-elementwise-semantic-cleanup.md) | Complete |
| 86 | [0018Q Masked reduction redesign](modules/model/tasks/0018q-masked-reduction-redesign.md) | Complete |
| 87 | [0018R Slice and window public-contract cleanup](modules/model/tasks/0018r-slice-and-window-public-contract-cleanup.md) | Complete |
| 88 | [0018S Tensor factory surface cleanup](modules/model/tasks/0018s-tensor-factory-surface-cleanup.md) | Complete |
| 89 | [0018T Scalar arithmetic family normalization](modules/model/tasks/0018t-scalar-arithmetic-family-normalization.md) | Complete |
| 90 | [0018T1 Unary numeric gaps and floating diagnostics](modules/model/tasks/0018t1-unary-numeric-gaps-and-floating-diagnostics.md) | Complete |
| 91 | [0018U Integral elementwise arithmetic and comparisons](modules/model/tasks/0018u-integral-elementwise-arithmetic-and-comparisons.md) | Complete |
| 92 | [0018U1 Integral reductions and arg-min normalization](modules/model/tasks/0018u1-integral-reductions-and-arg-min-normalization.md) | Complete |
| 93 | [0018V Multi-axis and statistical reductions](modules/model/tasks/0018v-multi-axis-and-statistical-reductions.md) | Complete |
| 94 | [0019 Matmul semantics and Tensor expression](modules/model/tasks/0019-matmul-semantics-and-tensor-expression.md) | Complete |
| 95 | [0019A Modern activation semantics and Tensor expressions](modules/model/tasks/0019a-modern-activation-semantics-and-tensor-expressions.md) | Complete |
| 96 | [0019A1 Embedding convenience](modules/model/tasks/0019a1-embedding-convenience.md) | Complete |
| 97 | [0019A2 One-hot encoding](modules/model/tasks/0019a2-one-hot-encoding.md) | Complete |
| 98 | [0019B Explicit graph RNG state foundation](modules/model/tasks/0019b-explicit-graph-rng-state-foundation.md) | Complete |
| 99 | [0019B1 Explicit graph dropout construction](modules/model/tasks/0019b1-explicit-graph-dropout-construction.md) | Complete |
| 100 | [0019C Sort and argsort](modules/model/tasks/0019c-sort-and-argsort.md) | Complete |
| 101 | [0019C1 Top-K values and indices](modules/model/tasks/0019c1-top-k-values-and-indices.md) | Complete |
| 102 | [0019D Linear convenience](modules/model/tasks/0019d-linear-convenience.md) | Complete |
| 103 | [0019E Scaled dot-product attention](modules/model/tasks/0019e-scaled-dot-product-attention.md) | Complete |
| 104 | [0020 NCHW Conv2d semantics and Tensor expressions](modules/model/tasks/0020-nchw-conv2d-semantics-and-tensor-expressions.md) | Complete |
| 105 | [0020A NCHW Max Pool2d semantics and Tensor expression](modules/model/tasks/0020a-nchw-max-pool2d-semantics-and-tensor-expression.md) | Complete |
| 106 | [0020A1 NCHW Average Pool2d semantics and Tensor expression](modules/model/tasks/0020a1-nchw-average-pool2d-semantics-and-tensor-expression.md) | Complete |
| 107 | [0021 Layer normalization semantics and Tensor expressions](modules/model/tasks/0021-layer-normalization-semantics-and-tensor-expressions.md) | Complete |
| 108 | [0021A RMS normalization semantics and Tensor expressions](modules/model/tasks/0021a-rms-normalization-semantics-and-tensor-expressions.md) | Complete |
| 109 | [0021B Batch-normalization inference](modules/model/tasks/0021b-batch-normalization-inference.md) | Complete |
| 110 | [0021C Batch-normalization training and statistic transition](modules/model/tasks/0021c-batch-normalization-training-and-statistic-transition.md) | Complete |
| 111 | [0022 Mean-squared-error loss](modules/model/tasks/0022-mean-squared-error-loss.md) | Complete |
| 112 | [0022A Dense-target categorical cross-entropy with logits](modules/model/tasks/0022a-dense-target-categorical-cross-entropy-with-logits.md) | Complete |
| 113 | [0022B Index-target categorical cross-entropy with logits](modules/model/tasks/0022b-index-target-categorical-cross-entropy-with-logits.md) | Complete |
| 114 | [0023 Adjoint expressibility audit](modules/model/tasks/0023-adjoint-expressibility-audit.md) | Complete |
| 115 | [0023A Binding-aware sum-to-Shape](modules/model/tasks/0023a-binding-aware-sum-to-shape.md) | Complete |
| 116 | [0023B Gather-compatible scatter-add](modules/model/tasks/0023b-gather-compatible-scatter-add.md) | Complete |
| 117 | [0023C Slice update and target-relative crop](modules/model/tasks/0023c-slice-update-and-target-relative-crop.md) | Complete |
| 118 | [0023D Public foldAxis and dynamic window transforms](modules/model/tasks/0023d-public-fold-axis-and-dynamic-window-transforms.md) | Complete |
| 119 | [0023E Cumulative scan normalization and product](modules/model/tasks/0023e-cumulative-scan-normalization-and-product.md) | Complete |
| 120 | [0023F Scaled dot-product attention weights output](modules/model/tasks/0023f-scaled-dot-product-attention-weights-output.md) | Complete |
| 121 | [0024 Model capability and contract closure audit](modules/model/tasks/0024-model-capability-and-contract-closure-audit.md) | Complete |
| 122 | [0024A GraphValue Tensor-status Javadoc correction](modules/model/tasks/0024a-graph-value-tensor-status-javadoc-correction.md) | Complete |
| 123 | [0025 Canonical TensorProducer outputs](modules/model/tasks/0025-canonical-tensor-producer-outputs.md) | Complete |
| 124 | [0025A Portable floating comparison, extrema, and clamp semantics](modules/model/tasks/0025a-portable-floating-comparison-extrema-and-clamp-semantics.md) | Complete |
| 125 | [0025B Binding-aware expansion](modules/model/tasks/0025b-binding-aware-expansion.md) | Complete |
| 126 | [0025C Portable functional-scatter reduction semantics](modules/model/tasks/0025c-portable-functional-scatter-reduction-semantics.md) | Complete |
| 127 | [0025D Dynamic-extent slice extraction and symbolic slice placement](modules/model/tasks/0025d-dynamic-extent-slice-extraction-and-symbolic-slice-placement.md) | Complete |
| 128 | 0026 IEEE FLOAT16 and mixed-precision semantic contracts | Draft (future interleave; no detailed specification) |

Task dependencies in the model master plan remain hard prerequisites. The table order is the default execution order even when a later task has no explicit dependency on an earlier task.

## Model foundation checkpoint result

The checkpoint reviewed the completed value, graph, storage, Tensor, provenance, and eager factory
contracts after task `0013A`. It selected continued sequential model operation-family work rather
than an immediate cross-module vertical slice.

The reason was concrete: model graph and provenance foundations existed, but no production
concrete `OperationKind` existed for compiler capture, capability analysis, backend ownership,
lowering, or execution. Task 0014 was therefore decomposed into semantic-vocabulary and public-
expression pairs. Completed task 0014A introduces the first typed family, and task 0014B now
implements its public Tensor expression construction. The family creates the intended integration
seam.

The post-0014B reassessment considered opening a cross-module compile-to-execution slice next, but
the required trace, backend-contract, config, planning, and compiler foundations still consist only
of placeholder production types and broad master plans. Treating that prerequisite chain as one
next task would violate the planning granularity and architecture-boundary rules. The ordered model
queue therefore continued with task 0014C, which completed the fifteen parameterless unary
elementwise semantic kinds. Task 0014D then completed their matching public Tensor expression
construction without crossing the model boundary. Task 0014E completed the typed scalar and clamp
semantic parameters without adding Tensor expression behavior. Task 0014F completed their public
Tensor expression construction without crossing the model boundary. The former broad task 0015
has been decomposed into comparison, BOOL logic, `where`, and cast semantic/expression pairs.
Task 0015A completed the six parameterless comparison semantics, and task 0015B completed their
floating-only, broadcast-aware public Tensor construction with fixed BOOL results and ordered
provenance. Task 0015C completed the parameterless AND, OR, and NOT semantic identities. Task
0015D completed their BOOL-only binary/unary public Tensor construction with fixed result facts and
exact provenance. Task 0015E completed the one parameterless `WHERE` identity and documented its
ternary logical roles separately from task 0015F's later Tensor validation, three-way broadcasting,
result construction, and provenance work. Task 0015F completed that public expression by composing
the current BOOL, floating-promotion, pairwise-broadcast, descriptor, provenance, and
derived-construction contracts without changing module boundaries or foundational APIs. Task
0015G completed the typed cast identity and target data-type parameter while isolating them from
task 0015H's Tensor/result construction and conversion-policy decisions. Task 0015H completed that
public Tensor construction with exact Shape retention, floating-only gradient eligibility, and a
fresh explicit cast for every valid request. Compiler work later owns redundant same-type and
cast-chain canonicalization. The broad former task 0016 is now decomposed into 0016A–0016J plus
0016F1 so aggregate semantics, focused Tensor expression groups, masked reductions, cumulative
scan, and softmax do not share one oversized task. Tasks 0016A through 0016E are complete. Tasks
0016F, 0016F1, 0016G, 0016H, 0016I, and 0016J are also complete. The broad former task 0017 is now
decomposed into 0017A–0017N plus 0017D1 and 0017F1; 0017A through 0017F, including 0017D1, are
complete, and 0017F1, 0017G, 0017H, 0017I, 0017J, 0017K, 0017L, 0017M, and 0017N are also
complete. The former broad task 0018 is decomposed into 0018A–0018J. Tasks 0018A and 0018B are
complete. Tasks 0018C, 0018D, 0018D1, 0018E, 0018F, and 0018G are also complete. Task 0018H is
also complete. Tasks 0018I and 0018J are complete. The capability reset inserts 0018K–0018V
before 0019. Tasks 0018K through 0018T1, task 0018U, task 0018U1, linked 0018V, and task 0019 are
complete. Tasks 0019A, 0019A1, 0019A2, 0019B, 0019B1, 0019C, 0019C1, and 0019D are also complete.
Tasks 0019E, 0020, 0020A, 0020A1, 0021, 0021A, and 0021B are complete. Task
[0021C](modules/model/tasks/0021c-batch-normalization-training-and-statistic-transition.md) is
Complete. [Task 0022](modules/model/tasks/0022-mean-squared-error-loss.md) is Complete.
[Task 0022A](modules/model/tasks/0022a-dense-target-categorical-cross-entropy-with-logits.md) is
Complete. Task 0022B is Complete. Task 0023 is Complete with its detailed specification and
result artifact. Tasks 0023A–0023F are Complete with their detailed specifications; established
task 0024 is Complete with its historical `BLOCKING_GAP` result artifact. Task 0024A is Complete,
its sole Javadoc blocker is resolved, and the selected model capability milestone is closed.
Focused task 0025 is the completed producer-output prerequisite for compiler-owned pre-capture
autograd. Focused task 0025A is also Complete and supplies the remaining portable floating
comparison/extrema/clamp forward-contract prerequisite for Compiler 0005A without altering the
historical closure result or selecting derivatives.
Focused task 0025B is Complete. It supplies the binding-aware EXPAND expression prerequisite for
Compiler 0005B without adding compiler behavior or altering the historical closure result.
Focused task 0025C is Complete. It supplies the configurable functional-scatter forward-semantics
prerequisite for Compiler 0005C without adding evaluation, backend behavior, or derivative policy.
Focused task 0025D is Complete and is the latest detailed Model task. It supplies the remaining
dynamic-slice construction prerequisite for Compiler 0005C without adding compiler constraints,
gradients, lowering, execution, or backend behavior. Compiler 0005C and detailed Compiler 0005D
are Complete; detailed Compiler 0005E is also Complete. Detailed Compiler 0006 is Complete, and no
later compiler task has a detailed specification.
Completed task 0016E originally added fixed-INT64 one-axis arg-max expression metadata without
changing the ordinary reduction helper or adding value comparison, empty-axis policy, or
execution. Completed task 0018U1 now supplies the shared arg-extrema model policy and integral
ordinary reduction baseline without adding execution.

This decision changes implementation order only. It does not change architecture dependencies or
authorize compiler, planning, runtime, prepare, or backend behavior inside modules/model. A future
explicit roadmap decision may still reorder work when a bounded cross-module task and its
prerequisites are concrete.

## Advancing the frontier

Before advancing to the next task or project area:

1. complete all acceptance criteria for the current task;
2. record validation evidence and the completion summary;
3. review documentation and Javadoc impact;
4. update the task and master-plan statuses;
5. update this roadmap when the active project area changes; and
6. create the next detailed task specification as a separate planning step.

## Roadmap changes

Update this roadmap when implementation order, active frontier, or project-area status changes. Record the reason for reordering. If reordering reveals an architecture conflict, stop and resolve it through the architecture process instead of changing this roadmap alone.
