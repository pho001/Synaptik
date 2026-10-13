# Repository review — findings and repair record, 2026-10-13

## Scope, method and limits

This is an engineering review report and implementation record, not a numerical certificate or
an architecture contract. The [review brief](project-review-2026-10-13.md) defines ownership and
acceptance. Findings refer to production baseline `728680eaabc44a485da9f32050914ca4f09b295c`;
source line numbers below are baseline locations and move after repairs unless a later inspected
revision is explicitly identified.

Six independent review areas covered all 22 Gradle leaf projects, native Metal/CUDA sources,
build wiring and all 23 area master plans. Review combined source tracing, existing focused-test
inspection, small reproductions, contract comparison and dependency/frontier reconciliation.
Nontrivial repairs use isolated implementation contexts and independent code/documentation review.
The coordinator integrates them and owns the combined validation checkpoint.

This was a risk-focused whole-project review, **not a line-by-line proof of every file**. Operation
kind counts do not establish all dtype/layout/strategy/gradient combinations. No exhaustive fuzzing,
large-model memory test, cross-platform OpenBLAS campaign, general performance comparison or new
historical five-fork benchmark matrix was performed. A passing test run is not a correctness proof.

The initial full `./gradlew test javadoc` checkpoint passed before repairs (100 tasks: 4 executed,
4 from cache, 92 up-to-date). That baseline did not cover the counterexamples found below.
Actual-device execution is recorded separately from source review and skipped tests.

## Result at a glance

Confirmed implementation findings: **6 P1 and 10 P2**, no confirmed P0. Separately: **G1, a P2
numerical-contract discrepancy requiring an owner decision**, two test defects and several
documentation inconsistencies. P1 means a supported path can return a wrong result, fail
for ordinary valid input, or access native metadata out of bounds. P2 covers narrower numerical,
resource, API-consumer or bounded-memory failures. These priorities are repair urgency, not a
formal security classification.

| ID | Priority | Finding | Repair |
|---|---|---|---|
| A1 | P1 | Batched matrix–vector right gradient loses the column axis. | Compiler 0006D |
| A2 | P1 | A connected sibling hides a required unsupported derivative path. | Compiler 0006D |
| B1 | P2 | Duplicate prepared resource causes trailing unique resources to leak. | Prepare 0009 |
| C1 | P2 | Large finite ERF/GELU can produce NaN; vector body disagrees with tail. | CPU 0022 |
| C2 | P1 | FLOAT16 softmax validator reads FLOAT32 storage. | CPU 0021 |
| C3 | P2 | Value validation runs before an internal producer executes. | CPU 0021 |
| C4 | P1 | Composed losses compare the wrong access representation. | CPU 0021 |
| C5 | P2 | Low activation composites use DOUBLE instead of named FLOAT32 sites. | CPU 0022 |
| D1 | P1 | Rank-16 WHERE false-input metadata overflows its native array. | Metal 0078 |
| D2 | P2 | Low ScatterAdd changes untouched raw words. | Metal 0078 |
| D3 | P2 | Low inclusive scan seed loses negative-zero prefixes. | Metal 0078 |
| D4 | P2 | Native invocation does not enforce exact creating-context identity. | Metal 0078 |
| D5 | P1 | Advertised scalar RELU/MUL are rejected by Java program preflight. | Metal 0079 |
| E1 | P2 | Adam divides before multiplying and rejects a finite documented update. | Training 0002C |
| E2 | P2 | NN public Model types are missing from an NN-only consumer's compile path. | NN 0025B |
| F1 | P2 | Model-plan cache checks its bound after aggregate allocation. | Tuning 0005 |
| G1 | P2, contract | Stable finite softmax changes the output class implied by literal EXP overflow. | Model 0033 Draft; owner decision required |
| B2 | P2, test | Several “mixed-backend” tests now run entirely on Metal. | Engine 0019 |
| B3 | P3, test | Copy/close test assumes executor submission guarantees admission. | Engine 0019 |
| B4 | P3, docs | Public tuning Javadoc incorrectly describes CPU-only support. | Engine 0019 |

## Detailed production findings

### A1 — Batched matvec gradient

Location: `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/LinearAlgebraGradientRules.java:86`.

For `A[..., M, K] @ x[K]`, the output cotangent is `g[..., M]`. The old right-gradient formula
used `transpose(A) @ g` directly. A batched vector cotangent then looks like a matrix operand.
With `A[2,3,4]`, compilation rejects inner dimensions 3 versus 2. With coincident dimensions,
it can silently compute a different contraction: `A[2,2,2] = 1..8`, unit cotangent gives
`[36,36]` instead of `[16,20]` when evaluating the emitted algebra.

Repair: explicitly form `g[..., M, 1]`, multiply, remove the last singleton axis and reduce
batch dimensions to `x.shape()`. No new derivative family is introduced. Regression evidence
uses independent host-loop oracles, unequal and coincident dimensions, multiple batch axes,
nonuniform seeds, both operands and mixed second derivatives, with optimization on/off.

### A2 — Required unsupported derivative is silently omitted

Location: `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/AutogradPreflight.java:506`.

The old preflight treated a target with any connected differentiable path as acceptable before
checking another required blocked path. It also conflated intentionally non-differentiable roles
with fail-closed roles. In a second derivative through BatchNorm training, saved mean/inverse
standard deviation can require derivatives that are not implemented. A direct sibling path
does not make those contributions optional. For `x=[-1,1]`, epsilon 1, the inspected two-stage
formula can yield `[0.7071,0.7071]` where the complete derivative is zero.

Repair: intersect selected-output cotangent demand with target reachability; reject a required
fail-closed role even when a sibling connects. Stop demand at intentional mask/index/comparison
boundaries. Ordinary first-order saved-value use remains valid. This is truthful rejection,
**not** implementation of saved-statistic higher derivatives.

### B1 — Resource rollback misses owners after a duplicate

Location: `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/BackendPartitionFinalizationHandoff.java:265`.

A finalizer returning resources `[A,A,B]` transfers both unique owners, but the old loop threw
at the second `A` before registering `B`. The diagnostic closed only `A`; expected cleanup is
`B,A`. The same issue occurs across results, e.g. first `[A]`, then `[A,B]`.

Repair: collect every unique identity from the returned result before rejecting its first
duplicate. Do not validate the rejected executable or call later finalizers. Close each owner
once in reverse order, preserving the primary error and suppressing distinct cleanup failures.
Tests cover identity versus equality, trailing owners, multiple duplicates and throwing cleanup.

### C1 — Activation polynomial overflow for finite input

Locations: `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuScalarEmitter.java:381`
and `.../internal/codegen/emit/CpuVectorMath.java:156`.

ERF's rational-polynomial tail evaluates overflowing numerator and denominator before multiplying
by an exponentially tiny value. `Infinity/Infinity` becomes NaN; multiplication by zero cannot
repair it. Actual generated FLOAT64 execution with `±1e100` returned NaN. A FLOAT32 vector
body plus scalar tail over five identical `1e10` inputs returned `[NaN,NaN,NaN,NaN,1]`.
GELU inherits the tail problem. The reference shared the defect, so agreement with it was weak evidence.

Repair scope: a mathematically justified saturation guard before unsafe evaluation, independent
special-value tests, consistent body/tail behavior, generated/direct Java shape inspection and
a generated compatibility identity change. Existing NaN/infinity/zero contracts must remain intact.

### C2 — FLOAT16 softmax is decoded with the wrong carrier width

Location: `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuSoftmaxInputValidator.java:36`.

The validator handled BFLOAT16 specially but let FLOAT16 fall through to FLOAT32 access.
A `short[]` then causes `ClassCastException`; a native half buffer can receive an out-of-bounds
four-byte read where only two bytes remain. The public Engine path reproduced the failure.

Repair: decode FLOAT16 exactly using its existing representation and validate in the declared
working precision. Cover softmax and log-softmax, arrays/native segments, offsets and strides.
Do not claim expanded infinity/NaN support merely by fixing the storage reader.

### C3 — Value-dependent validation reads unproduced intermediates

Locations: `CpuPreparedExecutable.java:1663` and `CpuPreparedPartitionExecutable.java:194`
under `backends/cpu/src/main/java/.../backend/cpu/internal/`.

The composite partition binds children before executing any of them. A softmax/loss/mask scan
performed during binding therefore reads an intermediate buffer before its producer has written
the current run's values. `[0,1000].exp().softmax()` returned NaNs while direct `[1,Infinity]`
softmax correctly rejected the backend's finite-input domain. Reused storage can similarly
validate stale values rather than this run's data.

Repair: keep structural, carrier, range and layout checks at bind time. Perform value-dependent
validation immediately before the consuming unit writes output, after its producers execute,
on the orchestrating path before parallel workers. Audit the same issue for loss indices and
Boolean/attention masks. This does not promise that earlier internal producer writes are rolled
back on rejection; publication and consumer-write boundaries remain as specified.

### C4 — Composed loss binding rejects valid unit topology

Locations: `CpuPartitionPreparationPlan.java:1596` and `CpuLossIr.java:131` in CPU internals.

`unitAccessesAgree` compares a loss's rank-zero cache-key projection instead of its actual
boundary accesses. Direct MSE over `[1,2]` and `[0,0]` with MEAN yields 2.5; inserting a valid
prediction NEG causes “general CPU unit topology is not stable”. A computed categorical target
using integer addition also fails.

Repair: validate concrete boundary accesses without changing the intentionally abstract cache
projection. Regression scope includes composed predictions/targets and strided loss bindings.

### C5 — Low composite activations bypass FLOAT32 working sites

Location: `CpuScalarEmitter.java:277,304` in the CPU generated scalar path.

BFLOAT16/FLOAT16 inputs were decoded and widened to DOUBLE for the entire composite expression,
with one narrowing at the end. The current Model contract instead names FLOAT32 intermediate
sites. For BF16 GELU(-6), the old result is raw `0xb1cb` (about `-5.9e-9`); at the required F32
ERF site the value rounds to -1, giving a zero composite result.

Repair: preserve exact low decode, F32 constants and named F32 operations/elementary-return
rounding, then final low narrowing. Verify emitted bytecode and independent expected values,
not merely a reference implementation with the same widening error. No numerical-policy reset
or new low-type capability is needed.

### D1 — Rank-16 WHERE metadata exceeds the native array

Location: `native/metal-macos-arm64/src/synaptik_metal_foundation.m:3993`, plus the exact and
low WHERE shader definitions.

The false-input rank was stored in `inputExtents[0]`, then its dimensions in positions `1..rank`.
The array has 16 elements, so valid rank 16 writes and later reads position 16 out of bounds.
This is a source-confirmed native safety defect; the audit did not present a device-result
counterexample as proof of memory corruption.

Repair: use the existing per-role rank/dimension fields for the third operand. Add actual-device
FLOAT32/FLOAT16 rank-16 cases whose selected false input and last-axis broadcasting exercise
the former boundary, plus source/schema checks and source-matched native packaging.

### D2 — ScatterAdd modifies untouched low words

Location: `native/metal-macos-arm64/src/synaptik_low_precision_kernels.h:91`.

The low kernel widened and narrowed every destination, including destinations with no update.
That turns an untouched BF16 negative NaN raw word `0xffc1` into `0x7fc0`, and an F16 `0xfe01`
into `0x7e00`. Example: base `[1, negative-NaN]`, index `[0]`, update `[1]` must not arithmetically
re-encode the second word.

Repair: raw-copy an unaddressed destination; only addressed values use the arithmetic contract.
Cover both low dtypes, both index widths and untouched NaN/zero/subnormal representations.

### D3 — Low scan seed changes a negative-zero prefix

Location: `synaptik_low_precision_kernels.h:64`.

Starting inclusive cumulative SUM at positive zero makes input `[-0,-0]` produce positive-zero
prefixes instead of starting from the first contributor. The same distinction matters in the
reverse direction and in nonempty exclusive prefixes.

Repair: seed from the first contributing value while preserving the specified empty-prefix
identity. Test inclusive/exclusive and forward/reverse variants independently.

### D4 — Native buffers are device-scoped instead of exact-context-scoped

Locations: `synaptik_metal_foundation.m:1372,9518,9843,9858,10000`.

Buffer boxes retained the device but not a durable identity for their creating native context.
Three invocation paths checked device and extent, allowing an executable from context A to
accept a buffer from context B on the same device. The normative boundary requires the exact
creating context. Java binding already enforced this; the finding is an exposed native-ABI
validation gap, not a demonstrated public Engine bypass.

Repair: retain a safe context identity whose equality cannot be confused by address reuse and
check it before submission/mutation on all three paths. Keep lifetime ownership explicit and
test cross-context rejection with unchanged destination bytes.

### D5 — Provider-true scalar operations fail during preparation

Locations: `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalCapabilityProvider.java:1480,1640`
and `MetalNativeApi.java:875,904`; the native schema decoder has corresponding guards around
`synaptik_metal_foundation.m:8551,9428` after the 0078 integration.

Discovered while restoring Engine mixed-owner tests, after the initial audit pass. With Metal
registered first, CPU second, a native-host FLOAT32 scalar 1.25 and `x.exp().relu()`, compilation
selects `[cpu,metal]`. Session preparation then throws “Metal MPSGraph RELU input/output shape is
unsupported” before invoking native code. The provider and its test explicitly admit scalar RELU,
but Java preflight requires a scalar RELU to be inside an anchor epilogue. The same source-level
mismatch affects scalar MUL. The later native operation-shape checks accept matching scalar
shapes, but earlier native decoder guards also require anchor membership. Existing raw-ABI tests
confirm UNSUPPORTED_SHAPE; this is not only a Java guard defect.

Repair scope: align Java/native admission with already-advertised occurrences, retain separate
anchor-record authentication, rebuild the native package and advance compatibility identity to
36. Prove direct/composed scalar execution on the actual device. Check advertised
low dtypes separately and scalar CLAMP as an adjacent control. Replacing the Engine fixture with
`exp().reshape(1)` made that transfer test valid but did not close this production finding.

### E1 — Adam/AdamW intermediate order rejects a finite update

Location: `extensions/training/src/main/java/io/github/pho001/synaptik/training/TrainingParameter.java:455,540`.

The implementation divided the corrected first moment by the denominator before multiplying by
the learning rate. The documented/direct formula multiplies first. In FLOAT32, parameter 1,
learning rate and epsilon `1e-30`, beta1 0.5, beta2 0, zero decay, and gradients `1e19` then 0
produce a finite second update around `3.333e18`. The old unscaled quotient overflows and rejects
it. Existing transaction behavior correctly leaves the completed-step count at one, but the
valid documented update should not have failed. FLOAT64 uses the same erroneous ordering.

Repair: validate the scaled numerator, then divide and validate the update, in both widths.
Keep staged candidates and atomic commit unchanged. Tests cover Adam/AdamW, both widths,
intermediate overflow, state/counter preservation and successful retry.

### E2 — NN public types are not exported to consumers

Location: `extensions/nn/build.gradle.kts:2`.

NN exposes Model `Tensor`, `Shape` and `DataType` in public signatures, but declared the Model
dependency as `implementation`. A Gradle consumer depending only on NN fails to compile with
“cannot access Tensor”; adding Model manually masks the API metadata error.

Repair: use `api` for the already permitted NN→Model edge. An isolated NN-only consumer now
compiles as part of architecture validation; the existing Training-only fixture remains separate.
This exports an existing boundary, not a new module dependency or an Engine facade inside NN.

### F1 — Cache size bound is enforced after the expensive allocation

Location: `tools/tuning/src/main/java/io/github/pho001/synaptik/tools/tuning/ModelPlanCacheFile.java:505`.

The encoder accumulated all entries in a growing byte stream, copied the payload and computed
its checksum before rejecting a total above 16 MiB. Individually valid fields can have an
arbitrarily excessive aggregate within the entry-count limit. The neighboring local-workload
cache already preflights aggregate size.

Repair: count the complete schema-1 encoding with checked arithmetic before allocating the
exact-sized output. Include all length prefixes, fixed fields and checksum; retain canonical
bytes and atomic file replacement. Tests cover the inclusive limit, one byte above/below,
large logical totals without enormous allocation, fixture compatibility and unchanged target.

## Test and documentation findings

### G1 — Stable softmax and literal intermediate overflow disagree

An independent source cross-check at integrated revision `c2461e1e` found a separate conflict
beyond CPU 0021's validation repair. The following G1 source locations refer to that revision.
For admitted contiguous FLOAT32 `[1000,1000]`, axis 0, `CpuSoftmaxEmitter` max-shifts the values
to `[0,0]`, yielding `[0.5,0.5]`. `SoftmaxKind` and the foundational contract's Composite inheritance
row instead specify literal unshifted EXP, sum and division: the EXP results overflow to infinity,
and infinity divided by infinity produces NaN. This is a result-class discrepancy, not a demand
for cross-backend bit equality.

Relevant sources: `CpuCapabilityProvider.java:529`, `CpuSoftmaxInputValidator.java:101`,
`CpuSoftmaxEmitter.java:146`, `SoftmaxKind.java:19`, and
`docs/architecture/contracts/foundational-modules.md:193,265,266,274`. The alternative-algorithm
clause requires special-value preservation; the source review found no express exception for
avoiding literal intermediate overflow. Existing Stage-A/normalization/conformance tests use a
stable oracle and cannot grant that permission. No fresh numerical diagnostic was run in this
cross-check; the trigger and both evaluations are source-derived.

**Unresolved, not silently repaired:** the recommended direction is to retain useful stable
softmax and clarify its explicit finite-overflow semantics, while explicitly preserving/specifying
existing NaN/infinity/empty rules and supported/rejected domains. This changes a numerical
contract and needs an owner decision plus
independent Class C review. [Model 0033](modules/model/tasks/0033-stable-softmax-overflow-semantics.md)
records the Draft decision task. No broader composite/fusion permission follows by analogy.
An asynchronous question requested this direction; until answered and implemented, G1 remains open.

### Test and explanation repairs

- **B2:** `testing/integration-tests/.../EngineExplicitCompositionMetalIntegrationTest.java:1970,2036`
  uses EXP/SIGMOID as old CPU-only separators. Both are now supported on Metal; several rank-1,
  rank-16, multi-hop and mixed-publication tests assert only `metal` ownership. They no longer
  test the boundary named by the fixture. Retain genuine rank-zero coverage and add operations
  that actually force the intended owner sequence, with transfer directions/counts and fan-out
  assertions. Do not hardcode a false provider capability to make a test pass.
- **B3:** `modules/engine/src/test/java/.../EngineTypedLifecycleTest.java:629` submits a second
  copy then closes the Engine. Submission does not prove admission via `beginOperation`.
  Close may legally win. The race is source-confirmed; this audit did not reproduce an actual
  flaky failure. Use deterministic synchronization or split the legal outcomes without sleeps
  or a production lifecycle change.
- **B4:** `modules/engine/src/main/java/.../Engine.java:380` describes `prepareTuned` as CPU-only,
  omitting the existing bounded Metal singleton NEG session route, same-owner fallback and
  heterogeneous rejection. The glossary's matching tuning entries repeat that stale claim.
  Correct both explanations without expanding tuning capability.
- Trace `LowPrecisionTraceMetadata` Javadoc says “three execution facts” for two route/dtype facts.
  This is a local wording repair; the DTO and runtime behavior stay unchanged.
- Independent documentation review found a stale glossary opening that still described Model
  0032 as private/uncommitted and a duplicated implementation inventory that called delivered
  Compiler, backend and Training work planned. Replace that inventory with a short current
  lifecycle/status convention; preserve the glossary's term definitions and historical task records.
  The roadmap also no longer implies an ONNX mapping task exists when its table is still empty.

## Whole-project coverage and intentional limits

Counts below describe reviewed-area inventory, not per-file proof. Java production/test counts
were captured after the first five scoped repairs; integration tests live in a separate project.

| Area | Production / test Java files | Review outcome and principal scope |
|---|---:|---|
| Model | 199 / 141 | Typed operation contracts, shape/promotion/casts, saved roles; no separate confirmed Model defect. |
| Compiler | 45 / 40 | Graph/gradient/preflight/layout paths; A1/A2. Conv3d adjoints and recurrent BPTT remain intentionally unsupported. |
| Planning | 13 / 9 | Ownership/partitioning and capability use; no confirmed executable defect. |
| Runtime | 26 / 17 | Binding, leases, isolation and closure; no confirmed executable defect. |
| Prepare | 27 / 12 | Transaction/finalization/ownership; B1. |
| Engine | 23 / 10 | Admission, sessions, transfer/publication and tuning; B2–B4 in tests/docs. |
| Config | 7 / 5 | Current leaves/validation and consumers; no confirmed executable defect. |
| Trace | 22 / 6 | Current payload facts/immutability and listeners; wording issue only. |
| Backend Contract | 9 / 4 | Minimal shared capability seam and dependency direction; no confirmed defect. |
| CPU | 172 / 217 | Generated/direct families, validator ordering, composed binding and identity; C1–C5. |
| OpenBLAS provider | 8 / 6 | Discovery, four-symbol boundary, thread/route qualifications; no confirmed defect. No new real-library run. |
| Metal | 35 / 47 | Native range/context/layout, low kernels, catalog/provider truth and source/package checks; D1–D4. |
| CUDA | 1 / 0 | Marker and native scaffold, not an executable backend. |
| NN | 46 / 41 | Public API/dependencies, state/load/init and layer graph construction; E2. |
| Training | 13 / 5 | Optimizer staging, accumulation, rollback/snapshot and orchestration; E1. |
| ONNX | 1 / 0 | Marker, not an importer/exporter. |
| Tuning | 15 / 12 | Cache bounds/encoding, candidate/decision ownership and compatibility; F1. |
| Benchmarks | 3 / 0 | Report-only harness/configuration, no general performance claim. |
| CLI | 1 / 0 | Marker; no completed CLI workflow. |
| Architecture tests | 1 / 7 | Dependency restrictions and isolated public consumers; NN fixture added. |
| Backend conformance | 1 / 9 | Relevant semantic/representation regressions reviewed and extended by repairs. |
| Integration tests | 1 / 20 | Public composed execution and native routes; gradient regressions added, mixed tests repaired. |

Data, Text, Vision and Checkpoint have plans but no implementation modules. Their architecture
decisions remain gates. No `.github` workflow was present at baseline; provisioning hosted GPU
CI is a separate unresolved operational choice, not a defect silently repaired in this audit.

The Metal baseline audit reports 88 supported / 27 unsupported representative kinds, structural
execution 103/12, MPSGraph catalog 74/36/5 and custom catalog 73/42/0. These are different questions,
not interchangeable support totals. CPU support similarly must be queried for the concrete typed
occurrence; no claim of universal vector/parallel/layout coverage follows from this review.

## Risks and follow-ups, not confirmed correctness findings

1. **Training capacity:** FLOAT32 Adam can retain roughly 36 bytes per parameter element,
   including caller storage, after an update; a retained snapshot adds roughly 20 bytes and
   accumulation staging can add another 4. At 100 million elements, the first two figures are
   about 3.6 GB plus 2 GB before backend buffers. Per-array limits do not impose an aggregate
   memory budget. Measure a representative model before choosing scratch reuse, snapshot
   streaming or backend optimizer state. Do not disable rollback silently.
2. **Cold planning complexity:** `LogicalMemoryPlanning` contains value/node scanning with a
   possible quadratic cost. No benchmark establishes a present bottleneck; measure graph scaling
   before changing the ownership/interval algorithm.
3. **CPU indexed-update uniqueness:** a path in `CpuPreparedExecutable` compares updates
   quadratically. A valid concern for large index sets, not evidence of a measured regression.
4. **Static recurrent expansion:** bidirectional sequence alignment can construct work
   proportional to sequence length times squared batch size. Runtime scan adoption is already
   a separate backend-dependent program; this audit does not silently replace it.
5. **Large implementation owners:** CPU prepared execution and native Metal remain large,
   multi-branch files. Extract responsibilities only at a demonstrated seam with focused tests;
   avoid a broad cosmetic rewrite during correctness repair.
6. **Platform confidence:** actual Apple-device tests and source/package authentication do not
   prove other GPU/OS behavior. Optional BF16 OpenBLAS remains blocked on ABI/full-accumulation
   semantics; that does not block current FLOAT32/FLOAT64 routes.

## Plan reconciliation and proposed next work

The original plan problem was primarily stale live prose, not task-table status disagreement.
The initial mechanical master-row/brief comparison found no mismatch. Roadmap and Metal master
plan had accumulated repeated chronology, obsolete global “no Ready task” statements, old
NumericalProfile gates and completed work described as prerequisites.

Changes preserve historical task records while making the current map actionable:

- Compact roadmap: one active repair frontier table, all 23 area links, explicit next decisions
  and non-goals. Compact Metal master: current rules/gates plus the retained historical task table.
  The original 75 historical Metal table lines remain unchanged. Across roadmap and 23 masters,
  active-index prose shrank from 6,128 to about 4,370 lines; completed task records were not deleted.
- Config/Trace/CPU/Engine and future extension plans now describe the existing profile-free
  contract rather than resurrecting `STRICT_IEEE`/`ACCELERATOR` API or certificate gates.
- Model's unbriefed obsolete FLOAT16 0026 is Superseded by delivered consolidated work, not a
  new prerequisite. Compiler exact rewrites no longer wait for obsolete relaxed-profile policy.
- Planning's old glossary/GraphCompilationPort follow-up is already closed by Compiler 0006B10.
- Checkpoint recognizes existing Engine host materialization; durable encoding/state handoff,
  not inability to read bytes, is the missing capability.
- Training 0003 no longer waits for an undefined “Data lifecycle” that Data explicitly excludes.
  Draft decision 0003A must define exact-resume participants; caller-owned progress/sampler
  state is a recommendation, not an architecture change already made.
- Training/Checkpoint dependencies name tasks and avoid the apparent adapter cycle:
  `Training 0003A → 0003`, `Checkpoint 0001 → 0004`, then
  `Training 0004 → Checkpoint 0005 → Checkpoint 0006`.
- CUDA/ONNX/CLI markers, absent future modules, hosted GPU runner blockage, BF16 native provider
  limits and missing adjoints remain visible. Completed strict-era probe failures are not
  implicit prerequisites for today's qualified routes.

Recommended order: finish bounded repairs and the combined checkpoint; resolve Model 0033's
stable-softmax contract decision before claiming numerical closure; then select Training 0003A's
resume-state decision. If durable saving is the product priority, authorize Checkpoint's
architecture/model-only work. Data/Text/Vision form a separately selected architecture stream.
These successor proposals remain Draft until their actual decisions and scope are approved.

## Repair, review and validation ledger

| Task | Implementation / main integration | Focused and module evidence | Independent review | State |
|---|---|---|---|---|
| Compiler 0006D | `e6fb9d28` / `5dead33c` | 292 Compiler tests; 4 numerical CPU integration tests; Javadoc | APPROVE | Integrated; final checkpoint pending |
| Prepare 0009 | `ec2e7ddf` / `d7de5721` | 18 focused / 65 module tests; Javadoc | APPROVE | Integrated; final checkpoint pending |
| Training 0002C | `97e66ee3` / `1f9ea4bd` | 11 focused / 68 module tests; Javadoc | APPROVE | Integrated; final checkpoint pending |
| NN 0025B | `6eefd08d` / `7905036f` | NN-only and Training-only consumer compilation; 306 NN tests; architecture test; Javadoc | APPROVE | Integrated; final checkpoint pending |
| Tuning 0005 | `ea9059dc` / `d324ee13` | 20 focused / 73 module tests; exact encoding/bounds; Javadoc | APPROVE | Integrated; final checkpoint pending |
| CPU 0021 | `ca21c80c` / `c0559e2b` | 1,026 CPU passed / 28 skipped; 2 conformance + 6 integration passed; Javadoc | APPROVE after softmax glossary correction | Integrated; final checkpoint pending |
| CPU 0022 | Pending | Pending | Pending | In progress, serial CPU successor |
| Metal 0078 | `50f570d8` + docs `12babf46` / `6e8d6b26`, `1db53756` | 4 new device regressions, 340 Metal passed / 10 optional observer skips, 37 conformance passed; native/package/source-AIR/Javadoc | APPROVE after version-document correction | Integrated; final checkpoint pending |
| Engine 0019 | `a73822ca` / `241ce92b` | 122 Engine tests, 3 final cleanup tests; 33 actual-device integration tests without skips; Javadoc | APPROVE including glossary/user-guide corrections | Integrated; final checkpoint pending |
| Metal 0079 | Pending | Pending | Pending | In progress, serial Metal successor for D5 |

Independent reviewers inspected the actual diffs and relevant contracts/tests/Javadocs and reused
successful executable evidence rather than rerunning it merely to duplicate a result. Known
limits: Compiler new numerical cases use static contiguous FLOAT32; Prepare cleanup tests use
resource doubles; cache allocation safety is structural plus deterministic size tests, not an
intentional out-of-memory experiment; NN isolated source consumption is not a published-artifact
release test. Training's new order stress tests cover FLOAT32/FLOAT64, with existing low routing
and narrowing tests reused.
CPU's composed invalid-mask regression rejects in the NOT producer, so it does not independently
prove failure atomicity of both attention outputs. Materialized BOOL validation lacks a combined
sentinel regression, although source ordering confirms materialization executes before consumer
validation. These are bounded coverage limitations, not demonstrated remaining defects.

Independent planning reconciliation review: APPROVE after correcting the glossary cutover status,
ONNX's undefined first task and the explicit architecture gate for backend discovery. All 287
post-introduction glossary heading/explicit anchors were retained. Added D5's narrow repair brief
and final status roster receive the coordinator's final checks separately.

Final combined checkpoint, native source identity and final plan-link validation: pending.
No push is part of this review request.

Status: Incomplete
Follow-up required: complete remaining scoped repairs/reviews, run and record integrated validation,
and close the planning reconciliation.
