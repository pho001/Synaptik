# [CPU](../../../../glossary.md#term-cpu) Task 0008O: Stable-Reduction [Vector](../../../../glossary.md#term-vector) Numerical Spike

## Reader context

This task records a numerical-permission spike for softmax/log-softmax, categorical losses with
logits, and attention, not production SIMD admission. `CpuSoftmaxIr`, `CpuLossIr`, and
`CpuAttentionIr` remain scalar identities. Candidate 2 maps vectors but folds in logical scalar
order; candidates 3 and 4 alter reduction grouping and encounter the Model-owned reassociation
boundary. Complete slices, samples, or query rows remain the units of caller-parallel ownership.

Here, Stage A provides semantic permission and result checks; Stage B supplies structural dossiers
excluding hidden helpers or fallback. Stage C compares candidate performance with scalar and
direct-Java controls. The retained fork-0 failures justify KEEP_SCALAR for candidate 2, but
interruption and cancellation leave no five-fork aggregate. No production route, schema, resource,
or shared contract changed.

Current-topic navigation (not a replacement for the historical record):

- [current first-class softmax expression meaning](../../../../reference/tensor/normalization.md#softmax-expressions) — Separates metadata construction and ideal slice meaning from backend-specific admission and execution.
- [current ordinary-production route policy](../../../../design/notes/cpu-kernel-strategy.md#current-ordinary-production-decision) — Prevents historical vector or parallel evidence from being read as current production-default authorization.

## Status

Cancelled

## Goal

Determine, before any production [SIMD](../../../../glossary.md#simd) ([single instruction, multiple data](../../../../glossary.md#simd)) route is admitted,
whether a vectorized stable-reduction inner loop can preserve the current observable [CPU](../../../../glossary.md#term-cpu) behavior
for [softmax](../../../../glossary.md#softmax--log-softmax)/[log-softmax](../../../../glossary.md#softmax--log-softmax), dense and index categorical cross-entropy with logits, and [scaled
dot-product attention](../../../../glossary.md#scaled-dot-product-attention). The spike produces a bounded numerical eligibility decision and [immutable](../../../../glossary.md#term-immutable)
evidence; it does not broaden CPU capability or production support.

## [Scope](../../../../glossary.md#term-scope)

The implementation context may add [CPU](../../../../glossary.md#term-cpu)-private test/evidence harnesses and this task's completed
decision record only. It must leave all production CPU source, generated-[artifact](../../../../glossary.md#term-artifact) [schema](../../../../glossary.md#term-schema),
capability reporting, route selection, resource declarations, and public/shared contracts
unchanged.

Assess only these existing [scalar](../../../../glossary.md#term-scalar) families and their existing resolved, fully static [occurrence](../../../../glossary.md#term-occurrence)
subsets:

| Family | Stable reduction being assessed | Candidate output work |
|---|---|---|
| [`SOFTMAX`](../../../../glossary.md#softmax--log-softmax) / `LOG_SOFTMAX` | selected-[axis](../../../../glossary.md#term-axis) maximum, shifted [`exp`](../../../../glossary.md#term-expx) sum, and [normalization](../../../../glossary.md#term-normalization)/[log](../../../../glossary.md#term-log) result | one complete selected-axis [slice](../../../../glossary.md#slice) |
| dense categorical cross-entropy with logits | class-axis maximum and shifted `exp` sum/[log-sum-exp](../../../../glossary.md#log-sum-exp), then dense contribution | one complete non-class [sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) |
| index categorical cross-entropy with logits | class-axis maximum and shifted `exp` sum/log-sum-exp, then selected-class contribution | one complete non-class sample |
| [scaled dot-product attention](../../../../glossary.md#scaled-dot-product-attention) | eligible-key score maximum and shifted `exp` sum/[weight](../../../../glossary.md#term-weight) normalization; weighted value accumulation is separately observed | one complete broadcast-[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)/query row |

Stage A is a numerical-permission and semantic test gate, not a performance stage. In one [JVM](../../../../glossary.md#term-java-virtual-machine-jvm),
for each [FLOAT32](../../../../glossary.md#term-float32) and [FLOAT64](../../../../glossary.md#term-float64) candidate, it compares the exact existing scalar generated/direct
[oracle](../../../../glossary.md#term-test-oracle) contract with the following candidates on the same typed carriers, cold geometry, range,
and logical order:

1. scalar ordered baseline;
2. [vector](../../../../glossary.md#term-vector) loads/map only, with scalar logical-order folds;
3. [lane](../../../../glossary.md#term-lane)-local max/sum followed by increasing-lane scalar folding; and
4. lane-reordered vector reduction (horizontal reduction or block/tree combine).

Candidates 2--4 must separately cover max pass, `exp`/sum pass, softmax division and [log-softmax](../../../../glossary.md#softmax--log-softmax)
log path, categorical log-sum-exp/contribution path, and [attention weight](../../../../glossary.md#attention-weight)/weighted-value path.
Candidate 2 retains scalar increasing-lane folding and therefore is assessed separately from the
lane-local/block candidate 3 and the lane-reordered candidate 4. If Stage A finds that candidate
3 or 4 needs permission for [reassociation](../../../../glossary.md#term-reassociation), changed special-value behavior, changed [rounding](../../../../glossary.md#term-rounding), or a
different deterministic result class, it records `STOP_MODEL_OR_ARCHITECTURE_DECISION` before
Stage B or C. Candidate 4 is investigatory only and is never a production fallback. Do not infer
permission from [Vector API](../../../../glossary.md#term-vector-api) availability, a [tolerance](../../../../glossary.md#term-tolerance) that happens to pass, or benchmark speed.

The candidate [matrix](../../../../glossary.md#term-matrix-meanings) is exactly:

| Type | carriers | access/[layout](../../../../glossary.md#layout) | domains | orchestration |
|---|---|---|---|---|
| FLOAT32, FLOAT64 | array/array, segment/segment, array/segment, segment/array | dense [contiguous](../../../../glossary.md#term-contiguous) and every actually vectorizable indexed positive-[stride](../../../../glossary.md#term-stride)/offset form, if Stage A proves one | widths 1, `species-1`, `species`, `species+1`, odd `2*species+3`; softmax axis first/middle/last; categorical [class axis](../../../../glossary.md#class-axis) first/middle/last; masked and causal attention rows | scalar plus existing complete-slice/sample/row caller-parallel semantic ranges |

`species` is the actual preferred species lane count for the [element](../../../../glossary.md#term-element) type. Widths below one full
species are Stage-A semantic scalar controls, never timed vector rows. A missing multi-lane
preferred species records an ineligible result, not a simulated vector result. Non-contiguous
candidates may use only proven legal indexed/[gather](../../../../glossary.md#gather) access if Java 26 Vector API support and the
current access form establish it; otherwise they are explicit scalar-only controls. No candidate
may copy, pack, materialize, allocate per invocation, or make a fixed [extent](../../../../glossary.md#term-extent) part of class
[identity](../../../../glossary.md#term-identity-meanings).

FLOAT32 and FLOAT64 are the entire spike arithmetic domain. [BFLOAT16](../../../../glossary.md#term-bfloat16), [FLOAT16](../../../../glossary.md#term-float16), integer/[BOOL](../../../../glossary.md#data-type--datatype)
arithmetic, cross-type promotion, vector transcendental approximation modes, native/provider
routes, external [fusion](../../../../glossary.md#term-fusion), decomposed-[graph](../../../../glossary.md#graph) recognition, dynamic [shapes](../../../../glossary.md#shape), negative strides, and any
new [carrier](../../../../glossary.md#term-storage-carrier) form are excluded. Existing BFLOAT16 behavior is review-only and gains no claim.

## Out of [scope](../../../../glossary.md#term-scope)

- Production [Vector API](../../../../glossary.md#term-vector-api) emission, new `ExecutionStrategy`, schema change, capability row,
  specialization, tuning selection, or generated Class-File identity change.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [specialization glossary](../../../../glossary.md#term-specialization); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- A partial reduction IR, per-worker partial buffer/workspace, cross-worker combine, atomics,
  split slice/sample/row, nested workers, or changed publication. CPU 0008P alone owns all
  deterministic partial-reduction parallelism, workspace, and combine design.<span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [buffer glossary](../../../../glossary.md#term-buffer); [slice glossary](../../../../glossary.md#slice); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [publication glossary](../../../../glossary.md#term-publication); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Any Model, Compiler, Prepare, Runtime, backend-contract, Training, architecture, build,
  conformance, integration, or public API change.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- A cross-backend bitwise guarantee, a new Model numerical tolerance, or a claim that lane
  reassociation is currently permitted.<span data-glossary-navigation="true"> ([tolerance glossary](../../../../glossary.md#term-tolerance); [lane glossary](../../../../glossary.md#term-lane); [reassociation glossary](../../../../glossary.md#term-reassociation))</span>
- Retry, outlier discard, benchmark replacement, threshold weakening, oracle slowdown, or use of
  an evidence result to select a route at Runtime.<span data-glossary-navigation="true"> ([outlier glossary](../../../../glossary.md#term-outlier); [oracle glossary](../../../../glossary.md#term-test-oracle))</span>

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [current architecture navigation](../../../../architecture/current-architecture-plan.md)
- [planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- [CPU master plan](../master-plan.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- [CPU softmax execution](0007e-portable-stable-softmax-and-log-softmax-coverage.md)<span data-glossary-navigation="true"> ([softmax glossary](../../../../glossary.md#softmax--log-softmax))</span>
- [CPU attention execution](0008h-portable-scaled-dot-product-attention-execution.md)
- [CPU loss execution](0008i-portable-loss-family-execution.md)<span data-glossary-navigation="true"> ([loss glossary](../../../../glossary.md#loss))</span>
- [Model softmax semantics](../../../modules/model/tasks/0016i-softmax-semantic-kinds-and-attributes.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- [Model softmax expressions](../../../modules/model/tasks/0016j-softmax-tensor-expressions.md)
- [Model attention semantics](../../../modules/model/tasks/0019e-scaled-dot-product-attention.md)
- [Model loss semantics](../../../modules/model/tasks/0022-mean-squared-error-loss.md)

## Architecture constraints

- Model owns operation meaning, special-value classes, permitted reassociation, and numerical
  conformance. CPU owns truthful private lowering, realization experiments, and evidence only.
  Current <span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [operation glossary](../../../../glossary.md#operation); [reassociation glossary](../../../../glossary.md#term-reassociation); [conformance glossary](../../../../glossary.md#term-conformance); [CPU glossary](../../../../glossary.md#term-cpu); [lowering glossary](../../../../glossary.md#lowering))</span>`CpuSoftmaxIr`, `CpuLossIr`, and `CpuAttentionIr` remain scalar identities; the spike
  must not reinterpret their <span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>`DIRECT_SCALAR` [tokens](../../../../glossary.md#term-token) as [vector](../../../../glossary.md#term-vector) permission.
- Preserve the existing generated/direct oracle rule: candidate and oracle have identical typed
  carrier signatures, cold geometry, <span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle); [carrier glossary](../../../../glossary.md#term-storage-carrier); [signatures glossary](../../../../glossary.md#term-signature))</span>`start`/`end`, pass structure, logical traversal, rounding
  points, and observable special-value behavior. The only investigated difference is the named
  vector operation/order. A direct oracle may not literalize fixture extents or hide a dispatcher,
  helper bridge, allocation, boxing, reflection, fallback, or slower algorithm.<span data-glossary-navigation="true"> ([rounding glossary](../../../../glossary.md#term-rounding); [fixture glossary](../../../../glossary.md#term-fixture))</span>
- Existing CPU behavior is the Stage A baseline: max scans increasing logical class/key/axis
  order; shifted sums use the current scalar accumulation order; softmax/log, categorical, and
  attention stores retain their current formula and narrowing points. Preserve NaN, infinities,
  signed zero, underflow/overflow, equal-max ties, empty-domain behavior, masks/causality, index
  ignore/bounds validation, and deterministic complete-range ownership.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [softmax glossary](../../../../glossary.md#softmax--log-softmax); [log glossary](../../../../glossary.md#term-log); [narrowing glossary](../../../../glossary.md#term-narrowing); [NaN glossary](../../../../glossary.md#term-nan); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [signed zero glossary](../../../../glossary.md#term-signed-zero); [underflow glossary](../../../../glossary.md#term-underflow); [overflow glossary](../../../../glossary.md#term-overflow); [masks glossary](../../../../glossary.md#term-boolean-mask); [validation glossary](../../../../glossary.md#term-validation); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Existing analysis-before-assignment/finalization remains intact. The spike declares no resource
  and calls no prepared production executable. Array/native-segment/mixed evidence must preserve
  the current cold binding/liveness/native-order/alignment/span/overlap checks; no carrier lookup
  occurs in a timed loop.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [liveness glossary](../../../../glossary.md#term-liveness); [alignment glossary](../../../../glossary.md#term-alignment); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- Planning selects CPU ownership and CPU prepare selects a route; Runtime executes prepared
  state only. Neither planning nor Runtime may consume spike measurements as route policy.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime))</span>

## Package impact

Existing packages used only if harness code is necessary:

- `io.github.pho001.synaptik.backend.cpu.internal.ir` -- review-only current family [identities](../../../../glossary.md#term-identity-meanings).
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` -- review-only scalar emitters;
  a test-only candidate emitter may live beside tests, not in production emission.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.prepare` -- review-only strategy/resource
  boundary.

Packages added or changed: none in production. Test packages mirror the relevant [CPU](../../../../glossary.md#term-cpu)-private
family package. No production type placement is authorized.

## Affected files

Expected [planning](../../../../glossary.md#term-planning) paths:

- this task;
- `docs/planning/backends/cpu/master-plan.md`; and
- `docs/planning/roadmap.md` only if its [CPU](../../../../glossary.md#term-cpu) frontier wording requires the new link.

Expected implementation-time paths are at most 12 CPU test/evidence-harness paths and one
[immutable](../../../../glossary.md#term-immutable), repository-tracked evidence [manifest](../../../../glossary.md#term-manifest) (commands, environment, inputs, generated
Class-File [hashes](../../../../glossary.md#term-hash), raw measurements, and decision). No production path may change.

## Maximum [scope](../../../../glossary.md#term-scope)

This task may modify or create at most 16 repository paths: the three [planning](../../../../glossary.md#term-planning) paths, at most 12
test/evidence-harness paths, and one [immutable](../../../../glossary.md#term-immutable) evidence [manifest](../../../../glossary.md#term-manifest). A production source edit, a
17th path, an unplanned [dependency](../../../../glossary.md#term-dependency), a resource declaration, or a request to change [Model](../../../../glossary.md#term-model)
permission is a stop/replan condition.

## Acceptance criteria

1. Stage A records a source-backed permission table for every family/pass/candidate/type, stating
   whether exact current behavior is required, whether Model expressly permits finite
   reassociation, and whether an architecture decision is needed. Any unpermitted lane-reordered
   reduction ends the relevant candidate at <span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [finite glossary](../../../../glossary.md#term-finite); [reassociation glossary](../../../../glossary.md#term-reassociation); [lane glossary](../../../../glossary.md#term-lane))</span>`STOP_MODEL_OR_ARCHITECTURE_DECISION`; it is not
   implemented or benchmarked as a potential production route.
2. For every permitted candidate, frozen optimal clean-Java algorithms are written before timing:
   ordered max; shifted [`exp`](../../../../glossary.md#term-expx) and sum; [softmax](../../../../glossary.md#softmax--log-softmax) division or [log-softmax](../../../../glossary.md#softmax--log-softmax) `shift - log(sum)`;
   categorical `max + log(sum)` and dense/index contribution; and attention eligible-score,
   normalized-weight, increasing-key weighted-value loops. They name scalar versus lane/block
   operation order and every FLOAT32/FLOAT64 narrowing point.<span data-glossary-navigation="true"> ([weight glossary](../../../../glossary.md#term-weight); [scalar glossary](../../../../glossary.md#term-scalar); [operation glossary](../../../../glossary.md#operation); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [narrowing glossary](../../../../glossary.md#term-narrowing))</span>
3. Stage A runs in one JVM without fork timing and covers the full type/family/carrier/special-
   value/axis/mask/tail matrix: NaN, both infinities, mixed infinities,
   signed-zero ties, subnormals, extreme finite shifts, underflow, overflow, singleton and odd
   tails, zero non-selected extents, softmax zero selected extent rejection, categorical empty and
   all-ignored cases, invalid non-ignored indexes, dense zero targets, all-masked attention,<span data-glossary-navigation="true"> ([JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm); [carrier glossary](../../../../glossary.md#term-storage-carrier); [axis glossary](../../../../glossary.md#term-axis); [mask glossary](../../../../glossary.md#term-boolean-mask); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [NaN glossary](../../../../glossary.md#term-nan); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [subnormals glossary](../../../../glossary.md#term-subnormal); [underflow glossary](../../../../glossary.md#term-underflow); [overflow glossary](../../../../glossary.md#term-overflow); [singleton glossary](../../../../glossary.md#term-singleton); [extent glossary](../../../../glossary.md#term-extent); [targets glossary](../../../../glossary.md#term-target))</span>
   `S==0`, `Ev==0`, causal boundaries, and requested attention weights. It asserts raw bits where
   existing CPU behavior is exact; otherwise it records the current Model conformance class and
   does not invent a tolerance.<span data-glossary-navigation="true"> ([raw bits glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [CPU glossary](../../../../glossary.md#term-cpu); [conformance glossary](../../../../glossary.md#term-conformance); [tolerance glossary](../../../../glossary.md#term-tolerance))</span>
4. Stage B starts only for a Stage-A-permitted candidate. Its exact raw probe inventory is 40
   contiguous bodies: five semantic forms x two types x four ordered carrier signatures
   (array/array, segment/segment, array/segment, segment/array). If and only if Stage A proves an
   indexed vector access, add exactly ten indexed bodies: five forms x two types x one declared
   mixed carrier signature. Axis, extent, special value, mask data, worker count, and range are
   cold facts and produce no extra body. Normalize the 40 or 50 emitted Class-Files by body,
   descriptor, owner calls, and carrier-access shape while excluding class name and constant-pool
   numbering; retain exactly one dossier per distinct normalized result and list every collapsed
   probe alias. Each dossier contains source, <span data-glossary-navigation="true"> ([contiguous glossary](../../../../glossary.md#term-contiguous); [signatures glossary](../../../../glossary.md#term-signature); [vector glossary](../../../../glossary.md#term-vector); [descriptor glossary](../../../../glossary.md#term-descriptor); [owner glossary](../../../../glossary.md#term-owner); [shape glossary](../../../../glossary.md#shape); [constant glossary](../../../../glossary.md#term-constant); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>`javap -c -v -p`, descriptor/constant-pool/
   instruction/call-owner scan, SHA-256, and normalized inventory. It proves final field-free
   direct entries, only intended Vector API/JDK primitive/<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan); [SHA-256 glossary](../../../../glossary.md#term-sha-256); [Vector API glossary](../../../../glossary.md#term-vector-api); [JDK glossary](../../../../glossary.md#term-java-development-kit-jdk))</span>`Math` or `StrictMath` calls, and no
   Synaptik numerical/reference helper, allocation, boxing, reflection, method handle,<span data-glossary-navigation="true"> ([method handle glossary](../../../../glossary.md#term-method-handle))</span>
   `invokedynamic`, collection/string dispatch, monitor, graph/layout/cache/route/resource/worker
   lookup, production fallback, or hidden scalar reference call.<span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch); [graph glossary](../../../../glossary.md#graph); [layout glossary](../../../../glossary.md#layout); [cache glossary](../../../../glossary.md#term-cache); [lookup glossary](../../../../glossary.md#term-lookup))</span>
5. Stage C starts only for a Stage-A-permitted candidate that has a full preferred-species block.
   Its contiguous core is exactly 20 timed rows: five semantic forms x two types x two widths
   (`species` exact and odd `2*species+3` tail). For each form, FLOAT32 exact/tail rows use
   array/array and array/segment respectively; FLOAT64 exact/tail rows use segment/segment and
   segment/array respectively. This covers all four carrier signatures without multiplying them.
   If Stage A proves a vectorizable indexed access, add exactly ten indexed rows: five forms x two
   types x the odd tail width, using the opposite mixed signature from that type's contiguous tail.
   Thus Stage C has exactly 30 rows per permitted candidate when indexed access is proved, or
   exactly 20 rows when it is not. Widths `1` and `species-1` remain Stage-A-only scalar controls.
   Current caller-parallel execution invokes the same `start`/`end` body from external workers;
   it has no distinct emitted body, so it adds zero timed rows and receives Stage-A range/
   determinism coverage. A later candidate with a distinct parallel body must add exactly one
   serial/parallel replacement row per affected semantic/type form and amend this task before
   timing; it cannot silently multiply the matrix.
6. Every Stage-C row uses five fresh Java 26 child JVM forks with the established fixed heap and
   synchronous C2 protocol: <span data-glossary-navigation="true"> ([JVM forks glossary](../../../../glossary.md#java-preview-incubator-modules-and-compiler-tiers))</span><span data-glossary-navigation="true"> ([heap glossary](../../../../glossary.md#term-heap); [synchronous glossary](../../../../glossary.md#term-synchronous); [C2 glossary](../../../../glossary.md#term-c2))</span>`-Xms1g -Xmx1g -XX:-TieredCompilation -Xbatch`. There is no reason in
   this spike to depart from the completed CPU 0008M/0008N1 discipline. Per fork, run five warmup
   symmetric pairs for each V/S and V/D comparison; conservatively calibrate one shared iteration
   count until each two-invocation side reaches 50 ms; then retain exactly nine randomized
   symmetric V/S and V/D pairs, four timings per pair, each side at least 25 ms. Record result
   consumption, identical typed carriers/cold geometry/range, seed/initial order, command, JDK,
   OS, CPU, heap, preferred species, affinity/governor facts when available, calibration, and all
   raw values. There are exactly five forks per row: no retry, discard, replacement, rerun-to-pass,
   changed fixture, or post-hoc batching/threshold change. A predeclared 12-minute per-candidate
   wall-clock limit includes warmup and calibration; exceeding it records <span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [warmup glossary](../../../../glossary.md#term-warmup); [seed glossary](../../../../glossary.md#term-seed); [fixture glossary](../../../../glossary.md#term-fixture))</span>`ENVIRONMENT_LIMIT` and
   fails the candidate without replacement.
7. Every retained V/S pair, [fork](../../../../glossary.md#term-fork-meanings) [median](../../../../glossary.md#term-median), and median-of-fork-medians must be `<= 0.95x`, a
   measurable five-percent benefit rather than a `<= 1.00x` non-regression claim. Every V/D pair,
   fork median, and aggregate remains <span data-glossary-navigation="true"> ([regression glossary](../../../../glossary.md#term-regression))</span>`<= 1.15x`, matching completed CPU generated/direct
   evidence. A missed gate is `KEEP_SCALAR`, never a reason to weaken a threshold. A candidate
   without a full preferred-species block is `INELIGIBLE`, not a failure or simulated pass.
8. Before the first measured fork, the manifest's inputs, code/hash identity, row inventory,
   protocol, thresholds, seed rule, and wall-clock limit become immutable; raw results are
   append-only in prescribed fork/row order. For a 30-row candidate it holds exactly 150 row/fork
   executions, 2,700 retained comparison pairs (30 x 5 x 9 x 2), 10,800 timed sides, 300 fork
   medians, and 60 aggregates; without proved indexed access the exact counts are 100 executions,
   1,800 pairs, 7,200 sides, 200 medians, and 40 aggregates. It also contains
   stopped/ineligible/N/A statuses and the final per-family decision. It must say exactly
   one of <span data-glossary-navigation="true"> ([manifest glossary](../../../../glossary.md#term-manifest); [hash glossary](../../../../glossary.md#term-hash); [identity glossary](../../../../glossary.md#term-identity-meanings); [immutable glossary](../../../../glossary.md#term-immutable))</span>`KEEP_SCALAR`, `ELIGIBLE_FOR_SEPARATE_PRODUCTION_TASK`, or
   `STOP_MODEL_OR_ARCHITECTURE_DECISION` for each candidate family/pass. `ELIGIBLE` authorizes no
   code change; a later detailed task must independently bound production design.
9. Documentation/Javadoc review concludes no Javadoc or explanatory guide change is needed if no
   production behavior changes; it updates the task/master plan/roadmap evidence and glossary only
   if a reusable term changes. A separate clean documentation-focused pass is mandatory before
   this task becomes Complete.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

Tier 1 runs focused [CPU](../../../../glossary.md#term-cpu) [IR](../../../../glossary.md#term-intermediate-representation-ir)/[lowering](../../../../glossary.md#lowering)/emitter/preparer/[binding](../../../../glossary.md#term-binding) tests unchanged, plus the one-[JVM](../../../../glossary.md#term-java-virtual-machine-jvm)
Stage-A semantic permission [matrix](../../../../glossary.md#term-matrix-meanings). Tier 2 runs the deduplicated Stage-B structural dossiers and
the opt-in compact Stage-C matrix above. The retained Stage-C minimum timing is 30 rows x five
[forks](../../../../glossary.md#term-fork-meanings) x two comparisons x nine pairs x four sides x 25 ms = 270,000 ms (4.5 minutes) per
candidate when indexed access is proved, or 180,000 ms (3 minutes) for 20 rows; its predeclared
12-minute per-candidate wall-clock limit bounds [warmup](../../../../glossary.md#term-warmup), calibration, and harness overhead. At most
three candidates exist, so Stage C has a 36-minute total wall-clock cap before ordinary build and
documentation validation. Tier 3 runs once after executable test-harness stabilization:

```bash
./gradlew :backends:cpu:test
./gradlew :backends:cpu:javadoc
git diff --check
```

The performance command must require an explicit evidence root and
`SYNAPTIK_CPU_STABLE_REDUCTION_SPIKE=true`; it must fail rather than time an unspecified protocol.
The final evidence records every command and result, Markdown links/anchors/fences, exact status
and task order, tracked-[manifest](../../../../glossary.md#term-manifest) completeness, and no-production-path inventory. Repository-wide
validation is deferred to CPU 0009/[CI](../../../../glossary.md#term-continuous-integration-ci) because this task changes no production behavior.

## Dependencies

- [CPU](../../../../glossary.md#term-cpu) 0008N1 is Complete and establishes the current bounded [SIMD](../../../../glossary.md#simd) evidence protocol precedent.
- CPU 0007E, 0008H, and 0008I are Complete and own the scalar stable-family behavior being
  preserved.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>
- [Model](../../../../glossary.md#term-model) [softmax](../../../../glossary.md#softmax--log-softmax), attention, and [loss](../../../../glossary.md#loss) contracts cited above remain the authority for Stage A.

## Follow-up tasks

- CPU 0008P remains the required, separate Draft follow-up for deterministic partial-reduction
  parallelism, partial IR/workspace, combine, and publication; this spike must not preempt it.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [publication glossary](../../../../glossary.md#term-publication))</span>
- Only a `ELIGIBLE_FOR_SEPARATE_PRODUCTION_TASK` evidence result may justify proposing a new,
  separately planned SIMD task. That task must not be created here unless it is the next valid
  frontier after 0008P.<span data-glossary-navigation="true"> ([SIMD glossary](../../../../glossary.md#simd))</span>
- CPU 0009 classifies this spike evidence alongside other generated coverage; it does not treat a
  spike result as whole-backend parity.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>

## Architecture impact

Expected impact: None.

If Stage A finds that [Model](../../../../glossary.md#term-model) does not settle [lane](../../../../glossary.md#term-lane)-reordered reduction behavior, stop the relevant
candidate and report the precise family, special-value/[rounding](../../../../glossary.md#term-rounding)/determinism conflict, and needed
Model or architecture decision. Do not edit architecture or Model documentation to manufacture
permission.

## Implementation prompt

Use this prompt in a separate agentic task/thread:

```text
You are working in the Synaptik repository. Read AGENTS.md, ARCHITECTURE.md,
docs/planning/planning-guide.md, and docs/planning/backends/cpu/tasks/
0008o-stable-reduction-vector-numerical-spike.md. Implement this task exactly as specified.
Do not change production Java, capability reporting, schema, route selection, resources, or public
contracts. Stop and report any Model/architecture permission conflict or scope conflict. Do not
commit or push unless separately authorized. After the implementation evidence is complete, hand
the diff and recorded validation evidence to a separate documentation-focused clean-context agent
for the required targeted review and final planning/documentation validation.
```

## Local decisions

- Broad Stage-A semantic coverage and compact Stage-C timing are deliberately separate. Stage C
  has 20 contiguous rows, plus ten indexed rows only after an indexed Vector API access proof;
  it uses the completed CPU five-fork <span data-glossary-navigation="true"> ([contiguous glossary](../../../../glossary.md#term-contiguous); [Vector API glossary](../../../../glossary.md#term-vector-api); [CPU glossary](../../../../glossary.md#term-cpu); [fork glossary](../../../../glossary.md#term-fork-meanings))</span>`-XX:-TieredCompilation -Xbatch` protocol and a measurable
  `V/S <= 0.95x` gate rather than treating `<= 1.00x` as a gain.
- Stage A distinguishes a vector map with scalar ordered folding from a lane-reordered reduction;
  only the latter requires explicit reassociation permission.<span data-glossary-navigation="true"> ([vector glossary](../../../../glossary.md#term-vector); [scalar glossary](../../../../glossary.md#term-scalar); [lane glossary](../../../../glossary.md#term-lane); [reassociation glossary](../../../../glossary.md#term-reassociation))</span>
- The user chose to stop Stage C after the retained bounded result rather than run replacement or
  additional forks.  Cancellation applies only to the unfinished five-fork acceptance protocol:
  it does not erase Stage A, Stage B, or the valid fork-0 <span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [Cancellation glossary](../../../../glossary.md#term-cancellation))</span>`KEEP_SCALAR` decision for candidate 2.
  CPU 0008P is therefore the next valid Draft task; this cancellation authorizes neither a SIMD
  production task nor a claim that the five-fork acceptance gate passed.<span data-glossary-navigation="true"> ([SIMD glossary](../../../../glossary.md#simd))</span>

## Known limitations

- This task cannot establish production SIMD support, partial parallel reductions, BFLOAT16
  behavior, native routes, or a cross-backend numerical promise.<span data-glossary-navigation="true"> ([SIMD glossary](../../../../glossary.md#simd); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [backend glossary](../../../../glossary.md#backend))</span>
- The exact candidate inventory may shrink only through a recorded `N/A`, `INELIGIBLE`, or
  `STOPPED` reason; it may not grow within this bounded spike.
- Candidate 2 has only fork 0. Fork 1 was externally interrupted and forks 2--4 did not run, so
  the required five-fork medians and median-of-fork-medians do not exist. Candidates 3 and 4
  remain <span data-glossary-navigation="true"> ([fork glossary](../../../../glossary.md#term-fork-meanings); [run glossary](../../../../glossary.md#term-run); [median glossary](../../../../glossary.md#term-median))</span>`STOP_MODEL_OR_ARCHITECTURE_DECISION`; the cancellation does not resolve their Model or
  architecture permission boundary.<span data-glossary-navigation="true"> ([cancellation glossary](../../../../glossary.md#term-cancellation); [Model glossary](../../../../glossary.md#term-model))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- Reused implementation context: `01a0779d-2c07-7690-b18e-a5cf037e016d`.
- Reused focused [CPU](../../../../glossary.md#term-cpu) command:
  `./gradlew :backends:cpu:test --tests 'io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuStableReduction*'`
  completed successfully: 24 tests, 23 passed, and one opt-in Stage-C measurement skipped.
  The focused opt-in Stage-C smoke also completed successfully. The documentation pass did not
  rerun Java tests because it changed no executable Java and must not repeat the implementation
  evidence.<span data-glossary-navigation="true"> ([smoke glossary](../../../../glossary.md#term-smoke-test))</span>
- Stage B retained 40 contiguous probes and 32 normalized dossiers, with no indexed body. Its
  normalized inventory SHA-256 is<span data-glossary-navigation="true"> ([contiguous glossary](../../../../glossary.md#term-contiguous))</span>
  `a0c8630ec4635b40122f250fd8ead9236ae062e524f05094eefdd4c51ce5dc21`.
- Stage C's [immutable](../../../../glossary.md#term-immutable) [fork](../../../../glossary.md#term-fork-meanings)-0 raw CSV [SHA-256](../../../../glossary.md#term-sha-256) is
  `6613a62c9c31f4ca45df0aec8e23050076ffc3daca69af74ca565ede385879c2`.
  It contains 20 row/fork executions, 360 retained pairs, and 1,440 timed sides; every side is
  at least 25 ms. Of 180 V/S pairs, 2 pass and 178 fail the `<= 0.95x` gate; all 20 fork-0 V/S
  medians fail. All 180 V/D pairs pass `<= 1.15x`.
- Other immutable Stage-C [identities](../../../../glossary.md#term-identity-meanings) retained by the [manifest](../../../../glossary.md#term-manifest) are inputs
  `fa1f0db4ff37343f48c1ddb47a273cc09792683bdc1392bd0dd1bdf6ba82e9c9`, environment
  `cfc3da5185cbc40356df6376d7c2875e66d404a2757951f5ccc13e36d0b5c349`, harness class content
  `5ad91b7e6a9f18c3635b23d83832285e0a814f9b9dbb7f1fbc3fe13196a5f7a2`, and direct-control
  class content `60b7a7ecc6b2aa4058af70087f33fc94d95d6dd4100b72f2f13337baeab51eef` (SHA-256).
- Documentation context: `01a077a1-2cca-7ed1-a047-a622a15fb949`; selected profile: Planning.
  Final documentation validation: local Markdown links, heading anchors, and fences checked;<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
  `git diff --check` passed after the documentation edit. The manifest was inspected as the
  immutable evidence index.

## Implementation notes

- No production path changed. The implementation added seven [CPU](../../../../glossary.md#term-cpu)-private `CpuStableReduction*`
  test/evidence harness files and the immutable retained-evidence manifest; all seven harness
  files and the manifest were reviewed by this documentation pass.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [manifest glossary](../../../../glossary.md#term-manifest))</span>
- Stage A permits only candidate 2, `VECTOR_MAP_ORDERED_FOLD`, for contiguous vector loads/map
  with increasing-logical-order scalar folds. It did not prove indexed Vector API access.
  Candidates 3 and 4 require an ungranted reassociation permission and remain<span data-glossary-navigation="true"> ([contiguous glossary](../../../../glossary.md#term-contiguous); [vector glossary](../../../../glossary.md#term-vector); [scalar glossary](../../../../glossary.md#term-scalar); [Vector API glossary](../../../../glossary.md#term-vector-api); [reassociation glossary](../../../../glossary.md#term-reassociation))</span>
  `STOP_MODEL_OR_ARCHITECTURE_DECISION`.
- Stage B recorded the 40-probe/32-dossier structural result above. Stage C uses the frozen
  candidate-2 20-row contiguous matrix. Fork 0 decisively assigns <span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings); [Fork glossary](../../../../glossary.md#term-fork-meanings))</span>`KEEP_SCALAR`: its 178 failed
  V/S pairs violate the acceptance rule that every retained V/S pair passes. This is a bounded
  candidate decision, not five-fork acceptance.
- Fork 1 was externally interrupted; forks 2--4 were not run. Thus only 20 of 100 executions,
  360 of 1,800 retained pairs, and 1,440 of 7,200 timed sides exist. No raw timing was rerun,
  replaced, discarded, or post-processed.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>
- The test-only Stage-C evaluator was corrected for future/replay use: it evaluates every V/S
  and V/D pair, every fork median, and the median-of-fork-medians before it can assign<span data-glossary-navigation="true"> ([median glossary](../../../../glossary.md#term-median))</span>
  `ELIGIBLE_FOR_SEPARATE_PRODUCTION_TASK`; any failed gate assigns `KEEP_SCALAR`. It cannot be
  applied to this partial capture because it truthfully requires all five raw fork files. The
  correction changes neither frozen source/class identity nor the retained raw CSV.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>

## Completion summary

- Completed changes: recorded the retained Stage-A/B/[fork](../../../../glossary.md#term-fork-meanings)-0 evidence, its candidate-2
  `KEEP_SCALAR` decision, the incomplete five-fork protocol, and the user's decision to stop it.
  Marked this task `Cancelled`, not `Complete`; cancellation concerns the unfinished protocol,
  not the retained evidence.<span data-glossary-navigation="true"> ([cancellation glossary](../../../../glossary.md#term-cancellation))</span>
- Files changed or created: CPU-private test/evidence harnesses and the immutable manifest were
  implementation-owned; this documentation pass finalizes this task, the CPU master plan, and
  the roadmap.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [immutable glossary](../../../../glossary.md#term-immutable); [manifest glossary](../../../../glossary.md#term-manifest))</span>
- Tests and validation: reused the successful focused CPU and Stage-C smoke evidence above;
  performed planning-document, glossary, Markdown link/anchor/fence, final-diff, and whitespace
  review. No Java suite was rerun because no executable Java changed in this pass.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [smoke glossary](../../../../glossary.md#term-smoke-test); [planning glossary](../../../../glossary.md#term-planning))</span>
- [Javadoc](../../../../glossary.md#term-javadoc) impact: none. No public [API](../../../../glossary.md#term-application-programming-interface-api), production implementation, or behavior changed.
- Glossary impact: none. `KEEP_SCALAR`, the five-fork protocol, and this evidence disposition are
  task-local planning terms, not reusable domain terminology.
- Unresolved issues: the exact five-fork acceptance counts were not and will not be met; no
  aggregate five-fork result exists. Candidate 3/4 permission remains unresolved.
- Follow-up required: CPU 0008P is the next valid Draft task. CPU 0009 may inventory this
  retained evidence as incomplete/cancelled evidence; it must not classify it as a passed
  five-fork result.

Status: Cancelled
