# CPU Task 0008O: Stable-Reduction Vector Numerical Spike

## Status

Cancelled

## Goal

Determine, before any production SIMD (single instruction, multiple data) route is admitted,
whether a vectorized stable-reduction inner loop can preserve the current observable CPU behavior
for softmax/log-softmax, dense and index categorical cross-entropy with logits, and scaled
dot-product attention. The spike produces a bounded numerical eligibility decision and immutable
evidence; it does not broaden CPU capability or production support.

## Scope

The implementation context may add CPU-private test/evidence harnesses and this task's completed
decision record only. It must leave all production CPU source, generated-artifact schema,
capability reporting, route selection, resource declarations, and public/shared contracts
unchanged.

Assess only these existing scalar families and their existing resolved, fully static occurrence
subsets:

| Family | Stable reduction being assessed | Candidate output work |
|---|---|---|
| `SOFTMAX` / `LOG_SOFTMAX` | selected-axis maximum, shifted `exp` sum, and normalization/log result | one complete selected-axis slice |
| dense categorical cross-entropy with logits | class-axis maximum and shifted `exp` sum/log-sum-exp, then dense contribution | one complete non-class sample |
| index categorical cross-entropy with logits | class-axis maximum and shifted `exp` sum/log-sum-exp, then selected-class contribution | one complete non-class sample |
| scaled dot-product attention | eligible-key score maximum and shifted `exp` sum/weight normalization; weighted value accumulation is separately observed | one complete broadcast-batch/query row |

Stage A is a numerical-permission and semantic test gate, not a performance stage. In one JVM,
for each FLOAT32 and FLOAT64 candidate, it compares the exact existing scalar generated/direct
oracle contract with the following candidates on the same typed carriers, cold geometry, range,
and logical order:

1. scalar ordered baseline;
2. vector loads/map only, with scalar logical-order folds;
3. lane-local max/sum followed by increasing-lane scalar folding; and
4. lane-reordered vector reduction (horizontal reduction or block/tree combine).

Candidates 2--4 must separately cover max pass, `exp`/sum pass, softmax division and log-softmax
log path, categorical log-sum-exp/contribution path, and attention weight/weighted-value path.
Candidate 2 retains scalar increasing-lane folding and therefore is assessed separately from the
lane-local/block candidate 3 and the lane-reordered candidate 4. If Stage A finds that candidate
3 or 4 needs permission for reassociation, changed special-value behavior, changed rounding, or a
different deterministic result class, it records `STOP_MODEL_OR_ARCHITECTURE_DECISION` before
Stage B or C. Candidate 4 is investigatory only and is never a production fallback. Do not infer
permission from Vector API availability, a tolerance that happens to pass, or benchmark speed.

The candidate matrix is exactly:

| Type | carriers | access/layout | domains | orchestration |
|---|---|---|---|---|
| FLOAT32, FLOAT64 | array/array, segment/segment, array/segment, segment/array | dense contiguous and every actually vectorizable indexed positive-stride/offset form, if Stage A proves one | widths 1, `species-1`, `species`, `species+1`, odd `2*species+3`; softmax axis first/middle/last; categorical class axis first/middle/last; masked and causal attention rows | scalar plus existing complete-slice/sample/row caller-parallel semantic ranges |

`species` is the actual preferred species lane count for the element type. Widths below one full
species are Stage-A semantic scalar controls, never timed vector rows. A missing multi-lane
preferred species records an ineligible result, not a simulated vector result. Non-contiguous
candidates may use only proven legal indexed/gather access if Java 26 Vector API support and the
current access form establish it; otherwise they are explicit scalar-only controls. No candidate
may copy, pack, materialize, allocate per invocation, or make a fixed extent part of class
identity.

FLOAT32 and FLOAT64 are the entire spike arithmetic domain. BFLOAT16, FLOAT16, integer/BOOL
arithmetic, cross-type promotion, vector transcendental approximation modes, native/provider
routes, external fusion, decomposed-graph recognition, dynamic shapes, negative strides, and any
new carrier form are excluded. Existing BFLOAT16 behavior is review-only and gains no claim.

## Out of scope

- Production Vector API emission, new `ExecutionStrategy`, schema change, capability row,
  specialization, tuning selection, or generated Class-File identity change.
- A partial reduction IR, per-worker partial buffer/workspace, cross-worker combine, atomics,
  split slice/sample/row, nested workers, or changed publication. CPU 0008P alone owns all
  deterministic partial-reduction parallelism, workspace, and combine design.
- Any Model, Compiler, Prepare, Runtime, backend-contract, Training, architecture, build,
  conformance, integration, or public API change.
- A cross-backend bitwise guarantee, a new Model numerical tolerance, or a claim that lane
  reassociation is currently permitted.
- Retry, outlier discard, benchmark replacement, threshold weakening, oracle slowdown, or use of
  an evidence result to select a route at Runtime.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [current architecture navigation](../../../../architecture/current-architecture-plan.md)
- [planning guide](../../../planning-guide.md)
- [CPU master plan](../master-plan.md)
- [CPU softmax execution](0007e-portable-stable-softmax-and-log-softmax-coverage.md)
- [CPU attention execution](0008h-portable-scaled-dot-product-attention-execution.md)
- [CPU loss execution](0008i-portable-loss-family-execution.md)
- [Model softmax semantics](../../../modules/model/tasks/0016i-softmax-semantic-kinds-and-attributes.md)
- [Model softmax expressions](../../../modules/model/tasks/0016j-softmax-tensor-expressions.md)
- [Model attention semantics](../../../modules/model/tasks/0019e-scaled-dot-product-attention.md)
- [Model loss semantics](../../../modules/model/tasks/0022-mean-squared-error-loss.md)

## Architecture constraints

- Model owns operation meaning, special-value classes, permitted reassociation, and numerical
  conformance. CPU owns truthful private lowering, realization experiments, and evidence only.
  Current `CpuSoftmaxIr`, `CpuLossIr`, and `CpuAttentionIr` remain scalar identities; the spike
  must not reinterpret their `DIRECT_SCALAR` tokens as vector permission.
- Preserve the existing generated/direct oracle rule: candidate and oracle have identical typed
  carrier signatures, cold geometry, `start`/`end`, pass structure, logical traversal, rounding
  points, and observable special-value behavior. The only investigated difference is the named
  vector operation/order. A direct oracle may not literalize fixture extents or hide a dispatcher,
  helper bridge, allocation, boxing, reflection, fallback, or slower algorithm.
- Existing CPU behavior is the Stage A baseline: max scans increasing logical class/key/axis
  order; shifted sums use the current scalar accumulation order; softmax/log, categorical, and
  attention stores retain their current formula and narrowing points. Preserve NaN, infinities,
  signed zero, underflow/overflow, equal-max ties, empty-domain behavior, masks/causality, index
  ignore/bounds validation, and deterministic complete-range ownership.
- Existing analysis-before-assignment/finalization remains intact. The spike declares no resource
  and calls no prepared production executable. Array/native-segment/mixed evidence must preserve
  the current cold binding/liveness/native-order/alignment/span/overlap checks; no carrier lookup
  occurs in a timed loop.
- Planning selects CPU ownership and CPU prepare selects a route; Runtime executes prepared
  state only. Neither planning nor Runtime may consume spike measurements as route policy.

## Package impact

Existing packages used only if harness code is necessary:

- `io.github.pho001.synaptik.backend.cpu.internal.ir` -- review-only current family identities.
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` -- review-only scalar emitters;
  a test-only candidate emitter may live beside tests, not in production emission.
- `io.github.pho001.synaptik.backend.cpu.internal.prepare` -- review-only strategy/resource
  boundary.

Packages added or changed: none in production. Test packages mirror the relevant CPU-private
family package. No production type placement is authorized.

## Affected files

Expected planning paths:

- this task;
- `docs/planning/backends/cpu/master-plan.md`; and
- `docs/planning/roadmap.md` only if its CPU frontier wording requires the new link.

Expected implementation-time paths are at most 12 CPU test/evidence-harness paths and one
immutable, repository-tracked evidence manifest (commands, environment, inputs, generated
Class-File hashes, raw measurements, and decision). No production path may change.

## Maximum scope

This task may modify or create at most 16 repository paths: the three planning paths, at most 12
test/evidence-harness paths, and one immutable evidence manifest. A production source edit, a
17th path, an unplanned dependency, a resource declaration, or a request to change Model
permission is a stop/replan condition.

## Acceptance criteria

1. Stage A records a source-backed permission table for every family/pass/candidate/type, stating
   whether exact current behavior is required, whether Model expressly permits finite
   reassociation, and whether an architecture decision is needed. Any unpermitted lane-reordered
   reduction ends the relevant candidate at `STOP_MODEL_OR_ARCHITECTURE_DECISION`; it is not
   implemented or benchmarked as a potential production route.
2. For every permitted candidate, frozen optimal clean-Java algorithms are written before timing:
   ordered max; shifted `exp` and sum; softmax division or log-softmax `shift - log(sum)`;
   categorical `max + log(sum)` and dense/index contribution; and attention eligible-score,
   normalized-weight, increasing-key weighted-value loops. They name scalar versus lane/block
   operation order and every FLOAT32/FLOAT64 narrowing point.
3. Stage A runs in one JVM without fork timing and covers the full type/family/carrier/special-
   value/axis/mask/tail matrix: NaN, both infinities, mixed infinities,
   signed-zero ties, subnormals, extreme finite shifts, underflow, overflow, singleton and odd
   tails, zero non-selected extents, softmax zero selected extent rejection, categorical empty and
   all-ignored cases, invalid non-ignored indexes, dense zero targets, all-masked attention,
   `S==0`, `Ev==0`, causal boundaries, and requested attention weights. It asserts raw bits where
   existing CPU behavior is exact; otherwise it records the current Model conformance class and
   does not invent a tolerance.
4. Stage B starts only for a Stage-A-permitted candidate. Its exact raw probe inventory is 40
   contiguous bodies: five semantic forms x two types x four ordered carrier signatures
   (array/array, segment/segment, array/segment, segment/array). If and only if Stage A proves an
   indexed vector access, add exactly ten indexed bodies: five forms x two types x one declared
   mixed carrier signature. Axis, extent, special value, mask data, worker count, and range are
   cold facts and produce no extra body. Normalize the 40 or 50 emitted Class-Files by body,
   descriptor, owner calls, and carrier-access shape while excluding class name and constant-pool
   numbering; retain exactly one dossier per distinct normalized result and list every collapsed
   probe alias. Each dossier contains source, `javap -c -v -p`, descriptor/constant-pool/
   instruction/call-owner scan, SHA-256, and normalized inventory. It proves final field-free
   direct entries, only intended Vector API/JDK primitive/`Math` or `StrictMath` calls, and no
   Synaptik numerical/reference helper, allocation, boxing, reflection, method handle,
   `invokedynamic`, collection/string dispatch, monitor, graph/layout/cache/route/resource/worker
   lookup, production fallback, or hidden scalar reference call.
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
   synchronous C2 protocol: `-Xms1g -Xmx1g -XX:-TieredCompilation -Xbatch`. There is no reason in
   this spike to depart from the completed CPU 0008M/0008N1 discipline. Per fork, run five warmup
   symmetric pairs for each V/S and V/D comparison; conservatively calibrate one shared iteration
   count until each two-invocation side reaches 50 ms; then retain exactly nine randomized
   symmetric V/S and V/D pairs, four timings per pair, each side at least 25 ms. Record result
   consumption, identical typed carriers/cold geometry/range, seed/initial order, command, JDK,
   OS, CPU, heap, preferred species, affinity/governor facts when available, calibration, and all
   raw values. There are exactly five forks per row: no retry, discard, replacement, rerun-to-pass,
   changed fixture, or post-hoc batching/threshold change. A predeclared 12-minute per-candidate
   wall-clock limit includes warmup and calibration; exceeding it records `ENVIRONMENT_LIMIT` and
   fails the candidate without replacement.
7. Every retained V/S pair, fork median, and median-of-fork-medians must be `<= 0.95x`, a
   measurable five-percent benefit rather than a `<= 1.00x` non-regression claim. Every V/D pair,
   fork median, and aggregate remains `<= 1.15x`, matching completed CPU generated/direct
   evidence. A missed gate is `KEEP_SCALAR`, never a reason to weaken a threshold. A candidate
   without a full preferred-species block is `INELIGIBLE`, not a failure or simulated pass.
8. Before the first measured fork, the manifest's inputs, code/hash identity, row inventory,
   protocol, thresholds, seed rule, and wall-clock limit become immutable; raw results are
   append-only in prescribed fork/row order. For a 30-row candidate it holds exactly 150 row/fork
   executions, 2,700 retained comparison pairs (30 x 5 x 9 x 2), 10,800 timed sides, 300 fork
   medians, and 60 aggregates; without proved indexed access the exact counts are 100 executions,
   1,800 pairs, 7,200 sides, 200 medians, and 40 aggregates. It also contains
   stopped/ineligible/N/A statuses and the final per-family decision. It must say exactly
   one of `KEEP_SCALAR`, `ELIGIBLE_FOR_SEPARATE_PRODUCTION_TASK`, or
   `STOP_MODEL_OR_ARCHITECTURE_DECISION` for each candidate family/pass. `ELIGIBLE` authorizes no
   code change; a later detailed task must independently bound production design.
9. Documentation/Javadoc review concludes no Javadoc or explanatory guide change is needed if no
   production behavior changes; it updates the task/master plan/roadmap evidence and glossary only
   if a reusable term changes. A separate clean documentation-focused pass is mandatory before
   this task becomes Complete.

## Tests / validation

Tier 1 runs focused CPU IR/lowering/emitter/preparer/binding tests unchanged, plus the one-JVM
Stage-A semantic permission matrix. Tier 2 runs the deduplicated Stage-B structural dossiers and
the opt-in compact Stage-C matrix above. The retained Stage-C minimum timing is 30 rows x five
forks x two comparisons x nine pairs x four sides x 25 ms = 270,000 ms (4.5 minutes) per
candidate when indexed access is proved, or 180,000 ms (3 minutes) for 20 rows; its predeclared
12-minute per-candidate wall-clock limit bounds warmup, calibration, and harness overhead. At most
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
and task order, tracked-manifest completeness, and no-production-path inventory. Repository-wide
validation is deferred to CPU 0009/CI because this task changes no production behavior.

## Dependencies

- CPU 0008N1 is Complete and establishes the current bounded SIMD evidence protocol precedent.
- CPU 0007E, 0008H, and 0008I are Complete and own the scalar stable-family behavior being
  preserved.
- Model softmax, attention, and loss contracts cited above remain the authority for Stage A.

## Follow-up tasks

- CPU 0008P remains the required, separate Draft follow-up for deterministic partial-reduction
  parallelism, partial IR/workspace, combine, and publication; this spike must not preempt it.
- Only a `ELIGIBLE_FOR_SEPARATE_PRODUCTION_TASK` evidence result may justify proposing a new,
  separately planned SIMD task. That task must not be created here unless it is the next valid
  frontier after 0008P.
- CPU 0009 classifies this spike evidence alongside other generated coverage; it does not treat a
  spike result as whole-backend parity.

## Architecture impact

Expected impact: None.

If Stage A finds that Model does not settle lane-reordered reduction behavior, stop the relevant
candidate and report the precise family, special-value/rounding/determinism conflict, and needed
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
  it uses the completed CPU five-fork `-XX:-TieredCompilation -Xbatch` protocol and a measurable
  `V/S <= 0.95x` gate rather than treating `<= 1.00x` as a gain.
- Stage A distinguishes a vector map with scalar ordered folding from a lane-reordered reduction;
  only the latter requires explicit reassociation permission.
- The user chose to stop Stage C after the retained bounded result rather than run replacement or
  additional forks.  Cancellation applies only to the unfinished five-fork acceptance protocol:
  it does not erase Stage A, Stage B, or the valid fork-0 `KEEP_SCALAR` decision for candidate 2.
  CPU 0008P is therefore the next valid Draft task; this cancellation authorizes neither a SIMD
  production task nor a claim that the five-fork acceptance gate passed.

## Known limitations

- This task cannot establish production SIMD support, partial parallel reductions, BFLOAT16
  behavior, native routes, or a cross-backend numerical promise.
- The exact candidate inventory may shrink only through a recorded `N/A`, `INELIGIBLE`, or
  `STOPPED` reason; it may not grow within this bounded spike.
- Candidate 2 has only fork 0. Fork 1 was externally interrupted and forks 2--4 did not run, so
  the required five-fork medians and median-of-fork-medians do not exist. Candidates 3 and 4
  remain `STOP_MODEL_OR_ARCHITECTURE_DECISION`; the cancellation does not resolve their Model or
  architecture permission boundary.

## Validation evidence

- Reused implementation context: `01a0779d-2c07-7690-b18e-a5cf037e016d`.
- Reused focused CPU command:
  `./gradlew :backends:cpu:test --tests 'io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuStableReduction*'`
  completed successfully: 24 tests, 23 passed, and one opt-in Stage-C measurement skipped.
  The focused opt-in Stage-C smoke also completed successfully. The documentation pass did not
  rerun Java tests because it changed no executable Java and must not repeat the implementation
  evidence.
- Stage B retained 40 contiguous probes and 32 normalized dossiers, with no indexed body. Its
  normalized inventory SHA-256 is
  `a0c8630ec4635b40122f250fd8ead9236ae062e524f05094eefdd4c51ce5dc21`.
- Stage C's immutable fork-0 raw CSV SHA-256 is
  `6613a62c9c31f4ca45df0aec8e23050076ffc3daca69af74ca565ede385879c2`.
  It contains 20 row/fork executions, 360 retained pairs, and 1,440 timed sides; every side is
  at least 25 ms. Of 180 V/S pairs, 2 pass and 178 fail the `<= 0.95x` gate; all 20 fork-0 V/S
  medians fail. All 180 V/D pairs pass `<= 1.15x`.
- Other immutable Stage-C identities retained by the manifest are inputs
  `fa1f0db4ff37343f48c1ddb47a273cc09792683bdc1392bd0dd1bdf6ba82e9c9`, environment
  `cfc3da5185cbc40356df6376d7c2875e66d404a2757951f5ccc13e36d0b5c349`, harness class content
  `5ad91b7e6a9f18c3635b23d83832285e0a814f9b9dbb7f1fbc3fe13196a5f7a2`, and direct-control
  class content `60b7a7ecc6b2aa4058af70087f33fc94d95d6dd4100b72f2f13337baeab51eef` (SHA-256).
- Documentation context: `01a077a1-2cca-7ed1-a047-a622a15fb949`; selected profile: Planning.
  Final documentation validation: local Markdown links, heading anchors, and fences checked;
  `git diff --check` passed after the documentation edit. The manifest was inspected as the
  immutable evidence index.

## Implementation notes

- No production path changed. The implementation added seven CPU-private `CpuStableReduction*`
  test/evidence harness files and the immutable retained-evidence manifest; all seven harness
  files and the manifest were reviewed by this documentation pass.
- Stage A permits only candidate 2, `VECTOR_MAP_ORDERED_FOLD`, for contiguous vector loads/map
  with increasing-logical-order scalar folds. It did not prove indexed Vector API access.
  Candidates 3 and 4 require an ungranted reassociation permission and remain
  `STOP_MODEL_OR_ARCHITECTURE_DECISION`.
- Stage B recorded the 40-probe/32-dossier structural result above. Stage C uses the frozen
  candidate-2 20-row contiguous matrix. Fork 0 decisively assigns `KEEP_SCALAR`: its 178 failed
  V/S pairs violate the acceptance rule that every retained V/S pair passes. This is a bounded
  candidate decision, not five-fork acceptance.
- Fork 1 was externally interrupted; forks 2--4 were not run. Thus only 20 of 100 executions,
  360 of 1,800 retained pairs, and 1,440 of 7,200 timed sides exist. No raw timing was rerun,
  replaced, discarded, or post-processed.
- The test-only Stage-C evaluator was corrected for future/replay use: it evaluates every V/S
  and V/D pair, every fork median, and the median-of-fork-medians before it can assign
  `ELIGIBLE_FOR_SEPARATE_PRODUCTION_TASK`; any failed gate assigns `KEEP_SCALAR`. It cannot be
  applied to this partial capture because it truthfully requires all five raw fork files. The
  correction changes neither frozen source/class identity nor the retained raw CSV.

## Completion summary

- Completed changes: recorded the retained Stage-A/B/fork-0 evidence, its candidate-2
  `KEEP_SCALAR` decision, the incomplete five-fork protocol, and the user's decision to stop it.
  Marked this task `Cancelled`, not `Complete`; cancellation concerns the unfinished protocol,
  not the retained evidence.
- Files changed or created: CPU-private test/evidence harnesses and the immutable manifest were
  implementation-owned; this documentation pass finalizes this task, the CPU master plan, and
  the roadmap.
- Tests and validation: reused the successful focused CPU and Stage-C smoke evidence above;
  performed planning-document, glossary, Markdown link/anchor/fence, final-diff, and whitespace
  review. No Java suite was rerun because no executable Java changed in this pass.
- Javadoc impact: none. No public API, production implementation, or behavior changed.
- Glossary impact: none. `KEEP_SCALAR`, the five-fork protocol, and this evidence disposition are
  task-local planning terms, not reusable domain terminology.
- Unresolved issues: the exact five-fork acceptance counts were not and will not be met; no
  aggregate five-fork result exists. Candidate 3/4 permission remains unresolved.
- Follow-up required: CPU 0008P is the next valid Draft task. CPU 0009 may inventory this
  retained evidence as incomplete/cancelled evidence; it must not classify it as a passed
  five-fork result.

Status: Cancelled
