# CPU Task 0008I: Portable Loss-Family Execution

## Status

Complete, with the corrected full performance gate waived/closed by project decision.

## Goal

Execute the complete currently represented Model loss inventory on the portable CPU route: exact-
shape mean-squared error (MSE), dense-target categorical cross-entropy with logits, and
index-target categorical cross-entropy with logits. The task makes every currently supported
reduction and index-ignore form executable only when a static, resolved, direct generated
realization can prove the Model/Compiler contract. It preserves the loss occurrence as one atomic
numerical unit and publishes its sole requested output exactly once.

## Scope

### Source-backed occurrence matrix

Admit only fixed two-input/one-output `LossKind` occurrences with their exact attribute class.
`NONE`, `SUM`, and `MEAN` are the complete `LossReduction` enum in its current order.

| Family | Ordered inputs | Attributes | Result type and `NONE` shape | Reduction denominator |
|---|---|---|---|---|
| `MEAN_SQUARED_ERROR` | prediction, target; both floating and positional-exact Shape | `MeanSquaredErrorAttrs` | floating promotion; prediction Shape | complete logical element count |
| `DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS` | logits, dense floating target; exact Shape | normalized class axis, reduction | floating promotion; logits Shape with class axis removed | non-class sample count |
| `INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS` | floating logits, exact INT32/INT64 target; target Shape is logits Shape with class axis removed | normalized axis, reduction, optional same-type ignore scalar | logits type; exact target Shape | sample count without ignore; non-ignored count with ignore |

All three families have exactly one output slot. They require fully static Shapes, resolved
layouts, non-negative offsets/strides, injective output writes, representable element/address/
span/range arithmetic, and BFLOAT16/FLOAT32/FLOAT64 floating operands only. MSE and dense use all
nine ordered floating type pairs; their result type follows `DataTypePromotion.promoteFloating`.
Index uses all three logits types, both index widths, and both absent/present ignore forms; the
index target never promotes or casts. Equal semantic input roles may share a read boundary only
when descriptors/types make that representation valid; lowering retains ordered role-to-unique-
boundary mapping.

Direct generated forms are: element-range MSE `NONE`; sample-range dense/index `NONE`; and one
complete reduced-domain range for each `SUM`/`MEAN` form. The latter deliberately has no partial
reduction tree, combine phase, or workspace: its one range owns the complete ordered domain and
its scalar output. `NONE` ranges own complete independent elements/samples and use existing scalar
or caller-parallel range selection. There is zero workspace for every form. Dynamic dimensions,
unresolved/non-injective outputs, malformed signatures, non-static index class extent, or an
unproved carrier/overflow/alias fact fail closed before generated realization; there is no
decomposition into public operations, generic interpreter, reference fallback, materialization,
or silent slower path.

### Frozen optimal-clean-Java oracle and exceptional-value contract

- Freeze one ordinary optimal clean-Java oracle for each exact generated identity and emit its same
  algorithm, selected hot-loop/dataflow shape, and avoidable-overhead profile. The direct oracle
  has the same typed carrier arguments, cold primitive geometry payload, and `start`/`end` range
  arguments as the generated entry; only Java source compiled by `javac`, rather than direct
  Class-File emission, differs. It is allocation-free during invocation, uses no production helper
  or fallback, and selects its direct typed loop before timing. MSE visits increasing logical
  element order, decodes both operands to the promoted accumulator, computes exactly
  `difference = prediction - target; loss = difference * difference`, stores `NONE` once, and for
  `SUM`/`MEAN` left-associatively accumulates each loss then divides once by the typed logical
  element count supplied through cold geometry. BFLOAT16/FLOAT32 use binary32; FLOAT64 uses
  binary64. Empty `NONE` is empty, empty `SUM` is positive zero, and empty `MEAN` is NaN.
- Freeze categorical execution as increasing non-class sample order and increasing class order.
  For every evaluated sample, make a classification/max pass (NaN, positive infinity,
  all-negative-infinity, maximum); make one stable `exp(logit - maximum)` sum and
  `maximum + log(sum)` pass in the selected accumulator; then make the contribution pass. Dense
  computes `q = lse - logit` and adds positive zero when the exact target is zero before it can
  multiply a non-finite `q`, otherwise adds `target * q`. Index first tests optional ignore before
  bounds or a logits read; it then validates `0 <= target < C`, calculates the same stable slice,
  and returns `lse - selectedLogit`. Reduced forms left-associatively add group losses and divide
  once by the typed sample/non-ignored count for `MEAN`.
- These traversals are a local CPU implementation decision, not a new Model promise: Model permits
  conforming reassociation, but generated and direct-oracle evidence must use this exact traversal,
  classification order, primitive arithmetic, narrowing points, stores, cold geometry, and range
  ownership. The direct oracle must not hard-code benchmark extents, sample/class/inner trip
  counts, or otherwise acquire compile-time geometry unavailable to the shape-polymorphic
  generated identity. Target values remain unnormalized caller obligations. The selected algorithm
  preserves the Model NaN, infinity, signed-zero, overflow, underflow, empty-domain, zero-weight,
  and reduction classes.
- Index categorical loss compares an optional exact ignore value before bounds or any logits read.
  A matching ignored sample writes/accumulates positive zero. A non-ignored target must satisfy
  `0 <= target < C`; preparation or binding validates every such target before any output write or
  generated call. `C == 0` is admitted only for an empty domain or an all-ignored bound domain;
  other non-ignored values fail pre-write. A finite `C == 1`, target zero has positive-zero loss.
  A non-ignored NaN/positive-infinity/all-negative-infinity logits slice is NaN; selecting a
  negative-infinity logit with another finite class is positive infinity. `MEAN` divides by the
  non-ignored count and is NaN for empty/all-ignored domains.
- No new cross-backend bitwise promise is introduced. Any generated/oracle deviation requires an
  explicit source-backed reason and a revised task.

### CPU lifecycle, partition, and generated boundary

- Add family-owned CPU-private loss IR, lowering, emitter, pre-write index validator, and
  reference-oracle owners. Thread them through existing capability, partition-DAG decomposition,
  specialization/cache identity, analysis, shared Prepare resource declaration, finalization,
  portable route planning, binding, and prepared executable dispatch.
- Each admitted loss occurrence is one atomic partition-DAG numerical-order barrier. Do not fuse a
  surrounding pointwise/reduction/softmax/log-softmax/one-hot operation and do not recognize a
  decomposed loss topology. No 0008E materialization candidate is offered.
- Capability `true` means CPU preparation can realize the exact occurrence and its static
  representation truthfully. It checks kind/attrs, arity, output slot, descriptor type/shape/
  gradient metadata, static class axis/depth, layouts, output injectivity, range arithmetic, and
  every current promotion/result rule. Compiler gradient ownership remains unchanged: Compiler
  owns loss pullbacks; CPU executes forward occurrences and does not add training/autograd state.
- Preparation declares no workspace, and finalization verifies the same direct family, schema,
  range plan, boundary roles, output publication, and zero-resource declaration. Binding checks
  carrier kind, native order, access mode, alignment, span, logical geometry, and all input/output
  overlap before validation mutation, output writes, or worker submission. Inputs may overlap one
  another; any output/input or output/output overlap is rejected.
- Advance the current-only `CpuGeneratorSchema` from 57. The new identity includes only actual
  emitted code/dataflow facts: loss family, ordered operand/result types, reduction, index width,
  ignore presence, direct carrier assignment, accumulator domain, output/domain range form,
  role-boundary aliasing, and zero-workspace entry shape. Normalized class axis, ranks/extents,
  layouts/strides, addresses, actual ignore bits, slots, and range bounds are cold geometry. Do
  not generate per-axis, per-rank, or per-extent classes; if the current emitter proves a finite
  code-shaping category is necessary, name it and amend the inventory before implementation.
  Schema-57 attention and all older family projections/bytes stay unchanged.
- A generated class is final, field-free, constructor-free, and has one public typed static entry.
  It contains direct typed array/segment access, primitive arithmetic/classification, permitted
  `StrictMath` stable-log operations, index-safe direct loads, and direct stores only. It contains
  no allocation, boxing, reflection, method handles, `invokedynamic`, collections/string dispatch,
  monitor, graph/layout/operation lookup, cache/route/resource/worker selection, reference call,
  fallback call, or Synaptik-owned hot helper.
- A finite fixed-trip body is not authorized for this task. The existing guarded contiguous
  int-address body remains a layout/address proof within one shape-polymorphic artifact, not an
  extent specialization: it must read its trip counts from the cold geometry/range payload. A
  later fixed-shape category requires workload evidence, an explicitly named bounded
  specialization budget, inventory amendment, and equal generated/direct evidence; a benchmark
  fixture alone is not that evidence.

## Out of scope

- New Model/Compiler/Prepare/Runtime/public API, operation kind, gradient rule, training route,
  hidden output, saved intermediate, probability-input loss, label smoothing, weights, masks,
  target broadcasting/casting, dynamic execution, or new loss family.
- Vector/native/OpenBLAS route, materialization/packing, reduction tree, partial-combine workspace,
  autotuning, generic loss abstraction, decomposed graph recognition, external fusion, or
  architecture/module/dependency/build/conformance/integration change.
- CPU 0009 or a 0008J/later detailed task specification.

## Architecture references

`ARCHITECTURE.md` is authoritative: Model owns declarative loss meaning; Compiler owns capture,
inference and gradients; Planning selects partition ownership; CPU analysis/Prepare owns lowering,
exact resource declarations and generated realization; shared Prepare assigns resources; Runtime
invokes only prepared primitive state. Preserve staged analysis-before-assignment/finalization,
atomic partition publication, and generated-code performance discipline. Model 0022/0022A/0022B
define the forward contracts; Compiler 0005D defines current inference and gradient ownership;
CPU 0008B–0008E1, 0008F–0008H provide the partition/resource/generation precedents.

## Architecture constraints

- Preserve Model semantic ownership, Compiler capture/inference/gradient ownership, CPU Prepare
  lowering/resource ownership, and Runtime's prepared-primitive-only hot path.
- Preserve atomic partition-DAG/publication boundaries and staged CPU analysis, shared assignment,
  and CPU finalization. No public/shared API, module dependency, architecture, build, conformance,
  or integration change is authorized.
- The generated oracle rule in `AGENTS.md` is mandatory: generated Class-Files must match the
  frozen clean Java oracle's semantic algorithm, hot-loop/dataflow shape, and avoidable-overhead
  profile.

## Package impact

Existing packages used and changed:

- `io.github.pho001.synaptik.backend.cpu`
- `io.github.pho001.synaptik.backend.cpu.internal.cache`
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit`
- `io.github.pho001.synaptik.backend.cpu.internal.executable`
- `io.github.pho001.synaptik.backend.cpu.internal.ir`
- `io.github.pho001.synaptik.backend.cpu.internal.lowering`
- `io.github.pho001.synaptik.backend.cpu.internal.reference`

Packages added or changed: none; new owners remain in the listed CPU-private packages.

Type placement:

- `...internal.ir.CpuLossIr` — closed code-shaping loss identity.
- `...internal.lowering.CpuLossLowering` — static occurrence and cold-geometry proof.
- `...internal.codegen.emit.CpuLossEmitter` — direct Class-File emission only.
- `...internal.executable.CpuLossInputValidator` — pre-write index/binding validation.
- `...internal.reference.CpuLossReferenceKernel` — test/performance oracle, unreachable from
  production execution.

## Affected files

Expected production/Javadoc paths are at most 30:
`CpuCapabilityProvider`; cache/schema/specialization owners; `CpuClassFileKernelGenerator` plus
new `CpuLossEmitter`; new `CpuLossIr`; `CpuPartitionLowering` plus new `CpuLossLowering`; existing
partition-DAG/recognition/profitability/preparation/finalization/portable-route/prepared-executable
owners; new `CpuLossInputValidator`; and new `CpuLossReferenceKernel`, with only directly affected
package documentation. Expected test/evidence paths are at most 30: existing capability/schema/
partition/preparation/executable controls plus new focused loss IR, lowering, reference, validator,
generated-kernel, structural-evidence, and opt-in performance owners. Documentation/planning is at
most nine paths after the required clean documentation pass.

## Maximum scope

This task may create or modify at most 69 paths: 30 production/Javadoc, 30 test/evidence, and
nine documentation/planning. Generated Class-Files, decompilation, raw benchmark CSV, manifests,
and reports belong under the explicit untracked evidence root and do not count. If another file,
package, route, public type, or module edge is needed, stop and propose a follow-up task.

## Acceptance criteria

1. Capability/lowering accept exactly the matrix above and reject every malformed, dynamic,
   non-static-depth, descriptor/promotion/gradient, layout, carrier, range, or overlap mismatch.
2. Direct generated results match the frozen clean Java oracle for all three reductions, MSE and
   dense's nine ordered floating pairs, index's three logits types times two target widths and
   both ignore states, all legal duplicate roles, arrays and
   segments, legal mixed carriers, dense/strided/broadcast read layouts, injective writes, and
   scalar/caller-parallel `NONE` ranges.
3. Tests prove exact empty, NaN, positive/negative infinity, signed-zero, overflow/underflow,
   promotion/BFLOAT16 narrowing, dense zero-target non-finite exclusion, index ignore-before-read,
   invalid-index pre-write rejection, `C == 0`, all-ignored, and denominator behavior.
4. Reduced forms use one complete ordered range and zero workspace; `NONE` partitions only
   independent complete elements/samples. Publication occurs only after successful invocation.
5. Schema/cache identity contains every code-shaping fact; unchanged family projections and bytes
   remain unchanged; no fallback/decomposition/materialization/helper bridge is selectable.
6. The exact generated inventory is 792 classes and tests enumerate every one: MSE is
   `9 ordered pairs * 3 reductions * 8` distinct three-boundary carrier assignments plus
   `3 equal-type aliasable pairs * 3 reductions * 4` shared-input/output assignments, or 252;
   dense is the same 252; index is
   `3 logits types * 2 index widths * 3 reductions * 2 ignore states * 8` distinct carrier
   assignments, or 288. The total is `252 + 252 + 288 = 792`. The established attention
   role-to-unique-boundary precedent permits the equal-type MSE/dense alias forms; index roles
   remain distinct. Axis/rank/extent/layout/address/ignore-bit/range variation creates no class.
7. Full generated/decompiled and normalized structural evidence covers all 792 classes, validates
   entry/member/instruction/descriptor/call-owner facts, and scans raw Class-Files for forbidden
   helpers, allocation, boxing, reflection, dispatch, and fallback/reference calls. The normalized
   inventory remains an audit aid, not a license to omit a carrier form from timing.
8. With an explicit evidence root and opt-in flag, five fresh Java-26 forks time all 792 exact
   classes in every fork using
   fixed heap, recorded seed/order, five warmups, nine measurements, adaptive at-least-25-ms
   batches, output consumption, no retries/discards, raw CSV/manifests/Class-Files/decompilation,
   and median-of-fork-medians. Each direct method receives the same bound typed carriers, cold
   geometry array, and range as its generated peer, and has the same selected direct loop and
   dataflow; it differs only by ordinary Java source and `javac`. Source and decompilation checks
   prove that neither timed side hides an indirect dispatcher, helper bridge, allocation, boxing,
   reflection, fallback/reference call, or deliberately slower oracle loop. Every row and
   aggregate is `<= 1.15x` generated/direct, consistent with completed CPU 0008F–0008H evidence.
9. The mandatory documentation-focused pass independently finalizes affected Javadocs/docs/
   glossary/planning evidence and records reasoned no-change conclusions where applicable.

Acceptance criterion 8 was not satisfied. The project owner explicitly accepted a validation
exception and closed the remaining CPU 0008I performance forks so implementation can progress.
This decision changes only the task-completion gate: it does not turn the failed old-protocol fork,
the targeted diagnostics, or the unrun corrected full protocol into passing performance evidence.

## Tests / validation

Tier 1: focused CPU capability, IR, lowering, frozen oracle, validator, generated binding,
partition, preparation/finalization, schema/cache, publication and overlap tests. Tier 2: generate
all 792 Class-Files and retain complete `javap -c -v -p`, raw constant-pool/descriptor/instruction/
call-owner forbidden scans, and normalized structural inventory. Tier 2 performance is opt-in only:
use `SYNAPTIK_CPU_LOSS_PERFORMANCE=true`, an explicit evidence-root property, five fresh Java 26
forks with `-Xms1g -Xmx1g`, five warmups, nine measured rounds, recorded randomized generated/
direct order, no retries/discards, adaptive batches of at least 25 ms per side, and output
consumption. Time all 792 classes in each fork.

The representative workload is static dense rank-three `[2, 32, 64]` for MSE/dense (class axis
one for dense), index logits `[2, 32, 64]` with index target `[2, 64]`, and a same-shape valid
alias workload for every equal-type MSE/dense shared-input mode. It is an ordinary bound geometry
fixture, not a code-shaping fixed-trip category: both timed sides receive the same geometry and
range arguments. The dense alias fixture uses valid one-hot target/logit slices, so sharing one
boundary does not evade the target obligation. It has non-empty per-element and per-sample work
for every reduction, valid in-range index values, and both a non-ignored and a recorded
ignore-present input; the actual ignore bits are cold data. Each exact carrier assignment uses its
matching array/segment storage. Focused semantic tests separately cover zero extents,
strides/broadcast reads, invalid indexes, all ignored, and special values. Retain commands,
JDK/OS/CPU/heap metadata, oracle identity, Class-Files, decompilation, raw CSV, 3,960 row/fork
results, per-row median-of-fork-medians, and summaries. Every individual and aggregate ratio is
`<= 1.15x`.

Tier 3: run `./gradlew :backends:cpu:test` once and `./gradlew :backends:cpu:javadoc` after the
documentation pass; then Markdown/link/anchor/fence checks, exact path/status/schema inventory,
tracked-evidence scan, and `git diff --check`. Tier 4: record no-change conformance/integration,
architecture-test, repository-wide-test, and documentation-agent Java-rerun conclusions unless a
real new boundary requires replanning.

## Dependencies

Dependencies are Complete CPU 0008H; Model 0022–0022B; Compiler 0005D; and the existing CPU
0008B–0008E1, 0008F–0008H generated/resource infrastructure.

## Follow-up tasks

- CPU 0008J is the next Draft CPU frontier; its detailed task specification belongs to the next
  separate planning context and is not created here.
- CPU 0009 remains the later Draft generated-coverage checkpoint. Its evidence inventory must
  classify CPU 0008I's corrected full 792-class by five-fork performance evidence as missing and
  rerun it unless then-current evidence makes it stale or otherwise insufficient under CPU 0009's
  explicit classification rules. This transfer does not establish whole-backend parity.

## Architecture impact

None expected. Stop for clarification if implementation needs a public/shared contract, dependency
edge, Runtime interpretation, changed Model semantics, or resource ownership outside this plan.

## Implementation prompt

Use this prompt in a separate clean implementation context:

```text
You are working in the Synaptik repository.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, and this exact CPU 0008I task.
Also inspect Model 0022/0022A/0022B, Compiler 0005D, CPU 0008F–0008H, and directly affected CPU
source/tests. Implement this task exactly as specified; do not implement out-of-scope work.

Stop and report any architecture, scope, oracle, or exact-792-inventory conflict before editing.
Run the specified validation tiers and retain reproducible structural and performance evidence.
After executable validation, hand the final diff and recorded evidence to a separate clean
documentation-focused context. That pass follows documentation-rules.md, finalizes affected
Javadoc/documentation/glossary impact and documentation validation, and does not repeat successful
Java tests unless it changes executable behavior or records a concrete reason.

Update this task's implementation notes, validation evidence, completion summary, and final status
only after that pass. Do not commit or push unless separately authorized.
```

## Local decisions

The benchmark fixture's concrete extents are cold bindings and remain excluded from the
schema-58/792-class identity. The direct oracle must therefore accept the same cold geometry and
range as its generated peer and express the same typed shape-polymorphic loop; this corrects the
comparison rather than weakening it. The current record does not justify a finite fixed-trip or
fixed-shape category: the measured `[2, 32, 64]` fixture is only one workload, and adding such a
category would violate the default cold-extent rule without the master plan's required explicit
evidence and bounded budget. The existing guarded contiguous int-address body is permitted only
as one geometry-guarded path in the same artifact and must not literalize fixture trip counts.

The performance evidence must mechanically inspect both timed entry shapes. Oracle source and
`javap` output must show the same typed carriers plus `long[]` cold geometry and `start`/`end`
range parameters as the generated entry; every logical trip bound, base, stride, axis, and
ignore/range fact must be loaded from those cold inputs rather than a representative-fixture
literal. The review also compares the selected direct loop and its primitive dataflow with the
generated path and rejects either side if it adds a dispatcher, helper bridge, allocation,
boxing, reflection, fallback/reference call, or avoidable per-element indirection. This is a
structural equivalence guard, not permission to make the Java oracle slower.

## Known limitations

Dynamic and unproved representations remain unsupported by this task and fail closed.

The corrected synchronous-C2, 32-interleaved-sub-batch full 792-class by five-fork protocol was
deliberately not run. Loss-family generated/direct near-parity therefore remains unproved across
the complete schema-58 inventory. The retained measurements show moving timing outliers but do not
prove that every outlier is environmental or exclude an undiscovered case-specific performance
defect. CPU 0009 owns classification and rerun of this missing evidence before any whole-backend
performance-parity claim.

## Validation evidence

- Production loss implementation is committed at `4658a58f`. The final diagnostic and harness
  correction changed no production generated code.
- Existing semantic/focused validation passed, and the complete 792-Class-File evidence passed.
  Retained structural evidence under
  `/private/tmp/synaptik-cpu-0008i-correction-20260904/structural` reports all 792 schema-58 classes,
  complete `javap -c -v -p`, and no hidden Synaptik helper/fallback, allocation, boxing,
  reflection, method-handle/`invokedynamic`, collection/string dispatch, monitor, or
  graph/layout/cache/route/resource/worker lookup defect.
- A retained 24-row by five-fork preflight at
  `/private/tmp/synaptik-cpu-0008i-retained-preflight-CEeO9Q` passed all rows; fork maxima were in
  the approximately `1.03x` to `1.06x` range and the overall maximum was `1.060877345937x`.
- One clean old-protocol full fork at
  `/private/tmp/synaptik-cpu-0008i-full-performance-final-clean` completed all 792 rows in about
  1 hour 31 minutes and failed 2 of 792: row 714 measured `1.2680900292x` and row 788 measured
  `1.1972692830x`. It is failure evidence, not a performance pass; forks 2 through 5 were not run.
- A retained targeted 16-row by five-fork diagnostic under
  `/private/tmp/synaptik-cpu-0008i-correction-20260904/baseline` showed those original two rows
  passing while a sibling row moved above the gate at `1.2196841898x`.
- The retained JIT diagnostic at
  `/private/tmp/synaptik-cpu-0008i-jit-diagnostic-20260904` covers 12 forks and 324 measurements
  across targeted/all-loaded and tiered/C2-only modes. It records moving two-sided outliers,
  normal and on-stack-replacement (OSR) C2 compilation of the target methods, no relevant uncommon
  traps, and symmetric inlining/carrier adaptation. It proved no emitter defect.
- The final uncommitted harness correction uses synchronous C2-only child JVMs and 32 alternating
  equal-count generated/direct sub-batches. It adds no retry, discarded interval, threshold
  weakening, oracle slowdown, specialization, or extra timed-loop work. The implementation-owned
  focused non-performance run passed 12 tests with 2 opt-in performance skips, and its
  `git diff --check` passed. This documentation context did not rerun stable Java or performance
  suites because it changed no executable Java or Javadoc.
- The corrected full 792-class by five-fork protocol was deliberately not rerun. By explicit
  project-owner decision, that gate is waived/closed for CPU 0008I only and its evidence is
  missing/deferred to CPU 0009; it did not pass.
- Documentation-focused completion context `01a06d82-d72e-7c93-b762-ef8e43ff1976` reviewed
  `AGENTS.md`, the architecture contract and focused performance-evidence explanation, the planning
  guide, CPU master plan, roadmap, this task, the General/Planning/API-and-Javadoc profiles, the
  final Java test diff, retained evidence, affected Javadocs, and glossary impact. Only planning
  status/evidence documents required edits. Production Javadocs remain accurate because no
  production Java behavior, API, invariant, lifecycle, parameter, return, failure, or ownership
  contract changed. The test-only private harness Javadoc accurately describes its corrected
  measurement behavior. No new or changed reusable project term requires a glossary edit.
- Documentation validation passed: a Ruby local-link/heading-anchor check over the three edited
  planning files, a Ruby heading-uniqueness check, a Ruby Markdown-fence check, `rg` stale-0008I-
  status inspection, an exact `git diff --name-only -- docs` three-path check, and final
  `git diff --check`. No Javadoc command ran because this pass changed no Java/Javadoc.

## Implementation notes

Implementation context `01a05d8b-dbd6-7032-a8e8-bbeae10b3ef3` established the initial
CPU-private loss identity and occurrence-local capability validation. The direct lowering,
binding, emission, validation, structural evidence, and performance protocol remain in progress;
the new capability must not be treated as completed execution support until that lifecycle is
wired and validated.

The subsequent generated-body segment added the CPU-private `CpuLossIr`, cold
`CpuLossLowering.Geometry`, direct `CpuLossEmitter`, frozen test-only reference oracle, and
schema-58 loss dispatch through the existing preparation/finalization/executable carrier path.
It intentionally does not add the separately assigned `CpuLossInputValidator` pre-write
lifecycle invocation or the complete 792-class structural/performance harness. Consequently, at
that implementation checkpoint the task remained `In progress`: the segment was not evidence that
all index validation, complete inventory evidence, or the task-level acceptance criteria were
complete. The required independent
documentation pass reviewed the affected Javadocs. A narrow direct-emission correction also
proves that a non-ignored invalid index throws before the generated body reads logits or writes
the affected output; it is a defensive generated-body property, not a replacement for the still
pending pre-write lifecycle validator. Neither result changes this task's status prematurely.

The first opt-in representative performance attempt reported generated/direct ratios of 1.346x,
1.431x, and 1.530x. Review found that its generated `lossContiguousInt` entry receives mutable
`long[]` geometry and invocation-time ranges, whereas the direct oracle literalized the fixture's
MSE `4096` and categorical `128/32/64` trips. Those are not the same shape-polymorphic
specialized case, so the values cannot support a performance conclusion. This task now requires a
same-cold-geometry ordinary clean-Java oracle and structural source/decompilation evidence before
any further performance gate. It does not authorize benchmark-only branches, a lower threshold,
reduced coverage, asymmetric warmup, an artificially slow oracle, or a fixed-shape category.

Clean planning/documentation review context `/root/cpu_0008i_doc_review` independently inspected
the current oracle, performance harness, loss lowering/IR/emitter, and governing architecture and
planning contracts. It finalized this task-only equivalence correction; the CPU master plan and
roadmap already state the default cold-extent rule and need no change. Markdown headings/fences,
the task's no-link state, and whitespace passed; no Java, Javadoc, glossary, architecture,
dependency, build, conformance, or integration change was made.

Documentation review context `/root/documentation_review` independently reviewed the third narrow
performance-fix iteration: categorical `SUM` removes only its dead included-count local from the
selected and generic generated bodies, while `MEAN` retains its denominator count; the added test
checks the selected F32/INT32 index forms' 26/27-local distinction against the ordinary Java
oracle. This is an internal code-shape correction with no Model, public API, loss semantics,
exception, lifecycle, architecture, or glossary impact. The pass corrected `CpuLossEmitter`
Javadoc for the actual `emit` parameters and clarified that its generated body has no *external*
Synaptik helper call; the CPU master plan and roadmap remain accurate and need no iteration-specific
change. `./gradlew :backends:cpu:javadoc` passed (57 pre-existing unrelated warnings); focused
Markdown link/anchor/fence/whitespace review and `git diff --check` passed. No Java tests were
rerun because this documentation pass did not alter executable behavior.

Documentation review context `/root/cpu_0008i_docs` independently reviewed the later narrow
contiguous-body/oracle-alignment iteration. The index `NONE` contiguous body keeps its direct
independent-sample form, while the categorical cold rank/axis prelude now follows the frozen
ordinary-Java oracle's forward coordinate order. Both remain shape-polymorphic: rank, axis,
extents, bases, strides, and range bounds are invocation geometry, not schema-58 identity or a
fixed-shape specialization. This is an internal generated/direct code-shape alignment; it changes
no Model or public API contract, loss semantics, exception/lifecycle promise, architecture,
glossary term, master-plan status, or roadmap status. The affected binding helper Javadoc now
documents that a failed contiguous proof uses the public generic-affine entry rather than
rejecting a valid invocation. `./gradlew :backends:cpu:javadoc` passed after that edit (57
pre-existing unrelated warnings); Markdown headings/fences, the task's no-link state, and
whitespace/diff review passed. No Java test suite was rerun by this documentation-only pass. At
that documentation checkpoint the task remained `In progress`: no complete 792-class structural
inventory or required five-fork performance evidence had yet been recorded.

Final diagnostic work retained complete semantic and 792-Class-File structural evidence, one
failed old-protocol 792-row fork, focused preflight/diagnostic measurements, and the 12-fork JIT
analysis described above. It then corrected only the private performance harness to synchronous
C2-only children and 32 alternating equal-count sub-batches. The project owner deliberately chose
not to spend the additional time on the corrected full 792-class by five-fork run and explicitly
accepted closure with that validation exception. CPU 0008I is therefore Complete without a
performance-pass claim; CPU 0009 receives the corrected full evidence as missing/deferred.

## Completion summary

- Completed changes: the committed schema-58 portable loss implementation is closed, and the
  final test-only harness correction plus truthful validation exception are recorded without
  changing production generated code.
- Files changed or created by this documentation pass:
  `docs/planning/backends/cpu/tasks/0008i-portable-loss-family-execution.md`,
  `docs/planning/backends/cpu/master-plan.md`, and `docs/planning/roadmap.md`.
- Tests and validation: reused the implementation-owned 12-test focused pass with 2 opt-in skips,
  complete 792-Class-File structural evidence, retained performance/JIT diagnostics, and prior
  `git diff --check`; this pass ran documentation status/link/anchor/fence/path checks and final
  `git diff --check` only.
- Documentation-agent review: clean documentation completion context
  `01a06d82-d72e-7c93-b762-ef8e43ff1976` finalized the task, CPU master-plan, and roadmap evidence.
- Documentation impact: planning status, validation exception, residual risk, and CPU 0009
  evidence transfer are synchronized; architecture and explanatory guides remain unchanged.
- Javadoc review: no production Javadoc change is needed because the latest correction is confined
  to private test-harness measurement behavior; its affected test Javadoc is accurate.
- Glossary impact: none; no reusable project term or existing term boundary changed.
- Unresolved issues: corrected full 792-class by five-fork generated/direct evidence is missing,
  so complete loss-family near-parity and whole-backend performance parity are not established.
- Follow-up required: CPU 0009 must classify this evidence as missing and rerun it when that
  checkpoint reaches the frontier; CPU 0008J is the next separate planning task.

Status: Complete
