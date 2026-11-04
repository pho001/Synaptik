# [CPU](../../../../glossary.md#term-cpu) Task 0008I: Portable [Loss](../../../../glossary.md#loss)-Family Execution

## Reader context

This task records generated MSE, dense-target categorical, and index-target categorical loss
execution, with completion explicitly separated from the waived full performance gate.
`CpuLossLowering` proves static descriptors and cold geometry; `CpuLossIr` identifies emitted forms;
`CpuLossEmitter` produces typed loops. `CpuLossInputValidator` owns pre-write index checks.
`CpuLossReferenceKernel` is an evidence oracle, not Runtime fallback; equivalent timing requires the
same geometry-bound traversal, not fixture-literal trip counts.

Here, each loss is an atomic PartitionDag numerical barrier. `NONE` ranges own independent elements
or samples; `SUM` and `MEAN` use one complete ordered range. Shared Prepare assigns declared buffers
with zero workspace, and Runtime invokes prepared state. Gradient eligibility is descriptor
metadata, not gradient storage or training execution. CPU 0009 receives missing performance
evidence, not a parity claim.

Current-topic navigation (not a replacement for the historical record):

- [current MSE input and reduction roles](../../../../reference/tensor/statistics-losses.md#mean-squared-error-loss-expressions) — Explains prediction versus target, exact-shape requirements, and NONE/SUM/MEAN result boundaries.
- [current dense and index categorical targets](../../../../reference/tensor/statistics-losses.md#categorical-cross-entropy-with-logits-expressions) — Distinguishes target data, class-axis semantics, and typed ignore-index metadata.

## Status

Complete, with the corrected full performance gate waived/closed by project decision.

## Goal

Execute the complete currently represented [Model](../../../../glossary.md#term-model) [loss](../../../../glossary.md#loss) inventory on the portable [CPU](../../../../glossary.md#term-cpu) route: exact-
[shape](../../../../glossary.md#shape) [mean-squared error](../../../../glossary.md#mean-squared-error--mse) ([MSE](../../../../glossary.md#mean-squared-error--mse)), dense-[target](../../../../glossary.md#term-target) categorical cross-entropy with logits, and
index-target categorical cross-entropy with logits. The task makes every currently supported
reduction and index-ignore form executable only when a static, resolved, direct generated
realization can prove the Model/[Compiler](../../../../glossary.md#term-compiler) contract. It preserves the loss [occurrence](../../../../glossary.md#term-occurrence) as one [atomic](../../../../glossary.md#resource-transaction-and-rollback)
numerical unit and publishes its sole requested output exactly once.

## [Scope](../../../../glossary.md#term-scope)

### Source-backed [occurrence](../../../../glossary.md#term-occurrence) [matrix](../../../../glossary.md#term-matrix-meanings)

Admit only fixed two-input/one-output `LossKind` occurrences with their exact attribute class.
`NONE`, `SUM`, and [`MEAN`](../../../../glossary.md#term-mean) are the complete `LossReduction` enum in its current order.

| Family | Ordered inputs | Attributes | Result type and `NONE` [shape](../../../../glossary.md#shape) | Reduction denominator |
|---|---|---|---|---|
| `MEAN_SQUARED_ERROR` | [prediction](../../../../glossary.md#term-prediction), [target](../../../../glossary.md#term-target); both floating and positional-exact Shape | `MeanSquaredErrorAttrs` | floating promotion; prediction Shape | complete logical [element](../../../../glossary.md#term-element) count |
| `DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS` | logits, dense floating target; exact Shape | normalized [class axis](../../../../glossary.md#class-axis), reduction | floating promotion; logits Shape with class axis removed | non-class [sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) count |
| `INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS` | floating logits, exact [INT32](../../../../glossary.md#data-type--datatype)/[INT64](../../../../glossary.md#data-type--datatype) target; target Shape is logits Shape with class axis removed | [normalized axis](../../../../glossary.md#normalized-axis), reduction, optional same-type ignore [scalar](../../../../glossary.md#term-scalar) | logits type; exact target Shape | sample count without ignore; non-ignored count with ignore |

All three families have exactly one output slot. They require fully static Shapes, resolved
[layouts](../../../../glossary.md#layout), non-negative offsets/[strides](../../../../glossary.md#term-stride), [injective](../../../../glossary.md#term-injective) output writes, representable element/address/
span/range arithmetic, and [BFLOAT16](../../../../glossary.md#term-bfloat16)/[FLOAT32](../../../../glossary.md#term-float32)/[FLOAT64](../../../../glossary.md#term-float64) floating operands only. [MSE](../../../../glossary.md#mean-squared-error--mse) and dense use all
nine ordered floating type pairs; their result type follows `DataTypePromotion.promoteFloating`.
Index uses all three logits types, both index widths, and both absent/present ignore forms; the
[index target](../../../../glossary.md#index-target) never promotes or casts. Equal semantic input roles may share a read boundary only
when [descriptors](../../../../glossary.md#term-descriptor)/types make that representation valid; [lowering](../../../../glossary.md#lowering) retains ordered role-to-unique-
boundary mapping.

Direct generated forms are: element-range MSE `NONE`; sample-range dense/index `NONE`; and one
complete reduced-domain range for each `SUM`/`MEAN` form. The latter deliberately has no partial
reduction tree, combine phase, or workspace: its one range owns the complete ordered domain and
its scalar output. `NONE` ranges own complete independent elements/samples and use existing scalar
or caller-parallel range selection. There is zero workspace for every form. Dynamic [dimensions](../../../../glossary.md#dimension),
unresolved/non-injective outputs, malformed [signatures](../../../../glossary.md#term-signature), non-static index class [extent](../../../../glossary.md#term-extent), or an
unproved [carrier](../../../../glossary.md#term-storage-carrier)/[overflow](../../../../glossary.md#term-overflow)/[alias](../../../../glossary.md#aliasing-and-arena) fact [fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation) before generated realization; there is no
[decomposition](../../../../glossary.md#term-decomposition) into public operations, generic interpreter, reference fallback, [materialization](../../../../glossary.md#materialization),
or silent slower path.

### Frozen optimal-clean-Java [oracle](../../../../glossary.md#term-test-oracle) and exceptional-value contract

- Freeze one ordinary optimal clean-Java oracle for each exact generated identity and emit its same
  algorithm, selected hot-loop/dataflow shape, and avoidable-overhead profile. The direct oracle
  has the same typed carrier arguments, cold primitive geometry payload, and <span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [dataflow glossary](../../../../glossary.md#term-dataflow); [shape glossary](../../../../glossary.md#shape); [carrier glossary](../../../../glossary.md#term-storage-carrier); [arguments glossary](../../../../glossary.md#term-argument); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication))</span>`start`/`end` range
  arguments as the generated entry; only Java source compiled by `javac`, rather than direct
  Class-File emission, differs. It is allocation-free during invocation, uses no production helper
  or fallback, and selects its direct typed loop before timing. MSE visits increasing logical
  element order, decodes both operands to the promoted accumulator, computes exactly<span data-glossary-navigation="true"> ([MSE glossary](../../../../glossary.md#mean-squared-error--mse); [element glossary](../../../../glossary.md#term-element); [accumulator glossary](../../../../glossary.md#accumulator-type))</span>
  `difference = prediction - target; loss = difference * difference`, stores `NONE` once, and for
  `SUM`/[`MEAN`](../../../../glossary.md#term-mean) left-associatively accumulates each loss then divides once by the typed logical
  element count supplied through cold geometry. BFLOAT16/FLOAT32 use binary32; FLOAT64 uses
  binary64. Empty <span data-glossary-navigation="true"> ([loss glossary](../../../../glossary.md#loss); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT32 glossary](../../../../glossary.md#term-float32); [binary32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [FLOAT64 glossary](../../../../glossary.md#term-float64); [binary64 glossary](../../../../glossary.md#floating-point-precision-and-rounding))</span>`NONE` is empty, empty `SUM` is [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero), and empty `MEAN` is [NaN](../../../../glossary.md#term-nan).
- Freeze categorical execution as increasing non-class sample order and increasing class order.
  For every evaluated sample, make a classification/max pass (NaN, positive infinity,
  all-negative-infinity, maximum); make one stable <span data-glossary-navigation="true"> ([sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [positive infinity glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [infinity glossary](../../../../glossary.md#term-infinity))</span>`exp(logit - maximum)` sum and
  `maximum + log(sum)` pass in the selected accumulator; then make the contribution pass. Dense
  computes `q = lse - logit` and adds positive zero when the exact target is zero before it can
  multiply a non-finite <span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [finite glossary](../../../../glossary.md#term-finite))</span>`q`, otherwise adds `target * q`. Index first tests optional ignore before
  bounds or a logits read; it then validates `0 <= target < C`, calculates the same stable slice,
  and returns <span data-glossary-navigation="true"> ([slice glossary](../../../../glossary.md#slice))</span>`lse - selectedLogit`. Reduced forms left-associatively add group losses and divide
  once by the typed sample/non-ignored count for `MEAN`.
- These traversals are a local CPU implementation decision, not a new Model promise: Model permits
  conforming reassociation, but generated and direct-oracle evidence must use this exact traversal,
  classification order, primitive arithmetic, narrowing points, stores, cold geometry, and range
  ownership. The direct oracle must not hard-code benchmark extents, sample/class/inner trip
  counts, or otherwise acquire compile-time geometry unavailable to the shape-polymorphic
  generated identity. Target values remain unnormalized caller obligations. The selected algorithm
  preserves the Model NaN, infinity, signed-zero, overflow, underflow, empty-domain, zero-weight,
  and reduction classes.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Model glossary](../../../../glossary.md#term-model); [reassociation glossary](../../../../glossary.md#term-reassociation); [narrowing glossary](../../../../glossary.md#term-narrowing); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [compile glossary](../../../../glossary.md#compile); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [overflow glossary](../../../../glossary.md#term-overflow); [underflow glossary](../../../../glossary.md#term-underflow); [weight glossary](../../../../glossary.md#term-weight))</span>
- Index categorical loss compares an optional exact ignore value before bounds or any logits read.
  A matching ignored sample writes/accumulates positive zero. A non-ignored target must satisfy
  `0 <= target < C`; preparation or binding validates every such target before any output write or
  generated call. <span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>`C == 0` is admitted only for an empty domain or an all-ignored bound domain;
  other non-ignored values fail pre-write. A finite `C == 1`, target zero has positive-zero loss.
  A non-ignored NaN/positive-infinity/all-negative-infinity logits slice is NaN; selecting a
  negative-infinity logit with another finite class is positive infinity. <span data-glossary-navigation="true"> ([logit glossary](../../../../glossary.md#logit))</span>`MEAN` divides by the
  non-ignored count and is NaN for empty/all-ignored domains.
- No new cross-backend bitwise promise is introduced. Any generated/oracle deviation requires an
  explicit source-backed reason and a revised task.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>

### [CPU](../../../../glossary.md#term-cpu) [lifecycle](../../../../glossary.md#lifecycle), [partition](../../../../glossary.md#partition), and generated boundary

- Add family-owned CPU-private loss IR, lowering, emitter, pre-write index validator, and
  reference-oracle owners. Thread them through existing capability, partition-DAG decomposition,
  specialization/cache identity, analysis, shared Prepare resource declaration, finalization,
  portable route planning, binding, and prepared executable dispatch.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([loss glossary](../../../../glossary.md#loss); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [lowering glossary](../../../../glossary.md#lowering); [oracle glossary](../../../../glossary.md#term-test-oracle); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [decomposition glossary](../../../../glossary.md#term-decomposition); [specialization glossary](../../../../glossary.md#term-specialization); [cache glossary](../../../../glossary.md#term-cache); [identity glossary](../../../../glossary.md#term-identity-meanings); [Prepare glossary](../../../../glossary.md#prepare); [planning glossary](../../../../glossary.md#term-planning); [binding glossary](../../../../glossary.md#term-binding); [prepared executable glossary](../../../../glossary.md#prepared-executable--preparedexecutable); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- Each admitted loss occurrence is one atomic partition-DAG numerical-order barrier. Do not fuse a
  surrounding pointwise/reduction/softmax/log-softmax/one-hot operation and do not recognize a
  decomposed loss topology. No 0008E materialization candidate is offered.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [pointwise glossary](../../../../glossary.md#term-pointwise); [softmax glossary](../../../../glossary.md#softmax--log-softmax); [log-softmax glossary](../../../../glossary.md#softmax--log-softmax); [operation glossary](../../../../glossary.md#operation); [topology glossary](../../../../glossary.md#term-topology); [materialization glossary](../../../../glossary.md#materialization))</span>
- Capability `true` means CPU preparation can realize the exact occurrence and its static
  representation truthfully. It checks kind/attrs, arity, output slot, descriptor type/shape/
  gradient metadata, static class axis/depth, layouts, output injectivity, range arithmetic, and
  every current promotion/result rule. Compiler gradient ownership remains unchanged: Compiler
  owns loss pullbacks; CPU executes forward occurrences and does not add training/autograd state.<span data-glossary-navigation="true"> ([arity glossary](../../../../glossary.md#term-arity); [descriptor glossary](../../../../glossary.md#term-descriptor); [shape glossary](../../../../glossary.md#shape); [gradient glossary](../../../../glossary.md#term-gradient); [metadata glossary](../../../../glossary.md#term-metadata); [class axis glossary](../../../../glossary.md#class-axis); [layouts glossary](../../../../glossary.md#layout); [injectivity glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [Compiler glossary](../../../../glossary.md#term-compiler); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [autograd glossary](../../../../glossary.md#autograd))</span>
- Preparation declares no workspace, and finalization verifies the same direct family, schema,
  range plan, boundary roles, output publication, and zero-resource declaration. Binding checks
  carrier kind, native order, access mode, alignment, span, logical geometry, and all input/output
  overlap before validation mutation, output writes, or worker submission. Inputs may overlap one
  another; any output/input or output/output overlap is rejected.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [publication glossary](../../../../glossary.md#term-publication); [carrier glossary](../../../../glossary.md#term-storage-carrier); [alignment glossary](../../../../glossary.md#term-alignment); [validation glossary](../../../../glossary.md#term-validation))</span>
- Advance the current-only `CpuGeneratorSchema` from 57. The new identity includes only actual
  emitted code/dataflow facts: loss family, ordered operand/result types, reduction, index width,
  ignore presence, direct carrier assignment, accumulator domain, output/domain range form,
  role-boundary aliasing, and zero-workspace entry shape. Normalized class axis, ranks/extents,
  layouts/strides, addresses, actual ignore bits, slots, and range bounds are cold geometry. Do
  not generate per-axis, per-rank, or per-extent classes; if the current emitter proves a finite
  code-shaping category is necessary, name it and amend the inventory before implementation.
  Schema-57 attention and all older family projections/bytes stay unchanged.<span data-glossary-navigation="true"> ([dataflow glossary](../../../../glossary.md#term-dataflow); [accumulator glossary](../../../../glossary.md#accumulator-type); [aliasing glossary](../../../../glossary.md#term-aliasing); [strides glossary](../../../../glossary.md#term-stride); [axis glossary](../../../../glossary.md#term-axis); [rank glossary](../../../../glossary.md#term-rank); [extent glossary](../../../../glossary.md#term-extent); [finite glossary](../../../../glossary.md#term-finite))</span>
- A generated class is final, field-free, constructor-free, and has one public typed static entry.
  It contains direct typed array/segment access, primitive arithmetic/classification, permitted
  `StrictMath` stable-log operations, index-safe direct loads, and direct stores only. It contains
  no allocation, boxing, reflection, method handles, <span data-glossary-navigation="true"> ([log glossary](../../../../glossary.md#term-log))</span>`invokedynamic`, collections/string dispatch,
  monitor, graph/layout/operation lookup, cache/route/resource/worker selection, reference call,
  fallback call, or Synaptik-owned hot helper.<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-method-handle-lookup))</span><span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph))</span>
- A finite fixed-trip body is not authorized for this task. The existing guarded contiguous
  int-address body remains a layout/address proof within one shape-polymorphic artifact, not an
  extent specialization: it must read its trip counts from the cold geometry/range payload. A
  later fixed-shape category requires workload evidence, an explicitly named bounded
  specialization budget, inventory amendment, and equal generated/direct evidence; a benchmark
  fixture alone is not that evidence.<span data-glossary-navigation="true"> ([contiguous glossary](../../../../glossary.md#term-contiguous); [artifact glossary](../../../../glossary.md#term-artifact); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [fixture glossary](../../../../glossary.md#term-fixture))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- New Model/Compiler/Prepare/Runtime/public API, operation kind, gradient rule, training route,
  hidden output, saved intermediate, probability-input loss, label smoothing, weights, masks,
  target broadcasting/casting, dynamic execution, or new loss family.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [operation glossary](../../../../glossary.md#operation); [gradient glossary](../../../../glossary.md#term-gradient); [loss glossary](../../../../glossary.md#loss); [weights glossary](../../../../glossary.md#term-weight); [masks glossary](../../../../glossary.md#term-boolean-mask); [target glossary](../../../../glossary.md#term-target); [broadcasting glossary](../../../../glossary.md#broadcasting))</span>
- Vector/native/OpenBLAS route, materialization/packing, reduction tree, partial-combine workspace,
  autotuning, generic loss abstraction, decomposed graph recognition, external fusion, or
  architecture/module/dependency/build/conformance/integration change.<span data-glossary-navigation="true"> ([Vector glossary](../../../../glossary.md#term-vector); [OpenBLAS glossary](../../../../glossary.md#term-openblas); [materialization glossary](../../../../glossary.md#materialization); [autotuning glossary](../../../../glossary.md#autotuning--model-autotuning); [graph glossary](../../../../glossary.md#graph); [fusion glossary](../../../../glossary.md#term-fusion); [module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- [CPU](../../../../glossary.md#term-cpu) 0009 or a 0008J/later detailed task specification.

## Architecture references

`ARCHITECTURE.md` is authoritative: [Model](../../../../glossary.md#term-model) owns declarative [loss](../../../../glossary.md#loss) meaning; [Compiler](../../../../glossary.md#term-compiler) owns [capture](../../../../glossary.md#term-capture-meanings),
[inference](../../../../glossary.md#term-inference) and [gradients](../../../../glossary.md#term-gradient); [Planning](../../../../glossary.md#term-planning) selects [partition](../../../../glossary.md#partition) [ownership](../../../../glossary.md#term-ownership-meanings); [CPU](../../../../glossary.md#term-cpu) analysis/[Prepare](../../../../glossary.md#prepare) owns [lowering](../../../../glossary.md#lowering),
exact resource declarations and generated realization; shared Prepare assigns resources; [Runtime](../../../../glossary.md#term-runtime)
invokes only prepared primitive state. Preserve staged analysis-before-assignment/finalization,
[atomic](../../../../glossary.md#resource-transaction-and-rollback) partition [publication](../../../../glossary.md#term-publication), and generated-code performance discipline. Model 0022/0022A/0022B
define the forward contracts; Compiler 0005D defines current inference and gradient ownership;
CPU 0008B–0008E1, 0008F–0008H provide the partition/resource/generation precedents.

## Architecture constraints

- Preserve Model semantic ownership, Compiler capture/inference/gradient ownership, CPU Prepare
  lowering/resource ownership, and Runtime's prepared-primitive-only hot path.<span data-glossary-navigation="true"> ([resource ownership glossary](../../../../glossary.md#term-resource-ownership))</span><span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [ownership glossary](../../../../glossary.md#term-ownership); [Compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [inference glossary](../../../../glossary.md#term-inference); [gradient glossary](../../../../glossary.md#term-gradient); [CPU glossary](../../../../glossary.md#term-cpu); [Prepare glossary](../../../../glossary.md#prepare); [lowering glossary](../../../../glossary.md#lowering); [Runtime glossary](../../../../glossary.md#term-runtime); [hot path glossary](../../../../glossary.md#term-hot-path))</span>
- Preserve atomic partition-DAG/publication boundaries and staged CPU analysis, shared assignment,
  and CPU finalization. No public/shared API, module dependency, architecture, build, conformance,
  or integration change is authorized.<span data-glossary-navigation="true"> ([atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [partition glossary](../../../../glossary.md#partition); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [publication glossary](../../../../glossary.md#term-publication); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- The generated [oracle](../../../../glossary.md#term-test-oracle) rule in `AGENTS.md` is mandatory: generated Class-Files must match the
  frozen clean Java oracle's semantic algorithm, hot-loop/dataflow shape, and avoidable-overhead
  profile.<span data-glossary-navigation="true"> ([dataflow glossary](../../../../glossary.md#term-dataflow); [shape glossary](../../../../glossary.md#shape))</span>

## Package impact

Existing packages used and changed:

- `io.github.pho001.synaptik.backend.cpu`
- `io.github.pho001.synaptik.backend.cpu.internal.cache`
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit`
- `io.github.pho001.synaptik.backend.cpu.internal.executable`
- `io.github.pho001.synaptik.backend.cpu.internal.ir`
- `io.github.pho001.synaptik.backend.cpu.internal.lowering`
- `io.github.pho001.synaptik.backend.cpu.internal.reference`

Packages added or changed: none; new [owners](../../../../glossary.md#term-owner) remain in the listed [CPU](../../../../glossary.md#term-cpu)-private packages.

Type placement:

- `...internal.ir.CpuLossIr` — closed code-shaping [loss](../../../../glossary.md#loss) [identity](../../../../glossary.md#term-identity-meanings).
- `...internal.lowering.CpuLossLowering` — static [occurrence](../../../../glossary.md#term-occurrence) and cold-geometry proof.
- `...internal.codegen.emit.CpuLossEmitter` — direct Class-File emission only.
- `...internal.executable.CpuLossInputValidator` — pre-write index/[binding](../../../../glossary.md#term-binding) [validation](../../../../glossary.md#term-validation).
- `...internal.reference.CpuLossReferenceKernel` — test/performance oracle, unreachable from
  production execution.<span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle))</span>

## Affected files

Expected production/[Javadoc](../../../../glossary.md#term-javadoc) paths are at most 30:
`CpuCapabilityProvider`; [cache](../../../../glossary.md#term-cache)/[schema](../../../../glossary.md#term-schema)/[specialization](../../../../glossary.md#term-specialization) [owners](../../../../glossary.md#term-owner); `CpuClassFileKernelGenerator` plus
new `CpuLossEmitter`; new `CpuLossIr`; `CpuPartitionLowering` plus new `CpuLossLowering`; existing
[partition](../../../../glossary.md#partition)-[DAG](../../../../glossary.md#term-directed-acyclic-graph-dag)/recognition/profitability/preparation/finalization/portable-route/prepared-executable
owners; new `CpuLossInputValidator`; and new `CpuLossReferenceKernel`, with only directly affected
package documentation. Expected test/evidence paths are at most 30: existing capability/schema/
partition/preparation/executable controls plus new focused [loss](../../../../glossary.md#loss) [IR](../../../../glossary.md#term-intermediate-representation-ir), [lowering](../../../../glossary.md#lowering), reference, validator,
generated-[kernel](../../../../glossary.md#kernel), structural-evidence, and opt-in performance owners. Documentation/[planning](../../../../glossary.md#term-planning) is at
most nine paths after the required clean documentation pass.

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most 69 paths: 30 production/[Javadoc](../../../../glossary.md#term-javadoc), 30 test/evidence, and
nine documentation/[planning](../../../../glossary.md#term-planning). Generated Class-Files, [decompilation](../../../../glossary.md#term-decompilation), raw benchmark CSV, manifests,
and reports belong under the explicit untracked evidence root and do not count. If another file,
package, route, public type, or [module](../../../../glossary.md#term-module) edge is needed, stop and propose a follow-up task.

## Acceptance criteria

1. Capability/lowering accept exactly the matrix above and reject every malformed, dynamic,
   non-static-depth, descriptor/promotion/gradient, layout, carrier, range, or overlap mismatch.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [descriptor glossary](../../../../glossary.md#term-descriptor); [gradient glossary](../../../../glossary.md#term-gradient); [layout glossary](../../../../glossary.md#layout))</span>
2. Direct generated results match the frozen clean Java oracle for all three reductions, MSE and
   dense's nine ordered floating pairs, index's three logits types times two target widths and
   both ignore states, all legal duplicate roles, arrays and
   segments, legal mixed carriers, dense/strided/broadcast read layouts, injective writes, and
   scalar/caller-parallel <span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle); [target glossary](../../../../glossary.md#term-target); [injective glossary](../../../../glossary.md#term-injective); [scalar glossary](../../../../glossary.md#term-scalar))</span>`NONE` ranges.
3. Tests prove exact empty, NaN, positive/negative infinity, signed-zero, overflow/underflow,
   promotion/BFLOAT16 narrowing, dense zero-target non-finite exclusion, index ignore-before-read,
   invalid-index pre-write rejection, <span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [negative infinity glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [overflow glossary](../../../../glossary.md#term-overflow); [underflow glossary](../../../../glossary.md#term-underflow); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [narrowing glossary](../../../../glossary.md#term-narrowing); [finite glossary](../../../../glossary.md#term-finite))</span>`C == 0`, all-ignored, and denominator behavior.
4. Reduced forms use one complete ordered range and zero workspace; `NONE` partitions only
   independent complete elements/samples. Publication occurs only after successful invocation.<span data-glossary-navigation="true"> ([elements glossary](../../../../glossary.md#term-element); [samples glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [Publication glossary](../../../../glossary.md#term-publication))</span>
5. Schema/cache identity contains every code-shaping fact; unchanged family projections and bytes
   remain unchanged; no fallback/decomposition/materialization/helper bridge is selectable.<span data-glossary-navigation="true"> ([Schema glossary](../../../../glossary.md#term-schema); [cache glossary](../../../../glossary.md#term-cache); [identity glossary](../../../../glossary.md#term-identity-meanings); [decomposition glossary](../../../../glossary.md#term-decomposition); [materialization glossary](../../../../glossary.md#materialization))</span>
6. The exact generated inventory is 792 classes and tests enumerate every one: [MSE](../../../../glossary.md#mean-squared-error--mse) is
   `9 ordered pairs * 3 reductions * 8` distinct three-boundary [carrier](../../../../glossary.md#term-storage-carrier) assignments plus
   `3 equal-type aliasable pairs * 3 reductions * 4` shared-input/output assignments, or 252;
   dense is the same 252; index is
   `3 logits types * 2 index widths * 3 reductions * 2 ignore states * 8` distinct carrier
   assignments, or 288. The total is `252 + 252 + 288 = 792`. The established attention
   role-to-unique-boundary precedent permits the equal-type MSE/dense alias forms; index roles
   remain distinct. Axis/rank/extent/layout/address/ignore-bit/range variation creates no class.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena); [Axis glossary](../../../../glossary.md#term-axis); [rank glossary](../../../../glossary.md#term-rank); [extent glossary](../../../../glossary.md#term-extent))</span>
7. Full generated/decompiled and normalized structural evidence covers all 792 classes, validates
   entry/member/instruction/descriptor/call-owner facts, and scans raw Class-Files for forbidden
   helpers, allocation, boxing, reflection, dispatch, and fallback/reference calls. The normalized
   inventory remains an audit aid, not a license to omit a carrier form from timing.<span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch))</span>
8. With an explicit evidence root and opt-in flag, five fresh Java-26 forks time all 792 exact
   classes in every fork using
   fixed heap, recorded seed/order, five warmups, nine measurements, adaptive at-least-25-ms
   batches, output consumption, no retries/discards, raw CSV/manifests/Class-Files/decompilation,
   and median-of-fork-medians. Each direct method receives the same bound typed carriers, cold
   geometry array, and range as its generated peer, and has the same selected direct loop and
   dataflow; it differs only by ordinary Java source and <span data-glossary-navigation="true"> ([heap glossary](../../../../glossary.md#term-heap); [seed glossary](../../../../glossary.md#term-seed); [batches glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [decompilation glossary](../../../../glossary.md#term-decompilation); [median glossary](../../../../glossary.md#term-median); [dataflow glossary](../../../../glossary.md#term-dataflow))</span>`javac`. Source and decompilation checks
   prove that neither timed side hides an indirect dispatcher, helper bridge, allocation, boxing,
   reflection, fallback/reference call, or deliberately slower oracle loop. Every row and
   aggregate is `<= 1.15x` generated/direct, consistent with completed [CPU](../../../../glossary.md#term-cpu) 0008F–0008H evidence.
9. The mandatory documentation-focused pass independently finalizes affected Javadocs/docs/
   glossary/planning evidence and records reasoned no-change conclusions where applicable.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>

Acceptance criterion 8 was not satisfied. The project [owner](../../../../glossary.md#term-owner) explicitly accepted a [validation](../../../../glossary.md#term-validation)
exception and closed the remaining CPU 0008I performance [forks](../../../../glossary.md#term-fork-meanings) so implementation can progress.
This decision changes only the task-completion gate: it does not turn the failed old-protocol [fork](../../../../glossary.md#term-fork-meanings),
the targeted diagnostics, or the unrun corrected full protocol into passing performance evidence.

## Tests / [validation](../../../../glossary.md#term-validation)

Tier 1: focused [CPU](../../../../glossary.md#term-cpu) capability, [IR](../../../../glossary.md#term-intermediate-representation-ir), [lowering](../../../../glossary.md#lowering), frozen [oracle](../../../../glossary.md#term-test-oracle), validator, generated [binding](../../../../glossary.md#term-binding),
[partition](../../../../glossary.md#partition), preparation/finalization, [schema](../../../../glossary.md#term-schema)/[cache](../../../../glossary.md#term-cache), [publication](../../../../glossary.md#term-publication) and overlap tests. Tier 2: generate
all 792 Class-Files and retain complete `javap -c -v -p`, raw [constant](../../../../glossary.md#term-constant)-pool/[descriptor](../../../../glossary.md#term-descriptor)/instruction/
call-[owner](../../../../glossary.md#term-owner) forbidden scans, and normalized structural inventory. Tier 2 performance is opt-in only:
use `SYNAPTIK_CPU_LOSS_PERFORMANCE=true`, an explicit evidence-root property, five fresh Java 26
[forks](../../../../glossary.md#term-fork-meanings) with `-Xms1g -Xmx1g`, five warmups, nine measured rounds, recorded randomized generated/
direct order, no retries/discards, adaptive [batches](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) of at least 25 ms per side, and output
consumption. Time all 792 classes in each [fork](../../../../glossary.md#term-fork-meanings).

The representative workload is static dense [rank](../../../../glossary.md#term-rank)-three `[2, 32, 64]` for [MSE](../../../../glossary.md#mean-squared-error--mse)/dense ([class axis](../../../../glossary.md#class-axis)
one for dense), index logits `[2, 32, 64]` with [index target](../../../../glossary.md#index-target) `[2, 64]`, and a same-[shape](../../../../glossary.md#shape) valid
[alias](../../../../glossary.md#aliasing-and-arena) workload for every equal-type MSE/dense shared-input mode. It is an ordinary bound geometry
[fixture](../../../../glossary.md#term-fixture), not a code-shaping fixed-trip category: both timed sides receive the same geometry and
range [arguments](../../../../glossary.md#term-argument). The dense alias fixture uses valid one-hot [target](../../../../glossary.md#term-target)/[logit](../../../../glossary.md#logit) slices, so sharing one
boundary does not evade the target obligation. It has non-empty per-[element](../../../../glossary.md#term-element) and per-[sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) work
for every reduction, valid in-range index values, and both a non-ignored and a recorded
ignore-present input; the actual ignore bits are cold data. Each exact [carrier](../../../../glossary.md#term-storage-carrier) assignment uses its
matching array/segment storage. Focused semantic tests separately cover zero extents,
[strides](../../../../glossary.md#term-stride)/broadcast reads, invalid indexes, all ignored, and special values. Retain commands,
[JDK](../../../../glossary.md#term-java-development-kit-jdk)/OS/CPU/[heap](../../../../glossary.md#term-heap) [metadata](../../../../glossary.md#term-metadata), oracle [identity](../../../../glossary.md#term-identity-meanings), Class-Files, [decompilation](../../../../glossary.md#term-decompilation), raw CSV, 3,960 row/fork
results, per-row [median](../../../../glossary.md#term-median)-of-fork-medians, and summaries. Every individual and aggregate ratio is
`<= 1.15x`.

Tier 3: [run](../../../../glossary.md#term-run) `./gradlew :backends:cpu:test` once and `./gradlew :backends:cpu:javadoc` after the
documentation pass; then Markdown/link/anchor/fence checks, exact path/status/schema inventory,
tracked-evidence [scan](../../../../glossary.md#term-cumulative-scan), and `git diff --check`. Tier 4: record no-change [conformance](../../../../glossary.md#term-conformance)/integration,
architecture-test, repository-wide-test, and documentation-agent Java-rerun conclusions unless a
real new boundary requires replanning.

## Dependencies

Dependencies are Complete [CPU](../../../../glossary.md#term-cpu) 0008H; [Model](../../../../glossary.md#term-model) 0022–0022B; [Compiler](../../../../glossary.md#term-compiler) 0005D; and the existing CPU
0008B–0008E1, 0008F–0008H generated/resource infrastructure.

## Follow-up tasks

- CPU 0008J is the next Draft CPU frontier; its detailed task specification belongs to the next
  separate planning context and is not created here.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [planning glossary](../../../../glossary.md#term-planning))</span>
- CPU 0009 remains the later Draft generated-coverage checkpoint. Its evidence inventory must
  classify CPU 0008I's corrected full 792-class by five-fork performance evidence as missing and
  rerun it unless then-current evidence makes it stale or otherwise insufficient under CPU 0009's
  explicit classification rules. This transfer does not establish whole-backend parity.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint); [fork glossary](../../../../glossary.md#term-fork-meanings); [backend glossary](../../../../glossary.md#backend))</span>

## Architecture impact

None expected. Stop for clarification if implementation needs a public/shared contract, [dependency
edge](../../../../glossary.md#term-dependency-edge), [Runtime](../../../../glossary.md#term-runtime) interpretation, changed [Model](../../../../glossary.md#term-model) semantics, or resource [ownership](../../../../glossary.md#term-ownership) outside this plan.

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

The benchmark [fixture](../../../../glossary.md#term-fixture)'s concrete extents are cold [bindings](../../../../glossary.md#term-binding) and remain excluded from the
[schema](../../../../glossary.md#term-schema)-58/792-class [identity](../../../../glossary.md#term-identity-meanings). The direct [oracle](../../../../glossary.md#term-test-oracle) must therefore accept the same cold geometry and
range as its generated peer and express the same typed [shape](../../../../glossary.md#shape)-polymorphic loop; this corrects the
comparison rather than weakening it. The current record does not justify a [finite](../../../../glossary.md#term-finite) fixed-trip or
fixed-shape category: the measured `[2, 32, 64]` fixture is only one workload, and adding such a
category would violate the default cold-[extent](../../../../glossary.md#term-extent) rule without the master plan's required explicit
evidence and bounded budget. The existing guarded [contiguous](../../../../glossary.md#term-contiguous) int-address body is permitted only
as one geometry-guarded path in the same [artifact](../../../../glossary.md#term-artifact) and must not literalize fixture trip counts.

The performance evidence must mechanically inspect both timed entry shapes. Oracle source and
`javap` output must show the same typed carriers plus `long[]` cold geometry and `start`/`end`
range [parameters](../../../../glossary.md#term-parameter) as the generated entry; every logical trip bound, base, [stride](../../../../glossary.md#term-stride), [axis](../../../../glossary.md#term-axis), and
ignore/range fact must be loaded from those cold inputs rather than a representative-fixture
literal. The review also compares the selected direct loop and its primitive [dataflow](../../../../glossary.md#term-dataflow) with the
generated path and rejects either side if it adds a dispatcher, helper bridge, allocation,
boxing, reflection, fallback/reference call, or avoidable per-[element](../../../../glossary.md#term-element) indirection. This is a
structural equivalence [guard](../../../../glossary.md#term-guard), not permission to make the Java oracle slower.

## Known limitations

Dynamic and unproved representations remain unsupported by this task and [fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation).

The corrected [synchronous](../../../../glossary.md#term-synchronous)-[C2](../../../../glossary.md#term-c2), 32-interleaved-sub-[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) full 792-class by five-[fork](../../../../glossary.md#term-fork-meanings) protocol was
deliberately not [run](../../../../glossary.md#term-run). [Loss](../../../../glossary.md#loss)-family generated/direct near-parity therefore remains unproved across
the complete [schema](../../../../glossary.md#term-schema)-58 inventory. The retained measurements show moving timing outliers but do not
prove that every [outlier](../../../../glossary.md#term-outlier) is environmental or exclude an undiscovered case-specific performance
defect. [CPU](../../../../glossary.md#term-cpu) 0009 owns classification and rerun of this missing evidence before any whole-[backend](../../../../glossary.md#backend)
performance-parity claim.

## [Validation](../../../../glossary.md#term-validation) evidence

- Production [loss](../../../../glossary.md#loss) implementation is committed at `4658a58f`. The final diagnostic and harness
  correction changed no production generated code.
- Existing semantic/focused validation passed, and the complete 792-Class-File evidence passed.
  Retained structural evidence under
  `/private/tmp/synaptik-cpu-0008i-correction-20260904/structural` reports all 792 schema-58 classes,
  complete <span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema))</span>`javap -c -v -p`, and no hidden Synaptik helper/fallback, allocation, boxing,
  reflection, method-handle/`invokedynamic`, collection/string dispatch, monitor, or
  graph/layout/cache/route/resource/worker lookup defect.<span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch); [graph glossary](../../../../glossary.md#graph); [layout glossary](../../../../glossary.md#layout); [cache glossary](../../../../glossary.md#term-cache); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- A retained 24-row by five-[fork](../../../../glossary.md#term-fork-meanings) [preflight](../../../../glossary.md#term-preflight) at
  `/private/tmp/synaptik-cpu-0008i-retained-preflight-CEeO9Q` passed all rows; fork maxima were in
  the approximately `1.03x` to `1.06x` range and the overall maximum was `1.060877345937x`.
- One clean old-protocol full fork at
  `/private/tmp/synaptik-cpu-0008i-full-performance-final-clean` completed all 792 rows in about
  1 hour 31 minutes and failed 2 of 792: row 714 measured `1.2680900292x` and row 788 measured
  `1.1972692830x`. It is failure evidence, not a performance pass; forks 2 through 5 were not [run](../../../../glossary.md#term-run).
- A retained targeted 16-row by five-fork diagnostic under
  `/private/tmp/synaptik-cpu-0008i-correction-20260904/baseline` showed those original two rows
  passing while a sibling row moved above the gate at `1.2196841898x`.
- The retained [JIT](../../../../glossary.md#term-just-in-time-compilation-jit) diagnostic at
  `/private/tmp/synaptik-cpu-0008i-jit-diagnostic-20260904` covers 12 forks and 324 measurements
  across targeted/all-loaded and tiered/C2-only modes. It records moving two-sided outliers,
  normal and on-stack-replacement (OSR) C2 compilation of the target methods, no relevant uncommon
  traps, and symmetric inlining/carrier adaptation. It proved no emitter defect.<span data-glossary-navigation="true"> ([C2 glossary](../../../../glossary.md#term-c2); [OSR glossary](../../../../glossary.md#term-on-stack-replacement-osr); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [target glossary](../../../../glossary.md#term-target); [inlining glossary](../../../../glossary.md#term-inlining); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- The final uncommitted harness correction uses synchronous C2-only child JVMs and 32 alternating
  equal-count generated/direct sub-batches. It adds no retry, discarded interval, threshold
  weakening, oracle slowdown, specialization, or extra timed-loop work. The implementation-owned
  focused non-performance run passed 12 tests with 2 opt-in performance skips, and its<span data-glossary-navigation="true"> ([synchronous glossary](../../../../glossary.md#term-synchronous); [batches glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [oracle glossary](../../../../glossary.md#term-test-oracle); [specialization glossary](../../../../glossary.md#term-specialization))</span>
  `git diff --check` passed. This documentation context did not rerun stable Java or performance
  suites because it changed no executable Java or Javadoc.
- The corrected full 792-class by five-fork protocol was deliberately not rerun. By explicit
  project-owner decision, that gate is waived/closed for CPU 0008I only and its evidence is
  missing/deferred to CPU 0009; it did not pass.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Documentation-focused completion context `01a06d82-d72e-7c93-b762-ef8e43ff1976` reviewed
  `AGENTS.md`, the architecture contract and focused performance-evidence explanation, the planning
  guide, CPU master plan, roadmap, this task, the General/Planning/API-and-Javadoc profiles, the
  final Java test diff, retained evidence, affected Javadocs, and glossary impact. Only planning
  status/evidence documents required edits. Production Javadocs remain accurate because no
  production Java behavior, API, invariant, lifecycle, parameter, return, failure, or ownership
  contract changed. The test-only private harness Javadoc accurately describes its corrected
  measurement behavior. No new or changed reusable project term requires a glossary edit.<span data-glossary-navigation="true"> ([architecture contract glossary](../../../../glossary.md#architecture-contract); [planning glossary](../../../../glossary.md#term-planning); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [invariant glossary](../../../../glossary.md#term-invariant); [lifecycle glossary](../../../../glossary.md#lifecycle); [parameter glossary](../../../../glossary.md#term-parameter); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Documentation validation passed: a Ruby local-link/heading-anchor check over the three edited
  planning files, a Ruby heading-uniqueness check, a Ruby Markdown-fence check, `rg` stale-0008I-
  status inspection, an exact `git diff --name-only -- docs` three-path check, and final
  `git diff --check`. No [Javadoc](../../../../glossary.md#term-javadoc) command ran because this pass changed no Java/Javadoc.

## Implementation notes

Implementation context `01a05d8b-dbd6-7032-a8e8-bbeae10b3ef3` established the initial
[CPU](../../../../glossary.md#term-cpu)-private [loss](../../../../glossary.md#loss) [identity](../../../../glossary.md#term-identity-meanings) and [occurrence](../../../../glossary.md#term-occurrence)-local capability [validation](../../../../glossary.md#term-validation). The direct [lowering](../../../../glossary.md#lowering),
[binding](../../../../glossary.md#term-binding), emission, validation, structural evidence, and performance protocol remain in progress;
the new capability must not be treated as completed execution support until that [lifecycle](../../../../glossary.md#lifecycle) is
wired and validated.

The subsequent generated-body segment added the CPU-private `CpuLossIr`, cold
`CpuLossLowering.Geometry`, direct `CpuLossEmitter`, frozen test-only reference [oracle](../../../../glossary.md#term-test-oracle), and
[schema](../../../../glossary.md#term-schema)-58 loss [dispatch](../../../../glossary.md#term-dispatch) through the existing preparation/finalization/executable [carrier](../../../../glossary.md#term-storage-carrier) path.
It intentionally does not add the separately assigned `CpuLossInputValidator` pre-write
lifecycle invocation or the complete 792-class structural/performance harness. Consequently, at
that implementation [checkpoint](../../../../glossary.md#term-checkpoint) the task remained `In progress`: the segment was not evidence that
all index validation, complete inventory evidence, or the task-level acceptance criteria were
complete. The required independent
documentation pass reviewed the affected Javadocs. A narrow direct-emission correction also
proves that a non-ignored invalid index throws before the generated body reads logits or writes
the affected output; it is a defensive generated-body property, not a replacement for the still
pending pre-write lifecycle validator. Neither result changes this task's status prematurely.

The first opt-in representative performance attempt reported generated/direct ratios of 1.346x,
1.431x, and 1.530x. Review found that its generated `lossContiguousInt` entry receives [mutable](../../../../glossary.md#term-mutable)
`long[]` geometry and invocation-time ranges, whereas the direct oracle literalized the [fixture](../../../../glossary.md#term-fixture)'s
[MSE](../../../../glossary.md#mean-squared-error--mse) `4096` and categorical `128/32/64` trips. Those are not the same [shape](../../../../glossary.md#shape)-polymorphic
specialized case, so the values cannot support a performance conclusion. This task now requires a
same-cold-geometry ordinary clean-Java oracle and structural source/[decompilation](../../../../glossary.md#term-decompilation) evidence before
any further performance gate. It does not authorize benchmark-only branches, a lower threshold,
reduced coverage, asymmetric [warmup](../../../../glossary.md#term-warmup), an artificially slow oracle, or a fixed-shape category.

Clean [planning](../../../../glossary.md#term-planning)/documentation review context `/root/cpu_0008i_doc_review` independently inspected
the current oracle, performance harness, loss lowering/[IR](../../../../glossary.md#term-intermediate-representation-ir)/emitter, and governing architecture and
planning contracts. It finalized this task-only equivalence correction; the CPU master plan and
roadmap already state the default cold-[extent](../../../../glossary.md#term-extent) rule and need no change. Markdown headings/fences,
the task's no-link state, and whitespace passed; no Java, [Javadoc](../../../../glossary.md#term-javadoc), glossary, architecture,
[dependency](../../../../glossary.md#term-dependency), build, [conformance](../../../../glossary.md#term-conformance), or integration change was made.

Documentation review context `/root/documentation_review` independently reviewed the third narrow
performance-fix iteration: categorical `SUM` removes only its dead included-count local from the
selected and generic generated bodies, while [`MEAN`](../../../../glossary.md#term-mean) retains its denominator count; the added test
checks the selected [F32](../../../../glossary.md#floating-point-precision-and-rounding)/[INT32](../../../../glossary.md#data-type--datatype) index forms' 26/27-local distinction against the ordinary Java
oracle. This is an internal code-shape correction with no [Model](../../../../glossary.md#term-model), public [API](../../../../glossary.md#term-application-programming-interface-api), loss semantics,
exception, lifecycle, architecture, or glossary impact. The pass corrected `CpuLossEmitter`
Javadoc for the actual `emit` [parameters](../../../../glossary.md#term-parameter) and clarified that its generated body has no *external*
Synaptik helper call; the CPU master plan and roadmap remain accurate and need no iteration-specific
change. `./gradlew :backends:cpu:javadoc` passed (57 pre-existing unrelated warnings); focused
Markdown link/anchor/fence/whitespace review and `git diff --check` passed. No Java tests were
rerun because this documentation pass did not alter executable behavior.

Documentation review context `/root/cpu_0008i_docs` independently reviewed the later narrow
[contiguous](../../../../glossary.md#term-contiguous)-body/oracle-[alignment](../../../../glossary.md#term-alignment) iteration. The index `NONE` contiguous body keeps its direct
independent-[sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) form, while the categorical cold [rank](../../../../glossary.md#term-rank)/[axis](../../../../glossary.md#term-axis) prelude now follows the frozen
ordinary-Java oracle's forward [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) order. Both remain shape-polymorphic: rank, axis,
extents, bases, [strides](../../../../glossary.md#term-stride), and range bounds are invocation geometry, not schema-58 identity or a
fixed-shape [specialization](../../../../glossary.md#term-specialization). This is an internal generated/direct code-shape alignment; it changes
no Model or public API contract, loss semantics, exception/lifecycle promise, architecture,
glossary term, master-plan status, or roadmap status. The affected binding helper Javadoc now
documents that a failed contiguous proof uses the public generic-affine entry rather than
rejecting a valid invocation. `./gradlew :backends:cpu:javadoc` passed after that edit (57
pre-existing unrelated warnings); Markdown headings/fences, the task's no-link state, and
whitespace/diff review passed. No Java test suite was rerun by this documentation-only pass. At
that documentation checkpoint the task remained `In progress`: no complete 792-class structural
inventory or required five-[fork](../../../../glossary.md#term-fork-meanings) performance evidence had yet been recorded.

Final diagnostic work retained complete semantic and 792-Class-File structural evidence, one
failed old-protocol 792-row fork, focused [preflight](../../../../glossary.md#term-preflight)/diagnostic measurements, and the 12-fork [JIT](../../../../glossary.md#term-just-in-time-compilation-jit)
analysis described above. It then corrected only the private performance harness to [synchronous](../../../../glossary.md#term-synchronous)
[C2](../../../../glossary.md#term-c2)-only children and 32 alternating equal-count sub-[batches](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters). The project [owner](../../../../glossary.md#term-owner) deliberately chose
not to spend the additional time on the corrected full 792-class by five-fork [run](../../../../glossary.md#term-run) and explicitly
accepted closure with that validation exception. CPU 0008I is therefore Complete without a
performance-pass claim; CPU 0009 receives the corrected full evidence as missing/deferred.

## Completion summary

- Completed changes: the committed schema-58 portable loss implementation is closed, and the
  final test-only harness correction plus truthful validation exception are recorded without
  changing production generated code.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [loss glossary](../../../../glossary.md#loss); [validation glossary](../../../../glossary.md#term-validation))</span>
- Files changed or created by this documentation pass:
  `docs/planning/backends/cpu/tasks/0008i-portable-loss-family-execution.md`,
  `docs/planning/backends/cpu/master-plan.md`, and `docs/planning/roadmap.md`.
- Tests and validation: reused the implementation-owned 12-test focused pass with 2 opt-in skips,
  complete 792-Class-File structural evidence, retained performance/JIT diagnostics, and prior<span data-glossary-navigation="true"> ([JIT glossary](../../../../glossary.md#term-just-in-time-compilation-jit))</span>
  `git diff --check`; this pass ran documentation status/link/anchor/fence/path checks and final
  `git diff --check` only.
- Documentation-agent review: clean documentation completion context
  `01a06d82-d72e-7c93-b762-ef8e43ff1976` finalized the task, [CPU](../../../../glossary.md#term-cpu) master-plan, and roadmap evidence.
- Documentation impact: planning status, validation exception, residual risk, and CPU 0009
  evidence transfer are synchronized; architecture and explanatory guides remain unchanged.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [residual glossary](../../../../glossary.md#scan-residual-port-and-history-demand))</span>
- Javadoc review: no production Javadoc change is needed because the latest correction is confined
  to private test-harness measurement behavior; its affected test Javadoc is accurate.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Glossary impact: none; no reusable project term or existing term boundary changed.
- Unresolved issues: corrected full 792-class by five-fork generated/direct evidence is missing,
  so complete loss-family near-parity and whole-backend performance parity are not established.<span data-glossary-navigation="true"> ([fork glossary](../../../../glossary.md#term-fork-meanings); [backend glossary](../../../../glossary.md#backend))</span>
- Follow-up required: CPU 0009 must classify this evidence as missing and rerun it when that
  checkpoint reaches the frontier; CPU 0008J is the next separate planning task.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>

Status: Complete
