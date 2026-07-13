# Implementation Roadmap

## Authority

This roadmap coordinates implementation order. It is not an architecture contract. The authoritative contract is [`ARCHITECTURE.md`](../../ARCHITECTURE.md), and it wins if this roadmap conflicts with it.

## Execution policy

Implementation advances through one active frontier at a time. Complete the current area's tasks in master-plan order before moving to the next area. Create a detailed task specification only for the next unfinished task.

Parallel work is not the default. It requires an explicit roadmap or master-plan note confirming that dependencies and affected files do not overlap.

## Ordered project areas

| Order | Project area | Status | Entry condition | Exit condition |
|---|---|---|---|---|
| 1 | [`modules/model`](modules/model/master-plan.md) | Draft | Repository and planning infrastructure are ready. | Selected model capabilities and all model task acceptance criteria are complete. |
| 2 | [`modules/trace`](modules/trace/master-plan.md) | Draft | Required model contracts are stable or confirmed unnecessary. | Typed trace DTO contracts and validation are complete. |
| 3 | [`modules/backend-contract`](modules/backend-contract/master-plan.md) | Draft | Foundational value-model conventions are stable. | Backend identity and declarative requirement contracts are complete. |
| 4 | [`modules/config`](modules/config/master-plan.md) | Draft | Model and backend identity contracts required by configuration are stable. | Compile, prepare, run, and profile configuration contracts are complete. |
| 5 | [`modules/planning`](modules/planning/master-plan.md) | Draft | Model, trace, backend-contract, and config contracts are ready. | Ownership, partitioning, scoring, and logical memory planning are complete. |
| 6 | [`modules/runtime`](modules/runtime/master-plan.md) | Draft | Runtime-facing config, backend identities, and trace contracts are ready. | Prepared runtime contracts and dynamic run-state foundations are complete. |
| 7 | [`modules/compiler`](modules/compiler/master-plan.md) | Draft | Model, config, planning, backend-contract, and trace contracts are ready. | Compile artifacts, graph transformations, and autograd compilation are complete. |
| 8 | [`modules/prepare`](modules/prepare/master-plan.md) | Draft | Compiler, planning, runtime, config, backend-contract, and trace contracts are ready. | Shared prepare contracts and validation are complete. |
| 9 | [`backends/openblas-provider`](backends/openblas-provider/master-plan.md) | Draft | Native interop conventions needed by the provider are decided. | The low-level provider contract and validation are complete. |
| 10 | [`backends/cpu`](backends/cpu/master-plan.md) | Draft | Model, config, planning, runtime, prepare, backend-contract, trace, and OpenBLAS contracts are ready. | CPU is a conforming reference backend for the selected capability set. |
| 11 | [`modules/engine`](modules/engine/master-plan.md) | Draft | Compiler, runtime, prepare, and the CPU backend can be composed. | The public compile, prepare, and run lifecycle works end to end on CPU. |
| 12 | [`backends/metal`](backends/metal/master-plan.md) | Draft | Shared backend contracts and CPU reference behavior are stable. | Metal passes the applicable backend-conformance suite. |
| 13 | [`backends/cuda`](backends/cuda/master-plan.md) | Draft | Shared backend contracts and CPU reference behavior are stable. | CUDA passes the applicable backend-conformance suite. |
| 14 | [`extensions/onnx`](extensions/onnx/master-plan.md) | Draft | The model representation and public tensor semantics are stable. | Selected import/export mappings and compatibility validation are complete. |
| 15 | [`extensions/nn`](extensions/nn/master-plan.md) | Draft | Model semantics, compiler capture, and execution foundations are stable. | Module, parameter, buffer, train/eval, and selected layer contracts are complete. |
| 16 | [`extensions/training`](extensions/training/master-plan.md) | Draft | NN parameter contracts, config, compiler autograd, and runtime publication contracts are stable. | Backend-independent optimizer and training-session capabilities are complete. |
| 17 | [`tools/tuning`](tools/tuning/master-plan.md) | Draft | Config and planning profiles are stable. | Tuning produces validated immutable profiles. |
| 18 | [`tools/benchmarks`](tools/benchmarks/master-plan.md) | Draft | Engine and selected execution paths are operational. | Repeatable benchmark suites and reporting are complete. |
| 19 | [`tools/cli`](tools/cli/master-plan.md) | Draft | Engine and diagnostic contracts are stable. | Selected diagnostic and execution commands are complete. |

The order above is the default delivery sequence, not a new dependency rule. Allowed and forbidden dependencies remain defined only by `ARCHITECTURE.md`.

## Current frontier

The current project area is [`modules/model`](modules/model/master-plan.md).

Its completed implementation frontier is:

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
task 0023 is Complete with its detailed specification and final matrix. Task 0023A is Complete
with its detailed specification. Task 0023B is Complete with its detailed specification. Task
0023C is the next Draft frontier; four concise Draft follow-up rows 0023C–0023F precede
established task 0024, which remains Draft without a detailed specification.
Task 0023B's focused 15-suite run passed 124 tests, its single final model suite passed 981 tests
across 125 suites, and the separate documentation pass validated model Javadoc, the executable
example, Markdown, exact 26-path scope, the 190-method public Tensor surface, and synchronized
status.

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
Task 0023B is Complete with its detailed specification. Task 0023C is the next Draft frontier;
tasks 0023C–0023F and task 0024 remain Draft without detailed specifications.

Task 0018R selects normalized start/length/signed-step slice attributes rather than a negative-end
sentinel, retains the general array primitive, adds explicit-step `sliceAxis` and one-occurrence
`flip(int... axes)`, and leaves every negative-step result layout-unresolved under the current
non-negative-stride descriptor. It removes public `Tensor.foldAxis` without an alias while
retaining `WindowTransformKind.FOLD_AXIS` and `FoldAxisAttrs` as public Java semantic contracts
without a public Tensor receiver/construction method.
[Task 0023](modules/model/tasks/0023-adjoint-expressibility-audit.md) selected Draft task 0023D to
restore the generally useful public overlap-add primitive and separately generalize 2D windows.
Public `unfold`, `unfold2d`, and `fold2d` remain unchanged.

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
contracts. [Task 0023](modules/model/tasks/0023-adjoint-expressibility-audit.md) has selected Draft
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
| 117 | 0023C Slice placement and dynamic crop | Draft |
| 118 | 0023D Public foldAxis and dynamic 2D windows | Draft |
| 119 | 0023E Cumulative-product scan | Draft |
| 120 | 0023F Attention weights output | Draft |
| 121 | 0024 Model capability selection audit | Draft |

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
result artifact. Tasks 0023A and 0023B are Complete with their detailed specifications. Task
0023C is the next Draft frontier; tasks 0023C–0023F and established task 0024 remain Draft without
detailed specifications.
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
