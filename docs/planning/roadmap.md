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
| 15 | [`extensions/training`](extensions/training/master-plan.md) | Draft | Model, config, compiler autograd, and runtime publication contracts are stable. | Backend-independent optimizer and training-session capabilities are complete. |
| 16 | [`tools/tuning`](tools/tuning/master-plan.md) | Draft | Config and planning profiles are stable. | Tuning produces validated immutable profiles. |
| 17 | [`tools/benchmarks`](tools/benchmarks/master-plan.md) | Draft | Engine and selected execution paths are operational. | Repeatable benchmark suites and reporting are complete. |
| 18 | [`tools/cli`](tools/cli/master-plan.md) | Draft | Engine and diagnostic contracts are stable. | Selected diagnostic and execution commands are complete. |

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
- [0017N Public unfold, foldAxis, unfold2d, and fold2d Tensor expressions](modules/model/tasks/0017n-unfold-and-fold-tensor-expressions.md)
  — Complete.
- [0018A Scalar select semantics](modules/model/tasks/0018a-scalar-select-semantics.md) — Complete.

The next planning frontier is:

- 0018B Scalar select Tensor expression — Draft without a detailed specification.

The former broad task 0017 is decomposed into tasks 0017A–0017N so parameterless contiguous
meaning, public expression construction, shape/view transformations, slicing, pad/tile,
composition, and unfold/fold contracts can be implemented and validated independently. Tasks
0017A–0017N have detailed specifications and are complete. The former broad task 0018 is now
decomposed into focused tasks 0018A–0018J for select, gather, and functional-scatter semantics and
expressions. Task 0018A is complete; tasks 0018B–0018J and all later tasks remain Draft without
detailed specifications.

Task [0018A](modules/model/tasks/0018a-scalar-select-semantics.md) is complete with the exact
`SELECT` identity and normalized scalar axis/index attributes. Its independent documentation
review passed focused 9-test, all 638-model-test/75-suite, model-Javadoc, root-test,
javap/reflection/import/generated-page, Markdown, exact eight-path, synchronized-status, and
no-0018B-spec checks. Public Tensor construction and every cross-layer behavior remain deferred.

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
helper while leaving value comparison and execution deferred.
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
construction, checked Shape arithmetic, identity-only dynamic preservation, unresolved result
layout, and fresh provenance. Task
[0017K](modules/model/tasks/0017k-tensor-composition-semantics.md) is complete with CONCAT, STACK,
and individually indexed UNSTACK-output semantics without provenance or graph changes. Task
[0017L](modules/model/tasks/0017l-tensor-composition-expressions.md) is complete with ordered public
concat/stack, immutable-list unstack expression construction, unresolved result layouts, and exact
ordered or individually indexed provenance without producer grouping or cross-layer behavior.
Task [0017M](modules/model/tasks/0017m-unfold-and-fold-semantics.md) is complete. It defines
general-axis sliding windows and their public/compiler-facing scatter-add fold, NCHW
im2col columns, and overlap-accumulating col2im through typed immutable semantic parameters.
Task 0017N owns all four public Tensor expressions. Task 0023 will later own compiler-generated
FOLD_AXIS use for backward graphs; neither task changes the semantic identity defined by 0017M.
Task [0017N](modules/model/tasks/0017n-unfold-and-fold-tensor-expressions.md) is complete with exact
public signatures, locally provable static/dynamic Shape rules, checked window arithmetic,
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
| 67 | [0017N Public unfold, foldAxis, unfold2d, and fold2d Tensor expressions](modules/model/tasks/0017n-unfold-and-fold-tensor-expressions.md) | Complete |
| 68 | [0018A Scalar select semantics](modules/model/tasks/0018a-scalar-select-semantics.md) | Complete |
| 69 | 0018B Scalar select Tensor expression | Draft |
| 70 | 0018C Axis gather semantics | Draft |
| 71 | 0018D Axis gather Tensor expressions | Draft |
| 72 | 0018E Gather-ND semantics | Draft |
| 73 | 0018F Gather-ND Tensor expression | Draft |
| 74 | 0018G Axis scatter semantics | Draft |
| 75 | 0018H Axis scatter Tensor expressions | Draft |
| 76 | 0018I Scatter-ND semantics | Draft |
| 77 | 0018J Scatter-ND Tensor expression | Draft |
| 78 | 0019 Linear algebra and attention operations | Draft |
| 79 | 0020 Convolution and pooling operations | Draft |
| 80 | 0021 Normalization operations | Draft |
| 81 | 0022 Loss operations | Draft |
| 82 | 0023 Compiler-generated semantic operations | Draft |
| 83 | 0024 Model capability parity audit | Draft |

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
complete. The former broad task 0018 is decomposed into 0018A–0018J. Task 0018A is complete;
tasks 0018B–0018J and every later task remain Draft without a detailed specification.
Completed task 0016E adds fixed-INT64,
one-axis arg-max expression metadata without changing the ordinary reduction helper or adding
value comparison, empty-axis policy, or execution.

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
