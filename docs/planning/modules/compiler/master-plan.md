# Compiler Master Plan

## Goal

Compile Tensor expressions into immutable compile artifacts through one phase-aware capture,
validation, exact graph transformation, compiler-owned autograd, and planning orchestration.

## Architecture references

- [Architecture contract](../../../../ARCHITECTURE.md)
- [Lifecycle](../../../architecture/lifecycle.md)
- [Training graph](../../../architecture/training-graph.md)
- [Module boundaries](../../../architecture/module-boundaries.md)
- [Dependency rules](../../../architecture/dependency-rules.md)
- [ADR 0009](../../../design/decisions/0009-compiler-owned-pre-capture-tensor-expression-autograd.md)

## Scope

- graph capture and indexing
- shape/data-type inference and validation
- canonicalization and exact graph optimization
- compiler-owned pre-capture Tensor-expression autograd
- combined forward/backward graph construction
- complete valid backend-neutral graph-transformation candidates for later bounded model tuning
- publication, planning, logical memory orchestration, and diagnostics

## Out of scope

- Tensor gradient/backward lifecycle state
- model-owned derivative rules
- a second low-level gradient algebra
- physical buffers
- prepared schedules and executions
- backend-specific lowering
- concrete kernel selection

## Module invariants

- Compiler output is immutable compile-time state.
- Compiler never constructs runtime execution units.
- Compiler has no concrete backend dependency.
- Autograd rules and reverse accumulation belong to compiler.
- Forward and generated gradient expressions share one model algebra, inference/validation
  contract, numerical-semantics contract, and exact optimization pipeline.
- Tensor identity maps exist only during one compile request and are not graph representations.
- One phase-aware capture assigns graph-local IDs once, including the combined graph for
  backward-capable modes.
- Compiler owns graph-candidate semantics and validity; tuning may measure bounded complete
  candidates but does not construct or reinterpret them.

## Allowed dependencies

- modules/model
- modules/config
- modules/planning
- modules/backend-contract
- modules/trace

## Forbidden dependencies

- runtime, prepare, engine, and concrete backend modules

## Package structure

```text
io.github.pho001.synaptik.compiler/
  <root>  package-private forward capture and backward-capable combined capture, inference and typed constraints,
          deterministic canonicalization, exact arithmetic rewriting, logical-splat facts and
          folding, DCE/CSE orchestration, named Tensor-expression gradient rules, reverse
          accumulation, combined-graph gradient result roles, the current narrow public
          artifact/cross-package Planning boundary justified by Compiler 0005, and the bounded
          public functional request/result/order contracts selected by Compiler 0006
```

The root package remains one cohesive internal compiler-front-end boundary. It must not become a
catch-all for public facades, pass registries, gradient registries, generic algebra builders,
artifacts, diagnostics, or planning adapters. Compiler task 0005 justifies its narrow
cross-package/public orchestration boundary from a concrete consumer.

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| 0001 | [Tensor expression graph capture](tasks/0001-tensor-expression-graph-capture.md) | Complete | Completed model graph/provenance/RNG-state foundations and model milestone closure | Added package-private deterministic forward capture from requested Tensor outputs to `CompiledGraphModel`, preserving exact producer identity, every output slot, graph boundaries, and opaque state edges. |
| 0002 | [Captured-graph inference and validation](tasks/0002-captured-graph-inference-and-validation.md) | Complete | 0001 | Independently derives and verifies every current operation descriptor, rejects semantic contradictions, and retains only genuinely unresolved typed Shape constraints. |
| 0003 | [Canonicalization and forward optimization](tasks/0003-canonicalization-and-forward-optimization.md) | Complete | 0002 | Added mandatory deterministic graph-local reindexing plus one config-controlled forward DCE/CSE/DCE sequence, revalidating every changed immutable candidate through 0002. |
| 0003A | [Exact arithmetic rewriting](tasks/0003a-exact-arithmetic-rewriting.md) | Complete | 0003 | Added the closed seven-rule guarded exact arithmetic matrix before the forward DCE/CSE/DCE sequence, with 0002 revalidation and no relaxed algebra. |
| 0003B | [Compile-time constants and constant folding](tasks/0003b-compile-time-constants-and-constant-folding.md) | Complete | 0003A | Added explicit logical-splat ingress, bounded BOOL/signed-integral folding, and sidecar-aware constant-source pruning without storage reads or physical values. |
| 0004 | [Compiler-owned pre-capture autograd and graph compilation](tasks/0004-compiler-owned-pre-capture-autograd-and-combined-graph-compilation.md) | Complete | Model 0025; Compiler 0001–0003B; Config 0002 | Added fail-closed preflight for one scalar-objective/implicit-unit-seed first-order request, the closed initial gradient matrix through ordinary public Tensor operations, one combined phase-aware capture, and proved exact whole-graph optimization with phase-local CSE. |
| 0004A | [Exact-composition gradient-rule extensions](tasks/0004a-exact-composition-gradient-rule-extensions.md) | Complete | 0004 | Added the bounded policy-free matrix for typed ERF, masked and locally invertible shape-target SUM, role-aware floating MATMUL, and selected exact data-movement adjoints through the existing one-capture pipeline. |
| 0004B | [Shared-algebra cotangent normalization and local derivative rules](tasks/0004b-shared-algebra-cotangent-normalization-and-local-derivative-rules.md) | Complete | 0004A | Added mixed-floating Shape/DataType normalization, ordinary DIV and MEAN formulas, and direct-zero FLOOR/CEIL/SIGN plus masked-all-false local conventions without a gradient-specific algebra or optimization policy. |
| 0005 | [Publication, planning orchestration, and compile artifacts](tasks/0005-publication-planning-orchestration-and-compile-artifacts.md) | Complete | 0001–0004B; Planning 0006 closure; stable Config 0001–0003 and Backend Contract 0001–0004 inputs | Added the package-private complete compile entry, ordered publication/constant/diagnostic artifacts, and Compiler-owned graph-wide orchestration through three narrow package-cohesive Planning operations without prepare/runtime/backend state. |
| 0005A | [Derivative policy and elementwise/activation gradient completion](tasks/0005a-derivative-policy-and-elementwise-activation-gradient-completion.md) | Complete | Model 0025A; Compiler 0005 | Completed the exact 48-kind binary/scalar arithmetic, selection/cast, unary, activation, comparison/logical/classification inventory with fixed tie, endpoint, discontinuity, domain, NaN, infinity, Shape/type-normalization, and non-differentiable-role policy. |
| 0005B | [Reduction, scan, softmax, statistics, and normalization gradient completion](tasks/0005b-reduction-scan-softmax-statistics-and-normalization-gradient-completion.md) | Complete | Model 0025B; 0005A | Adopted binding-aware EXPAND and completed binding-dependent sum-to-Shape, products, extrema, scans, softmax/log-softmax, statistics, norms, and layer/RMS/batch normalization, including explicit boundary policies, non-differentiable mask/index roles, and same-occurrence saved batch-statistic outputs. |
| 0005C | [Layout, window, indexing, scatter, ordering, and stochastic gradient completion](tasks/0005c-layout-window-indexing-scatter-ordering-and-stochastic-gradient-completion.md) | Complete | Models 0025C–0025D; 0005B | Completed layout/slice/composition and dynamic window rules, including retained dynamic extraction and target-relative placement obligations, Gather/scatter variants with fixed duplicate/zero/tie policies, exact sort/top-K routing, and dropout through canonical auxiliaries; coordinates, indices, one-hot/BOOL outputs, RNG state, masks, and configuration roles remain non-differentiable. |
| 0005D | [Attention, convolution, pooling, and loss gradient completion](tasks/0005d-attention-convolution-pooling-and-loss-gradient-completion.md) | Complete | 0005B, 0005C | Verified the implemented MATMUL/linear chain and completed the representable structured-ML inventory: two-output attention with canonical weights, grouped convolution, pooling, and every current loss role/reduction mode; one-output attention and dynamic/zero-depth index loss fail closed. |
| 0005E | [First-order gradient coverage closure checkpoint](tasks/0005e-first-order-gradient-coverage-closure-checkpoint.md) | Complete | 0005A, 0005B, 0005C, 0005D | Added one package-private coverage checker; closed all current 37 kind families, 107 constants, 128 signature variants, legal output slots, and ordered input roles as differentiable, intentionally non-differentiable, or explicitly fail-closed; and proved bounded transitive and connected nested-pass formula closure at the first-order capability checkpoint. |
| 0006 | [Explicit functional gradient requests and higher-order differentiation](tasks/0006-explicit-functional-gradient-requests-and-higher-order-differentiation.md) | Complete | 0005E and the stable public compile/artifact boundary from 0005 | Added one public immutable one/two-stage functional request, explicit/default cotangent seeds, ERROR/ZERO disconnected behavior, ordered `GradientPublicationBinding` values, and compiler-owned derivative-order metadata over the closed first-order formula matrix without Tensor gradient lifecycle state or another compile facade. |
| 0007 | Exact constant identities and permission-aware algebra | Draft | 0006; Config 0006 before any relaxed rule | Reassess remaining graph-level exact constant/algebra identities and any explicitly permitted relaxed rewrites without changing completed 0001–0006 history: preserve current guarded scalar `POW(+1) -> input`, require complete exceptional-value/constant-sidecar/output/publication/phase/autograd/descriptor proof before an exact `POW(0)` typed shape-correct one-splat, and never infer Tensor constants from storage or factory history. |

Tasks 0005A–0005D partition the complete current model operation inventory without claiming that
every role has a gradient. A task may claim family coverage only after it has implemented every
differentiable role in its assigned signatures, preserved every intentionally non-differentiable
BOOL, index, RNG-state, mask, or configuration role, and explicitly selected every required
subgradient, boundary, or exceptional-value policy. Dynamic or binding-dependent rules and
same-occurrence auxiliary outputs remain logical compiler concerns; they must not become static-
Shape assumptions, physical saved buffers, a runtime tape, or backend work.

The milestone extends rather than replaces completed Compiler 0004–0004B. Their supported matrices
remain the implemented baseline; each family task must preserve and revalidate its assigned
implemented rows while adding only the missing differentiable roles and explicit decisions.

Complete task 0005E is the first-order closure gate, not another formula-family task. It verifies
the current inventory against compiled production Model classes rather than a stale
hand-maintained family list, confirms that unknown or new operations still fail closed, and
checks a bounded cross-family set of generated formula edges plus connected nested expansions.
Existing 0005A–0005D suites remain the per-family formula evidence. This checkpoint preserves the
higher-order path without implementing higher-order requests before 0006.

## Milestones

- Capture and validation — Complete through task 0002.
- Exact optimization foundations — Complete through task 0003B.
- Pre-capture autograd and graph compilation — Complete through task 0004B and its compiler
  transformation/autograd capability checkpoint.
- Planning orchestration and compile artifacts — Complete through task 0005.
- Complete current-inventory first-order gradient coverage — Complete through
  [task 0005E](tasks/0005e-first-order-gradient-coverage-closure-checkpoint.md), including Model
  prerequisites 0025C–0025D; task 0005E is the completed closure checkpoint and dependency gate
  for task 0006.
- Explicit functional and bounded higher-order gradients — Complete through
  [task 0006](tasks/0006-explicit-functional-gradient-requests-and-higher-order-differentiation.md).

## Current status

Complete through task 0006. Tasks 0001–0006 are Complete with recorded source, tests,
documentation, and validation. Draft task 0007 records later work without a detailed
specification. Compiler 0004, 0004A, and 0004B are Complete with
recorded source, tests, documentation, and validation.
[Compiler 0004B](tasks/0004b-shared-algebra-cotangent-normalization-and-local-derivative-rules.md)
adds the closed mixed-floating cotangent
Shape/DataType normalization through ordinary `sumToShape` and `cast`, binary/scalar DIV local
formulas, direct-zero FLOOR/CEIL/SIGN conventions, and ordinary/masked MEAN formulas whose logical-
one denominators support static, dynamic, and expression Shapes. Forward and generated
expressions retain one shared algebra and exact optimization contract. The final exact 16-path
change contains five production files, four tests, and seven documentation/planning files. The
compiler module passed 18 suites/136 tests; the final transformation/autograd checkpoint passed
167 suites/1,275 tests with no skipped tests, failures, or errors; and the independent
documentation pass finalized Javadocs and current status.
[Compiler 0005](tasks/0005-publication-planning-orchestration-and-compile-artifacts.md) is Complete
with recorded source, tests, documentation, and validation. Focused
[Model 0025A](../model/tasks/0025a-portable-floating-comparison-extrema-and-clamp-semantics.md) is
Complete and fixes the shared floating comparison/extrema/clamp forward contract. Compiler
[0005A](tasks/0005a-derivative-policy-and-elementwise-activation-gradient-completion.md)
is Complete with the exact elementwise/activation derivative-policy matrix, fixed coefficient
bits, exact typed-splat cache, source-backed inventory, and one shared Tensor algebra. Complete
[Model 0025B](../model/tasks/0025b-binding-aware-expansion.md) supplies the remaining
implementation prerequisite. Detailed
[Compiler 0005B](tasks/0005b-reduction-scan-softmax-statistics-and-normalization-gradient-completion.md)
is Complete with binding-aware EXPAND inference/preflight, output-slot-aware accumulation, and the
complete assigned reduction/scan/softmax/statistics/norm/normalization matrix. Its exact
eighteen-path change passed 22 compiler suites/170 tests with no skips, failures, or errors plus
independent Javadoc, Markdown, surface, scope, and status validation. Complete
[Model 0025C](../model/tasks/0025c-portable-functional-scatter-reduction-semantics.md) supplies
the forward numerical-contract prerequisite. Complete
[Model 0025D](../model/tasks/0025d-dynamic-extent-slice-extraction-and-symbolic-slice-placement.md)
supplies the remaining dynamic-slice construction prerequisite. Detailed
[Compiler 0005C](tasks/0005c-layout-window-indexing-scatter-ordering-and-stochastic-gradient-completion.md)
is Complete through its exact compiler implementation, tests, API/Javadoc review, and
documentation gates. Detailed
[Compiler 0005D](tasks/0005d-attention-convolution-pooling-and-loss-gradient-completion.md)
is Complete with compiler-owned attention, grouped-convolution, pooling, and loss formulas,
canonical attention/max-pool auxiliary use, exact logical counts and coefficients, and explicit
fail-closed signatures.
[Compiler 0005E](tasks/0005e-first-order-gradient-coverage-closure-checkpoint.md) is Complete. Its
package-private checker closes the compiled-production 37-family, 107-kind, 128-signature role
inventory with exact `D`/`ND`/`FC`, owner/reason, ranged-cardinality, fail-closed, and Tensor-ID
evidence. A bounded 22-case cross-family expansion probe classified 64 generated edge
fingerprints, completed 10 connected nested passes, and identity-proved 12 disconnected skips.
The replacement focused command passed 43 tests, the full Compiler module passed 29 suites/194
tests, and the repository/architecture checkpoint passed.
[Compiler 0006](tasks/0006-explicit-functional-gradient-requests-and-higher-order-differentiation.md)
is Complete. Its approved final terminology correction distinguishes Model
`ForwardPublicationBinding` from Compiler `GradientPublicationBinding`; `PublicationPlan`
exposes `forwardBindings()` and `gradientBindings()` while the package-private
`GraphCompilation` component names remain unchanged. The final focused correction command passed
35 tests, and the single final affected-module command passed Model 127 suites/1,031 tests plus
Compiler 31 suites/208 tests with no skips, failures, or errors. No semantic, validation-order,
visibility, ownership, capture, optimization, publication, or ID behavior changed. No later
compiler task has a detailed specification.
The final focused seven-suite 0005D command passed. After the MSE negative-coefficient expression
order was corrected, the replacement full compiler-module run passed 28 suites and 189 tests with
no failures, errors, or skips. No executable Java changed after that evidence; the separate clean
documentation pass finalized Javadocs and the six affected documentation/planning paths under the
authorized 20-path ceiling.
The final exact fifteen-path 0005A change passed the compiler module suite with 22 suites/159
tests and no skips, failures, or errors; its independent documentation pass finalized all
affected Javadocs and explanatory/planning surfaces.

The first clean Compiler 0005 implementation context stopped before edits after confirming that
the originally planned `planning.compiler` facade could not call package-private top-level
operations in three sibling Java packages. The corrected specification kept the two
capability internals package-private, added one colocated public owner-selection collaboration, and
widened only the already-audited partition and logical-memory operations in their owning packages.
The completed 37-path change therefore includes their package Javadocs and existing
visibility-locking tests atomically. No compiler artifact, semantic, failure-order, or architecture
decision changed.

Accepted ADR 0009 changes the next compiler architecture from captured-forward placeholder
conversion to compiler-owned pre-capture Tensor-expression autograd. The prerequisite is
[Model task 0025](../model/tasks/0025-canonical-tensor-producer-outputs.md), which is Complete. It
makes exact hidden producer outputs retrievable without reconstructing wrappers. The dedicated
planning and implementation passes made
[Compiler 0004](tasks/0004-compiler-owned-pre-capture-autograd-and-combined-graph-compilation.md)
Complete with a bounded package-private scalar-objective/implicit-unit-seed request, an exact
fail-closed first rule matrix, one phase-aware capture, immutable result roles, and combined exact
optimization. The implementation first verified exactly six obsolete untracked Java prototypes,
deleted them without reading or adapting their contents, and verified their absence before Gradle
and at final status. Those removal-only paths plus 32 tracked create/modify paths stayed within
the 38 touched-path ceiling.

[Compiler 0004A](tasks/0004a-exact-composition-gradient-rule-extensions.md) is Complete with the
additive fail-closed `SUPPORTED_0004A` matrix. It adds fixed-bit typed ERF, masked SUM, locally
provable SUM_TO_SHAPE inversion, every floating MATMUL vector/matrix rank case, and guarded
SLICE/SLICE_UPDATE/SELECT/PAD/TILE/CONCAT/STACK cotangents. The implementation preserves
Compiler 0004's request, seed, identity accumulation, one-capture, phase, validation, and
optimization contracts.

The current general package-private entry owner is `GraphCompiler`, and its exact
parameter list is not wrapped in a request aggregate. It returns mode-neutral package-private
`GraphCompilation`: `FORWARD_ONLY` has no BACKWARD nodes and empty gradient results, while
backward-capable modes may carry the combined forward/backward graph. This internal graph-stage
result is distinct from the later-lifecycle `CompileArtifacts` aggregate added by task 0005.

Compiler 0005 preserves that entry and result while adding one package-private complete
compile overload. Compiler remains the graph-wide orchestrator: it validates ordered forward and
gradient publication roles, asks Planning for one owner decision per final graph node, assembles
the complete owner map, and invokes Planning's existing partition and logical-memory derivation.
The task exposes only `BackendOwnerPlanning.selectOwner(...)` in `planning.capability` plus the
existing `MaximalSameOwnerPartitioning.partition(...)` and `LogicalMemoryPlanning.plan(...)`
operations widened in their owning packages. `BackendEligibility` and `BackendOwnerSelection`
remain package-private. The task returns immutable `CompileArtifacts` containing output-only
`PublicationPlan`, `CompileConstantPlan`, and `CompileDiagnostics` values. Those artifacts retain
logical compile state only; they do not retain live providers, availability snapshots, selected
devices, routes, kernels, physical memory, prepared/runtime state, trace events, or executables.

After Compiler 0005 and the focused Model 0025A prerequisite, tasks 0005A–0005E form one explicit
dependency-ordered milestone
that closes first-order gradient coverage across the complete current model inventory before
higher-order work. The sequence first resolves elementwise/activation derivative policies and
formula foundations, then reductions/scans/softmax/statistics/normalization, then dynamic layout/
window/indexing/scatter/ordering/stochastic families, then structured attention/convolution/
pooling/loss families, and finally one source-backed coverage checkpoint. Tasks 0005A–0005E are
Complete. Detailed task 0006 is also Complete; no later compiler task is specified.

This reordering preserves completed history. Tasks 0003, 0003A, and 0003B were correctly completed
for a forward-only immutable graph. Compiler 0004 reused their existing exact rules only where
their guards are phase-safe: after one combined capture and initial validation, it performs exact
rewrite/fold, whole-graph liveness, phase-local CSE, and DCE cleanup once each and revalidates every
changed candidate through task 0002. It adds no new algebra.

Config 0004 remains Draft because these compiler transformations require no planning-cost
classification. Trace 0003 and later remain Draft because no stable emission schema is selected.
Compiler introduces no prepared or executable state. The post-0006 frontier reassessment advances
only Runtime 0001; Prepare remains Draft.

## Open questions

- No blocking question remains for task 0006. Its detailed specification fixes the bounded public
  one/two-stage functional request, exact forward/stage-one references, aligned seeds, ERROR/ZERO
  disconnected policy, result ordering, and compiler-owned derivative-order sidecar. A public
  Engine facade and more than two derivative stages remain future decisions.
- Complete task 0005C adopts retained slice/window constraints, preflight, and gradients; fixes
  scatter duplicate/zero/tie/NaN/signed-zero routing, stable ordering/cutoff routing, and dropout
  mask/probability behavior; and preserves dynamic geometry plus non-differentiable index/RNG
  roles.
- Complete task 0005D selects the remaining attention, convolution, pooling, and loss special-case
  policies, exact auxiliary-output use, and intentional fail-closed signatures. Complete task
  0005E found no unresolved policy for a current differentiable role.

## Decisions made

- Legacy code is read-only capability and formula evidence. Its mutable `Tensor.gradient`,
  `ThreadLocal` compilation scope, Tensor-owned derivative dispatch, and mutable graph cloning are
  rejected.
- `FORWARD_ONLY` skips autograd.
- `FORWARD_AND_BACKWARD` and the initial `TRAINING_STEP` construct the combined Tensor expression
  before capture. `TRAINING_STEP` adds no optimizer updates yet.
- Before backward construction, compiler inventories every backward-reachable producer occurrence,
  output role, exact attributes, cotangent-normalization path, and required local differentiation
  rule or convention. Unsupported work fails closed.
- Full inference/validation occurs after the one backward-capable combined capture. Later failures
  may consume temporary Tensor IDs; IDs are never rolled back or reused.
- Named compiler components such as `ElementwiseGradientRules` own dispatch. Formulas use only
  ordinary public Tensor operations such as `mul`, `add`, `sumToShape`, and `transpose`.
- Generated gradient expressions obey exactly the same model algebra, inference, validation,
  numerical semantics, and existing guarded optimization rules as forward expressions. Autograd
  adds local differentiation formulas, genuinely necessary local conventions, and cotangent
  Shape/DataType normalization; it adds no gradient-only numerical or optimization policy.
- Compiler 0003A's exact typed scalar `POW(+1) -> input` bypass remains implemented under its
  current internal-one-output `FORWARD`, descriptor-equality, false-gradient, and non-public-output
  guards. Later planning must not rewrite that completed history or imply broader current power
  algebra.
- Draft 0007 owns only future graph-level identities. An exact `POW(0)` replacement requires a
  typed shape-correct logical one-splat and a complete proof covering exceptional values, explicit
  constant sidecars, output/publication identity, graph phase, autograd, and descriptors. Tensor
  exponent forms are eligible only when compiler-owned immutable constant facts prove one exact
  typed uniform exponent; Tensor storage and factory history are never evidence.
- `POW(0.5)` must not be silently replaced by `SQRT`, nor may analogous fractional identities be
  assumed, because signed-zero, domain, exceptional-value, and rounding behavior can differ.
  `POW(2)`, `POW(-1)`, and other small integral exponents remain semantic `POW` in the compiled
  graph and are primarily backend-prepare realization choices.
- Forward and compiler-generated gradient operations use the same exact/default numerical policy.
  No gradient-specific numerical permission or rewrite policy is planned. Any later relaxed
  compiler rewrite requires the caller permission owned by Config 0006; hardware, providers,
  workload size, objectives, tuning, and benchmarks cannot supply it.
- One compile request may use `IdentityHashMap`-style Tensor-to-contribution and
  Tensor-to-accumulated-gradient bookkeeping. It is ephemeral compiler state, not Tensor state,
  graph IR, or a second graph.
- Contribution accumulation uses ordinary `Tensor.add` in deterministic contribution order.
- Seeds and derivative constants are storage-free Tensor leaves/expressions explicitly registered
  as logical splats. Tensor storage and factory history are never constant evidence.
- Model task 0025 supplies the exact canonical wrapper for every producer output slot, including
  hidden dropout and batch-normalization auxiliaries. Compiler does not reconstruct wrappers.
- Model task 0025A is the forward numerical-contract prerequisite for Compiler 0005A. It fixes
  ordinary ordered floating comparisons, numeric equality/inequality, NaN-propagating MIN/MAX,
  signed-zero extrema, and ordered `MIN(MAX(input, minValue), maxValue)` CLAMP without choosing any
  derivative tie, endpoint, discontinuity, singularity, or exceptional-value convention.
- Complete Model task 0025B is the semantic prerequisite for Compiler 0005B. It broadens existing
  EXPAND construction only when an aligned pair contains an unresolved Dimension, retains the
  exact target Shape, and leaves layout unresolved. Complete Compiler 0005B owns
  `AnyOf(DimensionEqual(source, 1), DimensionEqual(source, target))` inference/proof, preflight,
  and binding-dependent gradient consistency; Model adds no constraint object or binding.
- Complete Model task 0025C is the forward numerical-contract prerequisite for Compiler 0005C. For
  configurable Scatter Elements and Scatter-ND MUL/MIN/MAX, the base participates once, each
  addressed update participates once, duplicates remain distinct multiset members, and an empty
  update group preserves the exact base representation. Floating product/extrema and integral
  modular/signed behavior reuse existing Model contracts; encounter, layout, atomic, and tree
  order are not observable. Model 0025C chooses no derivative policy and changes neither NONE,
  ADD, SCATTER_ADD, bounds, nor duplicate validation.
- Complete Model task 0025D supplies the remaining construction prerequisite for Compiler 0005C. It adds
  finite length-defined extraction across unresolved selected extents and target-relative
  symbolic slice placement through existing slice kinds and attributes. Compiler 0005C owns
  retained-constraint inference/proof, preflight, and gradients; Model 0025D adds none of those
  compiler behaviors.
- Complete Compiler 0005C constructs one ordinary stable `ARGSORT` Tensor occurrence with the
  exact same input, normalized axis, and direction as a one-output `SORT` occurrence and use its
  indices in the SORT backward formula. This is recomputation through the existing separate
  one-output Model operation, not a hidden SORT output, sibling lookup, producer reconstruction,
  or public sort API change. Preflight must fail closed before partial backward construction if
  the exact SORT/ARGSORT signatures, kinds, or attributes cannot be matched.
- Complete Compiler 0005D reuses canonical weights for exact two-output attention, preserves
  group isolation through public unfold/matrix/fold convolution formulas, reconstructs the
  first eligible maximum-pool winner without saved indices, uses the fixed logical kernel count
  for average pooling, and restores loss reductions through logical typed counts. One-output
  attention and index loss with dynamic or zero class depth fail closed; MSE uses exact typed
  scalar-operation coefficients `2` and `-2`.
- Complete Compiler 0005B fixes the reduction, scan, softmax, statistics, norm, and normalization
  derivative policies and exact formulas. Product rules are division-free;
  extrema split equal ties; standard deviation and L2 norm select zero at zero result; statistical
  correction uses logical typed ones; and batch training consumes canonical same-occurrence saved
  outputs through output-slot-aware preflight and accumulation.
- Compiler 0005A fixes those separate first-order elementwise conventions: symmetric extrema
  ties, ordered-composition CLAMP endpoints, direct ABS/RELU discontinuities, raw analytic
  domains, activation infinity extensions, and intentionally non-differentiable comparison,
  logical, classification, condition, attribute, bound, and non-floating cast roles. It uses
  fixed typed coefficient bits and request-local exact splats without changing model semantics.
- Phase-aware capture receives forward outputs, gradient roots and target roles, the original
  forward-producer identity set, and explicit constant facts. It assigns `NodeId`/`ValueId` once
  and retains `GraphPhase` per node.
- Multiple targets may map independently to the same captured gradient `ValueId`; the graph output
  boundary lists each distinct gradient value once and adds no identity node for role separation.
- Initial combined optimization applies only the exact 0003A/0003B rules whose current guards are
  proved safe, whole-graph DCE, and phase-local CSE. Every changed graph is revalidated through
  0002. No new algebra follows from autograd.
- Generated gradients remain ordinary differentiable Tensor expressions. Detailed task 0006
  selects exactly one optional second reverse-mode stage, compile-local `createGraph`, reuse of
  the closed formula-operation coverage, and compiler-owned order metadata adjacent to unchanged
  Model `GraphPhase`.
- Compiler 0004A extends the closed preflight and named rule owners only. Its ERF coefficient uses
  fixed typed scalar-attribute bits; its MATMUL and SLICE_UPDATE selection is role-aware; and
  repeated operand positions remain repeated deterministic contributions.
- Compiler 0004 first verifies the exact path/status of six obsolete untracked Java prototypes
  under the production source root, deletes them before any implementation edit or compiler
  invocation, and never copies, adapts, moves, stages, or treats their contents as design
  authority.
- No task adds `Tensor.gradient`, `Tensor.backward`, mutable gradient state, placeholder
  `ValueId` conversion, a second low-level algebra, a public gradient registry/facade, a physical
  tape, or backend-owned global autograd.
- Compiler 0005 fixes the publication, constant, diagnostic, and immutable compile-artifact
  boundary and keeps graph-wide planning orchestration in Compiler. Planning exposes only one
  colocated owner-selection collaboration over its two internal capability stages and the two
  existing package-owned stateless operations needed to generate partitions and derive logical
  memory; the task does not inherit unfinished graph simplification.
- The first-order completion milestone is ordered 0005A -> 0005B -> 0005C -> 0005D -> 0005E.
  Formula families remain compiler-owned pre-capture Tensor-expression construction through the
  same request-local identity maps, one combined capture, inference/validation, and exact
  optimization pipeline established by 0004–0004B.
- Coverage means every current operation role is either supported for first-order
  differentiation or intentionally non-differentiable. It does not mean that BOOL, index,
  RNG-state, mask, or configuration roles receive cotangents.
- Derivative boundary and subgradient choices are implementation-frontier decisions in their
  assigned task. Compiler 0005B–0005D record implemented policies in Complete specifications.
  Model tasks choose only shared forward meaning and do
  not choose derivatives.
- Complete task 0005E proved the bounded current transitive formula-operation checkpoint before
  0006. It adds no create-graph behavior, derivative-order representation, or higher-order
  request.

## Risks

- Treating pre-capture Tensor expressions as graph-local IR.
- Leaving the six obsolete untracked prototypes under the compiler production source root, where
  Gradle would compile unauthorized code, or adapting their contents instead of deleting them.
- Publishing or reconstructing a sibling output instead of using the producer's canonical exact
  wrapper.
- Constructing partial backward expressions before discovering an unsupported exact attribute,
  normalization path, local differentiation rule, or convention.
- Losing repeated-operand contributions or changing deterministic accumulation order.
- Inferring constants from Tensor storage, labels, descriptors, provenance absence, or factory
  history.
- Assigning graph-local IDs in separate forward and backward passes.
- Replacing per-node phase with only a positional backward boundary.
- Manufacturing identity nodes when result roles share one gradient value.
- Applying forward-only rewrite/folding assumptions to backward nodes without proving their
  existing guards.
- Treating DIV, MEAN, casts, exceptional values, or optimization eligibility as a separate gradient
  algebra instead of using their ordinary shared operation contracts.
- Treating complete inventory coverage as a claim that BOOL, index, RNG-state, mask, or
  configuration roles are differentiable.
- Silently selecting a tie, subgradient, discontinuity, empty-domain, or exceptional-value policy
  while adding a family formula.
- Assuming static geometry where a current rule is binding-dependent, or reconstructing/resampling
  a saved auxiliary instead of using the canonical same-occurrence output.
- Entering higher-order work while an operation used by a generated first-order formula still
  fails the differentiable-role inventory.
- Merging equal expressions across phases before an explicit proof.
- Turning logical saved Tensor edges into physical buffers, recomputation policy, runtime
  scheduling, or a compiler-owned tape.
- Creating a public facade before the engine/config/artifact consumers are stable.
- Treating Tensor storage or construction history as an exponent constant, replacing fractional
  powers with superficially similar unary operations, or letting a benchmark grant numerical
  permission.

## Notes

Follow the planning guide's progressive-planning rule. Model 0025 and Compiler 0004–0004B are
Complete. Compiler 0004B stayed within its exact 16-path ceiling: five compiler production files,
four compiler tests, and seven documentation/planning files. Its compiler module validation,
independent documentation pass, and compiler transformation/autograd capability checkpoint all
passed. Compiler 0005 and Model 0025A are Complete.
[Compiler 0005A](tasks/0005a-derivative-policy-and-elementwise-activation-gradient-completion.md)
is Complete. It closed the complete 0005A policy and implementation boundary through the exact
fifteen authorized paths and independent documentation pass. Complete
[Model 0025B](../model/tasks/0025b-binding-aware-expansion.md) supplied the next compiler
prerequisite. Detailed
[Compiler 0005B](tasks/0005b-reduction-scan-softmax-statistics-and-normalization-gradient-completion.md)
is Complete through the exact eighteen authorized paths and independent documentation pass.
[Model 0025C](../model/tasks/0025c-portable-functional-scatter-reduction-semantics.md) and
[Model 0025D](../model/tasks/0025d-dynamic-extent-slice-extraction-and-symbolic-slice-placement.md)
are Complete as the forward numerical-contract and dynamic-slice construction prerequisites.
Detailed
[Compiler 0005C](tasks/0005c-layout-window-indexing-scatter-ordering-and-stochastic-gradient-completion.md)
is Complete. It fixes the complete assigned inventory, formulas, retained constraints, numerical
policies, canonical auxiliary-output use, fail-closed ordering, authorized 23-path ceiling, and
implementation/documentation handoff. Detailed 0005D and
[0005E](tasks/0005e-first-order-gradient-coverage-closure-checkpoint.md) are Complete. Detailed
[0006](tasks/0006-explicit-functional-gradient-requests-and-higher-order-differentiation.md) is
Complete with the approved forward/gradient publication-binding terminology correction;
Draft 0007 is the concise later row for remaining exact identities and permission-aware algebra.
It has no detailed task specification.
