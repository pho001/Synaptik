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
          accumulation, and combined-graph gradient result roles; later narrow public or cross-package
          orchestration only when a concrete consumer justifies it
```

The root package remains one cohesive internal compiler-front-end boundary. It must not become a
catch-all for public facades, pass registries, gradient registries, generic algebra builders,
artifacts, diagnostics, or planning adapters. Compiler task 0005 must justify any narrow
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
| 0004B | Derivative-policy selection and policy-dependent gradient rules | Draft | 0004A | Select explicit boundary, tie, discontinuity, singularity, and cross-floating conversion policies before adding policy-dependent formulas. |
| 0005 | Publication, planning orchestration, and compile artifacts | Draft | 0001–0004B, stable config/planning/trace consumers | Orchestrate publication, backend-neutral ownership/partition/logical-memory planning, diagnostics, and immutable `CompileArtifacts` without prepare/runtime/backend state. |
| 0006 | Explicit functional gradient requests and higher-order differentiation | Draft | 0005 and a stable public compile/artifact boundary | Define explicit objectives, targets, seeds, create-graph or derivative order, formula-operation coverage, and phase/order representation without Tensor gradient lifecycle state. |

## Milestones

- Capture and validation — Complete through task 0002.
- Exact optimization foundations — Complete through task 0003B.
- Pre-capture autograd and graph compilation — tasks 0004–0004B; run the compiler
  transformation/autograd capability checkpoint after 0004B.
- Planning orchestration and compile artifacts — task 0005.

## Current status

In progress through an explicitly bounded roadmap interleave. Compiler 0004 and 0004A are
Complete with recorded source, tests, documentation, and validation. Compiler 0004B is the next
unfinished compiler row but remains Draft without a detailed specification, as do later tasks.

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
result is distinct from the later `CompileArtifacts` aggregate owned by task 0005.

This reordering preserves completed history. Tasks 0003, 0003A, and 0003B were correctly completed
for a forward-only immutable graph. Compiler 0004 reused their existing exact rules only where
their guards are phase-safe: after one combined capture and initial validation, it performs exact
rewrite/fold, whole-graph liveness, phase-local CSE, and DCE cleanup once each and revalidates every
changed candidate through task 0002. It adds no new algebra.

Config 0004 remains Draft because these compiler transformations require no planning-cost
classification. Trace 0003 and later remain Draft because no stable emission schema is selected.
Runtime and prepare remain Draft because no prepared or executable state is introduced.

## Open questions

- The artifact boundary for unresolved constraints and gradient roles remains deferred until task
  0005 and its consumers are stable.
- The public functional boundary for explicit objectives, targets, seeds, and derivative order
  remains deferred to task 0006 after the compile/artifact consumer is stable.
- Task 0004A fixed the bounded policy-free exact-composition matrix. Binding-dependent shape
  inversion, indexing/scatter, windows, structured linear operations, pooling, losses,
  normalization, multi-output attention, ordering, and stochastic rules remain later cohesive
  planning decisions.
- Tie, discontinuity, singularity, exceptional-value, empty-domain, and cross-floating cotangent
  policies remain task-0004B decisions; 0004 and 0004A select none of them.
- The first cross-package collaboration with planning remains deferred to task 0005.

## Decisions made

- Legacy code is read-only capability and formula evidence. Its mutable `Tensor.gradient`,
  `ThreadLocal` compilation scope, Tensor-owned derivative dispatch, and mutable graph cloning are
  rejected.
- `FORWARD_ONLY` skips autograd.
- `FORWARD_AND_BACKWARD` and the initial `TRAINING_STEP` construct the combined Tensor expression
  before capture. `TRAINING_STEP` adds no optimizer updates yet.
- Before backward construction, compiler inventories every backward-reachable producer occurrence,
  output role, exact attributes, and required derivative policy. Unsupported work fails closed.
- Full inference/validation occurs after the one backward-capable combined capture. Later failures
  may consume temporary Tensor IDs; IDs are never rolled back or reused.
- Named compiler components such as `ElementwiseGradientRules` own dispatch. Formulas use only
  ordinary public Tensor operations such as `mul`, `add`, `sumToShape`, and `transpose`.
- One compile request may use `IdentityHashMap`-style Tensor-to-contribution and
  Tensor-to-accumulated-gradient bookkeeping. It is ephemeral compiler state, not Tensor state,
  graph IR, or a second graph.
- Contribution accumulation uses ordinary `Tensor.add` in deterministic contribution order.
- Seeds and derivative constants are storage-free Tensor leaves/expressions explicitly registered
  as logical splats. Tensor storage and factory history are never constant evidence.
- Model task 0025 supplies the exact canonical wrapper for every producer output slot, including
  hidden dropout and batch-normalization auxiliaries. Compiler does not reconstruct wrappers.
- Phase-aware capture receives forward outputs, gradient roots and target roles, the original
  forward-producer identity set, and explicit constant facts. It assigns `NodeId`/`ValueId` once
  and retains `GraphPhase` per node.
- Multiple targets may map independently to the same captured gradient `ValueId`; the graph output
  boundary lists each distinct gradient value once and adds no identity node for role separation.
- Initial combined optimization applies only the exact 0003A/0003B rules whose current guards are
  proved safe, whole-graph DCE, and phase-local CSE. Every changed graph is revalidated through
  0002. No new algebra follows from autograd.
- Generated gradients remain ordinary differentiable Tensor expressions. Higher derivatives wait
  for 0006's explicit create-graph/derivative-order lifecycle, complete rule coverage for formula
  operations, and phase/order representation.
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
- Compiler 0005 owns later publication and planning orchestration. It does not inherit unfinished
  graph simplification.

## Risks

- Treating pre-capture Tensor expressions as graph-local IR.
- Leaving the six obsolete untracked prototypes under the compiler production source root, where
  Gradle would compile unauthorized code, or adapting their contents instead of deleting them.
- Publishing or reconstructing a sibling output instead of using the producer's canonical exact
  wrapper.
- Constructing partial backward expressions before discovering an unsupported exact attribute or
  policy.
- Losing repeated-operand contributions or changing deterministic accumulation order.
- Inferring constants from Tensor storage, labels, descriptors, provenance absence, or factory
  history.
- Assigning graph-local IDs in separate forward and backward passes.
- Replacing per-node phase with only a positional backward boundary.
- Manufacturing identity nodes when result roles share one gradient value.
- Applying forward-only rewrite/folding assumptions to backward nodes without proving their
  existing guards.
- Merging equal expressions across phases before an explicit proof.
- Turning logical saved Tensor edges into physical buffers, recomputation policy, runtime
  scheduling, or a compiler-owned tape.
- Creating a public facade before the engine/config/artifact consumers are stable.

## Notes

Follow the planning guide's progressive-planning rule. Model 0025 and Compiler 0004–0004A are
Complete. Compiler 0004B is the next unfinished compiler row and remains Draft without a detailed
specification; this completion does not create or advance that specification. Later tasks also
remain Draft.
