# Compiler Master Plan

## Goal

Compile tensor expressions into immutable compile artifacts through capture, validation, transformation, autograd, and planning orchestration.

## Architecture references

- [Architecture contract](../../../../ARCHITECTURE.md)
- [Module boundaries](../../../architecture/module-boundaries.md)
- [Dependency rules](../../../architecture/dependency-rules.md)

## Scope

- graph capture and indexing
- shape and data type inference and validation
- canonicalization and graph optimization
- complete valid backend-neutral graph-transformation candidate generation for later bounded
  model tuning
- autograd and backward graph construction
- publication, planning, logical memory orchestration, and diagnostics

## Out of scope

- physical buffers
- prepared schedules and executions
- backend-specific lowering
- concrete kernel selection

## Module invariants

- Compiler output is immutable compile-time state.
- Compiler never constructs runtime execution units.
- Compiler has no concrete backend dependency.
- Compiler owns the semantics and validity of graph candidates; tuning tooling may measure a
  bounded set but does not construct or reinterpret them.

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
  <root>  package-private graph capture, captured-graph verification inference, typed deferred
          constraints, deterministic graph canonicalization, guarded exact arithmetic rewriting,
          and the bounded forward DCE/CSE/DCE pipeline; later narrow public or
          cross-package collaborations only when a concrete consumer justifies them
```

The root package remains one cohesive internal front-end and forward-transformation boundary
through task 0003A. Capture produces the structurally closed graph; verification inference
revalidates semantics and retains unresolved constraints; canonicalization normalizes graph-local
IDs; one guarded topological helper applies a closed seven-rule set comprising duplicate binary
MIN/MAX, exact typed scalar positive-one identities, and exact typed integral scalar-zero
identities; and one narrow orchestrator applies that helper before the existing forward
DCE/CSE/DCE sequence under the existing config permission. No pass registry,
generic rewrite framework, public optimizer, or candidate model is planned. The package must not
become a catch-all for artifacts, diagnostics, planning adapters, and public facades. Task 0005
must justify any narrow cross-package/public orchestration boundary from its concrete consumer.

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| 0001 | [Tensor expression graph capture](tasks/0001-tensor-expression-graph-capture.md) | Complete | Completed model graph/provenance/RNG-state foundations and model milestone closure | Replaced the placeholder with package-private deterministic forward capture from requested Tensor outputs to `CompiledGraphModel`, preserving identity, every producer output slot, graph boundaries, and opaque state edges without a public facade. |
| 0002 | [Captured-graph inference and validation](tasks/0002-captured-graph-inference-and-validation.md) | Complete | 0001 | Independently derives and verifies every current operation occurrence descriptor, rejects operand/domain contradictions, and retains only genuinely unresolved typed Shape constraints without transformation, binding, or backend decisions. |
| 0003 | [Canonicalization and forward optimization](tasks/0003-canonicalization-and-forward-optimization.md) | Complete | 0002 | Added mandatory deterministic graph-local reindexing plus one config-controlled forward DCE/CSE/DCE sequence, revalidating every changed immutable graph candidate through task 0002. |
| 0003A | [Exact arithmetic rewriting](tasks/0003a-exact-arithmetic-rewriting.md) | Complete | 0003 | Added a closed seven-rule set: guarded non-gradient internal forward binary `MIN(x, x)`/`MAX(x, x)`, exact typed scalar MUL-by-one for every current numeric domain, DIV/POW-by-one for floating domains, and ADD/SUB-zero for integral domains; it runs after canonical validation and before unchanged one-shot DCE/CSE/DCE, revalidates changes through Compiler 0002, distinguishes immutable scalar attributes from 0003B Tensor constants, and excludes broader algebra and relaxed/fast math. |
| 0003B | Compile-time constants and constant folding | Draft | 0003A | Define a compiler-owned immutable constant fact/ingress representation and exact deterministic folding, never treating mutable public Tensor host storage as authoritative compile-time data; revalidate every changed candidate through Compiler 0002 and exclude runtime/backend execution, physical allocation, broad partial evaluation, relaxed/fast-math, and architecture changes. |
| 0004 | Autograd and backward graph construction | Draft | 0002, 0003, 0003A, and 0003B | Expand selected compile modes into valid combined forward/backward graph state after exact forward rewriting and constant folding, then support post-autograd optimization. |
| 0005 | Publication, planning orchestration, and compile artifacts | Draft | 0001–0004 including 0003A–0003B, stable config/planning/trace consumers | Orchestrate publication, backend-neutral ownership/partition/logical-memory planning, diagnostics, and immutable `CompileArtifacts` without prepare/runtime/backend state. |


## Milestones

- Capture and validation
- Optimization and autograd — run the compiler transformation-and-autograd capability checkpoint
  after task 0004
- Planning orchestration and compile artifacts

## Current status

In progress through an explicitly bounded roadmap interleave. Tasks 0001–0002 are Complete and
close the compiler capture-and-validation milestone. Task 0001 captures the closed model
expression/provenance surface into the current immutable graph model. Task 0002 adds
package-private verification inference over that graph: every current production operation family
is revalidated, complete output descriptors are independently derived and compared, and current
dimension/Shape obligations are proven, rejected, or retained as typed internal constraints.
Task 0003 is Complete. It adds only mandatory deterministic graph-local reindexing plus
the smallest optional standard forward pipeline: dead-code elimination, exact common-subexpression
elimination, then one dead-code cleanup. Each newly constructed graph candidate is revalidated
through task 0002. The bounded sequence runs once and does not iterate to a fixed point.
Task 0003A is Complete. Its implementation provides a closed seven-rule set:
internal forward binary `MIN(x, x)`/`MAX(x, x)` plus exact typed scalar MUL-by-positive-one in all
numeric domains, DIV/POW-by-positive-one in floating domains, and ADD/SUB-zero in integral
domains. Every bypass requires complete descriptor equality, `requiresGrad == false`, a one-output
occurrence, and a non-graph-output result; scalar rows additionally require the exact immutable
`ScalarValueAttrs` carrier, type, and typed value. One scan runs after canonical validation and
before task 0003's unchanged one-shot DCE/CSE/DCE sequence; a changed candidate is revalidated
through task 0002. Scalar attributes are graph-visible semantic facts, not Tensor constants;
Tensor zero/one recognition and new result constants remain 0003B. Gradient-eligible occurrences
remain until task 0004 defines their operation, tie, and operand-multiplicity semantics. Draft
task 0003B then defines the compiler-owned immutable fact/ingress boundary for compile-time
constants and exact
deterministic folding without reading mutable public Tensor host storage as authoritative
compile-time data. It revalidates changed candidates through task 0002
and excludes runtime/backend execution, physical allocation, broad partial evaluation,
relaxed/fast-math, and architecture changes. Tasks 0003A and 0003B precede autograd so these
separately proved forward-only rewrite boundaries are stable before task 0004 introduces backward
occurrences and post-autograd optimization. Tasks 0003B, 0004, and 0005 remain Draft rows without
detailed specifications; no compiler task is Ready.

This interleave does not claim that the compiler project's full roadmap entry condition is met.
Tasks 0001–0003A depend only on completed model contracts and the completed config optimization
permission. They use none of the still-Draft config cost, trace payload, runtime, prepare,
planning-orchestration, publication, or compile-artifact surfaces. Task 0003 consumes only a
successful task-0002 result and the standalone `GraphOptimizationConfig`; it creates no compile
aggregate or public artifact. Task 0003A consumes the same internal graph/config boundary and adds
no new module or public dependency.

## Open questions

- The candidate boundary remains Draft until graph transformations, compile artifacts, and the
  prepare/tuning orchestration consumer are stable. No public Java declaration is selected here.
- The owning public/artifact boundary for unresolved graph constraints remains deferred until
  task 0005 and a concrete prepare/runtime binding consumer are stable. Task 0002 uses an internal
  result only and does not decide serialization or concrete binding.
- The first cross-package collaboration with planning remains deferred to task 0005. Planning's
  four current evaluator/generator operations stay package-private until that concrete compiler
  orchestrator can justify one narrow boundary.

## Decisions made

- The implementation must follow the current architecture contract.
- Legacy code is capability evidence only; new implementation is written from scratch.
- Model autotuning does not move backend-neutral graph-transformation ownership out of the
  compiler or backend-specific fusion ownership out of concrete backends.
- Task 0001 uses one package-private capture entry point returning only `CompiledGraphModel`.
  It adds no transitional public compiler facade or API.
- Capture assigns graph-local IDs from deterministic requested-output/input/output-slot encounter
  order, deduplicates exact producer identities, preserves every producer output slot, and keeps
  internal opaque RNG-state edges.
- Direct `GraphRngState` boundary selection remains unsupported because the model deliberately
  exposes no public state Tensor. Reachable state edges are captured through producer inputs.
- Task 0002 validates graph model data rather than reconstructing temporary Tensor expressions.
  It returns the exact accepted graph plus immutable unresolved constraints, fails closed for an
  unknown operation kind, and never binds named or expression dimensions.
- Task 0003 consumes only a successful `ValidatedGraph`. Mandatory canonicalization allocates
  graph inputs first and then node outputs in topological/output-slot order, with dense IDs from
  zero, while preserving exact operations, descriptors, phases, and graph boundaries.
- Task 0003's optional standard pipeline is one forward DCE pass, one exact forward CSE pass, then
  one forward DCE cleanup pass, without fixed-point iteration. CSE uses equal phase/operation/
  ordered-remapped-input/complete-descriptor keys, merges all outputs slotwise, and permits graph-
  output producers neither to merge nor to serve as representatives. DCE retains all graph inputs,
  graph outputs, non-forward work and dependencies, and every output slot of a live node.
- Task 0003 reuses task 0002 after mandatory canonicalization and after every changed optional
  candidate. Disabled optimization suppresses only the DCE/CSE/DCE sequence, never
  canonicalization or validation.
- Constant value execution, cast/arithmetic/algebraic/view rewrites, decomposition, fusion, a pass
  registry, graph-candidate collection, cost/tuning behavior, and backend-specific work are not
  justified by this first pipeline and remain absent.
- Complete task 0003A selects exactly seven non-gradient internal forward rules: duplicate binary
  MIN/MAX, exact typed scalar MUL-by-one in every current numeric domain, DIV/POW-by-one in current
  floating domains, and ADD/SUB-zero in current integral domains. Complete input/output descriptor
  equality, one output, and graph-output exclusion are common guards; a changed candidate is
  revalidated before task 0003's unchanged one-shot DCE/CSE/DCE sequence consumes it.
- Every gradient-eligible selected occurrence remains intact because task 0004 has not yet defined
  operation, saved-value, extrema-tie, or repeated-operand backward semantics.
- Task 0003A reads exact immutable scalar operation attributes but rejects Tensor constant
  recognition, floating ADD/SUB zero, MUL zero, cancellation, bounds/clamp identities, other POW
  exponents, broader algebra, reassociation, commutation, reciprocal substitution, and exceptional-
  value assumptions. POW 2, -1, and other small integers remain possible future backend-prepare
  strength reductions subject to explicit numerical/conformance contracts; relaxed or fast-math
  transformations require a future explicit numerical-permission contract.
- Draft task 0003B follows 0003A and owns compile-time constants and constant folding. It must
  define compiler-owned immutable constant facts and their ingress representation, perform only
  exact deterministic folding, never treat mutable public Tensor host storage as authoritative
  compile-time data, and revalidate every changed graph candidate through Compiler 0002. It must
  not add runtime/backend execution, physical allocation, broad partial evaluation,
  relaxed/fast-math behavior, or architecture changes.
- Config 0004 is not selected because capture, validation, and graph transformation consume no
  planning cost classification or unit. Trace 0003+ are not selected because none of these tasks
  emits a payload.
  Runtime is not selected because prepared contracts and runtime-facing config/trace consumers
  are not yet stable. Prepare is not selected because it requires compiler artifacts and runtime
  contracts, none of which tasks 0001–0003 invents.

## Risks

- Creating prepared or backend-specific state during compilation.
- Merging identity-distinct occurrences without the complete exact CSE key, or collapsing a
  graph-output/publication boundary that must remain distinct.
- Dropping an output slot from a retained multi-output producer that later compiler work may need
  as an auxiliary, saved, or opaque state value.
- Publishing a speculative compiler facade before engine/config/artifact consumers are stable.
- Duplicating model-time validation incompletely and silently trusting an unhandled operation
  family instead of failing closed.
- Treating an unresolved symbolic obligation as proven, or inventing runtime binding/public
  artifact contracts before their consumers exist.
- Applying a mathematically familiar arithmetic identity without proving it against the current
  operation, data-type, exceptional-value, signed-zero, overflow, and promotion semantics.
- Bypassing an arithmetic output whose descriptor differs from its input, or collapsing a
  graph-output occurrence whose distinct requested identity must remain observable.
- Deriving authoritative compile-time constants from mutable public Tensor host storage instead
  of a compiler-owned immutable ingress fact.

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and follow [the planning guide](../../planning-guide.md).
