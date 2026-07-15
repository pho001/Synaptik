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
  <root>  package-private graph capture now; later narrow public or cross-package compiler
          collaborations only when a concrete consumer justifies them
```

The root package begins with one cohesive internal capture operation. It must not become a
catch-all for unrelated passes, artifacts, diagnostics, planning adapters, and public facades.
Later tasks must refine this map before adding another package or making another detailed task
`Ready`.

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| 0001 | [Tensor expression graph capture](tasks/0001-tensor-expression-graph-capture.md) | Complete | Completed model graph/provenance/RNG-state foundations and model milestone closure | Replaced the placeholder with package-private deterministic forward capture from requested Tensor outputs to `CompiledGraphModel`, preserving identity, every producer output slot, graph boundaries, and opaque state edges without a public facade. |
| 0002 | Captured-graph inference and validation | Draft | 0001 | Revalidate operand domains and descriptors, represent and prove deferred constraints where possible, and reject invalid captured graphs without transformation or backend decisions. |
| 0003 | Canonicalization and forward optimization | Draft | 0002 | Add the bounded semantics-preserving canonicalization and forward optimization pipeline after graph validation is stable. |
| 0004 | Autograd and backward graph construction | Draft | 0002–0003 | Expand selected compile modes into valid combined forward/backward graph state, then support post-autograd optimization. |
| 0005 | Publication, planning orchestration, and compile artifacts | Draft | 0001–0004, stable config/planning/trace consumers | Orchestrate publication, backend-neutral ownership/partition/logical-memory planning, diagnostics, and immutable `CompileArtifacts` without prepare/runtime/backend state. |


## Milestones

- Capture and validation
- Optimization and autograd
- Planning orchestration and compile artifacts

## Current status

In progress through an explicitly bounded roadmap interleave. Task 0001 is Complete and remains
the only detailed compiler specification. It captures the now-closed model expression/provenance
surface into the already-current immutable graph model. Tasks 0002–0005 remain Draft rows without
detailed specifications; no next compiler task is `Ready` until its separate planning step.

This interleave does not claim that the compiler project's full roadmap entry condition is met.
Task 0001 depends only on completed model contracts and uses none of the still-Draft config cost,
trace payload, runtime, prepare, planning-orchestration, publication, or compile-artifact
surfaces. Completed capture creates the concrete compiler producer needed to make those later
contracts consumer-driven.

## Open questions

- The candidate boundary remains Draft until graph transformations, compile artifacts, and the
  prepare/tuning orchestration consumer are stable. No public Java declaration is selected here.
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
- Config 0004 is not selected because capture consumes no planning cost classification or unit.
  Trace 0003+ are not selected because capture emits no payload. Runtime is not selected because
  prepared contracts and runtime-facing config/trace consumers are not yet stable. Prepare is not
  selected because it requires compiler artifacts and runtime contracts, neither of which task
  0001 invents.

## Risks

- Creating prepared or backend-specific state during compilation.
- Accidentally merging structurally equal but identity-distinct Tensor producers.
- Dropping unrequested producer output slots that later compiler work may need as auxiliary or
  opaque state values.
- Publishing a speculative compiler facade before engine/config/artifact consumers are stable.

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and follow [the planning guide](../../planning-guide.md).
