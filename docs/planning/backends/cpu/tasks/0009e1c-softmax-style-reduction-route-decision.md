# Task 0009E1C: Softmax-Style Reduction Route Decision

## Status

Ready

## Goal

Decide whether the complete CPU `SOFTMAX`/`LOG_SOFTMAX` route remains bounded generated execution or becomes finite direct typed Java, choosing only the lower total implementation-and-verification-cost route. No benchmark, performance, or numerical redesign is authorized.

## Scope

Decide the complete current first-class one-input/one-output `SoftmaxKind.SOFTMAX` and `SoftmaxKind.LOG_SOFTMAX` family with `SoftmaxAttrs`: identical `FLOAT64`, `FLOAT32`, or `BFLOAT16` input/result types and gradient eligibility; normalized in-rank positive-width axis; result Shape equal to input Shape. A normalization slice fixes all non-selected coordinates.

Preserve static resolved non-negative layouts, dense-linear/general-odometer geometry, non-injective input reads/injective output writes, heap-array/`MemorySegment`/ordered mixed carriers, checked span/access/alignment/non-overlap, scalar or disjoint caller-parallel ranges owning complete slices, immutable prepared binding, zero scratch/workspace, and one final typed store per output. Preserve stable three-pass maximum, compensated shifted-exponential sum, final result, and represented narrowing: `SOFTMAX` divides exponentials by sum; `LOG_SOFTMAX` subtracts its logarithm. Preserve empty non-selected-extent no-op behavior, finite input/shift validation before write or submission, and rejection rather than a new NaN/infinity policy.

Cold-reject without output mutation or submitted work every current excluded fact: not exactly one supported first-class node; decomposed graph; wrong kind/attrs/arity/value identity/type/gradient eligibility/Shape/rank/axis/layout; non-floating/non-identical type; unresolved, negative, overflowing, inaccessible, undersized, or misaligned layout/carrier; non-injective output; zero selected extent; input/output overlap; invalid bases/range/workers/resources; non-finite represented input or maximum shift; and scratch/workspace request. Confirm this code boundary before implementation; no silent fallback.

## Out of scope

Layer/RMS/batch normalization, loss, attention, other reductions, partial/combine execution, vector/native routes, fusion or decomposed-softmax recognition, Model/Compiler semantics or gradients, public/shared Prepare/Runtime, architecture/modules/build, benchmarks, and performance claims. Do not create detailed E2, E3, F, or G tasks. Legacy is read-only evidence only.

## Architecture references and constraints

Read [architecture](../../../../../ARCHITECTURE.md), [current architecture plan](../../../../architecture/current-architecture-plan.md), [Planning Guide](../../../planning-guide.md), [documentation rules](../../../../developer-guide/documentation-rules.md), [CPU master plan](../master-plan.md), parent [CPU 0009](0009-portable-generated-coverage-closure-checkpoint.md), completed [E1](0009e1-partial-integral-reduction-direct-java-migration.md), [E1A](0009e1a-masked-reduction-route-decision.md), [E1B1](0009e1b1-log-sum-exp-route-decision.md), [E1B2](0009e1b2-statistical-reduction-route-decision.md), [E1B3](0009e1b3-norm-reduction-route-decision.md), and [0007E](0007e-portable-stable-softmax-and-log-softmax-coverage.md).

`ARCHITECTURE.md` is authoritative. CPU Prepare owns lowering, finite route selection, resource declaration, and finalization after slot assignment; Runtime invokes immutable prepared work only. Preserve borrowed carriers, isolated `RunState`, cold validation before writes/submission, and prepared-region ownership. No public/shared API, generic normalization executor, resource kind, dependency, architecture, or build change; stop on a need for one.

## Package impact

Existing CPU-private packages only: `internal.ir`/`internal.lowering` retain identity/geometry; `internal.codegen.emit` retains `CpuSoftmaxEmitter` if generated; `internal.prepare`, `internal.executable`, `internal.cache`, and `route.portable` retain selection, validator, artifact/binding, finalization, and invocation; `internal.reference` retains the independent clean-Java oracle. Retention adds no type. Direct Java may add one narrowly named CPU-private softmax executable, never a generic service.

## Affected files and maximum scope

Expected planning paths are this task plus the three synchronized records. Review-only seams: capability, `CpuSoftmaxIr`, lowering, emitter/classfile generator, validator, prepare/finalization/executable, route, artifact/cache, reference oracle, and focused tests. At most 28 paths (14 production/Javadoc, ten test/resource, four planning) and one new CPU-private type. A 29th path, second type, shared/public/resource/build/architecture/conformance/integration change, materialization, partial state, or generic element dispatch is stop-and-replan.

## Acceptance criteria

- The final decision covers the complete matrix by total implementation plus verification cost, without benchmark gate or performance claim.
- It preserves both kinds/types, Shape/axis/layout admission, dense/general geometry, heap/segment/mixed carriers, scalar/disjoint complete-slice ownership, zero workspace, immutable invocation, cold no-write failure, and one store.
- It preserves stable three-pass/compensation/narrowing, empty non-selected extent, finite-input/shift rejection, and special-value boundary.
- Direct Java is finite typed cold-selected carrier/geometry code with immutable range-owning calls; it has no generic per-element carrier/type/layout/kind dispatch, allocation, boxing, reflection, map/string, or avoidable virtual semantic dispatch; it fully retires generated selection, artifact/cache, invocation, inventory/evidence, and references.
- Retention records typed carrier/layout, numerical, validator, binding, artifact/cache, invocation, oracle, and retirement costs making replacement larger, and retains an optimal clean-Java oracle with matching stable dataflow/store shape.
- Retention requires generated Class-File/hot-loop inspection against that oracle, proving no hidden Synaptik helper calls or avoidable allocations, boxing, reflection, or generic/virtual semantic dispatch. No literal-bytecode, JIT, or performance assertion is required.
- Focused tests cover kinds/types, dense/general, carrier pairings, axis positions, empty/nonempty geometry, finite/narrowing/special values/one-store, range/canary/no-write failures, lowering/validator/prepare/finalization/invocation/artifact/cache/reference/inventory/retirement.
- A separate clean documentation-focused context finalizes Javadocs, explanatory/planning status, glossary impact, and documentation validation.

## Route-selection decision rule

First establish that either candidate preserves every Scope semantic and support boundary; an
unsupported candidate is not comparable. Record a bounded, auditable cost comparison with these
non-performance buckets: (1) generated retention's emitter/Class-File inspection, typed
carrier/layout binding, validator, artifact/cache, invocation, oracle, and focused-evidence
maintenance; (2) direct Java's finite typed implementation and cold selection/binding work; and
(3) direct Java's complete generated-route retirement plus semantic, rejection, invocation,
artifact/cache, inventory, oracle, and hot-loop verification work. Select direct Java only when
the documented sum of buckets (2) and (3) is demonstrably strictly lower than bucket (1).
Equal, uncertain, incomplete, or non-auditable evidence retains generation. Record the candidate
facts, each bucket, comparison, selected route, and retirement consequence in validation evidence.
This rule supplies neither a performance metric nor a benchmark or performance claim.

## Tests / validation

After executable stabilization run focused current softmax-owner tests: capability, package inventory, softmax IR/lowering/generated/input-validator/reference, generated matrix/direct-evidence/coverage inventory, and affected preparer/finalizer/prepared-executable/artifact-cache tests. Record exact commands/XML counts; add retirement tests before deletion. No benchmark or repository-wide suite.

Documentation context reuses stable Java evidence unless it changes executable Java; it validates links, anchors, fences, terminology/glossary, exact scope, dependencies/status/frontier, then runs:

```bash
git diff --check
git status --short -uall
```

Architecture, conformance, integration, and repository validation defer to 0009G/CI because no shared/end-to-end boundary may change.

## Dependencies and follow-up tasks

- E1, E1A, E1B1, E1B2, and E1B3 are Complete retained-generated-route decisions.
- Complete 0007E supplies semantic identity, stable algorithm, geometry, validator, emitter, artifact/binding, and reference oracle.
- E2, E3, F, and G remain Draft summaries only. E2 follows this decision; do not create it here.

## Architecture impact

Expected impact: None. Stop and report any public, dependency, resource, module, or architecture conflict.

## Implementation prompt

```text
You are the clean implementation agent for Synaptik CPU 0009E1C. Do not commit, push, stage, reset, revert, delete, or modify unrelated work; do not use GSD. Read AGENTS.md, ARCHITECTURE.md, current architecture plan, Planning Guide, CPU master plan, CPU 0009, E1/E1A/E1B1/E1B2/E1B3/0007E, this task, current softmax source/tests, documentation rules, and General/Planning profiles. Implement exactly this route decision. Choose direct finite typed Java only if total implementation and verification cost is lower; otherwise retain generation with concrete evidence and Class-File/hot-loop comparison to the clean-Java oracle. Preserve the confirmed boundary; stop on conflict. Hand the stabilized diff and exact test evidence to a separate clean documentation context. No benchmarks/repository-wide tests/commit/push/stage.
```

## Rationale

Both kinds share complete-slice geometry, cold carrier/binding, finite prevalidation, and three stable passes; only the final division versus logarithm/subtraction differs. This follows advanced reductions and precedes normalization without authorizing either adjacent family.

## Local decisions

Empty until implemented.

## Known limitations

No speed, vector, scaling, bytecode-parity, or performance conclusion follows. Dynamic Shapes/layouts, zero selected width, non-finite input, decomposed graphs, fusion, materialization, partial/combine, and native execution remain excluded.

## Validation evidence for planning

Before creating this Ready task, the planning context inspected required architecture/planning/documentation records; 0007E and E1--E1B3; Model/Compiler softmax references; `CpuCapabilityProvider`, `CpuSoftmaxIr`, lowering, emitter, validator, reference oracle, preparation/finalization/invocation, artifact/cache, and focused evidence tests. It confirmed the Scope rejection boundary and no architecture conflict. This planning-only change runs no Gradle or benchmark.

## Implementation notes

Empty until implemented.

## Completion summary

Empty until implemented.
