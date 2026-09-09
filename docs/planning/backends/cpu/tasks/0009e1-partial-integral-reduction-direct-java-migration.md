# Task 0009E1: Partial Integral Reduction Direct-Java Migration

## Status

Ready

## Goal

Replace the currently unselected generated modular partial-reduction implementation with finite
direct typed Java only if that lowers total implementation and verification work. No benchmark or
performance-parity admission gate is permitted.

## Scope

This is the first complete static-reduction subfamily, not a rewrite of every family formerly
listed under 0009E. It covers one static, one-node ordinary `SUM` or `PROD` reduction with
`CpuAggregateIr` form `FULL`, `SINGLE_AXIS`, or `MULTI_AXIS`; `INT32` or `INT64` input/output;
dense resolved layouts; typed heap-array input/output; and an injective dense output.

Preserve same-width two's-complement modular arithmetic, two or four quotient/remainder partial
domains per complete output cell, ascending partial-ordinal combine, one aligned run-owned state
slot per `[outputCell, partialOrdinal]`, and no floating reassociation. CPU Prepare cold-selects
only after all structural, carrier, worker, range, and workspace facts are known. Runtime binds a
typed direct invocation and does not select a route or carrier inside an element loop.

Direct Java may retain the CPU-private IR, preparation recipe, worker group, and run-owned
`MemorySegment` workspace where they express necessary ownership. It must replace generated
partial-body/combine selection only when finite direct `int[]` and `long[]` bodies have less code
and proof cost. No permanent selected generated/direct duplicate is allowed.

Reject exactly: a partition other than one ordinary aggregate node; non-`SUM`/`PROD` kind;
`SUM_TO_SHAPE`, scalar/invalid aggregate attributes, or a form outside full/single-/multi-axis;
wrong arity, non-distinct values, mismatched type or output Shape; `FLOAT64`, `FLOAT32`,
`BFLOAT16`, `BOOL`, or any non-`INT32`/`INT64` type; unresolved, negative, non-dense, or
non-injective layouts; `MemorySegment` or mixed carriers; zero/overflowing counts; partial count
other than two/four; insufficient workers or minimum domain; inaccessible/misaligned/wrong-sized
workspace; and input/output or workspace/buffer overlap. Every rejection is cold and leaves the
output unchanged. All excluded cases continue through the existing whole-cell route where that
route supports them, otherwise fail closed.

The inspected, distinct later groups remain summary-only and are not implementation permission:

| Order | Current family | Exact current surface | Separate because |
|---|---|---|---|
| E1A | Masked | `SUM`/`MEAN`; `FLOAT64`/`FLOAT32`/`BFLOAT16`; one removed axis; directional `BOOL` broadcast; dense/general and heap/segment/mixed carriers; exact state | Three boundaries and selected-count floating state. |
| E1B | Advanced | `LOG_SUM_EXP`, `VARIANCE`, `STANDARD_DEVIATION`, `L1_NORM`, `L2_NORM`; floating types; multi-axis/statistical attrs; keep/remove dimensions; dense/general and all carrier forms | Stable/two-pass/exact-state policies differ. |
| E1C | Softmax-style | first-class `SOFTMAX`/`LOG_SOFTMAX`; floating types; positive axis; shape-preserving complete slices; dense/general and all carrier forms | Three-pass normalization plus pre-write finite-input validation. |

## Out of scope

Masked, advanced, logarithmic, statistical, norm, softmax, log-softmax, ordinary whole-cell
aggregates, scans, arg extrema, normalizations, losses, vectors, native routes, public Model
semantics, shared Prepare/Runtime contracts, architecture, modules, Gradle, benchmarks, and
performance gates. Do not create E1A/E1B/E1C/E2/E3 task specifications. Do not use the immutable
non-passing 0008P evidence root as an admission criterion. Legacy is read-only evidence only.

## Architecture references and constraints

Read [the architecture contract](../../../../../ARCHITECTURE.md), [the current architecture index](../../../../architecture/current-architecture-plan.md),
[Planning Guide](../../../planning-guide.md), [documentation rules](../../../../developer-guide/documentation-rules.md),
[CPU master plan](../master-plan.md), parent [CPU 0009](0009-portable-generated-coverage-closure-checkpoint.md),
and [CPU 0008P](0008p-deterministic-partial-reduction-parallelism.md).

Planning selects CPU ownership only. CPU Prepare owns lowering, direct-route selection, exact
workspace declaration, and finalization after slot assignment; Runtime invokes immutable prepared
work. Preserve one `RunState`, borrowed caller arrays, run-owned workspace, cold compatibility
checking, and prepared-region ownership. Direct hot loops preserve the clean Java modular
algorithm without per-element allocation, boxing, reflection, string/map dispatch,
synchronization, `Operation`/`CompiledNode`, or avoidable virtual dispatch. A shared/public or
Model-permission change is a stop condition.

## Package impact

Existing CPU-private packages retain `CpuPartialReductionIr` and cold admission in `internal.ir`
and `internal.lowering`; `internal.prepare` owns recipe/workspace selection; `internal.executable`
owns typed direct worker/combine calls; and `internal.codegen.emit` loses partial-only generation
only if no selected consumer remains. Add no package or production type. Keep
`CpuPartialReductionExecution` package-private and focused, never a generic reduction service.

## Affected files and maximum scope

Expected production paths: `CpuPartialReductionLowering`, `CpuPartialReductionIr`,
`CpuPartitionPreparationPlan`, `CpuPartitionPreparer`, `CpuPartitionFinalizer`,
`CpuPreparedExecutable`, `CpuPartialReductionExecution`, and only partial-specific members in
`CpuGeneratedKernel`/`CpuClassFileKernelGenerator`; review affected package Javadocs. Expected
tests are partial IR/lowering/execution, preparer/finalizer/prepared-executable, replacement
direct-route, and inventory/reference tests that name partial artifacts.

Modify/create at most 18 CPU production/test/resource paths plus this task and the four
synchronized planning records. Add no production type. Stop and replan if direct migration needs
a public/shared path, workspace kind, segment support, generic carrier bridge, over 18 CPU paths,
or a second selected route. Delete partial-specific generation only after automated
no-selected-reference checks prove no remaining selected use.

## Acceptance criteria

- Explicitly decide the complete admitted family: migrate only when finite direct Java reduces
  total code/evidence complexity; otherwise retain generation with a concrete reason and no gate.
- A direct route admits exactly current `INT32`/`INT64` `SUM`/`PROD`, three aggregate forms,
  dense arrays/injective output, positive counts, two/four partials, worker capacity, and current
  minimum-domain facts; it rejects all currently excluded facts before output mutation.
- Typed `int[]`/`long[]` loops preserve quotient/remainder bounds and modular arithmetic; the
  caller combines ascending partial states after worker completion. State remains exact, aligned,
  isolated, and owned per active recipe.
- Cold binding exposes typed array calls. Element loops contain no `Object` carrier branch,
  `MethodHandle`, generated lookup, allocation, boxing, reflection, map/string, or generic
  semantic dispatch.
- Tests cover two types, operations, forms, partial counts, quotient/remainder edges, offsets,
  wraparound, repetition/concurrency, malformed/no-write failures, workspace isolation, and an
  independent modular oracle. Ineligible forms still select the whole-cell route.
- Preparation/finalization/invocation tests prove direct selection, worker/workspace declarations,
  typed binding, complete-output ownership, and absent generated artifact generation. Automated
  checks prove retired generated partial owners have no selected reference.
- Retire partial-specific generated selection/inventory/evidence only after semantic, invocation,
  relevant conformance review, and no-selected-reference checks pass. Do not remove shared
  class-file generation used by other families.
- A separate clean documentation-focused pass finalizes Javadocs and planning/docs/glossary review.

## Tests / validation

After executable code stabilizes, run:

```bash
./gradlew :backends:cpu:test \
  --tests io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPartialReductionIrTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPartialReductionLoweringTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPartialReductionExecutionTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizerTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedExecutableTest
```

Add direct-route/no-selected-reference/retirement tests before deletion. Run backend conformance
only if covered behavior changes; no integration suite absent an end-to-end impact. No benchmark,
fresh timing protocol, generated/direct ratio, or JIT claim is a gate. The documentation pass runs
`./gradlew :backends:cpu:javadoc`, validates links/anchors/fences/terminology/glossary and exact
scope, then runs `git diff --check`. Repository-wide validation is deferred to 0009G/CI.

## Dependencies and follow-up tasks

0009D4 is Complete. This is the sole detailed Ready task at the 0009E frontier. After it
completes, the master plan orders summary-only E1A masked, E1B advanced, E1C softmax-style, E2
normalization, and E3 loss. Do not detail those tasks before they become the frontier.

## Architecture impact

Expected impact: None. Report any public, dependency, shared-contract, or architecture conflict.

## Implementation prompt

```text
You are the clean implementation agent for Synaptik CPU 0009E1. Read AGENTS.md,
ARCHITECTURE.md, current architecture plan, Planning Guide, documentation rules and
General/Planning profiles, CPU master plan, CPU 0009, CPU 0008P, and this task. Implement only
this task's complete partial integral-reduction route decision/migration. Preserve modular
semantics, cold selection, typed binding, workspace and range ownership. Choose direct Java only
when it reduces total implementation and verification work; retain generation otherwise. Do not
create later E-task specs, add a benchmark gate, change architecture, commit, or push. Hand the
final diff and exact module-test evidence to a separate clean documentation-focused context.
```

## Local decisions

The former E1 summary mixed four non-cohesive families. Partial integral reduction is the sole
first slice because its typed-array carrier set, modular permission, two/four-state geometry, and
disabled generated candidate are self-contained. The other families remain ordered summaries.

## Known limitations

No speed, vector, parallel-scaling, or generated-bytecode parity claim follows. This task does
not authorize floating partials, segment carriers, arbitrary partial counts, or a silent fallback.
The 0008P evidence root remains historical and non-passing.

## Validation evidence

Planning context `/root` read the architecture, planning, and documentation contracts; CPU
master/0009/D/D1--D4 and 0008P; current capability, partial lowering/IR, generator,
prepare/finalize/executable, tests, inventory, reference, and prior masked/advanced/softmax
evidence. It found partial integral reduction is the only cohesive E1 slice: its generated
candidate is fail-closed by historical performance evidence, whereas the three remaining groups
have incompatible numerical, carrier, and workspace contracts. This planning pass makes no Java
change and runs no Java/Javadoc command. Replace this with final implementation evidence before
marking Complete.

## Implementation notes

None; this task is Ready.

## Completion summary

Pending implementation, validation, and the required separate documentation-focused pass.
