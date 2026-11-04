# Task 0009E1: Partial Integral Reduction Direct-Java Migration

## Status

Complete

## Goal

Replace the currently unselected generated modular partial-reduction implementation with [finite](../../../../glossary.md#term-finite)
direct typed Java only if that lowers total implementation and verification work. No benchmark or
performance-parity [admission](../../../../glossary.md#term-admission) gate is permitted.

## [Scope](../../../../glossary.md#term-scope)

This is the first complete static-reduction subfamily, not a rewrite of every family formerly
listed under 0009E. It covers one static, one-[node](../../../../glossary.md#node) ordinary `SUM` or `PROD` reduction with
`CpuAggregateIr` form `FULL`, `SINGLE_AXIS`, or `MULTI_AXIS`; [`INT32`](../../../../glossary.md#data-type--datatype) or [`INT64`](../../../../glossary.md#data-type--datatype) input/output;
dense resolved [layouts](../../../../glossary.md#layout); typed [heap](../../../../glossary.md#term-heap)-array input/output; and an [injective](../../../../glossary.md#term-injective) dense output.

Preserve same-width [two's-complement modular arithmetic](../../../../glossary.md#arithmetic-rearrangement-and-finite-difference-checks), two or four quotient/remainder partial
domains per complete output cell, ascending partial-ordinal combine, one aligned [run](../../../../glossary.md#term-run)-owned state
slot per `[outputCell, partialOrdinal]`, and no floating [reassociation](../../../../glossary.md#term-reassociation). [CPU](../../../../glossary.md#term-cpu) [Prepare](../../../../glossary.md#prepare) cold-selects
only after all structural, [carrier](../../../../glossary.md#term-storage-carrier), worker, range, and workspace facts are known. [Runtime](../../../../glossary.md#term-runtime) binds a
typed direct invocation and does not select a route or carrier inside an [element](../../../../glossary.md#term-element) loop.

Direct Java may retain the CPU-private [IR](../../../../glossary.md#term-intermediate-representation-ir), preparation recipe, worker group, and run-owned
[`MemorySegment`](../../../../glossary.md#term-memorysegment) workspace where they express necessary [ownership](../../../../glossary.md#term-ownership-meanings). It must replace generated
partial-body/combine selection only when [finite](../../../../glossary.md#term-finite) direct `int[]` and `long[]` bodies have less code
and proof cost. No permanent selected generated/direct duplicate is allowed.

Reject exactly: a [partition](../../../../glossary.md#partition) other than one ordinary aggregate node; non-`SUM`/`PROD` kind;
`SUM_TO_SHAPE`, [scalar](../../../../glossary.md#term-scalar)/invalid aggregate attributes, or a form outside full/single-/multi-[axis](../../../../glossary.md#term-axis);
wrong [arity](../../../../glossary.md#term-arity), non-distinct values, mismatched type or output [Shape](../../../../glossary.md#shape); [`FLOAT64`](../../../../glossary.md#term-float64), [`FLOAT32`](../../../../glossary.md#term-float32),
[`BFLOAT16`](../../../../glossary.md#term-bfloat16), [`BOOL`](../../../../glossary.md#data-type--datatype), or any non-`INT32`/`INT64` type; unresolved, negative, non-dense, or
non-injective layouts; `MemorySegment` or mixed carriers; zero/overflowing counts; partial count
other than two/four; insufficient workers or minimum domain; inaccessible/misaligned/wrong-sized
workspace; and input/output or workspace/[buffer](../../../../glossary.md#term-buffer) overlap. Every rejection is cold and leaves the
output unchanged. All excluded cases continue through the existing whole-cell route where that
route supports them, otherwise [fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation).

The inspected, distinct later groups remain summary-only and are not implementation permission:

| Order | Current family | Exact current surface | Separate because |
|---|---|---|---|
| E1A | Masked | `SUM`/[`MEAN`](../../../../glossary.md#term-mean); `FLOAT64`/`FLOAT32`/`BFLOAT16`; one removed axis; directional `BOOL` broadcast; dense/general and heap/segment/mixed carriers; exact state | Three boundaries and selected-count floating state. |
| E1B | Advanced | `LOG_SUM_EXP`, [`VARIANCE`](../../../../glossary.md#term-variance), `STANDARD_DEVIATION`, `L1_NORM`, `L2_NORM`; floating types; multi-axis/statistical attrs; keep/remove [dimensions](../../../../glossary.md#dimension); dense/general and all carrier forms | Stable/two-pass/exact-state policies differ. |
| E1C | [Softmax](../../../../glossary.md#softmax--log-softmax)-style | first-class `SOFTMAX`/`LOG_SOFTMAX`; floating types; positive axis; shape-preserving complete slices; dense/general and all carrier forms | Three-pass [normalization](../../../../glossary.md#term-normalization) plus pre-write finite-input [validation](../../../../glossary.md#term-validation). |

## Out of [scope](../../../../glossary.md#term-scope)

Masked, advanced, logarithmic, statistical, [norm](../../../../glossary.md#term-norm), [softmax](../../../../glossary.md#softmax--log-softmax), [log-softmax](../../../../glossary.md#softmax--log-softmax), ordinary whole-cell
aggregates, scans, arg [extrema](../../../../glossary.md#saturation-pole-and-extremum), normalizations, [losses](../../../../glossary.md#loss), [vectors](../../../../glossary.md#term-vector), native routes, public [Model](../../../../glossary.md#term-model)
semantics, shared [Prepare](../../../../glossary.md#prepare)/[Runtime](../../../../glossary.md#term-runtime) contracts, architecture, [modules](../../../../glossary.md#term-module), [Gradle](../../../../glossary.md#term-gradle), benchmarks, and
performance gates. Do not create E1A/E1B/E1C/E2/E3 task specifications. Do not use the [immutable](../../../../glossary.md#term-immutable)
non-passing 0008P evidence root as an [admission](../../../../glossary.md#term-admission) criterion. Legacy is read-only evidence only.

## Architecture references and constraints

Read [the architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([architecture contract glossary](../../../../glossary.md#architecture-contract))</span>, [the current architecture index](../../../../architecture/current-architecture-plan.md),
[Planning Guide](../../../planning-guide.md), [documentation rules](../../../../developer-guide/documentation-rules.md),
[CPU master plan](../master-plan.md), parent [CPU 0009](0009-portable-generated-coverage-closure-checkpoint.md),
and [CPU 0008P](0008p-deterministic-partial-reduction-parallelism.md).

[Planning](../../../../glossary.md#term-planning) selects [CPU](../../../../glossary.md#term-cpu) [ownership](../../../../glossary.md#term-ownership-meanings) only. CPU [Prepare](../../../../glossary.md#prepare) owns [lowering](../../../../glossary.md#lowering), direct-route selection, exact
workspace declaration, and finalization after slot assignment; [Runtime](../../../../glossary.md#term-runtime) invokes [immutable](../../../../glossary.md#term-immutable) prepared
work. Preserve one [`RunState`](../../../../glossary.md#run-state--runstate), [borrowed](../../../../glossary.md#ownership-borrowing-and-lease) caller arrays, [run](../../../../glossary.md#term-run)-owned workspace, cold compatibility
checking, and prepared-region ownership. Direct hot loops preserve the clean Java modular
algorithm without per-[element](../../../../glossary.md#term-element) allocation, boxing, reflection, string/map [dispatch](../../../../glossary.md#term-dispatch),
[synchronization](../../../../glossary.md#term-synchronization), [`Operation`](../../../../glossary.md#operation)/`CompiledNode`, or avoidable virtual dispatch. A shared/public or
[Model](../../../../glossary.md#term-model)-permission change is a stop condition.

## Package impact

Existing [CPU](../../../../glossary.md#term-cpu)-private packages retain `CpuPartialReductionIr` and cold [admission](../../../../glossary.md#term-admission) in `internal.ir`
and `internal.lowering`; `internal.prepare` owns recipe/workspace selection; `internal.executable`
owns typed direct worker/combine calls; and `internal.codegen.emit` loses partial-only generation
only if no selected consumer remains. Add no package or production type. Keep
`CpuPartialReductionExecution` package-private and focused, never a generic reduction service.

## Affected files and maximum [scope](../../../../glossary.md#term-scope)

Expected production paths: `CpuPartialReductionLowering`, `CpuPartialReductionIr`,
`CpuPartitionPreparationPlan`, `CpuPartitionPreparer`, `CpuPartitionFinalizer`,
`CpuPreparedExecutable`, `CpuPartialReductionExecution`, and only partial-specific members in
`CpuGeneratedKernel`/`CpuClassFileKernelGenerator`; review affected package Javadocs. Expected
tests are partial [IR](../../../../glossary.md#term-intermediate-representation-ir)/[lowering](../../../../glossary.md#lowering)/execution, preparer/finalizer/prepared-executable, replacement
direct-route, and inventory/reference tests that name partial [artifacts](../../../../glossary.md#term-artifact).

Modify/create at most 18 [CPU](../../../../glossary.md#term-cpu) production/test/resource paths plus this task and the four
synchronized [planning](../../../../glossary.md#term-planning) records. Add no production type. Stop and replan if direct migration needs
a public/shared path, workspace kind, segment support, generic [carrier](../../../../glossary.md#term-storage-carrier) bridge, over 18 CPU paths,
or a second selected route. Delete partial-specific generation only after automated
no-selected-reference checks prove no remaining selected use.

## Acceptance criteria

- Explicitly decide the complete admitted family: migrate only when finite direct Java reduces
  total code/evidence complexity; otherwise retain generation with a concrete reason and no gate.<span data-glossary-navigation="true"> ([finite glossary](../../../../glossary.md#term-finite))</span>
- A direct route admits exactly current [`INT32`](../../../../glossary.md#data-type--datatype)/[`INT64`](../../../../glossary.md#data-type--datatype) `SUM`/`PROD`, three aggregate forms,
  dense arrays/injective output, positive counts, two/four partials, worker capacity, and current
  minimum-domain facts; it rejects all currently excluded facts before output mutation.<span data-glossary-navigation="true"> ([injective glossary](../../../../glossary.md#term-injective))</span>
- Typed `int[]`/`long[]` loops preserve quotient/remainder bounds and modular arithmetic; the
  caller combines ascending partial states after worker completion. State remains exact, aligned,
  isolated, and owned per active recipe.<span data-glossary-navigation="true"> ([modular arithmetic glossary](../../../../glossary.md#term-modular-arithmetic))</span>
- Cold [binding](../../../../glossary.md#term-binding) exposes typed array calls. [Element](../../../../glossary.md#term-element) loops contain no `Object` [carrier](../../../../glossary.md#term-storage-carrier) branch,
  [`MethodHandle`](../../../../glossary.md#native-numerical-library-and-instruction-architecture), generated lookup, allocation, boxing, reflection, map/string, or generic
  semantic dispatch.<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- Tests cover two types, operations, forms, partial counts, quotient/remainder edges, offsets,
  wraparound, repetition/concurrency, malformed/no-write failures, workspace isolation, and an
  independent modular oracle. Ineligible forms still select the whole-cell route.<span data-glossary-navigation="true"> ([wraparound glossary](../../../../glossary.md#term-wraparound); [concurrency glossary](../../../../glossary.md#term-concurrency); [oracle glossary](../../../../glossary.md#term-test-oracle))</span>
- Preparation/finalization/invocation tests prove direct selection, worker/workspace declarations,
  typed binding, complete-output ownership, and absent generated artifact generation. Automated
  checks prove retired generated partial owners have no selected reference.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- Retire partial-specific generated selection/inventory/evidence only after semantic, invocation,
  relevant conformance review, and no-selected-reference checks pass. Do not remove shared
  class-file generation used by other families.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance))</span>
- A separate clean documentation-focused pass finalizes Javadocs and [planning](../../../../glossary.md#term-planning)/docs/glossary review.

## Tests / [validation](../../../../glossary.md#term-validation)

After executable code stabilizes, [run](../../../../glossary.md#term-run):

```bash
./gradlew :backends:cpu:test \
  --tests io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPartialReductionIrTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPartialReductionLoweringTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPartialReductionExecutionTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizerTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedExecutableTest
```

Add direct-route/no-selected-reference/retirement tests before deletion. Run [backend](../../../../glossary.md#backend) [conformance](../../../../glossary.md#term-conformance)
only if covered behavior changes; no integration suite absent an end-to-end impact. No benchmark,
fresh timing protocol, generated/direct ratio, or [JIT](../../../../glossary.md#term-just-in-time-compilation-jit) claim is a gate. The documentation pass
validates current route [owners](../../../../glossary.md#term-owner) and focused-test evidence, links/anchors/fences, terminology,
glossary impact, exact [scope](../../../../glossary.md#term-scope), and `git diff --check`; no [Javadoc](../../../../glossary.md#term-javadoc) generation is required because
this retained-route decision changes no Java or Javadoc. Repository-wide validation is deferred to
0009G/[CI](../../../../glossary.md#term-continuous-integration-ci).

## Dependencies and follow-up tasks

0009D4 is Complete. E1 retained the generated route, so E1A masked is the sole next summary
frontier. The master plan orders summary-only E1B advanced, E1C [softmax](../../../../glossary.md#softmax--log-softmax)-style, E2 [normalization](../../../../glossary.md#term-normalization),
and E3 [loss](../../../../glossary.md#loss) after E1A. Do not detail any later E task before it becomes the frontier.

## Architecture impact

Expected impact: None. Report any public, [dependency](../../../../glossary.md#term-dependency), shared-contract, or architecture conflict.

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
first [slice](../../../../glossary.md#slice) because its typed-array [carrier](../../../../glossary.md#term-storage-carrier) set, modular permission, two/four-state geometry, and
disabled generated candidate are self-contained. The other families remain ordered summaries.

## Known limitations

No speed, [vector](../../../../glossary.md#term-vector), parallel-scaling, or generated-[bytecode](../../../../glossary.md#term-bytecode) parity claim follows. This task does
not authorize floating partials, segment carriers, arbitrary partial counts, or a silent fallback.
The 0008P evidence root remains historical and non-passing.

## [Validation](../../../../glossary.md#term-validation) evidence

Implementation context `/root` inspected the complete admitted [matrix](../../../../glossary.md#term-matrix-meanings) and its current [owners](../../../../glossary.md#term-owner):
`CpuPartialReductionLowering`, `CpuPartialReductionIr`, `CpuPartitionPreparer`,
`CpuPartitionPreparationPlan`, `CpuPartitionFinalizer`, `CpuPreparedExecutable`,
`CpuPartialReductionExecution`, `CpuPartialReductionEmitter`, and
`CpuClassFileKernelGenerator`, together with their partial [IR](../../../../glossary.md#term-intermediate-representation-ir)/[lowering](../../../../glossary.md#lowering)/execution,
preparer/finalizer/prepared-executable, and generated-[artifact](../../../../glossary.md#term-artifact) tests. The route decision is to
retain the bounded generated partial implementation. It is not a performance decision: its one
partial emitter already contains the complete four typed modular fold bodies and four typed
ordered combines for [`INT32`](../../../../glossary.md#data-type--datatype)/[`INT64`](../../../../glossary.md#data-type--datatype) × `SUM`/`PROD`; form and partial count are [immutable](../../../../glossary.md#term-immutable) IR facts.
A direct replacement would additionally need cold direct-route [identity](../../../../glossary.md#term-identity-meanings) and selection, a typed
prepared invocation [carrier](../../../../glossary.md#term-storage-carrier), finalizer wiring, replacement semantics and malformed-[binding](../../../../glossary.md#term-binding)
coverage, and retirement/no-selected-reference proof. That is more implementation and
verification work than retaining the existing self-contained generated realization. The current
generated candidate remains unselected behind 0008P's historical [fail-closed](../../../../glossary.md#term-fail-closed) evidence; that
historical evidence is neither rerun nor used as this decision's [admission](../../../../glossary.md#term-admission) gate, and the
established whole-cell route remains the fallback.

The implementation context ran the specified focused [CPU](../../../../glossary.md#term-cpu) command after the no-code route decision:
110 tests passed with zero failures, errors, or skips across the six requested
IR/lowering/execution/preparer/finalizer/prepared-executable classes. This documentation-focused
context reused that evidence and did not rerun it because no executable Java changed afterward.
It independently checked the current lowering, IR, preparation, finalization, executable, emitter,
and focused tests; local Markdown links, anchors, and fences; synchronized E1/0009/master/roadmap
statuses and dependencies; terminology and glossary impact; the exact four changed [planning](../../../../glossary.md#term-planning) paths;
and `git diff --check`. No benchmark, repository-wide, [Javadoc](../../../../glossary.md#term-javadoc), [conformance](../../../../glossary.md#term-conformance), or integration command
was [run](../../../../glossary.md#term-run): this is a planning-only decision with no Java or public Javadoc change.

## Implementation notes

No production or test edit is warranted by the retained-route decision. E1A [masked reduction](../../../../glossary.md#masked-reduction) is
the sole next summary frontier; E1B--E3 remain ordered later summaries.

## Completion summary

- Completed changes: Finalized the retained generated-route decision and synchronized its current
  selection, status, dependency, and frontier evidence.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>
- Files changed or created: This task and the three synchronized [planning](../../../../glossary.md#term-planning) records only.
- Tests and validation: Reused the implementation context's focused six-class CPU result (110
  tests; zero failures, errors, or skips); this context validated the current route owners/tests,
  Markdown links/anchors/fences, terminology, exact scope, and <span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [CPU glossary](../../../../glossary.md#term-cpu); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check`.
- Documentation-agent review: This clean documentation-focused context finalized the planning
  diff without rerunning Java because no executable behavior changed.
- Documentation impact: Planning records updated; no public documentation change is needed.
- [Javadoc](../../../../glossary.md#term-javadoc) review: No Javadoc changed because no Java [API](../../../../glossary.md#term-application-programming-interface-api) or implementation contract changed.
- Glossary impact: No glossary change; this decision introduces no reusable term or changed term
  meaning.
- Unresolved issues: The generated partial route remains intentionally unselected until a separate
  trusted-evidence mechanism is scoped; E1A masked remains the sole next summary frontier.
- Follow-up required: None for E1.

Status: Complete
