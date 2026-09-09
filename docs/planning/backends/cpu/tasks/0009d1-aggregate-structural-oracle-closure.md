# Task 0009D1: Aggregate Direct-Java Migration and Generated-Retirement

## Status

Ready

## Goal

Replace the unfinished generated-aggregate structural-oracle program with the portable CPU
aggregate implementation that it was trying to validate: finite, direct, optimized, statically
typed Java kernels selected during prepare. This is a local CPU implementation/evidence-plan
change, not an architecture or module-boundary change.

## Scope

Migrate the current `CpuAggregateIr` frontier: SUM, PROD, MEAN, AGGREGATE_MIN,
AGGREGATE_MAX, ALL, ANY, and SUM_TO_SHAPE for their currently admitted types, layouts, carrier
patterns, scalar ranges, and caller-owned parallel ranges. Retain the current Model semantics,
lowering facts, validation, workspace contract, and exact rejection of integral MEAN.

Production ownership is deliberately narrow:

- `io.github.pho001.synaptik.backend.cpu.internal.executable.CpuAggregateDirectKernel` is a new
  package-private owner of the finite direct loops. It exposes package-private typed static entry
  methods, not an `Object` carrier loop: `execute{F64,F32,Bf16,I32,I64,Bool}{Dense,General}{Array,
  Segment,Mixed}(typed input, typed output, MemorySegment workspace when required, long[] geometry,
  long start, long end)`. Inapplicable type/kind combinations do not receive an entry.
- Its nested package-private `Binding` record holds the cold-selected exact typed entry and its
  carrier/address regime. `CpuPartitionPreparer` selects that binding from `CpuAggregateIr`, the
  already validated access bindings, and the carrier pattern during prepare.
- `CpuPreparedExecutable` owns invocation of the selected binding. It supplies one complete
  output-cell `[start,end)` range for scalar execution or gives disjoint complete-cell ranges to
  the existing worker group. It does not reselect a method, inspect storage, or dispatch per
  element.

Dense entries use primitive ordinal/address progression; general entries use the existing
lowered long-address geometry. Array, `MemorySegment`, and mixed patterns remain separate typed
methods. Floating SUM/MEAN/PROD preserve the admitted exact-state workspace and one represented
result conversion; extrema preserve first-logical-NaN and signed-zero behavior; BOOL preserves
canonical ALL/ANY values; integral arithmetic preserves its existing width and wrap semantics.
Vector entries are optional only where a type and algorithm have a useful fixed vector form; they
are not a prerequisite for this migration.

Use the existing clean-Java aggregate oracle and semantic fixtures as the first algorithmic source.
The read-only `legacy/pre-rewrite` reduction classes (including `CpuSumKernel`, `CpuCumSumKernel`,
`StorageAwareReductionKernel`, and their tests) may be inspected for observable cases, traversal,
and special-value tests only. Do not copy their source, packages, abstractions, dependencies,
runtime/storage coupling, or shortcuts.

After direct routes have semantic and invocation coverage, retire the selected generated aggregate
emitter route and only its now-unreferenced schema/inventory/evidence rows. Retire the incomplete
scaffolding in the same cohesive task: `AggregateTemplateOracle`, `AggregateClassFileGraph` and
its test, `FiniteAggregateFlow`, `LegacyFiniteAggregateFlow` nested in
`CpuAggregateStructuralOracleTest`, and `aggregate-clean-method-bindings.tsv`. Do not delete the
clean-Java oracle, semantic fixtures, or generated aggregate tests before their direct-Java
replacements cover their useful cases; delete or reduce them only after that proof.

## Out of scope

- Scan, ordering, fold, general reductions, normalization, loss, pooling, attention, batch norm,
  MATMUL, convolution, pointwise fusion, and completed 0009A--C/D1A evidence.
- A universal generated-versus-clean-Java structural verifier, generated-vs-Java benchmark or
  ratio gate, architecture update, module-boundary change, public API change, Gradle change, or
  broad inventory rewrite.
- Parallel permanent generated and direct aggregate implementations. A temporary overlap is only
  allowed while selected direct routes are being validated and must have an explicit deletion
  condition.

## Architecture references and constraints

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md): prepare owns lowering and kernel selection;
  Runtime invokes prepared work only; prepared recipes are immutable; run-owned state is isolated;
  generated code, when retained, still needs a clean Java review oracle.
- [CPU master plan](../master-plan.md), parent [CPU 0009](0009-portable-generated-coverage-closure-checkpoint.md),
  parent [CPU 0009D](0009d-aggregate-scan-ordering-fold-structural-oracle-closure.md), and the
  [Planning Guide](../../../planning-guide.md).
- Keep dispatch, carrier selection, geometry validation, and range partitioning cold. Element
  loops must not allocate, box, reflect, synchronize, string/map-dispatch, or make avoidable
  virtual/interface calls.

## Package impact

Only the existing CPU backend internal packages change: lowering continues to own
`CpuAggregateIr`; `internal.prepare` owns cold route binding; `internal.executable` owns direct
execution. No public package, module dependency, or architecture contract changes.

## Affected files and maximum scope

Expected production owners are `CpuPartitionPreparer`, `CpuPreparedExecutable`, the generator
selection point, `CpuAggregateEmitter`, and new `CpuAggregateDirectKernel`. Expected focused tests
cover lowering/prepare, direct aggregate execution, invocation/range binding, and existing semantic
fixtures. Generator schema, generated inventory, disposition/evidence resources, and retired
test-only scaffolding change only for aggregate rows after direct routing passes. Planning changes
are limited to this task, 0009D, 0009, CPU master plan, and roadmap. Stop if this requires another
family, a public/shared contract, or more than 18 CPU production/test/resource paths.

## Acceptance criteria

- Every currently advertised aggregate form is selected as a direct typed Java route or retains an
  exact rejection. No aggregate generated row remains selected after its replacement is complete.
- Prepare binds one exact static entry for datatype, array/segment carrier pattern, dense/general
  regime, operation form, workspace requirement, and range ownership. The runtime path has no
  per-element route/carrier/type dispatch.
- Direct routes reproduce semantic and canary coverage for empty and nonempty domains, nonzero
  starts/tails, SUM_TO_SHAPE, axis/keep dimensions, represented BFLOAT16 values, integral width,
  exact floating result finalization, NaN/signed-zero extrema, BOOL identities, and unsupported
  integral MEAN.
- Scalar and worker execution cover the same complete output-cell ownership semantics and detect
  overlap, untouched canaries, and invalid selected bindings.
- Focused source/hot-loop inspection records absence of allocation, boxing, reflection, string/map
  dispatch, synchronization, and per-element virtual dispatch. No benchmark gate is required.
- Generated aggregate emitter/schema/inventory/evidence entries and the named incomplete oracle
  scaffolding are removed only after replacement semantics, conformance, invocation, and
  no-selected-reference checks pass. Useful tests/oracles remain until then.
- Javadocs for changed private execution/prepare contracts are finalized by a separate clean
  documentation pass; public/API, architecture, glossary, Gradle, conformance, and integration
  documentation receive reasoned no-change conclusions unless the implementation proves otherwise.

## Tests / validation

Run focused CPU lowering, prepare, direct-execution, generated-retirement, and semantic/canary
tests; add route-selection tests that prove the generated aggregate route is no longer selected.
Run the affected CPU module suite once after the final executable changes. Inspect direct source and
compiled call sites for the stated hot-loop hygiene. Validate exact direct/rejected aggregate
inventory accounting, selected-route references, retirement of stale schema/evidence rows, and
the documented legacy boundary. Run `git diff --check`; do not run a generated-vs-Java benchmark
or require a ratio.

## Dependencies and follow-up tasks

Completed 0009A--C and 0009D1A remain historical evidence and are not reopened. D2 scan follows
only after D1 is complete; D3 ordering follows D2; D4 fold follows D3. CPU 0009E follows D4.

## Architecture impact

Expected impact: None. If direct binding cannot fit the existing prepare/executable boundary without
a public, shared-contract, dependency, or module-boundary change, stop and report that conflict
without editing architecture documents.

## Implementation prompt

```text
Implement only CPU 0009D1. Replace selected generated aggregate execution with finite direct typed
Java entries under internal.executable, selected during prepare and invoked over caller-owned
ranges. Preserve current aggregate semantics and rejections. Reuse current clean-Java algorithms
first; inspect legacy/pre-rewrite only as read-only behavioral/test evidence and never copy it.
Retire generated aggregate artifacts and incomplete structural scaffolding only after direct route
semantics, invocation, conformance, and no-reference checks pass. No benchmark gate, universal
structural verifier, architecture change, commit, or push. Hand the final diff and executable
evidence to a separate documentation-focused context.
```

## Local decisions

Direct Java is preferred here because the aggregate family is static, bounded, and currently has a
usable clean-Java algorithm, while bytecode structural proof is the remaining cost. This does not
invalidate retained generation where fusion or specialized compute makes it materially useful.

## Known limitations

Fresh performance measurements are optional final profiling, not a correctness or migration gate.
Existing performance facts, including NON_PASSING facts, remain historical evidence and are not
silently promoted or erased.

## Validation evidence

Planning replan only: inspected the current aggregate lowering, IR, emitter, generator,
preparer/executable route boundary, clean-Java oracle, structural scaffolding, semantic fixtures,
and read-only legacy reduction/scan paths. No Java, test, resource, schema, or behavior was changed
by this planning task.

## Implementation notes

None; implementation has not started.

## Completion summary

Status: Ready
