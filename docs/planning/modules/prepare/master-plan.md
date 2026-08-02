# Prepare Master Plan

## Goal

Define and validate the shared transition from immutable compile artifacts to prepared runtime state.

## Architecture references

- [Architecture contract](../../../../ARCHITECTURE.md)
- [Module boundaries](../../../architecture/module-boundaries.md)
- [Dependency rules](../../../architecture/dependency-rules.md)

## Scope

- prepare context
- backend partition preparer contract
- prepared partitions
- coverage, memory, and schedule validation
- a future narrow opaque-candidate orchestration boundary for model tuning, only after compiler,
  planning, concrete backend, and cache contracts are stable
- explicit workload-cache and model-plan artifact load/fallback handoff before runtime

## Out of scope

- concrete CPU, Metal, or CUDA lowering
- kernel selection
- backend executable and storage implementations
- backend-specific candidate vocabulary or search-space generation
- tuning measurement, comparison, and persistence algorithms

## Module invariants

- Shared prepare coordinates contracts and validation.
- Concrete backends own deterministic analysis, lowering, route selection, and executable
  construction.
- Backend analysis declares exact shared resource requirements before slot assignment.
- Backend finalization constructs executable state only after shared slot assignment and cannot
  revise the selected route or declarations.
- Shared prepare exposes complete candidates opaquely and does not interpret private backend
  fields.
- Compatible cache hits and safe heuristics can prepare correct work without a tuning search.
- Model-autotuning results become explicit prepared or cache state before runtime, never hidden
  global state.

## Allowed dependencies

- modules/runtime
- modules/planning
- modules/compiler
- modules/config
- modules/backend-contract
- modules/trace

## Forbidden dependencies

- concrete backend implementations

## Package structure

```text
io.github.pho001.synaptik.prepare/
  analysis/  public partition-scoped analysis request, opaque backend plan, exact resource
             declarations, and backend analyzer collaboration
  <root>     finalized prepared-partition result plus current complete graph-preparation,
             schedule-assembly, and validation contracts
```

Task 0001 opens only `analysis`. The request projects partition-scoped Model and Planning facts
plus one typed opaque backend input that a concrete backend uses for resolved bindings,
target/backend capabilities, configuration, and compatible cached decisions. It never exposes
`CompileArtifacts` or another Compiler-owned type. Later root contracts remain unimplemented.

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| 0001 | [Backend partition analysis and resource declaration](tasks/0001-backend-partition-analysis-and-resource-declaration.md) | Complete | Compiler 0006; Planning 0006; Runtime 0001; ADR 0010 | Defines the analysis-side Prepare projection, typed backend analyzer, opaque selected plan, and exact buffer/workspace declarations without assigning slots or finalizing executables. |
| 0002 | [Backend partition finalization handoff](tasks/0002-backend-partition-finalization-handoff.md) | Complete | 0001; Runtime 0002–0004 | Assigns deterministic conservative shared slots across the complete ordered analyses, retains exact source associations, and finalizes each typed backend plan into the minimal prepared partition/executable association. |
| 0003 | [Prepare orchestration and validation](tasks/0003-prepare-orchestration-and-validation.md) | Complete | 0001–0002; Compiler 0006; Planning 0006; Runtime 0002–0014 | Composes exact compile projection, typed backend analysis/finalization, initialized constant representations, prepared-memory assignment, complete schedule assembly/validation, and final prepared execution without concrete backend logic. |

## Milestones

- Prepare contracts
- Partition coverage validation
- Prepared memory and schedule validation

## Current status

Complete through
[task 0003](tasks/0003-prepare-orchestration-and-validation.md) after completion of
[task 0001](tasks/0001-backend-partition-analysis-and-resource-declaration.md). The current public
`prepare.analysis` package contains the exact typed projection, backend input/plan marker roles,
buffer/workspace requirement family, immutable analysis result, and backend preparer
collaboration selected by ADR 0010. Its focused and final implementation validation passed four
suites and 11 tests with no failures, errors, or skips. The separate documentation pass finalized
the production/package Javadocs, backend guide, focused architecture status, glossary, and
planning records without changing executable Java or repeating those successful tests.

Task 0001 remains deliberately analysis-only. `PrepareContext` accepts fully static partition
facts and one backend-specific immutable input object carrying target capabilities,
configuration, and compatible cached decisions. Its nodes match exact partition order; every
node-referenced value resolves to a unique projection; every projected value has one
descriptor-matching logical requirement; and exact-typed logical splats are limited to projected
graph inputs. The result retains the selected backend plan opaquely and declares exact
non-negative byte sizes plus positive power-of-two alignments.

Runtime 0002–0004 and
[task 0002](tasks/0002-backend-partition-finalization-handoff.md) are Complete. The Prepare root
now exposes exact declaration-to-slot assignments, typed finalization input and collaboration,
and the minimal `PreparedPartition(partition, executable)` association. Its package-private
complete-set handoff validates all ordered sources before assignment, uses first-declaration
buffer ordering with maximum geometry, gives every workspace declaration a distinct slot,
constructs every finalization before invoking a backend, and rejects an executable that does not
retain the exact shared memory plan.

Task 0002 deliberately adds no public orchestration, physical allocation, closeable prepared
resource, execution, or schedule. Complete Runtime 0005 defines the schedule consumer without a
distinct `PreparedUnit`; Complete Runtime 0006 supplies the smallest final prepared-execution
aggregate and exact-plan consistency contract; Runtime 0007–0010 complete representation
creation, transfer, publication, and runner recipe boundaries. Complete task 0003 implements the
smallest explicit complete-graph orchestration: positionally supplied typed backend
collaborators, exact compile projection, existing finalization handoff, one Prepare-owned schedule
assembler, one generic Runtime initialized-buffer origin for backend-materialized logical splats,
structural source/execution/publication validation, and one final `PreparedExecution`. It adds no
concrete backend, Engine facade, physical work in Prepare, or lookup. Its implementation pass
passed the combined Runtime/Prepare suites, its independent documentation pass finalized all
affected Javadocs and explanatory text, and the Prepare milestone checkpoint passed.

The implementation context's final Prepare module run passed 7 suites and 22 tests with no
skips, failures, or errors. The clean documentation pass finalized all six new production/package
Javadocs, the five explanatory documents, and synchronized planning records without changing
executable Java or repeating the successful tests. Prepare Javadoc, the Java 26 backend-finalizer
example, nine-file Markdown validation, exact public/package-private shape, mechanism, exact
18-path scope, unchanged architecture/build boundaries, status, and whitespace gates passed.

## Open questions

- Dynamic dimension binding remains deliberately unresolved. Complete task 0003 preserves the
  current fully-static `PrepareContext` boundary and fails before backend analysis when it cannot
  project exact geometry.
- A zero-node pass-through graph cannot currently obtain byte geometry from a backend analysis.
  Task 0003 fails closed on a requested value with no prepared buffer assignment instead of
  inventing a shared allocation rule.
- The smallest opaque candidate and artifact-lifecycle boundary waits for stable compiler,
  planning, backend, engine, and persistence consumers. No Java declaration or file format is
  selected here.

## Decisions made

- The implementation must follow the current architecture contract.
- Legacy code is capability evidence only; new implementation is written from scratch.
- Compiler, planning, and concrete backends generate candidates for decisions they own. Shared
  prepare coordinates complete candidates and validation without interpreting backend knobs.
- Cache incompatibility or corruption must fail closed to safe heuristics or an explicit miss;
  runtime never performs cache lookup or mutation.
- `BackendPartitionPreparer`, `PrepareContext`, and `PreparedPartition` belong to Prepare.
  `modules/backend-contract` remains closed and is not reopened for a prepare service.
- A backend-facing Prepare contract must not expose `CompileArtifacts` because concrete backends
  do not have an architecture-approved Compiler dependency.
- Backend analysis produces exact buffer/workspace byte-size and alignment declarations, not
  physical storage, resource handles, allocation, or per-run binding.
- Workspace identity is local to one analysis result until Runtime assigns a stable slot; every
  initial declaration receives a distinct slot.
- Compatible cached tuning decisions are explicit analysis inputs. Analysis performs no search
  or measurement and is deterministic from its complete inputs.
- Prepare 0002 assigns buffer slots in first-declaration order across the complete ordered
  analyses, combines repeated value declarations with maximum size/alignment geometry, and
  assigns every workspace declaration its own slot in declaration order.
- `PreparedPartition` retains only the exact planned partition and finalized executable.
  Complete Runtime 0005 establishes that list position plus `PreparedExecutable` is sufficient;
  no distinct `PreparedUnit` is planned for current scheduling.
- Prepare 0003 uses one explicit `PreparedScheduleAssembler` after complete backend finalization.
  It is a Prepare-owned recipe-construction seam supplied by future composition wiring, not a
  backend-facing Compiler leak, global registry, discovery mechanism, or Runtime concern.
- Prepare 0003 validates bindable-input, execution, representation-coordinate, and publication
  coverage before returning the exact Runtime `PreparedExecution`; concrete backends still own
  creators, transfers, executables, and their physical compatibility.
- Compile-time logical splats use the task-0003 `InitializedBuffer` origin: the backend creator
  materializes a fresh run-owned representation and Runtime records only that its contents start
  valid. Runtime receives no `ScalarValue`, graph value, or backend identity.

## Risks

- Accumulating concrete backend logic in the shared prepare layer.
- Turning an opaque orchestration boundary into a generic parameter language or central knob
  registry.
- Hiding a Compiler dependency inside a nominally Prepare-owned backend collaboration.
- Letting finalization change route choice or add resources after slot assignment.
- Letting the schedule assembler become a backend registry, tuning extension point, or substitute
  for Prepare-owned complete-result validation.

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and follow [the planning guide](../../planning-guide.md).
