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
  <root>     later orchestration, finalized prepared-partition result, and validation contracts
```

Task 0001 opens only `analysis`. The request projects partition-scoped Model and Planning facts
plus one typed opaque backend input that a concrete backend uses for resolved bindings,
target/backend capabilities, configuration, and compatible cached decisions. It never exposes
`CompileArtifacts` or another Compiler-owned type. Later root contracts remain unimplemented.

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| 0001 | [Backend partition analysis and resource declaration](tasks/0001-backend-partition-analysis-and-resource-declaration.md) | Complete | Compiler 0006; Planning 0006; Runtime 0001; ADR 0010 | Defines the analysis-side Prepare projection, typed backend analyzer, opaque selected plan, and exact buffer/workspace declarations without assigning slots or finalizing executables. |
| 0002 | Backend partition finalization handoff | Draft | 0001; Runtime 0002–0004 | Finalize one analysis against assigned Runtime slots and produce the smallest prepared partition/executable association. |
| 0003 | Prepare orchestration and validation | Draft | 0001–0002; Runtime prepared-memory and schedule contracts | Compose coverage, prepared-memory, schedule, and final prepared-execution validation without concrete backend logic. |

## Milestones

- Prepare contracts
- Partition coverage validation
- Prepared memory and schedule validation

## Current status

In progress after completion of
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

Runtime 0002 is now the next cross-area planning frontier because the declaration producer exists.
Prepare 0002 remains Draft and depends on Runtime 0002–0004 slot, access, and executable
contracts. No Prepare 0002 detailed specification exists.

## Open questions

- The later concrete binding contract for dynamic dimensions remains deferred. Task 0001 fails
  closed rather than inventing it.
- Runtime 0002 must define the exact `WorkspaceSlot`, assignment aggregate, and prepared-memory
  consumer before Prepare finalization becomes Ready; its former producer blocker is resolved.
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

## Risks

- Accumulating concrete backend logic in the shared prepare layer.
- Turning an opaque orchestration boundary into a generic parameter language or central knob
  registry.
- Hiding a Compiler dependency inside a nominally Prepare-owned backend collaboration.
- Letting finalization change route choice or add resources after slot assignment.

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and follow [the planning guide](../../planning-guide.md).
