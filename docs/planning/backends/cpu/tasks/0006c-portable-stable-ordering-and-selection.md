# Task 0006C: Portable Stable Ordering and Selection

## Status

Complete

## Goal

Add the next executable CPU frontier after completed CPU 0006B2: exactly one fully static,
resolved-layout current Model `SORT`, `ARGSORT`, or `TOP_K` occurrence through the existing
portable generated-kernel route.

The implementation preserves Model's stable logical-axis ordering exactly: non-NaN values use the
requested direction, negative zero precedes positive zero ascending and reverses descending, every
NaN remains after every non-NaN in both directions, and equal values or NaNs retain increasing
original logical-axis index. `SORT` writes values, `ARGSORT` writes INT64 logical indices, and
`TOP_K` writes values then INT64 indices from one two-output occurrence.

This task owns a deterministic portable scalar baseline, bounded slice-parallel execution, exact
scratch, multi-store preparation/invocation, and independent reference evidence. It does not add
Model semantics, general DAG decomposition, fusion, reductions, native routes, or autotuning.

## Scope

- Admit exactly one CPU-owned node with one input and either one `SORT` values output, one
  `ARGSORT` INT64 output, or ordered `TOP_K` outputs `[values, indices]`, with their exact current
  `SortAttrs` or `TopKAttrs` signatures.
- Require fully static input/output Shapes, resolved non-negative-offset/non-negative-stride
  layouts, exact Model Shape rules, and distinct injective writable outputs.
- Support FLOAT64, FLOAT32, BFLOAT16, INT32, INT64, and BOOL. Values outputs copy the exact
  represented input element; indices are INT64 logical-axis coordinates.
- Add focused CPU-private `CpuOrderingIr`, `CpuOrderingLowering`, and `CpuOrderingEmitter` owners.
  Structural identity includes family, represented type, direction/top-K flags, boundary roles,
  types/ranks, access plans, scratch policy, and output count. Concrete extents, offsets, stride
  magnitudes, axis, K, ranges, slots, and carriers remain cold when they do not shape bytes.
- Use deterministic stable ordering over logical indices without boxing, comparator/object arrays,
  per-element allocation, reflection, or runtime semantic dispatch. A permitted baseline is a
  bottom-up stable merge over two primitive INT64 index regions per worker range.
- Declare one exact run-owned scratch before shared assignment. Checked sizing must provide
  disjoint per-range regions sufficient for the selected stable algorithm; scratch never escapes
  the run and is accessible by every selected worker.
- Range work over complete independent logical-axis slices and never split one slice. Parallel-
  scalar is legal only for disjoint slices, outputs, and scratch; otherwise use scalar. Vector
  sorting is out of scope.
- Support dense, offset, positive-strided, and zero-strided input views; injective non-dense
  outputs; heap arrays; native-order `MemorySegment`; and compatible mixed carriers. Ordering and
  indices follow logical coordinates, never physical traversal.
- `SORT`/`ARGSORT` preserve axis extent. `TOP_K` selects the first `k` pairs of the complete stable
  order requested by `largest`; `sorted == true` retains it, while `sorted == false` orders the
  selected pairs by increasing original logical-axis index.
- Accept exact empty behavior. Empty selected SORT/ARGSORT axes produce empty outputs; TOP_K
  requires `0 <= k <= extent`, so an empty selected axis accepts only zero. Empty unselected
  dimensions and `k == 0` submit no generated or worker work. Scalar inputs remain unsupported.
- Reject every input/output and output/output physical overlap before scratch mutation, output
  write, or worker submission.
- Extend capability, lowering, preparation, assignment/finalization, generated entry, binding,
  reference, cache compatibility, package contracts, and tests only as this family requires.
- Advance generated compatibility from schema 17 to schema 18 with no migration reader.
- After Java behavior stabilizes, hand the uncommitted diff and CPU-test evidence to a distinct
  clean documentation-focused context for Javadocs, package summaries, CPU guide, glossary, task
  evidence, master plan, and roadmap.

## Exact ordering contract

- Each logical selected-axis slice is independent.
- FLOAT64/FLOAT32/BFLOAT16 non-NaNs use numerical order; ascending places negative infinity first
  and positive infinity last, and descending reverses non-NaNs.
- Negative zero is strictly before positive zero ascending and strictly after it descending.
- All NaNs form one final class after all non-NaNs in either direction. Sign, payload, and
  signaling state do not affect comparison, but values outputs preserve selected represented bits.
- INT32/INT64 use signed order; BOOL uses `false < true` ascending.
- Equal values and multiple NaNs retain increasing logical-axis index; stable merge chooses the
  left/logically earlier candidate on equality.
- ARGSORT and TOP_K indices are zero-based axis coordinates, not flattened storage offsets.

## Out of scope

- multiple/mixed nodes, fusion, general partition DAGs, materialized DAG splits, or representation
  reuse beyond this scratch
- unstable/custom ordering, algorithm/NaN options, radix/network/parallel merge, vector sorting,
  approximate top-K, dynamic Shapes/K, or runtime algorithm choice
- RNG/dropout, reductions/scans, normalization, heavy families, native routes, tuning, relaxed
  numerics, or persistence-policy changes
- new Model/compiler/shared Prepare/Runtime/public Config/Trace behavior, architecture/dependency/
  build changes, negative physical strides/offsets, unresolved layouts, device transfer, or Engine
- new backend-conformance/integration infrastructure; executable conformance closes at CPU 0009
  unless an applicable current reusable harness is found

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [CPU master plan](../master-plan.md)
- [CPU 0006B2 overlap fold](0006b2-portable-overlap-fold.md)
- [Model sort/argsort](../../../modules/model/tasks/0019c-sort-and-argsort.md)
- [Model top-K](../../../modules/model/tasks/0019c1-top-k-values-and-indices.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [CPU backend guide](../../../../backend-guide/cpu-backend.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- Model alone owns operation identity, attributes, Shape/output-slot rules, and ordering meaning.
  CPU independently revalidates and owns only its algorithm, IR, scratch, artifact, and invocation.
- CPU analysis declares exact resources before CPU-blind shared assignment; finalization validates
  assignments before realizing exactly one generated artifact.
- Runtime receives one direct prepared invocation and no graph, Model operation, comparator, or
  semantic-dispatch object.
- Work remains in `backends/cpu`, adds no dependency, and preserves package direction.
- Capability is no broader than lowering, assignment, binding, multi-output writes, scratch, and
  generated execution can complete.
- Stop if current Model source conflicts or exact behavior needs architecture, another module,
  shared-contract, build, or unresolved numerical-policy changes.

## Package impact

No package is added, removed, or moved.

- `internal.ir.CpuOrderingIr` owns immutable ordering/type/access/scratch/output identity.
- `internal.lowering.CpuOrderingLowering` owns one-node revalidation, slice geometry, layouts,
  scratch sizing, and boundaries.
- `internal.codegen.emit.CpuOrderingEmitter` owns generated stable ordering and represented
  load/store mechanics.
- Focused immutable geometry may be nested under lowering. Add no generic comparator, manager,
  utility package, registry, or execution facade.

## Affected files

Expected production work is the three focused new types plus directly necessary CPU capability,
IR permits/package docs, lowering dispatch/package docs, generator/schema/specialization,
preparation/finalization, executable/workspace, reference, route, and inventory seams.

Expected tests cover ordering IR/lowering/generated execution, preparation/finalization,
executable, capability, schema/cache, carriers/layouts, independent differential reference, and
package inventory using existing fixtures.

Expected documentation/planning is this task, CPU master plan, roadmap, CPU backend guide,
glossary, and affected package/Javadocs. Architecture/API guides, other modules, Gradle,
architecture tests, and empty conformance/integration harnesses are expected no-change.

## Maximum scope

At most 48 paths: 23 production/package, 18 CPU test, and 7 documentation/planning. Stop before a
49th path or any other module, build file, architecture contract, or shared Prepare/Runtime change.

## Acceptance criteria

- Capability/lowering accept exactly the three one-node families and six-type matrix under static
  resolved constraints; all unsupported work fails closed.
- Results exactly cover direction, stability, NaN-last, signed zero, infinity, BOOL, duplicates,
  empty/singleton/K, and TOP_K sorted/unsorted semantics.
- Boundaries are `[input, output]` for SORT/ARGSORT and `[input, values, indices]` for TOP_K; one
  TOP_K executable performs both stores.
- Exact scratch declaration/assignment/slicing/accessibility/overflow/cleanup and zero-work rules
  are tested, with no per-element heap allocation.
- Scalar and parallel-scalar are bitwise identical; ranges share no output/scratch coordinates.
- Heap/native/mixed carriers and dense/offset/strided/zero-stride input plus non-dense injective
  output layouts agree with an independent reference.
- All overlaps fail before mutation or submission.
- Schema 18 and structural type/carrier/output distinctions are deterministic and collision-tested;
  one artifact is realized.
- Existing pointwise, movement, indexing, scatter, and fold behavior remains unchanged.
- A separate documentation context finalizes docs and no-change conclusions before Complete.

## Tests / validation

During implementation run focused tests, then one final executable suite:

```bash
./gradlew :backends:cpu:test
```

The clean documentation pass reuses that evidence and runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
```

It also checks links/anchors/fences/newlines, changed-path scope, generated Javadoc, and task/master/
roadmap coherence. Repository-wide and executable conformance validation are deferred to CPU 0009
or CI unless a repository-wide contract unexpectedly changes.

## Dependencies

- CPU 0005A–0006B2: Complete.
- Model 0019C and 0019C1 ordering semantics: Complete.
- Portable Class-File route, typed carriers, workers, workspace, multi-boundary declarations, and
  artifact store: Complete.

## Follow-up tasks

- CPU 0006D remains next Draft and owns explicit-state RNG/dropout.
- CPU 0007 owns reduction/scan/statistics/normalization.
- CPU 0008A–0008D own later decomposition/fusion/profitability/representation reuse.
- CPU 0009 owns portable closure and executable conformance.

## Architecture impact

Expected impact: None.

## Implementation prompt

```text
Implement Synaptik CPU task 0006C exactly from its Ready specification. Do not use GSD. Read
AGENTS.md, ARCHITECTURE.md, current architecture plan, documentation rules, planning guide,
roadmap, CPU master plan, task 0006C, completed CPU 0006B2, Model 0019C/0019C1, and every affected
source/test seam in full before editing. Work on the existing uncommitted 0006C planning change.

Deliver one-node static resolved-layout generated SORT, ARGSORT, and two-output TOP_K for all six
types, exact stable Model order, represented value copies, INT64 indices, deterministic unsorted
TOP_K, bounded per-range scratch, scalar/slice-parallel parity, multi-output preparation/binding,
overlap rejection, independent reference evidence, and schema 18. Preserve exclusions and the
48-path ceiling. Stop on architecture/shared-contract/cross-module/semantic/numerical conflicts.

Run focused tests and one final ./gradlew :backends:cpu:test after executable Java stabilizes. Do
not commit/push, finalize explanatory documentation, or mark Complete. Return changed paths, exact
evidence, unresolved issues, documentation handoff, and Status: Complete/Incomplete.
```

## Local decisions

- CPU keeps Model ordering meaning separate from realization: Model owns axis-wise stable
  NaN-last/signed-zero order and output roles, while CPU owns primitive-index merge scratch,
  scalar/slice-parallel orchestration, carriers, artifact identity, and cold binding.
- TOP_K remains one three-boundary unit and one generated artifact. Its boundary order is input,
  represented values, then INT64 logical-axis indices; one invocation performs both stores.
- Every ordering plan declares one exact run-owned workspace. Each selected range receives two
  disjoint axis-extent INT64 regions, and parallel work owns complete logical-axis slices.
- Unsorted TOP_K selects from the complete stable value order, then deterministically orders the
  selected pairs by increasing original logical-axis index.

## Known limitations

- Coverage is one fully static resolved-layout occurrence. Dynamic Shapes/layouts, scalar input,
  negative physical offsets/strides, multi-node fusion, custom or unstable ordering, vector or
  native sorting, approximate selection, and runtime algorithm selection remain unsupported.
- The scalar and slice-parallel CPU realizations are bitwise identical, but this task makes no
  cross-backend bitwise or performance claim.
- `testing/backend-conformance` has no reusable executable cross-backend harness at this frontier.
  Executable conformance remains planned for CPU 0009; this task does not invent a cross-module
  harness. Repository-wide validation remains deferred to that checkpoint or CI.

## Validation evidence

- Implementation context initially ran the single final `./gradlew :backends:cpu:test` after the
  original executable Java stabilized: 257 tests, zero failures, zero errors, and one skip. The
  first documentation context `019ffbb7-f877-73f3-afc3-5cbd8b6f593d` reused that evidence. A later
  coordinator review found that one-output SORT/ARGSORT overlap validation ranged complete buffer
  bindings by slice ordinals, which could miss physical overlap outside that ordinal prefix. Fix
  context `019ffbc8-fb51-7a73-9504-9a3929cefe58` changed the overlap check to use complete boundary
  bindings, added the multidimensional SORT/ARGSORT regression, and ran the authoritative final
  `./gradlew :backends:cpu:test`: 258 tests, zero failures, zero errors, and one skip. Documentation
  re-review context `019ffbcb-4c30-7e53-8e4d-9474f5cda235` reused the 258-test evidence and did not
  rerun the suite because it changed only planning evidence.
- Documentation context ran `./gradlew :backends:cpu:javadoc` after final Java documentation edits.
  Documentation re-review context `019ffbcb-4c30-7e53-8e4d-9474f5cda235` reran it after the fix:
  `BUILD SUCCESSFUL`; 11 actionable tasks, 1 executed and 10 up-to-date. The only output warnings
  were the two expected incubating-module warnings for `jdk.incubator.vector`; Javadoc reported no
  missing-tag or content warnings.
- Final local Markdown validation checked links and heading anchors, balanced fences, one terminal
  newline, and trailing whitespace for the CPU guide, glossary, task, master plan, and roadmap;
  all five files passed. Two preliminary read-only Ruby validator invocations failed because the
  first regex was parsed as interpolation and the available Ruby lacked `Enumerator#filter_map`;
  the corrected compatible validator then passed and neither failed attempt changed a file.
- Final coherence checks confirmed task/master/roadmap mark 0006C `Complete`, CPU 0006D remains
  `Draft`, no detailed 0006D specification exists, generator schema 18 is current, all changed
  paths remain under the 48-path ceiling, and `git diff --check` passes.
- Generated Javadoc inspection found the ordering emitter, IR, lowering geometry, preparation,
  executable, reference, and package-summary contracts, including workspace ownership, stable
  order, output roles, overlap timing, and schema 18.
- The documentation pass made no executable Java statement or test change. It changed only Java
  comments/Javadocs, package summaries, the CPU backend guide, glossary, and planning records.
- Documentation re-review context `019ffbcb-4c30-7e53-8e4d-9474f5cda235` made no executable Java,
  test, Javadoc, package-summary, guide, or glossary change. Existing prose already accurately
  required complete pre-mutation overlap rejection; only final planning evidence changed.
- Package placement matches the approved map: `CpuOrderingIr`, `CpuOrderingLowering`, and
  `CpuOrderingEmitter` remain in existing `internal.ir`, `internal.lowering`, and
  `internal.codegen.emit` packages; no package, module, dependency, or build edge was added.
- Reasoned no-change conclusions: `ARCHITECTURE.md` and the current architecture plan remain
  accurate because the implementation stays inside the existing CPU-owned lowering/finalization/
  execution lifecycle and changes no dependency rule. Public Tensor and other API guides/modules
  remain unchanged because Model semantics and APIs did not change. Gradle/build structure,
  architecture tests, backend-conformance, and integration tests remain unchanged because no
  module boundary, dependency, shared build contract, reusable conformance harness, or end-to-end
  Engine path changed.

## Implementation notes

- Added focused ordering IR, lowering, generated scalar execution, reference evaluation,
  preparation/finalization, exact scratch binding, multi-output invocation, capability, cache
  schema, inventory, and regression evidence within the CPU module.
- The documentation-focused pass independently reviewed the complete uncommitted implementation
  diff, changed production sources and tests, relevant Model ordering contracts, CPU package
  summaries, backend guide, glossary, and synchronized planning records.
- Documentation-pass paths changed: `CpuCapabilityProvider.java`, `CpuGeneratorSchema.java`,
  `CpuOrderingEmitter.java`, `CpuPreparedExecutable.java`, `CpuOrderingIr.java`,
  `CpuPartitionLowering.java`, `CpuOrderingLowering.java`, `CpuPartitionPreparationPlan.java`,
  `CpuPartitionPreparer.java`, `CpuScalarReferenceKernel.java`; CPU `package-info.java` files for
  the public package and internal cache, codegen emitter, executable, IR, lowering, prepare,
  reference, and portable-route packages; `docs/backend-guide/cpu-backend.md`, `docs/glossary.md`,
  this task, the CPU master plan, and the roadmap.

## Completion summary

- Completed changes: one-node stable six-type SORT/ARGSORT and two-output TOP_K with exact Model
  order, represented-bit value copies, logical INT64 indices, deterministic unsorted selection,
  scalar/slice-parallel parity, exact per-range scratch, pre-mutation overlap rejection, and
  schema 18; finalized all affected documentation and planning evidence.
- Files changed or created: 39 total implementation-change paths, within the 48-path maximum; the
  24 documentation-pass paths are enumerated in Implementation notes.
- Tests and validation: reused the fix context's stabilized 258-test CPU pass; CPU Javadoc and final Markdown,
  generated-page, scope, status, schema, later-specification-absence, and whitespace gates passed.
- Documentation-agent review: clean context `019ffbb7-f877-73f3-afc3-5cbd8b6f593d` completed the
  original mandatory pass; clean re-review context `019ffbcb-4c30-7e53-8e4d-9474f5cda235`
  synchronized the post-fix evidence without executable Java or test changes.
- Documentation impact: CPU backend guide, glossary, package summaries, task, master plan, and
  roadmap now describe the exact supported boundary and separate Model meaning from CPU realization.
- Javadoc review: affected CPU capability, schema, ordering IR/lowering/emitter, preparation,
  executable, and reference contracts are meaningful and complete for parameters, results,
  failures, ownership, scratch, multi-output binding, and overlap timing.
- Glossary impact: added the reusable CPU stable-ordering realization boundary and advanced the
  current specialization/schema terminology to 18.
- Unresolved issues: None.
- Follow-up required: None. CPU 0006D remains Draft without a detailed specification.

Status: Complete
