# Task 0006C: Portable Stable Ordering and Selection

## Reader context

The ordering family produces values, logical indices, or both. SORT copied selected value
representations; ARGSORT wrote source-axis positions; TOP_K retained both outputs from one
semantic occurrence. “Stable” meant that equal values and NaNs preserved original index order,
including the explicit signed-zero and NaN placement rules below. A logical index is not a byte
address.

CPU analysis declared the result buffers and exact merge-index workspace before assignment.
Scalar or parallel-scalar execution processed whole independent axis slices using disjoint
scratch regions, so a worker did not split one stable merge arbitrarily. Runtime owned that
run’s result and scratch lifetime. The current sorting reference explains the public value/index
pairing separately from the original generated implementation.

Current reading alongside this historical record:

- [Stable sort and argsort semantics](../../../../reference/tensor/sort-topk.md#stable-sort-and-argsort-expressions)
  — explains logical indices, equal-value stability, NaN placement, and represented-bit selection.
- [CPU binding, validation, and ranges](../../../../explanation/packages/cpu.md#iogithubpho001synaptikbackendcpuinternalexecutable)
  — identifies current-value validation, direct invocation state, and worker/resource lifetimes.

## Status

Complete

## Goal

Add the next executable [CPU](../../../../glossary.md#term-cpu) frontier after completed CPU 0006B2: exactly one fully static,
resolved-[layout](../../../../glossary.md#layout) current [Model](../../../../glossary.md#term-model) `SORT`, `ARGSORT`, or `TOP_K` [occurrence](../../../../glossary.md#term-occurrence) through the existing
portable generated-[kernel](../../../../glossary.md#kernel) route.

The implementation preserves Model's stable logical-[axis](../../../../glossary.md#term-axis) ordering exactly: non-[NaN](../../../../glossary.md#term-nan) values use the
requested direction, [negative zero](../../../../glossary.md#nan-infinity-and-signed-zero) precedes [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero) ascending and reverses descending, every
NaN remains after every non-NaN in both directions, and equal values or NaNs retain increasing
original logical-[axis index](../../../../glossary.md#axis-rank-scalar-vector-and-matrix). `SORT` writes values, `ARGSORT` writes [INT64](../../../../glossary.md#data-type--datatype) logical indices, and
`TOP_K` writes values then INT64 indices from one two-output occurrence.

This task owns a deterministic portable [scalar](../../../../glossary.md#term-scalar) baseline, bounded [slice](../../../../glossary.md#slice)-parallel execution, exact
scratch, multi-store preparation/invocation, and independent reference evidence. It does not add
Model semantics, general [DAG](../../../../glossary.md#term-directed-acyclic-graph-dag) [decomposition](../../../../glossary.md#term-decomposition), [fusion](../../../../glossary.md#term-fusion), reductions, native routes, or [autotuning](../../../../glossary.md#autotuning--model-autotuning).

## [Scope](../../../../glossary.md#term-scope)

- Admit exactly one [CPU](../../../../glossary.md#term-cpu)-owned [node](../../../../glossary.md#node) with one input and either one `SORT` values output, one
  `ARGSORT` [INT64](../../../../glossary.md#data-type--datatype) output, or ordered `TOP_K` outputs `[values, indices]`, with their exact current
  `SortAttrs` or `TopKAttrs` [signatures](../../../../glossary.md#term-signature).
- Require fully static input/output Shapes, resolved non-negative-offset/non-negative-stride
  layouts, exact Model Shape rules, and distinct injective writable outputs.<span data-glossary-navigation="true"> ([Shapes glossary](../../../../glossary.md#shape); [stride glossary](../../../../glossary.md#term-stride); [layouts glossary](../../../../glossary.md#layout); [Model glossary](../../../../glossary.md#term-model); [injective glossary](../../../../glossary.md#term-injective))</span>
- Support FLOAT64, FLOAT32, BFLOAT16, INT32, INT64, and BOOL. Values outputs copy the exact
  represented input element; indices are INT64 logical-axis coordinates.<span data-glossary-navigation="true"> ([FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [INT32 glossary](../../../../glossary.md#data-type--datatype); [BOOL glossary](../../../../glossary.md#data-type--datatype); [element glossary](../../../../glossary.md#term-element); [coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- Add focused CPU-private `CpuOrderingIr`, `CpuOrderingLowering`, and `CpuOrderingEmitter` owners.
  Structural identity includes family, represented type, direction/top-K flags, boundary roles,
  types/ranks, access plans, scratch policy, and output count. Concrete extents, offsets, stride
  magnitudes, axis, K, ranges, slots, and carriers remain cold when they do not shape bytes.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Use deterministic stable ordering over logical indices without boxing, comparator/object arrays,
  per-element allocation, reflection, or runtime semantic dispatch. A permitted baseline is a
  bottom-up stable merge over two primitive INT64 index regions per worker range.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- Declare one exact run-owned scratch before shared assignment. Checked sizing must provide
  disjoint per-range regions sufficient for the selected stable algorithm; scratch never escapes
  the run and is accessible by every selected worker.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [stable algorithm glossary](../../../../glossary.md#conditioning-and-numerical-stability))</span>
- Range work over complete independent logical-axis slices and never split one slice. Parallel-
  scalar is legal only for disjoint slices, outputs, and scratch; otherwise use scalar. Vector
  sorting is out of scope.<span data-glossary-navigation="true"> ([slice glossary](../../../../glossary.md#slice); [Vector glossary](../../../../glossary.md#term-vector))</span>
- Support dense, offset, positive-strided, and zero-strided input views; injective non-dense
  outputs; heap arrays; native-order <span data-glossary-navigation="true"> ([views glossary](../../../../glossary.md#term-view-meanings); [heap glossary](../../../../glossary.md#term-heap))</span>[`MemorySegment`](../../../../glossary.md#term-memorysegment); and compatible mixed carriers. Ordering and
  indices follow logical coordinates, never physical traversal.
- `SORT`/`ARGSORT` preserve [axis](../../../../glossary.md#term-axis) [extent](../../../../glossary.md#term-extent). `TOP_K` selects the first `k` pairs of the complete stable
  order requested by `largest`; `sorted == true` retains it, while `sorted == false` orders the
  selected pairs by increasing original logical-axis index.<span data-glossary-navigation="true"> ([axis index glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- Accept exact empty behavior. Empty selected SORT/ARGSORT axes produce empty outputs; TOP_K
  requires `0 <= k <= extent`, so an empty selected axis accepts only zero. Empty unselected
  dimensions and <span data-glossary-navigation="true"> ([dimensions glossary](../../../../glossary.md#dimension))</span>`k == 0` submit no generated or worker work. [Scalar](../../../../glossary.md#term-scalar) inputs remain unsupported.
- Reject every input/output and output/output physical overlap before scratch mutation, output
  write, or worker submission.
- Extend capability, lowering, preparation, assignment/finalization, generated entry, binding,
  reference, cache compatibility, package contracts, and tests only as this family requires.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [binding glossary](../../../../glossary.md#term-binding); [cache glossary](../../../../glossary.md#term-cache))</span>
- Advance generated compatibility from [schema](../../../../glossary.md#term-schema) 17 to schema 18 with no migration reader.
- After Java behavior stabilizes, hand the uncommitted diff and CPU-test evidence to a distinct
  clean documentation-focused context for Javadocs, package summaries, CPU guide, glossary, task
  evidence, master plan, and roadmap.

## Exact ordering contract

- Each logical selected-[axis](../../../../glossary.md#term-axis) [slice](../../../../glossary.md#slice) is independent.
- FLOAT64/FLOAT32/BFLOAT16 non-NaNs use numerical order; ascending places negative infinity first
  and positive infinity last, and descending reverses non-NaNs.<span data-glossary-navigation="true"> ([FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [negative infinity glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [positive infinity glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- [Negative zero](../../../../glossary.md#nan-infinity-and-signed-zero) is strictly before [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero) ascending and strictly after it descending.
- All NaNs form one final class after all non-NaNs in either direction. Sign, payload, and
  signaling state do not affect comparison, but values outputs preserve selected represented bits.<span data-glossary-navigation="true"> ([payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [represented bits glossary](../../../../glossary.md#term-represented-bits))</span>
- [INT32](../../../../glossary.md#data-type--datatype)/[INT64](../../../../glossary.md#data-type--datatype) use signed order; [BOOL](../../../../glossary.md#data-type--datatype) uses `false < true` ascending.
- Equal values and multiple NaNs retain increasing logical-axis index; stable merge chooses the
  left/logically earlier candidate on equality.<span data-glossary-navigation="true"> ([axis index glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- ARGSORT and TOP_K indices are zero-based axis [coordinates](../../../../glossary.md#axis-rank-scalar-vector-and-matrix), not flattened storage offsets.

## Out of [scope](../../../../glossary.md#term-scope)

- multiple/mixed nodes, fusion, general partition DAGs, materialized DAG splits, or representation
  reuse beyond this scratch<span data-glossary-navigation="true"> ([nodes glossary](../../../../glossary.md#node); [fusion glossary](../../../../glossary.md#term-fusion); [partition glossary](../../../../glossary.md#partition); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag))</span>
- unstable/custom ordering, algorithm/NaN options, radix/network/parallel merge, vector sorting,
  approximate top-K, dynamic Shapes/K, or runtime algorithm choice<span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [vector glossary](../../../../glossary.md#term-vector); [Shapes glossary](../../../../glossary.md#shape); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- RNG/dropout, reductions/scans, normalization, heavy families, native routes, tuning, relaxed
  numerics, or persistence-policy changes<span data-glossary-navigation="true"> ([RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [normalization glossary](../../../../glossary.md#term-normalization); [persistence glossary](../../../../glossary.md#term-persistence))</span>
- new Model/compiler/shared Prepare/Runtime/public Config/Trace behavior, architecture/dependency/
  build changes, negative physical strides/offsets, unresolved layouts, device transfer, or Engine<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [compiler glossary](../../../../glossary.md#term-compiler); [Prepare glossary](../../../../glossary.md#prepare); [Trace glossary](../../../../glossary.md#trace); [dependency glossary](../../../../glossary.md#term-dependency); [strides glossary](../../../../glossary.md#term-stride); [layouts glossary](../../../../glossary.md#layout); [device glossary](../../../../glossary.md#term-device); [Engine glossary](../../../../glossary.md#term-engine))</span>
- new backend-conformance/integration infrastructure; executable conformance closes at CPU 0009
  unless an applicable current reusable harness is found<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [CPU glossary](../../../../glossary.md#term-cpu))</span>

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Roadmap](../../../roadmap.md)
- [CPU master plan](../master-plan.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- [CPU 0006B2 overlap fold](0006b2-portable-overlap-fold.md)
- [Model sort/argsort](../../../modules/model/tasks/0019c-sort-and-argsort.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- [Model top-K](../../../modules/model/tasks/0019c1-top-k-values-and-indices.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [CPU backend guide](../../../../backend-guide/cpu-backend.md)<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>
- [Glossary](../../../../glossary.md)

## Architecture constraints

- Model alone owns operation identity, attributes, Shape/output-slot rules, and ordering meaning.
  CPU independently revalidates and owns only its algorithm, IR, scratch, artifact, and invocation.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [operation glossary](../../../../glossary.md#operation); [identity glossary](../../../../glossary.md#term-identity-meanings); [Shape glossary](../../../../glossary.md#shape); [CPU glossary](../../../../glossary.md#term-cpu); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- CPU analysis declares exact resources before CPU-blind shared assignment; finalization validates
  assignments before realizing exactly one generated artifact.
- Runtime receives one direct prepared invocation and no graph, Model operation, comparator, or
  semantic-dispatch object.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [graph glossary](../../../../glossary.md#graph); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- Work remains in `backends/cpu`, adds no [dependency](../../../../glossary.md#term-dependency), and preserves package direction.
- Capability is no broader than lowering, assignment, binding, multi-output writes, scratch, and
  generated execution can complete.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [binding glossary](../../../../glossary.md#term-binding))</span>
- Stop if current Model source conflicts or exact behavior needs architecture, another module,
  shared-contract, build, or unresolved numerical-policy changes.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module))</span>

## Package impact

No package is added, removed, or moved.

- `internal.ir.CpuOrderingIr` owns [immutable](../../../../glossary.md#term-immutable) ordering/type/access/scratch/output [identity](../../../../glossary.md#term-identity-meanings).
- `internal.lowering.CpuOrderingLowering` owns one-node revalidation, slice geometry, layouts,
  scratch sizing, and boundaries.<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [slice glossary](../../../../glossary.md#slice); [layouts glossary](../../../../glossary.md#layout))</span>
- `internal.codegen.emit.CpuOrderingEmitter` owns generated stable ordering and represented
  load/store mechanics.
- Focused immutable geometry may be nested under lowering. Add no generic comparator, manager,
  utility package, registry, or execution facade.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [registry glossary](../../../../glossary.md#term-registry); [facade glossary](../../../../glossary.md#term-facade))</span>

## Affected files

Expected production work is the three focused new types plus directly necessary [CPU](../../../../glossary.md#term-cpu) capability,
[IR](../../../../glossary.md#term-intermediate-representation-ir) permits/package docs, [lowering](../../../../glossary.md#lowering) [dispatch](../../../../glossary.md#term-dispatch)/package docs, generator/[schema](../../../../glossary.md#term-schema)/[specialization](../../../../glossary.md#term-specialization),
preparation/finalization, executable/workspace, reference, route, and inventory seams.

Expected tests cover ordering IR/lowering/generated execution, preparation/finalization,
executable, capability, schema/[cache](../../../../glossary.md#term-cache), carriers/[layouts](../../../../glossary.md#layout), independent differential reference, and
package inventory using existing fixtures.

Expected documentation/[planning](../../../../glossary.md#term-planning) is this task, CPU master plan, roadmap, CPU [backend](../../../../glossary.md#backend) guide,
glossary, and affected package/Javadocs. Architecture/[API](../../../../glossary.md#term-application-programming-interface-api) guides, other [modules](../../../../glossary.md#term-module), [Gradle](../../../../glossary.md#term-gradle),
architecture tests, and empty [conformance](../../../../glossary.md#term-conformance)/integration harnesses are expected no-change.

## Maximum [scope](../../../../glossary.md#term-scope)

At most 48 paths: 23 production/package, 18 [CPU](../../../../glossary.md#term-cpu) test, and 7 documentation/[planning](../../../../glossary.md#term-planning). Stop before a
49th path or any other [module](../../../../glossary.md#term-module), build file, [architecture contract](../../../../glossary.md#architecture-contract), or shared [Prepare](../../../../glossary.md#prepare)/[Runtime](../../../../glossary.md#term-runtime) change.

## Acceptance criteria

- Capability/lowering accept exactly the three one-node families and six-type matrix under static
  resolved constraints; all unsupported work fails closed.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [node glossary](../../../../glossary.md#node); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
- Results exactly cover direction, stability, NaN-last, signed zero, infinity, BOOL, duplicates,
  empty/singleton/K, and TOP_K sorted/unsorted semantics.<span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [signed zero glossary](../../../../glossary.md#term-signed-zero); [infinity glossary](../../../../glossary.md#term-infinity); [BOOL glossary](../../../../glossary.md#data-type--datatype); [singleton glossary](../../../../glossary.md#term-singleton))</span>
- Boundaries are `[input, output]` for SORT/ARGSORT and `[input, values, indices]` for TOP_K; one
  TOP_K executable performs both stores.
- Exact scratch declaration/assignment/slicing/accessibility/overflow/cleanup and zero-work rules
  are tested, with no per-element heap allocation.<span data-glossary-navigation="true"> ([overflow glossary](../../../../glossary.md#term-overflow); [element glossary](../../../../glossary.md#term-element); [heap glossary](../../../../glossary.md#term-heap))</span>
- [Scalar](../../../../glossary.md#term-scalar) and parallel-scalar are bitwise identical; ranges share no output/scratch [coordinates](../../../../glossary.md#axis-rank-scalar-vector-and-matrix).
- Heap/native/mixed carriers and dense/offset/strided/zero-stride input plus non-dense injective
  output layouts agree with an independent reference.<span data-glossary-navigation="true"> ([stride glossary](../../../../glossary.md#term-stride); [injective glossary](../../../../glossary.md#term-injective); [layouts glossary](../../../../glossary.md#layout))</span>
- All overlaps fail before mutation or submission.
- Schema 18 and structural type/carrier/output distinctions are deterministic and collision-tested;
  one artifact is realized.<span data-glossary-navigation="true"> ([Schema glossary](../../../../glossary.md#term-schema); [carrier glossary](../../../../glossary.md#term-storage-carrier); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- Existing [pointwise](../../../../glossary.md#term-pointwise), movement, indexing, scatter, and fold behavior remains unchanged.
- A separate documentation context finalizes docs and no-change conclusions before Complete.

## Tests / [validation](../../../../glossary.md#term-validation)

During implementation [run](../../../../glossary.md#term-run) focused tests, then one final executable suite:

```bash
./gradlew :backends:cpu:test
```

The clean documentation pass reuses that evidence and runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
```

It also checks links/anchors/fences/newlines, changed-path [scope](../../../../glossary.md#term-scope), generated [Javadoc](../../../../glossary.md#term-javadoc), and task/master/
roadmap coherence. Repository-wide and executable [conformance](../../../../glossary.md#term-conformance) validation are deferred to [CPU](../../../../glossary.md#term-cpu) 0009
or [CI](../../../../glossary.md#term-continuous-integration-ci) unless a repository-wide contract unexpectedly changes.

## Dependencies

- [CPU](../../../../glossary.md#term-cpu) 0005A–0006B2: Complete.
- [Model](../../../../glossary.md#term-model) 0019C and 0019C1 ordering semantics: Complete.
- Portable Class-File route, typed carriers, workers, workspace, multi-boundary declarations, and
  artifact store: Complete.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact))</span>

## Follow-up tasks

- [CPU](../../../../glossary.md#term-cpu) 0006D remains next Draft and owns explicit-state [RNG](../../../../glossary.md#graph-rng-state--graphrngstate)/[dropout](../../../../glossary.md#dropout--inverted-dropout).
- CPU 0007 owns reduction/[scan](../../../../glossary.md#term-cumulative-scan)/statistics/[normalization](../../../../glossary.md#term-normalization).
- CPU 0008A–0008D own later [decomposition](../../../../glossary.md#term-decomposition)/[fusion](../../../../glossary.md#term-fusion)/profitability/representation reuse.
- CPU 0009 owns portable closure and executable [conformance](../../../../glossary.md#term-conformance).

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
  scalar/slice-parallel orchestration, carriers, artifact identity, and cold binding.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Model glossary](../../../../glossary.md#term-model); [axis glossary](../../../../glossary.md#term-axis); [NaN glossary](../../../../glossary.md#term-nan); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [scalar glossary](../../../../glossary.md#term-scalar); [slice glossary](../../../../glossary.md#slice); [artifact glossary](../../../../glossary.md#term-artifact); [identity glossary](../../../../glossary.md#term-identity-meanings); [binding glossary](../../../../glossary.md#term-binding))</span>
- TOP_K remains one three-boundary unit and one generated artifact. Its boundary order is input,
  represented values, then INT64 logical-axis indices; one invocation performs both stores.<span data-glossary-navigation="true"> ([INT64 glossary](../../../../glossary.md#data-type--datatype))</span>
- Every ordering plan declares one exact run-owned workspace. Each selected range receives two
  disjoint axis-extent INT64 regions, and parallel work owns complete logical-axis slices.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [extent glossary](../../../../glossary.md#term-extent))</span>
- Unsorted TOP_K selects from the complete stable value order, then deterministically orders the
  selected pairs by increasing original logical-axis index.<span data-glossary-navigation="true"> ([axis index glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>

## Known limitations

- Coverage is one fully static resolved-layout occurrence. Dynamic Shapes/layouts, scalar input,
  negative physical offsets/strides, multi-node fusion, custom or unstable ordering, vector or
  native sorting, approximate selection, and runtime algorithm selection remain unsupported.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [occurrence glossary](../../../../glossary.md#term-occurrence); [Shapes glossary](../../../../glossary.md#shape); [scalar glossary](../../../../glossary.md#term-scalar); [strides glossary](../../../../glossary.md#term-stride); [node glossary](../../../../glossary.md#node); [fusion glossary](../../../../glossary.md#term-fusion); [vector glossary](../../../../glossary.md#term-vector); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- The scalar and slice-parallel CPU realizations are bitwise identical, but this task makes no
  cross-backend bitwise or performance claim.<span data-glossary-navigation="true"> ([slice glossary](../../../../glossary.md#slice); [CPU glossary](../../../../glossary.md#term-cpu); [backend glossary](../../../../glossary.md#backend))</span>
- `testing/backend-conformance` has no reusable executable cross-backend harness at this frontier.
  Executable conformance remains planned for CPU 0009; this task does not invent a cross-module
  harness. Repository-wide validation remains deferred to that checkpoint or CI.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [module glossary](../../../../glossary.md#term-module); [validation glossary](../../../../glossary.md#term-validation); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [CI glossary](../../../../glossary.md#term-continuous-integration-ci))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- Implementation context initially ran the single final `./gradlew :backends:cpu:test` after the
  original executable Java stabilized: 257 tests, zero failures, zero errors, and one skip. The
  first documentation context `019ffbb7-f877-73f3-afc3-5cbd8b6f593d` reused that evidence. A later
  coordinator review found that one-output SORT/ARGSORT overlap validation ranged complete buffer
  bindings by slice ordinals, which could miss physical overlap outside that ordinal prefix. Fix
  context <span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer); [bindings glossary](../../../../glossary.md#term-binding); [slice glossary](../../../../glossary.md#slice))</span>`019ffbc8-fb51-7a73-9504-9a3929cefe58` changed the overlap check to use complete boundary
  bindings, added the multidimensional SORT/ARGSORT regression, and ran the authoritative final<span data-glossary-navigation="true"> ([regression glossary](../../../../glossary.md#term-regression))</span>
  `./gradlew :backends:cpu:test`: 258 tests, zero failures, zero errors, and one skip. Documentation
  re-review context `019ffbcb-4c30-7e53-8e4d-9474f5cda235` reused the 258-test evidence and did not
  rerun the suite because it changed only planning evidence.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Documentation context ran `./gradlew :backends:cpu:javadoc` after final Java documentation edits.
  Documentation re-review context `019ffbcb-4c30-7e53-8e4d-9474f5cda235` reran it after the fix:
  `BUILD SUCCESSFUL`; 11 actionable tasks, 1 executed and 10 up-to-date. The only output warnings
  were the two expected incubating-module warnings for `jdk.incubator.vector`; Javadoc reported no
  missing-tag or content warnings.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Final local Markdown validation checked links and heading anchors, balanced fences, one terminal
  newline, and trailing whitespace for the CPU guide, glossary, task, master plan, and roadmap;
  all five files passed. Two preliminary read-only Ruby validator invocations failed because the
  first regex was parsed as interpolation and the available Ruby lacked `Enumerator#filter_map`;
  the corrected compatible validator then passed and neither failed attempt changed a file.
- Final coherence checks confirmed task/master/roadmap mark 0006C `Complete`, [CPU](../../../../glossary.md#term-cpu) 0006D remains
  `Draft`, no detailed 0006D specification exists, generator schema 18 is current, all changed
  paths remain under the 48-path ceiling, and <span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema))</span>`git diff --check` passes.
- Generated Javadoc inspection found the ordering emitter, IR, lowering geometry, preparation,
  executable, reference, and package-summary contracts, including workspace ownership, stable
  order, output roles, overlap timing, and schema 18.<span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [lowering glossary](../../../../glossary.md#lowering); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- The documentation pass made no executable Java statement or test change. It changed only Java
  comments/Javadocs, package summaries, the CPU backend guide, glossary, and planning records.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>
- Documentation re-review context `019ffbcb-4c30-7e53-8e4d-9474f5cda235` made no executable Java,
  test, Javadoc, package-summary, guide, or glossary change. Existing prose already accurately
  required complete pre-mutation overlap rejection; only final planning evidence changed.
- Package placement matches the approved map: `CpuOrderingIr`, `CpuOrderingLowering`, and
  `CpuOrderingEmitter` remain in existing `internal.ir`, `internal.lowering`, and
  `internal.codegen.emit` packages; no package, [module](../../../../glossary.md#term-module), [dependency](../../../../glossary.md#term-dependency), or build edge was added.
- Reasoned no-change conclusions: `ARCHITECTURE.md` and the current architecture plan remain
  accurate because the implementation stays inside the existing CPU-owned lowering/finalization/
  execution lifecycle and changes no dependency rule. Public Tensor and other API guides/modules
  remain unchanged because Model semantics and APIs did not change. Gradle/build structure,
  architecture tests, backend-conformance, and integration tests remain unchanged because no
  module boundary, dependency, shared build contract, reusable conformance harness, or end-to-end
  Engine path changed.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Model glossary](../../../../glossary.md#term-model); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance); [Engine glossary](../../../../glossary.md#term-engine))</span>

## Implementation notes

- Added focused ordering IR, lowering, generated scalar execution, reference evaluation,
  preparation/finalization, exact scratch binding, multi-output invocation, capability, cache
  schema, inventory, and regression evidence within the CPU module.<span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [lowering glossary](../../../../glossary.md#lowering); [scalar glossary](../../../../glossary.md#term-scalar); [evaluation glossary](../../../../glossary.md#term-evaluation); [binding glossary](../../../../glossary.md#term-binding); [cache glossary](../../../../glossary.md#term-cache); [schema glossary](../../../../glossary.md#term-schema); [regression glossary](../../../../glossary.md#term-regression); [module glossary](../../../../glossary.md#term-module))</span>
- The documentation-focused pass independently reviewed the complete uncommitted implementation
  diff, changed production sources and tests, relevant Model ordering contracts, CPU package
  summaries, backend guide, glossary, and synchronized planning records.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [backend glossary](../../../../glossary.md#backend); [planning glossary](../../../../glossary.md#term-planning))</span>
- Documentation-pass paths changed: `CpuCapabilityProvider.java`, `CpuGeneratorSchema.java`,
  `CpuOrderingEmitter.java`, `CpuPreparedExecutable.java`, `CpuOrderingIr.java`,
  `CpuPartitionLowering.java`, `CpuOrderingLowering.java`, `CpuPartitionPreparationPlan.java`,
  `CpuPartitionPreparer.java`, `CpuScalarReferenceKernel.java`; [CPU](../../../../glossary.md#term-cpu) `package-info.java` files for
  the public package and internal cache, codegen emitter, executable, IR, lowering, prepare,
  reference, and portable-route packages; <span data-glossary-navigation="true"> ([prepare glossary](../../../../glossary.md#prepare))</span>`docs/backend-guide/cpu-backend.md`, `docs/glossary.md`,
  this task, the CPU master plan, and the roadmap.

## Completion summary

- Completed changes: one-node stable six-type SORT/ARGSORT and two-output TOP_K with exact Model
  order, represented-bit value copies, logical INT64 indices, deterministic unsorted selection,
  scalar/slice-parallel parity, exact per-range scratch, pre-mutation overlap rejection, and
  schema 18; finalized all affected documentation and planning evidence.<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [Model glossary](../../../../glossary.md#term-model); [represented-bit glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [INT64 glossary](../../../../glossary.md#data-type--datatype); [scalar glossary](../../../../glossary.md#term-scalar); [slice glossary](../../../../glossary.md#slice); [schema glossary](../../../../glossary.md#term-schema); [planning glossary](../../../../glossary.md#term-planning))</span>
- Files changed or created: 39 total implementation-change paths, within the 48-path maximum; the
  24 documentation-pass paths are enumerated in Implementation notes.
- Tests and validation: reused the fix context's stabilized 258-test CPU pass; CPU Javadoc and final Markdown,
  generated-page, scope, status, schema, later-specification-absence, and whitespace gates passed.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope))</span>
- Documentation-agent review: clean context `019ffbb7-f877-73f3-afc3-5cbd8b6f593d` completed the
  original mandatory pass; clean re-review context `019ffbcb-4c30-7e53-8e4d-9474f5cda235`
  synchronized the post-fix evidence without executable Java or test changes.
- Documentation impact: CPU backend guide, glossary, package summaries, task, master plan, and
  roadmap now describe the exact supported boundary and separate Model meaning from CPU realization.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>
- Javadoc review: affected CPU capability, schema, ordering IR/lowering/emitter, preparation,
  executable, and reference contracts are meaningful and complete for parameters, results,
  failures, ownership, scratch, multi-output binding, and overlap timing.<span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [lowering glossary](../../../../glossary.md#lowering); [parameters glossary](../../../../glossary.md#term-parameter); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [binding glossary](../../../../glossary.md#term-binding))</span>
- Glossary impact: added the reusable CPU stable-ordering realization boundary and advanced the
  current specialization/schema terminology to 18.<span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization))</span>
- Unresolved issues: None.
- Follow-up required: None. [CPU](../../../../glossary.md#term-cpu) 0006D remains Draft without a detailed specification.

Status: Complete
