# Task 0008Q1: Finite Scalar-Immediate and Clamp Matrix

## Status

Ready

## Goal

Establish the complete, finite, source-derived projection basis for every currently admitted CPU scalar-immediate operation and floating clamp before CPU 0009 inventories generated coverage. Prove generated execution equivalent to the matching optimal clean Java 26 specialization for each distinct form, and project only members whose Class-Files mechanically differ in declared constant locations. Do not enumerate arbitrary immediate bit patterns.

## Scope

- Derive the matrix from current `CpuCapabilityProvider`, `CpuPartitionLowering`, `CpuKernelIr`, `CpuScalarPowerAnalysis`, `CpuPartitionPreparer`, `CpuKernelSpecialization`, and `CpuClassFileKernelGenerator`; do not start from a manually assumed product.
- Cover every currently capability-admitted scalar occurrence: `ADD`, `SUB`, `MUL`, `DIV`, `MIN`, `MAX`, and `POW`, plus first-class `CLAMP`. Record Model capability, successful whole-partition lowering, preparation eligibility, and actual selected strategy as separate facts.
- Cover BFLOAT16, FLOAT32, and FLOAT64 for every floating scalar operation and clamp; INT32 and INT64 only for scalar ADD/SUB/MUL/MIN/MAX; and no BOOL, mixed-type scalar, integral DIV/POW, or integral clamp row. Assert each adjacent exclusion fails before artifact construction.
- Cover every generated direct-carrier signature actually reachable for each type: matching heap primitive arrays, native-order `MemorySegment`s, and every ordered legal array/segment mixed input/output pair. Keep a carrier pair separate whenever it changes entry descriptor, carrier load/store instruction, address arithmetic, or helper/invoke set.
- Cover direct inputs/outputs in `DENSE_LINEAR`, `SCALAR_ALL_ZERO`, `LAST_AXIS_BIAS`, `BLOCK_OUTER`, and `GENERAL_ODOMETER` whenever current lowering admits that regime. Exercise rank-zero, zero-work, one-dimensional contiguous, and minimally shaped right-aligned broadcast/non-contiguous witnesses only when they select a different generated loop or dataflow shape. Record materialization candidates and selection; include a materialized form only when current preparation can actually select it.
- Cover `SCALAR`, caller-parallel `PARALLEL_SCALAR`, `VECTOR`, and `PARALLEL_VECTOR` only when the preparer admits and selects them. Artifact sharing is permitted only after test proof that it is orchestration sharing; otherwise strategy changes remain separate.
- Derive finite immediate categories from source and emitted code shape. Ordinary arithmetic/extrema categories are distinct constant-encoding and semantic-edge categories discovered from the current emitter; scalar power is split by `POSITIVE_ONE`, `IDENTITY`, `SQUARE`, `RECIPROCAL`, and finite `DIRECT`; clamp is split by ordered lower/upper category pairs and any distinct lower-first/upper-second control-flow or constant-encoding shape. Include signed-zero ordering where it changes clamp/extrema semantics. Non-finite inputs remain oracle edge cases, but non-finite immediate or bound patterns are outside this finite matrix.
- Generate an exact fixture for every discovered non-projectable form. A projected unit records exact member hashes, specialization facts, normalizer version, declared constant locations, and source-derived category. It is `PROVED_CONSTANTS_ONLY` only if automated Class-File normalization proves equal members, method structure, control flow, invokes, carrier accesses, loops, and dataflow except for declared constants. Otherwise split or fail closed.
- Execute every exact fixture and an independent typed optimal clean-Java loop across ordinary values and applicable NaN, infinity, signed-zero, subnormal, overflow/underflow, and clamp-bound edge inputs. Preserve the operation's numerical order, BFLOAT16 decode/one-operation/encode boundary, carrier access, loop shape, and scalar-power realization. The oracle must not call generated code, the reference kernel, or CPU lowering.
- Add automated structural dossiers proving the matching clean-Java algorithm and generated body have no hidden Synaptik helper, allocation, boxing, reflection, map/string dispatch, generic carrier branch, or avoidable virtual/interface dispatch on the proved hot path. Keep hashes and normalized-member provenance in checked resources for CPU 0009.

## Out of scope

- Production Java, Model/Compiler/Prepare/Runtime contracts, schemas, cache identity, capability rules, route or strategy-selection policy, new materialization policy, performance tuning, native routes, or arbitrary immediate-bit enumeration.
- Projection across an opcode, type, scalar-power realization, ordered clamp category, carrier access, access regime, materialization disposition, selected compute strategy, vector species, loop/dataflow/control-flow shape, numerical order, invoke set, or semantic edge category that changes generated code or meaning.
- Five-fork performance claims or CPU 0009's full generated-coverage inventory.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md): CPU owns backend lowering/selection; generated code preserves the optimal clean-Java semantic algorithm and hot-loop/dataflow shape; unproved specialization is fail-closed.
- [Planning Guide](../../../planning-guide.md): status, bounded scope, validation, and documentation handoff.
- [CPU master plan](../master-plan.md): CPU package ownership, four strategies, and task order.
- [CPU 0005F](0005f-floating-division-and-exact-scalar-power-realization.md), [CPU 0005G](0005g-extrema-clamp-tensor-power-and-logical-coverage.md), [CPU 0005J](0005j-bounded-pointwise-coverage-and-parity-hardening.md), [CPU 0008J](0008j-bfloat16-scalar-pointwise-closure.md), [CPU 0008L](0008l-pointwise-simd-mask-output-closure.md), and [CPU 0008Q](0008q-scalar-immediate-clamp-generated-equivalence.md).

## Architecture constraints

- Model owns scalar and clamp semantics. CPU preserves exact typed raw immediate/bound bits in private IR, selects preparation facts before finalization, and Runtime only invokes the prepared artifact. No generated hot path sees `Operation`, `CompiledNode`, layouts, or string dispatch.
- Capability, complete-partition admission, and actual selection remain separate checked columns. A provider-positive form that cannot lower, bind, or select is fail-closed, not matrix coverage.
- Exact raw bits remain specialization identity facts even where a normalizer proves a constants-only relation. Test projection is evidence only and never relaxes production identity or cache reuse.
- Generated code and clean Java oracle use the same semantic order and direct typed carrier accesses. Direct scalar power cannot inherit identity, multiply, or reciprocal proof.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` for fixtures, independent oracles, Class-File normalization, structural assertions, and checked resources.
- Existing `.internal.lowering`, `.internal.ir`, `.internal.prepare`, and `.internal.cache` test packages only for a missing source-derived assertion.

Packages added or changed:

- None. No production type is added or moved.

Type placement:

- `...internal.codegen.emit.CpuScalarImmediateClampMatrixTest` — fixture, capability/admission/selection, oracle, and resource closure.
- `...internal.codegen.emit.CpuScalarImmediateClampMatrixStructuralTest` — Class-File/dataflow and normalizer mutation controls.
- `...internal.codegen.emit.CpuScalarImmediateClampMatrixOracle` — package-private independent clean-Java specializations; never a production dependency.

## Affected files

Expected implementation paths:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuScalarImmediateClampMatrixTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuScalarImmediateClampMatrixStructuralTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuScalarImmediateClampMatrixOracle.java`
- Up to two existing focused CPU test paths, only for an omitted capability/admission/selection witness.
- `backends/cpu/src/test/resources/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/scalar-immediate-clamp-matrix-fixtures.tsv`
- `backends/cpu/src/test/resources/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/scalar-immediate-clamp-matrix-projections.tsv`
- `backends/cpu/src/test/resources/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/scalar-immediate-clamp-matrix-forms.tsv`
- Up to three directly affected planning/Javadoc/glossary paths, this task, [CPU master plan](../master-plan.md), [CPU 0009](0009-portable-generated-coverage-closure-checkpoint.md), and [roadmap](../../../roadmap.md).

## Maximum scope

At most five CPU test sources, three checked matrix resources, three documentation paths, and four planning paths; zero production Java, Gradle, architecture, conformance, or integration paths. The fixture count is source-derived rather than capped. If it cannot fit the three resources and compositional mechanically verified dimensions, or needs production behavior, stop and create a family-specific Draft follow-up rather than hand-maintaining an unbounded Cartesian product.

## Stop conditions

Stop and leave this task `Incomplete` if any reachable finite category cannot be classified from source; an admitted form lacks an independent optimal clean-Java oracle; a required carrier/layout/strategy witness changes unaccounted code shape; a normalizer permits a nonconstant difference; source capability/admission/selection facts contradict one another; a semantic/structural test fails; or work needs production, architecture, schema, or selection changes. Record exact form, source predicate, fixture/hash where available, and required follow-up. Do not make CPU 0009 `Ready`.

## Acceptance criteria

- Checked resources reconcile every admitted scalar operation/clamp type row and every actual lowerable carrier/access/layout/strategy/materialization form, with explicit capability, admission, and selection results plus adjacent fail-closed exclusions.
- The category table is source-derived, includes every scalar-power realization and ordered clamp pair needing distinct semantics or emission, and never claims arbitrary immediate/bound enumeration.
- Every non-projectable form has exact generated hash, descriptor, schema, structural key, carrier pattern, regime, selected strategy, shape witness, category, oracle, and dossier. Every projection names exact members/hashes and passes strict constants-only normalization.
- Generated-vs-optimal-clean-Java comparisons cover every exact fixture and applicable ordinary/special inputs. Structural checks prove matching carrier and loop/dataflow shape and reject prohibited mechanisms.
- Artifact sharing is proven only for orchestration sharing; all other code-shaping changes remain distinct.
- A separate documentation-focused agent pass finalizes affected planning/Javadoc/glossary impact in the same change and records explicit no-change conclusions where applicable.

## Tests / validation

Run:

```bash
./gradlew :backends:cpu:test --tests '*ScalarImmediateClampMatrix*' --tests '*ScalarImmediateClampEquivalence*' --tests '*CpuPointwiseGeneratedKernelTest' --tests '*CpuPointwisePartitionLoweringTest' --tests '*CpuScalarPowerAnalysisTest' --rerun-tasks
```

Documentation pass:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
```

Validate exact TSV headers/columns/tabs/digests; hashes; bidirectional fixture/projection membership; source-derived closure; Class-File verification/decompilation; normalizer mutation controls; local Markdown links/anchors/fences; status/dependency/path-limit consistency; and `git status --short`. Repository-wide validation is deferred to CPU 0009/CI; architecture, conformance, and integration tests are unchanged unless this task stops for a boundary change.

## Dependencies

- Complete CPU 0005F, 0005G, 0005J, 0008J, 0008L, and 0008Q; current schema-63 pointwise lowering/preparation/generation; Java 26 Class-File/Vector toolchain.
- Current source is the only matrix authority. This task cannot infer a missing operation, carrier, layout, vector route, or materialization form from history.

## Follow-up tasks

- CPU 0009 may become `Ready` only after this task and its separate documentation pass are `Complete` with a finite proved matrix.
- A category or relation that fails semantic or constants-only proof becomes a narrowly named Draft remediation task; it is not projected into CPU 0009.

## Architecture impact

Expected impact: None. Stop for an architecture, dependency, production-identity, schema, capability, selection, carrier, or semantics change.

## Implementation prompt

```text
You are working in the Synaptik repository. Read AGENTS.md, ARCHITECTURE.md, the Planning Guide, the CPU master plan, CPU 0008Q, CPU 0009, and this task. Implement this test/resource matrix exactly as specified; do not change production behavior or enumerate arbitrary immediate bits. Stop for architecture or scope conflict. Do not commit or push. After executable validation, hand the stable diff and exact evidence to a separate clean documentation-focused context. That pass must follow `docs/developer-guide/documentation-rules.md` and finalize planning/Javadoc/glossary review. Update this task with results; do not mark it Complete before that pass finishes.
```

## Local decisions

- The finite matrix composes source-derived dimensions rather than a manually maintained product. Composition is allowed only after executable proof that the omitted dimension cannot change instructions, control flow, invokes, carrier access, loop/dataflow, numerical order, or semantics.
- A finite immediate category is an emitted-code and semantic boundary, not a decimal-value bucket. `SCALAR_POW` realizations are separate before normalization; clamp bounds retain lower-then-upper order.
- Completed 0008Q is a reusable normalizer/test seed for two BFLOAT16 scalar-MUL members, not a representative fixture for a new form or complete matrix evidence.

## Known limitations

The matrix proves finite source-reachable categories, not all raw immediate bit patterns. It has no performance result and cannot change selection. A new source category, generator change, or incompatible hash invalidates the relevant projection until regenerated.

## Validation evidence

Planning-only evidence: reviewed the required architecture/planning contracts, CPU master plan, 0008Q, blocked 0009, current scalar lowering/IR/power analysis/preparation/generator, and focused 0005F/0005G/0005J/0008J/0008L/0008Q test evidence. Current source admits the operation/type boundary, five access regimes, exact carrier specialization, four strategy names, and source-owned power realizations recorded above. No Java, Javadoc, Class-File, or performance command ran for this planning change.

## Implementation notes

Empty until implemented.

## Completion summary

Empty until implemented.
