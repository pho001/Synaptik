# Task 0008Q1: Finite Scalar-Immediate and Clamp Matrix

## Status

Complete

## Goal

Establish the complete, finite, source-derived projection basis for every currently admitted CPU scalar-immediate operation and floating clamp before CPU 0009 inventories generated coverage. Prove generated execution equivalent to the matching optimal clean Java 26 specialization for each distinct form, and project only mechanically proved relations. Do not enumerate arbitrary immediate bit patterns. Completed [CPU 0008Q1A](0008q1a-vector-scalar-power-hot-path-self-containment.md) removed the independently observed Synaptik helper calls from generated vector scalar-power hot paths under schema 64; this task regenerated and closed its matrix against that schema.

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
- [CPU 0008Q1A](0008q1a-vector-scalar-power-hot-path-self-containment.md): required production remediation discovered by this task's independent structural test.

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

Stop and leave this task `Incomplete` if any reachable finite category cannot be classified from source; an admitted form lacks an independent optimal clean-Java oracle; a required carrier/layout/strategy witness changes unaccounted code shape; a normalizer permits a nonconstant difference; source capability/admission/selection facts contradict one another; a semantic/structural test fails; or work needs another production, architecture, schema, or selection change. Record exact form, source predicate, fixture/hash where available, and required follow-up. Do not make CPU 0009 `Ready`.

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

- Complete CPU 0005F, 0005G, 0005J, 0008J, 0008L, 0008Q, and CPU 0008Q1A with its separate documentation pass; then-current pointwise lowering/preparation/generation schema; Java 26 Class-File/Vector toolchain.
- Current source is the only matrix authority. This task cannot infer a missing operation, carrier, layout, vector route, or materialization form from history.

## Follow-up tasks

- CPU 0009 may become `Ready` only after this task and its separate documentation pass are `Complete` with a finite proved matrix.
- A category or relation that fails semantic or constants-only proof becomes a narrowly named Draft remediation task; it is not projected into CPU 0009.

## Architecture impact

Expected impact: None. Stop for an architecture, dependency, production-identity, schema, capability, selection, carrier, or semantics change.

## Implementation prompt

```text
You are working in the Synaptik repository after CPU 0008Q1A and its documentation pass are Complete. Read AGENTS.md, ARCHITECTURE.md, the Planning Guide, the CPU master plan, CPU 0008Q, CPU 0008Q1A, CPU 0009, and this task. Implement this test/resource matrix exactly as specified; do not change production behavior or enumerate arbitrary immediate bits. Stop for architecture or scope conflict. Do not commit or push. After executable validation, hand the stable diff and exact evidence to a separate clean documentation-focused context. That pass must follow `docs/developer-guide/documentation-rules.md` and finalize planning/Javadoc/glossary review. Update this task with results; do not mark it Complete before that pass finishes.
```

## Local decisions

- The finite matrix composes source-derived dimensions rather than a manually maintained product. Composition is allowed only after executable proof that the omitted dimension cannot change instructions, control flow, invokes, carrier access, loop/dataflow, numerical order, or semantics.
- A finite immediate category is an emitted-code and semantic boundary, not a decimal-value bucket. `SCALAR_POW` realizations are separate before normalization; clamp bounds retain lower-then-upper order.
- Completed 0008Q is a reusable normalizer/test seed for two BFLOAT16 scalar-MUL members, not a representative fixture for a new form or complete matrix evidence.
- The failed vector positive-one fixture is not weakened, excluded, or reclassified. Completed CPU 0008Q1A corrected the complete currently generated vector scalar-power family; this task must now regenerate and re-prove that fixture and every affected projection under schema 64.

## Known limitations

The matrix proves finite source-reachable categories, not all raw immediate bit patterns. It has no performance result, does not establish whole-backend generated/direct performance parity, and cannot change production selection. A new source category, generator change, or incompatible hash invalidates the relevant fixture or projection until regenerated. Non-finite inputs are semantic oracle cases, but non-finite immediates and bounds remain outside this finite projection boundary.

## Validation evidence

Planning-only evidence initially reviewed the required architecture/planning contracts, CPU master plan, 0008Q, blocked 0009, current scalar lowering/IR/power analysis/preparation/generator, and focused 0005F/0005G/0005J/0008J/0008L/0008Q test evidence. Current source admits the operation/type boundary, five access regimes, exact carrier specialization, four strategy names, and source-owned power realizations recorded above.

The later independent planning/documentation review ran `./gradlew :backends:cpu:test --tests '*ScalarImmediateClampMatrixStructuralTest' --rerun-tasks`. Two tests executed and one failed at `CpuScalarImmediateClampMatrixStructuralTest.java:32`: `FIXTURE-POW-FLOAT32-POSITIVE_ONE contains io/github/pho001/synaptik`. That failure selected CPU 0008Q1A. The remediation completed with schema-64 self-contained vector scalar-power bodies before this task resumed.

The implementation context then finalized exactly six test/evidence paths: three new Java test/oracle files and three canonical tab-separated value (TSV) resources. The resources contain 203 semantic fixtures: 28 each for ADD, SUB, MUL, MIN, and MAX; 18 for DIV; 21 for POW; and 24 for CLAMP. Type counts are 51 each for BFLOAT16, FLOAT32, and FLOAT64 and 25 each for INT32 and INT64. The 21 finite code-shaping categories include JVM constant encodings, raw signed zero, seven power categories, and eight ordered clamp-bound pairs. Explicit adjacent exclusions cover BOOL, mixed boundary types, mixed immediate types, integral DIV/POW/CLAMP, and the finite/non-finite immediate or bound boundary.

The compositional ledger contains 256 forms covering all four ordered carrier pairs per numeric type, all five access regimes, rank-zero and zero-work Shapes, contiguous and non-contiguous layouts, and every selected strategy identity. Selected-strategy counts are 11 scalar, 75 parallel-scalar, one vector, and 169 parallel-vector. Exactly four forms expose a materialization candidate, zero select it, and 252 are direct-only. A separate BFLOAT16 assertion proves candidate absence. Candidate availability is therefore evidence about a complete alternative, not an active selected optimization.

Every generated artifact uses current generator schema 64. The unchanged family-specific class-identity distribution is 196 schema-52 and 60 schema-59 artifacts. Exactly two safe projections are retained, both byte-identical caller-orchestration pairs: scalar with parallel-scalar and vector with parallel-vector. No constants-only, fusion-topology, or unsafe projection is claimed.

The retained inspection root `/private/tmp/synaptik-cpu-0008q1-inspection-final` contains 256 generated classes, 256 normalized dossiers, and 256 complete `javap -c -v` reports. Its 257-line manifest has SHA-256 `01c4c8b1a077c5a0acaee59c50210ba5d9b5b730d760c45d17d31267a33a7cd9`. Structural inspection found zero Synaptik method references, `invokedynamic`, reflection, map/string dispatch, boxing, or allocation. It confirmed exact array/segment access, backward ranges, lower-maximum-then-upper-minimum clamp order, scalar tails, and only permitted Vector API or Foreign Function and Memory API calls.

Review strengthened the optimal clean-Java oracle so operation selection occurs outside hot loops and inputs include BFLOAT16 subnormal/raw-NaN and integral-overflow cases. It also enforced canonical TSV LF/final-newline/order and added direct assertions for source closure, schema provenance, materialization counts, signed-zero realizations, and all power realizations. These review changes affected only test/oracle rigor, not the generated inventory or retained evidence.

Validation chronology from the implementation context: the sealed exact-method run passed; the final matrix/review run executed 19 tests with 18 passes and one expected opt-in retained-inspection skip; the pre-review focused command passed eight suites, 96 tests, and two expected skips; and the genuine unfiltered pre-review `cleanTest` plus CPU run passed 157 suites and 818 tests with 27 expected skips and zero failures. After review changed only tests, the focused 19-test run passed, so the full CPU suite was not repeated under the Planning Guide's non-duplication rule. One earlier unrelated `CpuWorkerGroup` race did not recur in its isolated run or either of two later unfiltered runs. `git diff --check` passed. No implementation issue remains.

Clean documentation-focused context `01a0799a-8d14-7870-b032-8e77bcf29feb` independently reviewed the final six paths, retained metadata, architecture/planning contracts, CPU guide, public and code-generation package Javadocs, and glossary using the General, API/Javadoc, Planning, Backend Guide, Developer Guide, and example-format profiles. It changed planning documentation only and did not rerun Java, full CPU, Javadoc, or performance evidence because executable behavior and Javadoc did not change.

## Implementation notes

- The finite basis is compositional but fail-closed: semantic categories retain exact artifacts, while carrier, access, Shape, materialization-candidate, and selected-strategy witnesses close source-derived dimensions without claiming an arbitrary Cartesian projection.
- Exact generated bytes and hashes remain distinct except for the two proved byte-identical caller-orchestration pairs. No constants-only projection survived the completed matrix.
- All generated envelopes use schema 64, while schema 52 and schema 59 remain intentional unchanged class-identity projections for their respective forms.
- No production, schema, dependency, architecture, API, build, capability, lowering, preparation, selection, or Runtime behavior changed.

## Completion summary

- Completed changes: added and independently reviewed the complete finite scalar-immediate and clamp test/evidence matrix with 203 semantic fixtures, 256 generated forms, strict semantic/structural oracles, canonical sealed resources, and two safe byte-identical orchestration projections.
- Files changed or created: three CPU test sources, three CPU TSV resources, this task, CPU master plan, CPU 0009, and roadmap.
- Tests and validation: reused the implementation context's sealed exact-method pass, focused 8-suite/96-test pre-review pass, final 19-test matrix/review result with 18 passes and one expected opt-in skip, and genuine unfiltered 157-suite/818-test/27-expected-skip clean CPU pass. The documentation pass ran Markdown link/anchor/fence, status/dependency, exact-path, newline/whitespace, retained-manifest, and `git diff --check` checks only.
- Documentation-agent review: clean context `01a0799a-8d14-7870-b032-8e77bcf29feb` completed the targeted review and synchronized CPU 0008Q1/0009 planning status.
- Documentation impact: planning documents only. The CPU guide already explains current scalar/clamp semantics, candidate-only materialization, selected strategies, schema-64 vector scalar-power behavior, and the absence of whole-backend performance claims; adding a test-ledger inventory would not change its explanatory contract.
- Javadoc review: public CPU and code-generation package Javadocs remain accurate because no production type, contract, behavior, or schema changed; Javadoc generation was therefore not run.
- Glossary impact: no reusable domain term or existing definition changed. Current entries already distinguish scalar-power realization, exact scalar/clamp identity facts, portable strategy, and candidate-only materialization from selected work.
- Architecture and broader validation impact: none. `ARCHITECTURE.md`, current architecture explanations, ADRs, architecture tests, public API, conformance/integration tests, other modules, production Javadocs, dependencies, Gradle, and schema require no change because this task adds test/evidence closure only.
- Unresolved issues: None.
- Follow-up required: CPU 0009 is now the next Ready task and may reuse this completed finite matrix provenance. Its full generated-coverage and performance closure remains its own scope.

Status: Complete
