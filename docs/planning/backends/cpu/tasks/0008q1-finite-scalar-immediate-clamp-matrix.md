# Task 0008Q1: [Finite](../../../../glossary.md#term-finite) [Scalar](../../../../glossary.md#term-scalar)-Immediate and Clamp [Matrix](../../../../glossary.md#term-matrix-meanings)

## Reader context

This task records finite scalar-immediate and clamp coverage derived from CPU source predicates, not
an assumed Cartesian product. `CpuCapabilityProvider` describes admission, `CpuPartitionLowering`
checks whole partitions, and `CpuPartitionPreparer` records eligible and selected strategies
separately. Model owns typed semantics; Runtime invokes prepared artifacts through exact
array/`MemorySegment` carrier signatures. Materialization candidates are alternatives, not selected
work in the recorded matrix.

Independent clean-Java loops serve as semantic oracles; Class-File inspection and mutation controls
serve as structural oracles for carrier access, loops, and prohibited calls. The completed evidence
separates 203 semantic fixtures from 256 generated forms. Only two byte-identical
caller-orchestration projections survive. Generator schema 64 is distinct from unchanged family
class-identity schemas 52 and 59; normalization never relaxes production identity.

Current-topic navigation (not a replacement for the historical record):

- [current exact typed scalar metadata](../../../../reference/tensor/data-types.md#exact-typed-scalar-values) — Distinguishes immutable immediate bits from a rank-zero Tensor or live storage.
- [current generated-class boundary](../../../../explanation/packages/cpu.md#iogithubpho001synaptikbackendcpuinternalcodegenemit) — Explains emitters, typed entries, hidden classes, and the limits of Class-File inspection.

## Status

Complete

## Goal

Establish the complete, [finite](../../../../glossary.md#term-finite), source-derived projection basis for every currently admitted [CPU](../../../../glossary.md#term-cpu) [scalar](../../../../glossary.md#term-scalar)-immediate [operation](../../../../glossary.md#operation) and floating clamp before CPU 0009 inventories generated coverage. Prove generated execution equivalent to the matching optimal clean Java 26 [specialization](../../../../glossary.md#term-specialization) for each distinct form, and project only mechanically proved relations. Do not enumerate arbitrary immediate bit patterns. Completed [CPU 0008Q1A](0008q1a-vector-scalar-power-hot-path-self-containment.md) removed the independently observed Synaptik helper calls from generated [vector](../../../../glossary.md#term-vector) scalar-power hot paths under [schema](../../../../glossary.md#term-schema) 64; this task regenerated and closed its [matrix](../../../../glossary.md#term-matrix-meanings) against that schema.

## [Scope](../../../../glossary.md#term-scope)

- Derive the [matrix](../../../../glossary.md#term-matrix-meanings) from current `CpuCapabilityProvider`, `CpuPartitionLowering`, `CpuKernelIr`, `CpuScalarPowerAnalysis`, `CpuPartitionPreparer`, `CpuKernelSpecialization`, and `CpuClassFileKernelGenerator`; do not start from a manually assumed product.
- Cover every currently capability-admitted [scalar](../../../../glossary.md#term-scalar) [occurrence](../../../../glossary.md#term-occurrence): `ADD`, `SUB`, `MUL`, `DIV`, `MIN`, `MAX`, and `POW`, plus first-class `CLAMP`. Record [Model](../../../../glossary.md#term-model) capability, successful whole-[partition](../../../../glossary.md#partition) [lowering](../../../../glossary.md#lowering), preparation eligibility, and actual selected strategy as separate facts.
- Cover [BFLOAT16](../../../../glossary.md#term-bfloat16), [FLOAT32](../../../../glossary.md#term-float32), and [FLOAT64](../../../../glossary.md#term-float64) for every floating scalar [operation](../../../../glossary.md#operation) and clamp; [INT32](../../../../glossary.md#data-type--datatype) and [INT64](../../../../glossary.md#data-type--datatype) only for scalar ADD/SUB/MUL/MIN/MAX; and no [BOOL](../../../../glossary.md#data-type--datatype), mixed-type scalar, integral DIV/POW, or integral clamp row. Assert each adjacent exclusion fails before [artifact](../../../../glossary.md#term-artifact) construction.
- Cover every generated direct-[carrier](../../../../glossary.md#term-storage-carrier) [signature](../../../../glossary.md#term-signature) actually reachable for each type: matching [heap](../../../../glossary.md#term-heap) primitive arrays, native-order [`MemorySegment`](../../../../glossary.md#term-memorysegment)s, and every ordered legal array/segment mixed input/output pair. Keep a carrier pair separate whenever it changes entry [descriptor](../../../../glossary.md#term-descriptor), carrier load/store instruction, address arithmetic, or helper/invoke set.
- Cover direct inputs/outputs in `DENSE_LINEAR`, `SCALAR_ALL_ZERO`, `LAST_AXIS_BIAS`, `BLOCK_OUTER`, and `GENERAL_ODOMETER` whenever current lowering admits that regime. Exercise [rank-zero](../../../../glossary.md#axis-rank-scalar-vector-and-matrix), zero-work, one-dimensional [contiguous](../../../../glossary.md#term-contiguous), and minimally shaped right-aligned broadcast/non-contiguous witnesses only when they select a different generated loop or [dataflow](../../../../glossary.md#term-dataflow) [shape](../../../../glossary.md#shape). Record [materialization](../../../../glossary.md#materialization) candidates and selection; include a materialized form only when current preparation can actually select it.
- Cover `SCALAR`, caller-parallel `PARALLEL_SCALAR`, [`VECTOR`](../../../../glossary.md#term-vector), and `PARALLEL_VECTOR` only when the preparer admits and selects them. Artifact sharing is permitted only after test proof that it is orchestration sharing; otherwise strategy changes remain separate.
- Derive [finite](../../../../glossary.md#term-finite) immediate categories from source and emitted code shape. Ordinary arithmetic/[extrema](../../../../glossary.md#saturation-pole-and-extremum) categories are distinct [constant](../../../../glossary.md#term-constant)-encoding and semantic-edge categories discovered from the current emitter; scalar power is split by `POSITIVE_ONE`, [`IDENTITY`](../../../../glossary.md#term-identity-meanings), `SQUARE`, `RECIPROCAL`, and finite `DIRECT`; clamp is split by ordered lower/upper category pairs and any distinct lower-first/upper-second control-flow or constant-encoding shape. Include [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero) ordering where it changes clamp/extrema semantics. Non-finite inputs remain [oracle](../../../../glossary.md#term-test-oracle) edge cases, but non-finite immediate or bound patterns are outside this finite matrix.
- Generate an exact [fixture](../../../../glossary.md#term-fixture) for every discovered non-projectable form. A projected unit records exact member [hashes](../../../../glossary.md#term-hash), [specialization](../../../../glossary.md#term-specialization) facts, normalizer version, declared constant locations, and source-derived category. It is `PROVED_CONSTANTS_ONLY` only if automated Class-File [normalization](../../../../glossary.md#term-normalization) proves equal members, method structure, control flow, invokes, carrier accesses, loops, and dataflow except for declared constants. Otherwise split or [fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation).
- Execute every exact fixture and an independent typed optimal clean-Java loop across ordinary values and applicable [NaN](../../../../glossary.md#term-nan), [infinity](../../../../glossary.md#term-infinity), signed-zero, [subnormal](../../../../glossary.md#term-subnormal), [overflow](../../../../glossary.md#term-overflow)/[underflow](../../../../glossary.md#term-underflow), and clamp-bound edge inputs. Preserve the operation's numerical order, BFLOAT16 decode/one-operation/encode boundary, carrier access, loop shape, and [scalar-power realization](../../../../glossary.md#scalar-power-realization). The oracle must not call generated code, the reference [kernel](../../../../glossary.md#kernel), or [CPU](../../../../glossary.md#term-cpu) lowering.
- Add automated structural dossiers proving the matching clean-Java algorithm and generated body have no hidden Synaptik helper, allocation, boxing, reflection, map/string [dispatch](../../../../glossary.md#term-dispatch), generic carrier branch, or avoidable virtual/interface dispatch on the proved [hot path](../../../../glossary.md#term-hot-path). Keep hashes and normalized-member [provenance](../../../../glossary.md#provenance) in checked resources for CPU 0009.

## Out of [scope](../../../../glossary.md#term-scope)

- Production Java, [Model](../../../../glossary.md#term-model)/[Compiler](../../../../glossary.md#term-compiler)/[Prepare](../../../../glossary.md#prepare)/[Runtime](../../../../glossary.md#term-runtime) contracts, [schemas](../../../../glossary.md#term-schema), [cache](../../../../glossary.md#term-cache) [identity](../../../../glossary.md#term-identity-meanings), capability rules, route or strategy-selection policy, new [materialization](../../../../glossary.md#materialization) policy, performance tuning, native routes, or arbitrary immediate-bit enumeration.
- Projection across an [opcode](../../../../glossary.md#term-opcode), type, [scalar-power realization](../../../../glossary.md#scalar-power-realization), ordered clamp category, [carrier](../../../../glossary.md#term-storage-carrier) access, access regime, materialization disposition, selected compute strategy, [vector species](../../../../glossary.md#term-vector-species), loop/[dataflow](../../../../glossary.md#term-dataflow)/control-flow [shape](../../../../glossary.md#shape), numerical order, invoke set, or semantic edge category that changes generated code or meaning.
- Five-[fork](../../../../glossary.md#term-fork-meanings) performance claims or [CPU](../../../../glossary.md#term-cpu) 0009's full generated-coverage inventory.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md): [CPU](../../../../glossary.md#term-cpu) owns [backend](../../../../glossary.md#backend) [lowering](../../../../glossary.md#lowering)/selection; generated code preserves the optimal clean-Java semantic algorithm and hot-loop/[dataflow](../../../../glossary.md#term-dataflow) [shape](../../../../glossary.md#shape); unproved [specialization](../../../../glossary.md#term-specialization) is [fail-closed](../../../../glossary.md#term-fail-closed).
- [Planning Guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>: status, bounded [scope](../../../../glossary.md#term-scope), [validation](../../../../glossary.md#term-validation), and documentation [handoff](../../../../glossary.md#term-handoff).
- [CPU master plan](../master-plan.md): CPU package [ownership](../../../../glossary.md#term-ownership-meanings), four strategies, and task order.
- [CPU 0005F](0005f-floating-division-and-exact-scalar-power-realization.md), [CPU 0005G](0005g-extrema-clamp-tensor-power-and-logical-coverage.md), [CPU 0005J](0005j-bounded-pointwise-coverage-and-parity-hardening.md), [CPU 0008J](0008j-bfloat16-scalar-pointwise-closure.md), [CPU 0008L](0008l-pointwise-simd-mask-output-closure.md), and [CPU 0008Q](0008q-scalar-immediate-clamp-generated-equivalence.md).
- [CPU 0008Q1A](0008q1a-vector-scalar-power-hot-path-self-containment.md): required production remediation discovered by this task's independent structural test.

## Architecture constraints

- [Model](../../../../glossary.md#term-model) owns [scalar](../../../../glossary.md#term-scalar) and clamp semantics. [CPU](../../../../glossary.md#term-cpu) preserves exact typed raw immediate/bound bits in private [IR](../../../../glossary.md#term-intermediate-representation-ir), selects preparation facts before finalization, and [Runtime](../../../../glossary.md#term-runtime) only invokes the prepared [artifact](../../../../glossary.md#term-artifact). No generated [hot path](../../../../glossary.md#term-hot-path) sees [`Operation`](../../../../glossary.md#operation), `CompiledNode`, [layouts](../../../../glossary.md#layout), or string [dispatch](../../../../glossary.md#term-dispatch).
- Capability, complete-[partition](../../../../glossary.md#partition) [admission](../../../../glossary.md#term-admission), and actual selection remain separate checked columns. A provider-positive form that cannot lower, bind, or select is [fail-closed](../../../../glossary.md#term-fail-closed), not [matrix](../../../../glossary.md#term-matrix-meanings) coverage.
- Exact [raw bits](../../../../glossary.md#storage-mapping-contiguity-and-injectivity) remain [specialization](../../../../glossary.md#term-specialization) [identity](../../../../glossary.md#term-identity-meanings) facts even where a normalizer proves a [constants](../../../../glossary.md#term-constant)-only relation. Test projection is evidence only and never relaxes production identity or [cache](../../../../glossary.md#term-cache) reuse.
- Generated code and clean Java [oracle](../../../../glossary.md#term-test-oracle) use the same semantic order and direct typed [carrier](../../../../glossary.md#term-storage-carrier) accesses. Direct scalar power cannot inherit identity, multiply, or reciprocal proof.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` for fixtures, independent oracles, Class-File [normalization](../../../../glossary.md#term-normalization), structural assertions, and checked resources.
- Existing `.internal.lowering`, `.internal.ir`, `.internal.prepare`, and `.internal.cache` test packages only for a missing source-derived assertion.

Packages added or changed:

- None. No production type is added or moved.

Type placement:

- `...internal.codegen.emit.CpuScalarImmediateClampMatrixTest` — [fixture](../../../../glossary.md#term-fixture), capability/[admission](../../../../glossary.md#term-admission)/selection, [oracle](../../../../glossary.md#term-test-oracle), and resource closure.
- `...internal.codegen.emit.CpuScalarImmediateClampMatrixStructuralTest` — Class-File/[dataflow](../../../../glossary.md#term-dataflow) and normalizer mutation controls.
- `...internal.codegen.emit.CpuScalarImmediateClampMatrixOracle` — package-private independent clean-Java specializations; never a production [dependency](../../../../glossary.md#term-dependency).

## Affected files

Expected implementation paths:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuScalarImmediateClampMatrixTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuScalarImmediateClampMatrixStructuralTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuScalarImmediateClampMatrixOracle.java`
- Up to two existing focused [CPU](../../../../glossary.md#term-cpu) test paths, only for an omitted capability/[admission](../../../../glossary.md#term-admission)/selection witness.
- `backends/cpu/src/test/resources/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/scalar-immediate-clamp-matrix-fixtures.tsv`
- `backends/cpu/src/test/resources/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/scalar-immediate-clamp-matrix-projections.tsv`
- `backends/cpu/src/test/resources/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/scalar-immediate-clamp-matrix-forms.tsv`
- Up to three directly affected [planning](../../../../glossary.md#term-planning)/[Javadoc](../../../../glossary.md#term-javadoc)/glossary paths, this task, [CPU master plan](../master-plan.md), [CPU 0009](0009-portable-generated-coverage-closure-checkpoint.md), and [roadmap](../../../roadmap.md).

## Maximum [scope](../../../../glossary.md#term-scope)

At most five [CPU](../../../../glossary.md#term-cpu) test sources, three checked [matrix](../../../../glossary.md#term-matrix-meanings) resources, three documentation paths, and four [planning](../../../../glossary.md#term-planning) paths; zero production Java, [Gradle](../../../../glossary.md#term-gradle), architecture, [conformance](../../../../glossary.md#term-conformance), or integration paths. The [fixture](../../../../glossary.md#term-fixture) count is source-derived rather than capped. If it cannot fit the three resources and compositional mechanically verified [dimensions](../../../../glossary.md#dimension), or needs production behavior, stop and create a family-specific Draft follow-up rather than hand-maintaining an unbounded [Cartesian product](../../../../glossary.md#term-cartesian-product).

## Stop conditions

Stop and leave this task `Incomplete` if any reachable [finite](../../../../glossary.md#term-finite) category cannot be classified from source; an admitted form lacks an independent optimal clean-Java [oracle](../../../../glossary.md#term-test-oracle); a required [carrier](../../../../glossary.md#term-storage-carrier)/[layout](../../../../glossary.md#layout)/strategy witness changes unaccounted code [shape](../../../../glossary.md#shape); a normalizer permits a nonconstant difference; source capability/[admission](../../../../glossary.md#term-admission)/selection facts contradict one another; a semantic/structural test fails; or work needs another production, architecture, [schema](../../../../glossary.md#term-schema), or selection change. Record exact form, source predicate, [fixture](../../../../glossary.md#term-fixture)/[hash](../../../../glossary.md#term-hash) where available, and required follow-up. Do not make [CPU](../../../../glossary.md#term-cpu) 0009 [`Ready`](../../../../glossary.md#term-ready).

## Acceptance criteria

- Checked resources reconcile every admitted [scalar](../../../../glossary.md#term-scalar) [operation](../../../../glossary.md#operation)/clamp type row and every actual lowerable [carrier](../../../../glossary.md#term-storage-carrier)/access/[layout](../../../../glossary.md#layout)/strategy/[materialization](../../../../glossary.md#materialization) form, with explicit capability, [admission](../../../../glossary.md#term-admission), and selection results plus adjacent [fail-closed](../../../../glossary.md#term-fail-closed) exclusions.
- The category table is source-derived, includes every [scalar-power realization](../../../../glossary.md#scalar-power-realization) and ordered clamp pair needing distinct semantics or emission, and never claims arbitrary immediate/bound enumeration.
- Every non-projectable form has exact generated [hash](../../../../glossary.md#term-hash), [descriptor](../../../../glossary.md#term-descriptor), [schema](../../../../glossary.md#term-schema), structural key, carrier pattern, regime, selected strategy, [shape](../../../../glossary.md#shape) witness, category, [oracle](../../../../glossary.md#term-test-oracle), and dossier. Every projection names exact members/hashes and passes strict [constants](../../../../glossary.md#term-constant)-only [normalization](../../../../glossary.md#term-normalization).
- Generated-vs-optimal-clean-Java comparisons cover every exact [fixture](../../../../glossary.md#term-fixture) and applicable ordinary/special inputs. Structural checks prove matching carrier and loop/[dataflow](../../../../glossary.md#term-dataflow) shape and reject prohibited mechanisms.
- [Artifact](../../../../glossary.md#term-artifact) sharing is proven only for orchestration sharing; all other code-shaping changes remain distinct.
- A separate documentation-focused agent pass finalizes affected [planning](../../../../glossary.md#term-planning)/[Javadoc](../../../../glossary.md#term-javadoc)/glossary impact in the same change and records explicit no-change conclusions where applicable.

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run):

```bash
./gradlew :backends:cpu:test --tests '*ScalarImmediateClampMatrix*' --tests '*ScalarImmediateClampEquivalence*' --tests '*CpuPointwiseGeneratedKernelTest' --tests '*CpuPointwisePartitionLoweringTest' --tests '*CpuScalarPowerAnalysisTest' --rerun-tasks
```

Documentation pass:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
```

Validate exact TSV headers/columns/tabs/digests; [hashes](../../../../glossary.md#term-hash); bidirectional [fixture](../../../../glossary.md#term-fixture)/projection membership; source-derived closure; Class-File verification/[decompilation](../../../../glossary.md#term-decompilation); normalizer mutation controls; local Markdown links/anchors/fences; status/[dependency](../../../../glossary.md#term-dependency)/path-limit consistency; and `git status --short`. Repository-wide validation is deferred to [CPU](../../../../glossary.md#term-cpu) 0009/[CI](../../../../glossary.md#term-continuous-integration-ci); architecture, [conformance](../../../../glossary.md#term-conformance), and integration tests are unchanged unless this task stops for a boundary change.

## Dependencies

- Complete [CPU](../../../../glossary.md#term-cpu) 0005F, 0005G, 0005J, 0008J, 0008L, 0008Q, and CPU 0008Q1A with its separate documentation pass; then-current [pointwise](../../../../glossary.md#term-pointwise) [lowering](../../../../glossary.md#lowering)/preparation/generation [schema](../../../../glossary.md#term-schema); Java 26 Class-File/[Vector](../../../../glossary.md#term-vector) [toolchain](../../../../glossary.md#term-toolchain).
- Current source is the only [matrix](../../../../glossary.md#term-matrix-meanings) authority. This task cannot infer a missing [operation](../../../../glossary.md#operation), [carrier](../../../../glossary.md#term-storage-carrier), [layout](../../../../glossary.md#layout), vector route, or [materialization](../../../../glossary.md#materialization) form from history.

## Follow-up tasks

- [CPU](../../../../glossary.md#term-cpu) 0009 may become [`Ready`](../../../../glossary.md#term-ready) only after this task and its separate documentation pass are `Complete` with a [finite](../../../../glossary.md#term-finite) proved [matrix](../../../../glossary.md#term-matrix-meanings).
- A category or relation that fails semantic or [constants](../../../../glossary.md#term-constant)-only proof becomes a narrowly named Draft remediation task; it is not projected into CPU 0009.

## Architecture impact

Expected impact: None. Stop for an architecture, [dependency](../../../../glossary.md#term-dependency), production-[identity](../../../../glossary.md#term-identity-meanings), [schema](../../../../glossary.md#term-schema), capability, selection, [carrier](../../../../glossary.md#term-storage-carrier), or semantics change.

## Implementation prompt

```text
You are working in the Synaptik repository after CPU 0008Q1A and its documentation pass are Complete. Read AGENTS.md, ARCHITECTURE.md, the Planning Guide, the CPU master plan, CPU 0008Q, CPU 0008Q1A, CPU 0009, and this task. Implement this test/resource matrix exactly as specified; do not change production behavior or enumerate arbitrary immediate bits. Stop for architecture or scope conflict. Do not commit or push. After executable validation, hand the stable diff and exact evidence to a separate clean documentation-focused context. That pass must follow `docs/developer-guide/documentation-rules.md` and finalize planning/Javadoc/glossary review. Update this task with results; do not mark it Complete before that pass finishes.
```

## Local decisions

- The [finite](../../../../glossary.md#term-finite) [matrix](../../../../glossary.md#term-matrix-meanings) composes source-derived [dimensions](../../../../glossary.md#dimension) rather than a manually maintained product. Composition is allowed only after executable proof that the omitted dimension cannot change instructions, control flow, invokes, [carrier](../../../../glossary.md#term-storage-carrier) access, loop/[dataflow](../../../../glossary.md#term-dataflow), numerical order, or semantics.
- A finite immediate category is an emitted-code and semantic boundary, not a decimal-value bucket. `SCALAR_POW` realizations are separate before [normalization](../../../../glossary.md#term-normalization); clamp bounds retain lower-then-upper order.
- Completed 0008Q is a reusable normalizer/test [seed](../../../../glossary.md#term-seed) for two [BFLOAT16](../../../../glossary.md#term-bfloat16) [scalar](../../../../glossary.md#term-scalar)-MUL members, not a representative [fixture](../../../../glossary.md#term-fixture) for a new form or complete matrix evidence.
- The failed [vector](../../../../glossary.md#term-vector) positive-one fixture is not weakened, excluded, or reclassified. Completed [CPU](../../../../glossary.md#term-cpu) 0008Q1A corrected the complete currently generated vector scalar-power family; this task must now regenerate and re-prove that fixture and every affected projection under [schema](../../../../glossary.md#term-schema) 64.

## Known limitations

The [matrix](../../../../glossary.md#term-matrix-meanings) proves [finite](../../../../glossary.md#term-finite) source-reachable categories, not all raw immediate bit patterns. It has no performance result, does not establish whole-[backend](../../../../glossary.md#backend) generated/direct performance parity, and cannot change production selection. A new source category, generator change, or incompatible [hash](../../../../glossary.md#term-hash) invalidates the relevant [fixture](../../../../glossary.md#term-fixture) or projection until regenerated. Non-finite inputs are semantic [oracle](../../../../glossary.md#term-test-oracle) cases, but non-finite immediates and bounds remain outside this finite projection boundary.

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning)-only evidence initially reviewed the required architecture/planning contracts, [CPU](../../../../glossary.md#term-cpu) master plan, 0008Q, blocked 0009, current [scalar](../../../../glossary.md#term-scalar) [lowering](../../../../glossary.md#lowering)/[IR](../../../../glossary.md#term-intermediate-representation-ir)/power analysis/preparation/generator, and focused 0005F/0005G/0005J/0008J/0008L/0008Q test evidence. Current source admits the [operation](../../../../glossary.md#operation)/type boundary, five access regimes, exact [carrier](../../../../glossary.md#term-storage-carrier) [specialization](../../../../glossary.md#term-specialization), four strategy names, and source-owned power realizations recorded above.

The later independent planning/documentation review ran `./gradlew :backends:cpu:test --tests '*ScalarImmediateClampMatrixStructuralTest' --rerun-tasks`. Two tests executed and one failed at `CpuScalarImmediateClampMatrixStructuralTest.java:32`: `FIXTURE-POW-FLOAT32-POSITIVE_ONE contains io/github/pho001/synaptik`. That failure selected CPU 0008Q1A. The remediation completed with [schema](../../../../glossary.md#term-schema)-64 self-contained [vector](../../../../glossary.md#term-vector) scalar-power bodies before this task resumed.

The implementation context then finalized exactly six test/evidence paths: three new Java test/[oracle](../../../../glossary.md#term-test-oracle) files and three [canonical](../../../../glossary.md#term-canonical-form) tab-separated value (TSV) resources. The resources contain 203 semantic fixtures: 28 each for ADD, SUB, MUL, MIN, and MAX; 18 for DIV; 21 for POW; and 24 for CLAMP. Type counts are 51 each for [BFLOAT16](../../../../glossary.md#term-bfloat16), [FLOAT32](../../../../glossary.md#term-float32), and [FLOAT64](../../../../glossary.md#term-float64) and 25 each for [INT32](../../../../glossary.md#data-type--datatype) and [INT64](../../../../glossary.md#data-type--datatype). The 21 [finite](../../../../glossary.md#term-finite) code-shaping categories include [JVM](../../../../glossary.md#term-java-virtual-machine-jvm) [constant](../../../../glossary.md#term-constant) encodings, raw [signed zero](../../../../glossary.md#term-signed-zero), seven power categories, and eight ordered clamp-bound pairs. Explicit adjacent exclusions cover [BOOL](../../../../glossary.md#data-type--datatype), mixed boundary types, mixed immediate types, integral DIV/POW/CLAMP, and the finite/non-finite immediate or bound boundary.

The compositional ledger contains 256 forms covering all four ordered carrier pairs per numeric type, all five access regimes, [rank-zero](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) and zero-work [Shapes](../../../../glossary.md#shape), [contiguous](../../../../glossary.md#term-contiguous) and non-contiguous [layouts](../../../../glossary.md#layout), and every selected strategy [identity](../../../../glossary.md#term-identity-meanings). Selected-strategy counts are 11 scalar, 75 parallel-scalar, one vector, and 169 parallel-vector. Exactly four forms expose a [materialization](../../../../glossary.md#materialization) candidate, zero select it, and 252 are direct-only. A separate BFLOAT16 assertion proves candidate absence. Candidate availability is therefore evidence about a complete alternative, not an active selected optimization.

Every generated [artifact](../../../../glossary.md#term-artifact) uses current generator schema 64. The unchanged family-specific class-identity [distribution](../../../../glossary.md#term-distribution) is 196 schema-52 and 60 schema-59 artifacts. Exactly two safe projections are retained, both byte-identical caller-orchestration pairs: scalar with parallel-scalar and vector with parallel-vector. No constants-only, [fusion](../../../../glossary.md#term-fusion)-[topology](../../../../glossary.md#term-topology), or unsafe projection is claimed.

The retained inspection root `/private/tmp/synaptik-cpu-0008q1-inspection-final` contains 256 generated classes, 256 normalized dossiers, and 256 complete `javap -c -v` reports. Its 257-line [manifest](../../../../glossary.md#term-manifest) has [SHA-256](../../../../glossary.md#term-sha-256) `01c4c8b1a077c5a0acaee59c50210ba5d9b5b730d760c45d17d31267a33a7cd9`. Structural inspection found zero Synaptik method references, `invokedynamic`, reflection, map/string [dispatch](../../../../glossary.md#term-dispatch), boxing, or allocation. It confirmed exact array/segment access, backward ranges, lower-maximum-then-upper-minimum clamp order, scalar tails, and only permitted [Vector API](../../../../glossary.md#term-vector-api) or [Foreign Function and Memory API](../../../../glossary.md#application-binary-interface-and-native-bridge) calls.

Review strengthened the optimal clean-Java oracle so operation selection occurs outside hot loops and inputs include BFLOAT16 [subnormal](../../../../glossary.md#term-subnormal)/raw-[NaN](../../../../glossary.md#term-nan) and integral-[overflow](../../../../glossary.md#term-overflow) cases. It also enforced canonical TSV LF/final-newline/order and added direct assertions for source closure, schema [provenance](../../../../glossary.md#provenance), materialization counts, [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero) realizations, and all power realizations. These review changes affected only test/oracle rigor, not the generated inventory or retained evidence.

Validation chronology from the implementation context: the [sealed](../../../../glossary.md#term-sealed) exact-method [run](../../../../glossary.md#term-run) passed; the final [matrix](../../../../glossary.md#term-matrix-meanings)/review run executed 19 tests with 18 passes and one expected opt-in retained-inspection skip; the pre-review focused command passed eight suites, 96 tests, and two expected skips; and the genuine unfiltered pre-review `cleanTest` plus CPU run passed 157 suites and 818 tests with 27 expected skips and zero failures. After review changed only tests, the focused 19-test run passed, so the full CPU suite was not repeated under the Planning Guide's non-duplication rule. One earlier unrelated [`CpuWorkerGroup`](../../../../glossary.md#cpu-worker-group--cpuworkergroup) [race](../../../../glossary.md#term-race) did not recur in its isolated run or either of two later unfiltered runs. `git diff --check` passed. No implementation issue remains.

Clean documentation-focused context `01a0799a-8d14-7870-b032-8e77bcf29feb` independently reviewed the final six paths, retained [metadata](../../../../glossary.md#term-metadata), architecture/planning contracts, CPU guide, public and code-generation package Javadocs, and glossary using the General, [API](../../../../glossary.md#term-application-programming-interface-api)/[Javadoc](../../../../glossary.md#term-javadoc), Planning, [Backend](../../../../glossary.md#backend) Guide, Developer Guide, and example-format profiles. It changed planning documentation only and did not rerun Java, full CPU, Javadoc, or performance evidence because executable behavior and Javadoc did not change.

## Implementation notes

- The [finite](../../../../glossary.md#term-finite) basis is compositional but [fail-closed](../../../../glossary.md#term-fail-closed): semantic categories retain exact [artifacts](../../../../glossary.md#term-artifact), while [carrier](../../../../glossary.md#term-storage-carrier), access, [Shape](../../../../glossary.md#shape), [materialization](../../../../glossary.md#materialization)-candidate, and selected-strategy witnesses close source-derived [dimensions](../../../../glossary.md#dimension) without claiming an arbitrary Cartesian projection.
- Exact generated bytes and [hashes](../../../../glossary.md#term-hash) remain distinct except for the two proved byte-identical caller-orchestration pairs. No [constants](../../../../glossary.md#term-constant)-only projection survived the completed [matrix](../../../../glossary.md#term-matrix-meanings).
- All generated envelopes use [schema](../../../../glossary.md#term-schema) 64, while schema 52 and schema 59 remain intentional unchanged class-[identity](../../../../glossary.md#term-identity-meanings) projections for their respective forms.
- No production, schema, [dependency](../../../../glossary.md#term-dependency), architecture, [API](../../../../glossary.md#term-application-programming-interface-api), build, capability, [lowering](../../../../glossary.md#lowering), preparation, selection, or [Runtime](../../../../glossary.md#term-runtime) behavior changed.

## Completion summary

- Completed changes: added and independently reviewed the complete [finite](../../../../glossary.md#term-finite) [scalar](../../../../glossary.md#term-scalar)-immediate and clamp test/evidence [matrix](../../../../glossary.md#term-review-matrix) with 203 semantic fixtures, 256 generated forms, strict semantic/structural oracles, [canonical](../../../../glossary.md#term-canonical-form) [sealed](../../../../glossary.md#term-sealed) resources, and two safe byte-identical orchestration projections.
- Files changed or created: three [CPU](../../../../glossary.md#term-cpu) test sources, three CPU TSV resources, this task, CPU master plan, CPU 0009, and roadmap.
- Tests and [validation](../../../../glossary.md#term-validation): reused the implementation context's sealed exact-method pass, focused 8-suite/96-test pre-review pass, final 19-test matrix/review result with 18 passes and one expected opt-in skip, and genuine unfiltered 157-suite/818-test/27-expected-skip clean CPU pass. The documentation pass ran Markdown link/anchor/fence, status/[dependency](../../../../glossary.md#term-dependency), exact-path, newline/whitespace, retained-[manifest](../../../../glossary.md#term-manifest), and `git diff --check` checks only.
- Documentation-agent review: clean context `01a0799a-8d14-7870-b032-8e77bcf29feb` completed the targeted review and synchronized CPU 0008Q1/0009 [planning](../../../../glossary.md#term-planning) status.
- Documentation impact: planning documents only. The CPU guide already explains current scalar/clamp semantics, candidate-only [materialization](../../../../glossary.md#materialization), selected strategies, [schema](../../../../glossary.md#term-schema)-64 [vector](../../../../glossary.md#term-vector) scalar-power behavior, and the absence of whole-[backend](../../../../glossary.md#backend) performance claims; adding a test-ledger inventory would not change its explanatory contract.
- [Javadoc](../../../../glossary.md#term-javadoc) review: public CPU and code-generation package Javadocs remain accurate because no production type, contract, behavior, or schema changed; Javadoc generation was therefore not [run](../../../../glossary.md#term-run).
- Glossary impact: no reusable domain term or existing definition changed. Current entries already distinguish [scalar-power realization](../../../../glossary.md#scalar-power-realization), exact scalar/clamp [identity](../../../../glossary.md#term-identity-meanings) facts, portable strategy, and candidate-only materialization from selected work.
- Architecture and broader validation impact: none. `ARCHITECTURE.md`, current architecture explanations, ADRs, architecture tests, public [API](../../../../glossary.md#term-application-programming-interface-api), [conformance](../../../../glossary.md#term-conformance)/integration tests, other [modules](../../../../glossary.md#term-module), production Javadocs, dependencies, [Gradle](../../../../glossary.md#term-gradle), and schema require no change because this task adds test/evidence closure only.
- Unresolved issues: None.
- Follow-up required: CPU 0009 is now the next [Ready](../../../../glossary.md#term-ready) task and may reuse this completed finite [matrix](../../../../glossary.md#term-matrix-meanings) [provenance](../../../../glossary.md#provenance). Its full generated-coverage and performance closure remains its own [scope](../../../../glossary.md#term-scope).

Status: Complete
