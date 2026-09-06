# Task 0009: Portable Generated-Coverage Closure Checkpoint

## Status

Blocked

## Goal

Close the portable CPU generated-coverage checkpoint with a reproducible, source-derived ledger. It remains blocked because completed CPU 0008Q supplies only a strict BFLOAT16 scalar-MUL projection mechanism, not the required complete scalar-immediate/clamp operation/type/carrier/layout/strategy/shape matrix. Complete [CPU 0008Q1](0008q1-finite-scalar-immediate-clamp-matrix.md) and its documentation-focused pass are required before this checkpoint can become `Ready`.

## Scope

- Add a checkpoint test and checked TSV/JSON resources deriving one row per currently supported CPU family/form from `CpuCapabilityProvider`, `CpuPartitionPreparer`, family lowerers, `CpuClassFileKernelGenerator`, and schema 63. A row names operation/family/form, applicable FLOAT64/FLOAT32/BFLOAT16/INT32/INT64/BOOL types, ordered heap primitive-array/native-order `MemorySegment`/mixed carrier pattern, contiguous/general layout, materialized/direct disposition, available scalar/vector/parallel-scalar/parallel-vector strategies, fallback, admission, and production selection.
- Cover pointwise/cast/view/movement; indexing, scatter, fold, ordering, random, scans, ordinary/masked/advanced reductions and arg extrema; softmax/log-softmax, Layer/RMS and batch normalization; Conv1d composition, Conv2d/Conv3d, pools, MATMUL, attention, and losses. Metadata-only and zero-work views get `NO_GENERATED_HOT_LOOP`, not timing obligations. Provider-positive but whole-partition-inadmissible occurrences are not counted as selected generation.
- Generate each finite fixture and retain binary name, entry descriptor, schema, SHA-256, code-shaping facts, normalized body identity, semantic oracle, structural dossier, and evidence disposition: `CURRENT`, `REPRESENTATIVE_ONLY`, `STALE`, `MISSING`, `NON_PASSING`, or `WAIVED`. Reuse requires matching actual hash/form plus protocol scope.
- Inventory every finite code-shaping form and every unit proved by CPU 0008Q1, recording exact class hashes for representative fixtures and exact member-hash provenance for every projected fixture. Do not claim enumeration of arbitrary scalar-immediate or clamp bit patterns. Project only a checked `PROVED_CONSTANTS_ONLY` unit whose automated normalization proves constants-only variance; otherwise record a distinct finite form or fail closed.
- Preserve CPU 0008I corrected 792-class five-fork evidence as `MISSING` (waived, not passed), CPU 0008O as cancelled `NON_PASSING` (only fork 0; `KEEP_SCALAR`; 178/180 V/S failures), and CPU 0008P as `NON_PASSING` pre-seal profitability evidence with production `KEEP_WHOLE_CELL`. Neither admits SIMD or partial selection.
- Freeze a finite gap matrix only for missing, stale, or materially non-representative hot forms. Group only identical normalized body, carrier access, loop shape, numerical algorithm, strategy, and provenance from CPU 0008Q1; otherwise split. Distinct opcode/type/carrier/layout/strategy, scalar-power realization, structurally relevant constant encoding category, edge-value semantic category, or materially distinct shape remains distinct. Each timed row uses the matching optimal clean Java oracle, semantic comparison, Class-File/decompilation dossier, forbidden helper/allocation/boxing/reflection/dispatch scan, and immutable five-fork protocol/gates. Failed, interrupted, unsealed, or unproved-projection rows remain fail-closed.
- Add provider/admission/selection closure and unsupported fail-closed tests. Finalize affected Javadocs, CPU guide, glossary conclusion, and planning evidence in the same change through a separate documentation-focused pass.

## Out of scope

- New semantics, generated algorithms, schemas, strategies, native routes, route-selection policy, tuning, Model/Compiler/Prepare contracts, architecture changes, or a claim that every arbitrary immediate bit pattern was enumerated.
- Promising vector or parallel-vector code for scalar-only families; timing metadata-only rows; treating a reference kernel as production fallback; or calling 0008I/0008O/0008P passed.
- Indiscriminate historical reruns or repository-tracked measurements.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md): generated-kernel, direct-carrier, four-strategy, fail-closed, oracle, and evidence rules.
- [CPU master plan](../master-plan.md): ownership, schema-63 identity, and frontier.
- [Planning Guide](../../../planning-guide.md): checkpoint validation and status requirements.
- [Performance evidence and tuning](../../../../architecture/performance-evidence-and-tuning.md).

## Architecture constraints

- CPU owns lowering, generation, preparation, and evidence; Planning owns capability truth. Provider support never implies admission or selection.
- Generated classes retain exact bytecode-relevant specialization and direct carrier signatures; no Runtime dispatch, address/object identity, or cold extent enters identity unless existing specialization already does so. CPU 0008Q fixture hashes and normalizer are provenance, not a production identity rule.
- The clean Java oracle has equivalent semantics, numerical order, and hot-loop/dataflow shape. Structural review rejects hidden Synaptik helpers, allocation, boxing, reflection, maps, string dispatch, and avoidable virtual dispatch on proved hot paths.

## Package impact

Existing test packages used: `io.github.pho001.synaptik.backend.cpu` and matching `.internal.prepare`, `.internal.lowering`, `.internal.ir`, `.internal.codegen.emit`, `.internal.cache`, and `.internal.executable` packages. No package or production type is added or moved.

## Affected files

Expected implementation paths:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuGeneratedCoverageCheckpointTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuGeneratedCoverageEvidenceTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuGeneratedCoverageUnsupportedBoundaryTest.java`
- `backends/cpu/src/test/resources/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/generated-coverage-inventory.tsv`
- `backends/cpu/src/test/resources/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/generated-coverage-evidence-ledger.tsv`
- `backends/cpu/src/test/resources/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/generated-coverage-gap-matrix.json`
- up to four existing focused CPU family-test paths only to expose a missing fixture
- up to five affected Javadocs/package summaries, CPU guide, or glossary paths
- this task, [CPU master plan](../master-plan.md), and [roadmap](../../../roadmap.md).

## Maximum scope

At most 11 CPU test/evidence paths, five documentation paths, and three planning paths; zero production Java, Gradle, architecture, conformance, or integration paths. If a gap needs generator/lowering/preparer change, a public documentation concept, or more fixtures, stop and propose a family-specific Draft follow-up.

All performance artifacts use a caller-supplied, initially empty, untracked root outside the repository. It contains immutable protocol/environment/source-class-input hashes, class dossiers, raw fork CSVs, decisions, and a SHA-256 manifest. Record its absolute path and manifest digest; never place measurements or generated classes under source trees.

## Stop conditions

Stop without changing production behavior and leave the task `Blocked` or `Incomplete` if source discovery finds an unbounded reachable-form set, a capability/admission/selection contradiction requiring a contract decision, a code-shaping form outside the finite fixture budget, an evidence root that cannot be sealed, or any required gap row that fails, is interrupted, or cannot be structurally matched to its clean-Java oracle. Record the exact family, form, hash, and required family-specific follow-up.

## Acceptance criteria

- The checked inventory is reconciled from current source and has an accountable row for every provider-supported family/form, every finite code-shaping form, and every unit proved by CPU 0008Q1; capability, admission, and selection are distinct.
- Rows truthfully record type/carrier/layout/materialization/strategy/fallback. Vector strategies appear only where the current preparer admits them; scalar or caller-parallel fallback is explicit otherwise.
- Every generated row has reproducible Class-File identity and semantic/structural/oracle mapping. Reused performance evidence is byte-identical at normalized-body and class-hash/form levels with matching protocol scope; projected scalar-immediate/clamp coverage additionally cites exact unit/member hashes and constants-only proof from CPU 0008Q1. Otherwise it is classified without a pass claim.
- 0008I, 0008O, and 0008P retain exactly the stated non-passing dispositions. Forged/incomplete evidence cannot select a route; unsupported provider queries and mixed partitions fail closed.
- The finite gap matrix gives a body-level reason and checked projection provenance for each grouping. Required fresh rows pass sealed five forks or the production form remains fail-closed and this checkpoint is not Complete.
- A separate documentation-focused pass finalizes documentation/Javadoc/glossary impact and explicit no-change conclusions.

## Tests / validation

```bash
./gradlew :backends:cpu:test --tests '*GeneratedCoverage*' --tests '*CapabilityProviderTest' --tests '*PartitionPreparerTest'
./gradlew :backends:cpu:test
```

For a non-empty gap matrix, run its checked harness in exactly five fresh Java 26 forks with the explicit external root; it must reject an absent or changed seal before timing. Run Class-File/decompilation and forbidden-reference scans from that root. Then run:

```bash
./gradlew :backends:cpu:test --tests 'io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneratedCoverageEvidenceTest' -Dsynaptik.cpu.generatedCoverageEvidenceRoot=/absolute/external/evidence-root
./gradlew test
./gradlew :backends:cpu:javadoc
git diff --check
```

Validate TSV/JSON parsing, manifest, Markdown links/anchors/fences, whitespace, path/package ceilings, task/master-plan/roadmap status synchronization, and `git status --short`. Architecture, conformance, and integration suites are unchanged; the full repository suite is this checkpoint validation, not a claim they changed.

## Dependencies

- Complete CPU 0005A–0005J, 0006–0007F2, 0008–0008P, Prepare 0003A, and their ledgers; especially 0008I, 0008M, 0008N/0008N1, 0008O, and 0008P.
- Complete CPU 0008Q1 and its documentation pass, including exact fixture hashes and automated constants-only normalization proof.
- Current Java 26 Class-File/Vector toolchain and a writable external evidence root.

## Follow-up tasks

- A hot form or scalar-immediate/clamp relation that cannot fit the finite matrix, lacks future complete-matrix provenance, or fails its gate becomes one family-specific Draft remediation task; do not enlarge this checkpoint.
- CPU 0010–0015 consume no non-passing or waived performance claim.

## Architecture impact

Expected impact: None. Stop for required architecture, dependency, production-selection, or schema changes.

## Implementation prompt

```text
Read AGENTS.md, ARCHITECTURE.md, the Planning Guide, CPU master plan, and this task. Implement only this bounded inventory/evidence closure; do not add capability or production behavior. Stop for architecture or scope conflict. Do not commit or push. After executable validation, hand the stable diff and exact evidence to a separate clean documentation-focused context for documentation, Javadoc, glossary, and Markdown review. Update status only from results.
```

## Local decisions

- 0009 remains one cohesive checkpoint: it has one output contract, the source-derived inventory and evidence disposition, and changes no production behavior. A family failure is a follow-up, not a backend rewrite.
- `WAIVED` is an owner-approved absence, never passing or production eligibility.
- CPU 0009 is `Blocked` because the preceding attempt reached its explicit unbounded-reachable-form stop condition. Completed CPU 0008Q supplies only a narrow two-fixture scalar-MUL mechanism, not the permitted complete scalar-immediate/clamp projection basis. Return CPU 0009 to `Ready` only after CPU 0008Q1 and its documentation pass are complete.

## Known limitations

Planning has not generated the inventory or run fresh gaps. Current retained evidence is only source input; the task becomes Complete only after every required current row and checkpoint validation pass. It cannot begin while CPU 0008Q1's complete scalar-immediate/clamp matrix projection is unproved.

## Validation evidence

Planning-only evidence: reviewed required contracts; current provider/preparer/schema sources; `CpuKernelIr` scalar/clamp immediates; `CpuKernelSpecialization` class identity; pointwise lowering/preparation/generation and structural evidence tests; CPU 0008I, 0008M, 0008N/0008N1, 0008O, and 0008P ledgers; master-plan row; roadmap; and recent CPU history. No Java, Javadoc, Class-File, or performance command was run for this planning change.

## Implementation notes

Empty until implemented.

## Completion summary

Empty until implemented.
