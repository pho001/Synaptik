# Task 0008Q: Finite Scalar-Immediate and Clamp Generated-Code Equivalence

## Status

Complete

## Goal

Establish the smallest source-derived, fail-closed generated-code projection mechanism that CPU 0009 can later extend: the exact BFLOAT16 scalar-MUL represented-immediate pair on the current contiguous heap scalar route. This task does not establish a coverage-equivalence contract for every scalar-immediate or clamp form; 0009 remains blocked pending the separate complete matrix task.

## Scope

- Add focused CPU test support and checked resources for exactly BFLOAT16 scalar MUL, `SHORT_ARRAY` input/output carriers, contiguous layout, scalar strategy, and shape `[8]`. It is a complete two-immediate mechanism fixture, not a projection for unrepresented dimensions.
- Keep distinct opcode/type/carrier/layout/strategy, scalar-power realization, structurally relevant constant encoding category, edge-value semantic category, and materially distinct shape as separate evidence units. Clamp retains ordered lower/upper categories.
- Record each fixture's exact production binary name, specialization structural key, Class-File SHA-256, fixture-resource SHA-256, entry descriptor, schema, and normalization result. Do not alter `CpuKernelIr`, `CpuKernelSpecialization`, lowering, preparation, generation, naming, or identity projection.
- Implement a test-only automated Class-File normalizer. A unit is accepted only when differing code/constant operands decode to declared immediate(s), every non-immediate constant matches, and opcode/control-flow/invoke/carrier/hot-loop/dataflow shape matches. Exact class hashes remain fixture-specific.
- Add semantic tests against an optimal clean Java oracle with the same typed operation, numerical order, immediate conversion, branch behavior, carrier access, and hot-loop/dataflow shape. Add structural tests rejecting allocation, boxing, reflection, maps, string dispatch, hidden Synaptik helpers, and avoidable dispatch on proved hot paths.
- Emit a checked mechanism-projection resource with exact member hashes, normalizer version, full dimensions, and canonical fixture-resource SHA-256. CPU 0009 may not consume it until a later task supplies the omitted matrix.

## Out of scope

- Enumerating arbitrary immediate bit patterns or claiming every scalar value has equivalent generated code.
- Production identity, fixture hashes, schema, lowering, codegen, capability, selection, semantics, vector eligibility, timing, broad coverage inventory, or CPU 0009 completion.
- Grouping across scalar-power realization, constant encoding category, semantic edge category, type, carrier, layout, strategy, opcode, or materially distinct generated shape.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md): CPU ownership, direct carrier signatures, existing specialization identity, and optimal-clean-Java semantic/hot-path parity.
- [CPU master plan](../master-plan.md): schema-63 identity and ordered frontier.
- [CPU 0009](0009-portable-generated-coverage-closure-checkpoint.md): downstream inventory/evidence consumer.
- [Planning Guide](../../../planning-guide.md): bounded Ready-task scope and validation.

## Architecture constraints

- Existing `CpuKernelIr` and `CpuKernelSpecialization` identity remains exact; no runtime/address/object/extent/worker/artifact-root fact enters identity and this task changes no such rule.
- Normalization is test-only evidence: it cannot rewrite a class, normalize production identity, merge artifacts, authorize cache reuse, or change selection.
- If normalization cannot identify exact immediate operands or finds a changed instruction, branch, invoke, carrier access, loop/dataflow shape, or non-immediate constant, leave the form ungrouped and fail closed.

## Package impact

Existing test packages used:

- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit`
- `io.github.pho001.synaptik.backend.cpu.internal.lowering`
- `io.github.pho001.synaptik.backend.cpu.internal.prepare`

No production package or type is added, moved, or changed.

## Affected files

Expected implementation paths:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuScalarImmediateClampEquivalenceTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuScalarImmediateClampEquivalenceStructuralTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuScalarImmediateClampEquivalenceOracle.java`
- `backends/cpu/src/test/resources/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/scalar-immediate-clamp-fixtures.tsv`
- `backends/cpu/src/test/resources/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/scalar-immediate-clamp-equivalence.tsv`
- this task, [CPU 0009](0009-portable-generated-coverage-closure-checkpoint.md), [CPU master plan](../master-plan.md), and [roadmap](../../../roadmap.md).

## Maximum scope

At most three CPU test-source paths, two CPU test-resource paths, and four planning paths; zero production Java, Javadoc, backend-guide, glossary, Gradle, architecture, conformance, integration, or performance-artifact paths. If a fixture exposes a production difference or needs a new source fixture family, stop and create a bounded follow-up.

## Stop conditions

Stop without changing production behavior and leave the task `Blocked` or `Incomplete` if any
reachable scalar-immediate/clamp form cannot be reduced to a finite fixture set, a proposed unit
changes an instruction, control-flow/invoke/carrier/hot-loop/dataflow shape, a fixture cannot retain
its exact existing identity/hash, the normalizer cannot prove the declared constants-only relation,
or the matching clean-Java semantic/structural test fails. Record the exact opcode, type, carrier,
layout, strategy, immediate category, fixture hash, and required family-specific follow-up.

## Acceptance criteria

- The checked fixture resource is finite, gives exact fixture hashes, and records every dimension of its explicitly bounded mechanism.
- Each fixture regenerates its exact existing binary name, class-identity/structural key, entry descriptor, schema, and Class-File SHA-256. No production identity or bytes change.
- A `PROVED_CONSTANTS_ONLY` unit contains only fixtures for which automated normalization proves declared immediate constants are the sole difference. Failed or ambiguous comparisons are explicitly ungrouped and cannot be projected by CPU 0009.
- The projection resource rejects duplicate members, missing hashes, missing shape dimensions, unrecognized normalizer versions, and a representative standing in for a member with no exact hash.
- Semantic tests cover every fixture against the matching optimal clean Java oracle; structural tests cover every proved unit and reject prohibited hot-path structure.
- CPU 0009 remains `Blocked`: this task supplies a strict projection mechanism only, not the promised operation/type/carrier/layout/strategy/shape matrix; this task never claims arbitrary-immediate enumeration.
- A separate documentation-focused pass reviews the final test/evidence and planning diff, and records the reasoned no-change conclusions for public/production Javadoc, backend guides, glossary, architecture documents, ADRs, architecture tests, conformance/integration tests, Gradle, and production documentation.

## Tests / validation

```bash
./gradlew :backends:cpu:test --tests '*ScalarImmediateClampEquivalence*' --tests '*CpuPointwiseGeneratedKernelTest' --tests '*CpuPointwisePartitionLoweringTest' --tests '*CpuScalarPowerAnalysisTest'
./gradlew :backends:cpu:test
git diff --check
```

Also validate resource parsing; exact fixture/Class-File SHA-256 values; binary names, schema, structural keys, and entry descriptors; normalization false-positive/negative controls; projection provenance; no production paths; Markdown links/anchors/fences; task/master-plan/roadmap synchronization; and `git status --short`. Repository-wide, architecture, conformance, integration, Javadoc, glossary, and performance validation are not required because this task changes no production or public documentation.

## Dependencies

- Complete CPU 0005F, CPU 0005G, CPU 0005J, CPU 0008J, CPU 0008L, and current schema-63 pointwise lowering/preparation/generation.
- Current Java 26 Class-File API toolchain.

## Follow-up tasks

- [CPU 0008Q1](0008q1-finite-scalar-immediate-clamp-matrix.md), with its own documentation pass, is required before CPU 0009 can become `Ready`.
- Any non-equivalent immediate/code-shaping form becomes a family-specific Draft task and must not be asserted into this contract or CPU 0009.

## Architecture impact

Expected impact: None. Stop for an architecture, dependency, production-identity, schema, capability, selection, or semantics change.

## Implementation prompt

```text
Read AGENTS.md, ARCHITECTURE.md, the Planning Guide, CPU master plan, CPU 0009, and this task. Implement only this finite test/resource equivalence contract. Do not change production behavior, identity, hashes, schema, or selection. Stop for architecture or scope conflict. Do not commit or push. After executable validation, hand the stable diff and evidence to a separate clean documentation-focused context for the planning/documentation review. Update status only from results.
```

## Local decisions

- A finite unit is evidence about declared fixture members, not a universal statement over raw immediate bit patterns.
- Exact hashes are source-of-truth fixture identities; normalization is separately checked structural evidence and cannot erase hash differences.
- `SCALAR_POW` realization and constant encoding category are boundaries before normalization; clamp has two ordered operands and never inherits scalar-arithmetic units.

## Known limitations

No finite fixture set can establish behavior for every arbitrary immediate. The task stops at relations mechanically proved from Class-Files. The complete scalar-immediate/clamp matrix is CPU 0008Q1; its completion and documentation pass are prerequisites of CPU 0009.

## Validation evidence

Reviewed the required architecture/planning documents, `CpuKernelIr` immediate/clamp identity,
`CpuKernelSpecialization` class identity, and pointwise lowering/preparation/generation. The
following forced focused validation passed after the implementation repair:

```bash
./gradlew :backends:cpu:test --tests '*ScalarImmediateClampEquivalence*' --tests '*CpuPointwiseGeneratedKernelTest' --tests '*CpuPointwisePartitionLoweringTest' --tests '*CpuScalarPowerAnalysisTest' --rerun-tasks
```

The task matrix is included in `*ScalarImmediateClampEquivalence*`: it regenerates exact hashes,
parses strict tab-separated resources, executes the generated hidden classes, compares raw
BFLOAT16 results with clean Java, and runs normalizer/structural mutation controls. `git diff
--check`, path/status checks, literal-escape rejection, and exact 16/13-column TSV checks passed.
No full CPU rerun was performed: the prior result was 789 tests, 25 skips, and one unrelated
historical timing failure, so this task makes no full-CPU-green claim.

Documentation-focused context ID: `/root`. It independently reviewed the final diff using the General and
Planning documentation profiles. It confirmed the two exact BFLOAT16 scalar-MUL represented
immediates on contiguous `SHORT_ARRAY` shape `[8]`, rather than an arbitrary-immediate or full
scalar/clamp claim. It ran local link/anchor/fence/profile, status/dependency, changed-path, TSV
tab/column, and whitespace checks. No executable Java changed after the recorded focused run.

No-change conclusions: public/production Javadoc and production documentation are unaffected
because no production Java or public behavior changed; backend guides and the glossary have no
new reusable user-facing term or workflow; `ARCHITECTURE.md` and architecture explanations need
no update because ownership, dependencies, and lifecycle are unchanged; ADRs are unnecessary
because no decision changed; architecture, conformance, and integration tests do not apply
because no boundary, backend behavior, or end-to-end behavior changed; and Gradle is unchanged
because no build configuration changed.

## Implementation notes

The checked pair derives from a real PrepareContext, CpuPartitionPreparer, lowering, selected
portable route, and generator. Its strict normalizer admits a difference only at its declared
immediate instruction location and compares every other instruction operand, member structure,
and prohibited hot-path feature. Generated classes execute against independent typed clean-Java
loops with ordinary, NaN, signed-zero, and infinity inputs. A full scalar/clamp matrix was not
possible under this task's three-source/two-resource ceiling without inventing fixture families;
it is intentionally deferred and 0009 remains blocked.

## Completion summary

Completed changes: added the bounded test-only source-derived projection evidence for exactly two
BFLOAT16 scalar-MUL represented immediates, and finalized the matching planning coordination.

Files changed or created: three focused CPU test sources, two checked TSV resources, this task,
CPU master plan, CPU 0009, and roadmap.

Tests and validation: reused the recorded passing forced focused Gradle command; task-matrix
resource/hash/provenance, local Markdown link/anchor/fence/profile, status/dependency,
changed-path, TSV tab/column, and `git diff --check` checks passed. No Java tests were rerun by
this documentation pass because executable Java did not change. The prior full CPU observation
remains 789 tests, 25 skips, and one unrelated historical timing failure; no full-CPU-green claim
is made.

Documentation-agent review: clean documentation-focused context ID `/root` completed the targeted
General + Planning profile review.

Documentation impact: planning documents only; the reasoned no-change conclusions are recorded
above for Javadoc, backend guides, glossary, architecture, ADRs, architecture tests,
conformance/integration tests, Gradle, and production documentation.

Unresolved issues: CPU 0008Q1's complete scalar-immediate/clamp operation/type/carrier/layout/
strategy/shape matrix remains unimplemented.

Follow-up required: complete CPU 0008Q1, including its documentation pass, before CPU 0009 can become `Ready`.

Status: Complete
