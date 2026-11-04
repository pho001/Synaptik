# Task 0008Q: [Finite](../../../../glossary.md#term-finite) [Scalar](../../../../glossary.md#term-scalar)-Immediate and Clamp Generated-Code Equivalence

## Reader context

This task records a test-only equivalence mechanism for two `BFLOAT16` scalar-MUL immediates on
contiguous `SHORT_ARRAY` carriers of shape [8]. `CpuPartitionPreparer` and the selected portable
generator produce exact artifacts; `CpuKernelIr` and `CpuKernelSpecialization` retain their existing
identity rules. The Class-File normalizer is a structural oracle: it checks that only declared
immediate operands differ, without authorizing artifact reuse or changing selection.

The semantic oracle is an independent typed clean-Java loop comparing raw `BFLOAT16` results,
including exceptional inputs. Structural tests separately reject hidden helpers and avoidable
hot-path machinery. Checked TSV resources preserve exact hashes and projection provenance. Here,
completion covers this mechanism only; the document leaves CPU 0009 blocked pending CPU 0008Q1,
rather than establishing the broader matrix itself.

Current-topic navigation (not a replacement for the historical record):

- [current exact typed scalar metadata](../../../../reference/tensor/data-types.md#exact-typed-scalar-values) — Distinguishes immutable immediate bits from a rank-zero Tensor or live storage.
- [current generated-class boundary](../../../../explanation/packages/cpu.md#iogithubpho001synaptikbackendcpuinternalcodegenemit) — Explains emitters, typed entries, hidden classes, and the limits of Class-File inspection.

## Status

Complete

## Goal

Establish the smallest source-derived, [fail-closed](../../../../glossary.md#term-fail-closed) generated-code projection mechanism that [CPU](../../../../glossary.md#term-cpu) 0009 can later extend: the exact [BFLOAT16](../../../../glossary.md#term-bfloat16) [scalar](../../../../glossary.md#term-scalar)-MUL represented-immediate pair on the current [contiguous](../../../../glossary.md#term-contiguous) [heap](../../../../glossary.md#term-heap) scalar route. This task does not establish a coverage-equivalence contract for every scalar-immediate or clamp form; 0009 remains blocked pending the separate complete [matrix](../../../../glossary.md#term-matrix-meanings) task.

## [Scope](../../../../glossary.md#term-scope)

- Add focused [CPU](../../../../glossary.md#term-cpu) test support and checked resources for exactly [BFLOAT16](../../../../glossary.md#term-bfloat16) [scalar](../../../../glossary.md#term-scalar) MUL, `SHORT_ARRAY` input/output carriers, [contiguous](../../../../glossary.md#term-contiguous) [layout](../../../../glossary.md#layout), scalar strategy, and [shape](../../../../glossary.md#shape) `[8]`. It is a complete two-immediate mechanism [fixture](../../../../glossary.md#term-fixture), not a projection for unrepresented [dimensions](../../../../glossary.md#dimension).
- Keep distinct [opcode](../../../../glossary.md#term-opcode)/type/[carrier](../../../../glossary.md#term-storage-carrier)/layout/strategy, [scalar-power realization](../../../../glossary.md#scalar-power-realization), structurally relevant [constant](../../../../glossary.md#term-constant) encoding category, edge-value semantic category, and materially distinct shape as separate evidence units. Clamp retains ordered lower/upper categories.
- Record each fixture's exact production [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) name, [specialization](../../../../glossary.md#term-specialization) structural key, Class-File [SHA-256](../../../../glossary.md#term-sha-256), fixture-resource SHA-256, entry [descriptor](../../../../glossary.md#term-descriptor), [schema](../../../../glossary.md#term-schema), and [normalization](../../../../glossary.md#term-normalization) result. Do not alter `CpuKernelIr`, `CpuKernelSpecialization`, [lowering](../../../../glossary.md#lowering), preparation, generation, naming, or [identity](../../../../glossary.md#term-identity-meanings) projection.
- Implement a test-only automated Class-File normalizer. A unit is accepted only when differing code/constant operands decode to declared immediate(s), every non-immediate constant matches, and opcode/control-flow/invoke/carrier/hot-loop/[dataflow](../../../../glossary.md#term-dataflow) shape matches. Exact class [hashes](../../../../glossary.md#term-hash) remain fixture-specific.
- Add semantic tests against an optimal clean Java [oracle](../../../../glossary.md#term-test-oracle) with the same typed [operation](../../../../glossary.md#operation), numerical order, immediate conversion, branch behavior, carrier access, and hot-loop/dataflow shape. Add structural tests rejecting allocation, boxing, reflection, maps, string [dispatch](../../../../glossary.md#term-dispatch), hidden Synaptik helpers, and avoidable dispatch on proved hot paths.
- Emit a checked mechanism-projection resource with exact member hashes, normalizer version, full dimensions, and [canonical](../../../../glossary.md#term-canonical-form) fixture-resource SHA-256. CPU 0009 may not consume it until a later task supplies the omitted [matrix](../../../../glossary.md#term-matrix-meanings).

## Out of [scope](../../../../glossary.md#term-scope)

- Enumerating arbitrary immediate bit patterns or claiming every [scalar](../../../../glossary.md#term-scalar) value has equivalent generated code.
- Production [identity](../../../../glossary.md#term-identity-meanings), [fixture](../../../../glossary.md#term-fixture) [hashes](../../../../glossary.md#term-hash), [schema](../../../../glossary.md#term-schema), [lowering](../../../../glossary.md#lowering), codegen, capability, selection, semantics, [vector](../../../../glossary.md#term-vector) eligibility, timing, broad coverage inventory, or [CPU](../../../../glossary.md#term-cpu) 0009 completion.
- Grouping across [scalar-power realization](../../../../glossary.md#scalar-power-realization), [constant](../../../../glossary.md#term-constant) encoding category, semantic edge category, type, [carrier](../../../../glossary.md#term-storage-carrier), [layout](../../../../glossary.md#layout), strategy, [opcode](../../../../glossary.md#term-opcode), or materially distinct generated shape.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md): [CPU](../../../../glossary.md#term-cpu) [ownership](../../../../glossary.md#term-ownership-meanings), direct [carrier](../../../../glossary.md#term-storage-carrier) [signatures](../../../../glossary.md#term-signature), existing [specialization](../../../../glossary.md#term-specialization) [identity](../../../../glossary.md#term-identity-meanings), and optimal-clean-Java semantic/hot-path parity.
- [CPU master plan](../master-plan.md): [schema](../../../../glossary.md#term-schema)-63 identity and ordered frontier.
- [CPU 0009](0009-portable-generated-coverage-closure-checkpoint.md): downstream inventory/evidence consumer.
- [Planning Guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>: bounded [Ready](../../../../glossary.md#term-ready)-task [scope](../../../../glossary.md#term-scope) and [validation](../../../../glossary.md#term-validation).

## Architecture constraints

- Existing `CpuKernelIr` and `CpuKernelSpecialization` [identity](../../../../glossary.md#term-identity-meanings) remains exact; no [runtime](../../../../glossary.md#term-runtime)/address/object/[extent](../../../../glossary.md#term-extent)/worker/[artifact](../../../../glossary.md#term-artifact)-root fact enters identity and this task changes no such rule.
- [Normalization](../../../../glossary.md#term-normalization) is test-only evidence: it cannot rewrite a class, normalize production identity, merge artifacts, authorize [cache](../../../../glossary.md#term-cache) reuse, or change selection.
- If normalization cannot identify exact immediate operands or finds a changed instruction, branch, invoke, [carrier](../../../../glossary.md#term-storage-carrier) access, loop/[dataflow](../../../../glossary.md#term-dataflow) shape, or non-immediate [constant](../../../../glossary.md#term-constant), leave the form ungrouped and [fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation).

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
- this task, [CPU 0009](0009-portable-generated-coverage-closure-checkpoint.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>, [CPU master plan](../master-plan.md), and [roadmap](../../../roadmap.md).

## Maximum [scope](../../../../glossary.md#term-scope)

At most three [CPU](../../../../glossary.md#term-cpu) test-source paths, two CPU test-resource paths, and four [planning](../../../../glossary.md#term-planning) paths; zero production Java, [Javadoc](../../../../glossary.md#term-javadoc), [backend](../../../../glossary.md#backend)-guide, glossary, [Gradle](../../../../glossary.md#term-gradle), architecture, [conformance](../../../../glossary.md#term-conformance), integration, or performance-[artifact](../../../../glossary.md#term-artifact) paths. If a [fixture](../../../../glossary.md#term-fixture) exposes a production difference or needs a new source fixture family, stop and create a bounded follow-up.

## Stop conditions

Stop without changing production behavior and leave the task `Blocked` or `Incomplete` if any
reachable [scalar](../../../../glossary.md#term-scalar)-immediate/clamp form cannot be reduced to a [finite](../../../../glossary.md#term-finite) [fixture](../../../../glossary.md#term-fixture) set, a proposed unit
changes an instruction, control-flow/invoke/[carrier](../../../../glossary.md#term-storage-carrier)/hot-loop/[dataflow](../../../../glossary.md#term-dataflow) shape, a fixture cannot retain
its exact existing [identity](../../../../glossary.md#term-identity-meanings)/[hash](../../../../glossary.md#term-hash), the normalizer cannot prove the declared [constants](../../../../glossary.md#term-constant)-only relation,
or the matching clean-Java semantic/structural test fails. Record the exact [opcode](../../../../glossary.md#term-opcode), type, carrier,
[layout](../../../../glossary.md#layout), strategy, immediate category, fixture hash, and required family-specific follow-up.

## Acceptance criteria

- The checked [fixture](../../../../glossary.md#term-fixture) resource is [finite](../../../../glossary.md#term-finite), gives exact fixture [hashes](../../../../glossary.md#term-hash), and records every [dimension](../../../../glossary.md#dimension) of its explicitly bounded mechanism.
- Each fixture regenerates its exact existing [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) name, class-[identity](../../../../glossary.md#term-identity-meanings)/structural key, entry [descriptor](../../../../glossary.md#term-descriptor), [schema](../../../../glossary.md#term-schema), and Class-File [SHA-256](../../../../glossary.md#term-sha-256). No production identity or bytes change.
- A `PROVED_CONSTANTS_ONLY` unit contains only fixtures for which automated [normalization](../../../../glossary.md#term-normalization) proves declared immediate [constants](../../../../glossary.md#term-constant) are the sole difference. Failed or ambiguous comparisons are explicitly ungrouped and cannot be projected by [CPU](../../../../glossary.md#term-cpu) 0009.
- The projection resource rejects duplicate members, missing hashes, missing [shape](../../../../glossary.md#shape) dimensions, unrecognized normalizer versions, and a representative standing in for a member with no exact [hash](../../../../glossary.md#term-hash).
- Semantic tests cover every fixture against the matching optimal clean Java [oracle](../../../../glossary.md#term-test-oracle); structural tests cover every proved unit and reject prohibited hot-path structure.
- CPU 0009 remains `Blocked`: this task supplies a strict projection mechanism only, not the promised [operation](../../../../glossary.md#operation)/type/[carrier](../../../../glossary.md#term-storage-carrier)/[layout](../../../../glossary.md#layout)/strategy/shape [matrix](../../../../glossary.md#term-matrix-meanings); this task never claims arbitrary-immediate enumeration.
- A separate documentation-focused pass reviews the final test/evidence and [planning](../../../../glossary.md#term-planning) diff, and records the reasoned no-change conclusions for public/production [Javadoc](../../../../glossary.md#term-javadoc), [backend](../../../../glossary.md#backend) guides, glossary, architecture documents, ADRs, architecture tests, [conformance](../../../../glossary.md#term-conformance)/integration tests, [Gradle](../../../../glossary.md#term-gradle), and production documentation.

## Tests / [validation](../../../../glossary.md#term-validation)

```bash
./gradlew :backends:cpu:test --tests '*ScalarImmediateClampEquivalence*' --tests '*CpuPointwiseGeneratedKernelTest' --tests '*CpuPointwisePartitionLoweringTest' --tests '*CpuScalarPowerAnalysisTest'
./gradlew :backends:cpu:test
git diff --check
```

Also validate resource parsing; exact [fixture](../../../../glossary.md#term-fixture)/Class-File [SHA-256](../../../../glossary.md#term-sha-256) values; [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) names, [schema](../../../../glossary.md#term-schema), structural keys, and entry [descriptors](../../../../glossary.md#term-descriptor); [normalization](../../../../glossary.md#term-normalization) false-positive/negative controls; projection [provenance](../../../../glossary.md#provenance); no production paths; Markdown links/anchors/fences; task/master-plan/roadmap [synchronization](../../../../glossary.md#term-synchronization); and `git status --short`. Repository-wide, architecture, [conformance](../../../../glossary.md#term-conformance), integration, [Javadoc](../../../../glossary.md#term-javadoc), glossary, and performance validation are not required because this task changes no production or public documentation.

## Dependencies

- Complete [CPU](../../../../glossary.md#term-cpu) 0005F, CPU 0005G, CPU 0005J, CPU 0008J, CPU 0008L, and current [schema](../../../../glossary.md#term-schema)-63 [pointwise](../../../../glossary.md#term-pointwise) [lowering](../../../../glossary.md#lowering)/preparation/generation.
- Current Java 26 [Class-File API](../../../../glossary.md#term-class-file-api) [toolchain](../../../../glossary.md#term-toolchain).

## Follow-up tasks

- [CPU 0008Q1](0008q1-finite-scalar-immediate-clamp-matrix.md), with its own documentation pass, is required before [CPU](../../../../glossary.md#term-cpu) 0009 can become [`Ready`](../../../../glossary.md#term-ready).
- Any non-equivalent immediate/code-shaping form becomes a family-specific Draft task and must not be asserted into this contract or CPU 0009.

## Architecture impact

Expected impact: None. Stop for an architecture, [dependency](../../../../glossary.md#term-dependency), production-[identity](../../../../glossary.md#term-identity-meanings), [schema](../../../../glossary.md#term-schema), capability, selection, or semantics change.

## Implementation prompt

```text
Read AGENTS.md, ARCHITECTURE.md, the Planning Guide, CPU master plan, CPU 0009, and this task. Implement only this finite test/resource equivalence contract. Do not change production behavior, identity, hashes, schema, or selection. Stop for architecture or scope conflict. Do not commit or push. After executable validation, hand the stable diff and evidence to a separate clean documentation-focused context for the planning/documentation review. Update status only from results.
```

## Local decisions

- A [finite](../../../../glossary.md#term-finite) unit is evidence about declared [fixture](../../../../glossary.md#term-fixture) members, not a universal statement over raw immediate bit patterns.
- Exact [hashes](../../../../glossary.md#term-hash) are source-of-truth fixture [identities](../../../../glossary.md#term-identity-meanings); [normalization](../../../../glossary.md#term-normalization) is separately checked structural evidence and cannot erase [hash](../../../../glossary.md#term-hash) differences.
- `SCALAR_POW` realization and [constant](../../../../glossary.md#term-constant) encoding category are boundaries before normalization; clamp has two ordered operands and never inherits [scalar](../../../../glossary.md#term-scalar)-arithmetic units.

## Known limitations

No [finite](../../../../glossary.md#term-finite) [fixture](../../../../glossary.md#term-fixture) set can establish behavior for every arbitrary immediate. The task stops at relations mechanically proved from Class-Files. The complete [scalar](../../../../glossary.md#term-scalar)-immediate/clamp [matrix](../../../../glossary.md#term-matrix-meanings) is [CPU](../../../../glossary.md#term-cpu) 0008Q1; its completion and documentation pass are prerequisites of CPU 0009.

## [Validation](../../../../glossary.md#term-validation) evidence

Reviewed the required architecture/[planning](../../../../glossary.md#term-planning) documents, `CpuKernelIr` immediate/clamp [identity](../../../../glossary.md#term-identity-meanings),
`CpuKernelSpecialization` class identity, and [pointwise](../../../../glossary.md#term-pointwise) [lowering](../../../../glossary.md#lowering)/preparation/generation. The
following forced focused validation passed after the implementation repair:

```bash
./gradlew :backends:cpu:test --tests '*ScalarImmediateClampEquivalence*' --tests '*CpuPointwiseGeneratedKernelTest' --tests '*CpuPointwisePartitionLoweringTest' --tests '*CpuScalarPowerAnalysisTest' --rerun-tasks
```

The task [matrix](../../../../glossary.md#term-matrix-meanings) is included in `*ScalarImmediateClampEquivalence*`: it regenerates exact [hashes](../../../../glossary.md#term-hash),
parses strict tab-separated resources, executes the generated hidden classes, compares raw
[BFLOAT16](../../../../glossary.md#term-bfloat16) results with clean Java, and runs normalizer/structural mutation controls. `git diff
--check`, path/status checks, literal-escape rejection, and exact 16/13-column TSV checks passed.
No full [CPU](../../../../glossary.md#term-cpu) rerun was performed: the prior result was 789 tests, 25 skips, and one unrelated
historical timing failure, so this task makes no full-CPU-green claim.

Documentation-focused context ID: `/root`. It independently reviewed the final diff using the General and
Planning documentation profiles. It confirmed the two exact BFLOAT16 [scalar](../../../../glossary.md#term-scalar)-MUL represented
immediates on [contiguous](../../../../glossary.md#term-contiguous) `SHORT_ARRAY` [shape](../../../../glossary.md#shape) `[8]`, rather than an arbitrary-immediate or full
scalar/clamp claim. It ran local link/anchor/fence/profile, status/[dependency](../../../../glossary.md#term-dependency), changed-path, TSV
tab/column, and whitespace checks. No executable Java changed after the recorded focused [run](../../../../glossary.md#term-run).

No-change conclusions: public/production [Javadoc](../../../../glossary.md#term-javadoc) and production documentation are unaffected
because no production Java or public behavior changed; [backend](../../../../glossary.md#backend) guides and the glossary have no
new reusable user-facing term or workflow; `ARCHITECTURE.md` and architecture explanations need
no update because [ownership](../../../../glossary.md#term-ownership-meanings), dependencies, and [lifecycle](../../../../glossary.md#lifecycle) are unchanged; ADRs are unnecessary
because no decision changed; architecture, [conformance](../../../../glossary.md#term-conformance), and integration tests do not apply
because no boundary, backend behavior, or end-to-end behavior changed; and [Gradle](../../../../glossary.md#term-gradle) is unchanged
because no build configuration changed.

## Implementation notes

The checked pair derives from a real [PrepareContext](../../../../glossary.md#prepare-context-preparecontext), CpuPartitionPreparer, [lowering](../../../../glossary.md#lowering), selected
portable route, and generator. Its strict normalizer admits a difference only at its declared
immediate instruction location and compares every other instruction operand, member structure,
and prohibited hot-path [feature](../../../../glossary.md#term-software-feature). Generated classes execute against independent typed clean-Java
loops with ordinary, [NaN](../../../../glossary.md#term-nan), [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero), and [infinity](../../../../glossary.md#term-infinity) inputs. A full [scalar](../../../../glossary.md#term-scalar)/clamp [matrix](../../../../glossary.md#term-matrix-meanings) was not
possible under this task's three-source/two-resource ceiling without inventing [fixture](../../../../glossary.md#term-fixture) families;
it is intentionally deferred and 0009 remains blocked.

## Completion summary

Completed changes: added the bounded test-only source-derived projection evidence for exactly two
[BFLOAT16](../../../../glossary.md#term-bfloat16) [scalar](../../../../glossary.md#term-scalar)-MUL represented immediates, and finalized the matching [planning](../../../../glossary.md#term-planning) coordination.

Files changed or created: three focused [CPU](../../../../glossary.md#term-cpu) test sources, two checked TSV resources, this task,
CPU master plan, CPU 0009, and roadmap.

Tests and [validation](../../../../glossary.md#term-validation): reused the recorded passing forced focused [Gradle](../../../../glossary.md#term-gradle) command; task-[matrix](../../../../glossary.md#term-matrix-meanings)
resource/[hash](../../../../glossary.md#term-hash)/[provenance](../../../../glossary.md#provenance), local Markdown link/anchor/fence/profile, status/[dependency](../../../../glossary.md#term-dependency),
changed-path, TSV tab/column, and `git diff --check` checks passed. No Java tests were rerun by
this documentation pass because executable Java did not change. The prior full CPU observation
remains 789 tests, 25 skips, and one unrelated historical timing failure; no full-CPU-green claim
is made.

Documentation-agent review: clean documentation-focused context ID `/root` completed the targeted
General + Planning profile review.

Documentation impact: planning documents only; the reasoned no-change conclusions are recorded
above for [Javadoc](../../../../glossary.md#term-javadoc), [backend](../../../../glossary.md#backend) guides, glossary, architecture, ADRs, architecture tests,
[conformance](../../../../glossary.md#term-conformance)/integration tests, Gradle, and production documentation.

Unresolved issues: CPU 0008Q1's complete scalar-immediate/clamp [operation](../../../../glossary.md#operation)/type/[carrier](../../../../glossary.md#term-storage-carrier)/[layout](../../../../glossary.md#layout)/
strategy/[shape](../../../../glossary.md#shape) matrix remains unimplemented.

Follow-up required: complete CPU 0008Q1, including its documentation pass, before CPU 0009 can become [`Ready`](../../../../glossary.md#term-ready).

Status: Complete
