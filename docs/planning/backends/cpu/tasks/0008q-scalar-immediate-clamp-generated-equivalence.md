# Task 0008Q: Finite Scalar-Immediate and Clamp Generated-Code Equivalence

## Status

Ready

## Goal

Establish a finite, fail-closed generated-code coverage-equivalence contract for the current portable pointwise scalar-immediate and floating-clamp forms. The task proves equivalence only for a finite fixture set when automated Class-File normalization proves that changing an admitted immediate changes constants only, while preserving generated opcode, control flow, invokes, carrier access, hot loop, and dataflow shape. It supplies the finite evidence unit CPU 0009 needs before it can inventory coverage honestly.

## Scope

- Add focused CPU test support and checked resources for a finite fixture matrix covering scalar ADD/SUB/MUL/DIV/MIN/MAX/POW and CLAMP across every currently admitted type, carrier, layout/addressing, execution strategy, and materially distinct pointwise shape reached by selected fixtures.
- Keep distinct opcode/type/carrier/layout/strategy, scalar-power realization, structurally relevant constant encoding category, edge-value semantic category, and materially distinct shape as separate evidence units. Clamp retains ordered lower/upper categories.
- Record each fixture's exact production binary name, specialization structural key, Class-File SHA-256, fixture-resource SHA-256, entry descriptor, schema, and normalization result. Do not alter `CpuKernelIr`, `CpuKernelSpecialization`, lowering, preparation, generation, naming, or identity projection.
- Implement a test-only automated Class-File normalizer. A unit is accepted only when differing code/constant operands decode to declared immediate(s), every non-immediate constant matches, and opcode/control-flow/invoke/carrier/hot-loop/dataflow shape matches. Exact class hashes remain fixture-specific.
- Add semantic tests against an optimal clean Java oracle with the same typed operation, numerical order, immediate conversion, branch behavior, carrier access, and hot-loop/dataflow shape. Add structural tests rejecting allocation, boxing, reflection, maps, string dispatch, hidden Synaptik helpers, and avoidable dispatch on proved hot paths.
- Emit a checked coverage-projection resource naming every finite unit, exact member fixture/hash, normalizer version, fixture provenance, and `PROVED_CONSTANTS_ONLY` or a fail-closed non-equivalent disposition. CPU 0009 may consume only this provenance.

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

- The checked fixture resource is finite, gives exact fixture hashes, and records every retained distinction.
- Each fixture regenerates its exact existing binary name, class-identity/structural key, entry descriptor, schema, and Class-File SHA-256. No production identity or bytes change.
- A `PROVED_CONSTANTS_ONLY` unit contains only fixtures for which automated normalization proves declared immediate constants are the sole difference. Failed or ambiguous comparisons are explicitly ungrouped and cannot be projected by CPU 0009.
- The projection resource rejects duplicate members, missing hashes, missing shape dimensions, unrecognized normalizer versions, and a representative standing in for a member with no exact hash.
- Semantic tests cover every fixture against the matching optimal clean Java oracle; structural tests cover every proved unit and reject prohibited hot-path structure.
- CPU 0009 can return to `Ready` only after this task is `Complete`; this task never claims arbitrary-immediate enumeration.
- A separate documentation-focused pass reviews the final planning-only diff and records no production Javadoc, backend-guide, or glossary change because it changes only tests and planning coordination.

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

- CPU 0009 consumes only proved finite units and returns to `Ready` after this task completes.
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

No finite fixture set can establish behavior for every arbitrary immediate. The task stops at relations mechanically proved from Class-Files.

## Validation evidence

Planning-only evidence: reviewed the required architecture/planning documents, `CpuKernelIr` immediate/clamp identity, `CpuKernelSpecialization` class identity, pointwise lowering/preparation/generation, and pointwise structural-evidence tests. No Java, Class-File, Javadoc, or performance command was run for this planning change.

## Implementation notes

Empty until implemented.

## Completion summary

Empty until implemented.
