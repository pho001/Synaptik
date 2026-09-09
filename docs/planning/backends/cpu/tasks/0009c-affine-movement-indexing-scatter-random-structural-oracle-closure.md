# Task 0009C: Affine, Movement, Indexing, Scatter, and Random Structural-Oracle Closure

## Status

Complete

## Goal

CPU 0009C1 is Complete, satisfying the narrow CONCAT/STACK selected-entry hygiene prerequisite
found during this task's earlier review. Close the next exact, bounded portable-CPU generated-
versus-clean-Java structural-oracle projection. The checked inventory owns exactly 2,252 generated
rows: 1,536 affine rows, 128 non-fold movement rows, 152 indexing rows, 416 scatter rows, and 20
random rows. Each row must retain source-derived provenance, generated definition/loading/
invocation, independent typed clean-Java semantic execution, applicable realization coverage,
hygiene, and an explicit disposition. Structural promotion is permitted only for a family-specific
paired projection that observes facts the clean Java and generated artifacts independently expose;
performance remains separately fail-closed.

## Scope

- Own the exact generated inventory projection below, joined by owner ID, evidence key, SHA-256, typed entry descriptor, ordered carriers, access regime, strategy, structural key, and disposition. The 17,636-row canonical inventory remains 17,463 `GENERATED` rows plus 173 exact rejected rows.

  | Family | Generated denominator | Exact forms |
  |---|---:|---|
  | Affine | 1,536 | `CONTIGUOUS`, `EXPAND`, `EXPAND_DIMS`, `PERMUTE`, `RESHAPE`, `SELECT`, `SLICE`, `SQUEEZE` — 192 each |
  | Movement | 128 | `PAD` 4; `TILE`, `CONCAT`, `STACK`, `SLICE_UPDATE`, `UNFOLD_AXIS` 24 each; `UNFOLD2D` 4 |
  | Indexing | 152 | `GATHER`, `GATHER_ELEMENTS`, `GATHER_ND` 48 each; canonical-BOOL `ONE_HOT` 8 |
  | Scatter | 416 | `SCATTER_ELEMENTS` 208; `SCATTER_ND` 208 |
  | Random | 20 | `DROPOUT` 16; `INITIAL_STATE` 4 |

- Keep `FOLD_AXIS` (20) and `FOLD2D` (12), every aggregate/scan/ordering row, and all later families out of this task. CPU 0009D owns aggregate, scan, ordering, and fold. The mixed current semantic test's 160-row movement/fold projection is not this task's denominator: its 128 non-fold movement rows are the only movement rows here.
- Derive fixtures from the provider, preparer, lowerer, IR, emitter, and checked inventory. Do not use operation names, catalog categories, normalized-body keys, hashes, or a common helper as equivalence proof.
- For every owned row, generate deterministic bytes twice, define and invoke the actual typed entry, and compare actual normal, empty/non-empty, nonzero-start/tail, and untouched-sentinel behavior with independently authored, specialized typed clean Java. Preserve represented bits for all six Model data types where the family admits them; cover floating special values, integral wrap boundaries, canonical BOOL, BFLOAT16 represented bits, index widths, and stated family-specific state semantics when applicable.
- Cover each applicable ordered array/`MemorySegment` carrier role, dense/general or admitted non-contiguous access regime, scalar/vector/caller-parallel selection, range, tail, materialization, workspace/scratch, and legal alias/overlap condition from the checked fixtures. A missing or inapplicable realization must be recorded with its exact provider, preparer, binding, semantic, or selection boundary rather than inferred from another row.
- Treat the four relevant checked rejections as adjacent fail-closed boundaries, not generated coverage: one `GATHER` invalid index role, one `ONE_HOT` invalid output role, one BOOL `SCATTER_ADD` inapplicability, and one BFLOAT16 `DROPOUT` inapplicability. The integral `FOLD2D` rejection belongs to CPU 0009D, not this task.

## Out of scope

- Production Java, generator/lowering/IR schemas, provider support, admission, selection, cache identity, materialization policy, Model/Compiler/Prepare/Runtime/Engine contracts, native routes, or architecture changes. CPU 0009C1 alone owns the narrow CONCAT/STACK production correction and any proved identity update.
- CPU 0009A scalar-immediate/clamp, CPU 0009B ordinary pointwise/CAST, fold, aggregate, scan, ordering, reductions, normalizations, loss, matmul, convolution, pooling, attention, batch norm, and CPU 0009D--0009G work.
- Literal bytecode, `javac`, CFG, or JIT identity; general bytecode verification; full abstract interpretation; an unbounded predecessor walk presented as a proof; or mutation for every theoretical schema dimension.
- New five-fork measurements, a performance disposition promotion, or a representative-performance claim. Existing performance evidence is reusable only with exact row, generated hash, clean counterpart, protocol, and projection agreement.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md): backend prepare owns lowering and selection; generated implementations must preserve an optimal clean-Java semantic algorithm, hot-loop/dataflow shape, and avoidable-overhead profile without requiring byte-for-byte, `javac`, CFG, or JIT identity.
- [Current architecture navigation](../../../../architecture/current-architecture-plan.md), [Planning Guide](../../../planning-guide.md), [documentation rules](../../../../developer-guide/documentation-rules.md), [General profile](../../../../developer-guide/documentation/general-style.md), [Planning profile](../../../../developer-guide/documentation/planning-style.md), [CPU master plan](../master-plan.md), parent [CPU 0009](0009-portable-generated-coverage-closure-checkpoint.md), and completed [CPU 0009A](0009a-scalar-immediate-clamp-clean-java-structural-equivalence.md) and [CPU 0009B](0009b-ordinary-pointwise-and-cast-structural-oracle-closure.md).

## Architecture constraints

- Evidence is test-only: it must not alter production capability, admission, lowering, selection, identity, routes, or runtime behavior. Prepare owns lowering and selection; Runtime invokes prepared artifacts only.
- The clean-Java oracle is independently authored, typed, and specialized. Shared immutable fixture inputs are allowed; generated code, emitters, lowering, reference kernels, generic carrier bridges, and production execution helpers are forbidden from oracle execution.
- Provider support, preparer admission, binding/invocation, selected strategy, semantic agreement, structural equivalence, and performance are independent fail-closed facts. A row not proved by the exact comparator remains `PARTIAL_CLASSFILE_NO_CLEAN_JAVA_ORACLE`; all rows remain `PARTIAL_NO_REPRESENTATIVE_BENCHMARK` absent exact reusable evidence.
- Preserve ordered carrier roles, addresses, input/output topology, materialization placement, conversion/represented-bit rules, stores, range ownership, and state/scratch facts. Never flatten these into an operation-name, unordered-carrier, or generic-hygiene projection.

## Package impact

Existing test-only package:

- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` owns inventory reconstruction, typed clean-Java counterparts, generated-entry invocation, structural comparison, hygiene, and disposition evidence.

Packages added or changed:

- None. No production package, type, public Javadoc, or module dependency is authorized.

Type placement:

- `...CpuAffineGeneratedCoverageFixtureTest` — exact affine source fixture/invocation basis, extended only for missing source-derived witnesses.
- `...CpuOrdinaryMovementFoldSemanticClosureTest`, `...CpuIndexingOrderingSemanticClosureTest`, `...CpuScatterSemanticClosureTest`, and `...CpuRandomOneHotSemanticClosureTest` — source-derived direct-entry fixtures; extract only a non-fold, non-ordering projection where needed.
- `...CpuAffineMovementIndexingScatterRandomCleanJavaOracle` (new test-only) — independent typed clean-Java ABI and semantic counterparts.
- `...CpuAffineMovementIndexingScatterRandomStructuralOracleTest` (new test-only) — paired family projections, active hygiene, dispositions, and negative controls.

## Affected files

Expected CPU test/evidence paths:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuAffineGeneratedCoverageFixtureTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuOrdinaryMovementFoldSemanticClosureTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuIndexingOrderingSemanticClosureTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuScatterSemanticClosureTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuRandomOneHotSemanticClosureTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuAffineMovementIndexingScatterRandomCleanJavaOracle.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuAffineMovementIndexingScatterRandomStructuralOracleTest.java`
- Existing checkpoint/unsupported-boundary/registry tests and inventory, disposition-registry, or evidence-ledger resources only where a row-exact disposition synchronization requires them.

Planning paths: this task, completed [CPU 0009B](0009b-ordinary-pointwise-and-cast-structural-oracle-closure.md), parent [CPU 0009](0009-portable-generated-coverage-closure-checkpoint.md), [CPU master plan](../master-plan.md), and [roadmap](../../../roadmap.md).

## Maximum scope

This task may modify at most ten CPU test/evidence paths and five planning paths. It may not modify production Java, Gradle, architecture documents/ADRs, public/API Javadocs, guides, glossary, architecture/conformance/integration tests, or unrelated modules. If an additional path, production change, semantic family, or unproved structural fact is required, stop, leave the affected row partial, and propose a narrow Draft follow-up.

## Acceptance criteria

- The exact 2,252-row generated projection reconciles once with the checked inventory and its 1,536/128/152/416/20 family denominators and form counts. It neither admits the 32 fold rows nor ordering; the parent total remains 17,463 generated plus 173 rejected.
- Every owned row has SHA-bound provenance, deterministic generation, definition/loading, typed invocation, an independent typed clean-Java counterpart, and agreement on applicable semantic, range, sentinel, data-type, carrier, access, selected-strategy, alias/overlap, scratch/state, and exceptional conditions.
- The comparator declares distinct projections rather than one flattened schema:
  - affine compares the single input/output ABI, represented-bit copy, exact composed affine source/result address progression, and dense-int versus general-long loop form;
  - movement compares each topology separately: PAD fill versus source map; TILE carry/wrap; CONCAT boundary source selection; STACK occurrence placement; SLICE_UPDATE base/update selection; and UNFOLD_AXIS/UNFOLD2D window map. It preserves one-versus-multi-input role order and the no-fold boundary;
  - indexing compares prevalidated writer ABI, typed index load/address map, GATHER/GATHER_ELEMENTS/GATHER_ND rank-specific mapping, and ONE_HOT's canonical BOOL write. Validation is not silently treated as generated hot-loop work;
  - scatter compares base-copy-before-update topology, ordered base/index/update/output roles, target mapping, reduction order, exact `NONE` uniqueness boundary, output-range ownership, and the floating-`MUL` scratch/output-group state machine separately from scratch-free reductions;
  - random compares `INITIAL_STATE`'s exact state stores separately from DROPOUT's `[0,0)` state prologue, global-ordinal word/mapping/threshold order, FLOAT64/FLOAT32 value scaling, canonical BOOL mask store, counter advance, three-output ordering, and zero/nonzero range behavior.
- Each paired projection includes active Class-File hygiene with an exact invocation/overhead allowlist. It rejects generated/reference/oracle/helper leakage, allocation, boxing, reflection, method-handle/dynamic invocation, string/map/generic-carrier dispatch, avoidable virtual/interface dispatch, and unjustified calls. Exact numerical, Vector API, and Foreign Function and Memory API targets are allowed only when the selected family projection justifies them.
- Negative controls fail for altered ABI/carrier role/order, address/map/store, range direction or tail, conversion/represented-bit rule, state/scratch topology, invoke target, and a deliberately changed allowed projection. IDs, hashes, normalized-body keys, catalog membership, and self-comparison cannot make a negative control pass.
- Inventory/disposition/evidence resources fail closed for stale digest, duplicate/orphan/missing owner, changed ABI/hash, malformed schema/header/tab/LF/final newline, unknown disposition, mismatched totals, or a scoped rejection reclassified as generated. Unproved rows remain partial; no performance promotion occurs.
- A separate documentation-focused pass finalizes the implementation diff and records reasoned no-change conclusions for public/API Javadocs, guides, glossary, architecture/ADRs, architecture/conformance/integration tests, Gradle, production packages, and unrelated modules.

## Tests / validation

Implementation context runs the exact source-derived manifest/direct-entry bases, independent clean-Java matrix, paired structural/hygiene/negative controls, checkpoint, and unsupported boundaries:

```bash
./gradlew :backends:cpu:test --tests '*CpuAffineGeneratedCoverageFixtureTest' --tests '*CpuOrdinaryMovementFoldSemanticClosureTest' --tests '*CpuIndexingOrderingSemanticClosureTest' --tests '*CpuScatterSemanticClosureTest' --tests '*CpuRandomOneHotSemanticClosureTest' --tests '*CpuAffineMovementIndexingScatterRandomStructuralOracleTest' --tests '*CpuGeneratedCoverageCheckpointTest' --tests '*CpuGeneratedCoverageUnsupportedBoundaryTest' --tests '*CpuGeneratedCoverageEvidenceTest'
```

It validates the exact 2,252 generated denominator; 1,536/128/152/416/20 family counts; every listed form; the four scoped rejections; SHA-bound owner/disposition joins; headers, tabs, LF endings, final newline, and digests; then runs `git diff --check`.

Documentation context reuses successful Java evidence unless executable Java changes. It validates Markdown links, anchors, fences, path scope, status/dependency synchronization, terminology, whitespace, and:

```bash
git diff --check
```

Repository-wide/capability-checkpoint validation and representative performance closure defer to CPU 0009G/CI. No five-fork command is required.

## Dependencies

- Completed CPU 0009C1 and its documentation pass: the required production hygiene correction
  discovered during this task's review; it depends on completed CPU 0009B.
- Current checked generated inventory, disposition registry, evidence ledger, semantic-closure fixtures, and CPU preparer/lowering/IR/generator/emitter contracts.
- Java 26 Class-File, Vector API, and Foreign Function and Memory API test environment.

## Follow-up tasks

- CPU 0009D remains the next ordered Draft child after CPU 0009C resumes and is Complete. It owns aggregate, scan, ordering, and fold; do not create its detailed specification until then.
- A row requiring production, semantic, or broad-comparator work remains partial and becomes a narrowly named Draft follow-up only at the active frontier.
- CPU 0009G owns representative performance and final generated-coverage checkpoint closure.

## Architecture impact

Expected impact: None. Stop for an architecture, dependency, production-selection, schema, generator, capability, admission, or semantics change.

## Implementation prompt

```text
You are the clean implementation agent for Synaptik CPU task 0009C. Read AGENTS.md, ARCHITECTURE.md, docs/architecture/current-architecture-plan.md, the Planning Guide, CPU master plan, CPU 0009, CPU 0009A, CPU 0009B, completed CPU 0009C1, and this task. Implement exactly this bounded 2,252-row affine/movement/indexing/scatter/random test-evidence closure. Do not change production behavior, include fold or ordering, weaken the CONCAT/STACK hygiene gate, or claim general bytecode/CFG equivalence. Promote structural evidence only through the declared non-tautological family-specific paired clean-Java projections; leave every unproved row partial. Do not run or promote new five-fork performance evidence. Stop for a scope or architecture conflict. Do not commit or push. After stable evidence, hand the diff and test evidence to a separate clean documentation-focused context under docs/developer-guide/documentation-rules.md. Update this task with results; do not mark Complete before that pass finishes.
```

## Local decisions

- The denominator is the source-checked 2,252-row projection, not the 6,009 catalog and not the 160-row movement/fold semantic fixture. Fold's 32 rows remain exclusively in CPU 0009D.
- `ONE_HOT` belongs to indexing (8 canonical-BOOL rows), while `DROPOUT` and `INITIAL_STATE` alone form random (20 rows); the current combined 28-row semantic fixture is split by operation semantics for this task.
- Structural evidence may share parsing/hygiene infrastructure but not a topology-erasing comparator. Scratch-free scatter and floating-`MUL`, and state-only initialization and dropout, remain separate projections.

## Known limitations

This task proves no arbitrary-bytecode equivalence, universal alias/overlap behavior beyond exact checked legal fixtures, literal Class-File/CFG/JIT identity, fold/ordering behavior, or generated/direct performance parity. The 6,009 catalog remains discovery/accounting support, never a closure denominator or proof.

## Validation evidence

The canonical inventory SHA-256 is
`e821d08cad4e5903510d4b6f700f94708c01b913f9e3a5e8b4e5d6f3bd718c1d`: 17,636 rows comprising
17,463 generated and 173 rejected rows. CPU 0009C proved its exact 2,252 generated rows: 1,536
affine, 128 non-fold movement, 152 indexing, 416 scatter, and 20 random. The four adjacent
unsupported cases remain rejected: invalid-index-role `GATHER`, invalid-output-role `ONE_HOT`,
BOOL `SCATTER_ADD`, and BFLOAT16 `DROPOUT`.

Serial focused validation passed checkpoint 9/9, evidence 2/2, unsupported 3/3, structural
25/25, catalog 2/2, and ordinary matrix 2/2. Independent reviews approved the structural
promotion and registry. CONCAT and STACK's 48 rows have generated and independent javac
clean-Java normalized provenance relations, including repeated occurrence, boundary,
source/output address, range, and semantic-mutant evidence; all other owned families retain their
completed paired semantic/structural evidence. Earlier concurrent Gradle EOF/`NoSuchFile`
observations were superseded by serial green runs.

Live resolution marks exactly 2,252 generated rows `ORACLE_PROVED`, leaves 15,211 generated rows
oracle-partial, and leaves all 17,463 generated rows
`PARTIAL_NO_REPRESENTATIVE_BENCHMARK`. No performance disposition was promoted. CPU 0009C1's
schema-65 prerequisite remains historically complete and is not rewritten here.

This clean documentation context applied the Planning and API/Javadoc profiles; it reviewed the
two 0009C oracle classes, live evidence/disposition registries and tests, canonical inventory and
ledger, parent, master plan, roadmap, and 0009C1. It did not rerun successful Java behavioral
suites because no executable behavior changed. Final documentation validation ran CPU Javadoc,
local Markdown link/anchor/fence and terminology/status checks, changed-path audit, and
`git diff --check`.

## Implementation notes

The test-only oracle separates frozen inventory accounting from live execution-produced registry
facts. A canonical disposition requires one most-specific applicability relation; stale, missing,
or ambiguous relations fail closed. The paired oracles retain ordered ABI, topology, carrier,
range, address, represented-bit, state, and scratch facts rather than operation-name categories.

## Completion summary

- Completed changes: closed the exact 2,252-row affine/movement/indexing/scatter/random paired
  clean-Java structural-oracle projection and promoted only its exact live registry owners.
- Files changed or created: bounded CPU test/evidence sources and registry/ledger resources in the
  implementation diff; this task, parent CPU 0009, CPU master plan, and roadmap.
- Tests and validation: reused the recorded serial green focused results; this context ran CPU
  Javadoc, Markdown link/anchor/fence/whitespace, terminology/status, changed-path, and diff checks.
- Documentation-agent review: clean context `/root`; Planning and API/Javadoc profiles.
- Documentation impact: private oracle/registry Javadocs finalized. Public/API docs, guides,
  glossary, `ARCHITECTURE.md`, architecture explanations, ADRs, Gradle, production code,
  architecture/conformance/integration tests, and unrelated modules require no change because
  this task changes neither public behavior, architecture, terminology, nor build policy.
- Glossary impact: None; no reusable project term changed.
- Unresolved issues: CPU 0009 remains unfinished. CPU 0009D is the next Draft task and owns fold,
  aggregate, scan, and ordering; all non-0009C generated rows remain oracle-partial and every
  generated row remains performance-partial.
- Follow-up required: CPU 0009D.

Status: Complete
