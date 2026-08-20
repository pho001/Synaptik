# Task 0007A1E: Movement General-Address-Loop Parity

## Status

Complete

## Goal

Correct the strongest single-owner residual cluster left by CPU 0007A1D: the frozen `M-PAD`,
`M-CONCAT`, `M-UNFOLD-AXIS`, and `M-UNFOLD2D` generated movement rows still spend substantially
more time than their equivalent direct primitive Java comparators after scalar segment-layout
construction has been hoisted.

Preserve each exact movement mapping while making invocation setup prove and retain bounded
address/coordinate state so repeated generated work does not reload unchanged geometry or
reconstruct an output coordinate/address from the general `long[]` geometry on every element.
Every target must meet the unchanged generated/direct `<= 1.15x` gate.

## Scope

- Use the unchanged frozen A1C sources under
  `/private/tmp/synaptik-cpu-0007a1c-oMAmuVhF` and the schema-32 A1D evidence under
  `/private/tmp/synaptik-cpu-0007a1d-HWoXtXQQ`.
- Correct only `CpuDataMovementEmitter`, which owns all four target mappings and their general
  long-address/output-odometer loops.
- Preserve PAD bounds/padding bits, CONCAT occurrence order and repeated-boundary mapping,
  UNFOLD_AXIS selected-axis step/window mapping, and UNFOLD2D canonical NCHW column order,
  padding, dilation, and output order.
- At generated invocation setup, cold-prove and load the target's loop-invariant extents, bases,
  strides, prefixes, and family facts into primitive locals. Maintain increment/carry/reset
  coordinate and address cursors in the repeated body, or select the existing bounded integer
  form when all accessed ranges and cursor transitions are proved to fit it independently of
  heap-versus-segment carrier choice.
- Keep a typed long-address fallback for any geometry whose range or transition cannot be proved.
  Carrier form alone must not force broader or narrower addressing.
- Add stable Class-File tests for the selected movement shapes. Assert the intended primitive
  invocation-local/cursor form semantically, without fixed bytecode offsets, constant-pool
  indexes, generated names, or JIT assembly.
- Advance the generated-kernel schema exactly once when emitted bytes change. Earlier artifacts
  become incompatible safe misses with no migration, alias, dual-schema path, or partial reuse.
- Re-run the unchanged full 20-row process. This task is accepted only on the four target rows and
  the three existing controls; every non-target result remains evidence for later ordered work and
  cannot be pooled with or used to weaken a target gate.

### Frozen target rows

| Row | Schema-32 fork-1 ratio | Shared code-shape evidence |
|---|---:|---|
| `M-PAD` | `7.743893186x` | General output odometer reloads rank geometry and recomputes the conditional source address. |
| `M-CONCAT` | `60.725558334x` | General loop repeatedly selects an occurrence and reconstructs coordinates/addresses through prefix and stride arrays. |
| `M-UNFOLD-AXIS` | `8.525273772x` | General loop repeatedly selects the transformed axis and rebuilds its source address from geometry. |
| `M-UNFOLD2D` | `11.226516498x` | General loop repeatedly loads fixed convolution geometry and advances a multi-coordinate output odometer. |

The diagnostic version with all `ByteOrder.nativeOrder()` and `ValueLayout.withOrder(...)`
construction removed still measured these four rows at `1.962945423x`, `10.053775572x`,
`2.065186933x`, and `3.088574486x`, respectively. The residual therefore survives complete
layout-construction removal and is falsifiable independently of CPU 0007A1D's partial fix.

## Out of scope

- Changing or refreezing either retained evidence corpus, probe, case, comparator, dimension,
  threshold, warmup, sample order, fork count, or algorithm.
- Correcting `M-STACK`; its BOOL inserted-axis/occurrence shape remains in CPU 0007A1F.
- Any indexing, scatter, fold, ordering, random, scan, aggregate, pointwise, affine-copy, or
  slice-update correction.
- Changing movement semantics, represented bits, carrier pattern, output layout, resources,
  materialization, worker ownership, routes, capability, public API, Prepare/Runtime behavior,
  dependency, architecture, build, or another module.
- A general IR/lowering redesign, new shared geometry contract, generated fields, constructors,
  secondary methods, generic carriers, mutable static state, per-element allocation/helper calls,
  benchmark tooling, tuning, or a universal performance claim.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [`current architecture plan`](../../../../architecture/current-architecture-plan.md)
- [`performance evidence and tuning`](../../../../architecture/performance-evidence-and-tuning.md)
- [`runtime, Prepare, and backend boundary`](../../../../architecture/runtime-prepare-backend-boundary.md)
- [`planning guide`](../../../planning-guide.md)
- [`documentation rules`](../../../../developer-guide/documentation-rules.md)
- [`general profile`](../../../../developer-guide/documentation/general-style.md)
- [`planning profile`](../../../../developer-guide/documentation/planning-style.md)
- [`API and Javadoc profile`](../../../../developer-guide/documentation/api-and-javadoc-style.md)
- [`backend-guide profile`](../../../../developer-guide/documentation/backend-guide-style.md)
- [`CPU backend guide`](../../../../backend-guide/cpu-backend.md)
- [`glossary`](../../../../glossary.md)
- [`CPU master plan`](../master-plan.md)
- [`CPU 0007A1C`](0007a1c-generated-direct-evidence-closure.md)
- [`CPU 0007A1D`](0007a1d-native-order-segment-layout-hoisting.md)

## Architecture constraints

- Model owns movement semantics; only CPU-private generated realization and its evidence may
  change.
- CPU analysis owns lowering and exact declarations before assignment; CPU finalization owns
  generated bytes afterward; Runtime executes the immutable prepared result without interpreting
  movement geometry or selecting a route.
- Generated entries remain typed, direct, field-free, allocation-free in normal hot work, and
  free of mutable static state or Synaptik-owned per-element calls.
- Benchmark evidence accepts, rejects, or diagnoses the fixed code shape. It never selects or
  mutates production settings.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` — the existing movement emitter,
  generated structural tests, and package documentation.
- `io.github.pho001.synaptik.backend.cpu.internal.cache` — current-only generated compatibility
  schema and package documentation if emitted bytes change.

Packages added or changed:

- None.

Type placement:

- `CpuDataMovementEmitter` remains the sole owner of PAD, CONCAT, UNFOLD_AXIS, and UNFOLD2D
  generated mapping, address, coordinate, and carry/reset emission.
- No second production code-shaping owner is authorized.

## Affected files

Expected:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuDataMovementEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratorSchema.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/package-info.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuDataMovementGeneratedKernelTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuGeneratedDirectEvidenceClosureTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratedKernelArtifactStoreTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecializationTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizerTest.java`
- this task, CPU 0007A1C, CPU 0007A1D, CPU master plan, and roadmap.

The two preparation/finalization tests may change only their existing schema expectation. The CPU
guide, glossary, and prepare package summary change only where schema-33 makes their current
contract text inaccurate. After A1E completes, planning discipline additionally requires the one
next detailed Ready A1F task.

## Maximum scope

The executable implementation context may create or modify at most 15 repository paths:

- 4 production/package paths: one code-shaping owner, schema, and two package summaries;
- 6 tests: movement, closure, two cache/specialization, and two schema-propagation owners; and
- 5 planning paths: this task, CPU 0007A1C, CPU 0007A1D, CPU master plan, and roadmap.

If a second production owner, another test owner, a guide/glossary edit, IR/lowering/resource
change, another module, or more paths are needed, stop and propose a follow-up task.

The mandatory documentation context may additionally correct inaccurate current-contract prose in
the CPU guide, glossary, and prepare package summary and create the detailed A1F specification
required after A1E completion. Those documentation-only additions do not broaden executable scope
or authorize Java behavior changes.

## Acceptance criteria

- Frozen semantic verification passes all 20 rows exactly.
- `M-PAD`, `M-CONCAT`, `M-UNFOLD-AXIS`, and `M-UNFOLD2D` each pass every one of five new isolated
  forks and the median of fork medians at `<= 1.15x` under the unchanged A1C protocol.
- `P-VECTOR-SEGMENT`, `P-INTEGRAL-MIXED`, and `O-ARGSORT` remain `<= 1.15x` in every new fork and
  aggregate.
- Generated target/direct comparators retain exact represented outputs, untouched inputs,
  canaries, scratch/state invariants, sampling, and mapping/order equivalence.
- Stable Class-File assertions show target loop invariants loaded once per invocation and bounded
  primitive coordinate/address cursors in repeated work, while unproved cases retain the typed
  general long fallback.
- Every representative remains field-free with exactly one typed static entry and no constructor,
  bridge, generic `Object`, method-handle/type constant, `invokedynamic`, dynamic constant,
  bootstrap, reflection, collection/boxing, allocation, or forbidden Synaptik-owned reference.
- Emitted-byte changes advance schema exactly once from 32 to 33; schema-32 artifacts become safe
  incompatible misses with no migration, alias, converter, or dual-schema path.
- Only `CpuDataMovementEmitter` changes generated algorithm shape. Semantics, resources, routes,
  capability, public API, Prepare/Runtime behavior, architecture, dependencies, build, and
  unrelated dirty paths remain unchanged.
- A distinct clean documentation context independently finalizes affected Javadocs/package
  summaries, planning evidence/status, CPU guide and glossary impact, and documentation gates.

## Tests / validation

After executable Java stabilizes, run one focused matrix and one authoritative module suite:

```bash
./gradlew :backends:cpu:test \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneratedDirectEvidenceClosureTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuDataMovementGeneratedKernelTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratedKernelArtifactStoreTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecializationTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizerTest
./gradlew :backends:cpu:test
```

Recompile the unchanged frozen A1C sources against stabilized production classes, verify all 20
semantics, and run five new isolated `-Xms1g -Xmx1g` forks. Retain raw results, summaries,
four-row aggregate, regenerated classes, full `javap -c -p` and `javap -v -p`, semantic
Class-File reports, member inventories, environment, commands, and SHA-256 manifests in exactly
one new `/private/tmp/synaptik-cpu-0007a1e-XXXXXXXX` directory. Do not modify either earlier
evidence directory. A nonzero full-probe exit is acceptable only for explicitly deferred rows;
never describe the full process as passed unless all 20 rows pass.

The documentation context reuses stable Java/timing evidence and runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
git diff --cached --check
git status --short
```

It also validates Markdown links/anchors/fences/newlines/whitespace; exact path/owner/schema scope;
one detailed Ready CPU task; A1C/A1D incomplete status; strict correction order; A1C frozen-corpus
preservation; empty staged diff; and worktree preservation. Repository-wide, architecture,
backend-conformance, and integration suites remain deferred to CPU 0009 or CI because no shared
contract may change.

## Dependencies

- CPU 0007A1C remains Incomplete with the immutable 20-row corpus and unchanged `<= 1.15x` gate.
- CPU 0007A1D is Incomplete but supplies stable schema-32 invocation-local segment layouts,
  passing Java/semantic/Class-File evidence, and the failed post-change fork.
- Completed CPU 0006A, CPU 0006A1, and CPU 0007A0A define movement semantics and generated-loop
  foundations that this correction must preserve.

## Follow-up tasks

- CPU 0007A1F is now the detailed Ready task for `M-STACK` and `X-ANY-SINGLE`.
- CPU 0007A1G remains Draft for `F-FOLD2D` and `R-DROPOUT-GENERAL`; create its detailed
  specification only after CPU 0007A1F completes.
- The next evidence-backed owner cluster among the other nine failed A1D targets must remain Draft
  until this task's final evidence is available.
- CPU 0007A1C resumes its unchanged full closure only after all residual corrections complete.
- CPU 0007A2 remains blocked until CPU 0007A1C is Complete.

## Architecture impact

Expected impact: None.

If this task requires an architecture, dependency, public API, semantic, resource, route,
Prepare, Runtime, build, IR, or lowering change, stop and report the issue.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
You are the clean implementation agent for Synaptik CPU task 0007A1E. Work on the existing dirty
worktree without committing, pushing, staging, resetting, reverting, deleting, or modifying
unrelated work. Do not use a GSD skill or workflow.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, CPU task 0007A1E, CPU task
0007A1D, and CPU 0007A1C's frozen evidence contract in full. Implement exactly the bounded
single-owner movement general-address-loop correction. Preserve both retained evidence corpora,
all four frozen mappings/comparators, and the typed general fallback. Stop before a second
production owner or any IR/lowering/resource/architecture/lifecycle change.

Run the specified focused/full CPU, frozen semantic/Class-File/five-fork, schema, scope, and
preservation gates. Hand the stabilized diff and exact evidence to a distinct clean documentation
context for Javadocs/package summaries, guide/glossary/planning review, and documentation gates.
Do not mark Complete until all four targets and all three controls pass every required gate.
```

## Local decisions

- The four targets form one cluster because final generated disassembly places their mapping,
  address, output-odometer, and carry/reset work in `CpuDataMovementEmitter`.
- The diagnostic removal of segment-layout construction still leaves all four above threshold,
  separating this hypothesis from A1D's completed code shape.
- Address-width proof is independent of carrier kind. A segment may use bounded integer cursors
  when its complete accessed byte range and every transition are proved; an array may still need
  typed long addressing when those proofs fail.
- `M-STACK` remains separate because its BOOL inserted-axis occurrence selection and extreme ratio
  require an independent correction hypothesis rather than being assumed to share this fix.

## Known limitations

- Ratios apply only to the frozen cases, JVM, host, and protocol; they grant no universal parity
  or tuning claim.
- This task does not close CPU 0007A1C, make CPU 0007A1D Complete, or authorize CPU 0007A2.
- Non-target rows may remain above threshold and must be recorded honestly without weakening this
  task's four target gates.

## Validation evidence

Implementation evidence is retained at
`/private/tmp/synaptik-cpu-0007a1e-1LiNbBUz`. The implementation context ran the exact focused
six-owner Gradle command in this task and `./gradlew :backends:cpu:test`; both completed with
`BUILD SUCCESSFUL`. No executable Java changed after those results. Frozen-source compilation
used the unchanged A1C `ClosureProbe.java`, `ClosureCases.java`, and `ExactClosure.java`; their
preservation check passed and semantic verification is exactly `VERIFIED,20`.

All five isolated forks passed for every required target and control. Target ratios were:

| Row | Fork 1 | Fork 2 | Fork 3 | Fork 4 | Fork 5 | Median of fork medians | Worst fork |
|---|---:|---:|---:|---:|---:|---:|---:|
| `M-PAD` | `0.823098645x` | `0.819549209x` | `0.816475987x` | `0.817465944x` | `0.815224067x` | `0.817465944x` | `0.823098645x` |
| `M-CONCAT` | `1.101164392x` | `1.004023918x` | `0.970872237x` | `0.983129241x` | `1.143960287x` | `1.004023918x` | `1.143960287x` |
| `M-UNFOLD-AXIS` | `0.892298334x` | `0.896544010x` | `0.891180073x` | `0.894784006x` | `0.892410803x` | `0.892410803x` | `0.896544010x` |
| `M-UNFOLD2D` | `0.777068865x` | `0.801719116x` | `0.825152484x` | `0.811480881x` | `0.791206345x` | `0.801719116x` | `0.825152484x` |
| `P-VECTOR-SEGMENT` | `0.982636686x` | `0.941038455x` | `0.979673878x` | `0.994817395x` | `0.973437126x` | `0.979673878x` | `0.994817395x` |
| `P-INTEGRAL-MIXED` | `0.317414478x` | `0.318093759x` | `0.312088791x` | `0.316726759x` | `0.316688290x` | `0.316726759x` | `0.318093759x` |
| `O-ARGSORT` | `0.894866026x` | `0.886879758x` | `0.851355147x` | `0.878866716x` | `0.864578601x` | `0.878866716x` | `0.894866026x` |

The unchanged full 20-row process exited nonzero in every fork because twelve deferred diagnostic
rows exceeded `1.15x` in forks 1 and 2 and thirteen did so in forks 3 and 4; fork 5 again had
twelve. This is explicitly not a full 20-row performance pass. The four targets and three controls
alone are the task acceptance set.

Retained Class-File evidence includes all 20 generated classes, full `javap -c -p` and
`javap -v -p` output, SHA-256 manifests, semantic reports, and the one-entry member inventory.
`target-forbidden-reference-summary.txt` confirms no target class contains invokedynamic/bootstrap,
reflection, collection, boxing, or a Synaptik-owned runtime reference. `scope-check.txt` confirms
that `CpuDataMovementEmitter` is the only production algorithm-shaping owner, schema advanced
exactly once from 32 to 33, and supporting executable changes are one movement Class-File test and
four schema assertions. The staged diff remained empty and all prior evidence corpora were
preserved.

Clean documentation context `01a01e40-883b-79c3-9bf1-e4ede5e5d05c` applied the General,
API/Javadoc, Backend Guide, Planning, Example, glossary, and package-summary requirements. It
inspected the implementation, structural
test, generated classes/disassembly, retained evidence, affected package contracts, CPU guide,
glossary, A1C–A1E, master plan, and roadmap without changing executable behavior or rerunning Java
tests or performance forks. It finalized schema-33 movement/cache/prepare Javadocs, guide and
glossary accuracy, completion evidence, and the next detailed Ready task. Final documentation
validation passed: `./gradlew :backends:cpu:javadoc` completed with `BUILD SUCCESSFUL` and only the
two expected incubating-Vector warnings; generated package/schema pages contain the bounded,
fallback, and schema-33 text; the eight-file Markdown link/anchor/fence/final-newline check passed;
the exact one-Ready/status check passed; exact retained evidence and deferred-row counts passed;
`git diff --check` and `git diff --cached --check` passed; and the staged diff is empty.

## Implementation notes

- `CpuDataMovementEmitter` now attempts a checked bounded invocation form for PAD, CONCAT,
  UNFOLD_AXIS, and UNFOLD2D before entering the original typed general-long fallback. Geometry,
  entry bounds, address ranges, and cursor transitions must all fit the bounded proof.
- The repeated bounded body advances primitive coordinate and address cursors without reloading
  invariant geometry. Carrier kind does not select the address width.
- Schema 33 is current-only. Schema-32 and earlier envelopes are incompatible safe misses with no
  migration, alias, dual-schema path, or converter.
- Stable Class-File tests assert the bounded cursor loop and retained long fallback without
  depending on bytecode offsets, constant-pool indexes, generated names, or JIT assembly.

## Completion summary

- Completed changes: added cold-proved bounded primitive geometry/cursor loops for the four target
  movement families while retaining the typed general-long fallback; advanced compatibility from
  schema 32 to 33; added the structural Class-File gate; finalized Javadocs, guide, glossary, and
  planning consequences.
- Files changed or created: the permitted movement emitter, schema, movement/cache/prepare package
  documentation, one movement structural test, four schema assertions, A1C–A1E planning records,
  CPU master plan, roadmap, CPU guide, glossary, and the new A1F task specification.
- Tests and validation: reused successful focused and authoritative CPU Gradle results; frozen
  semantics passed `VERIFIED,20`; all 20 target fork ratios and all 15 control fork ratios
  passed `<= 1.15x`; retained structure, forbidden-reference, schema, scope, and preservation gates
  passed; the separate documentation gates passed as recorded above.
- Documentation-agent review: clean context `01a01e40-883b-79c3-9bf1-e4ede5e5d05c`
  independently finalized affected Javadocs, package summaries, guide, glossary, task
  evidence/status, master plan, roadmap, and A1F.
- Documentation impact: the CPU guide now explains schema-33 bounded movement cursors, fallback,
  target-only timing evidence, and the deferred diagnostic boundary.
- Javadoc review: movement, cache, and prepare contracts now consistently describe schema 33,
  carrier-independent bounded movement, typed general-long fallback, direct generated work, and
  safe incompatibility of older artifacts.
- Glossary impact: updated the reusable generated-kernel, specialization, artifact, portable-route,
  and static-data-movement entries for schema 33; no new reusable term was introduced.
- Unresolved issues: CPU 0007A1C remains incomplete because deferred diagnostic rows still fail;
  A1E does not claim full-probe closure.
- Follow-up required: implement Ready CPU 0007A1F for `M-STACK` and `X-ANY-SINGLE`; keep A1G and
  all other residual groups Draft until ordered evidence permits advancement.

Status: Complete
