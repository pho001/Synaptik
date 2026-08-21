# Task 0007A1J: Cumulative scan residual parity

## Status

Ready

## Goal

Correct exactly the persistent `C-SCAN-GENERAL` generated/direct residual through its sole
code-shaping owner, `CpuScanEmitter`. Preserve the frozen CPU 0007A1C semantics, exclusive reverse
INT64 cumulative-product order, mixed segment carriers, arbitrary legal complete-slice subranges,
zero-workspace contract, and typed general fallback while passing the unchanged five-fork
`<= 1.15x` gate.

## Scope

- Use the immutable A1C probe sources and the accepted A1I evidence bundle at
  `/private/tmp/synaptik-cpu-0007a1i-jNhVIZ7F`.
- Correct only `C-SCAN-GENERAL`: exclusive reverse INT64 `CUM_PROD` over Shape `[1024, 1024]`,
  axis one, offset/strided `MemorySegment` input and output, and arbitrary legal complete-slice
  ordinal ranges.
- Preserve reverse traversal, exclusive placement, same-width modular multiplication, exact
  represented loads/stores, logical slice order, scalar or whole-slice parallel-scalar reuse,
  output injectivity, complete overlap rejection, and zero workspace/materialization.
- Treat the frozen optimal direct primitive Java loop as the generated-code design and review
  oracle. Preserve its semantic algorithm, cursor progression, hot-loop dataflow, and
  avoidable-overhead profile.
- Add only a fully guarded proved body to `CpuScanEmitter`. Every geometry or carrier pattern not
  completely proved must retain the existing typed general body.
- Keep the generated target field-free with exactly one typed static `invoke` and no constructor,
  bridge, allocation, reflection, dynamic construct, generic `Object` descriptor, runtime
  operation dispatch, or Synaptik-runtime call.
- Advance the generator schema exactly once if emitted bytes change, propagate it to existing
  schema assertions, and keep older envelopes as incompatible safe misses without migration or
  dual-schema reuse.
- Retain the complete evidence bundle: immutable inputs, exact commands, environment, semantics,
  generated classes, full disassembly, structure/hot-loop reports, five forks, summaries,
  validation, context identifiers, inventory, and checksums.

## Out of scope

- Changing the frozen probe, case, Shape, carriers, comparator, flags, fork protocol, threshold,
  direct oracle, or semantics.
- Correcting `P-SCALAR-GENERAL`, `A-GENERAL`, `S-GENERAL-MIN`, `X-MIN-MULTI`, or any already
  completed row.
- Changing `CpuCarrierEmitter`, another production code-shaping owner, scan lowering/IR,
  validation, resources, orchestration, route selection, Prepare/Runtime behavior, capabilities,
  public API, dependencies, build structure, conformance, or integration.
- Adding a generic scan framework, workspace, partial/combine algorithm, vector/native scan,
  dynamic Shape/layout support, tuning input, or universal performance claim.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [`current architecture plan`](../../../../architecture/current-architecture-plan.md)
- [`planning guide`](../../../planning-guide.md)
- [`documentation rules`](../../../../developer-guide/documentation-rules.md)
- [`CPU backend guide`](../../../../backend-guide/cpu-backend.md)
- [`glossary`](../../../../glossary.md)
- [`CPU master plan`](../master-plan.md)
- [`CPU 0007 cumulative scan`](0007-portable-cumulative-scan-coverage.md)
- [`CPU 0007A0 generated parity correction`](0007a0-generated-hot-path-parity-correction.md)
- [`CPU 0007A1C evidence closure`](0007a1c-generated-direct-evidence-closure.md)
- [`CPU 0007A1I indexing residual parity`](0007a1i-indexing-residual-parity.md)

## Architecture constraints

- Model owns cumulative-scan meaning. This task changes only CPU-private generated realization and
  its evidence.
- CPU analysis retains lowering, strategy, resource declaration, and route selection before shared
  assignment; CPU finalization realizes one artifact afterward; Runtime only invokes prepared
  direct carriers and complete-slice bounds.
- Generated code must preserve the optimal clean Java scan algorithm and hot-loop/dataflow shape.
- Benchmark observations accept or reject this bounded code shape and never select production
  settings.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` — sole scan code-shaping owner and
  generated-kernel tests.
- `io.github.pho001.synaptik.backend.cpu.internal.cache` — schema compatibility when bytes change.
- `io.github.pho001.synaptik.backend.cpu.internal.prepare` — existing schema assertion and
  finalization wording only when the current version changes.

Packages added or changed: None.

Type placement:

- `CpuScanEmitter` remains the only authorized production algorithm-shaping owner.
- No helper, public type, package, or module may be added.

## Affected files

Expected implementation paths:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuScanEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratorSchema.java`
- directly affected existing CPU cache/code-generation/prepare package Javadocs only as required;
- `CpuScanGeneratedKernelTest.java` and `CpuGeneratedDirectEvidenceClosureTest.java` only for
  stable semantic/Class-File assertions;
- the four existing schema-assertion tests when schema advances; and
- CPU backend guide, glossary, this task, CPU 0007A1C, CPU master plan, and roadmap.

## Maximum scope

At most one production algorithm-shaping owner may change. With schema/package propagation, scan
and closure tests, four schema assertion owners, and documentation/planning, the overall change
must remain at or below 18 repository paths. If a second production owner, new helper, IR/lowering
change, resource change, module, or broader path set is required, stop and replan.

## Acceptance criteria

- Frozen semantics remain exactly `VERIFIED,20` against byte-identical sources.
- `C-SCAN-GENERAL` passes `<= 1.15x` in every one of five fresh isolated forks and in the
  cross-fork median.
- Controls `P-VECTOR-SEGMENT`, `P-INTEGRAL-MIXED`, and `O-ARGSORT` pass every fork and median.
- The unchanged full probe exits nonzero only for the four explicitly deferred rows.
- Exact output, input immutability, canaries, complete-slice range ownership, exclusive reverse
  order, modular INT64 multiplication, and zero-workspace/resource contracts remain unchanged.
- Target Class-File and disassembly prove the typed guarded scan body and retained fallback with
  no forbidden structure or avoidable repeated division, remainder, or packed-geometry lookup in
  the proved hot loop.
- Focused scan/closure/schema owners pass, followed by one uncached CPU suite.
- A distinct clean documentation context finalizes affected Javadocs, guide/glossary impact,
  evidence/status records, and documentation validation without rerunning stable Java tests.

## Tests / validation

Run the focused `CpuScanGeneratedKernelTest`, `CpuGeneratedDirectEvidenceClosureTest`, and four
schema-assertion owners, followed once by:

```bash
./gradlew :backends:cpu:test --rerun-tasks
```

Recompile and checksum the frozen sources, run `VERIFIED,20`, retain the target and all required
control/deferred generated classes and complete disassembly, run five sequential fresh JVM forks
with the unchanged `-Xms1g -Xmx1g` protocol, and regenerate the evidence inventory/checksums.

The documentation pass runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
git diff --cached --check
git status --short
```

It also checks Markdown links/anchors/fences/final newlines, schema propagation, exact path bounds,
one Ready task, A1C incomplete, A1J Complete only after every gate, CPU 0007A2 Draft/blocked, and
empty staging. Repository-wide, architecture, conformance, and integration suites remain deferred
to CPU 0009 or CI because no shared contract may change.

## Dependencies

- CPU 0007A1C supplies the immutable twenty-row corpus and remains incomplete.
- CPU 0007 and CPU 0007A0 supply the exact scan semantics and generated typed-body foundation.
- CPU 0007A1A supplies the self-contained BFLOAT16 scan boundary without changing this INT64 row.
- CPU 0007A1D supplies stable invocation-local segment layout handling.
- CPU 0007A1I is Complete at schema 37 and supplies the fresh five-fork residual frontier.

## Follow-up tasks

- The remaining `P-SCALAR-GENERAL`, `A-GENERAL`, `S-GENERAL-MIN`, and `X-MIN-MULTI` rows remain
  Draft and unassigned until fresh accepted evidence selects one bounded owner cluster.
- Resume CPU 0007A1C closure only after every persistent row closes.
- CPU 0007A2 remains Draft and blocked behind CPU 0007A1C and all corrective residual work.

## Architecture impact

Expected impact: None.

If the task requires an architecture, dependency, public API, semantic, resource, route,
Prepare/Runtime, build, IR, lowering, conformance, or integration change, stop and report it.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
You are the clean implementation agent for Synaptik CPU task 0007A1J. Work on the existing dirty
worktree without committing, pushing, staging, resetting, reverting, deleting, or modifying
unrelated work. Do not use a GSD skill or workflow.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, CPU task 0007A1J, CPU tasks
0007/0007A0/0007A1C/0007A1I, the relevant scan/cache/package source and tests, and the complete A1I
evidence bundle. Implement exactly the bounded one-owner C-SCAN-GENERAL correction through
CpuScanEmitter. Preserve the immutable probe, exclusive reverse INT64 modular-product semantics,
mixed carriers, arbitrary legal complete-slice subranges, zero resources, typed fallback, and
optimal direct-clean-Java algorithm/dataflow shape. Stop before changing a second production
owner or any IR/lowering/resource/architecture/lifecycle contract.

Run the specified focused/full CPU, semantic/Class-File/five-fork, forbidden-reference, schema,
scope, and preservation gates. Hand the stabilized diff and exact evidence to a distinct clean
documentation context. Do not mark Complete until the target and all controls pass every gate.
Do not commit, push, or stage.
```

## Local decisions

- A1I's fresh five forks show `C-SCAN-GENERAL` fails every fork from `9.306481960x` to
  `9.766779261x`, so it is persistent rather than a one-fork anomaly.
- The row is selected before the numerically larger scatter residual because scan has one exact
  code-shaping owner, one fixed sequential algorithm, zero workspace, and already completed scan
  semantics/generated-body dependencies. Scatter retains separate range-copy, update filtering,
  reduction, and duplicate-order concerns that should not be pulled into this bounded task.
- No other residual shares the scan owner or algorithm, so this remains a one-row task.

## Known limitations

- Ratios apply only to the frozen case, JVM, host, and protocol; they are not a universal scan
  claim or production tuning input.
- This task does not close A1C or authorize CPU 0007A2.

## Validation evidence

Empty until implemented.

## Implementation notes

Empty until implemented.

## Completion summary

Empty until implemented.
