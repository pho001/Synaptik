# Task 0007A1K: Affine-copy residual parity

## Status

Ready

## Goal

Correct exactly the persistent `A-GENERAL` generated/direct residual through its sole code-shaping
owner, `CpuAffineCopyEmitter`. Preserve the frozen CPU 0007A1C raw-BFLOAT16 copy semantics,
composed PERMUTE/SLICE mapping, segment-to-heap carriers, nonzero offsets, positive non-unit
strides, arbitrary legal output subranges, and typed general fallback while passing the unchanged
five-fork `<= 1.15x` gate.

## Scope

- Use the immutable A1C probe sources and the accepted A1J evidence bundle at
  `/private/tmp/synaptik-cpu-0007a1j-96RvT1wf`.
- Correct only `A-GENERAL`: raw-BFLOAT16 PERMUTE/SLICE affine copy over Shape `[256,32,32]`,
  `MemorySegment` input to `short[]` output, nonzero offsets, positive non-unit strides, and the
  frozen composed logical mapping.
- Preserve represented bits exactly, including NaN payloads and signed zeros; the copy performs no
  BFLOAT16 arithmetic, conversion, promotion, canonicalization, or numerical interpretation.
- Preserve output-domain ownership, distinct-address writes, complete overlap rejection, input
  immutability, arbitrary legal half-open output ranges, scalar or parallel-scalar reuse, and zero
  workspace/materialization beyond the already owned affine boundary copy itself.
- Treat the frozen optimal direct primitive Java loop as the generated-code design and review
  oracle. Preserve its semantic mapping, cursor/odometer progression, carrier work, hot-loop
  dataflow, and avoidable-overhead profile.
- Add only a completely guarded proved body to `CpuAffineCopyEmitter`. Every mapping, carrier, or
  layout fact not completely proved must retain the existing typed general-long body.
- Keep the generated target final and field-free with exactly one typed static `invoke`, no
  constructor or bridge, and no allocation, reflection, dynamic construct, generic `Object`
  descriptor, runtime operation dispatch, or Synaptik-runtime call.
- Advance the generator schema exactly once if emitted bytes change, propagate it to existing
  schema assertions, and keep older envelopes as incompatible safe misses without migration or
  dual-schema reuse.
- Retain immutable inputs, exact commands, environment, semantics, generated classes, complete
  disassembly, structure/hot-loop reports, five accepted forks, rejected samples, summaries, XML,
  context identifiers, inventories, and checksums.

## Out of scope

- Changing the frozen probe, case, Shape, carriers, comparator, JVM flags, fork protocol,
  threshold, direct oracle, or semantics.
- Correcting `P-SCALAR-GENERAL`, `S-GENERAL-MIN`, `X-MIN-MULTI`, or any completed row.
- Changing `CpuCarrierEmitter`, another production code-shaping owner, affine lowering/IR, view
  folding, validation, resources, orchestration, route selection, Prepare/Runtime behavior,
  capabilities, public API, dependencies, build structure, conformance, or integration.
- Adding a general layout optimizer, per-element address table, workspace, materialization route,
  vector/native affine copy, dynamic Shape/layout support, tuning input, or universal performance
  claim.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [`current architecture plan`](../../../../architecture/current-architecture-plan.md)
- [`planning guide`](../../../planning-guide.md)
- [`documentation rules`](../../../../developer-guide/documentation-rules.md)
- [`CPU backend guide`](../../../../backend-guide/cpu-backend.md)
- [`glossary`](../../../../glossary.md)
- [`CPU master plan`](../master-plan.md)
- [`CPU 0006 affine views`](0006-portable-static-affine-views-and-boundary-materialization.md)
- [`CPU 0007A0A affine and movement parity`](0007a0a-affine-and-movement-generated-loop-parity.md)
- [`CPU 0007A1C evidence closure`](0007a1c-generated-direct-evidence-closure.md)
- [`CPU 0007A1J cumulative-scan residual parity`](0007a1j-cumulative-scan-residual-parity.md)

## Architecture constraints

- Model owns view semantics and represented data types. This task changes only CPU-private
  generated affine-copy realization and evidence.
- CPU analysis retains view composition, lowering, strategy, resource declaration, and route
  selection before shared assignment; CPU finalization realizes one artifact afterward; Runtime
  only invokes prepared direct carriers and output bounds.
- Generated code must preserve the optimal clean Java affine-copy mapping and hot-loop/dataflow
  shape without interpreting Model operations or layouts at Runtime.
- Benchmark observations accept or reject this bounded code shape and never select production
  settings.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` — sole affine-copy code-shaping
  owner and generated-kernel tests.
- `io.github.pho001.synaptik.backend.cpu.internal.cache` — schema compatibility when bytes change.
- `io.github.pho001.synaptik.backend.cpu.internal.prepare` — existing schema assertion and
  finalization wording only when the current version changes.

Packages added or changed: None.

Type placement:

- `CpuAffineCopyEmitter` remains the only authorized production algorithm-shaping owner.
- No helper, public type, package, or module may be added.

## Affected files

Expected implementation paths:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuAffineCopyEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratorSchema.java`
- directly affected existing CPU cache/code-generation/prepare package Javadocs only as required;
- `CpuAffineCopyGeneratedKernelTest.java` and `CpuGeneratedDirectEvidenceClosureTest.java` only for
  stable semantic/Class-File assertions;
- the four existing schema-assertion tests when schema advances; and
- CPU backend guide, glossary, this task, CPU 0007A1C, CPU master plan, and roadmap.

## Maximum scope

At most one production algorithm-shaping owner may change. With schema/package propagation,
affine/closure tests, four schema assertion owners, and documentation/planning, the overall change
must remain at or below 18 repository paths. If a second production owner, new helper,
IR/lowering/resource change, module, or broader path set is required, stop and replan.

## Acceptance criteria

- Frozen sources recompile byte-identically and exact semantics remain `VERIFIED,20`.
- `A-GENERAL` passes `<= 1.15x` in every one of five fresh isolated forks and in the cross-fork
  median.
- Controls `P-VECTOR-SEGMENT`, `P-INTEGRAL-MIXED`, and `O-ARGSORT` pass every fork and median.
- The unchanged full probe exits nonzero only for the three explicitly deferred rows.
- Exact raw-BFLOAT16 output bits, input immutability, canaries, composed mapping, arbitrary legal
  output-range ownership, and zero-resource contracts remain unchanged.
- Target Class-File and complete disassembly prove the typed guarded affine body and retained
  fallback, with no avoidable per-element coordinate division/remainder, packed-geometry lookup,
  layout construction, generic dispatch, or helper call in the proved hot loop.
- Focused affine/closure/schema owners pass, followed by one uncached CPU suite.
- A distinct clean documentation context finalizes affected Javadocs, guide/glossary impact,
  evidence/status records, and documentation validation without rerunning stable Java tests.

## Tests / validation

Run the focused `CpuAffineCopyGeneratedKernelTest`, `CpuGeneratedDirectEvidenceClosureTest`, and
four schema-assertion owners, followed once by:

```bash
./gradlew :backends:cpu:test --rerun-tasks
```

Recompile and checksum the frozen sources, run `VERIFIED,20`, retain the target and all required
control/deferred generated classes and complete disassembly, run five sequential fresh JVM forks
with the unchanged `-Xms1g -Xmx1g` protocol, and regenerate the evidence inventory/checksums. A
complete five-fork sample is accepted only if the target, controls, and every non-deferred row
pass; retain and reject the whole sample otherwise.

The documentation pass runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
git diff --cached --check
git status --short
```

It also checks Markdown links/anchors/fences/final newlines, schema propagation, exact path bounds,
one Ready task, A1C incomplete, A1K Complete only after every gate, CPU 0007A2 Draft/blocked, and
empty staging. Repository-wide, architecture, conformance, and integration suites remain deferred
to CPU 0009 or CI because no shared contract may change.

## Dependencies

- CPU 0007A1C supplies the immutable twenty-row corpus and remains incomplete.
- CPU 0006 and CPU 0007A0A supply exact affine-view composition and generated typed-copy
  foundations.
- CPU 0007A1D supplies stable invocation-local segment layout handling.
- CPU 0007A1J is Complete at schema 38 and supplies the accepted five-fork residual frontier.

## Follow-up tasks

- The remaining `P-SCALAR-GENERAL`, `S-GENERAL-MIN`, and `X-MIN-MULTI` rows remain Draft and
  unassigned until accepted evidence selects one bounded owner cluster.
- Resume CPU 0007A1C closure only after every persistent row closes.
- CPU 0007A2 remains Draft and blocked behind CPU 0007A1C and all corrective residual work.

## Architecture impact

Expected impact: None.

If the task requires an architecture, dependency, public API, semantic, resource, route,
Prepare/Runtime, build, IR, lowering, conformance, or integration change, stop and report it.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
You are the clean implementation agent for Synaptik CPU task 0007A1K. Work on the existing dirty
worktree without committing, pushing, staging, resetting, reverting, deleting, or modifying
unrelated work. Do not use a GSD skill or workflow.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, CPU task 0007A1K, CPU tasks
0006/0007A0A/0007A1C/0007A1J, the relevant affine/cache/package source and tests, and the complete
A1J evidence bundle. Implement exactly the bounded one-owner A-GENERAL correction through
CpuAffineCopyEmitter. Preserve the immutable probe, raw-BFLOAT16 represented-bit copy, composed
PERMUTE/SLICE mapping, segment-to-heap carriers, offsets/strides, arbitrary legal output ranges,
zero resources, typed fallback, and optimal direct-clean-Java algorithm/dataflow shape. Stop
before changing a second production owner or any IR/lowering/resource/architecture/lifecycle
contract.

Run the specified focused/full CPU, semantic/Class-File/five-fork, forbidden-reference, schema,
scope, and preservation gates. Hand the stabilized diff and exact evidence to a distinct clean
documentation context. Do not mark Complete until the target and all controls pass every gate.
Do not commit, push, or stage.
```

## Local decisions

- Accepted A1J forks measure `A-GENERAL` at `4.569629881`, `4.621004147`, `4.614792202`,
  `4.547987466`, and `4.651517540`, with median `4.614792202`, while all three controls pass.
  The row is therefore persistent rather than a one-fork anomaly.
- This row is selected before the other residuals because it has one exact copy owner, represented-
  bit semantics, one output-domain pass, and zero workspace. Pointwise general addressing,
  scatter copy/update/reduction, and multi-axis extrema retain distinct owner and algorithm risks.
- No other residual shares the affine-copy owner or raw-copy algorithm, so this remains a one-row
  task.

## Known limitations

- Ratios apply only to the frozen case, JVM, host, and protocol; they are not a universal affine-
  copy claim or production tuning input.
- This task does not close A1C or authorize CPU 0007A2.

## Validation evidence

Empty until implemented.

## Implementation notes

Empty until implemented.

## Completion summary

Empty until implemented.
