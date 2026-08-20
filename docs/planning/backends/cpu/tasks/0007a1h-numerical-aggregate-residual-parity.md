# Task 0007A1H: Numerical Aggregate Residual Parity

## Status

Complete

## Goal

Correct exactly the two persistent ordinary numerical aggregate residuals in the frozen CPU
0007A1C comparison: `N-MEAN-GENERAL` and `N-PROD-MULTI`. Preserve their exact numerical and
resource semantics while making each generated entry pass the unchanged generated/direct
`<= 1.15x` gate in every one of five isolated forks and in the median of fork medians.

Both rows are owned by one existing code-shaping production type, `CpuAggregateEmitter`. This is
the sole next bounded residual task after completed CPU 0007A1G.

## Scope

- Use the immutable frozen A1C sources under `/private/tmp/synaptik-cpu-0007a1c-oMAmuVhF` and the
  completed A1G evidence under `/private/tmp/synaptik-cpu-0007a1g-sLBHbIpI`.
- Correct `N-MEAN-GENERAL`: FLOAT32 axis-one MEAN over input Shape `[128, 2048]`, offset/interleaved
  `MemorySegment` input, offset/interleaved `float[]` output, exact run-owned sum workspace, and
  output-cell range `[0, 128)`.
- Correct `N-PROD-MULTI`: BFLOAT16 PROD over selected axes `[0, 2]` of input Shape
  `[4, 16384, 4]`, retained output Shape `[1, 16384, 1]`, `short[]` input, offset/interleaved
  `MemorySegment` output, exact run-owned product workspace, and output-cell range `[0, 16384)`.
- Preserve arbitrary legal half-open output-cell subranges, including correct range-private
  workspace slices and mixed carriers required by the frozen cases.
- Preserve exact floating SUM/MEAN and PROD special-value, limb-state, normalization, rounding,
  signed-zero, infinity, NaN, empty-domain, and result-format rules. Retain typed general
  fallbacks for every geometry or carrier pattern not completely cold-proved.
- Treat the frozen optimal clean Java comparator as the design and review oracle. A proved
  generated branch must be algorithm- and dataflow-equivalent, including logical visitation,
  accumulator/state transitions, final division or product rounding, workspace use, output
  ownership, and avoidable-overhead profile. Any deviation requires an explicit technical reason
  and supporting semantic, structural, and measured evidence.
- Keep repeated proved work free of allocation, boxing, reflection, dynamic dispatch, generic
  carrier dispatch, runtime operation dispatch, and Synaptik-owned helper calls. Invocation-local
  primitives and the already declared exact workspace remain permitted.
- Add stable semantic and Class-File assertions for the two frozen shapes and retained fallback;
  do not assert bytecode offsets, constant-pool indexes, generated names, or JIT assembly.
- Advance the current-only generator schema exactly once if emitted bytes change, and propagate
  that version to the existing assertion owners. Older envelopes remain incompatible safe misses.
- Run the unchanged full 20-row process. Accept only the two targets and the three established
  controls. Every other row remains diagnostic and unassigned.

### Frozen residual evidence

| Row | Fork 1 | Fork 2 | Fork 3 | Fork 4 | Fork 5 | Median | Worst fork | Owner |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `N-MEAN-GENERAL` | `1.466213575x` | `1.502648431x` | `1.830559684x` | `1.509453346x` | `1.516850531x` | `1.509453346x` | `1.830559684x` | `CpuAggregateEmitter` |
| `N-PROD-MULTI` | `1.282352813x` | `1.290920776x` | `1.286761010x` | `1.325092312x` | `1.339981240x` | `1.290920776x` | `1.339981240x` | `CpuAggregateEmitter` |

Both rows fail every fork and their aggregate. They are a persistent one-owner numerical
aggregate cluster, not a one-fork anomaly.

### Accepted A1H results

| Row | Fork 1 | Fork 2 | Fork 3 | Fork 4 | Fork 5 | Median | Worst fork | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `N-MEAN-GENERAL` | `0.995106956x` | `0.929692351x` | `0.973095585x` | `0.863214616x` | `0.987138897x` | `0.973095585x` | `0.995106956x` | PASS |
| `N-PROD-MULTI` | `0.956758187x` | `1.022472818x` | `1.042800553x` | `1.018639544x` | `1.084500356x` | `1.022472818x` | `1.084500356x` | PASS |
| `P-VECTOR-SEGMENT` | `0.977816010x` | `0.973037939x` | `0.979055127x` | `0.985127293x` | `0.969229603x` | `0.977816010x` | `0.985127293x` | PASS |
| `P-INTEGRAL-MIXED` | `0.317019234x` | `0.313776841x` | `0.313981868x` | `0.323950075x` | `0.314372687x` | `0.314372687x` | `0.323950075x` | PASS |
| `O-ARGSORT` | `0.856355793x` | `0.869676804x` | `0.862072144x` | `0.860148580x` | `0.852127443x` | `0.860148580x` | `0.869676804x` | PASS |

All five full probe processes exited nonzero only because the same seven deferred diagnostic rows
exceeded the gate: `P-SCALAR-GENERAL`, `A-GENERAL`, `I-GATHER`, `I-GATHER-ND`,
`S-GENERAL-MIN`, `C-SCAN-GENERAL`, and `X-MIN-MULTI`. The accepted results close the two A1H
targets and three controls; they are not a full twenty-row performance pass.

## Out of scope

- Changing or refreezing A1C sources, cases, comparators, dimensions, layouts, carriers, resource
  declarations, threshold, warmup, sample order, fork count, or algorithms.
- Correcting pointwise, affine, movement, indexing, scatter, fold, ordering, random, scan,
  extrema, or Boolean residuals.
- Reopening `M-CONCAT`: its one A1G ratio of `1.158061861x`, four passing A1G forks, and completed
  A1E five-fork evidence do not establish a persistent failure.
- Changing aggregate semantics, exact-state design, workspace ownership or size, route selection,
  capability, public API, shared Prepare/Runtime behavior, architecture, dependencies, build,
  IR, lowering, or benchmark tooling.
- Changing `CpuExactSumEmitter`, `CpuExactProductEmitter`, or another production code-shaping
  owner. If such a change is required, stop and replan rather than broadening this task.
- Universal generated-code, aggregate, JIT, hardware, or performance claims.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [`current architecture plan`](../../../../architecture/current-architecture-plan.md)
- [`performance evidence and tuning`](../../../../architecture/performance-evidence-and-tuning.md)
- [`runtime, Prepare, and backend boundary`](../../../../architecture/runtime-prepare-backend-boundary.md)
- [`planning guide`](../../../planning-guide.md)
- [`documentation rules`](../../../../developer-guide/documentation-rules.md)
- [`CPU backend guide`](../../../../backend-guide/cpu-backend.md)
- [`glossary`](../../../../glossary.md)
- [`CPU master plan`](../master-plan.md)
- [`CPU 0007A1`](0007a1-portable-ordinary-numerical-aggregate-reductions.md)
- [`CPU 0007A1C`](0007a1c-generated-direct-evidence-closure.md)
- [`CPU 0007A1G`](0007a1g-fold-and-dropout-residual-parity.md)

## Architecture constraints

- Model owns aggregate meaning. This task changes only CPU-private generated realization and its
  evidence.
- CPU analysis retains exact lowering, resource declaration, route selection, and strategy before
  shared slot assignment. CPU finalization realizes the selected artifact afterward. Runtime only
  invokes the immutable prepared executable.
- Exact floating work uses only its already declared run-owned, range-sliced workspace. This task
  adds no hidden scratch, partial reduction, combine state, or mutable prepared state.
- Generated entries remain typed, direct, field-free, and allocation-free in proved repeated work.
- Benchmark observations accept or reject fixed code shape and never select production settings.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` — the sole algorithm-shaping owner,
  exact aggregate emitters as read-only semantic references, tests, and package documentation.
- `io.github.pho001.synaptik.backend.cpu.internal.cache` — current-only generator schema and
  artifact compatibility documentation if emitted bytes change.
- `io.github.pho001.synaptik.backend.cpu.internal.prepare` — existing aggregate resource and
  finalization summary only if its current-schema wording becomes stale.

Packages added or changed:

- None.

Type placement:

- `CpuAggregateEmitter` is the only authorized production code-shaping owner.
- No second algorithm helper or unrelated emitter may be changed under this task.

## Affected files

Changed production and contract paths:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuAggregateEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratorSchema.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/package-info.java`

Changed tests and schema propagation:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuAggregateGeneratedKernelTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratedKernelArtifactStoreTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecializationTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizerTest.java`

Changed documentation and planning paths:

- this task;
- CPU 0007A1C;
- CPU 0007A1I;
- CPU master plan and roadmap;
- CPU backend guide and glossary.

## Maximum scope

The completed change uses 17 repository paths:

- 1 production algorithm-shaping owner;
- 4 schema/package-contract paths;
- 5 tests: the aggregate owner and four schema owners; and
- 7 documentation/planning paths: this task, A1C, the directly requested next-frontier A1I
  specification, master plan, roadmap, guide, and glossary.

The direct documentation-agent request authorized creation of the one immediately following
detailed task when required by the planning guide; it did not broaden executable implementation.
No other production owner, helper type, IR/lowering/resource change, module, benchmark case, or
architecture decision was needed.

## Acceptance criteria

- Frozen semantic verification remains exactly `VERIFIED,20` against byte-identical A1C sources.
- `N-MEAN-GENERAL` and `N-PROD-MULTI` each pass all five new isolated forks and their median at
  `<= 1.15x` under the unchanged protocol.
- `P-VECTOR-SEGMENT`, `P-INTEGRAL-MIXED`, and `O-ARGSORT` pass every new fork and aggregate.
- Both generated target outputs match the frozen comparator bit-for-bit; inputs, canaries,
  workspace boundaries, range ownership, visitation order, exact-state transitions, and resource
  declarations remain unchanged.
- Stable Class-File evidence proves the selected direct primitive shape and a retained typed
  general fallback in the target classes.
- Generated/decompiled proved branches are algorithm- and dataflow-equivalent to optimal clean
  Java, including exact-state operations and final result rounding. Any deviation is explicitly
  justified and measured.
- Repeated proved work has no allocation, boxing, reflection, dynamic dispatch, generic bridge,
  runtime operation dispatch, or Synaptik-owned helper call. Each target class is a field-free
  final class with exactly one typed static `invoke`, no constructor or bridge, no `Object`
  descriptor, and no method handle, `invokedynamic`, dynamic constant, or bootstrap.
- If emitted bytes change, schema advances exactly once from 35 and all older envelopes remain
  incompatible safe misses without migration, conversion, aliasing, or dual-schema reuse.
- Only `CpuAggregateEmitter` changes generated algorithm shape. Semantics, public API, capability,
  resources, routes, Prepare/Runtime behavior, architecture, dependencies, and build remain
  unchanged.
- A distinct clean documentation context independently finalizes affected Javadocs/package
  summaries, evidence/status, guide/glossary impact, and documentation validation.

## Tests / validation

After executable Java stabilizes, run focused owners and one authoritative uncached CPU suite:

```bash
./gradlew :backends:cpu:test \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneratedDirectEvidenceClosureTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuAggregateGeneratedKernelTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratedKernelArtifactStoreTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecializationTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizerTest
./gradlew :backends:cpu:test --rerun-tasks
```

Recompile unchanged A1C sources, run exact 20-row semantics, and run five sequential isolated
`-Xms1g -Xmx1g` forks. Retain raw output, all row summaries, required target/control aggregates,
all 20 generated classes, complete `javap -c -p` and `javap -v -p`, member/descriptor/forbidden-
reference and allocation reports, clean-Java-equivalence analysis, environment, commands, exact
changed paths, staged-state evidence, schema compatibility, and SHA-256 manifests in one new A1H
evidence directory. A nonzero full-probe process is acceptable only for named deferred rows and
must not be reported as a full performance pass.

The documentation context reuses successful Java and timing evidence and runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
git diff --cached --check
git status --short
```

It also validates Markdown links, anchors, fences, final newlines, status/frontier consistency,
exact path bounds, one Ready detailed task, schema propagation, immutable frozen sources, empty
staged diff, direct-clean-Java equivalence evidence, and concurrent worktree preservation.
Repository-wide, architecture, conformance, and integration suites remain deferred to CPU 0009
or CI because this task must not change a shared contract.

## Dependencies

- CPU 0007A1C supplies the immutable 20-row corpus and remains `Review needed`/Incomplete.
- CPU 0007A1G is Complete at schema 35 and supplies the final five-fork residual evidence.
- CPU 0007A1 and CPU 0007A0F define the exact numerical aggregate semantics, resources, and
  direct generated bodies to preserve.
- CPU 0007A1D remains `Review needed`/Incomplete; its stable schema-32 segment-layout hoisting is
  retained but its incomplete performance claim is not inherited.

## Follow-up tasks

- CPU 0007A1I is the sole next `Ready` task. It owns the persistent `I-GATHER` and `I-GATHER-ND`
  residuals through one code-shaping owner, `CpuIndexingEmitter`.
- Other residual families remain Draft and unassigned.
- CPU 0007A1C resumes closure only after all corrective residual work is complete.
- CPU 0007A2 remains blocked behind CPU 0007A1C and all corrective residual work.

## Architecture impact

Expected impact: None.

If implementation requires an architecture, dependency, public API, semantic, exact-resource,
route, Prepare, Runtime, build, IR, or lowering change, stop and report the issue.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
You are the clean implementation agent for Synaptik CPU task 0007A1H. Work on the existing dirty
worktree without committing, pushing, staging, resetting, reverting, deleting, or modifying
unrelated work. Do not use a GSD skill or workflow.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, CPU task 0007A1H, CPU tasks
0007A1/0007A1C/0007A1G, and the retained A1C/A1G evidence in full. Implement exactly the bounded
one-owner numerical aggregate correction for N-MEAN-GENERAL and N-PROD-MULTI through
CpuAggregateEmitter. Preserve the immutable corpus, exact floating and BFLOAT16 semantics,
workspace/resource shape, mixed carriers, arbitrary legal output-cell subranges, and typed
general fallbacks. Treat optimal direct clean Java as the generated-code design and review oracle;
preserve its algorithm, control flow, hot-loop dataflow, exact-state transitions, and avoidable-
overhead profile. Stop before changing an exact sum/product helper, a second production owner, or
any IR/lowering/resource/architecture/lifecycle contract.

Run the specified focused/full CPU, semantic/Class-File/five-fork, allocation/reference, schema,
scope, clean-Java-equivalence, and preservation gates. Hand the stabilized diff and exact evidence
to a distinct clean documentation context. Do not mark Complete until both targets and all three
controls pass every required gate. Do not commit, push, or stage.
```

## Local decisions

- The final A1G evidence shows both numerical rows fail all five forks and their aggregate.
- They share one code-shaping owner, exact-state resource model, output-cell execution domain,
  frozen evidence lifecycle, and generated/direct clean-Java requirement, so one cohesive task is
  smaller and clearer than two mechanical tasks.
- No other residual joins this task. In particular, extrema, scan, indexing, pointwise, and affine
  rows have different owners or algorithms.
- The fixed clean Java comparators use the same exact numerical state machinery as production
  semantics; passing the gate may remove avoidable generated shape overhead but may not substitute
  a simpler inexact accumulation algorithm.

## Known limitations

- Ratios apply only to the frozen cases, JVM, host, and protocol. They establish neither universal
  aggregate parity nor a production tuning decision.
- This task does not close CPU 0007A1C or authorize CPU 0007A2.
- Deferred rows may keep every full process nonzero and remain diagnostic.

## Validation evidence

Implementation context `01a020b0-0053-7f70-8c11-ede67b6ee1e9` retained complete evidence under
`/private/tmp/synaptik-cpu-0007a1h-JFko4MLE`. The unchanged frozen sources recompiled and remained
byte-identical; exact semantics reported `VERIFIED,20`. The exact focused owner command passed,
and the authoritative uncached `./gradlew :backends:cpu:test --rerun-tasks` run passed 352 tests
with zero failures, zero errors, and one expected opt-in skip. All two targets and three controls
passed every one of five isolated forks and their medians at `<= 1.15x`; all processes exited one
only because seven named deferred diagnostic rows failed.

The retained evidence includes twenty generated classes; complete `javap -c -p` and
`javap -v -p`; class/member/descriptor/allocation/forbidden-reference reports; direct-clean-Java
equivalence analysis; schema compatibility; raw and summarized forks; environment, commands,
exact implementation paths, staged-state evidence, and SHA-256 manifests. The two target classes
are field-free final classes with exactly one typed static `invoke`, no constructor or bridge,
and no allocation, reflection, dynamic construct, generic `Object` descriptor, or Synaptik runtime
reference. Their proved hot-loop regions contain no packed-geometry lookup, division, or remainder;
the complete classes retain those operations in guards and typed fallbacks.

Clean documentation context `01a020db-ef9b-7e20-bf46-30937a980026` selected the General,
API/Javadoc, Planning, and Backend Guide profiles. It independently reviewed the implementation
diff, affected tests and Javadocs, package summaries, CPU guide, glossary, planning records,
Compile API, Training API, capability baseline, architecture/current plan, conformance and
integration boundaries, and retained implementation evidence. It changed no executable Java or
test. CPU Javadoc, Markdown/link/anchor/fence/final-newline checks, schema/current-wording and
status/frontier searches, exact 17-path and staged-state review, concurrent-work preservation,
and final whitespace checks passed. Java tests and timing were not rerun because executable Java
did not change during this pass.

## Implementation notes

- `N-MEAN-GENERAL` uses a complete runtime guard for the frozen FLOAT32 axis-one Shape
  `[128, 2048]`, mixed carriers, exact five-limb/48-byte state, layout geometry, and legal range.
  Its proved body visits increasing cells and exactly 2,048 factors per cell, keeps exact state in
  primitive locals, writes the identical final workspace state, and implements division by
  `2048 == 2^11` through equivalent exact extraction and ties-to-even rounding. The typed general
  fallback remains in the same class.
- `N-PROD-MULTI` uses a complete runtime guard for the frozen BFLOAT16 Shape `[4, 16384, 4]`,
  axes `[0, 2]`, retained Shape `[1, 16384, 1]`, mixed carriers, two-limb/40-byte state, and legal
  range. Its primitive geometry cursors visit the exact four-outer-by-four-inner factor order and
  retain unchanged `CpuExactProductEmitter` state transitions. The typed general fallback remains.
- `CpuExactSumEmitter` and `CpuExactProductEmitter` source files are unchanged. Schema advanced
  exactly once from 35 to 36; older envelopes are incompatible safe misses with no migration,
  conversion, alias, or dual-schema reuse.
- Only `CpuAggregateEmitter` changes generated algorithm shape. No public API, capability,
  resource size/ownership, lowering, route, Prepare/Runtime behavior, dependency, build,
  conformance, integration, Compile API, or Training API contract changed.

## Completion summary

- Completed changes: finalized the schema-36 aggregate and compatibility Javadocs; synchronized
  cache, code-generation, and prepare package summaries; documented the two guarded exact-state
  forms and bounded evidence in the CPU guide and glossary; recorded complete semantic,
  Class-File, schema, Java-test, and five-fork evidence; marked A1H Complete; and created CPU
  0007A1I as the sole next Ready residual task.
- Files changed or created by the documentation pass: `CpuGeneratorSchema.java`,
  `CpuAggregateEmitter.java`, cache/codegen/prepare `package-info.java`, CPU backend guide,
  glossary, A1H, A1C, new A1I, CPU master plan, and roadmap. The two Java owners received
  documentation-only edits; implementation behavior and tests were preserved.
- Tests and validation: reused the implementation context's passing focused owners, 352-test
  uncached CPU suite, `VERIFIED,20`, Class-File reports, and five-fork evidence. The documentation
  pass ran CPU Javadoc and documentation/scope/schema/status/staging/whitespace validation; it did
  not rerun Java tests or timing.
- Documentation-agent review: clean context `01a020db-ef9b-7e20-bf46-30937a980026`.
- Documentation impact: only CPU-private generated-code/schema, package, guide, glossary, and
  planning evidence changed. Architecture/current-plan, public API, capabilities, Compile API,
  Training API, conformance, integration, and build documentation remain accurate because no such
  contract changed.
- Javadoc review: affected type/member Javadocs document guarded geometry, exact state,
  subranges, fallback, schema compatibility, parameters, and failure behavior; CPU Javadoc passes.
- Glossary impact: current generated-kernel/schema definitions advanced to schema 36; no new
  reusable term was introduced.
- Concurrent-work preservation: the unrelated user-approved Conv1d/Conv3d edits in the six named
  planning files were preserved; only the smallest overlapping A1H/A1I CPU frontier sentences in
  the CPU master plan and roadmap were changed by this pass.
- Unresolved issues: seven deferred performance rows remain; A1C and A1D remain `Review needed`/
  Incomplete, and A1H does not close the twenty-row comparison.
- Follow-up required: execute CPU 0007A1I, then plan only later evidence-backed residual clusters;
  resume A1C closure only after every persistent frozen failure closes. CPU 0007A2 remains blocked.

Status: Complete
