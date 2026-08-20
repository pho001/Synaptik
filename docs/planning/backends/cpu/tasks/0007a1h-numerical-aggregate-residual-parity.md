# Task 0007A1H: Numerical Aggregate Residual Parity

## Status

Ready

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

Expected production and contract paths:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuAggregateEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratorSchema.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/package-info.java`, only if stale.

Expected tests and schema propagation:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuAggregateGeneratedKernelTest.java`
- the existing four schema-assertion owners, only if the schema advances.
- `CpuGeneratedDirectEvidenceClosureTest` may change only for a stable structural assertion that
  cannot belong in the aggregate owner test; the frozen external corpus itself must not change.

Expected documentation and planning paths:

- this task;
- CPU 0007A1C;
- CPU master plan and roadmap;
- CPU backend guide and glossary only when current contract or evidence text requires an update.

## Maximum scope

This task may modify at most 17 repository paths:

- 1 production algorithm-shaping owner;
- 4 schema/package-contract paths;
- 6 tests: the aggregate owner, optional closure assertion, and four schema owners; and
- 6 documentation/planning paths: this task, A1C, master plan, roadmap, guide, and glossary.

If another production owner, a new helper type, IR/lowering/resource change, another module,
another benchmark case, or an architecture decision is needed, stop and report the conflict.

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

- Other residual families remain Draft and unassigned until A1H evidence supports the next
  bounded owner cluster.
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

Planning context selected this task from the checksummed A1G five-fork summaries. The source
ratios and frozen case definitions are retained under
`/private/tmp/synaptik-cpu-0007a1g-sLBHbIpI`; no A1H executable Java, test, schema, probe, API,
architecture, or build change has been made.

## Implementation notes

Empty until implemented.

## Completion summary

Empty until implemented.
