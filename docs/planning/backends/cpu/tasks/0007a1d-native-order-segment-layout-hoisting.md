# Task 0007A1D: Native-Order Segment Layout Hoisting

## Status

Review needed

## Goal

Correct the first evidence-backed code-shape defect exposed by CPU 0007A1C: generated scalar
`MemorySegment` access repeatedly reconstructs native-order typed layouts inside hot loops.

Hoist the exact typed native-order layout state once per generated invocation and make the 13
frozen affected rows meet their unchanged generated/direct `<= 1.15x` gate. Preserve semantics,
carrier/address geometry, descriptors, the frozen probe and comparators, and every non-target
family algorithm.

## Scope

- Use the unchanged frozen sources and evidence under
  `/private/tmp/synaptik-cpu-0007a1c-oMAmuVhF`.
- Correct only the shared generated segment-access shape owned by `CpuCarrierEmitter`; a necessary
  entry-body setup change may also use `CpuClassFileKernelGenerator` as the second and final
  code-shaping owner.
- Initialize each required native-order typed layout outside repeated scalar load/store work and
  reuse its typed local. Do not add generated fields, constructors, secondary methods, generic
  carriers, mutable static state, or per-element helper calls.
- Preserve exact primitive array and Vector API access. A vector segment access may continue to
  consume `ByteOrder.nativeOrder()` only if retained Class-File and performance evidence proves the
  call is outside repeated vector work or otherwise remains at parity.
- Regenerate representatives and advance the generated-kernel schema exactly once when bytes
  change. Earlier artifacts become safe incompatible misses with no migration path.
- Propagate that schema-only change through the existing schema-version expectations in
  `CpuPartitionPreparerTest` and `CpuPartitionFinalizerTest`. In those two test owners, only the
  existing expected version may change from 31 to 32; preparation/finalization behavior and all
  other assertions remain frozen.
- Add stable Class-File assertions that fail on repeated hot-loop `ByteOrder.nativeOrder()` or
  `ValueLayout.withOrder(...)` construction for the affected scalar segment forms.
- Re-run the full frozen 20-row process without modifying its cases, thresholds, algorithms,
  dimensions, order, warmup, sampling, or comparators. Evaluate this task against the 13 target
  rows below. The unchanged A1D gate requires every target row to be at most `1.15x`; the final
  fork failed all 13 rows, and no later corrective task changes or retroactively satisfies that
  failed acceptance criterion.

### Frozen target rows

The 13 rows are fixed by the first 0007A1C fork and retained disassembly:

| Row | Fork-1 ratio | Retained code-shape fact |
|---|---:|---|
| `P-SCALAR-GENERAL` | `9.602663980x` | Scalar FLOAT32 segment loads reconstruct native-order layout. |
| `A-GENERAL` | `28.418421710x` | BFLOAT16 segment load reconstructs native-order layout. |
| `M-PAD` | `44.118935579x` | BFLOAT16 segment load reconstructs native-order layout. |
| `M-CONCAT` | `299.529062690x` | Repeated INT32 segment loads reconstruct native-order layout. |
| `M-UNFOLD-AXIS` | `45.015260656x` | FLOAT32 segment load reconstructs native-order layout. |
| `M-UNFOLD2D` | `55.387577809x` | FLOAT64 segment store reconstructs native-order layout. |
| `I-GATHER` | `210.847850287x` | FLOAT64 segment load/store reconstruct native-order layouts. |
| `I-GATHER-ND` | `80.696243403x` | FLOAT32 segment store reconstructs native-order layout. |
| `S-GENERAL-MIN` | `185.427959907x` | INT64 segment load/store reconstruct native-order layouts. |
| `C-SCAN-GENERAL` | `81.940741374x` | INT64 segment load/store reconstruct native-order layouts. |
| `X-MIN-MULTI` | `45.882524599x` | BFLOAT16 segment load reconstructs native-order layout. |
| `N-MEAN-GENERAL` | `6.013050288x` | FLOAT32 segment load reconstructs native-order layout. |
| `N-PROD-MULTI` | `1.735452755x` | BFLOAT16 segment store reconstructs native-order layout. |

The three already passing rows remain regression controls. `M-STACK`, `F-FOLD2D`,
`R-DROPOUT-GENERAL`, and `X-ANY-SINGLE` are deliberately excluded from this correction because
their retained Class-File shapes do not contain the same native-order-layout construction defect.

## Out of scope

- Changing or refreezing the CPU 0007A1C probe, ledger, cases, direct comparators, threshold,
  algorithms, work sizes, fork count, sampling, or evidence directory.
- Correcting any residual row owned by CPU 0007A1E through CPU 0007A1G.
- Family-specific mapping, fold, dropout, scan, aggregate, exact-sum/product, scalar-formula, or
  address-geometry redesign.
- Preparation/finalization behavior, new preparation/finalization coverage, unrelated assertions
  in the two schema-propagation test owners, or any other test owner.
- Changes to IR, lowering, resources, scratch, materialization, routes, capability, semantics,
  public API, Prepare, Runtime, dependencies, architecture, build, other modules, or later work.
- Production benchmark tooling, tuning, caches other than schema compatibility, JMH, or a
  universal performance claim.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [`performance evidence and tuning`](../../../../architecture/performance-evidence-and-tuning.md)
- [`runtime, Prepare, and backend boundary`](../../../../architecture/runtime-prepare-backend-boundary.md)
- [`planning guide`](../../../planning-guide.md)
- [`documentation rules`](../../../../developer-guide/documentation-rules.md)
- [`CPU backend guide`](../../../../backend-guide/cpu-backend.md)
- [`CPU master plan`](../master-plan.md)
- [`CPU 0007A1C`](0007a1c-generated-direct-evidence-closure.md)

## Architecture constraints

- CPU finalization owns deterministic generated code; Runtime executes only cold-bound immutable
  prepared work.
- Generated classes remain direct, typed, allocation-free in normal hot work, field-free, and
  free of mutable static run state and Synaptik-owned scalar hot calls.
- Benchmark evidence accepts, rejects, or diagnoses this fixed change. It never selects or mutates
  production settings.
- No Model semantics, backend ownership, lifecycle boundary, resource declaration, or dependency
  direction changes.

## Package impact

Existing package used:

- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` — shared direct carrier emission,
  optional entry setup, and structural tests.
- `io.github.pho001.synaptik.backend.cpu.internal.cache` — current-only generated compatibility
  schema and package documentation after emitted bytes change.

No package, export, supported public type, or module is added or moved.

Type placement:

- `CpuCarrierEmitter` remains the sole owner of typed array/segment load and store bytecode.
- `CpuClassFileKernelGenerator` may coordinate one-time entry setup only if the carrier emitter
  cannot do so without emitting setup inside a loop.

## Affected files

Expected:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuCarrierEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratorSchema.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/package-info.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuGeneratedDirectEvidenceClosureTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratedKernelArtifactStoreTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecializationTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizerTest.java`
- this task, CPU 0007A1C, CPU master plan, roadmap, and CPU backend guide if the completed result
  adds a useful bounded implementation/evidence statement.

Optional second production owner:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGenerator.java`

## Maximum scope

At most 15 repository paths: two code-shaping owners, schema, two package summaries, five tests,
and five documentation/planning paths. The preparation/finalization tests are permitted only as
schema-propagation owners for their existing version expectations. If a third code-shaping owner,
a family emitter, a sixth test owner, any other change in either schema-propagation test, or any
architecture/resource/lowering change is required, stop and replan.

## Acceptance criteria

- Frozen semantic verification still passes all 20 cases exactly.
- Every one of the 13 target rows passes every new isolated fork and its median of fork medians at
  `<= 1.15x` under the unchanged CPU 0007A1C protocol.
- The three original passing rows remain `<= 1.15x` in every new fork and aggregate.
- A target generated method constructs each needed native-order typed layout no more than once per
  invocation and never inside its repeated scalar body. Stable Class-File tests enforce this
  without brittle pool indices, bytecode offsets, or generated names.
- Each representative remains field-free with exactly one typed static entry and no constructor,
  bridge, generic `Object`, method handle/type constant, `invokedynamic`, dynamic constant,
  bootstrap, reflection, collection/boxing, or forbidden Synaptik-owned reference.
- Emitted-byte changes advance schema exactly once from 31 to 32; schema-31 artifacts are safe
  incompatible misses and no migration/alias/dual-schema path exists.
- `CpuPartitionPreparerTest` and `CpuPartitionFinalizerTest` may change only their existing schema
  expectation from 31 to 32; their preparation/finalization behavior and every unrelated test
  remain unchanged.
- Only the two permitted code-shaping owners change; no family algorithm, semantics, resources,
  routes, capability, public API, Prepare/Runtime behavior, architecture, dependency, or build
  contract changes.
- A separate clean documentation context finalizes Javadocs/package summaries, bounded guide
  evidence if useful, glossary impact, task evidence/status, CPU 0007A1C follow-up state, master
  plan, and roadmap.

## Tests / validation

Reuse the frozen semantic and fork-1 evidence as the pre-edit failure. After Java stabilizes, run:

```bash
./gradlew :backends:cpu:test \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneratedDirectEvidenceClosureTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuPointwiseGeneratedKernelTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuAffineCopyGeneratedKernelTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuDataMovementGeneratedKernelTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuIndexingGeneratedKernelTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuScatterGeneratedKernelTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuScanGeneratedKernelTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuAggregateGeneratedKernelTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratedKernelArtifactStoreTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecializationTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizerTest
./gradlew :backends:cpu:test
```

Recompile the unchanged frozen source against the stabilized production classes, verify all 20
semantics, and run five new isolated `-Xms1g -Xmx1g` forks. Retain raw results, per-fork summaries,
the 13-row aggregate, regenerated class bytes, full `javap -c -p`/`javap -v -p`, semantic
Class-File reports, member inventories, environment, commands, and SHA-256 manifests in a new
separate sibling directory created as `/private/tmp/synaptik-cpu-0007a1d-XXXXXXXX`; do not add,
remove, or modify anything in the frozen 0007A1C corpus. A nonzero full-probe exit is acceptable
only when the output proves all 16 target/control rows pass and failures are confined exactly to
the four deferred rows; do not describe that full process as passed.

The documentation context reuses stabilized Java/timing evidence and runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
git diff --cached --check
git status --short
```

It also checks Markdown links/anchors/fences/newlines/whitespace, exact path/owner/schema scope,
empty staged diff, preserved dirty paths, one Ready frontier, and CPU 0007A2 still blocked.
Repository-wide, architecture, conformance, and integration suites remain deferred to CPU 0009 or
CI because no shared contract changes.

## Dependencies

- CPU 0007A1C is Incomplete with a retained frozen 20-case corpus and first-fork failure.
- CPU 0007A0–0007A1B provide the completed generated-family and schema-31 baseline.
- Current Java 26 Class-File and foreign-memory APIs define the emitted bytecode shape.

## Follow-up tasks

- CPU 0007A1E: Ready movement general-address-loop parity correction for `M-PAD`, `M-CONCAT`,
  `M-UNFOLD-AXIS`, and `M-UNFOLD2D` through the single `CpuDataMovementEmitter` owner.
- CPU 0007A1F: Draft residual BOOL movement/aggregate correction for `M-STACK` and
  `X-ANY-SINGLE`; create its detailed specification only after CPU 0007A1E completes.
- CPU 0007A1G: Draft fold/dropout residual correction for `F-FOLD2D` and
  `R-DROPOUT-GENERAL`; create its detailed specification only after CPU 0007A1F completes.
- The other nine failed A1D targets remain unassigned Draft residual work until the movement
  correction supplies new frozen evidence for the next strongest bounded owner cluster.
- CPU 0007A1C resumes its unchanged full closure protocol only after every residual group closes.
- CPU 0007A2 remains blocked until CPU 0007A1C is Complete.

## Architecture impact

Expected impact: None. Stop if the correction requires an architecture, dependency, public API,
semantic, resource, route, Prepare, Runtime, or build change.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
You are the clean implementation agent for Synaptik CPU task 0007A1D. Work on the existing dirty
worktree without committing, pushing, staging, reverting, deleting, or modifying unrelated work.
Do not use a GSD skill/workflow.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, CPU task 0007A1D and its linked
CPU 0007A1C frozen evidence contract in full. Implement exactly the bounded native-order segment
layout-hoisting correction. Preserve the frozen probe/comparators and stop before a third
code-shaping owner or any family/architecture/resource/lifecycle change. When emitted bytes move
schema 31 to 32, change only the existing schema-version expectations in
`CpuPartitionPreparerTest` and `CpuPartitionFinalizerTest`; do not change their behavior or any
other assertion.

Run the specified focused/full CPU, unchanged frozen semantic/Class-File/five-fork, schema, and
scope gates. Hand the stabilized diff and exact evidence to a distinct clean documentation
context for Javadocs/package summaries, guide/glossary/planning review, and documentation gates.
Do not mark Complete until all 13 target and three control rows pass every required gate.
```

## Local decisions

- Fix the shared causal code shape before family-specific residual rows.
- The first task owns 13 rows because retained `javap` shows the same repeated native-order typed
  layout construction in each generated path; their operation families remain unchanged.
- The four rows without that shape are not used to justify this production change.
- The frozen full probe remains immutable even though its expected process exit may stay nonzero
  until later correction groups complete.
- The two preparation/finalization tests are schema-propagation owners only; admitting their
  existing version assertions does not expand the task's production owners or lifecycle behavior.

## Known limitations

- The evidence is specific to the frozen JVM, machine, cases, and protocol and grants no universal
  parity or tuning claim.
- This task does not close CPU 0007A1C or authorize CPU 0007A2.
- Residual full-probe failure is permitted only for the four named later rows and must be recorded
  as an expected incomplete closure result, not a passing full benchmark.
- The final required fork failed all 13 A1D targets. The task therefore did not reach its
  five-fork or aggregate acceptance gates and remains incomplete despite stable semantic,
  structural, and Java-test evidence.
- A diagnostic generated version that eliminated `ByteOrder.nativeOrder()` and
  `ValueLayout.withOrder(...)` entirely still left ten A1D targets above `1.15x`. Layout
  construction was one partial cause, not a sufficient explanation of the remaining ratios.

## Validation evidence

Planning/documentation context `/root` verified the frozen sources, ledger, and all 20 generated
class checksums; exact semantics `VERIFIED,20`; the raw fork-1 ratios; current source schema 31;
and no 0007A1C production-owner diff. Retained `javap` identifies the 13 target classes above with
native-order typed layout construction and the four residual classes without that same shape.
`schema-after.txt` is malformed and is not accepted as evidence; current source supplies the
independent schema-31 baseline. No executable command was rerun while creating this specification.

At that earlier planning-only checkpoint, the scope repair admitted the two existing authoritative
preparation/finalization schema assertions and raised the maximum from 13 to 15 paths without
changing the production-owner limit, frozen corpus, semantics, performance gates, or then-`Ready`
status. The CPU master plan and roadmap then described CPU 0007A1D by its 13 frozen rows and
two-owner correction; neither described its test-owner budget, so neither required an edit at that
checkpoint. That context changed only this already-untracked task path. All eight local Markdown
targets existed, there were no anchor-fragment links, all six fences were balanced, the file ended
with a newline, and it had no trailing whitespace. Scope searches confirmed the stale
13-path/three-test wording was gone, both schema assertions expected source schema 31, A1D was the
only detailed `Ready` CPU task at that time, and CPU 0007A2 remained blocked. `git diff --check` and
`git diff --cached --check` pass, and the staged diff is empty. `git status --short` was inspected;
all unrelated dirty paths were left untouched, including additional concurrent paths that appeared
during this planning pass.

The implementation context `01a01dcf-2ad5-75c1-9e60-94cb4c69a659` then changed only the two
permitted code-shaping owners, advanced current-only generated compatibility to schema 32, and
updated the five permitted test owners. The exact focused matrix passed 119 tests. The final
authoritative CPU suite passed 54 suites and 345 tests with zero failures, zero errors, and one
existing opt-in skip. No executable Java changed after those results.

The retained A1D evidence directory is
`/private/tmp/synaptik-cpu-0007a1d-HWoXtXQQ`. Frozen-source and generated-class SHA-256 manifests
verify. Frozen semantic verification reports `VERIFIED,20`. All 20 retained generated classes
remain field-free with exactly one static entry, and the final `javap` output places at most one
required typed scalar segment-layout construction before the first scalar segment access. Current
source independently reports schema 32; schema-31 artifacts are incompatible safe misses.

The final required isolated fork produced the following generated/direct ratios for the 13 A1D
targets; every value exceeds the unchanged `1.15x` gate:

| Target | Ratio | Target | Ratio |
|---|---:|---|---:|
| `P-SCALAR-GENERAL` | `2.608760079x` | `A-GENERAL` | `4.656029094x` |
| `M-PAD` | `7.743893186x` | `M-CONCAT` | `60.725558334x` |
| `M-UNFOLD-AXIS` | `8.525273772x` | `M-UNFOLD2D` | `11.226516498x` |
| `I-GATHER` | `33.120494786x` | `I-GATHER-ND` | `14.529261309x` |
| `S-GENERAL-MIN` | `23.316236605x` | `C-SCAN-GENERAL` | `9.511804623x` |
| `X-MIN-MULTI` | `9.045297757x` | `N-MEAN-GENERAL` | `1.832304357x` |
| `N-PROD-MULTI` | `1.298619443x` |  |  |

The three controls passed: `P-VECTOR-SEGMENT` at `0.988031893x`, `P-INTEGRAL-MIXED` at
`0.316327620x`, and `O-ARGSORT` at `0.888401179x`. The four previously deferred rows also remain
above the gate: `M-STACK` at `69.508672343x`, `F-FOLD2D` at `1.155424701x`,
`R-DROPOUT-GENERAL` at `1.150352245x`, and `X-ANY-SINGLE` at `1.208797167x`. The process failed
with 17 ratio failures. Forks two through five correctly did not run, so no aggregate exists.

The diagnostic direct-static experiment removed native-order and with-order construction from the
generated version without changing the frozen comparator. It improved three A1D targets to at
most `1.15x`, but ten remained above the gate. This diagnostic is causal evidence only; it is not
an accepted production version or performance pass.

Mandatory clean documentation/planning context `/root` independently reviewed the final diff,
source/Javadocs/tests, retained sources and hashes, semantic result, raw and summarized fork,
diagnostic experiment, generated classes, complete `javap` output, member inventory, CPU guide,
and glossary. General and Planning profiles are primary; API/Javadoc and Backend Guide profiles
govern the targeted no-change review. The existing `CpuCarrierEmitter`, schema, cache-package,
and emitter-package Javadocs accurately describe the stable schema-32 invocation-local behavior
without claiming performance parity, so no Java documentation edit is required. The CPU guide
receives no incomplete local performance claim, and the glossary receives no non-reusable
implementation term.

## Implementation notes

- `CpuClassFileKernelGenerator` invokes `CpuCarrierEmitter.prepareSegmentLayouts(...)` once at
  generated entry setup. `CpuCarrierEmitter` reserves a fixed typed-layout local block, initializes
  only types used by segment boundaries, and loads those locals for repeated scalar access.
- Array access, vector access, family semantics, carrier/address geometry, resources, and the
  frozen probe/comparators remain unchanged. Schema 32 is a current-only compatibility boundary.
- Stable Class-File assertions cover actual carrier-emitter consumers and place typed layout
  construction before the first scalar segment access.
- The retained final fork rejects the correction as a complete parity fix. No threshold,
  comparator, result, or acceptance criterion is weakened or reinterpreted.
- Final generated/direct disassembly and direct comparators show that `M-PAD`, `M-CONCAT`,
  `M-UNFOLD-AXIS`, and `M-UNFOLD2D` share `CpuDataMovementEmitter`'s general long-address/output-
  odometer owner. That single-owner cluster is the next bounded correction task.

## Completion summary

- Completed changes: hoisted required native-order typed scalar segment layouts into invocation
  locals, advanced schema 31 to 32, propagated compatibility expectations, and added stable
  cross-family Class-File assertions.
- Files changed or created: two permitted code-shaping owners, schema and two package summaries,
  five permitted tests, plus the synchronized planning paths in this documentation pass.
- Tests and validation: focused matrix 119/119 passed; final CPU suite 345 tests passed with zero
  failures/errors and one skip; frozen semantics passed 20/20; all three controls passed.
- Documentation-agent review: clean context `/root`; affected Javadocs/package summaries remain
  accurate without further edit, and the CPU guide/glossary receive reasoned no-change results.
- Unresolved issues: every one of the 13 required A1D performance targets failed fork 1; forks
  two through five and aggregate evaluation do not exist.
- Follow-up required: CPU 0007A1E is now Complete. Execute Ready CPU 0007A1F, then later residual
  groups in strict order before resuming CPU 0007A1C. CPU 0007A2 remains blocked.

Status: Incomplete
Follow-up required: CPU 0007A1F and later ordered residual corrections must close the remaining frozen performance failures before CPU 0007A1C or CPU 0007A2 can advance.
