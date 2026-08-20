# Task 0007A1F: BOOL Movement and Aggregate Residual Parity

## Status

Complete

## Goal

Correct the next bounded residual group in the frozen CPU 0007A1C comparison: `M-STACK` and
`X-ANY-SINGLE`. Preserve their exact canonical-BOOL semantics while making each generated entry
meet the unchanged generated/direct `<= 1.15x` gate in every one of five isolated forks.

`M-STACK` is the remaining movement row after CPU 0007A1E. `X-ANY-SINGLE` is the smallest remaining
aggregate row and exercises a single selected axis over a zero-stride canonical-BOOL input. The
task may change no more than their two existing code-shaping owners.

## Scope

- Use the unchanged frozen A1C sources under `/private/tmp/synaptik-cpu-0007a1c-oMAmuVhF` and the
  completed schema-33 A1E evidence under `/private/tmp/synaptik-cpu-0007a1e-1LiNbBUz`.
- Correct `M-STACK` only through `CpuDataMovementEmitter`. Preserve inserted-axis occurrence
  selection, ordered occurrence map `[0, 1, 0, 2]`, repeated input reuse, mixed BOOL carriers,
  arbitrary legal half-open ranges, and represented canonical bytes.
- Correct `X-ANY-SINGLE` only through `CpuAggregateEmitter`. Preserve its selected-axis membership,
  logical row-major visitation, canonical false identity, canonical true/false result, zero-stride
  input semantics, output-cell ownership, and zero-workspace resource contract.
- Cold-prove and retain only the primitive loop, occurrence, address, or selected-domain facts
  needed by the two fixed code shapes. Retain typed general-long fallbacks whenever geometry,
  ranges, or transitions cannot be proved safe.
- Keep generated target classes direct, typed, field-free, allocation-free in repeated work, and
  free of hidden Synaptik runtime calls.
- Add stable Class-File assertions for both corrected shapes without fixed bytecode offsets,
  constant-pool indexes, generated names, or JIT assembly.
- Advance the current-only generated schema exactly once if emitted bytes change. Older artifacts
  must remain incompatible safe misses without migration, aliasing, conversion, or dual-schema
  reuse.
- Run the unchanged full 20-row process. Accept only the two targets and the three established
  controls; every other row remains diagnostic evidence for later ordered work.

### Frozen target evidence

| Row | Fork 1 | Fork 2 | Fork 3 | Fork 4 | Fork 5 | Worst fork | Existing owner |
|---|---:|---:|---:|---:|---:|---:|---|
| `M-STACK` | `72.565477178x` | `72.793000000x` | `72.044510386x` | `76.856816183x` | `61.540028090x` | `76.856816183x` | `CpuDataMovementEmitter` |
| `X-ANY-SINGLE` | `1.209398307x` | `1.207831644x` | `1.214668151x` | `1.202673602x` | `1.212934559x` | `1.214668151x` | `CpuAggregateEmitter` |

These ratios are diagnostic baselines from A1E, not accepted results. `M-STACK` uses Shape
`[4, 64, 64, 16]`, canonical BOOL, mixed `byte[]`/`MemorySegment` inputs, a `byte[]` output, and
four semantic occurrences backed by three unique inputs. `X-ANY-SINGLE` reduces axis 1 of Shape
`[1024, 256]`; its physical input has strides `[1, 0]`, so each output cell reads one repeated
canonical BOOL value 256 times. The direct comparator preserves that repeated logical fold.

## Out of scope

- Changing or refreezing the A1C probe, cases, comparators, dimensions, threshold, warmup, sample
  order, fork count, or algorithm.
- Correcting PAD, CONCAT, UNFOLD_AXIS, or UNFOLD2D again; CPU 0007A1E is complete.
- Correcting `F-FOLD2D`, `R-DROPOUT-GENERAL`, or any other deferred diagnostic row.
- Changing BOOL semantics, permitting noncanonical bytes, short-circuiting the required logical
  fold in a way that changes access/failure behavior, changing output layout, or changing resources.
- New IR/lowering/resource contracts, capability changes, route selection, public API, shared
  Prepare/Runtime behavior, architecture, dependencies, build configuration, generated fields,
  helper methods, generic carriers, mutable static state, tuning, or a universal performance claim.

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
- [`CPU 0007A1C`](0007a1c-generated-direct-evidence-closure.md)
- [`CPU 0007A1D`](0007a1d-native-order-segment-layout-hoisting.md)
- [`CPU 0007A1E`](0007a1e-movement-general-address-loop-parity.md)

## Architecture constraints

- Model owns STACK and ANY semantics; this task changes only CPU-private generated realization and
  evidence.
- CPU analysis retains lowering, route selection, and exact declarations before shared assignment;
  CPU finalization realizes generated bytes afterward; Runtime only executes the prepared result.
- Runtime hot work receives no `Operation`, `CompiledNode`, Shape, route selection, or generated-
  code decision.
- Performance evidence accepts or rejects fixed code shapes and never selects production settings.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` — existing movement and aggregate
  emitters, Class-File tests, and package documentation.
- `io.github.pho001.synaptik.backend.cpu.internal.cache` — current-only schema and cache package
  documentation if emitted bytes change.

Packages added or changed:

- None.

Type placement:

- `CpuDataMovementEmitter` remains the sole STACK code-shaping owner.
- `CpuAggregateEmitter` remains the sole ANY code-shaping owner.
- No third production owner is authorized.

## Affected files

Expected production and contract paths:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuDataMovementEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuAggregateEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratorSchema.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/package-info.java`

Expected tests and schema propagation:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuDataMovementGeneratedKernelTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuAggregateGeneratedKernelTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuGeneratedDirectEvidenceClosureTest.java`
- the existing four schema-assertion owners changed by CPU 0007A1E, only if the schema advances.

Planning paths are this task, CPU 0007A1C, CPU master plan, and roadmap. CPU 0007A1D/E historical
evidence changes only if a direct status cross-reference becomes inaccurate. The CPU guide and
glossary are review-only unless a current reusable contract changes.

## Maximum scope

This task may modify at most 20 repository paths:

- 2 production algorithm-shaping owners;
- 3 schema/package-contract paths;
- 7 test paths: two target Class-File tests, the frozen closure test, and four existing schema
  assertion owners; and
- 8 documentation/planning paths: this task, A1C, optional A1D/E cross-reference corrections,
  CPU master plan, roadmap, and guide/glossary only when current contract text changes.

If a third production owner, IR/lowering/resource change, another module, another benchmark case,
or an architecture decision is needed, stop and report the conflict.

## Acceptance criteria

- Frozen semantic verification remains exactly `VERIFIED,20`.
- `M-STACK` and `X-ANY-SINGLE` each pass every one of five new isolated forks and the median of
  fork medians at `<= 1.15x` under the unchanged A1C protocol.
- `P-VECTOR-SEGMENT`, `P-INTEGRAL-MIXED`, and `O-ARGSORT` pass every new fork and aggregate.
- Exact target outputs, untouched inputs, canaries, scratch/state invariants, logical visitation,
  mapping, and comparator behavior remain unchanged.
- Stable Class-File assertions prove the selected primitive hot form and the retained typed
  general-long fallback where applicable.
- Every target class has exactly one typed static entry, no fields or constructors, and no bridge,
  generic `Object`, method-handle/type constant, `invokedynamic`, dynamic constant, bootstrap,
  reflection, collection/boxing, allocation, or Synaptik-owned runtime reference.
- If emitted bytes change, the schema advances exactly once from 33 and every prior-schema
  envelope is an incompatible safe miss.
- Only the two named emitters change generated algorithm shape. Semantics, resources, routes,
  capability, public API, Prepare/Runtime behavior, architecture, dependencies, and build remain
  unchanged.
- A distinct clean documentation context independently finalizes Javadocs/package summaries,
  planning evidence/status, guide and glossary impact, and documentation validation.

## Tests / validation

Run the focused owners and the one authoritative module suite after executable Java stabilizes:

```bash
./gradlew :backends:cpu:test \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneratedDirectEvidenceClosureTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuDataMovementGeneratedKernelTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuAggregateGeneratedKernelTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratedKernelArtifactStoreTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecializationTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizerTest
./gradlew :backends:cpu:test
```

Recompile the unchanged A1C sources, run exact 20-row semantics, and run five isolated
`-Xms1g -Xmx1g` forks. Retain raw/summarized results, two-target/control aggregates, all generated
classes, complete `javap -c -p` and `javap -v -p`, semantic/member/forbidden-reference reports,
environment, commands, and SHA-256 manifests in one new A1F evidence directory. A nonzero full-
probe exit is acceptable only for explicitly deferred rows and must not be described as a full
performance pass.

The documentation context reuses stable Java/timing evidence and runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
git diff --cached --check
git status --short
```

It also validates Markdown links/anchors/fences/newlines, terminology/glossary, exact owners and
schema propagation, immutable A1C sources, empty staged diff, one detailed Ready task, strict
correction order, and worktree preservation. Repository-wide, architecture, conformance, and
integration suites remain deferred to CPU 0009 or CI because no shared contract may change.

## Dependencies

- CPU 0007A1C supplies the immutable 20-row corpus and remains Incomplete.
- CPU 0007A1D supplies stable invocation-local segment layouts.
- CPU 0007A1E is Complete at schema 33 and supplies the current five-fork residual evidence.
- Completed CPU 0006A and CPU 0007A define STACK and ANY semantics/resources to preserve.

## Follow-up tasks

- [CPU 0007A1G](0007a1g-fold-and-dropout-residual-parity.md) is the sole detailed `Ready` task for
  `F-FOLD2D` and `R-DROPOUT-GENERAL`.
- Other residual target groups remain Draft until the ordered evidence supports a bounded owner
  cluster.
- CPU 0007A1C resumes its unchanged closure only after all residual corrections complete.
- CPU 0007A2 remains blocked until CPU 0007A1C is Complete.

## Architecture impact

Expected impact: None.

If implementation requires an architecture, dependency, public API, semantic, resource, route,
Prepare, Runtime, build, IR, or lowering change, stop and report the issue.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
You are the clean implementation agent for Synaptik CPU task 0007A1F. Work on the existing dirty
worktree without committing, pushing, staging, resetting, reverting, deleting, or modifying
unrelated work. Do not use a GSD skill or workflow.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, CPU task 0007A1F, CPU tasks
0007A1C–0007A1E, and the retained A1C/A1E evidence in full. Implement exactly the bounded two-owner
BOOL movement/aggregate correction. Preserve the frozen sources, STACK/ANY semantics and resources,
all comparators, and typed fallbacks. Stop before a third production owner or any IR/lowering/
resource/architecture/lifecycle change.

Run the specified focused/full CPU, frozen semantic/Class-File/five-fork, schema, scope, and
preservation gates. Hand the stabilized diff and exact evidence to a distinct clean documentation
context. Do not mark Complete until both targets and all three controls pass every required gate.
Do not commit or push.
```

## Local decisions

- A1E evidence confirms both rows fail consistently across five isolated forks, so neither is a
  one-fork anomaly.
- The rows use exactly two existing code-shaping owners, matching A1C's maximum owner boundary.
- STACK remains separate from A1E because its inserted-axis occurrence selection and extreme BOOL
  ratio need an independent hypothesis. ANY remains in this task because its small, consistent
  overage and zero-stride selected domain are bounded to the aggregate emitter.
- The task preserves repeated logical ANY visits; a semantic shortcut is not assumed safe merely
  because the physical address repeats.

## Known limitations

- Ratios apply only to the frozen cases, JVM, host, and protocol; they grant no universal parity,
  tuning, or routing claim.
- This task does not close CPU 0007A1C or authorize CPU 0007A2.
- Deferred rows may keep the full process nonzero and must remain explicitly diagnostic.

## Validation evidence

Implementation evidence is retained at
`/private/tmp/synaptik-cpu-0007a1f-OsajC4ko`. Implementation context
`01a01e62-9d2d-7ff1-a02c-3c3fc0bd664b` ran the exact seven-owner focused Gradle command in this
task and the authoritative uncached `./gradlew :backends:cpu:test --rerun-tasks`; both completed
with `BUILD SUCCESSFUL`. No executable Java changed afterward. The unchanged A1C sources compiled
against final production classes, their before/after SHA-256 manifests match, and exact semantic
verification reports `VERIFIED,20`.

All five isolated forks passed for every required target and control:

| Row | Fork 1 | Fork 2 | Fork 3 | Fork 4 | Fork 5 | Median of fork medians | Worst fork |
|---|---:|---:|---:|---:|---:|---:|---:|
| `M-STACK` | `0.933040078x` | `0.878074544x` | `0.869558736x` | `0.932181070x` | `0.986374612x` | `0.932181070x` | `0.986374612x` |
| `X-ANY-SINGLE` | `0.307561645x` | `0.306641529x` | `0.304177560x` | `0.306305081x` | `0.307136502x` | `0.306641529x` | `0.307561645x` |
| `P-VECTOR-SEGMENT` | `0.973187118x` | `0.993918738x` | `0.968807230x` | `0.961917957x` | `0.978440665x` | `0.973187118x` | `0.993918738x` |
| `P-INTEGRAL-MIXED` | `0.317426099x` | `0.317541461x` | `0.313788337x` | `0.316096824x` | `0.314939159x` | `0.316096824x` | `0.317541461x` |
| `O-ARGSORT` | `0.849541316x` | `0.868323844x` | `0.862000991x` | `0.860249631x` | `0.856880990x` | `0.860249631x` | `0.868323844x` |

Every full 20-row process exited nonzero because ten to twelve explicitly deferred rows exceeded
`1.15x`; no required row failed. This is not a full 20-row performance pass.

The retained Class-File assessment reports one typed static entry and zero fields for each target.
The proved `M-STACK` branch emits the frozen occurrence order with direct `byte[]` and
`MemorySegment` reads and primitive byte stores. Its sole `NEW` instruction belongs to the retained
general-long fallback's exceptional failure path. The proved `X-ANY-SINGLE` branch performs every
selected-domain `MemorySegment.get`, combines with `IOR`, has no accumulator early exit, and
canonicalizes once at output. Neither proved branch contains allocation, boxing, reflection,
`invokedynamic`, or a Synaptik-owned runtime reference. Both retain typed general-long fallbacks.

Exactly nine implementation-owned CPU paths changed: the two emitters, schema, two target
Class-File tests, and four existing schema-assertion tests. Schema advanced exactly once from 33
to 34; older envelopes are incompatible safe misses with no migration or dual-schema path.
Generated classes, complete `javap -c -p` and `javap -v -p`, semantic and forbidden-reference
reports, summaries, raw fork output, environment, commands, and SHA-256 manifests are retained in
the evidence directory.

Clean documentation context `01a01e83-bb41-7513-8b8a-9a17a0c15c3d` independently reviewed the
implementation, tests, retained evidence, affected emitter/schema/package Javadocs, CPU guide,
glossary, A1C–A1F, master plan, and roadmap. It changed no executable behavior and did not rerun
successful Java tests, frozen semantics, or performance forks. It finalized schema-34 package
documentation, the bounded current guide/glossary descriptions, this evidence/status record, and
the sole detailed Ready A1G plan. Final documentation validation is recorded below.

`./gradlew :backends:cpu:javadoc` completed with `BUILD SUCCESSFUL` and only the two expected
incubating-Vector warnings. Local Markdown targets and balanced fences passed for all seven
changed Markdown documents; the new A1G links introduce no fragment anchors. Final-newline,
whitespace, schema-34/current-text, exact evidence-path, frozen-hash, status synchronization,
sole-Ready, A2-blocking, and 19-path combined-scope checks passed. `git diff --check` and
`git diff --cached --check` passed, and the staged diff is empty. `git status --short` was
inspected; only the A1F implementation and focused documentation/planning paths remain dirty.

## Implementation notes

- `CpuDataMovementEmitter` cold-proves the frozen axis-zero canonical-BOOL STACK geometry before
  using one direct primitive loop per semantic occurrence. Repeated mapping position `0` remains a
  repeated copy; mixed input carriers and byte-array output are embedded in the generated entry.
- `CpuAggregateEmitter` cold-proves the rank-two canonical-BOOL ANY geometry before using direct
  primitive output-cell/domain loops. It deliberately preserves all 256 logical reads and applies
  canonicalization only after `IOR` accumulation.
- Geometry outside either narrow proof continues through the existing typed general-long body.
- Schema 34 is current-only. Schema-33 and earlier artifacts are incompatible safe misses.

## Completion summary

- Completed changes: added the two narrow cold-proved direct canonical-BOOL forms with retained
  typed fallbacks, advanced generated compatibility to schema 34, added structural tests, and
  finalized Javadocs, guide, glossary, evidence, and planning state.
- Files changed or created: nine implementation-owned CPU paths; three affected package-summary
  Javadocs; CPU backend guide; glossary; A1F, A1C follow-up record, CPU master plan, roadmap; and
  the new A1G task specification.
- Tests and validation: reused the successful focused and authoritative CPU test evidence;
  frozen semantics passed `VERIFIED,20`; all 25 required fork ratios and five aggregates passed;
  target Class-File, forbidden-reference, schema, frozen-source, and exact-scope gates passed.
  The separate documentation checks and CPU Javadoc passed as recorded above.
- Documentation-agent review: clean context `01a01e83-bb41-7513-8b8a-9a17a0c15c3d` finalized
  affected Javadocs/package summaries, bounded guide/glossary text, task evidence/status, master
  plan, roadmap, and A1G without executable changes or duplicate Java/performance runs.
- Documentation impact: the guide and glossary now describe schema-34 STACK/ANY forms as narrow
  cold-proved paths with typed fallbacks and bounded retained evidence, not universal support or
  performance claims.
- Javadoc review: emitter/schema type documentation was accurate; package summaries required the
  schema-34 and narrow-form updates. Prepare package documentation required only the current
  artifact-version correction.
- Unresolved issues: A1C and A1D remain `Review needed`; deferred rows keep the full probe nonzero.
- Follow-up required: implement sole Ready CPU 0007A1G, then plan later residual owner clusters in
  evidence-backed order. CPU 0007A2 remains blocked until A1C closes.

Status: Complete
