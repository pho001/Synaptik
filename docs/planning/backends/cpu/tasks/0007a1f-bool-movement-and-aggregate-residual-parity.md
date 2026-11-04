# Task 0007A1F: [BOOL](../../../../glossary.md#data-type--datatype) Movement and Aggregate [Residual](../../../../glossary.md#scan-residual-port-and-history-demand) Parity

## Status

Complete

## Goal

Correct the next bounded [residual](../../../../glossary.md#scan-residual-port-and-history-demand) group in the frozen [CPU](../../../../glossary.md#term-cpu) 0007A1C comparison: `M-STACK` and
`X-ANY-SINGLE`. Preserve their exact [canonical](../../../../glossary.md#term-canonical-form)-[BOOL](../../../../glossary.md#data-type--datatype) semantics while making each generated entry
meet the unchanged generated/direct `<= 1.15x` gate in every one of five isolated [forks](../../../../glossary.md#term-fork-meanings).

`M-STACK` is the remaining movement row after CPU 0007A1E. `X-ANY-SINGLE` is the smallest remaining
aggregate row and exercises a single selected [axis](../../../../glossary.md#term-axis) over a zero-[stride](../../../../glossary.md#term-stride) canonical-BOOL input. The
task may change no more than their two existing code-shaping [owners](../../../../glossary.md#term-owner).

## [Scope](../../../../glossary.md#term-scope)

- Use the unchanged frozen A1C sources under `/private/tmp/synaptik-cpu-0007a1c-oMAmuVhF` and the
  completed schema-33 A1E evidence under <span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema))</span>`/private/tmp/synaptik-cpu-0007a1e-1LiNbBUz`.
- Correct `M-STACK` only through `CpuDataMovementEmitter`. Preserve inserted-axis occurrence
  selection, ordered occurrence map <span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>`[0, 1, 0, 2]`, repeated input reuse, mixed BOOL carriers,
  arbitrary legal half-open ranges, and represented canonical bytes.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [half-open glossary](../../../../glossary.md#term-half-open-interval); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- Correct `X-ANY-SINGLE` only through `CpuAggregateEmitter`. Preserve its selected-axis membership,
  logical row-major visitation, canonical false identity, canonical true/false result, zero-stride
  input semantics, output-cell ownership, and zero-workspace resource contract.<span data-glossary-navigation="true"> ([row-major glossary](../../../../glossary.md#term-row-major); [identity glossary](../../../../glossary.md#term-identity-meanings); [stride glossary](../../../../glossary.md#term-stride); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Cold-prove and retain only the primitive loop, occurrence, address, or selected-domain facts
  needed by the two fixed code shapes. Retain typed general-long fallbacks whenever geometry,
  ranges, or transitions cannot be proved safe.<span data-glossary-navigation="true"> ([shapes glossary](../../../../glossary.md#shape))</span>
- Keep generated target classes direct, typed, field-free, allocation-free in repeated work, and
  free of hidden Synaptik runtime calls.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Add stable Class-File assertions for both corrected shapes without fixed bytecode offsets,
  constant-pool indexes, generated names, or JIT assembly.<span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode); [constant glossary](../../../../glossary.md#term-constant); [JIT glossary](../../../../glossary.md#term-just-in-time-compilation-jit))</span>
- Advance the current-only generated schema exactly once if emitted bytes change. Older artifacts
  must remain incompatible safe misses without migration, aliasing, conversion, or dual-schema
  reuse.<span data-glossary-navigation="true"> ([artifacts glossary](../../../../glossary.md#term-artifact); [aliasing glossary](../../../../glossary.md#term-aliasing))</span>
- Run the unchanged full 20-row process. Accept only the two targets and the three established
  controls; every other row remains diagnostic evidence for later ordered work.<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run))</span>

### Frozen [target](../../../../glossary.md#term-target) evidence

| Row | [Fork](../../../../glossary.md#term-fork-meanings) 1 | Fork 2 | Fork 3 | Fork 4 | Fork 5 | Worst fork | Existing [owner](../../../../glossary.md#term-owner) |
|---|---:|---:|---:|---:|---:|---:|---|
| `M-STACK` | `72.565477178x` | `72.793000000x` | `72.044510386x` | `76.856816183x` | `61.540028090x` | `76.856816183x` | `CpuDataMovementEmitter` |
| `X-ANY-SINGLE` | `1.209398307x` | `1.207831644x` | `1.214668151x` | `1.202673602x` | `1.212934559x` | `1.214668151x` | `CpuAggregateEmitter` |

These ratios are diagnostic baselines from A1E, not accepted results. `M-STACK` uses [Shape](../../../../glossary.md#shape)
`[4, 64, 64, 16]`, [canonical](../../../../glossary.md#term-canonical-form) [BOOL](../../../../glossary.md#data-type--datatype), mixed `byte[]`/[`MemorySegment`](../../../../glossary.md#term-memorysegment) inputs, a `byte[]` output, and
four semantic occurrences backed by three unique inputs. `X-ANY-SINGLE` reduces [axis](../../../../glossary.md#term-axis) 1 of Shape
`[1024, 256]`; its physical input has [strides](../../../../glossary.md#term-stride) `[1, 0]`, so each output cell reads one repeated
canonical BOOL value 256 times. The direct comparator preserves that repeated logical fold.

## Out of [scope](../../../../glossary.md#term-scope)

- Changing or refreezing the A1C probe, cases, comparators, dimensions, threshold, warmup, sample
  order, fork count, or algorithm.<span data-glossary-navigation="true"> ([dimensions glossary](../../../../glossary.md#dimension); [warmup glossary](../../../../glossary.md#term-warmup); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [fork glossary](../../../../glossary.md#term-fork-meanings))</span>
- Correcting [PAD](../../../../glossary.md#padding), CONCAT, UNFOLD_AXIS, or UNFOLD2D again; [CPU](../../../../glossary.md#term-cpu) 0007A1E is complete.
- Correcting `F-FOLD2D`, `R-DROPOUT-GENERAL`, or any other deferred diagnostic row.
- Changing BOOL semantics, permitting noncanonical bytes, short-circuiting the required logical
  fold in a way that changes access/failure behavior, changing output layout, or changing resources.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [layout glossary](../../../../glossary.md#layout))</span>
- New IR/lowering/resource contracts, capability changes, route selection, public API, shared
  Prepare/Runtime behavior, architecture, dependencies, build configuration, generated fields,
  helper methods, generic carriers, mutable static state, tuning, or a universal performance claim.<span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [lowering glossary](../../../../glossary.md#lowering); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [mutable glossary](../../../../glossary.md#term-mutable))</span>

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
  evidence.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- CPU analysis retains lowering, route selection, and exact declarations before shared assignment;
  CPU finalization realizes generated bytes afterward; Runtime only executes the prepared result.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering))</span>
- [Runtime](../../../../glossary.md#term-runtime) hot work receives no [`Operation`](../../../../glossary.md#operation), `CompiledNode`, Shape, route selection, or generated-
  code decision.
- Performance evidence accepts or rejects fixed code [shapes](../../../../glossary.md#shape) and never selects production settings.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` — existing movement and aggregate
  emitters, Class-File tests, and package documentation.
- `io.github.pho001.synaptik.backend.cpu.internal.cache` — current-only schema and cache package
  documentation if emitted bytes change.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [cache glossary](../../../../glossary.md#term-cache))</span>

Packages added or changed:

- None.

Type placement:

- `CpuDataMovementEmitter` remains the sole STACK code-shaping [owner](../../../../glossary.md#term-owner).
- `CpuAggregateEmitter` remains the sole ANY code-shaping owner.
- No third production owner is authorized.

## Affected files

Expected production and contract paths:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuDataMovementEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuAggregateEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratorSchema.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/package-info.java`

Expected tests and [schema](../../../../glossary.md#term-schema) propagation:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuDataMovementGeneratedKernelTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuAggregateGeneratedKernelTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuGeneratedDirectEvidenceClosureTest.java`
- the existing four schema-assertion [owners](../../../../glossary.md#term-owner) changed by [CPU](../../../../glossary.md#term-cpu) 0007A1E, only if the schema advances.

[Planning](../../../../glossary.md#term-planning) paths are this task, CPU 0007A1C, CPU master plan, and roadmap. CPU 0007A1D/E historical
evidence changes only if a direct status cross-reference becomes inaccurate. The CPU guide and
glossary are review-only unless a current reusable contract changes.

## Maximum [scope](../../../../glossary.md#term-scope)

This task may modify at most 20 repository paths:

- 2 production algorithm-shaping [owners](../../../../glossary.md#term-owner);
- 3 [schema](../../../../glossary.md#term-schema)/package-contract paths;
- 7 test paths: two target Class-File tests, the frozen closure test, and four existing schema
  assertion owners; and<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>
- 8 documentation/planning paths: this task, A1C, optional A1D/E cross-reference corrections,
  CPU master plan, roadmap, and guide/glossary only when current contract text changes.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [CPU glossary](../../../../glossary.md#term-cpu))</span>

If a third production [owner](../../../../glossary.md#term-owner), [IR](../../../../glossary.md#term-intermediate-representation-ir)/[lowering](../../../../glossary.md#lowering)/resource change, another [module](../../../../glossary.md#term-module), another benchmark case,
or an architecture decision is needed, stop and report the conflict.

## Acceptance criteria

- Frozen semantic verification remains exactly `VERIFIED,20`.
- `M-STACK` and `X-ANY-SINGLE` each pass every one of five new isolated forks and the median of
  fork medians at <span data-glossary-navigation="true"> ([median glossary](../../../../glossary.md#term-median))</span>`<= 1.15x` under the unchanged A1C protocol.
- `P-VECTOR-SEGMENT`, `P-INTEGRAL-MIXED`, and `O-ARGSORT` pass every new [fork](../../../../glossary.md#term-fork-meanings) and aggregate.
- Exact target outputs, untouched inputs, canaries, scratch/state invariants, logical visitation,
  mapping, and comparator behavior remain unchanged.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>
- Stable Class-File assertions prove the selected primitive hot form and the retained typed
  general-long fallback where applicable.
- Every target class has exactly one typed static entry, no fields or constructors, and no bridge,
  generic `Object`, method-handle/type [constant](../../../../glossary.md#term-constant), `invokedynamic`, dynamic constant, bootstrap,
  reflection, collection/boxing, allocation, or Synaptik-owned runtime reference.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime))</span>
- If emitted bytes change, the schema advances exactly once from 33 and every prior-schema
  envelope is an incompatible safe miss.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema))</span>
- Only the two named emitters change generated algorithm shape. Semantics, resources, routes,
  capability, public API, Prepare/Runtime behavior, architecture, dependencies, and build remain
  unchanged.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Prepare glossary](../../../../glossary.md#prepare))</span>
- A distinct clean documentation context independently finalizes Javadocs/package summaries,
  planning evidence/status, guide and glossary impact, and documentation validation.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [validation glossary](../../../../glossary.md#term-validation))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) the focused [owners](../../../../glossary.md#term-owner) and the one authoritative [module](../../../../glossary.md#term-module) suite after executable Java stabilizes:

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
`-Xms1g -Xmx1g` [forks](../../../../glossary.md#term-fork-meanings). Retain raw/summarized results, two-[target](../../../../glossary.md#term-target)/control aggregates, all generated
classes, complete `javap -c -p` and `javap -v -p`, semantic/member/forbidden-reference reports,
environment, commands, and [SHA-256](../../../../glossary.md#term-sha-256) manifests in one new A1F evidence directory. A nonzero full-
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
[schema](../../../../glossary.md#term-schema) propagation, [immutable](../../../../glossary.md#term-immutable) A1C sources, empty staged diff, one detailed [Ready](../../../../glossary.md#term-ready) task, strict
correction order, and worktree preservation. Repository-wide, architecture, [conformance](../../../../glossary.md#term-conformance), and
integration suites remain deferred to [CPU](../../../../glossary.md#term-cpu) 0009 or [CI](../../../../glossary.md#term-continuous-integration-ci) because no shared contract may change.

## Dependencies

- [CPU](../../../../glossary.md#term-cpu) 0007A1C supplies the [immutable](../../../../glossary.md#term-immutable) 20-row corpus and remains Incomplete.
- CPU 0007A1D supplies stable invocation-local segment [layouts](../../../../glossary.md#layout).
- CPU 0007A1E is Complete at [schema](../../../../glossary.md#term-schema) 33 and supplies the current five-[fork](../../../../glossary.md#term-fork-meanings) [residual](../../../../glossary.md#scan-residual-port-and-history-demand) evidence.
- Completed CPU 0006A and CPU 0007A define STACK and ANY semantics/resources to preserve.

## Follow-up tasks

- [CPU 0007A1G](0007a1g-fold-and-dropout-residual-parity.md) is Complete for `F-FOLD2D` and
  `R-DROPOUT-GENERAL`.
- [CPU 0007A1H](0007a1h-numerical-aggregate-residual-parity.md) is the sole detailed [`Ready`](../../../../glossary.md#term-ready) task
  for `N-MEAN-GENERAL` and `N-PROD-MULTI`. Other [residual](../../../../glossary.md#scan-residual-port-and-history-demand) groups remain Draft and unassigned.
- [CPU](../../../../glossary.md#term-cpu) 0007A1C resumes its unchanged closure only after all residual corrections complete.
- CPU 0007A2 remains blocked until CPU 0007A1C is Complete.

## Architecture impact

Expected impact: None.

If implementation requires an architecture, [dependency](../../../../glossary.md#term-dependency), public [API](../../../../glossary.md#term-application-programming-interface-api), semantic, resource, route,
[Prepare](../../../../glossary.md#prepare), [Runtime](../../../../glossary.md#term-runtime), build, [IR](../../../../glossary.md#term-intermediate-representation-ir), or [lowering](../../../../glossary.md#lowering) change, stop and report the issue.

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
  one-fork anomaly.<span data-glossary-navigation="true"> ([fork glossary](../../../../glossary.md#term-fork-meanings))</span>
- The rows use exactly two existing code-shaping [owners](../../../../glossary.md#term-owner), matching A1C's maximum [owner](../../../../glossary.md#term-owner) boundary.
- STACK remains separate from A1E because its inserted-axis occurrence selection and extreme BOOL
  ratio need an independent hypothesis. ANY remains in this task because its small, consistent
  overage and zero-stride selected domain are bounded to the aggregate emitter.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [occurrence glossary](../../../../glossary.md#term-occurrence); [BOOL glossary](../../../../glossary.md#data-type--datatype); [stride glossary](../../../../glossary.md#term-stride))</span>
- The task preserves repeated logical ANY visits; a semantic shortcut is not assumed safe merely
  because the physical address repeats.

## Known limitations

- Ratios apply only to the frozen cases, JVM, host, and protocol; they grant no universal parity,
  tuning, or routing claim.<span data-glossary-navigation="true"> ([JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm); [host glossary](../../../../glossary.md#term-host))</span>
- This task does not close [CPU](../../../../glossary.md#term-cpu) 0007A1C or authorize CPU 0007A2.
- Deferred rows may keep the full process nonzero and must remain explicitly diagnostic.

## [Validation](../../../../glossary.md#term-validation) evidence

Implementation evidence is retained at
`/private/tmp/synaptik-cpu-0007a1f-OsajC4ko`. Implementation context
`01a01e62-9d2d-7ff1-a02c-3c3fc0bd664b` ran the exact seven-[owner](../../../../glossary.md#term-owner) focused [Gradle](../../../../glossary.md#term-gradle) command in this
task and the authoritative uncached `./gradlew :backends:cpu:test --rerun-tasks`; both completed
with `BUILD SUCCESSFUL`. No executable Java changed afterward. The unchanged A1C sources compiled
against final production classes, their before/after [SHA-256](../../../../glossary.md#term-sha-256) manifests match, and exact semantic
verification reports `VERIFIED,20`.

All five isolated [forks](../../../../glossary.md#term-fork-meanings) passed for every required [target](../../../../glossary.md#term-target) and control:

| Row | [Fork](../../../../glossary.md#term-fork-meanings) 1 | Fork 2 | Fork 3 | Fork 4 | Fork 5 | [Median](../../../../glossary.md#term-median) of fork medians | Worst fork |
|---|---:|---:|---:|---:|---:|---:|---:|
| `M-STACK` | `0.933040078x` | `0.878074544x` | `0.869558736x` | `0.932181070x` | `0.986374612x` | `0.932181070x` | `0.986374612x` |
| `X-ANY-SINGLE` | `0.307561645x` | `0.306641529x` | `0.304177560x` | `0.306305081x` | `0.307136502x` | `0.306641529x` | `0.307561645x` |
| `P-VECTOR-SEGMENT` | `0.973187118x` | `0.993918738x` | `0.968807230x` | `0.961917957x` | `0.978440665x` | `0.973187118x` | `0.993918738x` |
| `P-INTEGRAL-MIXED` | `0.317426099x` | `0.317541461x` | `0.313788337x` | `0.316096824x` | `0.314939159x` | `0.316096824x` | `0.317541461x` |
| `O-ARGSORT` | `0.849541316x` | `0.868323844x` | `0.862000991x` | `0.860249631x` | `0.856880990x` | `0.860249631x` | `0.868323844x` |

Every full 20-row process exited nonzero because ten to twelve explicitly deferred rows exceeded
`1.15x`; no required row failed. This is not a full 20-row performance pass.

The retained Class-File assessment reports one typed static entry and zero fields for each target.
The proved `M-STACK` branch emits the frozen [occurrence](../../../../glossary.md#term-occurrence) order with direct `byte[]` and
[`MemorySegment`](../../../../glossary.md#term-memorysegment) reads and primitive byte stores. Its sole `NEW` instruction belongs to the retained
general-long fallback's exceptional failure path. The proved `X-ANY-SINGLE` branch performs every
selected-domain `MemorySegment.get`, combines with `IOR`, has no [accumulator](../../../../glossary.md#accumulator-type) early exit, and
canonicalizes once at output. Neither proved branch contains allocation, boxing, reflection,
`invokedynamic`, or a Synaptik-owned [runtime](../../../../glossary.md#term-runtime) reference. Both retain typed general-long fallbacks.

Exactly nine implementation-owned [CPU](../../../../glossary.md#term-cpu) paths changed: the two emitters, [schema](../../../../glossary.md#term-schema), two target
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
the then-current sole detailed [Ready](../../../../glossary.md#term-ready) A1G plan. Final documentation validation is recorded below.

`./gradlew :backends:cpu:javadoc` completed with `BUILD SUCCESSFUL` and only the two expected
incubating-[Vector](../../../../glossary.md#term-vector) warnings. Local Markdown targets and balanced fences passed for all seven
changed Markdown documents; the new A1G links introduce no fragment anchors. Final-newline,
whitespace, schema-34/current-text, exact evidence-path, frozen-[hash](../../../../glossary.md#term-hash), status [synchronization](../../../../glossary.md#term-synchronization),
sole-Ready, A2-blocking, and 19-path combined-[scope](../../../../glossary.md#term-scope) checks passed. `git diff --check` and
`git diff --cached --check` passed, and the staged diff is empty. `git status --short` was
inspected; only the A1F implementation and focused documentation/[planning](../../../../glossary.md#term-planning) paths remain dirty.

## Implementation notes

- `CpuDataMovementEmitter` cold-proves the frozen axis-zero canonical-BOOL STACK geometry before
  using one direct primitive loop per semantic occurrence. Repeated mapping position <span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [canonical glossary](../../../../glossary.md#term-canonical-form); [BOOL glossary](../../../../glossary.md#data-type--datatype); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>`0` remains a
  repeated copy; mixed input carriers and byte-array output are embedded in the generated entry.
- `CpuAggregateEmitter` cold-proves the rank-two canonical-BOOL ANY geometry before using direct
  primitive output-cell/domain loops. It deliberately preserves all 256 logical reads and applies
  canonicalization only after <span data-glossary-navigation="true"> ([rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form))</span>`IOR` accumulation.
- Geometry outside either narrow proof continues through the existing typed general-long body.
- [Schema](../../../../glossary.md#term-schema) 34 is current-only. Schema-33 and earlier [artifacts](../../../../glossary.md#term-artifact) are incompatible safe misses.

## Completion summary

- Completed changes: added the two narrow cold-proved direct canonical-BOOL forms with retained
  typed fallbacks, advanced generated compatibility to schema 34, added structural tests, and
  finalized Javadocs, guide, glossary, evidence, and planning state.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [BOOL glossary](../../../../glossary.md#data-type--datatype); [schema glossary](../../../../glossary.md#term-schema); [planning glossary](../../../../glossary.md#term-planning))</span>
- Files changed or created: nine implementation-owned CPU paths; three affected package-summary
  Javadocs; CPU backend guide; glossary; A1F, A1C follow-up record, CPU master plan, roadmap; and
  the new A1G task specification.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [backend glossary](../../../../glossary.md#backend))</span>
- Tests and validation: reused the successful focused and authoritative CPU test evidence;
  frozen semantics passed <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>`VERIFIED,20`; all 25 required fork ratios and five aggregates passed;
  target Class-File, forbidden-reference, schema, frozen-source, and exact-scope gates passed.
  The separate documentation checks and CPU Javadoc passed as recorded above.<span data-glossary-navigation="true"> ([fork glossary](../../../../glossary.md#term-fork-meanings); [target glossary](../../../../glossary.md#term-target); [scope glossary](../../../../glossary.md#term-scope); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Documentation-agent review: clean context `01a01e83-bb41-7513-8b8a-9a17a0c15c3d` finalized
  affected Javadocs/package summaries, bounded guide/glossary text, task evidence/status, master
  plan, roadmap, and A1G without executable changes or duplicate Java/performance runs.
- Documentation impact: the guide and glossary now describe schema-34 STACK/ANY forms as narrow
  cold-proved paths with typed fallbacks and bounded retained evidence, not universal support or
  performance claims.
- Javadoc review: emitter/schema type documentation was accurate; package summaries required the
  schema-34 and narrow-form updates. Prepare package documentation required only the current
  artifact-version correction.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- Unresolved issues: A1C and A1D remain `Review needed`; deferred rows keep the full probe nonzero.
- Follow-up required: CPU 0007A1G is Complete; implement sole Ready CPU 0007A1H, then plan later
  residual owner clusters in evidence-backed order. CPU 0007A2 remains blocked until A1C closes.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready); [residual glossary](../../../../glossary.md#scan-residual-port-and-history-demand); [owner glossary](../../../../glossary.md#term-owner))</span>

Status: Complete
