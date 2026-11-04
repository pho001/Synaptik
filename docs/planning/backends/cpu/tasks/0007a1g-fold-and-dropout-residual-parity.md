# Task 0007A1G: Fold and [Dropout](../../../../glossary.md#dropout--inverted-dropout) [Residual](../../../../glossary.md#term-residual-meanings) Parity

## Reader context

These remaining timing gaps—not saved differentiation values—belonged to different emitters:
FOLD2D accumulated overlapping image-window contributions, while DROPOUT mapped explicit
key/counter state to keep decisions. Each gained a fully guarded body for one frozen FLOAT32
mixed-carrier geometry, alongside the original general path in the same class. “Mixed carrier”
meant that different arguments used arrays and segments; it did not mean mixed numerical
precision.

Fold retained output-owned contribution order and positive-zero initialization. Dropout retained
global-ordinal replay, next-state handling, and arbitrary legal subranges. Neither acquired
hidden shared state or changed Runtime ownership. The unusually small recorded fold ratio is
evidence for the guarded comparator case, not a claim about every fold. Current window and
randomness references explain the separate meanings.

Current reading alongside this historical record:

- [Window-transform expressions](../../../../reference/tensor/windows.md#window-transform-expressions)
  — distinguishes extraction, columns geometry, and overlap accumulation.
- [Explicit-state dropout construction](../../../../reference/tensor/graph-randomness.md#explicit-state-dropout-construction)
  — names value, mask, and next-state positions without promising a portable bitstream.

## Status

Complete

## Goal

Correct the next bounded [residual](../../../../glossary.md#term-residual-meanings) group in the frozen [CPU](../../../../glossary.md#term-cpu) 0007A1C comparison:
`F-FOLD2D` and `R-DROPOUT-GENERAL`. Preserve their exact fold and explicit-state [dropout](../../../../glossary.md#dropout--inverted-dropout)
semantics while making each generated entry meet the unchanged generated/direct `<= 1.15x` gate
in every one of five isolated [forks](../../../../glossary.md#term-fork-meanings).

These are the two smallest remaining measured residuals after CPU 0007A1F. They belong to exactly
two existing code-shaping [owners](../../../../glossary.md#term-owner), `CpuFoldEmitter` and `CpuRandomEmitter`.

## [Scope](../../../../glossary.md#term-scope)

- Use the unchanged frozen A1C sources under `/private/tmp/synaptik-cpu-0007a1c-oMAmuVhF` and the
  completed schema-34 A1F evidence under <span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema))</span>`/private/tmp/synaptik-cpu-0007a1f-OsajC4ko`.
- Correct `F-FOLD2D` only through `CpuFoldEmitter`. Preserve padded and dilated overlapping
  FOLD2D mapping, output-cell ownership, canonical logical input-occurrence order, represented
  FLOAT32 sequential addition, mixed carriers, arbitrary legal half-open output ranges, and zero
  workspace.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [canonical glossary](../../../../glossary.md#term-canonical-form); [occurrence glossary](../../../../glossary.md#term-occurrence); [FLOAT32 glossary](../../../../glossary.md#term-float32); [half-open glossary](../../../../glossary.md#term-half-open-interval))</span>
- Correct `R-DROPOUT-GENERAL` only through `CpuRandomEmitter`. Preserve the exact
  `SYNAPTIK_CPU_SPLITMIX64_COUNTER_V1` word mapping, key/counter roles, probability threshold,
  FLOAT32 scaling/narrowing, canonical BOOL mask, final-state update, mixed general-long carriers,
  arbitrary legal element ranges, and the dedicated <span data-glossary-navigation="true"> ([narrowing glossary](../../../../glossary.md#term-narrowing); [BOOL glossary](../../../../glossary.md#data-type--datatype); [mask glossary](../../../../glossary.md#term-boolean-mask); [element glossary](../../../../glossary.md#term-element))</span>`[0,0)` state prologue.
- Cold-prove and retain only primitive geometry, addresses, coordinates, or state facts needed by
  the two fixed shapes. Retain typed general-long fallbacks whenever required bounds or
  transitions cannot be proved.<span data-glossary-navigation="true"> ([coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [shapes glossary](../../../../glossary.md#shape))</span>
- Treat optimal direct clean Java as the implementation and review oracle. Generated and
  decompiled target branches must preserve the same semantic algorithm, control-flow shape,
  hot-loop dataflow, represented operations, and avoidable-overhead profile. Any deviation must
  have an explicit technical reason and supporting structural and measured evidence.<span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle); [target glossary](../../../../glossary.md#term-target); [dataflow glossary](../../../../glossary.md#term-dataflow))</span>
- Keep repeated proved work free of allocation, boxing, reflection, dynamic dispatch, generic-
  fallback overhead, and Synaptik-owned runtime calls.<span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Add stable Class-File assertions for both corrected shapes without fixed bytecode offsets,
  constant-pool indexes, generated names, or JIT assembly.<span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode); [constant glossary](../../../../glossary.md#term-constant); [JIT glossary](../../../../glossary.md#term-just-in-time-compilation-jit))</span>
- Advance the current-only generated schema exactly once if emitted bytes change. Older artifacts
  remain incompatible safe misses without migration, aliasing, conversion, or dual-schema reuse.<span data-glossary-navigation="true"> ([artifacts glossary](../../../../glossary.md#term-artifact); [aliasing glossary](../../../../glossary.md#term-aliasing))</span>
- Run the unchanged full 20-row process. Accept only the two targets and the three established
  controls; all other rows remain diagnostic evidence for later ordered work.<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run))</span>

### Frozen [target](../../../../glossary.md#term-target) evidence

| Row | [Fork](../../../../glossary.md#term-fork-meanings) 1 | Fork 2 | Fork 3 | Fork 4 | Fork 5 | [Median](../../../../glossary.md#term-median) | Worst fork | Existing [owner](../../../../glossary.md#term-owner) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `F-FOLD2D` | `1.154454122x` | `1.170489456x` | `1.160103362x` | `1.166606799x` | `1.157989230x` | `1.160103362x` | `1.170489456x` | `CpuFoldEmitter` |
| `R-DROPOUT-GENERAL` | `1.144406640x` | `1.154724914x` | `1.151115377x` | `1.165588748x` | `1.158634012x` | `1.154724914x` | `1.165588748x` | `CpuRandomEmitter` |

The FOLD2D row fails every fork. The [dropout](../../../../glossary.md#dropout--inverted-dropout) row passes fork 1 but fails four forks and its
aggregate. Both remain reproducible bounded residuals rather than one-fork anomalies. FOLD2D is
the fixed mixed-[carrier](../../../../glossary.md#term-storage-carrier) general-long padded/dilated overlapping [FLOAT32](../../../../glossary.md#term-float32) case. Dropout is the fixed
1,048,576-[element](../../../../glossary.md#term-element) mixed general-long FLOAT32 case with exact V1 state, value, and [mask](../../../../glossary.md#term-boolean-mask) work.

### Accepted A1G results

| Row | [Fork](../../../../glossary.md#term-fork-meanings) 1 | Fork 2 | Fork 3 | Fork 4 | Fork 5 | [Median](../../../../glossary.md#term-median) | Worst fork | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `F-FOLD2D` | `0.234704046x` | `0.231469222x` | `0.233527780x` | `0.232217638x` | `0.231026459x` | `0.232217638x` | `0.234704046x` | PASS |
| `R-DROPOUT-GENERAL` | `1.052637855x` | `1.053737922x` | `1.056324388x` | `1.048828706x` | `1.050161298x` | `1.052637855x` | `1.056324388x` | PASS |
| `P-VECTOR-SEGMENT` | `0.972062342x` | `0.961146655x` | `0.983583863x` | `0.979836134x` | `0.974884851x` | `0.974884851x` | `0.983583863x` | PASS |
| `P-INTEGRAL-MIXED` | `0.316515333x` | `0.315026507x` | `0.314328088x` | `0.313621556x` | `0.314718409x` | `0.314718409x` | `0.316515333x` | PASS |
| `O-ARGSORT` | `0.862641168x` | `0.877979349x` | `0.903613277x` | `0.885096605x` | `0.857920056x` | `0.877979349x` | `0.903613277x` | PASS |

All five full probe processes exited nonzero only because nine or ten deferred diagnostic rows
still exceeded the gate. These accepted results close the two A1G [targets](../../../../glossary.md#term-target) and three controls;
they are not a full 20-row performance pass.

## Out of [scope](../../../../glossary.md#term-scope)

- Changing or refreezing the A1C sources, cases, comparators, dimensions, threshold, warmup,
  sample order, fork count, state initialization, or algorithm.<span data-glossary-navigation="true"> ([dimensions glossary](../../../../glossary.md#dimension); [warmup glossary](../../../../glossary.md#term-warmup); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [fork glossary](../../../../glossary.md#term-fork-meanings))</span>
- Correcting any pointwise, affine, movement, indexing, scatter, ordering, scan, aggregate, or
  numerical residual row.<span data-glossary-navigation="true"> ([pointwise glossary](../../../../glossary.md#term-pointwise); [scan glossary](../../../../glossary.md#term-cumulative-scan); [residual glossary](../../../../glossary.md#term-residual-meanings))</span>
- Changing fold addition order or bits, dropout RNG mapping or outputs, carrier pattern, layout,
  resources, materialization, worker ownership, route, capability, public API, Prepare/Runtime
  behavior, dependency direction, architecture, or build configuration.<span data-glossary-navigation="true"> ([dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [carrier glossary](../../../../glossary.md#term-storage-carrier); [layout glossary](../../../../glossary.md#layout); [materialization glossary](../../../../glossary.md#materialization); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools))</span>
- New IR/lowering/resource contracts, generated fields, constructors, secondary helpers, generic
  carriers, mutable static state, benchmark tooling, tuning, or a universal performance claim.<span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [lowering glossary](../../../../glossary.md#lowering); [mutable glossary](../../../../glossary.md#term-mutable))</span>

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
- [`CPU 0007A1F`](0007a1f-bool-movement-and-aggregate-residual-parity.md)

## Architecture constraints

- Model owns FOLD2D and DROPOUT semantics; this task changes only CPU-private generated
  realization and evidence.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [DROPOUT glossary](../../../../glossary.md#dropout--inverted-dropout); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- CPU analysis retains lowering, route selection, and exact declarations before shared slot
  assignment. CPU finalization realizes generated bytes afterward. Runtime only executes the
  immutable prepared result.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [Runtime glossary](../../../../glossary.md#term-runtime); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- Generated entries remain typed, direct, field-free, allocation-free in normal proved work, free
  of mutable static state, and free of Synaptik-owned per-element calls.<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [element glossary](../../../../glossary.md#term-element))</span>
- Benchmark evidence accepts or rejects fixed code shapes and never selects production settings.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` — existing fold and random
  emitters, Class-File tests, and package documentation.
- `io.github.pho001.synaptik.backend.cpu.internal.cache` — current-only schema and cache package
  documentation if emitted bytes change.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [cache glossary](../../../../glossary.md#term-cache))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.prepare` — existing fold/random/aggregate
  declaration and current-artifact lifecycle summary.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>

Packages added or changed:

- None.

Type placement:

- `CpuFoldEmitter` remains the sole FOLD2D code-shaping [owner](../../../../glossary.md#term-owner).
- `CpuRandomEmitter` remains the sole [DROPOUT](../../../../glossary.md#dropout--inverted-dropout) code-shaping owner.
- No third production owner is authorized.

## Affected files

Changed production and contract paths:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuFoldEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuRandomEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratorSchema.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/package-info.java`

Changed tests and [schema](../../../../glossary.md#term-schema) propagation:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuFoldGeneratedKernelTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuRandomGeneratedKernelTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratedKernelArtifactStoreTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecializationTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizerTest.java`

Changed documentation and [planning](../../../../glossary.md#term-planning) paths:

- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`
- `docs/planning/backends/cpu/tasks/0007a1c-generated-direct-evidence-closure.md`
- `docs/planning/backends/cpu/tasks/0007a1f-bool-movement-and-aggregate-residual-parity.md`
- this task
- `docs/planning/backends/cpu/tasks/0007a1h-numerical-aggregate-residual-parity.md`
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

No A1D or A1E historical task file changed. A1F changed only to replace its now-stale next-task
cross-reference. The implementation retained the existing closure test unchanged because the
frozen external evidence supplied the required complete structural report.

## Maximum [scope](../../../../glossary.md#term-scope)

The completed change uses the permitted maximum of 20 repository paths:

- 2 production algorithm-shaping [owners](../../../../glossary.md#term-owner);
- 4 [schema](../../../../glossary.md#term-schema)/package-contract paths;
- 6 test paths: two [target](../../../../glossary.md#term-target) Class-File tests and four schema-assertion owners; and
- 8 documentation/planning paths: this task, A1C, the A1F cross-reference, A1H, CPU master plan,
  roadmap, guide, and glossary.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [CPU glossary](../../../../glossary.md#term-cpu))</span>

If a third production [owner](../../../../glossary.md#term-owner), [IR](../../../../glossary.md#term-intermediate-representation-ir)/[lowering](../../../../glossary.md#lowering)/resource change, another [module](../../../../glossary.md#term-module), another benchmark case,
or an architecture decision is needed, stop and report the conflict.

## Acceptance criteria

- Frozen semantic verification remains exactly `VERIFIED,20`.
- `F-FOLD2D` and `R-DROPOUT-GENERAL` each pass every one of five new isolated forks and the
  median of fork medians at <span data-glossary-navigation="true"> ([median glossary](../../../../glossary.md#term-median))</span>`<= 1.15x` under the unchanged A1C protocol.
- `P-VECTOR-SEGMENT`, `P-INTEGRAL-MIXED`, and `O-ARGSORT` pass every new [fork](../../../../glossary.md#term-fork-meanings) and aggregate.
- Exact target outputs, untouched inputs, canaries, scratch/state invariants, logical visitation,
  mapping, addition order, RNG mapping, and comparator behavior remain unchanged.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>
- Stable Class-File assertions prove the selected primitive hot form and retained typed
  general-long fallback where applicable.
- Generated/decompiled proved branches match optimal direct clean Java in semantic algorithm,
  control flow, hot-path dataflow, and avoidable-overhead profile. Any deviation is explicitly
  justified and measured.<span data-glossary-navigation="true"> ([dataflow glossary](../../../../glossary.md#term-dataflow))</span>
- Repeated proved work contains no allocation, boxing, reflection, dynamic dispatch, generic-
  fallback overhead, or Synaptik-owned runtime call. Each target class has exactly one typed static
  entry and no fields, constructor, bridge, generic <span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch); [runtime glossary](../../../../glossary.md#term-runtime))</span>`Object`, method-handle/type [constant](../../../../glossary.md#term-constant),
  `invokedynamic`, dynamic constant, or bootstrap.
- If emitted bytes change, schema advances exactly once from 34 and every prior-schema envelope is
  an incompatible safe miss.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema))</span>
- Only the two named emitters change generated algorithm shape. Semantics, resources, routes,
  capability, public API, Prepare/Runtime behavior, architecture, dependencies, and build remain
  unchanged.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Prepare glossary](../../../../glossary.md#prepare))</span>
- A distinct clean documentation context independently finalizes Javadocs/package summaries,
  planning evidence/status, guide and glossary impact, and documentation validation.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [validation glossary](../../../../glossary.md#term-validation))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) the focused [owners](../../../../glossary.md#term-owner) and one authoritative uncached [module](../../../../glossary.md#term-module) suite after executable Java stabilizes:

```bash
./gradlew :backends:cpu:test \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneratedDirectEvidenceClosureTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuFoldGeneratedKernelTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuRandomGeneratedKernelTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratedKernelArtifactStoreTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecializationTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizerTest
./gradlew :backends:cpu:test --rerun-tasks
```

Recompile the unchanged A1C sources, run exact 20-row semantics, and run five isolated
`-Xms1g -Xmx1g` [forks](../../../../glossary.md#term-fork-meanings). Retain raw and summarized results, two-[target](../../../../glossary.md#term-target)/control aggregates, all
generated classes, complete `javap -c -p` and `javap -v -p`, semantic/member/forbidden-reference
reports, environment, commands, and [SHA-256](../../../../glossary.md#term-sha-256) manifests in one new A1G evidence directory. A
nonzero full-probe exit is acceptable only for explicitly deferred rows and must not be described
as a full performance pass.

The documentation context reuses stable Java/timing evidence and runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
git diff --cached --check
git status --short
```

It also validates Markdown links/anchors/fences/newlines, terminology/glossary, exact owners and
[schema](../../../../glossary.md#term-schema) propagation, [immutable](../../../../glossary.md#term-immutable) A1C sources, empty staged diff, one detailed [Ready](../../../../glossary.md#term-ready) task, strict
correction order, clean-Java equivalence evidence, and worktree preservation. Repository-wide,
architecture, [conformance](../../../../glossary.md#term-conformance), and integration suites remain deferred to [CPU](../../../../glossary.md#term-cpu) 0009 or [CI](../../../../glossary.md#term-continuous-integration-ci) because no
shared contract may change.

## Dependencies

- [CPU](../../../../glossary.md#term-cpu) 0007A1C supplies the [immutable](../../../../glossary.md#term-immutable) 20-row corpus and remains Incomplete.
- CPU 0007A1D supplies stable invocation-local segment [layouts](../../../../glossary.md#layout).
- CPU 0007A1E is Complete at [schema](../../../../glossary.md#term-schema) 33.
- CPU 0007A1F is Complete at schema 34 and supplies the current five-[fork](../../../../glossary.md#term-fork-meanings) [residual](../../../../glossary.md#term-residual-meanings) evidence.
- Completed CPU 0006B2, CPU 0006D, CPU 0007A0D, and CPU 0007A0F define fold/dropout semantics,
  resources, and generated foundations to preserve.<span data-glossary-navigation="true"> ([dropout glossary](../../../../glossary.md#dropout--inverted-dropout))</span>

## Follow-up tasks

- [CPU](../../../../glossary.md#term-cpu) 0007A1H is the sole next [`Ready`](../../../../glossary.md#term-ready) task. It owns persistent `N-MEAN-GENERAL` and
  `N-PROD-MULTI` residuals through one code-shaping [owner](../../../../glossary.md#term-owner), `CpuAggregateEmitter`.
- Other [residual](../../../../glossary.md#term-residual-meanings) [target](../../../../glossary.md#term-target) groups remain Draft and unassigned.
- CPU 0007A1C resumes its unchanged closure only after all residual corrections complete.
- CPU 0007A2 remains blocked until CPU 0007A1C is Complete.

## Architecture impact

Expected impact: None.

If implementation requires an architecture, [dependency](../../../../glossary.md#term-dependency), public [API](../../../../glossary.md#term-application-programming-interface-api), semantic, resource, route,
[Prepare](../../../../glossary.md#prepare), [Runtime](../../../../glossary.md#term-runtime), build, [IR](../../../../glossary.md#term-intermediate-representation-ir), or [lowering](../../../../glossary.md#lowering) change, stop and report the issue.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
You are the clean implementation agent for Synaptik CPU task 0007A1G. Work on the existing dirty
worktree without committing, pushing, staging, resetting, reverting, deleting, or modifying
unrelated work. Do not use a GSD skill or workflow.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, CPU task 0007A1G, CPU tasks
0007A1C–0007A1F, and the retained A1C/A1F evidence in full. Implement exactly the bounded
two-owner fold/dropout correction. Preserve the frozen sources, semantics, resources, comparators,
and typed fallbacks. Use optimal direct clean Java as the generated-code design and review oracle;
preserve its algorithm, control flow, hot-loop dataflow, and avoidable-overhead profile, and
justify and measure any deviation. Stop before a third production owner or any IR/lowering/
resource/architecture/lifecycle change.

Run the specified focused/full CPU, frozen semantic/Class-File/five-fork, schema, scope,
clean-Java-equivalence, and preservation gates. Hand the stabilized diff and exact evidence to a
distinct clean documentation context. Do not mark Complete until both targets and all three
controls pass every required gate. Do not commit or push.
```

## Local decisions

- A1F evidence confirms FOLD2D fails all five forks and general dropout fails four forks plus its
  aggregate. Their small but repeated overages are bounded residuals.<span data-glossary-navigation="true"> ([forks glossary](../../../../glossary.md#term-fork-meanings))</span><span data-glossary-navigation="true"> ([dropout glossary](../../../../glossary.md#dropout--inverted-dropout))</span>
- The rows use exactly two existing code-shaping [owners](../../../../glossary.md#term-owner), matching the correction-[owner](../../../../glossary.md#term-owner) boundary.
- They remain one task because they are the next two residual ratios, have the same frozen
  protocol and evidence lifecycle, and together remain within the two-owner/task-size guardrails.<span data-glossary-navigation="true"> ([residual glossary](../../../../glossary.md#term-residual-meanings); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- No optimization may replace the exact sequential fold or V1 dropout work with an easier direct
  algorithm merely to pass the ratio gate.

## Known limitations

- Ratios apply only to the frozen cases, JVM, host, and protocol; they grant no universal parity,
  tuning, or routing claim.<span data-glossary-navigation="true"> ([JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm); [host glossary](../../../../glossary.md#term-host))</span>
- This task does not close [CPU](../../../../glossary.md#term-cpu) 0007A1C or authorize CPU 0007A2.
- Deferred rows may keep the full process nonzero and must remain explicitly diagnostic.

## [Validation](../../../../glossary.md#term-validation) evidence

Implementation context `01a01ee5-da54-7090-831b-769bf2fa87a7` retained complete evidence under
`/private/tmp/synaptik-cpu-0007a1g-sLBHbIpI`. Frozen sources remained byte-identical and exact
semantics reported `VERIFIED,20`. The focused [owners](../../../../glossary.md#term-owner) passed, and the authoritative uncached [CPU](../../../../glossary.md#term-cpu)
[module](../../../../glossary.md#term-module) [run](../../../../glossary.md#term-run) passed 350 tests with zero failures or errors. Twenty generated Class-Files, complete
`javap -c -p` and `javap -v -p` output, member/forbidden-reference checks, clean-Java equivalence,
[schema](../../../../glossary.md#term-schema) compatibility, five raw [forks](../../../../glossary.md#term-fork-meanings), summaries, commands, and [SHA-256](../../../../glossary.md#term-sha-256) manifests are retained.
The evidence [manifest](../../../../glossary.md#term-manifest) reverified without mismatch during the documentation pass.

The independent documentation pass ran `./gradlew :backends:cpu:javadoc` successfully with only
the expected incubating [Vector API](../../../../glossary.md#term-vector-api) warnings. Focused Markdown file/link/anchor/fence/final-newline,
status/frontier, schema, exact 20-path, frozen-[hash](../../../../glossary.md#term-hash), and staged-state checks passed. Final
`git diff --check` and `git diff --cached --check` passed, and the staged path inventory remained
empty. Java tests and performance forks were not rerun because this pass changed no executable
Java behavior or tests.

## Implementation notes

- `F-FOLD2D` uses a complete cold guard for the frozen FLOAT32 mixed-carrier padded/dilated shape.
  Its proved branch follows output cell -> kernel position <span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [carrier glossary](../../../../glossary.md#term-storage-carrier); [shape glossary](../../../../glossary.md#shape); [kernel glossary](../../../../glossary.md#kernel))</span>`q` -> column order, initializes with
  `+0.0f`, performs canonical sequential FLOAT32 additions, stores once, and removes singleton
  dimensions only under the guard. Its typed general-long body remains in the same class.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [singleton glossary](../../../../glossary.md#term-singleton); [dimensions glossary](../../../../glossary.md#dimension))</span>
- `R-DROPOUT-GENERAL` uses a complete cold [guard](../../../../glossary.md#term-guard) for the frozen [rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) `1 << 20` FLOAT32 mixed-
  carrier shape. Its proved branch uses primitive integer cursors while retaining exact SplitMix64
  V1 key/counter/mix/uniform/threshold/mask/value order, <span data-glossary-navigation="true"> ([SplitMix64 glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [mask glossary](../../../../glossary.md#term-boolean-mask))</span>`[0,0)` state handling, and arbitrary
  subranges. Its typed general-long body remains in the same class.
- [Schema](../../../../glossary.md#term-schema) advanced exactly once from 34 to 35. Older envelopes remain incompatible safe misses.
- Both [target](../../../../glossary.md#term-target) classes are field-free final classes with exactly one typed static `invoke`; the
  retained reports find no constructor, bridge, `Object` descriptor, allocation, reflection,
  dynamic dispatch, method handle, <span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [dispatch glossary](../../../../glossary.md#term-dispatch); [method handle glossary](../../../../glossary.md#term-method-handle))</span>`invokedynamic`, dynamic constant, bootstrap, Synaptik runtime,
  or emitter reference.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- The unusually low fold ratio reflects the same frozen direct comparator algorithm with cold-
  proved constants and removed singleton/general coordinate machinery. It does not establish a
  broader algorithm or performance claim.<span data-glossary-navigation="true"> ([coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>

## Completion summary

Completed changes:

- finalized [schema](../../../../glossary.md#term-schema)-35 Javadocs and [cache](../../../../glossary.md#term-cache)/codegen/[prepare](../../../../glossary.md#prepare) package summaries;
- documented the two guarded generated forms and their typed general-long fallbacks in the CPU
  guide and glossary;<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- recorded exact A1G semantic, Class-File, schema, Java-test, and five-[fork](../../../../glossary.md#term-fork-meanings) evidence;
- synchronized A1G as Complete while leaving A1C and A1D `Review needed`/Incomplete and A1E/A1F
  Complete; and
- specified A1H as the sole next [Ready](../../../../glossary.md#term-ready) [residual](../../../../glossary.md#term-residual-meanings) task. The one-fork `M-CONCAT` value
  `1.158061861x` does not reopen completed A1E because its other four A1G forks and completed A1E
  five-fork evidence do not establish a persistent failure.

No architecture, [ADR](../../../../glossary.md#term-architecture-decision-record-adr), public [API](../../../../glossary.md#term-application-programming-interface-api), capability, [conformance](../../../../glossary.md#term-conformance), integration, [dependency](../../../../glossary.md#term-dependency), resource,
Prepare/[Runtime](../../../../glossary.md#term-runtime) contract, or build change was needed. Cache, codegen, and prepare summaries were
updated only where schema-35 or guarded-form wording was stale; other relevant packages and
Javadocs remain accurate. Java tests and performance forks were not rerun because this independent
pass changed no executable Java behavior or tests.

Status: Complete
