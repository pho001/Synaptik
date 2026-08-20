# Task 0007A1G: Fold and Dropout Residual Parity

## Status

Ready

## Goal

Correct the next bounded residual group in the frozen CPU 0007A1C comparison:
`F-FOLD2D` and `R-DROPOUT-GENERAL`. Preserve their exact fold and explicit-state dropout
semantics while making each generated entry meet the unchanged generated/direct `<= 1.15x` gate
in every one of five isolated forks.

These are the two smallest remaining measured residuals after CPU 0007A1F. They belong to exactly
two existing code-shaping owners, `CpuFoldEmitter` and `CpuRandomEmitter`.

## Scope

- Use the unchanged frozen A1C sources under `/private/tmp/synaptik-cpu-0007a1c-oMAmuVhF` and the
  completed schema-34 A1F evidence under `/private/tmp/synaptik-cpu-0007a1f-OsajC4ko`.
- Correct `F-FOLD2D` only through `CpuFoldEmitter`. Preserve padded and dilated overlapping
  FOLD2D mapping, output-cell ownership, canonical logical input-occurrence order, represented
  FLOAT32 sequential addition, mixed carriers, arbitrary legal half-open output ranges, and zero
  workspace.
- Correct `R-DROPOUT-GENERAL` only through `CpuRandomEmitter`. Preserve the exact
  `SYNAPTIK_CPU_SPLITMIX64_COUNTER_V1` word mapping, key/counter roles, probability threshold,
  FLOAT32 scaling/narrowing, canonical BOOL mask, final-state update, mixed general-long carriers,
  arbitrary legal element ranges, and the dedicated `[0,0)` state prologue.
- Cold-prove and retain only primitive geometry, addresses, coordinates, or state facts needed by
  the two fixed shapes. Retain typed general-long fallbacks whenever required bounds or
  transitions cannot be proved.
- Treat optimal direct clean Java as the implementation and review oracle. Generated and
  decompiled target branches must preserve the same semantic algorithm, control-flow shape,
  hot-loop dataflow, represented operations, and avoidable-overhead profile. Any deviation must
  have an explicit technical reason and supporting structural and measured evidence.
- Keep repeated proved work free of allocation, boxing, reflection, dynamic dispatch, generic-
  fallback overhead, and Synaptik-owned runtime calls.
- Add stable Class-File assertions for both corrected shapes without fixed bytecode offsets,
  constant-pool indexes, generated names, or JIT assembly.
- Advance the current-only generated schema exactly once if emitted bytes change. Older artifacts
  remain incompatible safe misses without migration, aliasing, conversion, or dual-schema reuse.
- Run the unchanged full 20-row process. Accept only the two targets and the three established
  controls; all other rows remain diagnostic evidence for later ordered work.

### Frozen target evidence

| Row | Fork 1 | Fork 2 | Fork 3 | Fork 4 | Fork 5 | Median | Worst fork | Existing owner |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `F-FOLD2D` | `1.154454122x` | `1.170489456x` | `1.160103362x` | `1.166606799x` | `1.157989230x` | `1.160103362x` | `1.170489456x` | `CpuFoldEmitter` |
| `R-DROPOUT-GENERAL` | `1.144406640x` | `1.154724914x` | `1.151115377x` | `1.165588748x` | `1.158634012x` | `1.154724914x` | `1.165588748x` | `CpuRandomEmitter` |

The FOLD2D row fails every fork. The dropout row passes fork 1 but fails four forks and its
aggregate. Both remain reproducible bounded residuals rather than one-fork anomalies. FOLD2D is
the fixed mixed-carrier general-long padded/dilated overlapping FLOAT32 case. Dropout is the fixed
1,048,576-element mixed general-long FLOAT32 case with exact V1 state, value, and mask work.

## Out of scope

- Changing or refreezing the A1C sources, cases, comparators, dimensions, threshold, warmup,
  sample order, fork count, state initialization, or algorithm.
- Correcting any pointwise, affine, movement, indexing, scatter, ordering, scan, aggregate, or
  numerical residual row.
- Changing fold addition order or bits, dropout RNG mapping or outputs, carrier pattern, layout,
  resources, materialization, worker ownership, route, capability, public API, Prepare/Runtime
  behavior, dependency direction, architecture, or build configuration.
- New IR/lowering/resource contracts, generated fields, constructors, secondary helpers, generic
  carriers, mutable static state, benchmark tooling, tuning, or a universal performance claim.

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
  realization and evidence.
- CPU analysis retains lowering, route selection, and exact declarations before shared slot
  assignment. CPU finalization realizes generated bytes afterward. Runtime only executes the
  immutable prepared result.
- Generated entries remain typed, direct, field-free, allocation-free in normal proved work, free
  of mutable static state, and free of Synaptik-owned per-element calls.
- Benchmark evidence accepts or rejects fixed code shapes and never selects production settings.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` — existing fold and random
  emitters, Class-File tests, and package documentation.
- `io.github.pho001.synaptik.backend.cpu.internal.cache` — current-only schema and cache package
  documentation if emitted bytes change.

Packages added or changed:

- None.

Type placement:

- `CpuFoldEmitter` remains the sole FOLD2D code-shaping owner.
- `CpuRandomEmitter` remains the sole DROPOUT code-shaping owner.
- No third production owner is authorized.

## Affected files

Expected production and contract paths:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuFoldEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuRandomEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratorSchema.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/package-info.java`

Expected tests and schema propagation:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuFoldGeneratedKernelTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuRandomGeneratedKernelTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuGeneratedDirectEvidenceClosureTest.java`
- the existing four schema-assertion owners changed by CPU 0007A1F, only if schema advances.

Planning paths are this task, CPU 0007A1C, CPU master plan, and roadmap. A1D–A1F historical
evidence changes only if a direct status cross-reference becomes inaccurate. The CPU guide,
glossary, and prepare package summary change only when current contract text requires it.

## Maximum scope

This task may modify at most 20 repository paths:

- 2 production algorithm-shaping owners;
- 3 schema/package-contract paths;
- 7 test paths: two target Class-File tests, the frozen closure test, and four existing schema
  assertion owners; and
- 8 documentation/planning paths: this task, A1C, optional A1D–A1F cross-reference corrections,
  CPU master plan, roadmap, and guide/glossary only when current contract text changes.

If a third production owner, IR/lowering/resource change, another module, another benchmark case,
or an architecture decision is needed, stop and report the conflict.

## Acceptance criteria

- Frozen semantic verification remains exactly `VERIFIED,20`.
- `F-FOLD2D` and `R-DROPOUT-GENERAL` each pass every one of five new isolated forks and the
  median of fork medians at `<= 1.15x` under the unchanged A1C protocol.
- `P-VECTOR-SEGMENT`, `P-INTEGRAL-MIXED`, and `O-ARGSORT` pass every new fork and aggregate.
- Exact target outputs, untouched inputs, canaries, scratch/state invariants, logical visitation,
  mapping, addition order, RNG mapping, and comparator behavior remain unchanged.
- Stable Class-File assertions prove the selected primitive hot form and retained typed
  general-long fallback where applicable.
- Generated/decompiled proved branches match optimal direct clean Java in semantic algorithm,
  control flow, hot-path dataflow, and avoidable-overhead profile. Any deviation is explicitly
  justified and measured.
- Repeated proved work contains no allocation, boxing, reflection, dynamic dispatch, generic-
  fallback overhead, or Synaptik-owned runtime call. Each target class has exactly one typed static
  entry and no fields, constructor, bridge, generic `Object`, method-handle/type constant,
  `invokedynamic`, dynamic constant, or bootstrap.
- If emitted bytes change, schema advances exactly once from 34 and every prior-schema envelope is
  an incompatible safe miss.
- Only the two named emitters change generated algorithm shape. Semantics, resources, routes,
  capability, public API, Prepare/Runtime behavior, architecture, dependencies, and build remain
  unchanged.
- A distinct clean documentation context independently finalizes Javadocs/package summaries,
  planning evidence/status, guide and glossary impact, and documentation validation.

## Tests / validation

Run the focused owners and one authoritative uncached module suite after executable Java stabilizes:

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
`-Xms1g -Xmx1g` forks. Retain raw and summarized results, two-target/control aggregates, all
generated classes, complete `javap -c -p` and `javap -v -p`, semantic/member/forbidden-reference
reports, environment, commands, and SHA-256 manifests in one new A1G evidence directory. A
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
schema propagation, immutable A1C sources, empty staged diff, one detailed Ready task, strict
correction order, clean-Java equivalence evidence, and worktree preservation. Repository-wide,
architecture, conformance, and integration suites remain deferred to CPU 0009 or CI because no
shared contract may change.

## Dependencies

- CPU 0007A1C supplies the immutable 20-row corpus and remains Incomplete.
- CPU 0007A1D supplies stable invocation-local segment layouts.
- CPU 0007A1E is Complete at schema 33.
- CPU 0007A1F is Complete at schema 34 and supplies the current five-fork residual evidence.
- Completed CPU 0006B2, CPU 0006D, CPU 0007A0D, and CPU 0007A0F define fold/dropout semantics,
  resources, and generated foundations to preserve.

## Follow-up tasks

- Other residual target groups remain Draft until A1G evidence supports the next bounded owner
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
  aggregate. Their small but repeated overages are bounded residuals.
- The rows use exactly two existing code-shaping owners, matching the correction-owner boundary.
- They remain one task because they are the next two residual ratios, have the same frozen
  protocol and evidence lifecycle, and together remain within the two-owner/task-size guardrails.
- No optimization may replace the exact sequential fold or V1 dropout work with an easier direct
  algorithm merely to pass the ratio gate.

## Known limitations

- Ratios apply only to the frozen cases, JVM, host, and protocol; they grant no universal parity,
  tuning, or routing claim.
- This task does not close CPU 0007A1C or authorize CPU 0007A2.
- Deferred rows may keep the full process nonzero and must remain explicitly diagnostic.

## Validation evidence

Planning/documentation context `01a01e83-bb41-7513-8b8a-9a17a0c15c3d` verified the A1F
`VERIFIED,20` result, all five target residual ratios above, generated-class inventory and hashes,
final disassembly, and the frozen direct comparator/owner mapping for both rows. No A1G executable
Java, test, schema, probe, architecture, API, or build change has been made.

## Implementation notes

Empty until implemented.

## Completion summary

Empty until implemented.
