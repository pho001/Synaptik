# Task 0006A1: Portable Static Window Extraction

## Status

Complete

## Goal

Extend the completed CPU 0006A portable movement foundation with represented-bit materialization
for exactly one fully static, resolved-layout `WindowTransformKind.UNFOLD_AXIS` or
`WindowTransformKind.UNFOLD2D` occurrence. `UNFOLD2D` preserves the Model signature order:
direct `Window2dAttrs` first, with conceptual positive-zero padding, and `Unfold2dAttrs` second,
with its exact typed `ScalarValue` padding.

The partition has one semantic node, one input, and one distinct injective output. CPU analysis
owns exact route selection, checked geometry, compact range-start state, and declarations;
finalization realizes one schema-13 portable scalar artifact after assignment. Runtime receives
only prepared direct carriers and never inspects `Operation`, `CompiledNode`, Shape, or layout.
`UNFOLD_AXIS` copies represented bits for all six current Model data types; both `UNFOLD2D`
variants copy only the current floating types `FLOAT64`, `FLOAT32`, and `BFLOAT16`, exactly as the
completed Model contracts permit. Execution is scalar or parallel-scalar. No fold/scatter
accumulation, vector/native route, tuning, general directed-acyclic-graph (DAG) work, dynamic
binding, or performance claim is added.

## Scope

### Exact semantic and occurrence matrix

| Kind | Exact attributes | Exact boundary | Result |
|---|---|---|---|
| `UNFOLD_AXIS` | `UnfoldAxisAttrs(axis, size, step)` | one same-typed input and output; `FLOAT64`, `FLOAT32`, `BFLOAT16`, `INT32`, `INT64`, or `BOOL` | materialized general-axis windows |
| `UNFOLD2D` | direct `Window2dAttrs` | one same-typed rank-four NCHW input and rank-three output; `FLOAT64`, `FLOAT32`, or `BFLOAT16` | im2col with conceptual positive zero |
| `UNFOLD2D` | `Unfold2dAttrs(window, paddingValue)` | one same-typed rank-four NCHW input and rank-three output; `FLOAT64`, `FLOAT32`, or `BFLOAT16`, with matching padding type | im2col with exact typed padding |

`UNFOLD_AXIS` admits exactly all six current types: `FLOAT64`, `FLOAT32`, `BFLOAT16`, `INT32`,
`INT64`, and `BOOL`. Both ordered `UNFOLD2D` variants admit exactly the three current floating
types: `FLOAT64`, `FLOAT32`, and `BFLOAT16`. Model 0017M defines floating 2D unfold behavior, and
Model 0023D explicitly leaves integral and Boolean `UNFOLD2D` out of scope; CPU must not broaden
that backend-independent matrix merely because extraction is represented-bit movement. Existing
canonical `0`/`1` input-byte validation therefore applies to the BOOL `UNFOLD_AXIS` row only.

Only one node, input occurrence, and output are admitted. `FOLD_AXIS`, `FOLD2D`, every wrong
attribute pairing, mixed types, unresolved layouts, dynamic dimensions, non-injective outputs,
and multi-node or mixed-family partitions fail closed.

### `UNFOLD_AXIS` geometry

Let static input rank be `R`, normalized axis `a`, selected extent `D`, positive size `S`, and
positive step `T`. Analysis proves `0 <= a < R`, `S <= D`, and checked arithmetic. The position
count is:

```text
P = floor((D - S) / T) + 1
```

The output rank is `R + 1`: input extents with axis `a` replaced by `P`, then final extent `S`.
For output coordinate `y[0..R]`, the exact source coordinate is:

```text
x[i] = y[i]                         when i != a
x[a] = y[a] * T + y[R]
```

Analysis checks the supplied output Shape exactly, including the maximum source coordinate, and
uses resolved input/output offsets and strides. There is no padding, dilation, image assumption,
or wrap. Scalar input is ineligible because it has no axis.

Zero work means an unaffected extent makes the checked output element count zero. Rank, axis,
size, step, shape, layouts, spans, and overflow are still validated; execution submits no
non-empty range and touches no carrier.

### `UNFOLD2D` geometry and order

Interpret input as `[N,C,H,W]`. Let kernel, stride, symmetric padding, and dilation be
`(kH,kW)`, `(sH,sW)`, `(pH,pW)`, and `(dH,dW)`. Kernel, stride, and dilation are positive;
padding is non-negative. Cold analysis uses checked `long` arithmetic:

```text
effectiveH = dH * (kH - 1) + 1
effectiveW = dW * (kW - 1) + 1
paddedH    = H + 2 * pH
paddedW    = W + 2 * pW
numeratorH = paddedH - effectiveH
numeratorW = paddedW - effectiveW
```

Both numerators must be non-negative. Floor mode is
`out = floor(numerator / stride) + 1`. Ceil mode is the checked integer form
`out = floor(numerator / stride) + (numerator % stride == 0 ? 0 : 1) + 1`; division and remainder
occur only on the cold path. Output Shape must be exactly:

```text
[N, C * kH * kW, outH * outW]
```

For output coordinate `[n,q,p]`, use this canonical order and mapping:

```text
q  = ((c * kH) + kh) * kW + kw      // kw fastest
p  = oh * outW + ow                 // ow fastest
ih = oh * sH - pH + kh * dH
iw = ow * sW - pW + kw * dW
```

If `0 <= ih < H` and `0 <= iw < W`, copy `[n,c,ih,iw]`; otherwise, including a terminal
ceil-grid sample beyond the padded extent, emit padding. For the three admitted floating types,
direct `Window2dAttrs` uses exact conceptual positive zero: `+0.0` bits for FLOAT64/FLOAT32 and
zero BFLOAT16 bits. `Unfold2dAttrs` accepts only matching FLOAT64/FLOAT32/BFLOAT16 padding and
extracts its exact primitive bits through the corresponding type-specific `ScalarValue` inspector,
reusing the completed movement `scalarBits` rule. Typed negative zero and NaN payloads survive
unchanged; no canonicalization occurs. There is no integral or BOOL `UNFOLD2D` padding row.

Zero work occurs when `N == 0` or `C == 0` after spatial geometry and overflow checks. `outH` and
`outW` remain positive checked Model results; zero work does not excuse invalid spatial geometry.

### Layout, compact geometry, and hot execution

Both Shapes are fully static and both layouts resolved. Input offset and non-negative strides,
including read-zero strides, are honored. Output offset and strides are honored only after the
completed movement injectivity proof succeeds. Output is a materialized distinct boundary whose
complete accessed span must not overlap input.

Extend `CpuNonAffineMovementLowering.Geometry`; do not create parallel prepared geometry. Its
common packer must support unequal input/output ranks by sizing each input stride block from that
input's rank while preserving existing PAD/TILE/CONCAT/STACK behavior. Add closed `UnfoldAxis`
and `Unfold2d` variants containing checked instance facts only. Never retain a selector, address,
or padding decision per output element.

Cold `Geometry.pack(...)` may divide/remainder to decode arbitrary range start. `UNFOLD2D` packs
initial `c/kh/kw` and `oh/ow` plus kernel/stride/padding/dilation and spatial facts; generated code
advances them with carry/reset odometers as flattened `q` and `p` advance. `UNFOLD_AXIS` uses the
output odometer plus `a/S/T`. The hot loop has no per-element division, modulo, allocation, map,
reflection, or semantic inspection.

### IR, lowering, reference, and emission

Extend `CpuDataMovementIr.MovementPlan` with `UnfoldAxisPlan` and `Unfold2dPlan`; do not add a
top-level portable IR family. Each plan has occurrence map `[0]`, one input access, one injective
output access, and the represented type. Structural identity records family, input/output rank
and access structure, and exact `UNFOLD2D` padding bits. Axis, size, step, extents, effective
kernels, stride, dilation, padding widths, output grid, layout offsets/stride magnitudes, and
range-start coordinates remain cold geometry.

Direct conceptual zero and typed positive zero lower to the same `Unfold2dPlan` and compatible
artifact identity when all other structural facts match. Attribute-class provenance is excluded
because bytecode is identical. Negative zero or a distinct NaN payload changes exact padding bits
and compatibility.

`CpuPartitionLowering` routes only exact single-node occurrences to the existing non-affine
movement lowerer. `CpuDataMovementEmitter` adds family branches using completed carrier and
odometer helpers. `CpuScalarReferenceKernel` independently evaluates the formulas for differential
tests; it is not a Runtime fallback.

### Preparation, strategy, schema, and key

Analysis declares exactly input then output, no workspace. Movement remains ineligible for
pointwise materialization and vector compute. Scalar handles one range or zero work;
parallel-scalar uses the completed deterministic disjoint-output policy. Vector preference falls
back to scalar. Finalization realizes exactly one selected artifact without revisiting semantics.

Generated bytecode changes, so advance `CpuGeneratorSchema.CURRENT_VERSION` exactly `12 -> 13`.
There is no migration reader; schema 12 and older envelopes are safe misses.

The key includes schema 13, movement family and rank/access structure, ordered types and carrier
pattern, scalar compute form, materialized position `-1`, and exact `UNFOLD2D` padding bits. It
excludes extents/counts, axis/size/step, kernel/stride/padding/dilation/grid/effective-kernel facts,
layout offsets/stride magnitudes, ranges/chunks/workers, carrier objects/byte offsets, slots,
addresses, `ValueId`, graph/run identity, and artifact-root state. Tests prove compatible cold
geometry shares identity/bytes and that family, rank/access, type/carrier, or padding-bit changes
do not.

### Failure behavior and stop conditions

Capability returns false and lowering fails closed for wrong signature/count/rank/type/padding,
dynamic Shape, unresolved layout, invalid axis, wrong output formula, non-injective output,
unsupported fold, or checked arithmetic/span overflow. Current static contracts allow all of
these decisions before invocation.

Cold binding retains rejection of wrong carrier/type, undersized/misaligned storage,
inaccessible/read-only output, non-canonical BOOL input for `UNFOLD_AXIS`, and input/output overlap
before writing. `UNFOLD2D` has no BOOL row. Extraction introduces no value-dependent execution
failure.

Stop before editing outside this task if implementation requires any Model, Compiler, Planning,
Prepare, Runtime, Config, Trace, Engine, dependency, Gradle, architecture, architecture-test,
backend-conformance, integration, or public API change; per-element prepared table; hot graph
inspection; second window representation; or shared alias/slot change. Report the exact missing
contract instead of inventing architecture.

## Out of scope

- `FOLD_AXIS`, `FOLD2D`, overlap/scatter accumulation, averaging, and duplicate-target policy.
- Gather/one-hot or value-dependent index validation.
- Asymmetric/nonconstant/negative padding; `UNFOLD_AXIS` padding/dilation; non-NCHW `UNFOLD2D`;
  or non-flattened columns.
- Integral or Boolean `UNFOLD2D`; the completed Model contracts admit only floating 2D unfold.
- Dynamic Shape/layout binding or unresolved output-layout selection.
- Vector/native/tuning/benchmark/fixed-shape/unrolled movement and performance claims.
- Multi-node/mixed-family partitions, general DAG decomposition/fusion, and multiple outputs.
- Shared/public/build/architecture/conformance/integration changes or the 0006A2 specification.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [`docs/architecture/current-architecture-plan.md`](../../../../architecture/current-architecture-plan.md)
- [`docs/architecture/module-boundaries.md`](../../../../architecture/module-boundaries.md)
- [`docs/architecture/dependency-rules.md`](../../../../architecture/dependency-rules.md)
- [`docs/architecture/lifecycle.md`](../../../../architecture/lifecycle.md)
- [`docs/architecture/runtime-prepare-backend-boundary.md`](../../../../architecture/runtime-prepare-backend-boundary.md)
- [`docs/planning/planning-guide.md`](../../../planning-guide.md)
- [`docs/developer-guide/documentation-rules.md`](../../../../developer-guide/documentation-rules.md)

## Architecture constraints

- Planning selects CPU only; CPU analysis/lowering owns realization, geometry, declarations, and
  specialization.
- Shared Prepare assigns exact buffers opaquely; CPU finalization realizes after assignment.
- Runtime sees only prepared direct invocation, never semantic/graph/layout/IR state.
- The boundary is one materialized window node, one input, and one distinct injective output.
- Scalar/parallel-scalar `UNFOLD_AXIS` copies all six current types; both `UNFOLD2D` variants copy
  only FLOAT64/FLOAT32/BFLOAT16. BFLOAT16 remains opaque represented-bit movement.
- Unsupported, dynamic, unresolved, overflowing, non-injective, or overlapping cases fail closed.

## Package impact

Existing packages remain responsible: public CPU capability; internal IR; lowering; codegen
emission; prepare; executable; reference; and cache. No package or top-level production type is
added. `CpuDataMovementIr`, `CpuNonAffineMovementLowering`, and `CpuDataMovementEmitter` remain
the family owners. Do not add a manager, registry, route, selector table, or duplicate geometry.

## Affected files

Authorized CPU production/package paths:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProvider.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratorSchema.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecialization.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuDataMovementEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutable.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuDataMovementIr.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuPortableKernelIr.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuNonAffineMovementLowering.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPartitionLowering.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparationPlan.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuScalarReferenceKernel.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/reference/package-info.java`

Authorized CPU test paths:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProviderTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratedKernelArtifactStoreTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecializationTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuDataMovementGeneratedKernelTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutableTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuDataMovementIrTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuNonAffineMovementLoweringTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuReferenceDifferentialTest.java`

Authorized documentation/planning paths:

- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`
- `docs/planning/backends/cpu/tasks/0006a1-portable-static-window-extraction.md`
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

No other path is authorized. Listed paths change only when their actual contract or regression
ownership is affected; record review-only no-change conclusions.

## Maximum scope

| Category | Maximum | Accounting |
|---|---:|---|
| CPU production/package | 20 | Existing movement/lifecycle/schema owners; no new type |
| CPU tests | 10 | Existing focused regression owners |
| Explanatory documentation | 2 | CPU guide and glossary |
| Planning/status | 3 | This task, master plan, roadmap |
| **Total** | **35** | **20 + 10 + 2 + 3** |

This exception is required because one generated family crosses truthful capability, sealed IR,
geometry, generation, preparation/finalization evidence, binding, reference, schema, Javadocs,
and explanatory documentation atomically. Splitting the two window forms repeats those seams and
leaves an arbitrary half-signature. No production type is added. Do not spend unused ceiling on
refactoring; stop and revise the plan before a 36th path.

## Acceptance criteria

- Capability admits all six current types for `UNFOLD_AXIS` and only FLOAT64/FLOAT32/BFLOAT16 for
  both ordered `UNFOLD2D` signatures; it rejects integral/BOOL 2D unfold plus every other
  wrong/dynamic/unresolved/fold/non-injective case.
- `UNFOLD_AXIS` proves the exact formula/mapping, checked geometry, arbitrary resolved layouts,
  and zero-work behavior.
- `UNFOLD2D` proves NCHW columns order, effective kernels, dilation, stride, floor/ceil grids,
  source/padding mapping, exact output Shape, overflow, and terminal ceil-grid padding.
- Direct zero and matching typed FLOAT64/FLOAT32/BFLOAT16 padding preserve exact representations,
  including signed-zero/NaN payload fixtures. Generated/reference `UNFOLD_AXIS` outputs agree
  bit-for-bit for all six types; `UNFOLD2D` outputs agree for the three floating types only.
- Compact unequal-rank geometry supports arbitrary ranges without per-element tables; hot loops
  use odometers with no per-element division/modulo/allocation/semantic inspection.
- Offset/strided/read-zero inputs and injective offset/strided outputs work; cold binding rejects
  carrier, geometry, `UNFOLD_AXIS` canonical-BOOL, and overlap failures before writes.
- Analysis declares input/output only, no workspace and one artifact; only scalar/parallel-scalar
  compute is selected and vector preference falls back.
- Existing movement, affine, pointwise, range, parallel, persistence, and carrier tests stay green.
- Schema advances exactly 12 to 13; schema 12 misses; key inclusion/exclusion tests include direct
  zero versus typed-zero equality and distinct padding-bit inequality.
- No excluded path/layer changes; a clean documentation pass finalizes all affected contracts and
  evidence after executable stabilization.

## Tests / validation

```bash
./gradlew :backends:cpu:test --tests 'io.github.pho001.synaptik.backend.cpu.CpuCapabilityProviderTest' --tests 'io.github.pho001.synaptik.backend.cpu.internal.ir.CpuDataMovementIrTest' --tests 'io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuNonAffineMovementLoweringTest' --tests 'io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuDataMovementGeneratedKernelTest' --tests 'io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest' --tests 'io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizerTest' --tests 'io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedExecutableTest' --tests 'io.github.pho001.synaptik.backend.cpu.internal.reference.CpuReferenceDifferentialTest' --tests 'io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecializationTest' --tests 'io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratedKernelArtifactStoreTest'
./gradlew :backends:cpu:test
```

The focused matrix covers axes/ranks/steps, NCHW order, floor/ceil/all-padding terminal windows,
offsets/strides, arbitrary flattened-boundary ranges, zero work, parallel chunks, all six
`UNFOLD_AXIS` types, only the three floating `UNFOLD2D` types, rejection of integral/BOOL 2D
unfold, matching FLOAT64/FLOAT32/BFLOAT16 explicit padding including adversarial bits,
overflow/failure, declarations, overlap-before-write, differential parity, and schema/key
behavior. Run the final CPU suite exactly once after Java stabilizes and record totals and Java 26
identity.

The distinct documentation pass then runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
```

It also validates local Markdown links/anchors, balanced fences, headings, final newlines,
trailing whitespace, exact allowlist/35-path ceiling, schema/status synchronization, no 0006A2
specification, and no Java/build change after reused evidence.

Repository-wide tests are unnecessary because this is CPU-private with no shared/build change;
CI and CPU 0009 own that checkpoint. Architecture tests are unnecessary because dependency and
hot-path boundaries do not change. Backend conformance is unnecessary because no cross-backend
claim is made and the CPU scalar reference provides differential evidence. Integration tests are
unnecessary because public Engine composition/output access is unchanged.

## Dependencies

- [CPU 0006A](0006a-portable-pad-tile-and-tensor-composition-movement.md) is `Complete`.
- [Model 0017M](../../../modules/model/tasks/0017m-unfold-and-fold-semantics.md),
  [Model 0017N](../../../modules/model/tasks/0017n-unfold-and-fold-tensor-expressions.md), and
  [Model 0023D](../../../modules/model/tasks/0023d-public-fold-axis-and-dynamic-window-transforms.md)
  are `Complete`.
- Current window kinds/attributes, `TensorWindowExpressions`, `Shape`, `LayoutDescriptor`,
  `OperationSignature`, `ScalarValue`, and completed CPU prepare/finalize/carrier/range/cache/bind
  contracts.

## Follow-up tasks

- CPU 0006A2 remains `Draft` without a detailed specification for Gather/one-hot and complete
  deterministic pre-write index validation.
- CPU 0006B remains `Draft` for functional update, scatter, `FOLD_AXIS`, and `FOLD2D` accumulation.
- CPU 0008A remains `Draft` for general DAG decomposition/fusion.
- CPU 0009 owns portable closure plus repository/conformance checkpoint.

## Architecture impact

Expected impact: None.

Existing architecture already assigns exact realization to CPU analysis/lowering, declarations
to staged preparation, artifacts to CPU finalization, and direct execution to Runtime. Unequal-rank
geometry is CPU-private. If implementation disproves this, stop at the first shared/architecture
need and report the exact gap.

## Implementation prompt

```text
You are the isolated implementation agent for Synaptik CPU task 0006A1. Do not commit or push.
Do not use any GSD skill, command, directory, artifact, or workflow.

Read in full AGENTS.md, ARCHITECTURE.md, current architecture plan, module/dependency/lifecycle/
runtime-prepare-backend boundary docs, planning guide/roadmap, CPU master plan, completed CPU 0006
and 0006A, this task, completed Model 0017M/0017N/0023D, and every current source/Javadoc/test,
CPU-guide, and glossary path named here.

Implement exactly one static resolved-layout UNFOLD_AXIS or UNFOLD2D node through existing
CpuDataMovementIr, CpuNonAffineMovementLowering.Geometry, CpuDataMovementEmitter, prepare/finalize/
executable, reference, and artifact seams. Preserve all six current types for UNFOLD_AXIS, only
FLOAT64/FLOAT32/BFLOAT16 for both UNFOLD2D variants, and rejection of integral/BOOL 2D unfold.
Preserve every formula, order, signature, matching floating padding bit, zero-work, overflow,
injectivity, overlap, scalar/parallel-scalar, compact unequal-rank, cold-decomposition, and
hot-odometer rule. Advance schema exactly 12 to 13 and apply the locked key inclusions/exclusions.

Do not add fold/scatter, Gather/one-hot, vector/native/tuning/general-DAG work, per-element tables,
shared/public/build/architecture/conformance/integration changes, 0006A2 planning, commits, or
pushes. Stop on such a need or a 35-path breach.

Run the focused ten-class command and sole final :backends:cpu:test after Java stabilizes. Then
hand the diff/evidence to a distinct clean documentation agent. That agent reads documentation
rules and applicable general/API-Javadoc/planning/example profiles, independently finalizes
Javadocs/package summaries, CPU guide, glossary, task/master/roadmap evidence, runs CPU Javadoc
and documentation/scope/status/whitespace gates, and reuses Java evidence unless executable code
changes or a concrete stale-evidence risk is identified.

Report completed changes, exact files, validation, unresolved issues, follow-up, context ID if
available, and finish with AGENTS.md `Status: Complete`, or `Status: Incomplete` and a specific
`Follow-up required:` line. Complete requires implementation, tests, documentation, and all gates.
```

## Local decisions

- Extend the existing movement family, not a new top-level IR or route.
- One input and one distinct injective materialized output are the partition boundary.
- Support all six represented types for `UNFOLD_AXIS`, but only FLOAT64/FLOAT32/BFLOAT16 for both
  `UNFOLD2D` variants; CPU does not broaden the completed Model semantic matrix.
- For floating-only `UNFOLD2D`, direct zero and matching typed positive zero share identity; exact
  padding bits shape compatibility.
- Unequal ranks use compact variable-rank geometry; start decomposition is cold, flattened state
  uses hot odometers.
- Static failures happen by prepare time; extraction adds no value-dependent partial-write policy.
- Schema 13 is mandatory for new bytecode; no migration.
- Extend existing owners, add no production abstraction, and keep the 35-path ceiling.

## Known limitations

- Only one fully static resolved-layout unfold occurrence is executable.
- Output is injective, distinct, and non-overlapping with input.
- `UNFOLD2D` is floating-only symmetric constant-padding canonical flattened NCHW; integral and
  BOOL rows remain unsupported.
- No fold/index/vector/native/dynamic/DAG/Engine/tuning/benchmark/performance result exists.
- Tensor expressions normally retain unresolved layouts; existing lifecycle work must supply exact
  descriptors. This task does not add layout selection.

## Validation evidence

Implementation context supplied the stabilized Java evidence:

- The required focused ten-class matrix passed 10 suites and 63 tests with zero failures, zero
  errors, and zero skipped tests.
- The sole final `./gradlew :backends:cpu:test` passed 32 suites and 163 tests with zero failures,
  zero errors, and one expected skipped opt-in persistence test.
- The toolchain was Java OpenJDK 26.0.1, build 26.0.1+8-34.
- Executable Java did not change after that final suite. The implementation advanced the private
  generated-artifact schema exactly from 12 to 13.

Clean documentation context `019ff061-4aa7-7f62-b2c5-7a1aed7b16f4` independently reviewed the
architecture, documentation profiles, task and dependency contracts, complete changed
production/test diff, current
Javadocs/package summaries, CPU guide, glossary, and relevant Model window contracts. It found no
implementation defect or architecture uncertainty and changed no executable Java behavior or
test. It therefore reused the stabilized Java evidence rather than rerunning Java tests.

- `./gradlew :backends:cpu:javadoc` passed after final Javadoc edits.
- Local Markdown target and anchor validation passed for every changed Markdown file; balanced
  fences, canonical headings/statuses, final newlines, and trailing-whitespace checks passed.
- `git diff --check` passed on the final combined change.
- Exact scope validation passed the task allowlist and 35-path ceiling. Schema/status checks
  confirmed schema 13 and synchronized 0006A1 `Complete` records.
- Inventory checks confirmed that no 0006A2 task specification exists and 0006A2 remains `Draft`.
- Forbidden-change checks confirmed no shared Model, Compiler, Planning, Prepare, Runtime, Config,
  Trace, or Engine source; Gradle/dependency; architecture/ADR/test; backend-conformance; or
  integration path changed.

## Implementation notes

- `CpuCapabilityProvider` now admits exactly one fully static resolved-layout `UNFOLD_AXIS` for
  all six represented types or one floating `UNFOLD2D` with either ordered attribute form.
- `CpuDataMovementIr`, `CpuNonAffineMovementLowering.Geometry`, generated emission, and the scalar
  reference implement compact unequal-rank axis/NCHW mapping. Exact two-dimensional padding bits
  shape artifact identity; concrete geometry remains cold.
- Preparation declares input then output with no workspace, scalar or parallel-scalar execution,
  and one schema-13 artifact. Existing cold binding rejects incompatible carriers, noncanonical
  BOOL input, and input/output overlap before writing.
- Package summaries remain accurate without modification: their existing movement, lowering,
  emission, preparation, executable, cache, reference, and public-CPU ownership descriptions
  already cover family extensions without enumerating a stale closed operation list.
- The two authorized tests not changed—`CpuKernelSpecializationTest` and
  `CpuReferenceDifferentialTest`—remain accurate because specialization/window identity and
  generated/reference parity are covered through the changed IR/generated-kernel owners; no
  separate assertion or fixture ownership moved into those files.
- `CpuKernelSpecialization`, `CpuPortableKernelIr`, `CpuPreparedExecutable`, and the preparation
  plan/preparer Javadocs remain accurate because the new family uses their existing generic
  structural-key, movement-geometry, binding, and lifecycle contracts without changing those
  APIs.

## Completion summary

CPU 0006A1 is complete. The portable CPU movement route now materializes exactly one fully static,
resolved-layout general-axis unfold for all six represented types or one floating canonical NCHW
two-dimensional unfold with conceptual positive-zero or exact typed padding. Compact unequal-rank
geometry, generated and scalar-reference mappings, scalar/parallel-scalar orchestration, exact
schema-13 compatibility, one distinct injective output, and fail-closed exclusions are documented
and validated. No unresolved issue remains. CPU 0006A2 and later work remain Draft without a
detailed 0006A2 specification.

- Completed changes: portable static window extraction plus finalized Javadocs, CPU guide,
  glossary, task evidence, CPU master plan, and roadmap.
- Files changed or created: the exact final allowlisted paths recorded by repository status.
- Tests and validation: reused 10-suite/63-test focused and 32-suite/163-test final CPU evidence;
  CPU Javadoc and all requested documentation/scope/status/schema/whitespace gates passed.
- Documentation-agent review: clean context `019ff061-4aa7-7f62-b2c5-7a1aed7b16f4`; complete
  with no executable Java changes.
- Documentation impact: CPU guide and planning/status records now describe current window support.
- Javadoc review: affected contract Javadocs finalized; unchanged authorized contracts reviewed.
- Glossary impact: the existing CPU static-data-movement term now includes window extraction.
- Unresolved issues: None.
- Follow-up required: None.

Status: Complete
