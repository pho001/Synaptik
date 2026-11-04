# Task 0006A1: Portable Static Window Extraction

## Reader context

Unfolding extracts sliding windows into an explicit result; it is not overlap accumulation or
pooling. Model attributes described the axis or image-window geometry, and CPU analysis derived
compact source mappings before assigning the input and distinct output buffers. The generated
scalar body copied each selected representation, including the exact padding bits required by
the chosen UNFOLD2D attribute form.

The image notation NCHW means batch, channel, height, and width; the columns result groups
window positions for later consumers. The original BFLOAT16 support was raw movement, not newly
granted arithmetic, and the historical six-type/three-floating-type matrix remains below.
Current window-transform semantics explain the mapping independently of CPU’s selected carriers
and strategy.

Current reading alongside this historical record:

- [Window-transform expressions](../../../../reference/tensor/windows.md#window-transform-expressions)
  — distinguishes extraction, columns geometry, and overlap accumulation.
- [CPU partition and unit lowering](../../../../explanation/packages/cpu.md#iogithubpho001synaptikbackendcpuinternallowering)
  — explains virtual intermediates, boundary stores, and family-specific mappings.

## Status

Complete

## Goal

Extend the completed [CPU](../../../../glossary.md#term-cpu) 0006A portable movement foundation with [represented-bit](../../../../glossary.md#storage-mapping-contiguity-and-injectivity) [materialization](../../../../glossary.md#materialization)
for exactly one fully static, resolved-[layout](../../../../glossary.md#layout) `WindowTransformKind.UNFOLD_AXIS` or
`WindowTransformKind.UNFOLD2D` [occurrence](../../../../glossary.md#term-occurrence). `UNFOLD2D` preserves the [Model](../../../../glossary.md#term-model) [signature](../../../../glossary.md#term-signature) order:
direct `Window2dAttrs` first, with conceptual positive-zero [padding](../../../../glossary.md#padding), and `Unfold2dAttrs` second,
with its exact typed [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue) padding.

The [partition](../../../../glossary.md#partition) has one semantic [node](../../../../glossary.md#node), one input, and one distinct [injective](../../../../glossary.md#term-injective) output. CPU analysis
owns exact route selection, checked geometry, compact range-start state, and declarations;
finalization realizes one [schema](../../../../glossary.md#term-schema)-13 portable [scalar](../../../../glossary.md#term-scalar) [artifact](../../../../glossary.md#term-artifact) after assignment. [Runtime](../../../../glossary.md#term-runtime) receives
only prepared direct carriers and never inspects [`Operation`](../../../../glossary.md#operation), `CompiledNode`, [Shape](../../../../glossary.md#shape), or layout.
`UNFOLD_AXIS` copies [represented bits](../../../../glossary.md#term-represented-bits) for all six current Model data types; both `UNFOLD2D`
variants copy only the current floating types [`FLOAT64`](../../../../glossary.md#term-float64), [`FLOAT32`](../../../../glossary.md#term-float32), and [`BFLOAT16`](../../../../glossary.md#term-bfloat16), exactly as the
completed Model contracts permit. Execution is scalar or parallel-scalar. No fold/scatter
accumulation, [vector](../../../../glossary.md#term-vector)/native route, tuning, general directed-acyclic-[graph](../../../../glossary.md#graph) ([DAG](../../../../glossary.md#term-directed-acyclic-graph-dag)) work, dynamic
[binding](../../../../glossary.md#term-binding), or performance claim is added.

## [Scope](../../../../glossary.md#term-scope)

### Exact semantic and [occurrence](../../../../glossary.md#term-occurrence) [matrix](../../../../glossary.md#term-matrix-meanings)

| Kind | Exact attributes | Exact boundary | Result |
|---|---|---|---|
| `UNFOLD_AXIS` | `UnfoldAxisAttrs(axis, size, step)` | one same-typed input and output; [`FLOAT64`](../../../../glossary.md#term-float64), [`FLOAT32`](../../../../glossary.md#term-float32), [`BFLOAT16`](../../../../glossary.md#term-bfloat16), [`INT32`](../../../../glossary.md#data-type--datatype), [`INT64`](../../../../glossary.md#data-type--datatype), or [`BOOL`](../../../../glossary.md#data-type--datatype) | materialized general-[axis](../../../../glossary.md#term-axis) windows |
| `UNFOLD2D` | direct `Window2dAttrs` | one same-typed [rank](../../../../glossary.md#term-rank)-four [NCHW](../../../../glossary.md#term-nchw) input and rank-three output; `FLOAT64`, `FLOAT32`, or `BFLOAT16` | [im2col](../../../../glossary.md#term-im2col) with conceptual [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero) |
| `UNFOLD2D` | `Unfold2dAttrs(window, paddingValue)` | one same-typed rank-four NCHW input and rank-three output; `FLOAT64`, `FLOAT32`, or `BFLOAT16`, with matching [padding](../../../../glossary.md#padding) type | im2col with exact typed padding |

`UNFOLD_AXIS` admits exactly all six current types: `FLOAT64`, `FLOAT32`, `BFLOAT16`, `INT32`,
`INT64`, and `BOOL`. Both ordered `UNFOLD2D` variants admit exactly the three current floating
types: `FLOAT64`, `FLOAT32`, and `BFLOAT16`. [Model](../../../../glossary.md#term-model) 0017M defines floating 2D unfold behavior, and
Model 0023D explicitly leaves integral and [Boolean](../../../../glossary.md#data-type--datatype) `UNFOLD2D` out of [scope](../../../../glossary.md#term-scope); [CPU](../../../../glossary.md#term-cpu) must not broaden
that [backend](../../../../glossary.md#backend)-independent matrix merely because extraction is [represented-bit](../../../../glossary.md#storage-mapping-contiguity-and-injectivity) movement. Existing
[canonical](../../../../glossary.md#term-canonical-form) `0`/`1` input-byte [validation](../../../../glossary.md#term-validation) therefore applies to the BOOL `UNFOLD_AXIS` row only.

Only one [node](../../../../glossary.md#node), input occurrence, and output are admitted. `FOLD_AXIS`, `FOLD2D`, every wrong
attribute pairing, mixed types, unresolved [layouts](../../../../glossary.md#layout), dynamic [dimensions](../../../../glossary.md#dimension), non-[injective](../../../../glossary.md#term-injective) outputs,
and multi-node or mixed-family partitions [fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation).

### `UNFOLD_AXIS` geometry

Let static input [rank](../../../../glossary.md#term-rank) be `R`, [normalized axis](../../../../glossary.md#normalized-axis) `a`, selected [extent](../../../../glossary.md#term-extent) `D`, positive size `S`, and
positive step `T`. Analysis proves `0 <= a < R`, `S <= D`, and checked arithmetic. The position
count is:

```text
P = floor((D - S) / T) + 1
```

The output rank is `R + 1`: input extents with [axis](../../../../glossary.md#term-axis) `a` replaced by `P`, then final extent `S`.
For output [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) `y[0..R]`, the exact source coordinate is:

```text
x[i] = y[i]                         when i != a
x[a] = y[a] * T + y[R]
```

Analysis checks the supplied output [Shape](../../../../glossary.md#shape) exactly, including the maximum source coordinate, and
uses resolved input/output offsets and [strides](../../../../glossary.md#term-stride). There is no [padding](../../../../glossary.md#padding), [dilation](../../../../glossary.md#term-dilation), image assumption,
or wrap. [Scalar](../../../../glossary.md#term-scalar) input is ineligible because it has no axis.

Zero work means an unaffected extent makes the checked output [element](../../../../glossary.md#term-element) count zero. Rank, axis,
size, step, shape, [layouts](../../../../glossary.md#layout), spans, and [overflow](../../../../glossary.md#term-overflow) are still validated; execution submits no
non-empty range and touches no [carrier](../../../../glossary.md#term-storage-carrier).

### `UNFOLD2D` geometry and order

Interpret input as `[N,C,H,W]`. Let [kernel](../../../../glossary.md#kernel), [stride](../../../../glossary.md#term-stride), symmetric [padding](../../../../glossary.md#padding), and [dilation](../../../../glossary.md#term-dilation) be
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
occur only on the [cold path](../../../../glossary.md#term-cold-path). Output [Shape](../../../../glossary.md#shape) must be exactly:

```text
[N, C * kH * kW, outH * outW]
```

For output [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) `[n,q,p]`, use this [canonical](../../../../glossary.md#term-canonical-form) order and mapping:

```text
q  = ((c * kH) + kh) * kW + kw      // kw fastest
p  = oh * outW + ow                 // ow fastest
ih = oh * sH - pH + kh * dH
iw = ow * sW - pW + kw * dW
```

If `0 <= ih < H` and `0 <= iw < W`, copy `[n,c,ih,iw]`; otherwise, including a terminal
ceil-grid [sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) beyond the padded [extent](../../../../glossary.md#term-extent), emit padding. For the three admitted floating types,
direct `Window2dAttrs` uses exact conceptual [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero): `+0.0` bits for [FLOAT64](../../../../glossary.md#term-float64)/[FLOAT32](../../../../glossary.md#term-float32) and
zero [BFLOAT16](../../../../glossary.md#term-bfloat16) bits. `Unfold2dAttrs` accepts only matching FLOAT64/FLOAT32/BFLOAT16 padding and
extracts its exact primitive bits through the corresponding type-specific [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue) inspector,
reusing the completed movement `scalarBits` rule. Typed [negative zero](../../../../glossary.md#nan-infinity-and-signed-zero) and [NaN](../../../../glossary.md#term-nan) payloads survive
unchanged; no [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form) occurs. There is no integral or [BOOL](../../../../glossary.md#data-type--datatype) `UNFOLD2D` padding row.

Zero work occurs when `N == 0` or `C == 0` after [spatial](../../../../glossary.md#term-spatial-axes) geometry and [overflow](../../../../glossary.md#term-overflow) checks. `outH` and
`outW` remain positive checked [Model](../../../../glossary.md#term-model) results; zero work does not excuse invalid spatial geometry.

### [Layout](../../../../glossary.md#layout), compact geometry, and hot execution

Both [Shapes](../../../../glossary.md#shape) are fully static and both layouts resolved. Input offset and non-negative [strides](../../../../glossary.md#term-stride),
including read-zero strides, are honored. Output offset and strides are honored only after the
completed movement [injectivity](../../../../glossary.md#storage-mapping-contiguity-and-injectivity) proof succeeds. Output is a materialized distinct boundary whose
complete accessed span must not overlap input.

Extend `CpuNonAffineMovementLowering.Geometry`; do not create parallel prepared geometry. Its
common packer must support unequal input/output ranks by sizing each input stride block from that
input's [rank](../../../../glossary.md#term-rank) while preserving existing [PAD](../../../../glossary.md#padding)/TILE/CONCAT/STACK behavior. Add closed `UnfoldAxis`
and `Unfold2d` variants containing checked instance facts only. Never retain a selector, address,
or [padding](../../../../glossary.md#padding) decision per output [element](../../../../glossary.md#term-element).

Cold `Geometry.pack(...)` may divide/remainder to decode arbitrary range start. `UNFOLD2D` packs
initial `c/kh/kw` and `oh/ow` plus [kernel](../../../../glossary.md#kernel)/stride/padding/[dilation](../../../../glossary.md#term-dilation) and [spatial](../../../../glossary.md#term-spatial-axes) facts; generated code
advances them with [carry](../../../../glossary.md#term-carry)/reset odometers as flattened `q` and `p` advance. `UNFOLD_AXIS` uses the
output odometer plus `a/S/T`. The hot loop has no per-element division, modulo, allocation, map,
reflection, or semantic inspection.

### [IR](../../../../glossary.md#term-intermediate-representation-ir), [lowering](../../../../glossary.md#lowering), reference, and emission

Extend `CpuDataMovementIr.MovementPlan` with `UnfoldAxisPlan` and `Unfold2dPlan`; do not add a
top-level portable IR family. Each plan has [occurrence](../../../../glossary.md#term-occurrence) map `[0]`, one input access, one [injective](../../../../glossary.md#term-injective)
output access, and the represented type. Structural [identity](../../../../glossary.md#term-identity-meanings) records family, input/output [rank](../../../../glossary.md#term-rank)
and access structure, and exact `UNFOLD2D` [padding](../../../../glossary.md#padding) bits. [Axis](../../../../glossary.md#term-axis), size, step, extents, effective
[kernels](../../../../glossary.md#kernel), [stride](../../../../glossary.md#term-stride), [dilation](../../../../glossary.md#term-dilation), padding widths, output grid, [layout](../../../../glossary.md#layout) offsets/stride magnitudes, and
range-start [coordinates](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) remain cold geometry.

Direct conceptual zero and typed [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero) lower to the same `Unfold2dPlan` and compatible
[artifact](../../../../glossary.md#term-artifact) identity when all other structural facts match. Attribute-class [provenance](../../../../glossary.md#provenance) is excluded
because [bytecode](../../../../glossary.md#term-bytecode) is identical. [Negative zero](../../../../glossary.md#nan-infinity-and-signed-zero) or a distinct [NaN payload](../../../../glossary.md#nan-infinity-and-signed-zero) changes exact padding bits
and compatibility.

`CpuPartitionLowering` routes only exact single-[node](../../../../glossary.md#node) occurrences to the existing non-affine
movement lowerer. `CpuDataMovementEmitter` adds family branches using completed [carrier](../../../../glossary.md#term-storage-carrier) and
odometer helpers. `CpuScalarReferenceKernel` independently evaluates the formulas for differential
tests; it is not a [Runtime](../../../../glossary.md#term-runtime) fallback.

### Preparation, strategy, [schema](../../../../glossary.md#term-schema), and key

Analysis declares exactly input then output, no workspace. Movement remains ineligible for
[pointwise](../../../../glossary.md#term-pointwise) [materialization](../../../../glossary.md#materialization) and [vector](../../../../glossary.md#term-vector) compute. [Scalar](../../../../glossary.md#term-scalar) handles one range or zero work;
parallel-scalar uses the completed deterministic disjoint-output policy. Vector preference falls
back to scalar. Finalization realizes exactly one selected [artifact](../../../../glossary.md#term-artifact) without revisiting semantics.

Generated [bytecode](../../../../glossary.md#term-bytecode) changes, so advance `CpuGeneratorSchema.CURRENT_VERSION` exactly `12 -> 13`.
There is no migration reader; schema 12 and older envelopes are safe misses.

The key includes schema 13, movement family and [rank](../../../../glossary.md#term-rank)/access structure, ordered types and [carrier](../../../../glossary.md#term-storage-carrier)
pattern, scalar compute form, materialized position `-1`, and exact `UNFOLD2D` [padding](../../../../glossary.md#padding) bits. It
excludes extents/counts, [axis](../../../../glossary.md#term-axis)/size/step, [kernel](../../../../glossary.md#kernel)/[stride](../../../../glossary.md#term-stride)/padding/[dilation](../../../../glossary.md#term-dilation)/grid/effective-kernel facts,
[layout](../../../../glossary.md#layout) offsets/stride magnitudes, ranges/chunks/workers, carrier objects/byte offsets, slots,
addresses, [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot), [graph](../../../../glossary.md#graph)/[run](../../../../glossary.md#term-run) [identity](../../../../glossary.md#term-identity-meanings), and artifact-root state. Tests prove compatible cold
geometry shares identity/bytes and that family, rank/access, type/carrier, or padding-bit changes
do not.

### Failure behavior and stop conditions

Capability returns false and [lowering](../../../../glossary.md#lowering) fails closed for wrong [signature](../../../../glossary.md#term-signature)/count/[rank](../../../../glossary.md#term-rank)/type/[padding](../../../../glossary.md#padding),
dynamic [Shape](../../../../glossary.md#shape), unresolved [layout](../../../../glossary.md#layout), invalid [axis](../../../../glossary.md#term-axis), wrong output formula, non-[injective](../../../../glossary.md#term-injective) output,
unsupported fold, or checked arithmetic/span [overflow](../../../../glossary.md#term-overflow). Current static contracts allow all of
these decisions before invocation.

Cold [binding](../../../../glossary.md#term-binding) retains rejection of wrong [carrier](../../../../glossary.md#term-storage-carrier)/type, undersized/misaligned storage,
inaccessible/read-only output, non-[canonical](../../../../glossary.md#term-canonical-form) [BOOL](../../../../glossary.md#data-type--datatype) input for `UNFOLD_AXIS`, and input/output overlap
before writing. `UNFOLD2D` has no BOOL row. Extraction introduces no value-dependent execution
failure.

Stop before editing outside this task if implementation requires any [Model](../../../../glossary.md#term-model), [Compiler](../../../../glossary.md#term-compiler), [Planning](../../../../glossary.md#term-planning),
[Prepare](../../../../glossary.md#prepare), [Runtime](../../../../glossary.md#term-runtime), Config, [Trace](../../../../glossary.md#trace), [Engine](../../../../glossary.md#term-engine), [dependency](../../../../glossary.md#term-dependency), [Gradle](../../../../glossary.md#term-gradle), architecture, architecture-test,
[backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance), integration, or public [API](../../../../glossary.md#term-application-programming-interface-api) change; per-[element](../../../../glossary.md#term-element) prepared table; hot [graph](../../../../glossary.md#graph)
inspection; second window representation; or shared [alias](../../../../glossary.md#aliasing-and-arena)/slot change. Report the exact missing
contract instead of inventing architecture.

## Out of [scope](../../../../glossary.md#term-scope)

- `FOLD_AXIS`, `FOLD2D`, overlap/scatter accumulation, averaging, and duplicate-[target](../../../../glossary.md#term-target) policy.
- [Gather](../../../../glossary.md#gather)/one-hot or value-dependent index [validation](../../../../glossary.md#term-validation).
- Asymmetric/nonconstant/negative [padding](../../../../glossary.md#padding); `UNFOLD_AXIS` padding/[dilation](../../../../glossary.md#term-dilation); non-[NCHW](../../../../glossary.md#term-nchw) `UNFOLD2D`;
  or non-flattened columns.
- Integral or [Boolean](../../../../glossary.md#data-type--datatype) `UNFOLD2D`; the completed [Model](../../../../glossary.md#term-model) contracts admit only floating 2D unfold.
- Dynamic [Shape](../../../../glossary.md#shape)/[layout](../../../../glossary.md#layout) [binding](../../../../glossary.md#term-binding) or unresolved output-layout selection.
- [Vector](../../../../glossary.md#term-vector)/native/tuning/benchmark/fixed-shape/[unrolled](../../../../glossary.md#backpropagation-through-time-and-recurrent-state) movement and performance claims.
- Multi-[node](../../../../glossary.md#node)/mixed-family partitions, general [DAG](../../../../glossary.md#term-directed-acyclic-graph-dag) [decomposition](../../../../glossary.md#term-decomposition)/[fusion](../../../../glossary.md#term-fusion), and multiple outputs.
- Shared/public/build/architecture/[conformance](../../../../glossary.md#term-conformance)/integration changes or the 0006A2 specification.

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
  specialization.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [lowering glossary](../../../../glossary.md#lowering); [specialization glossary](../../../../glossary.md#term-specialization))</span>
- Shared [Prepare](../../../../glossary.md#prepare) assigns exact [buffers](../../../../glossary.md#term-buffer) opaquely; [CPU](../../../../glossary.md#term-cpu) finalization realizes after assignment.
- [Runtime](../../../../glossary.md#term-runtime) sees only prepared direct invocation, never semantic/[graph](../../../../glossary.md#graph)/[layout](../../../../glossary.md#layout)/[IR](../../../../glossary.md#term-intermediate-representation-ir) state.
- The boundary is one materialized window [node](../../../../glossary.md#node), one input, and one distinct [injective](../../../../glossary.md#term-injective) output.
- [Scalar](../../../../glossary.md#term-scalar)/parallel-scalar `UNFOLD_AXIS` copies all six current types; both `UNFOLD2D` variants copy
  only FLOAT64/FLOAT32/BFLOAT16. BFLOAT16 remains opaque represented-bit movement.<span data-glossary-navigation="true"> ([FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [opaque glossary](../../../../glossary.md#term-opaque); [represented-bit glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity))</span>
- Unsupported, dynamic, unresolved, overflowing, non-injective, or overlapping cases [fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation).

## Package impact

Existing packages remain responsible: public [CPU](../../../../glossary.md#term-cpu) capability; internal [IR](../../../../glossary.md#term-intermediate-representation-ir); [lowering](../../../../glossary.md#lowering); codegen
emission; [prepare](../../../../glossary.md#prepare); executable; reference; and [cache](../../../../glossary.md#term-cache). No package or top-level production type is
added. `CpuDataMovementIr`, `CpuNonAffineMovementLowering`, and `CpuDataMovementEmitter` remain
the family [owners](../../../../glossary.md#term-owner). Do not add a manager, [registry](../../../../glossary.md#term-registry), route, selector table, or duplicate geometry.

## Affected files

Authorized [CPU](../../../../glossary.md#term-cpu) production/package paths:

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

Authorized documentation/[planning](../../../../glossary.md#term-planning) paths:

- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`
- `docs/planning/backends/cpu/tasks/0006a1-portable-static-window-extraction.md`
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

No other path is authorized. Listed paths change only when their actual contract or [regression](../../../../glossary.md#term-regression)
[ownership](../../../../glossary.md#term-ownership-meanings) is affected; record review-only no-change conclusions.

## Maximum [scope](../../../../glossary.md#term-scope)

| Category | Maximum | Accounting |
|---|---:|---|
| [CPU](../../../../glossary.md#term-cpu) production/package | 20 | Existing movement/[lifecycle](../../../../glossary.md#lifecycle)/[schema](../../../../glossary.md#term-schema) [owners](../../../../glossary.md#term-owner); no new type |
| CPU tests | 10 | Existing focused [regression](../../../../glossary.md#term-regression) owners |
| Explanatory documentation | 2 | CPU guide and glossary |
| [Planning](../../../../glossary.md#term-planning)/status | 3 | This task, master plan, roadmap |
| **Total** | **35** | **20 + 10 + 2 + 3** |

This exception is required because one generated family crosses truthful capability, [sealed](../../../../glossary.md#term-sealed) [IR](../../../../glossary.md#term-intermediate-representation-ir),
geometry, generation, preparation/finalization evidence, [binding](../../../../glossary.md#term-binding), reference, schema, Javadocs,
and explanatory documentation atomically. Splitting the two window forms repeats those seams and
leaves an arbitrary half-[signature](../../../../glossary.md#term-signature). No production type is added. Do not spend unused ceiling on
refactoring; stop and revise the plan before a 36th path.

## Acceptance criteria

- Capability admits all six current types for `UNFOLD_AXIS` and only FLOAT64/FLOAT32/BFLOAT16 for
  both ordered <span data-glossary-navigation="true"> ([FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16))</span>`UNFOLD2D` signatures; it rejects integral/BOOL 2D unfold plus every other
  wrong/dynamic/unresolved/fold/non-injective case.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature); [injective glossary](../../../../glossary.md#term-injective))</span>
- `UNFOLD_AXIS` proves the exact formula/mapping, checked geometry, arbitrary resolved layouts,
  and zero-work behavior.<span data-glossary-navigation="true"> ([layouts glossary](../../../../glossary.md#layout))</span>
- `UNFOLD2D` proves NCHW columns order, effective kernels, dilation, stride, floor/ceil grids,
  source/padding mapping, exact output Shape, overflow, and terminal ceil-grid padding.<span data-glossary-navigation="true"> ([NCHW glossary](../../../../glossary.md#term-nchw); [kernels glossary](../../../../glossary.md#kernel); [dilation glossary](../../../../glossary.md#term-dilation); [stride glossary](../../../../glossary.md#term-stride); [padding glossary](../../../../glossary.md#padding); [Shape glossary](../../../../glossary.md#shape); [overflow glossary](../../../../glossary.md#term-overflow))</span>
- Direct zero and matching typed FLOAT64/FLOAT32/BFLOAT16 padding preserve exact representations,
  including signed-zero/NaN payload fixtures. Generated/reference <span data-glossary-navigation="true"> ([signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [NaN payload glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>`UNFOLD_AXIS` outputs agree
  bit-for-bit for all six types; `UNFOLD2D` outputs agree for the three floating types only.
- Compact unequal-rank geometry supports arbitrary ranges without per-element tables; hot loops
  use odometers with no per-element division/modulo/allocation/semantic inspection.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [element glossary](../../../../glossary.md#term-element))</span>
- Offset/strided/read-zero inputs and injective offset/strided outputs work; cold binding rejects
  carrier, geometry, <span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>`UNFOLD_AXIS` [canonical](../../../../glossary.md#term-canonical-form)-[BOOL](../../../../glossary.md#data-type--datatype), and overlap failures before writes.
- Analysis declares input/output only, no workspace and one artifact; only scalar/parallel-scalar
  compute is selected and vector preference falls back.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact); [scalar glossary](../../../../glossary.md#term-scalar); [vector glossary](../../../../glossary.md#term-vector))</span>
- Existing movement, affine, [pointwise](../../../../glossary.md#term-pointwise), range, parallel, [persistence](../../../../glossary.md#term-persistence), and [carrier](../../../../glossary.md#term-storage-carrier) tests stay green.
- Schema advances exactly 12 to 13; schema 12 misses; key inclusion/exclusion tests include direct
  zero versus typed-zero equality and distinct padding-bit inequality.<span data-glossary-navigation="true"> ([Schema glossary](../../../../glossary.md#term-schema))</span>
- No excluded path/layer changes; a clean documentation pass finalizes all affected contracts and
  evidence after executable stabilization.

## Tests / [validation](../../../../glossary.md#term-validation)

```bash
./gradlew :backends:cpu:test --tests 'io.github.pho001.synaptik.backend.cpu.CpuCapabilityProviderTest' --tests 'io.github.pho001.synaptik.backend.cpu.internal.ir.CpuDataMovementIrTest' --tests 'io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuNonAffineMovementLoweringTest' --tests 'io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuDataMovementGeneratedKernelTest' --tests 'io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest' --tests 'io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizerTest' --tests 'io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedExecutableTest' --tests 'io.github.pho001.synaptik.backend.cpu.internal.reference.CpuReferenceDifferentialTest' --tests 'io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecializationTest' --tests 'io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratedKernelArtifactStoreTest'
./gradlew :backends:cpu:test
```

The focused [matrix](../../../../glossary.md#term-matrix-meanings) covers [axes](../../../../glossary.md#term-axis)/ranks/steps, [NCHW](../../../../glossary.md#term-nchw) order, floor/ceil/all-[padding](../../../../glossary.md#padding) terminal windows,
offsets/[strides](../../../../glossary.md#term-stride), arbitrary flattened-boundary ranges, zero work, parallel chunks, all six
`UNFOLD_AXIS` types, only the three floating `UNFOLD2D` types, rejection of integral/[BOOL](../../../../glossary.md#data-type--datatype) 2D
unfold, matching [FLOAT64](../../../../glossary.md#term-float64)/[FLOAT32](../../../../glossary.md#term-float32)/[BFLOAT16](../../../../glossary.md#term-bfloat16) explicit padding including adversarial bits,
[overflow](../../../../glossary.md#term-overflow)/failure, declarations, overlap-before-write, differential parity, and [schema](../../../../glossary.md#term-schema)/key
behavior. [Run](../../../../glossary.md#term-run) the final [CPU](../../../../glossary.md#term-cpu) suite exactly once after Java stabilizes and record totals and Java 26
[identity](../../../../glossary.md#term-identity-meanings).

The distinct documentation pass then runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
```

It also validates local Markdown links/anchors, balanced fences, headings, final newlines,
trailing whitespace, exact allowlist/35-path ceiling, schema/status [synchronization](../../../../glossary.md#term-synchronization), no 0006A2
specification, and no Java/build change after reused evidence.

Repository-wide tests are unnecessary because this is CPU-private with no shared/build change;
[CI](../../../../glossary.md#term-continuous-integration-ci) and CPU 0009 own that [checkpoint](../../../../glossary.md#term-checkpoint). Architecture tests are unnecessary because [dependency](../../../../glossary.md#term-dependency) and
hot-path boundaries do not change. [Backend](../../../../glossary.md#backend) [conformance](../../../../glossary.md#term-conformance) is unnecessary because no cross-backend
claim is made and the CPU [scalar](../../../../glossary.md#term-scalar) reference provides differential evidence. Integration tests are
unnecessary because public [Engine](../../../../glossary.md#term-engine) composition/output access is unchanged.

## Dependencies

- [CPU 0006A](0006a-portable-pad-tile-and-tensor-composition-movement.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span> is `Complete`.
- [Model 0017M](../../../modules/model/tasks/0017m-unfold-and-fold-semantics.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>,
  [Model 0017N](../../../modules/model/tasks/0017n-unfold-and-fold-tensor-expressions.md), and
  [Model 0023D](../../../modules/model/tasks/0023d-public-fold-axis-and-dynamic-window-transforms.md)
  are `Complete`.
- Current window kinds/attributes, `TensorWindowExpressions`, [`Shape`](../../../../glossary.md#shape), `LayoutDescriptor`,
  `OperationSignature`, [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue), and completed CPU prepare/finalize/carrier/range/cache/bind
  contracts.<span data-glossary-navigation="true"> ([prepare glossary](../../../../glossary.md#prepare); [carrier glossary](../../../../glossary.md#term-storage-carrier); [cache glossary](../../../../glossary.md#term-cache))</span>

## Follow-up tasks

- [CPU](../../../../glossary.md#term-cpu) 0006A2 remains `Draft` without a detailed specification for Gather/one-hot and complete
  deterministic pre-write index validation.<span data-glossary-navigation="true"> ([Gather glossary](../../../../glossary.md#gather); [validation glossary](../../../../glossary.md#term-validation))</span>
- CPU 0006B remains `Draft` for functional update, scatter, `FOLD_AXIS`, and `FOLD2D` accumulation.
- CPU 0008A remains `Draft` for general [DAG](../../../../glossary.md#term-directed-acyclic-graph-dag) [decomposition](../../../../glossary.md#term-decomposition)/[fusion](../../../../glossary.md#term-fusion).
- CPU 0009 owns portable closure plus repository/[conformance](../../../../glossary.md#term-conformance) [checkpoint](../../../../glossary.md#term-checkpoint).

## Architecture impact

Expected impact: None.

Existing architecture already assigns exact realization to [CPU](../../../../glossary.md#term-cpu) analysis/[lowering](../../../../glossary.md#lowering), declarations
to staged preparation, [artifacts](../../../../glossary.md#term-artifact) to CPU finalization, and direct execution to [Runtime](../../../../glossary.md#term-runtime). Unequal-[rank](../../../../glossary.md#term-rank)
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

- Extend the existing movement family, not a new top-level [IR](../../../../glossary.md#term-intermediate-representation-ir) or route.
- One input and one distinct [injective](../../../../glossary.md#term-injective) materialized output are the [partition](../../../../glossary.md#partition) boundary.
- Support all six represented types for `UNFOLD_AXIS`, but only [FLOAT64](../../../../glossary.md#term-float64)/[FLOAT32](../../../../glossary.md#term-float32)/[BFLOAT16](../../../../glossary.md#term-bfloat16) for both
  `UNFOLD2D` variants; [CPU](../../../../glossary.md#term-cpu) does not broaden the completed [Model](../../../../glossary.md#term-model) semantic [matrix](../../../../glossary.md#term-matrix-meanings).
- For floating-only `UNFOLD2D`, direct zero and matching typed positive zero share identity; exact
  padding bits shape compatibility.<span data-glossary-navigation="true"> ([positive zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [identity glossary](../../../../glossary.md#term-identity-meanings); [padding glossary](../../../../glossary.md#padding); [shape glossary](../../../../glossary.md#shape))</span>
- Unequal ranks use compact variable-rank geometry; start decomposition is cold, flattened state
  uses hot odometers.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [decomposition glossary](../../../../glossary.md#term-decomposition))</span>
- Static failures happen by [prepare](../../../../glossary.md#prepare) time; extraction adds no value-dependent partial-write policy.
- [Schema](../../../../glossary.md#term-schema) 13 is mandatory for new [bytecode](../../../../glossary.md#term-bytecode); no migration.
- Extend existing [owners](../../../../glossary.md#term-owner), add no production abstraction, and keep the 35-path ceiling.

## Known limitations

- Only one fully static resolved-[layout](../../../../glossary.md#layout) unfold [occurrence](../../../../glossary.md#term-occurrence) is executable.
- Output is [injective](../../../../glossary.md#term-injective), distinct, and non-overlapping with input.
- `UNFOLD2D` is floating-only symmetric constant-padding canonical flattened NCHW; integral and
  BOOL rows remain unsupported.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [padding glossary](../../../../glossary.md#padding); [canonical glossary](../../../../glossary.md#term-canonical-form); [NCHW glossary](../../../../glossary.md#term-nchw); [BOOL glossary](../../../../glossary.md#data-type--datatype))</span>
- No fold/index/[vector](../../../../glossary.md#term-vector)/native/dynamic/[DAG](../../../../glossary.md#term-directed-acyclic-graph-dag)/[Engine](../../../../glossary.md#term-engine)/tuning/benchmark/performance result exists.
- Tensor expressions normally retain unresolved layouts; existing lifecycle work must supply exact
  descriptors. This task does not add layout selection.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [lifecycle glossary](../../../../glossary.md#lifecycle); [descriptors glossary](../../../../glossary.md#term-descriptor))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

Implementation context supplied the stabilized Java evidence:

- The required focused ten-class matrix passed 10 suites and 63 tests with zero failures, zero
  errors, and zero skipped tests.<span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
- The sole final `./gradlew :backends:cpu:test` passed 32 suites and 163 tests with zero failures,
  zero errors, and one expected skipped opt-in persistence test.<span data-glossary-navigation="true"> ([persistence glossary](../../../../glossary.md#term-persistence))</span>
- The [toolchain](../../../../glossary.md#term-toolchain) was Java OpenJDK 26.0.1, build 26.0.1+8-34.
- Executable Java did not change after that final suite. The implementation advanced the private
  generated-artifact schema exactly from 12 to 13.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact); [schema glossary](../../../../glossary.md#term-schema))</span>

Clean documentation context `019ff061-4aa7-7f62-b2c5-7a1aed7b16f4` independently reviewed the
architecture, documentation profiles, task and [dependency](../../../../glossary.md#term-dependency) contracts, complete changed
production/test diff, current
Javadocs/package summaries, [CPU](../../../../glossary.md#term-cpu) guide, glossary, and relevant [Model](../../../../glossary.md#term-model) window contracts. It found no
implementation defect or architecture uncertainty and changed no executable Java behavior or
test. It therefore reused the stabilized Java evidence rather than rerunning Java tests.

- `./gradlew :backends:cpu:javadoc` passed after final [Javadoc](../../../../glossary.md#term-javadoc) edits.
- Local Markdown target and anchor validation passed for every changed Markdown file; balanced
  fences, canonical headings/statuses, final newlines, and trailing-whitespace checks passed.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- `git diff --check` passed on the final combined change.
- Exact scope validation passed the task allowlist and 35-path ceiling. Schema/status checks
  confirmed schema 13 and synchronized 0006A1 <span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>`Complete` records.
- Inventory checks confirmed that no 0006A2 task specification exists and 0006A2 remains `Draft`.
- Forbidden-change checks confirmed no shared Model, Compiler, Planning, Prepare, Runtime, Config,
  Trace, or Engine source; Gradle/dependency; architecture/ADR/test; backend-conformance; or
  integration path changed.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [Planning glossary](../../../../glossary.md#term-planning); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [Trace glossary](../../../../glossary.md#trace); [Engine glossary](../../../../glossary.md#term-engine); [Gradle glossary](../../../../glossary.md#term-gradle); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## Implementation notes

- `CpuCapabilityProvider` now admits exactly one fully static resolved-[layout](../../../../glossary.md#layout) `UNFOLD_AXIS` for
  all six represented types or one floating `UNFOLD2D` with either ordered attribute form.
- `CpuDataMovementIr`, `CpuNonAffineMovementLowering.Geometry`, generated emission, and the scalar
  reference implement compact unequal-rank axis/NCHW mapping. Exact two-dimensional padding bits
  shape artifact identity; concrete geometry remains cold.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [rank glossary](../../../../glossary.md#term-rank); [axis glossary](../../../../glossary.md#term-axis); [NCHW glossary](../../../../glossary.md#term-nchw); [padding glossary](../../../../glossary.md#padding); [shape glossary](../../../../glossary.md#shape); [artifact glossary](../../../../glossary.md#term-artifact); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Preparation declares input then output with no workspace, scalar or parallel-scalar execution,
  and one schema-13 artifact. Existing cold binding rejects incompatible carriers, noncanonical
  BOOL input, and input/output overlap before writing.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [binding glossary](../../../../glossary.md#term-binding); [BOOL glossary](../../../../glossary.md#data-type--datatype))</span>
- Package summaries remain accurate without modification: their existing movement, lowering,
  emission, preparation, executable, cache, reference, and public-CPU ownership descriptions
  already cover family extensions without enumerating a stale closed operation list.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [cache glossary](../../../../glossary.md#term-cache); [CPU glossary](../../../../glossary.md#term-cpu); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [operation glossary](../../../../glossary.md#operation))</span>
- The two authorized tests not changed—`CpuKernelSpecializationTest` and
  `CpuReferenceDifferentialTest`—remain accurate because specialization/window identity and
  generated/reference parity are covered through the changed IR/generated-kernel owners; no
  separate assertion or fixture ownership moved into those files.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [kernel glossary](../../../../glossary.md#kernel); [fixture glossary](../../../../glossary.md#term-fixture))</span>
- `CpuKernelSpecialization`, `CpuPortableKernelIr`, `CpuPreparedExecutable`, and the preparation
  plan/preparer Javadocs remain accurate because the new family uses their existing generic
  structural-key, movement-geometry, binding, and lifecycle contracts without changing those
  APIs.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>

## Completion summary

[CPU](../../../../glossary.md#term-cpu) 0006A1 is complete. The portable CPU movement route now materializes exactly one fully static,
resolved-[layout](../../../../glossary.md#layout) general-[axis](../../../../glossary.md#term-axis) unfold for all six represented types or one floating [canonical](../../../../glossary.md#term-canonical-form) [NCHW](../../../../glossary.md#term-nchw)
two-dimensional unfold with conceptual positive-zero or exact typed [padding](../../../../glossary.md#padding). Compact unequal-[rank](../../../../glossary.md#term-rank)
geometry, generated and [scalar](../../../../glossary.md#term-scalar)-reference mappings, scalar/parallel-scalar orchestration, exact
[schema](../../../../glossary.md#term-schema)-13 compatibility, one distinct [injective](../../../../glossary.md#term-injective) output, and [fail-closed](../../../../glossary.md#term-fail-closed) exclusions are documented
and validated. No unresolved issue remains. CPU 0006A2 and later work remain Draft without a
detailed 0006A2 specification.

- Completed changes: portable static window extraction plus finalized Javadocs, CPU guide,
  glossary, task evidence, CPU master plan, and roadmap.
- Files changed or created: the exact final allowlisted paths recorded by repository status.
- Tests and validation: reused 10-suite/63-test focused and 32-suite/163-test final CPU evidence;
  CPU Javadoc and all requested documentation/scope/status/schema/whitespace gates passed.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [scope glossary](../../../../glossary.md#term-scope))</span>
- Documentation-agent review: clean context `019ff061-4aa7-7f62-b2c5-7a1aed7b16f4`; complete
  with no executable Java changes.
- Documentation impact: CPU guide and [planning](../../../../glossary.md#term-planning)/status records now describe current window support.
- [Javadoc](../../../../glossary.md#term-javadoc) review: affected contract Javadocs finalized; unchanged authorized contracts reviewed.
- Glossary impact: the existing CPU static-data-movement term now includes window extraction.
- Unresolved issues: None.
- Follow-up required: None.

Status: Complete
