# ONNX Import And Export

ONNX support is an interchange boundary. It converts between external ONNX protobuf models and Synaptik semantic tensor graphs. It does not choose CPU, Metal, CUDA, runtime thresholds, publication behavior, calibration profiles, or graph autotune winners.

## Layer Boundary

The ONNX layer owns:

- protobuf read/write for ONNX `ModelProto`;
- mapping ONNX dense tensor metadata to Synaptik dtype, shape, and labels;
- mapping supported ONNX nodes to existing `Tensor` operations;
- clear diagnostics for unsupported operators, dtypes, dynamic shapes, external data, and attributes.

The ONNX layer does not own:

- `CompileConfig`, backend planning, region ownership, or memory planning;
- `RuntimeConfig`, accelerator availability, buffer binding, BLAS, or publication policy;
- CPU kernels, Metal MPSGraph routing, CUDA routing, or fallback decisions;
- calibration, benchmark persistence, or autotune search.

The intended flow is:

```text
ONNX ModelProto
  -> ONNX import mapper
  -> regular Synaptik Tensor graph
  -> CompileConfig
  -> backend planning / region optimization / memory planning
  -> RuntimeConfig
  -> execution
```

Export follows the reverse semantic direction:

```text
Synaptik Tensor graph
  -> ONNX export mapper
  -> ONNX ModelProto
```

## Public API

```java
Tensor out = a.matmul(b.transpose());

OnnxModel model = Onnx.exportModel(
        out,
        OnnxExportOptions.defaults().withLeafTensorPolicy(OnnxLeafTensorPolicy.INPUTS)
);

model.write(Path.of("model.onnx"));
```

```java
// Multi-output export is intentionally narrow and currently exists for ONNX
// boundary patterns such as Split. The outputs must be regular Synaptik tensors.
OnnxModel splitModel = Onnx.exportModel(List.of(leftSlice, rightSlice));
```

```java
ImportedOnnxModel imported = Onnx.read(Path.of("model.onnx"));

imported.input("a").setData(new float[]{1f, 2f, 3f, 4f, 5f, 6f});
imported.input("b").setData(new float[]{7f, 8f, 9f, 10f, 11f, 12f});

CompiledGraph compiled = imported.compile("scores", CompileConfig.inference());
compiled.execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
```

`OnnxLeafTensorPolicy` controls how operation-free Synaptik leaf tensors are exported. The default is `INPUTS`, so exporting a normal computation graph does not accidentally serialize user input values as model weights.

- `INPUTS`: every leaf is an ONNX graph input and storage is not serialized.
- `INITIALIZERS`: every leaf is serialized as an ONNX initializer.
- `TRAINABLE_INPUTS`: trainable leaves become inputs and non-trainable leaves become initializers.

## Supported Subset

The first implementation intentionally supports static dense inference graphs.

Supported dtypes:

| ONNX dtype | Synaptik dtype | Notes |
|---|---|---|
| `FLOAT` | `FLOAT32` | Tensor input, initializer, compute value. |
| `DOUBLE` | `FLOAT64` | Tensor input, initializer, compute value where Synaptik op supports it. |
| `BFLOAT16` | `BFLOAT16` | Tensor input/initializer; stored as ONNX `int32_data` bit patterns when exported. |
| `INT32` | `INT32` | Tensor input/initializer for supported integer roles. |
| `BOOL` | `BOOL` | Tensor input/initializer for supported boolean roles. |
| `INT64` | `INT64` | Tensor input/initializer for supported integer roles and ONNX-compatible shape/index plumbing. CPU supports runtime `INT64`; accelerator native support remains backend-scoped and conservative. |

Supported node families:

| ONNX op | Synaptik mapping |
|---|---|
| `Add`, `Sub`, `Mul`, `Div`, `Min`, `Max` | Floating tensor ops with existing broadcasting rules. ONNX variadic `Min`/`Max` import is lowered to a left-associated binary chain such as `Min(Min(a, b), c)`. Export currently writes the Synaptik binary DAG rather than re-packing chains into a single variadic ONNX node. |
| `Pow` | Scalar-exponent and tensor-exponent power. Scalar Synaptik `pow(Tensor, double)` exports as ONNX `Pow` with a scalar initializer. Tensor exponent `pow(base, exponent)` imports and exports as binary ONNX `Pow` with normal broadcasting. Accelerator native coverage remains scalar-pow scoped unless a backend explicitly proves binary power support. |
| `Neg`, `Abs`, `Relu`, `Tanh`, `Sigmoid`, `Exp`, `Log`, `Sqrt`, `Reciprocal`, `Erf`, `Floor`, `Ceil`, `Sign` | Unary floating tensor ops. `Reciprocal` maps to Synaptik `inv`. `Erf` uses a scalar CPU approximation because Java has no standard `Math.erf`. `Floor`, `Ceil`, and `Sign` are inference-friendly non-smooth unary ops and currently do not define useful gradients. |
| `LeakyRelu`, `Elu`, `HardSigmoid`, `Softplus` | Composed activation lowerings. `LeakyRelu` lowers to `Where(x >= 0, x, alpha * x)`, `Elu` lowers to `Where(x >= 0, x, alpha * (Exp(x) - 1))`, `HardSigmoid` lowers to `Clip(alpha * x + beta, 0, 1)`, and `Softplus` lowers to `Log(Exp(x) + 1)`. Export has conservative canonical recognizers for those exact Synaptik compositions; near-miss graphs remain primitive ONNX nodes. |
| `Equal`, `Greater`, `GreaterOrEqual`, `Less`, `LessOrEqual` | Binary floating comparisons with boolean output. |
| `And`, `Or`, `Not` | Boolean tensor logic. |
| `Where` | Boolean condition plus two floating branches using Synaptik broadcast and dtype promotion rules. |
| `Identity` | Import-only pass-through mapping to the input tensor. |
| `Clip` | Scalar min/max clamp. Opset-style optional min/max inputs are supported when present as scalar initializers or scalar `Constant` nodes; legacy float `min`/`max` attributes are also accepted. Export emits one-sided `Clip` nodes for Synaptik `clampMin` and `clampMax`. |
| `Cast` | Explicit graph dtype conversion for supported Synaptik dtypes, including runtime `INT64` on CPU. Shape-only `INT64`/`INT32` casts are evaluated during import. Accelerator casts remain backend-scoped; Metal currently does not support `INT64` cast pairs. |
| `MatMul` | `Tensor.matmul`. |
| `Gemm` | `matmul` plus optional bias and scalar `alpha`/`beta`; rank-2 transpose flags are supported. |
| `Conv` | Rank-4 NCHW convolution mapped to `Tensor.conv2d`. Weights must be OIHW, bias is optional rank-1, and attributes are static. Symmetric spatial pads stay in `Conv2dOptions`; asymmetric ONNX pads import as an explicit static `Pad -> Conv` DAG. |
| `MaxPool` | Rank-4 NCHW max pooling mapped to `Tensor.maxPool2d`. `kernel_shape`, `strides`, symmetric `pads`, and static `ceil_mode` are represented in `Pool2dOptions`. Accelerator-native pool rows may still reject `ceil_mode=true`. |
| `AveragePool` | Rank-4 NCHW average pooling mapped to `Tensor.avgPool2d`. `count_include_pad` and static `ceil_mode` are preserved in `Pool2dOptions`, but backend-native support remains backend-specific. |
| `LayerNormalization` | Single-output inference form mapped to `Tensor.layerNorm`. The ONNX `axis` must select trailing normalized dimensions so it matches Synaptik's tail-parameter contract. Missing bias is imported as a zero tensor matching scale. |
| `BatchNormalization` | Single-output inference form mapped to external-statistics `Tensor.batchNorm` with channel axis 1. `training_mode=1` and multi-output training forms are rejected. Export recognizes the canonical external-statistics Synaptik batch-norm DAG and writes a single ONNX `BatchNormalization` node. |
| `Transpose` | `Tensor.permute`. |
| `Reshape` | `Tensor.reshape` with constant shape input. |
| `Flatten` | Static reshape using the ONNX `axis` attribute. Export emits canonical `Flatten` only for rank > 2 reshapes whose rank-2 target exactly matches an ONNX flatten split; other reshapes remain `Reshape`. |
| `Expand` | `Tensor.expand` with constant target shape. |
| `Pad` | Constant-mode padding mapped to `Tensor.pad`. Pads must be static, non-negative, and have length `2 * rank`; the optional pad value must be a scalar initializer or scalar `Constant` node. Reflect/edge/wrap modes are rejected because they require different boundary semantics. |
| `Tile` | `Tensor.tile` with static positive repeat counts. The repeat vector length must match input rank. |
| `ConstantOfShape` | Import-time materialization of a constant leaf tensor. The shape input must be a static `INT64` or `INT32` constant vector. The optional `value` attribute must contain exactly one element. Supported output dtypes are `FLOAT`, `DOUBLE`, `BFLOAT16`, `INT32`, `INT64`, and `BOOL`. |
| `Range` | Import-time constant folding. Static `INT64` inputs produce both a materialized `INT64` constant tensor and importer-internal shape values for later shape plumbing. Supported scalar tensor initializers or scalar `Constant` nodes produce a materialized constant leaf tensor. Positive and negative `delta` are supported, and `delta=0` is rejected. Runtime `Range` is not supported because output length is data-dependent. |
| `Squeeze`, `Unsqueeze` | `Tensor.squeeze` / `Tensor.expandDims` with constant axes. |
| `Slice` | Static positive-step slice with constant `starts`, `ends`, `axes`, and `steps`. Runtime tensor slicing maps to `Tensor.slice`; importer-internal shape-vector slicing is evaluated during import. Very large ONNX end sentinels such as `INT64_MAX` are saturated and then clamped to the known static dimension. |
| `Concat` | Runtime tensor concat for matching dtypes/ranks; shape-only concat for importer-internal `INT64` shape vectors. |
| `Split` | Import lowers to one `Tensor.slice` per output. Export supports the reverse pattern when the requested graph outputs are sibling `Slice` tensors over the same input that exactly cover one axis with static split sizes. This is deliberately not a general multi-output runtime architecture; it is a narrow ONNX boundary adapter. |
| `Shape`, `Size`, `Gather` | Runtime `Gather` maps to ONNX-style `Tensor.gatherAxis`, where the index tensor shape is inserted at the gathered axis. Shape-only `Gather` remains import-time shape plumbing and is limited to axis `0` because the importer represents shape tensors as flat compile-time vectors. |
| `GatherElements` | Runtime `GatherElements` maps to `Tensor.takeAlongAxis`. The data and index tensors must have the same rank, the output shape equals the index tensor shape, and all non-axis dimensions must match. Runtime indices may be `INT32` or `INT64` on CPU/ONNX; accelerator native rows remain backend-scoped. Export writes `takeAlongAxis` as `GatherElements`. |
| `GatherND` | Runtime `GatherND` maps to `Tensor.gatherNd`. The final dimension of `indices` is the coordinate tuple length. With `batch_dims=B` and tuple length `K`, the output shape is `indices.shape[:B] + indices.shape[B:-1] + data.shape[B + K:]`; `batch_dims=0` is the usual `indices.shape[:-1] + data.shape[K:]` case. Runtime indices may be `INT32` or `INT64` on CPU/ONNX; accelerator native rows remain backend-scoped. |
| `ScatterElements` | Runtime `ScatterElements` maps to functional `Tensor.scatterElements`. The output shape equals `data.shape`; `indices` and `updates` must have the same rank and shape, and non-axis dimensions must match `data`. Supported reductions are `none`, `add`, `mul`, `max`, and `min` for inference; backward is defined only for `none` and `add`. Runtime indices may be `INT32` or `INT64` on CPU/ONNX; accelerator native rows remain backend-scoped. |
| `ScatterND` | Runtime `ScatterND` maps to functional `Tensor.scatterNd`. The output shape equals `data.shape`; the final dimension of `indices` is the coordinate tuple length; `updates.shape` must equal `indices.shape[:-1] + data.shape[indices.shape[-1]:]`. Supported reductions are `none`, `add`, `mul`, `max`, and `min` for inference; backward is defined only for `none` and `add`. Runtime indices may be `INT32` or `INT64` on CPU/ONNX; accelerator native rows remain backend-scoped. |
| `ReduceSum`, `ReduceMean`, `ReduceMax`, `ReduceMin`, `ReduceProd` | Axis reductions; multi-axis reductions are applied as repeated Synaptik reductions. `ReduceProd` is currently an inference primitive and does not define autograd. |
| `ReduceL1` | Composed lowering: `Abs` followed by `ReduceSum`. Multi-axis import behavior matches `ReduceSum`. Export recognizes the exact single-axis `abs(x).sum(axis, keepDims)` pattern. |
| `ReduceL2` | Composed lowering: square with `Mul`, reduce with `ReduceSum`, then apply `Sqrt` after all axes have been reduced. Applying `Sqrt` once at the end is required for correct multi-axis import math. Export recognizes the exact single-axis `sqrt(sum(x * x, axis, keepDims))` pattern. |
| `ReduceLogSum` | Composed lowering: `ReduceSum` followed by `Log` after all axes have been reduced. Export recognizes the exact single-axis `log(sum(x, axis, keepDims))` pattern. |
| `ReduceLogSumExp` | Composed lowering: `Exp`, then `ReduceSum`, then `Log`. This is the direct ONNX formula, not the numerically stabilized max-shift variant. Export recognizes the exact single-axis `log(sum(exp(x), axis, keepDims))` pattern. |
| `ArgMax` | Axis argmax with explicit tie policy. `select_last_index=0` maps to `FIRST_INDEX`; `select_last_index=1` maps to `LAST_INDEX`. Output is `INT64`, matching ONNX's usual index dtype. Accelerator-native rows remain backend-scoped; a backend may support only first-index ties until last-index behavior is proven. |
| `CumSum` | First-class `Tensor.cumSum(axis, exclusive, reverse)` with shape-preserving output. The ONNX axis input must be a static scalar `INT64`/`INT32` constant. Floating dtypes and `INT32` are supported; `BOOL` is rejected. CPU execution is layout-aware. Metal supports dense `FLOAT32/BFLOAT16` inputs; CUDA remains unsupported. |
| `GlobalAveragePool` | Lowering to repeated `Tensor.mean(axis, keepDims=true)` over spatial axes. Import supports static rank >= 3. Export recognizes the rank-4 NCHW spatial mean chain and writes canonical `GlobalAveragePool`. |
| `Softmax`, `LogSoftmax` | Axis normalization ops. |
| `Constant` | Tensor initializer in graph-node form. Export can emit leaf tensors as ONNX `Constant` nodes with `OnnxLeafTensorPolicy.CONSTANT_NODES`; the other leaf policies continue to use graph inputs or initializers. |

## Coverage Matrix

The code-level source of truth for interchange coverage is `onnx.OnnxCoverageMatrix`. The generated human-readable report is `docs/onnx-coverage.md`; it is rendered by `onnx.OnnxCoverageReport` and checked by `OnnxCoverageMatrixTest`, so status rows are not hand-maintained. Each row separates:

- ONNX import support: whether an ONNX node can be translated into a Synaptik graph.
- ONNX export support: whether a Synaptik semantic op can be serialized as that ONNX op.
- CPU support: whether the imported graph has a CPU execution path.
- Metal/CUDA support: whether the mapped Synaptik operation is covered by the native GPU lowering matrix.

This distinction matters. For example, `Pad`, `Tile`, `ReduceProd`, `ArgMax`, scoped non-negative `GatherND`, `ScatterElements`, and `ScatterND` have Metal rows, while CUDA still keeps explicit blockers for the index-write rows. `ArgMax` is now a CPU/ONNX `INT64` output row and Metal produces that public `INT64` index-output contract rather than the older scoped `INT32` bridge output. `Split`, `Shape`, `Size`, `ConstantOfShape`, and `Range` are static or import-boundary rows; they should not be read as native GPU operation promises. `Split` export is available only through the multi-output export API and only for graph-output slice siblings. Conversely, Metal supports internal operations such as SDPA, selected losses, and backward-adjacent ops that are not ONNX interchange rows yet.

Index conformance is covered by checked-in miniature ONNX models under `src/test/resources/onnx/index/`. Those fixtures are regenerated from the Java builder in `OnnxIndexFixtureModels` and then byte-compared in tests, so review can inspect both executable ONNX files and the source definition. The current fixture set covers executable `GatherElements`, `GatherND`, `ScatterElements`, and `ScatterND` variants, including axes, negative axes/indices, tuple slices, `GatherND batch_dims`, and `ScatterND` inference reductions. Invalid duplicate-write cases are kept as code-built rejection tests instead of executable fixture files.

NN inference conformance is covered the same way under `src/test/resources/onnx/nn/`. `OnnxNnFixtureModels` generates the checked-in fixture files and `OnnxNnFixtureTest` byte-compares them before execution. The current fixture set covers `Conv`, `MaxPool`, `AveragePool`, `LayerNormalization`, and inference `BatchNormalization`.

Static breadth conformance is covered under `src/test/resources/onnx/breadth/`. `OnnxBreadthFixtureModels` generates checked-in fixtures for `Pad`, `Split`, `Tile`, `ArgMax`, `ReduceProd`, and `GlobalAveragePool`; `OnnxBreadthFixtureTest` byte-compares them and executes every declared output, including both outputs of the special-case `Split` lowering.
The same fixture set also covers wave 3 static inference breadth: `ConstantOfShape`, static tensor `Range`, `ReduceL1`, `ReduceL2`, `ReduceLogSum`, `ReduceLogSumExp`, and `CumSum`.

Activation/math conformance is covered under `src/test/resources/onnx/activation/`. `OnnxActivationFixtureModels` generates the checked-in fixture file and `OnnxActivationFixtureTest` byte-compares it before execution. The fixture exercises first-class unary interchange rows (`Reciprocal`, `Erf`, `Floor`, `Ceil`, `Sign`) and composed activation imports (`LeakyRelu`, `Elu`, `HardSigmoid`, `Softplus`). `OnnxWave4ActivationExecutionTest` also checks export operator names and Synaptik graph -> ONNX export -> ONNX import -> execution round trips for the new unary primitives.

A small compatibility harness lives under `src/test/resources/onnx/compat/`. `OnnxCompatibilityFixtureModels` classifies each miniature model as `IMPORTED`, `EXECUTED`, or `REJECTED_WITH_REASON`; `OnnxCompatibilityHarnessTest` byte-compares every checked-in fixture and then either executes expected outputs or verifies the expected rejection. Current cases cover activation MLPs, `Erf`/`Softplus`, conv/pool/classifier, shape-helper/reduction, layernorm/residual/broadcast, gather/scatter, global average pool, and explicit dynamic-shape or runtime-static-parameter rejections.

Practical examples:

```text
Static shape supported:
Shape(x) -> Gather(dim) -> Unsqueeze -> Concat -> Reshape(x)
```

```text
Rejected dynamic shape:
Reshape(x, runtime_shape_input)
Slice(x, runtime_starts, static_ends)
```

```text
Import/export supported, GPU not guaranteed:
GatherND(data, runtime_int32_indices)
ScatterND(data, runtime_int32_indices, updates)
```

```text
Composite canonical export:
log(sum(exp(x), axis)) -> ReduceLogSumExp
sqrt(sum(x * x, axis)) -> ReduceL2
```

Static helper terminology in this importer:

- **Importer-internal shape constant** means a Java `long[]` tracked by the ONNX importer for shape plumbing. It is not a Synaptik runtime tensor and cannot be a graph output.
- **Constant tensor leaf** means a normal Synaptik `Tensor` with storage materialized during import. It can feed runtime graph ops or be a graph output.
- **Runtime tensor op** means a Synaptik graph node executed by CPU/Metal/CUDA after compile. `ConstantOfShape` and `Range` are not runtime tensor ops in this implementation; only their static subsets are folded during ONNX import.

Coverage limitation categories in `docs/onnx-coverage.md`:

- **static_semantic_limit**: the ONNX meaning is represented by the current primitive algebra, but only through a scoped semantic form, composed pattern, or conservative export recognizer.
- **static_attribute_limit**: the op is static and executable, but only for specific ranks, axes, layouts, attributes, dtypes, or backend-native scopes.
- **multi_output_limit**: the ONNX op crosses the current single-output core graph boundary. `Split` remains a narrow static boundary adapter; training `BatchNormalization` and general multi-output ops need a real multi-output model.
- **runtime_shape_limit**: the op needs runtime shape tensor values or execute-time allocation to support the full ONNX form. Current support is importer-time/static only.
- **data_dependent_shape_limit**: the output shape depends on runtime data values, as with `NonZero`.

Explicit non-goals in the current algebra subset:

- Variadic ONNX `Min`/`Max` export packing is not implemented. Import accepts variadic nodes, but export preserves the explicit binary Synaptik DAG because that keeps graph structure and intermediate consumer semantics unambiguous.
- `Softplus` currently uses the direct mathematical lowering `Log(Exp(x) + 1)`. It is correct for the small/static compatibility fixtures, but it is not the numerically stabilized thresholded implementation used by some inference runtimes for very large positive inputs.
- Canonical export recognizers are intentionally conservative. A Synaptik graph must match the supported composed activation or reduction pattern exactly and all internal nodes must have a single consumer; otherwise export writes primitive ONNX nodes.
- Runtime ONNX `Gather` is supported through the dedicated `gatherAxis` graph op, not the older Synaptik `gather` helper with reduced output shape. Runtime ONNX `GatherElements` is supported through `takeAlongAxis`, which preserves rank and uses the index tensor shape as the output shape. `GatherND` supports ONNX `batch_dims`; the leading batch dimensions select matching slices and are not part of the coordinate tuple stored in the final index dimension.
- General multi-output runtime graph support is still not part of the importer. `Split` is a named exception because it can be lowered immediately to independent `Slice` tensors with static shapes and no shared mutable output state; export can write canonical `Split` only when those slices are graph outputs.
- Dynamic shape, slice, reshape, and expand parameters are rejected; the current importer remains static dense inference.
- Runtime `NonZero` is rejected because its output shape depends on input data. ONNX defines the second output dimension as the number of non-zero values, which is unknown until execution. Supporting it as a normal runtime op would require a dynamic-shape graph/execution model, not just another CPU kernel.
- Runtime shape tensors and general multi-output ops are tracked separately in `todo/74-onnx-runtime-shape-and-multi-output-architecture.md`. Higher-level layer-aware import/export belongs in `todo/75-nn-layer-api-and-layer-aware-onnx-interchange.md`, not in the primitive core ONNX importer.

Unsupported by design in the first subset:

- dynamic or symbolic dimensions;
- sparse tensors;
- quantized tensors;
- string tensors;
- sequence/map/optional ONNX values;
- training metadata;
- external data files;
- custom domains;
- multi-output nodes other than the narrow static `Split` lowering;
- data-dependent output shapes such as runtime `NonZero`;
- broad dynamic-shape, quantized, sparse, control-flow, and full training import coverage.

## Failure Mode

Unsupported models fail during import or export with `OnnxUnsupportedException`. The message names the ONNX node, op type, tensor, dtype, or attribute that crossed the supported subset boundary. This is deliberate: ONNX import/export should not silently rewrite semantics or fall back to a backend-specific path.

External data is rejected because it is filesystem-sensitive. A later implementation can enable it only with path traversal checks and model-directory-relative resolution.

## Maintenance Workflow

After changing ONNX importer/exporter behavior, run the focused ONNX suite:

```bash
./gradlew test --tests 'onnx.*'
./gradlew test --tests SourceTreeHygieneTest
git diff --check
```

After changing a fixture builder, regenerate the checked-in ONNX resources and let the harness byte-compare them:

```bash
./gradlew testClasses
java --add-modules=jdk.incubator.vector \
  -cp build/classes/java/main:build/classes/java/test:<protobuf jar> \
  onnx.OnnxCompatibilityFixtureModels src/test/resources/onnx/compat
./gradlew test --tests 'onnx.OnnxCompatibilityHarnessTest'
```

After changing `OnnxCoverageMatrix`, regenerate the report instead of hand-editing rows:

```bash
./gradlew classes
java --add-modules=jdk.incubator.vector \
  -cp build/classes/java/main:<protobuf jar> \
  onnx.OnnxCoverageReport docs/onnx-coverage.md
./gradlew test --tests 'onnx.OnnxCoverageMatrixTest'
```
