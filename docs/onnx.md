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
| `INT64` | shape constant only | Accepted for ONNX axes/shape initializers; rejected as a normal tensor value. |

Supported node families:

| ONNX op | Synaptik mapping |
|---|---|
| `Add`, `Sub`, `Mul`, `Div`, `Min`, `Max` | Binary floating tensor ops with existing broadcasting rules. ONNX variadic `Min`/`Max` forms are not expanded; the supported form has exactly two inputs. |
| `Pow` | Scalar-exponent power. The exponent must be a scalar initializer or scalar `Constant` node because Synaptik's graph op is `pow(Tensor, double)`, not tensor-by-tensor exponentiation. |
| `Neg`, `Abs`, `Relu`, `Tanh`, `Sigmoid`, `Exp`, `Log`, `Sqrt` | Unary floating tensor ops. |
| `Equal`, `Greater`, `GreaterOrEqual`, `Less`, `LessOrEqual` | Binary floating comparisons with boolean output. |
| `And`, `Or`, `Not` | Boolean tensor logic. |
| `Where` | Boolean condition plus two floating branches using Synaptik broadcast and dtype promotion rules. |
| `Identity` | Import-only pass-through mapping to the input tensor. |
| `Clip` | Scalar min/max clamp. Opset-style optional min/max inputs are supported when present as scalar initializers or scalar `Constant` nodes; legacy float `min`/`max` attributes are also accepted. Export emits one-sided `Clip` nodes for Synaptik `clampMin` and `clampMax`. |
| `Cast` | Explicit graph dtype conversion for supported Synaptik dtypes except runtime `INT64`. Shape-only `INT64`/`INT32` casts are evaluated during import. |
| `MatMul` | `Tensor.matmul`. |
| `Gemm` | `matmul` plus optional bias and scalar `alpha`/`beta`; rank-2 transpose flags are supported. |
| `Conv` | Rank-4 NCHW convolution mapped to `Tensor.conv2d`. Weights must be OIHW, bias is optional rank-1, attributes are static, and pads must be symmetric spatial NCHW pads. |
| `MaxPool` | Rank-4 NCHW max pooling mapped to `Tensor.maxPool2d`. `kernel_shape` is required, `strides` and symmetric `pads` are supported, and `ceil_mode=1` is rejected. |
| `AveragePool` | Rank-4 NCHW average pooling mapped to `Tensor.avgPool2d`. `count_include_pad` is preserved in the Synaptik pool options, but backend-native support remains backend-specific. |
| `LayerNormalization` | Single-output inference form mapped to `Tensor.layerNorm`. The ONNX `axis` must select trailing normalized dimensions so it matches Synaptik's tail-parameter contract. Missing bias is imported as a zero tensor matching scale. |
| `BatchNormalization` | Single-output inference form mapped to external-statistics `Tensor.batchNorm` with channel axis 1. `training_mode=1` and multi-output training forms are rejected. Export is not first-class because Synaptik currently represents batch norm as a composed graph, not a single descriptor. |
| `Transpose` | `Tensor.permute`. |
| `Reshape` | `Tensor.reshape` with constant shape input. |
| `Flatten` | Static reshape using the ONNX `axis` attribute. |
| `Expand` | `Tensor.expand` with constant target shape. |
| `Pad` | Constant-mode padding mapped to `Tensor.pad`. Pads must be static, non-negative, and have length `2 * rank`; the optional pad value must be a scalar initializer or scalar `Constant` node. Reflect/edge/wrap modes are rejected because they require different boundary semantics. |
| `Tile` | `Tensor.tile` with static positive repeat counts. The repeat vector length must match input rank. |
| `Squeeze`, `Unsqueeze` | `Tensor.squeeze` / `Tensor.expandDims` with constant axes. |
| `Slice` | Static positive-step slice with constant `starts`, `ends`, `axes`, and `steps`. Runtime tensor slicing maps to `Tensor.slice`; importer-internal shape-vector slicing is evaluated during import. Very large ONNX end sentinels such as `INT64_MAX` are saturated and then clamped to the known static dimension. |
| `Concat` | Runtime tensor concat for matching dtypes/ranks; shape-only concat for importer-internal `INT64` shape vectors. |
| `Split` | Import-only lowering to one `Tensor.slice` per output. Split sizes must be static, either from the second input, legacy `split` attribute, or equal division when no explicit sizes are supplied. This is deliberately not a general multi-output graph architecture; it is a narrow ONNX boundary adapter. |
| `Shape`, `Size`, `Gather` | Runtime `Gather` maps to ONNX-style `Tensor.gatherAxis`, where the index tensor shape is inserted at the gathered axis. Shape-only `Gather` remains import-time shape plumbing and is limited to axis `0` because the importer represents shape tensors as flat compile-time vectors. |
| `GatherElements` | Runtime `GatherElements` maps to `Tensor.takeAlongAxis`. The data and index tensors must have the same rank, the output shape equals the index tensor shape, and all non-axis dimensions must match. Runtime indices are `INT32`; ONNX `INT64` remains shape-constant-only in this importer. Export writes `takeAlongAxis` as `GatherElements`. |
| `GatherND` | Runtime `GatherND` maps to `Tensor.gatherNd`. The final dimension of `indices` is the coordinate tuple length. With `batch_dims=B` and tuple length `K`, the output shape is `indices.shape[:B] + indices.shape[B:-1] + data.shape[B + K:]`; `batch_dims=0` is the usual `indices.shape[:-1] + data.shape[K:]` case. Runtime indices are `INT32`; ONNX `INT64` remains shape-constant-only. |
| `ScatterElements` | Runtime `ScatterElements` maps to functional `Tensor.scatterElements`. The output shape equals `data.shape`; `indices` and `updates` must have the same rank and shape, and non-axis dimensions must match `data`. Supported reductions are `none`, `add`, `mul`, `max`, and `min` for inference; backward is defined only for `none` and `add`. Runtime indices are `INT32`; ONNX `INT64` remains shape-constant-only. |
| `ScatterND` | Runtime `ScatterND` maps to functional `Tensor.scatterNd`. The output shape equals `data.shape`; the final dimension of `indices` is the coordinate tuple length; `updates.shape` must equal `indices.shape[:-1] + data.shape[indices.shape[-1]:]`. Supported reductions are `none`, `add`, `mul`, `max`, and `min` for inference; backward is defined only for `none` and `add`. Runtime indices are `INT32`; ONNX `INT64` remains shape-constant-only. |
| `ReduceSum`, `ReduceMean`, `ReduceMax`, `ReduceMin`, `ReduceProd` | Axis reductions; multi-axis reductions are applied as repeated Synaptik reductions. `ReduceProd` is currently an inference primitive and does not define autograd. |
| `ArgMax` | Axis argmax with first-index tie behavior. Export and import support `select_last_index=0`; `select_last_index=1` is rejected. Output is `INT32`, not ONNX's usual `INT64`, because Synaptik runtime tensors do not support `INT64`. |
| `GlobalAveragePool` | Import-only lowering to repeated `Tensor.mean(axis, keepDims=true)` over spatial axes `2..rank-1`. This preserves the ONNX `N,C,1,1...` output shape for static dense inference. |
| `Softmax`, `LogSoftmax` | Axis normalization ops. |
| `Constant` | Tensor initializer in graph-node form. |

## Coverage Matrix

The code-level source of truth for interchange coverage is `onnx.OnnxCoverageMatrix`. Each row separates:

- ONNX import support: whether an ONNX node can be translated into a Synaptik graph.
- ONNX export support: whether a Synaptik semantic op can be serialized as that ONNX op.
- CPU support: whether the imported graph has a CPU execution path.
- Metal/CUDA support: whether the mapped Synaptik operation is covered by the native GPU lowering matrix.

This distinction matters. For example, `Pad`, `Tile`, `ReduceProd`, and `ArgMax` are valid ONNX import/export rows and execute on CPU, but they are explicit GPU coverage rows through the mapped Synaptik operation status, not implied native accelerator support. `Split` and `GlobalAveragePool` are import-supported even though they do not have first-class export rows because they lower to existing Synaptik graph primitives at the interchange boundary. Conversely, Metal supports internal operations such as SDPA, selected losses, and backward-adjacent ops that are not ONNX interchange rows yet.

Index conformance is covered by checked-in miniature ONNX models under `src/test/resources/onnx/index/`. Those fixtures are regenerated from the Java builder in `OnnxIndexFixtureModels` and then byte-compared in tests, so review can inspect both executable ONNX files and the source definition. The current fixture set covers executable `GatherElements`, `GatherND`, `ScatterElements`, and `ScatterND` variants, including axes, negative axes/indices, tuple slices, `GatherND batch_dims`, and `ScatterND` inference reductions. Invalid duplicate-write cases are kept as code-built rejection tests instead of executable fixture files.

NN inference conformance is covered the same way under `src/test/resources/onnx/nn/`. `OnnxNnFixtureModels` generates the checked-in fixture files and `OnnxNnFixtureTest` byte-compares them before execution. The current fixture set covers `Conv`, `MaxPool`, `AveragePool`, `LayerNormalization`, and inference `BatchNormalization`.

Static breadth conformance is covered under `src/test/resources/onnx/breadth/`. `OnnxBreadthFixtureModels` generates checked-in fixtures for `Pad`, `Split`, `Tile`, `ArgMax`, `ReduceProd`, and `GlobalAveragePool`; `OnnxBreadthFixtureTest` byte-compares them and executes every declared output, including both outputs of the special-case `Split` lowering.

Explicit non-goals in the current algebra subset:

- Tensor-by-tensor `Pow` is rejected. It needs either a first-class Synaptik tensor exponent op or a documented lowering strategy before import/export can claim support.
- Variadic ONNX `Min`/`Max` are rejected unless represented as binary nodes. A future importer can lower a variadic ONNX node into a left-associated chain if that behavior is intentionally added.
- Runtime ONNX `Gather` is supported through the dedicated `gatherAxis` graph op, not the older Synaptik `gather` helper with reduced output shape. Runtime ONNX `GatherElements` is supported through `takeAlongAxis`, which preserves rank and uses the index tensor shape as the output shape. `GatherND` supports ONNX `batch_dims`; the leading batch dimensions select matching slices and are not part of the coordinate tuple stored in the final index dimension.
- General multi-output graph support is still not part of the importer. `Split` is a named exception because it can be lowered immediately to independent `Slice` tensors with static shapes and no shared mutable output state.
- Dynamic shape, slice, reshape, and expand parameters are rejected; the current importer remains static dense inference.

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
- broad dynamic-shape, quantized, sparse, control-flow, and full training import coverage.

## Failure Mode

Unsupported models fail during import or export with `OnnxUnsupportedException`. The message names the ONNX node, op type, tensor, dtype, or attribute that crossed the supported subset boundary. This is deliberate: ONNX import/export should not silently rewrite semantics or fall back to a backend-specific path.

External data is rejected because it is filesystem-sensitive. A later implementation can enable it only with path traversal checks and model-directory-relative resolution.
