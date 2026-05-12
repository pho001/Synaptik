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
| `Transpose` | `Tensor.permute`. |
| `Reshape` | `Tensor.reshape` with constant shape input. |
| `Flatten` | Static reshape using the ONNX `axis` attribute. |
| `Expand` | `Tensor.expand` with constant target shape. |
| `Squeeze`, `Unsqueeze` | `Tensor.squeeze` / `Tensor.expandDims` with constant axes. |
| `Slice` | Static positive-step slice with constant `starts`, `ends`, `axes`, and `steps`. Runtime tensor slicing maps to `Tensor.slice`; importer-internal shape-vector slicing is evaluated during import. Very large ONNX end sentinels such as `INT64_MAX` are saturated and then clamped to the known static dimension. |
| `Concat` | Runtime tensor concat for matching dtypes/ranks; shape-only concat for importer-internal `INT64` shape vectors. |
| `Shape`, `Size`, shape-only `Gather` | Import-time shape constants used to feed `Reshape`, `Expand`, and similar shape parameters. `Shape` supports static `start`/`end` attributes. Shape-only `Gather` and `Slice` are limited to axis `0` because the importer represents shape tensors as flat compile-time vectors. |
| `ReduceSum`, `ReduceMean`, `ReduceMax`, `ReduceMin` | Axis reductions; multi-axis reductions are applied as repeated Synaptik reductions. |
| `Softmax`, `LogSoftmax` | Axis normalization ops. |
| `Constant` | Tensor initializer in graph-node form. |

Explicit non-goals in the current algebra subset:

- Tensor-by-tensor `Pow` is rejected. It needs either a first-class Synaptik tensor exponent op or a documented lowering strategy before import/export can claim support.
- Variadic ONNX `Min`/`Max` are rejected unless represented as binary nodes. A future importer can lower a variadic ONNX node into a left-associated chain if that behavior is intentionally added.
- Runtime ONNX `Gather` is rejected except for importer-internal shape vectors. ONNX `Gather` has broader output-shape semantics than Synaptik's current `gather` graph op, so broad support needs a dedicated semantic mapping.
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
- multi-output nodes;
- broad convolution, pooling, gather/scatter, and indexing import coverage.

## Failure Mode

Unsupported models fail during import or export with `OnnxUnsupportedException`. The message names the ONNX node, op type, tensor, dtype, or attribute that crossed the supported subset boundary. This is deliberate: ONNX import/export should not silently rewrite semantics or fall back to a backend-specific path.

External data is rejected because it is filesystem-sensitive. A later implementation can enable it only with path traversal checks and model-directory-relative resolution.
