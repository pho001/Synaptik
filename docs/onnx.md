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
| `Add`, `Sub`, `Mul`, `Div` | Binary tensor ops with existing broadcasting rules. |
| `Neg`, `Abs`, `Relu`, `Tanh`, `Sigmoid`, `Exp`, `Log`, `Sqrt` | Unary tensor ops. |
| `MatMul` | `Tensor.matmul`. |
| `Gemm` | `matmul` plus optional bias and scalar `alpha`/`beta`; rank-2 transpose flags are supported. |
| `Transpose` | `Tensor.permute`. |
| `Reshape` | `Tensor.reshape` with constant shape input. |
| `Squeeze`, `Unsqueeze` | `Tensor.squeeze` / `Tensor.expandDims` with constant axes. |
| `ReduceSum`, `ReduceMean`, `ReduceMax`, `ReduceMin` | Axis reductions; multi-axis reductions are applied as repeated Synaptik reductions. |
| `Softmax`, `LogSoftmax` | Axis normalization ops. |
| `Constant` | Tensor initializer in graph-node form. |

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
