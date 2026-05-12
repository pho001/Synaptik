package onnx;

import tensor.DataType;
import tensor.Tensor;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

public final class OnnxCompatibilityFixtureModels {
    private OnnxCompatibilityFixtureModels() {
    }

    enum ExpectedStatus {
        IMPORTED,
        EXECUTED,
        REJECTED_WITH_REASON
    }

    public record ExpectedOutput(
            int[] shape,
            DataType dataType,
            double[] values
    ) {
    }

    public record Fixture(
            String fileName,
            OnnxModel model,
            ExpectedStatus status,
            String rejectionReason,
            Map<String, ExpectedOutput> outputs,
            Map<String, Object> inputs
    ) {
    }

    public static Map<String, Fixture> fixtures() {
        Map<String, Fixture> out = new LinkedHashMap<>();
        add(out, activationMlp());
        add(out, softplusErfTiny());
        add(out, convPoolClassifierTiny());
        add(out, shapeHelperReduceTiny());
        add(out, residualMlpTiny());
        add(out, layernormActivationMlp());
        add(out, convBiasActivationPool());
        add(out, broadcastBiasWhereTiny());
        add(out, shapeStaticLayoutChain());
        add(out, rejectedNonZeroDynamicShape());
        add(out, rejectedDynamicReduceAxes());
        return out;
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            throw new IllegalArgumentException("Usage: OnnxCompatibilityFixtureModels <output-dir>");
        }
        Path root = Path.of(args[0]);
        Files.createDirectories(root);
        for (Fixture fixture : fixtures().values()) {
            fixture.model().write(root.resolve(fixture.fileName()));
        }
    }

    private static void add(Map<String, Fixture> out, Fixture fixture) {
        out.put(fixture.fileName(), fixture);
    }

    private static Fixture activationMlp() {
        OnnxModel model = model("activation_mlp", graph -> graph
                .addInput(OnnxTensorProtoUtil.valueInfo("x", DataType.FLOAT32, new int[]{1, 3}))
                .addInitializer(OnnxTensorProtoUtil.tensorInitializer("w1",
                        new Tensor(new float[]{1f, 0.5f, -1f, 2f, 0.25f, -0.5f}, new int[]{3, 2}, null, "w1", DataType.FLOAT32)))
                .addInitializer(OnnxTensorProtoUtil.tensorInitializer("b1",
                        new Tensor(new float[]{0.1f, -0.2f}, new int[]{2}, null, "b1", DataType.FLOAT32)))
                .addInitializer(OnnxTensorProtoUtil.tensorInitializer("w2",
                        new Tensor(new float[]{0.5f, -1f, 1.5f, 0.25f}, new int[]{2, 2}, null, "w2", DataType.FLOAT32)))
                .addInitializer(OnnxTensorProtoUtil.tensorInitializer("b2",
                        new Tensor(new float[]{0f, 1f}, new int[]{2}, null, "b2", DataType.FLOAT32)))
                .addNode(nodeBuilder("gemm1", "Gemm", "h0", "x", "w1", "b1").build())
                .addNode(nodeBuilder("leaky", "LeakyRelu", "h1", "h0")
                        .addAttribute(floatAttr("alpha", 0.1f))
                        .build())
                .addNode(nodeBuilder("gemm2", "Gemm", "logits", "h1", "w2", "b2").build())
                .addNode(nodeBuilder("hard", "HardSigmoid", "y", "logits")
                        .addAttribute(floatAttr("alpha", 0.2f))
                        .addAttribute(floatAttr("beta", 0.5f))
                        .build())
                .addOutput(OnnxTensorProtoUtil.valueInfo("y", DataType.FLOAT32, new int[]{1, 2})));
        return executable(
                "activation_mlp.onnx",
                model,
                Map.of("y", expected(new int[]{1, 2}, DataType.FLOAT32, 0.704, 0.03525)),
                Map.of("x", new float[]{1f, -2f, 0.5f})
        );
    }

    private static Fixture softplusErfTiny() {
        OnnxModel model = model("softplus_erf_tiny", graph -> graph
                .addInput(OnnxTensorProtoUtil.valueInfo("x", DataType.FLOAT32, new int[]{3}))
                .addNode(nodeBuilder("erf", "Erf", "erf", "x").build())
                .addNode(nodeBuilder("softplus", "Softplus", "softplus", "x").build())
                .addNode(nodeBuilder("add", "Add", "y", "erf", "softplus").build())
                .addOutput(OnnxTensorProtoUtil.valueInfo("y", DataType.FLOAT32, new int[]{3})));
        return executable(
                "softplus_erf_tiny.onnx",
                model,
                Map.of("y", expected(new int[]{3}, DataType.FLOAT32,
                        utils.SpecialFunctions.erf(-1.0) + Math.log(Math.exp(-1.0) + 1.0),
                        Math.log(2.0),
                        utils.SpecialFunctions.erf(1.0) + Math.log(Math.E + 1.0))),
                Map.of("x", new float[]{-1f, 0f, 1f})
        );
    }

    private static Fixture convPoolClassifierTiny() {
        double z0 = 2.8d;
        double z1 = 5.6d;
        double denom = Math.exp(z0) + Math.exp(z1);
        OnnxModel model = model("conv_pool_classifier_tiny", graph -> graph
                .addInput(OnnxTensorProtoUtil.valueInfo("x", DataType.FLOAT32, new int[]{1, 1, 3, 3}))
                .addInitializer(OnnxTensorProtoUtil.tensorInitializer("w",
                        new Tensor(new float[]{1f, 1f, 1f, 1f}, new int[]{1, 1, 2, 2}, null, "w", DataType.FLOAT32)))
                .addInitializer(OnnxTensorProtoUtil.tensorInitializer("fc_w",
                        new Tensor(new float[]{0.1f, 0.2f}, new int[]{1, 2}, null, "fc_w", DataType.FLOAT32)))
                .addInitializer(OnnxTensorProtoUtil.tensorInitializer("fc_b",
                        new Tensor(new float[]{0f, 0f}, new int[]{2}, null, "fc_b", DataType.FLOAT32)))
                .addNode(nodeBuilder("conv", "Conv", "conv", "x", "w")
                        .addAttribute(intsAttr("kernel_shape", 2, 2))
                        .build())
                .addNode(nodeBuilder("relu", "Relu", "relu", "conv").build())
                .addNode(nodeBuilder("pool", "MaxPool", "pool", "relu")
                        .addAttribute(intsAttr("kernel_shape", 2, 2))
                        .addAttribute(intsAttr("strides", 2, 2))
                        .build())
                .addNode(nodeBuilder("flatten", "Flatten", "flat", "pool")
                        .addAttribute(intAttr("axis", 1))
                        .build())
                .addNode(nodeBuilder("gemm", "Gemm", "logits", "flat", "fc_w", "fc_b").build())
                .addNode(nodeBuilder("softmax", "Softmax", "y", "logits")
                        .addAttribute(intAttr("axis", 1))
                        .build())
                .addOutput(OnnxTensorProtoUtil.valueInfo("y", DataType.FLOAT32, new int[]{1, 2})));
        return executable(
                "conv_pool_classifier_tiny.onnx",
                model,
                Map.of("y", expected(new int[]{1, 2}, DataType.FLOAT32, Math.exp(z0) / denom, Math.exp(z1) / denom)),
                Map.of("x", new float[]{1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f})
        );
    }

    private static Fixture shapeHelperReduceTiny() {
        OnnxModel model = model("shape_helper_reduce_tiny", graph -> graph
                .addInput(OnnxTensorProtoUtil.valueInfo("x", DataType.FLOAT32, new int[]{2, 3}))
                .addInitializer(OnnxTensorProtoUtil.int64Initializer("idx0", new long[]{0}))
                .addInitializer(OnnxTensorProtoUtil.int64Initializer("axes0", new long[]{0}))
                .addInitializer(OnnxTensorProtoUtil.int64Initializer("three", new long[]{3}))
                .addInitializer(OnnxTensorProtoUtil.int64Initializer("reduce_axes", new long[]{1}))
                .addNode(nodeBuilder("shape", "Shape", "shape", "x").build())
                .addNode(nodeBuilder("gather", "Gather", "dim0_scalar", "shape", "idx0").build())
                .addNode(nodeBuilder("unsqueeze", "Unsqueeze", "dim0", "dim0_scalar", "axes0").build())
                .addNode(nodeBuilder("concat", "Concat", "target_shape", "dim0", "three")
                        .addAttribute(intAttr("axis", 0))
                        .build())
                .addNode(nodeBuilder("reshape", "Reshape", "reshaped", "x", "target_shape").build())
                .addNode(nodeBuilder("reduce_l2", "ReduceL2", "y", "reshaped", "reduce_axes")
                        .addAttribute(intAttr("keepdims", 0))
                        .build())
                .addOutput(OnnxTensorProtoUtil.valueInfo("y", DataType.FLOAT32, new int[]{2})));
        return executable(
                "shape_helper_reduce_tiny.onnx",
                model,
                Map.of("y", expected(new int[]{2}, DataType.FLOAT32, Math.sqrt(14.0), Math.sqrt(77.0))),
                Map.of("x", new float[]{1f, 2f, 3f, 4f, 5f, 6f})
        );
    }

    private static Fixture residualMlpTiny() {
        OnnxModel model = model("residual_mlp_tiny", graph -> graph
                .addInput(OnnxTensorProtoUtil.valueInfo("x", DataType.FLOAT32, new int[]{1, 3}))
                .addInitializer(OnnxTensorProtoUtil.tensorInitializer("w",
                        new Tensor(new float[]{
                                1f, 0f, 0f,
                                0f, 1f, 0f,
                                0f, 0f, 1f
                        }, new int[]{3, 3}, null, "w", DataType.FLOAT32)))
                .addInitializer(OnnxTensorProtoUtil.tensorInitializer("b",
                        new Tensor(new float[]{0.5f, 0.5f, -4f}, new int[]{3}, null, "b", DataType.FLOAT32)))
                .addNode(nodeBuilder("gemm", "Gemm", "h0", "x", "w", "b").build())
                .addNode(nodeBuilder("relu", "Relu", "h1", "h0").build())
                .addNode(nodeBuilder("residual", "Add", "y", "h1", "x").build())
                .addOutput(OnnxTensorProtoUtil.valueInfo("y", DataType.FLOAT32, new int[]{1, 3})));
        return executable(
                "residual_mlp_tiny.onnx",
                model,
                Map.of("y", expected(new int[]{1, 3}, DataType.FLOAT32, 2.5, -2.0, 3.0)),
                Map.of("x", new float[]{1f, -2f, 3f})
        );
    }

    private static Fixture layernormActivationMlp() {
        double n0 = -1.3416407864998738d;
        double n3 = 1.3416407864998738d;
        OnnxModel model = model("layernorm_activation_mlp", graph -> graph
                .addInput(OnnxTensorProtoUtil.valueInfo("x", DataType.FLOAT32, new int[]{1, 4}))
                .addInitializer(OnnxTensorProtoUtil.tensorInitializer("scale",
                        new Tensor(new float[]{1f, 1f, 1f, 1f}, new int[]{4}, null, "scale", DataType.FLOAT32)))
                .addInitializer(OnnxTensorProtoUtil.tensorInitializer("bias",
                        new Tensor(new float[]{0f, 0f, 0f, 0f}, new int[]{4}, null, "bias", DataType.FLOAT32)))
                .addInitializer(OnnxTensorProtoUtil.tensorInitializer("w",
                        new Tensor(new float[]{
                                1f, 0f,
                                0f, 0f,
                                0f, 0f,
                                0f, 1f
                        }, new int[]{4, 2}, null, "w", DataType.FLOAT32)))
                .addInitializer(OnnxTensorProtoUtil.tensorInitializer("b",
                        new Tensor(new float[]{0.1f, -0.1f}, new int[]{2}, null, "b", DataType.FLOAT32)))
                .addNode(nodeBuilder("layernorm", "LayerNormalization", "norm", "x", "scale", "bias")
                        .addAttribute(intAttr("axis", -1))
                        .addAttribute(floatAttr("epsilon", 1.0e-12f))
                        .build())
                .addNode(nodeBuilder("elu", "Elu", "act", "norm")
                        .addAttribute(floatAttr("alpha", 1.0f))
                        .build())
                .addNode(nodeBuilder("gemm", "Gemm", "y", "act", "w", "b").build())
                .addOutput(OnnxTensorProtoUtil.valueInfo("y", DataType.FLOAT32, new int[]{1, 2})));
        return executable(
                "layernorm_activation_mlp.onnx",
                model,
                Map.of("y", expected(new int[]{1, 2}, DataType.FLOAT32, Math.exp(n0) - 1.0d + 0.1d, n3 - 0.1d)),
                Map.of("x", new float[]{1f, 2f, 3f, 4f})
        );
    }

    private static Fixture convBiasActivationPool() {
        OnnxModel model = model("conv_bias_activation_pool", graph -> graph
                .addInput(OnnxTensorProtoUtil.valueInfo("x", DataType.FLOAT32, new int[]{1, 1, 3, 3}))
                .addInitializer(OnnxTensorProtoUtil.tensorInitializer("w",
                        new Tensor(new float[]{1f, 1f, 1f, 1f}, new int[]{1, 1, 2, 2}, null, "w", DataType.FLOAT32)))
                .addInitializer(OnnxTensorProtoUtil.tensorInitializer("b",
                        new Tensor(new float[]{-5f}, new int[]{1}, null, "b", DataType.FLOAT32)))
                .addNode(nodeBuilder("conv", "Conv", "conv", "x", "w", "b")
                        .addAttribute(intsAttr("kernel_shape", 2, 2))
                        .build())
                .addNode(nodeBuilder("hard_sigmoid", "HardSigmoid", "act", "conv")
                        .addAttribute(floatAttr("alpha", 0.05f))
                        .addAttribute(floatAttr("beta", 0.2f))
                        .build())
                .addNode(nodeBuilder("pool", "AveragePool", "y", "act")
                        .addAttribute(intsAttr("kernel_shape", 2, 2))
                        .addAttribute(intsAttr("strides", 2, 2))
                        .build())
                .addOutput(OnnxTensorProtoUtil.valueInfo("y", DataType.FLOAT32, new int[]{1, 1, 1, 1})));
        return executable(
                "conv_bias_activation_pool.onnx",
                model,
                Map.of("y", expected(new int[]{1, 1, 1, 1}, DataType.FLOAT32, 0.825)),
                Map.of("x", new float[]{1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f})
        );
    }

    private static Fixture broadcastBiasWhereTiny() {
        OnnxModel model = model("broadcast_bias_where_tiny", graph -> graph
                .addInput(OnnxTensorProtoUtil.valueInfo("x", DataType.FLOAT32, new int[]{2, 3}))
                .addInput(OnnxTensorProtoUtil.valueInfo("fallback", DataType.FLOAT32, new int[]{1, 3}))
                .addInitializer(OnnxTensorProtoUtil.tensorInitializer("bias",
                        new Tensor(new float[]{0.5f, -2f, 1f}, new int[]{3}, null, "bias", DataType.FLOAT32)))
                .addInitializer(OnnxTensorProtoUtil.tensorInitializer("zero", Tensor.scalar(0.0d, DataType.FLOAT32)))
                .addNode(nodeBuilder("add", "Add", "biased", "x", "bias").build())
                .addNode(nodeBuilder("mask", "Greater", "mask", "biased", "zero").build())
                .addNode(nodeBuilder("where", "Where", "y", "mask", "biased", "fallback").build())
                .addOutput(OnnxTensorProtoUtil.valueInfo("y", DataType.FLOAT32, new int[]{2, 3})));
        return executable(
                "broadcast_bias_where_tiny.onnx",
                model,
                Map.of("y", expected(new int[]{2, 3}, DataType.FLOAT32, 1.5, 20.0, 4.0, 10.0, 20.0, 5.0)),
                Map.of(
                        "x", new float[]{1f, 2f, 3f, -1f, -2f, 4f},
                        "fallback", new float[]{10f, 20f, 30f}
                )
        );
    }

    private static Fixture shapeStaticLayoutChain() {
        OnnxModel model = model("shape_static_layout_chain", graph -> graph
                .addInput(OnnxTensorProtoUtil.valueInfo("x", DataType.FLOAT32, new int[]{2, 3}))
                .addInitializer(OnnxTensorProtoUtil.int64Initializer("idx1", new long[]{1}))
                .addInitializer(OnnxTensorProtoUtil.int64Initializer("axes0", new long[]{0}))
                .addInitializer(OnnxTensorProtoUtil.int64Initializer("two", new long[]{2}))
                .addInitializer(OnnxTensorProtoUtil.int64Initializer("repeats", new long[]{1, 1}))
                .addNode(nodeBuilder("shape", "Shape", "shape", "x").build())
                .addNode(nodeBuilder("gather_dim1", "Gather", "dim1_scalar", "shape", "idx1").build())
                .addNode(nodeBuilder("unsqueeze", "Unsqueeze", "dim1", "dim1_scalar", "axes0").build())
                .addNode(nodeBuilder("concat", "Concat", "target_shape", "dim1", "two")
                        .addAttribute(intAttr("axis", 0))
                        .build())
                .addNode(nodeBuilder("reshape", "Reshape", "reshaped", "x", "target_shape").build())
                .addNode(nodeBuilder("transpose", "Transpose", "transposed", "reshaped")
                        .addAttribute(intsAttr("perm", 1, 0))
                        .build())
                .addNode(nodeBuilder("tile", "Tile", "y", "transposed", "repeats").build())
                .addOutput(OnnxTensorProtoUtil.valueInfo("y", DataType.FLOAT32, new int[]{2, 3})));
        return executable(
                "shape_static_layout_chain.onnx",
                model,
                Map.of("y", expected(new int[]{2, 3}, DataType.FLOAT32, 1.0, 3.0, 5.0, 2.0, 4.0, 6.0)),
                Map.of("x", new float[]{1f, 2f, 3f, 4f, 5f, 6f})
        );
    }

    private static Fixture rejectedNonZeroDynamicShape() {
        OnnxModel model = model("rejected_nonzero_dynamic_shape", graph -> graph
                .addInput(OnnxTensorProtoUtil.valueInfo("x", DataType.FLOAT32, new int[]{3}))
                .addNode(nodeBuilder("nonzero", "NonZero", "idx", "x").build())
                .addOutput(OnnxTensorProtoUtil.valueInfo("idx", DataType.INT32, new int[]{1, 3})));
        return new Fixture(
                "rejected_nonzero_dynamic_shape.onnx",
                model,
                ExpectedStatus.REJECTED_WITH_REASON,
                "dynamic-shape",
                Map.of(),
                Map.of("x", new float[]{0f, 1f, 2f})
        );
    }

    private static Fixture rejectedDynamicReduceAxes() {
        OnnxModel model = model("rejected_dynamic_reduce_axes", graph -> graph
                .addInput(OnnxTensorProtoUtil.valueInfo("x", DataType.FLOAT32, new int[]{2, 3}))
                .addInput(OnnxTensorProtoUtil.valueInfo("axes", DataType.INT32, new int[]{1}))
                .addNode(nodeBuilder("reduce", "ReduceMean", "y", "x", "axes")
                        .addAttribute(intAttr("keepdims", 0))
                        .build())
                .addOutput(OnnxTensorProtoUtil.valueInfo("y", DataType.FLOAT32, new int[]{2})));
        return new Fixture(
                "rejected_dynamic_reduce_axes.onnx",
                model,
                ExpectedStatus.REJECTED_WITH_REASON,
                "constant initializer or Constant node",
                Map.of(),
                Map.of("x", new float[]{1f, 2f, 3f, 4f, 5f, 6f})
        );
    }

    private static Fixture executable(String fileName, OnnxModel model, Map<String, ExpectedOutput> outputs, Map<String, Object> inputs) {
        return new Fixture(fileName, model, ExpectedStatus.EXECUTED, "", outputs, inputs);
    }

    private static ExpectedOutput expected(int[] shape, DataType dataType, double... values) {
        return new ExpectedOutput(shape, dataType, values);
    }

    private static OnnxModel model(String graphName, java.util.function.Consumer<OnnxProto.GraphProto.Builder> configure) {
        OnnxProto.GraphProto.Builder graph = OnnxProto.GraphProto.newBuilder().setName(graphName);
        configure.accept(graph);
        return new OnnxModel(OnnxProto.ModelProto.newBuilder()
                .setIrVersion(OnnxProto.Version.IR_VERSION_2023_5_5.getNumber())
                .addOpsetImport(OnnxProto.OperatorSetIdProto.newBuilder()
                        .setDomain("")
                        .setVersion(OnnxExportOptions.DEFAULT_OPSET))
                .setGraph(graph)
                .build());
    }

    private static OnnxProto.NodeProto.Builder nodeBuilder(String name, String opType, String output, String... inputs) {
        OnnxProto.NodeProto.Builder node = OnnxProto.NodeProto.newBuilder()
                .setName(name)
                .setOpType(opType);
        for (String input : inputs) {
            node.addInput(input);
        }
        return node.addOutput(output);
    }

    private static OnnxProto.AttributeProto intAttr(String name, long value) {
        return OnnxProto.AttributeProto.newBuilder().setName(name).setI(value).build();
    }

    private static OnnxProto.AttributeProto intsAttr(String name, long... values) {
        OnnxProto.AttributeProto.Builder attr = OnnxProto.AttributeProto.newBuilder().setName(name);
        for (long value : values) {
            attr.addInts(value);
        }
        return attr.build();
    }

    private static OnnxProto.AttributeProto floatAttr(String name, float value) {
        return OnnxProto.AttributeProto.newBuilder().setName(name).setF(value).build();
    }
}
