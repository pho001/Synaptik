package onnx;

import tensor.DataType;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

public final class OnnxActivationFixtureModels {
    private OnnxActivationFixtureModels() {
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
            Map<String, ExpectedOutput> outputs,
            Map<String, Object> inputs
    ) {
    }

    public static Map<String, Fixture> fixtures() {
        Map<String, Fixture> out = new LinkedHashMap<>();
        Fixture fixture = activationMathBreadth();
        out.put(fixture.fileName(), fixture);
        return out;
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            throw new IllegalArgumentException("Usage: OnnxActivationFixtureModels <output-dir>");
        }
        Path root = Path.of(args[0]);
        Files.createDirectories(root);
        for (Fixture fixture : fixtures().values()) {
            fixture.model().write(root.resolve(fixture.fileName()));
        }
    }

    private static Fixture activationMathBreadth() {
        OnnxModel model = model("activation_math_breadth", graph -> graph
                .addInput(OnnxTensorProtoUtil.valueInfo("input", DataType.FLOAT32, new int[]{4}))
                .addNode(nodeBuilder("reciprocal", "Reciprocal", "reciprocal", "input").build())
                .addNode(nodeBuilder("erf", "Erf", "erf", "input").build())
                .addNode(nodeBuilder("floor", "Floor", "floor", "input").build())
                .addNode(nodeBuilder("ceil", "Ceil", "ceil", "input").build())
                .addNode(nodeBuilder("sign", "Sign", "sign", "input").build())
                .addNode(nodeBuilder("leaky", "LeakyRelu", "leaky", "input")
                        .addAttribute(floatAttr("alpha", 0.2f))
                        .build())
                .addNode(nodeBuilder("elu", "Elu", "elu", "input")
                        .addAttribute(floatAttr("alpha", 1.5f))
                        .build())
                .addNode(nodeBuilder("hard", "HardSigmoid", "hard", "input")
                        .addAttribute(floatAttr("alpha", 0.25f))
                        .addAttribute(floatAttr("beta", 0.5f))
                        .build())
                .addNode(nodeBuilder("softplus", "Softplus", "softplus", "input").build())
                .addOutput(OnnxTensorProtoUtil.valueInfo("reciprocal", DataType.FLOAT32, new int[]{4}))
                .addOutput(OnnxTensorProtoUtil.valueInfo("erf", DataType.FLOAT32, new int[]{4}))
                .addOutput(OnnxTensorProtoUtil.valueInfo("floor", DataType.FLOAT32, new int[]{4}))
                .addOutput(OnnxTensorProtoUtil.valueInfo("ceil", DataType.FLOAT32, new int[]{4}))
                .addOutput(OnnxTensorProtoUtil.valueInfo("sign", DataType.FLOAT32, new int[]{4}))
                .addOutput(OnnxTensorProtoUtil.valueInfo("leaky", DataType.FLOAT32, new int[]{4}))
                .addOutput(OnnxTensorProtoUtil.valueInfo("elu", DataType.FLOAT32, new int[]{4}))
                .addOutput(OnnxTensorProtoUtil.valueInfo("hard", DataType.FLOAT32, new int[]{4}))
                .addOutput(OnnxTensorProtoUtil.valueInfo("softplus", DataType.FLOAT32, new int[]{4})));

        double[] input = {-1.5, -0.25, 0.5, 2.0};
        return new Fixture(
                "activation_math_breadth.onnx",
                model,
                Map.of(
                        "reciprocal", expected(new int[]{4}, DataType.FLOAT32, -0.6666667, -4.0, 2.0, 0.5),
                        "erf", expected(new int[]{4}, DataType.FLOAT32,
                                utils.SpecialFunctions.erf(input[0]),
                                utils.SpecialFunctions.erf(input[1]),
                                utils.SpecialFunctions.erf(input[2]),
                                utils.SpecialFunctions.erf(input[3])),
                        "floor", expected(new int[]{4}, DataType.FLOAT32, -2, -1, 0, 2),
                        "ceil", expected(new int[]{4}, DataType.FLOAT32, -1, 0, 1, 2),
                        "sign", expected(new int[]{4}, DataType.FLOAT32, -1, -1, 1, 1),
                        "leaky", expected(new int[]{4}, DataType.FLOAT32, -0.3, -0.05, 0.5, 2.0),
                        "elu", expected(new int[]{4}, DataType.FLOAT32,
                                1.5 * (Math.exp(-1.5) - 1.0),
                                1.5 * (Math.exp(-0.25) - 1.0),
                                0.5,
                                2.0),
                        "hard", expected(new int[]{4}, DataType.FLOAT32, 0.125, 0.4375, 0.625, 1.0),
                        "softplus", expected(new int[]{4}, DataType.FLOAT32,
                                Math.log(Math.exp(-1.5) + 1.0),
                                Math.log(Math.exp(-0.25) + 1.0),
                                Math.log(Math.exp(0.5) + 1.0),
                                Math.log(Math.exp(2.0) + 1.0))
                ),
                Map.of("input", new float[]{-1.5f, -0.25f, 0.5f, 2.0f})
        );
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

    private static OnnxProto.AttributeProto floatAttr(String name, float value) {
        return OnnxProto.AttributeProto.newBuilder().setName(name).setF(value).build();
    }
}
