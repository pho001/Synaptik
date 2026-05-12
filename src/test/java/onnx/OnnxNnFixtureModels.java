package onnx;

import tensor.DataType;
import tensor.Tensor;
import tensor.options.Conv2dOptions;
import tensor.options.Pool2dOptions;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

public final class OnnxNnFixtureModels {
    private OnnxNnFixtureModels() {
    }

    public record Fixture(
            String fileName,
            OnnxModel model,
            String outputName,
            int[] expectedShape,
            double[] expectedOutput,
            Map<String, Object> inputs
    ) {
    }

    public static Map<String, Fixture> fixtures() {
        Map<String, Fixture> out = new LinkedHashMap<>();
        add(out, conv2dNoBias());
        add(out, conv2dBiasStridePadding());
        add(out, maxPool2d());
        add(out, averagePool2d());
        add(out, layerNormalization());
        add(out, batchNormalization());
        return out;
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            throw new IllegalArgumentException("Usage: OnnxNnFixtureModels <output-dir>");
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

    private static Fixture conv2dNoBias() {
        Tensor input = new Tensor(new float[9], new int[]{1, 1, 3, 3}, null, "input", DataType.FLOAT32);
        Tensor weight = new Tensor(new float[4], new int[]{1, 1, 2, 2}, null, "weight", DataType.FLOAT32);
        Tensor y = input.conv2d(weight, Conv2dOptions.defaults());
        y.setLabel("y");
        return fixture(
                "conv2d_no_bias.onnx",
                y,
                new int[]{1, 1, 2, 2},
                new double[]{6.0, 8.0, 12.0, 14.0},
                Map.of(
                        "input", new float[]{1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f},
                        "weight", new float[]{1f, 0f, 0f, 1f}
                )
        );
    }

    private static Fixture conv2dBiasStridePadding() {
        Tensor input = new Tensor(new float[16], new int[]{1, 1, 4, 4}, null, "input", DataType.FLOAT32);
        Tensor weight = new Tensor(new float[4], new int[]{1, 1, 2, 2}, null, "weight", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[1], new int[]{1}, null, "bias", DataType.FLOAT32);
        Tensor y = input.conv2d(weight, bias, Conv2dOptions.defaults().withStride(2, 2).withPadding(1, 1));
        y.setLabel("y");
        return fixture(
                "conv2d_bias_stride_padding.onnx",
                y,
                new int[]{1, 1, 3, 3},
                new double[]{1.5, 3.5, 0.5, 9.5, 17.5, 8.5, 0.5, 14.5, 16.5},
                Map.of(
                        "input", new float[]{1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f, 15f, 16f},
                        "weight", new float[]{1f, 0f, 0f, 1f},
                        "bias", new float[]{0.5f}
                )
        );
    }

    private static Fixture maxPool2d() {
        Tensor input = new Tensor(new float[16], new int[]{1, 1, 4, 4}, null, "input", DataType.FLOAT32);
        Tensor y = input.maxPool2d(Pool2dOptions.square(2));
        y.setLabel("y");
        return fixture(
                "max_pool2d.onnx",
                y,
                new int[]{1, 1, 2, 2},
                new double[]{6.0, 8.0, 14.0, 16.0},
                Map.of("input", new float[]{1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f, 15f, 16f})
        );
    }

    private static Fixture averagePool2d() {
        Tensor input = new Tensor(new float[16], new int[]{1, 1, 4, 4}, null, "input", DataType.FLOAT32);
        Tensor y = input.avgPool2d(Pool2dOptions.square(2));
        y.setLabel("y");
        return fixture(
                "average_pool2d.onnx",
                y,
                new int[]{1, 1, 2, 2},
                new double[]{3.5, 5.5, 11.5, 13.5},
                Map.of("input", new float[]{1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f, 15f, 16f})
        );
    }

    private static Fixture layerNormalization() {
        Tensor input = new Tensor(new float[6], new int[]{2, 3}, null, "input", DataType.FLOAT32);
        Tensor scale = new Tensor(new float[]{1f, 1f, 1f}, new int[]{3}, null, "scale", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{0f, 0f, 0f}, new int[]{3}, null, "bias", DataType.FLOAT32);
        Tensor y = input.layerNorm(scale, bias, 1.0e-12);
        y.setLabel("y");
        return fixture(
                "layer_normalization.onnx",
                y,
                new int[]{2, 3},
                new double[]{-1.224744, 0.0, 1.224744, -1.224744, 0.0, 1.224744},
                Map.of(
                        "input", new float[]{1f, 2f, 3f, 2f, 4f, 6f},
                        "scale", new float[]{1f, 1f, 1f},
                        "bias", new float[]{0f, 0f, 0f}
                )
        );
    }

    private static Fixture batchNormalization() {
        return fixture(
                "batch_normalization.onnx",
                model("batch_normalization", graph -> graph
                        .addInput(OnnxTensorProtoUtil.valueInfo("input", DataType.FLOAT32, new int[]{1, 2, 1, 2}))
                        .addInitializer(OnnxTensorProtoUtil.tensorInitializer("scale",
                                new Tensor(new float[]{2f, 3f}, new int[]{2}, null, "scale", DataType.FLOAT32)))
                        .addInitializer(OnnxTensorProtoUtil.tensorInitializer("bias",
                                new Tensor(new float[]{1f, -1f}, new int[]{2}, null, "bias", DataType.FLOAT32)))
                        .addInitializer(OnnxTensorProtoUtil.tensorInitializer("mean",
                                new Tensor(new float[]{2f, 12f}, new int[]{2}, null, "mean", DataType.FLOAT32)))
                        .addInitializer(OnnxTensorProtoUtil.tensorInitializer("variance",
                                new Tensor(new float[]{1f, 4f}, new int[]{2}, null, "variance", DataType.FLOAT32)))
                        .addNode(nodeBuilder("batch_norm", "BatchNormalization", "y", "input", "scale", "bias", "mean", "variance")
                                .addAttribute(floatAttr("epsilon", 1.0e-12f))
                                .build())
                        .addOutput(OnnxTensorProtoUtil.valueInfo("y", DataType.FLOAT32, new int[]{1, 2, 1, 2}))),
                new int[]{1, 2, 1, 2},
                new double[]{-1.0, 3.0, -4.0, 2.0},
                Map.of("input", new float[]{1f, 3f, 10f, 14f})
        );
    }

    private static Fixture fixture(
            String fileName,
            Tensor output,
            int[] expectedShape,
            double[] expectedOutput,
            Map<String, Object> inputs
    ) {
        return fixture(
                fileName,
                Onnx.exportModel(output, OnnxExportOptions.defaults().withLeafTensorPolicy(OnnxLeafTensorPolicy.INPUTS)),
                expectedShape,
                expectedOutput,
                inputs
        );
    }

    private static Fixture fixture(
            String fileName,
            OnnxModel model,
            int[] expectedShape,
            double[] expectedOutput,
            Map<String, Object> inputs
    ) {
        return new Fixture(fileName, model, "y", expectedShape, expectedOutput, inputs);
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
