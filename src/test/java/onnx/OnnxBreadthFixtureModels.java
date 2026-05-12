package onnx;

import tensor.DataType;
import tensor.Tensor;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

public final class OnnxBreadthFixtureModels {
    private OnnxBreadthFixtureModels() {
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
        add(out, constantPad());
        add(out, splitAxis1());
        add(out, tileRank2());
        add(out, argMaxKeepDimsTrue());
        add(out, argMaxKeepDimsFalse());
        add(out, reduceProdAxis());
        add(out, globalAveragePool());
        return out;
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            throw new IllegalArgumentException("Usage: OnnxBreadthFixtureModels <output-dir>");
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

    private static Fixture constantPad() {
        Tensor input = new Tensor(new float[4], new int[]{2, 2}, null, "input", DataType.FLOAT32);
        Tensor y = input.pad(new int[]{1, 0}, new int[]{0, 1}, -1.0);
        y.setLabel("y");
        return fixture(
                "pad_constant.onnx",
                y,
                Map.of("y", expected(new int[]{3, 3}, DataType.FLOAT32, -1, -1, -1, 1, 2, -1, 3, 4, -1)),
                Map.of("input", new float[]{1f, 2f, 3f, 4f})
        );
    }

    private static Fixture splitAxis1() {
        OnnxModel model = model("split_axis1", graph -> graph
                .addInput(OnnxTensorProtoUtil.valueInfo("input", DataType.FLOAT32, new int[]{2, 4}))
                .addInitializer(OnnxTensorProtoUtil.int64Initializer("split_sizes", new long[]{1, 3}))
                .addNode(nodeBuilder("split", "Split", "left", "input", "split_sizes")
                        .addOutput("right")
                        .addAttribute(intAttr("axis", 1))
                        .build())
                .addOutput(OnnxTensorProtoUtil.valueInfo("left", DataType.FLOAT32, new int[]{2, 1}))
                .addOutput(OnnxTensorProtoUtil.valueInfo("right", DataType.FLOAT32, new int[]{2, 3})));
        return fixture(
                "split_axis1.onnx",
                model,
                Map.of(
                        "left", expected(new int[]{2, 1}, DataType.FLOAT32, 1, 5),
                        "right", expected(new int[]{2, 3}, DataType.FLOAT32, 2, 3, 4, 6, 7, 8)
                ),
                Map.of("input", new float[]{1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f})
        );
    }

    private static Fixture tileRank2() {
        Tensor input = new Tensor(new float[4], new int[]{2, 2}, null, "input", DataType.FLOAT32);
        Tensor y = input.tile(2, 3);
        y.setLabel("y");
        return fixture(
                "tile_rank2.onnx",
                y,
                Map.of("y", expected(new int[]{4, 6}, DataType.FLOAT32,
                        1, 2, 1, 2, 1, 2,
                        3, 4, 3, 4, 3, 4,
                        1, 2, 1, 2, 1, 2,
                        3, 4, 3, 4, 3, 4)),
                Map.of("input", new float[]{1f, 2f, 3f, 4f})
        );
    }

    private static Fixture argMaxKeepDimsTrue() {
        Tensor input = new Tensor(new float[6], new int[]{2, 3}, null, "input", DataType.FLOAT32);
        Tensor y = input.argMax(1, true);
        y.setLabel("y");
        return fixture(
                "argmax_keepdims_true.onnx",
                y,
                Map.of("y", expected(new int[]{2, 1}, DataType.INT32, 1, 0)),
                Map.of("input", new float[]{1f, 4f, 4f, 7f, 6f, 7f})
        );
    }

    private static Fixture argMaxKeepDimsFalse() {
        Tensor input = new Tensor(new float[6], new int[]{2, 3}, null, "input", DataType.FLOAT32);
        Tensor y = input.argMax(1, false);
        y.setLabel("y");
        return fixture(
                "argmax_keepdims_false.onnx",
                y,
                Map.of("y", expected(new int[]{2}, DataType.INT32, 1, 0)),
                Map.of("input", new float[]{1f, 4f, 4f, 7f, 6f, 7f})
        );
    }

    private static Fixture reduceProdAxis() {
        Tensor input = new Tensor(new float[6], new int[]{2, 3}, null, "input", DataType.FLOAT32);
        Tensor y = input.prod(1, false);
        y.setLabel("y");
        return fixture(
                "reduceprod_axis.onnx",
                y,
                Map.of("y", expected(new int[]{2}, DataType.FLOAT32, 6, 120)),
                Map.of("input", new float[]{1f, 2f, 3f, 4f, 5f, 6f})
        );
    }

    private static Fixture globalAveragePool() {
        OnnxModel model = model("global_average_pool", graph -> graph
                .addInput(OnnxTensorProtoUtil.valueInfo("input", DataType.FLOAT32, new int[]{1, 2, 2, 2}))
                .addNode(nodeBuilder("gap", "GlobalAveragePool", "y", "input").build())
                .addOutput(OnnxTensorProtoUtil.valueInfo("y", DataType.FLOAT32, new int[]{1, 2, 1, 1})));
        return fixture(
                "global_average_pool.onnx",
                model,
                Map.of("y", expected(new int[]{1, 2, 1, 1}, DataType.FLOAT32, 2.5, 25.0)),
                Map.of("input", new float[]{1f, 2f, 3f, 4f, 10f, 20f, 30f, 40f})
        );
    }

    private static Fixture fixture(
            String fileName,
            Tensor output,
            Map<String, ExpectedOutput> outputs,
            Map<String, Object> inputs
    ) {
        return fixture(
                fileName,
                Onnx.exportModel(output, OnnxExportOptions.defaults().withLeafTensorPolicy(OnnxLeafTensorPolicy.INPUTS)),
                outputs,
                inputs
        );
    }

    private static Fixture fixture(
            String fileName,
            OnnxModel model,
            Map<String, ExpectedOutput> outputs,
            Map<String, Object> inputs
    ) {
        return new Fixture(fileName, model, outputs, inputs);
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
}
