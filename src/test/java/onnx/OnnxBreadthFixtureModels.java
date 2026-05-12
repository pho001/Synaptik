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
        add(out, constantOfShapeFloat());
        add(out, constantOfShapeBool());
        add(out, constantOfShapeInt32());
        add(out, rangeInt32Positive());
        add(out, rangeFloat32Negative());
        add(out, rangeInt64ShapeAxes());
        add(out, reduceL1Axis());
        add(out, reduceL2MultiAxisKeepDims());
        add(out, reduceLogSumAxis());
        add(out, reduceLogSumExpAxis());
        add(out, cumSumExclusiveReverse());
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

    private static Fixture constantOfShapeFloat() {
        OnnxModel model = model("constant_of_shape_float", graph -> graph
                .addInitializer(OnnxTensorProtoUtil.int64Initializer("shape", new long[]{2, 3}))
                .addNode(nodeBuilder("constant_of_shape", "ConstantOfShape", "y", "shape")
                        .addAttribute(tensorAttr("value", Tensor.scalar(2.5, DataType.FLOAT32)))
                        .build())
                .addOutput(OnnxTensorProtoUtil.valueInfo("y", DataType.FLOAT32, new int[]{2, 3})));
        return fixture(
                "constant_of_shape_float.onnx",
                model,
                Map.of("y", expected(new int[]{2, 3}, DataType.FLOAT32, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5)),
                Map.of()
        );
    }

    private static Fixture constantOfShapeBool() {
        OnnxModel model = model("constant_of_shape_bool", graph -> graph
                .addInitializer(OnnxTensorProtoUtil.int64Initializer("shape", new long[]{2}))
                .addNode(nodeBuilder("constant_of_shape", "ConstantOfShape", "y", "shape")
                        .addAttribute(tensorAttr("value", new Tensor(new byte[]{1}, new int[]{1}, null, "value", DataType.BOOL)))
                        .build())
                .addOutput(OnnxTensorProtoUtil.valueInfo("y", DataType.BOOL, new int[]{2})));
        return fixture(
                "constant_of_shape_bool.onnx",
                model,
                Map.of("y", expected(new int[]{2}, DataType.BOOL, 1, 1)),
                Map.of()
        );
    }

    private static Fixture constantOfShapeInt32() {
        OnnxModel model = model("constant_of_shape_int32", graph -> graph
                .addInitializer(OnnxTensorProtoUtil.int64Initializer("shape", new long[]{2, 2}))
                .addNode(nodeBuilder("constant_of_shape", "ConstantOfShape", "y", "shape")
                        .addAttribute(tensorAttr("value", Tensor.scalar(7, DataType.INT32)))
                        .build())
                .addOutput(OnnxTensorProtoUtil.valueInfo("y", DataType.INT32, new int[]{2, 2})));
        return fixture(
                "constant_of_shape_int32.onnx",
                model,
                Map.of("y", expected(new int[]{2, 2}, DataType.INT32, 7, 7, 7, 7)),
                Map.of()
        );
    }

    private static Fixture rangeInt32Positive() {
        OnnxModel model = model("range_int32_positive", graph -> graph
                .addInitializer(OnnxTensorProtoUtil.tensorInitializer("start", Tensor.scalar(0, DataType.INT32)))
                .addInitializer(OnnxTensorProtoUtil.tensorInitializer("limit", Tensor.scalar(6, DataType.INT32)))
                .addInitializer(OnnxTensorProtoUtil.tensorInitializer("delta", Tensor.scalar(2, DataType.INT32)))
                .addNode(nodeBuilder("range", "Range", "y", "start", "limit", "delta").build())
                .addOutput(OnnxTensorProtoUtil.valueInfo("y", DataType.INT32, new int[]{3})));
        return fixture(
                "range_int32_positive.onnx",
                model,
                Map.of("y", expected(new int[]{3}, DataType.INT32, 0, 2, 4)),
                Map.of()
        );
    }

    private static Fixture rangeFloat32Negative() {
        OnnxModel model = model("range_float32_negative", graph -> graph
                .addInitializer(OnnxTensorProtoUtil.tensorInitializer("start", Tensor.scalar(5, DataType.FLOAT32)))
                .addInitializer(OnnxTensorProtoUtil.tensorInitializer("limit", Tensor.scalar(-1, DataType.FLOAT32)))
                .addInitializer(OnnxTensorProtoUtil.tensorInitializer("delta", Tensor.scalar(-2, DataType.FLOAT32)))
                .addNode(nodeBuilder("range", "Range", "y", "start", "limit", "delta").build())
                .addOutput(OnnxTensorProtoUtil.valueInfo("y", DataType.FLOAT32, new int[]{3})));
        return fixture(
                "range_float32_negative.onnx",
                model,
                Map.of("y", expected(new int[]{3}, DataType.FLOAT32, 5, 3, 1)),
                Map.of()
        );
    }

    private static Fixture rangeInt64ShapeAxes() {
        OnnxModel model = model("range_int64_shape_axes", graph -> graph
                .addInput(OnnxTensorProtoUtil.valueInfo("input", DataType.FLOAT32, new int[]{2, 3}))
                .addInitializer(OnnxTensorProtoUtil.int64Initializer("start", new long[]{0}))
                .addInitializer(OnnxTensorProtoUtil.int64Initializer("limit", new long[]{2}))
                .addInitializer(OnnxTensorProtoUtil.int64Initializer("delta", new long[]{1}))
                .addNode(nodeBuilder("range", "Range", "axes", "start", "limit", "delta").build())
                .addNode(nodeBuilder("reduce", "ReduceSum", "y", "input", "axes")
                        .addAttribute(intAttr("keepdims", 1))
                        .build())
                .addOutput(OnnxTensorProtoUtil.valueInfo("y", DataType.FLOAT32, new int[]{1, 1})));
        return fixture(
                "range_int64_shape_axes.onnx",
                model,
                Map.of("y", expected(new int[]{1, 1}, DataType.FLOAT32, 21)),
                Map.of("input", new float[]{1f, 2f, 3f, 4f, 5f, 6f})
        );
    }

    private static Fixture reduceL1Axis() {
        OnnxModel model = reduceFixtureModel("reduce_l1_axis", "ReduceL1", new long[]{1}, 0, new int[]{2});
        return fixture(
                "reduce_l1_axis.onnx",
                model,
                Map.of("y", expected(new int[]{2}, DataType.FLOAT32, 6, 15)),
                Map.of("input", new float[]{-1f, 2f, -3f, 4f, -5f, 6f})
        );
    }

    private static Fixture reduceL2MultiAxisKeepDims() {
        OnnxModel model = reduceFixtureModel("reduce_l2_multi_axis", "ReduceL2", new long[]{0, 1}, 1, new int[]{1, 1});
        return fixture(
                "reduce_l2_multi_axis.onnx",
                model,
                Map.of("y", expected(new int[]{1, 1}, DataType.FLOAT32, Math.sqrt(91.0))),
                Map.of("input", new float[]{1f, 2f, 3f, 4f, 5f, 6f})
        );
    }

    private static Fixture reduceLogSumAxis() {
        OnnxModel model = reduceFixtureModel("reduce_log_sum_axis", "ReduceLogSum", new long[]{1}, 0, new int[]{2});
        return fixture(
                "reduce_log_sum_axis.onnx",
                model,
                Map.of("y", expected(new int[]{2}, DataType.FLOAT32, Math.log(6.0), Math.log(15.0))),
                Map.of("input", new float[]{1f, 2f, 3f, 4f, 5f, 6f})
        );
    }

    private static Fixture reduceLogSumExpAxis() {
        OnnxModel model = reduceFixtureModel("reduce_log_sum_exp_axis", "ReduceLogSumExp", new long[]{1}, 0, new int[]{2});
        return fixture(
                "reduce_log_sum_exp_axis.onnx",
                model,
                Map.of("y", expected(new int[]{2}, DataType.FLOAT32,
                        Math.log(Math.exp(0.0) + Math.exp(1.0) + Math.exp(2.0)),
                        Math.log(Math.exp(3.0) + Math.exp(4.0) + Math.exp(5.0)))),
                Map.of("input", new float[]{0f, 1f, 2f, 3f, 4f, 5f})
        );
    }

    private static Fixture cumSumExclusiveReverse() {
        Tensor input = new Tensor(new float[6], new int[]{2, 3}, null, "input", DataType.FLOAT32);
        Tensor y = input.cumSum(1, true, true);
        y.setLabel("y");
        return fixture(
                "cumsum_exclusive_reverse.onnx",
                y,
                Map.of("y", expected(new int[]{2, 3}, DataType.FLOAT32, 5, 3, 0, 11, 6, 0)),
                Map.of("input", new float[]{1f, 2f, 3f, 4f, 5f, 6f})
        );
    }

    private static OnnxModel reduceFixtureModel(String graphName, String opType, long[] axes, int keepDims, int[] outputShape) {
        return model(graphName, graph -> graph
                .addInput(OnnxTensorProtoUtil.valueInfo("input", DataType.FLOAT32, new int[]{2, 3}))
                .addInitializer(OnnxTensorProtoUtil.int64Initializer("axes", axes))
                .addNode(nodeBuilder("reduce", opType, "y", "input", "axes")
                        .addAttribute(intAttr("keepdims", keepDims))
                        .build())
                .addOutput(OnnxTensorProtoUtil.valueInfo("y", DataType.FLOAT32, outputShape)));
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

    private static OnnxProto.AttributeProto tensorAttr(String name, Tensor value) {
        return OnnxProto.AttributeProto.newBuilder()
                .setName(name)
                .setT(OnnxTensorProtoUtil.tensorInitializer("", value))
                .build();
    }
}
