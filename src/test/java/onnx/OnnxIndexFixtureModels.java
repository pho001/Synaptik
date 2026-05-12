package onnx;

import operations.index.ScatterReduction;
import tensor.DataType;
import tensor.Tensor;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

public final class OnnxIndexFixtureModels {
    private OnnxIndexFixtureModels() {
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
        add(out, gatherNdElementBatch0());
        add(out, gatherNdSliceBatch0());
        add(out, gatherNdBatch1());
        add(out, gatherNdBatch2());
        add(out, gatherElementsAxis0());
        add(out, gatherElementsAxis1());
        add(out, gatherElementsNegativeAxis());
        add(out, scatterElementsNoneAxis0());
        add(out, scatterElementsAddAxis1());
        add(out, scatterElementsNegativeAxis());
        add(out, scatterNdNone());
        add(out, scatterNdAdd());
        add(out, scatterNdSliceNone());
        add(out, scatterNdAddDuplicates());
        add(out, scatterNdMul());
        add(out, scatterNdMax());
        add(out, scatterNdMin());
        return out;
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            throw new IllegalArgumentException("Usage: OnnxIndexFixtureModels <output-dir>");
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

    private static Fixture gatherNdElementBatch0() {
        Tensor data = new Tensor(new float[6], new int[]{2, 3}, null, "data", DataType.FLOAT32);
        Tensor indices = new Tensor(new int[4], new int[]{2, 2}, null, "indices", DataType.INT32);
        Tensor y = data.gatherNd(indices);
        y.setLabel("y");
        return fixture(
                "gather_nd_element_batch0.onnx",
                y,
                new int[]{2},
                new double[]{30.0, 40.0},
                Map.of(
                        "data", new float[]{10f, 20f, 30f, 40f, 50f, 60f},
                        "indices", new int[]{0, 2, 1, 0}
                )
        );
    }

    private static Fixture gatherNdSliceBatch0() {
        Tensor data = new Tensor(new float[6], new int[]{2, 3}, null, "data", DataType.FLOAT32);
        Tensor indices = new Tensor(new int[2], new int[]{2, 1}, null, "indices", DataType.INT32);
        Tensor y = data.gatherNd(indices);
        y.setLabel("y");
        return fixture(
                "gather_nd_slice_batch0.onnx",
                y,
                new int[]{2, 3},
                new double[]{4.0, 5.0, 6.0, 1.0, 2.0, 3.0},
                Map.of(
                        "data", new float[]{1f, 2f, 3f, 4f, 5f, 6f},
                        "indices", new int[]{1, 0}
                )
        );
    }

    private static Fixture gatherNdBatch1() {
        Tensor data = new Tensor(new float[12], new int[]{2, 3, 2}, null, "data", DataType.FLOAT32);
        Tensor indices = new Tensor(new int[4], new int[]{2, 2, 1}, null, "indices", DataType.INT32);
        Tensor y = data.gatherNd(indices, 1);
        y.setLabel("y");
        return fixture(
                "gather_nd_batch1.onnx",
                y,
                new int[]{2, 2, 2},
                new double[]{5.0, 6.0, 1.0, 2.0, 9.0, 10.0, 7.0, 8.0},
                Map.of(
                        "data", new float[]{1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f},
                        "indices", new int[]{2, 0, 1, 0}
                )
        );
    }

    private static Fixture gatherNdBatch2() {
        Tensor data = new Tensor(new float[12], new int[]{2, 2, 3}, null, "data", DataType.FLOAT32);
        Tensor indices = new Tensor(new int[4], new int[]{2, 2, 1}, null, "indices", DataType.INT32);
        Tensor y = data.gatherNd(indices, 2);
        y.setLabel("y");
        return fixture(
                "gather_nd_batch2.onnx",
                y,
                new int[]{2, 2},
                new double[]{3.0, 4.0, 8.0, 12.0},
                Map.of(
                        "data", new float[]{1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f},
                        "indices", new int[]{2, 0, 1, 2}
                )
        );
    }

    private static Fixture gatherElementsAxis0() {
        Tensor data = new Tensor(new float[6], new int[]{3, 2}, null, "data", DataType.FLOAT32);
        Tensor indices = new Tensor(new int[4], new int[]{2, 2}, null, "indices", DataType.INT32);
        Tensor y = data.takeAlongAxis(indices, 0);
        y.setLabel("y");
        return fixture(
                "gather_elements_axis0.onnx",
                y,
                new int[]{2, 2},
                new double[]{5.0, 2.0, 3.0, 6.0},
                Map.of(
                        "data", new float[]{1f, 2f, 3f, 4f, 5f, 6f},
                        "indices", new int[]{2, 0, 1, 2}
                )
        );
    }

    private static Fixture gatherElementsAxis1() {
        Tensor data = new Tensor(new float[6], new int[]{2, 3}, null, "data", DataType.FLOAT32);
        Tensor indices = new Tensor(new int[4], new int[]{2, 2}, null, "indices", DataType.INT32);
        Tensor y = data.takeAlongAxis(indices, 1);
        y.setLabel("y");
        return fixture(
                "gather_elements_axis1.onnx",
                y,
                new int[]{2, 2},
                new double[]{3.0, 2.0, 4.0, 4.0},
                Map.of(
                        "data", new float[]{1f, 2f, 3f, 4f, 5f, 6f},
                        "indices", new int[]{2, 1, 0, 0}
                )
        );
    }

    private static Fixture gatherElementsNegativeAxis() {
        return fixture(
                "gather_elements_negative_axis.onnx",
                model("gather_elements_negative_axis", graph -> graph
                        .addInput(OnnxTensorProtoUtil.valueInfo("data", DataType.FLOAT32, new int[]{2, 3}))
                        .addInput(OnnxTensorProtoUtil.valueInfo("indices", DataType.INT32, new int[]{2, 2}))
                        .addNode(axisNode("gather_elements", "GatherElements", "y", -1, "data", "indices"))
                        .addOutput(OnnxTensorProtoUtil.valueInfo("y", DataType.FLOAT32, new int[]{2, 2}))),
                new int[]{2, 2},
                new double[]{3.0, 1.0, 4.0, 6.0},
                Map.of(
                        "data", new float[]{1f, 2f, 3f, 4f, 5f, 6f},
                        "indices", new int[]{-1, 0, 0, -1}
                )
        );
    }

    private static Fixture scatterElementsNoneAxis0() {
        Tensor data = new Tensor(new float[6], new int[]{3, 2}, null, "data", DataType.FLOAT32);
        Tensor indices = new Tensor(new int[4], new int[]{2, 2}, null, "indices", DataType.INT32);
        Tensor updates = new Tensor(new float[4], new int[]{2, 2}, null, "updates", DataType.FLOAT32);
        Tensor y = data.scatterElements(indices, updates, 0);
        y.setLabel("y");
        return fixture(
                "scatter_elements_none_axis0.onnx",
                y,
                new int[]{3, 2},
                new double[]{10.0, 50.0, 3.0, 4.0, 1.0, 20.0},
                Map.of(
                        "data", new float[]{1f, 2f, 3f, 4f, 5f, 6f},
                        "indices", new int[]{2, 0, 0, 2},
                        "updates", new float[]{1f, 50f, 10f, 20f}
                )
        );
    }

    private static Fixture scatterElementsAddAxis1() {
        Tensor data = new Tensor(new float[6], new int[]{2, 3}, null, "data", DataType.FLOAT32);
        Tensor indices = new Tensor(new int[4], new int[]{2, 2}, null, "indices", DataType.INT32);
        Tensor updates = new Tensor(new float[4], new int[]{2, 2}, null, "updates", DataType.FLOAT32);
        Tensor y = data.scatterElements(indices, updates, 1, ScatterReduction.ADD);
        y.setLabel("y");
        return fixture(
                "scatter_elements_add_axis1.onnx",
                y,
                new int[]{2, 3},
                new double[]{10.0, 26.0, 30.0, 47.0, 50.0, 69.0},
                Map.of(
                        "data", new float[]{10f, 20f, 30f, 40f, 50f, 60f},
                        "indices", new int[]{1, 1, 0, 2},
                        "updates", new float[]{1f, 5f, 7f, 9f}
                )
        );
    }

    private static Fixture scatterElementsNegativeAxis() {
        return fixture(
                "scatter_elements_negative_axis.onnx",
                model("scatter_elements_negative_axis", graph -> graph
                        .addInput(OnnxTensorProtoUtil.valueInfo("data", DataType.FLOAT32, new int[]{2, 3}))
                        .addInput(OnnxTensorProtoUtil.valueInfo("indices", DataType.INT32, new int[]{2, 2}))
                        .addInput(OnnxTensorProtoUtil.valueInfo("updates", DataType.FLOAT32, new int[]{2, 2}))
                        .addNode(axisNode("scatter_elements", "ScatterElements", "y", -1, "data", "indices", "updates"))
                        .addOutput(OnnxTensorProtoUtil.valueInfo("y", DataType.FLOAT32, new int[]{2, 3}))),
                new int[]{2, 3},
                new double[]{5.0, 20.0, 1.0, 7.0, 50.0, 9.0},
                Map.of(
                        "data", new float[]{10f, 20f, 30f, 40f, 50f, 60f},
                        "indices", new int[]{-1, 0, 0, -1},
                        "updates", new float[]{1f, 5f, 7f, 9f}
                )
        );
    }

    private static Fixture scatterNdNone() {
        Tensor data = new Tensor(new float[6], new int[]{2, 3}, null, "data", DataType.FLOAT32);
        Tensor indices = new Tensor(new int[4], new int[]{2, 2}, null, "indices", DataType.INT32);
        Tensor updates = new Tensor(new float[2], new int[]{2}, null, "updates", DataType.FLOAT32);
        Tensor y = data.scatterNd(indices, updates);
        y.setLabel("y");
        return fixture(
                "scatter_nd_none.onnx",
                y,
                new int[]{2, 3},
                new double[]{10.0, 20.0, 1.0, 7.0, 50.0, 60.0},
                Map.of(
                        "data", new float[]{10f, 20f, 30f, 40f, 50f, 60f},
                        "indices", new int[]{0, 2, 1, 0},
                        "updates", new float[]{1f, 7f}
                )
        );
    }

    private static Fixture scatterNdAdd() {
        Tensor data = new Tensor(new float[6], new int[]{2, 3}, null, "data", DataType.FLOAT32);
        Tensor indices = new Tensor(new int[4], new int[]{2, 2}, null, "indices", DataType.INT32);
        Tensor updates = new Tensor(new float[2], new int[]{2}, null, "updates", DataType.FLOAT32);
        Tensor y = data.scatterNd(indices, updates, ScatterReduction.ADD);
        y.setLabel("y");
        return fixture(
                "scatter_nd_add.onnx",
                y,
                new int[]{2, 3},
                new double[]{10.0, 32.0, 30.0, 40.0, 50.0, 60.0},
                Map.of(
                        "data", new float[]{10f, 20f, 30f, 40f, 50f, 60f},
                        "indices", new int[]{0, 1, 0, 1},
                        "updates", new float[]{5f, 7f}
                )
        );
    }

    private static Fixture scatterNdSliceNone() {
        Tensor data = new Tensor(new float[6], new int[]{2, 3}, null, "data", DataType.FLOAT32);
        Tensor indices = new Tensor(new int[1], new int[]{1, 1}, null, "indices", DataType.INT32);
        Tensor updates = new Tensor(new float[3], new int[]{1, 3}, null, "updates", DataType.FLOAT32);
        Tensor y = data.scatterNd(indices, updates);
        y.setLabel("y");
        return fixture(
                "scatter_nd_slice_none.onnx",
                y,
                new int[]{2, 3},
                new double[]{1.0, 2.0, 3.0, 40.0, 50.0, 60.0},
                Map.of(
                        "data", new float[]{1f, 2f, 3f, 4f, 5f, 6f},
                        "indices", new int[]{1},
                        "updates", new float[]{40f, 50f, 60f}
                )
        );
    }

    private static Fixture scatterNdAddDuplicates() {
        Tensor data = new Tensor(new float[6], new int[]{2, 3}, null, "data", DataType.FLOAT32);
        Tensor indices = new Tensor(new int[4], new int[]{2, 2}, null, "indices", DataType.INT32);
        Tensor updates = new Tensor(new float[2], new int[]{2}, null, "updates", DataType.FLOAT32);
        Tensor y = data.scatterNd(indices, updates, ScatterReduction.ADD);
        y.setLabel("y");
        return fixture(
                "scatter_nd_add_duplicates.onnx",
                y,
                new int[]{2, 3},
                new double[]{10.0, 32.0, 30.0, 40.0, 50.0, 60.0},
                Map.of(
                        "data", new float[]{10f, 20f, 30f, 40f, 50f, 60f},
                        "indices", new int[]{0, 1, 0, 1},
                        "updates", new float[]{5f, 7f}
                )
        );
    }

    private static Fixture scatterNdMul() {
        Tensor data = new Tensor(new float[4], new int[]{2, 2}, null, "data", DataType.FLOAT32);
        Tensor indices = new Tensor(new int[4], new int[]{2, 2}, null, "indices", DataType.INT32);
        Tensor updates = new Tensor(new float[2], new int[]{2}, null, "updates", DataType.FLOAT32);
        Tensor y = data.scatterNd(indices, updates, ScatterReduction.MUL);
        y.setLabel("y");
        return fixture(
                "scatter_nd_mul.onnx",
                y,
                new int[]{2, 2},
                new double[]{2.0, 120.0, 4.0, 8.0},
                Map.of(
                        "data", new float[]{2f, 10f, 4f, 8f},
                        "indices", new int[]{0, 1, 0, 1},
                        "updates", new float[]{3f, 4f}
                )
        );
    }

    private static Fixture scatterNdMax() {
        Tensor data = new Tensor(new float[4], new int[]{2, 2}, null, "data", DataType.FLOAT32);
        Tensor indices = new Tensor(new int[4], new int[]{2, 2}, null, "indices", DataType.INT32);
        Tensor updates = new Tensor(new float[2], new int[]{2}, null, "updates", DataType.FLOAT32);
        Tensor y = data.scatterNd(indices, updates, ScatterReduction.MAX);
        y.setLabel("y");
        return fixture(
                "scatter_nd_max.onnx",
                y,
                new int[]{2, 2},
                new double[]{2.0, 10.0, 4.0, 8.0},
                Map.of(
                        "data", new float[]{2f, 10f, 4f, 8f},
                        "indices", new int[]{0, 1, 0, 1},
                        "updates", new float[]{3f, 4f}
                )
        );
    }

    private static Fixture scatterNdMin() {
        Tensor data = new Tensor(new float[4], new int[]{2, 2}, null, "data", DataType.FLOAT32);
        Tensor indices = new Tensor(new int[4], new int[]{2, 2}, null, "indices", DataType.INT32);
        Tensor updates = new Tensor(new float[2], new int[]{2}, null, "updates", DataType.FLOAT32);
        Tensor y = data.scatterNd(indices, updates, ScatterReduction.MIN);
        y.setLabel("y");
        return fixture(
                "scatter_nd_min.onnx",
                y,
                new int[]{2, 2},
                new double[]{2.0, 3.0, 4.0, 8.0},
                Map.of(
                        "data", new float[]{2f, 10f, 4f, 8f},
                        "indices", new int[]{0, 1, 0, 1},
                        "updates", new float[]{3f, 4f}
                )
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
        return new Fixture(
                fileName,
                model,
                "y",
                expectedShape,
                expectedOutput,
                inputs
        );
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

    private static OnnxProto.NodeProto axisNode(String name, String opType, String output, int axis, String... inputs) {
        OnnxProto.NodeProto.Builder node = OnnxProto.NodeProto.newBuilder()
                .setName(name)
                .setOpType(opType);
        for (String input : inputs) {
            node.addInput(input);
        }
        return node.addOutput(output)
                .addAttribute(OnnxProto.AttributeProto.newBuilder().setName("axis").setI(axis))
                .build();
    }
}
