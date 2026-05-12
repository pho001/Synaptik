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
        add(out, scatterNdNone());
        add(out, scatterNdAdd());
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

    private static Fixture fixture(
            String fileName,
            Tensor output,
            int[] expectedShape,
            double[] expectedOutput,
            Map<String, Object> inputs
    ) {
        return new Fixture(
                fileName,
                Onnx.exportModel(output, OnnxExportOptions.defaults().withLeafTensorPolicy(OnnxLeafTensorPolicy.INPUTS)),
                "y",
                expectedShape,
                expectedOutput,
                inputs
        );
    }
}
