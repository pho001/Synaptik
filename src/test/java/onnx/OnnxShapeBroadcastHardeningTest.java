package onnx;

import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import java.util.function.Consumer;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class OnnxShapeBroadcastHardeningTest {
    @Test
    void constantVectorAndScalarBroadcastAcrossHigherRankInput() {
        OnnxProto.ModelProto model = model("constant_broadcast", graph -> graph
                .addInput(OnnxTensorProtoUtil.valueInfo("x", DataType.FLOAT32, new int[]{2, 1, 3}))
                .addNode(nodeBuilder("bias", "Constant", "bias")
                        .addAttribute(tensorAttr("value", new Tensor(new float[]{10f, 20f, 30f}, new int[]{3}, null, "bias", DataType.FLOAT32)))
                        .build())
                .addNode(nodeBuilder("scale", "Constant", "scale")
                        .addAttribute(tensorAttr("value", Tensor.scalar(0.5, DataType.FLOAT32)))
                        .build())
                .addNode(nodeBuilder("add", "Add", "biased", "x", "bias").build())
                .addNode(nodeBuilder("mul", "Mul", "y", "biased", "scale").build())
                .addOutput(OnnxTensorProtoUtil.valueInfo("y", DataType.FLOAT32, new int[]{2, 1, 3})));

        ImportedOnnxModel imported = Onnx.importModel(model);
        imported.input("x").setData(new float[]{1f, 2f, 3f, 4f, 5f, 6f});

        assertArrayEquals(new double[]{5.5, 11.0, 16.5, 7.0, 12.5, 18.0},
                OnnxRoundTripTestSupport.executeImported(imported, "y"), 1e-6);
    }

    @Test
    void constantBoolBroadcastFeedsWhere() {
        OnnxProto.ModelProto model = model("constant_bool_broadcast", graph -> graph
                .addInput(OnnxTensorProtoUtil.valueInfo("x", DataType.FLOAT32, new int[]{2, 3}))
                .addInput(OnnxTensorProtoUtil.valueInfo("fallback", DataType.FLOAT32, new int[]{1, 3}))
                .addNode(nodeBuilder("mask", "Constant", "mask")
                        .addAttribute(tensorAttr("value", new Tensor(new byte[]{1, 0, 1}, new int[]{3}, null, "mask", DataType.BOOL)))
                        .build())
                .addNode(nodeBuilder("where", "Where", "y", "mask", "x", "fallback").build())
                .addOutput(OnnxTensorProtoUtil.valueInfo("y", DataType.FLOAT32, new int[]{2, 3})));

        ImportedOnnxModel imported = Onnx.importModel(model);
        imported.input("x").setData(new float[]{1f, 2f, 3f, 4f, 5f, 6f});
        imported.input("fallback").setData(new float[]{10f, 20f, 30f});

        assertArrayEquals(new double[]{1.0, 20.0, 3.0, 4.0, 20.0, 6.0},
                OnnxRoundTripTestSupport.executeImported(imported, "y"), 1e-6);
    }

    @Test
    void dynamicReshapeAndExpandShapeInputsAreRejectedClearly() {
        OnnxProto.ModelProto dynamicReshape = model("dynamic_reshape", graph -> graph
                .addInput(OnnxTensorProtoUtil.valueInfo("x", DataType.FLOAT32, new int[]{2, 3}))
                .addInput(OnnxTensorProtoUtil.valueInfo("shape", DataType.INT32, new int[]{2}))
                .addNode(nodeBuilder("reshape", "Reshape", "y", "x", "shape").build())
                .addOutput(OnnxTensorProtoUtil.valueInfo("y", DataType.FLOAT32, new int[]{3, 2})));
        OnnxProto.ModelProto dynamicExpand = model("dynamic_expand", graph -> graph
                .addInput(OnnxTensorProtoUtil.valueInfo("x", DataType.FLOAT32, new int[]{1, 3}))
                .addInput(OnnxTensorProtoUtil.valueInfo("shape", DataType.INT32, new int[]{2}))
                .addNode(nodeBuilder("expand", "Expand", "y", "x", "shape").build())
                .addOutput(OnnxTensorProtoUtil.valueInfo("y", DataType.FLOAT32, new int[]{2, 3})));

        OnnxUnsupportedException reshapeEx = assertThrows(OnnxUnsupportedException.class, () -> Onnx.importModel(dynamicReshape));
        OnnxUnsupportedException expandEx = assertThrows(OnnxUnsupportedException.class, () -> Onnx.importModel(dynamicExpand));

        assertTrue(reshapeEx.getMessage().contains("constant initializer or Constant node"));
        assertTrue(expandEx.getMessage().contains("constant initializer or Constant node"));
    }

    @Test
    void dynamicStaticParameterInputsAreRejectedClearly() {
        OnnxProto.ModelProto dynamicTile = model("dynamic_tile", graph -> graph
                .addInput(OnnxTensorProtoUtil.valueInfo("x", DataType.FLOAT32, new int[]{2, 3}))
                .addInput(OnnxTensorProtoUtil.valueInfo("repeats", DataType.INT32, new int[]{2}))
                .addNode(nodeBuilder("tile", "Tile", "y", "x", "repeats").build())
                .addOutput(OnnxTensorProtoUtil.valueInfo("y", DataType.FLOAT32, new int[]{2, 3})));
        OnnxProto.ModelProto dynamicReduceAxes = model("dynamic_reduce_axes", graph -> graph
                .addInput(OnnxTensorProtoUtil.valueInfo("x", DataType.FLOAT32, new int[]{2, 3}))
                .addInput(OnnxTensorProtoUtil.valueInfo("axes", DataType.INT32, new int[]{1}))
                .addNode(nodeBuilder("reduce", "ReduceSum", "y", "x", "axes").build())
                .addOutput(OnnxTensorProtoUtil.valueInfo("y", DataType.FLOAT32, new int[]{2})));
        OnnxProto.ModelProto dynamicSqueezeAxes = model("dynamic_squeeze_axes", graph -> graph
                .addInput(OnnxTensorProtoUtil.valueInfo("x", DataType.FLOAT32, new int[]{1, 3}))
                .addInput(OnnxTensorProtoUtil.valueInfo("axes", DataType.INT32, new int[]{1}))
                .addNode(nodeBuilder("squeeze", "Squeeze", "y", "x", "axes").build())
                .addOutput(OnnxTensorProtoUtil.valueInfo("y", DataType.FLOAT32, new int[]{3})));
        OnnxProto.ModelProto dynamicUnsqueezeAxes = model("dynamic_unsqueeze_axes", graph -> graph
                .addInput(OnnxTensorProtoUtil.valueInfo("x", DataType.FLOAT32, new int[]{3}))
                .addInput(OnnxTensorProtoUtil.valueInfo("axes", DataType.INT32, new int[]{1}))
                .addNode(nodeBuilder("unsqueeze", "Unsqueeze", "y", "x", "axes").build())
                .addOutput(OnnxTensorProtoUtil.valueInfo("y", DataType.FLOAT32, new int[]{1, 3})));

        assertStaticParameterRejected(dynamicTile);
        assertStaticParameterRejected(dynamicReduceAxes);
        assertStaticParameterRejected(dynamicSqueezeAxes);
        assertStaticParameterRejected(dynamicUnsqueezeAxes);
    }

    @Test
    void invalidBinaryBroadcastShapesAreRejectedAsOnnxUnsupported() {
        OnnxProto.ModelProto model = model("invalid_binary_broadcast", graph -> graph
                .addInput(OnnxTensorProtoUtil.valueInfo("a", DataType.FLOAT32, new int[]{2, 3}))
                .addInput(OnnxTensorProtoUtil.valueInfo("b", DataType.FLOAT32, new int[]{2}))
                .addNode(nodeBuilder("add", "Add", "y", "a", "b").build())
                .addOutput(OnnxTensorProtoUtil.valueInfo("y", DataType.FLOAT32, new int[]{2, 3})));

        OnnxUnsupportedException ex = assertThrows(OnnxUnsupportedException.class, () -> Onnx.importModel(model));

        assertTrue(ex.getMessage().contains("add"));
    }

    @Test
    void invalidWhereBranchBroadcastShapesAreRejectedAsOnnxUnsupported() {
        OnnxProto.ModelProto model = model("invalid_where_branch_broadcast", graph -> graph
                .addInput(OnnxTensorProtoUtil.valueInfo("condition", DataType.BOOL, new int[]{2, 3}))
                .addInput(OnnxTensorProtoUtil.valueInfo("x", DataType.FLOAT32, new int[]{2, 3}))
                .addInput(OnnxTensorProtoUtil.valueInfo("fallback", DataType.FLOAT32, new int[]{2}))
                .addNode(nodeBuilder("where", "Where", "y", "condition", "x", "fallback").build())
                .addOutput(OnnxTensorProtoUtil.valueInfo("y", DataType.FLOAT32, new int[]{2, 3})));

        OnnxUnsupportedException ex = assertThrows(OnnxUnsupportedException.class, () -> Onnx.importModel(model));

        assertTrue(ex.getMessage().contains("where"));
    }

    @Test
    void invalidConstantTensorBroadcastShapesAreRejectedAsOnnxUnsupported() {
        OnnxProto.ModelProto model = model("invalid_constant_tensor_broadcast", graph -> graph
                .addInput(OnnxTensorProtoUtil.valueInfo("x", DataType.FLOAT32, new int[]{2, 3}))
                .addNode(nodeBuilder("bias", "Constant", "bias")
                        .addAttribute(tensorAttr("value", new Tensor(new float[]{10f, 20f}, new int[]{2}, null, "bias", DataType.FLOAT32)))
                        .build())
                .addNode(nodeBuilder("add", "Add", "y", "x", "bias").build())
                .addOutput(OnnxTensorProtoUtil.valueInfo("y", DataType.FLOAT32, new int[]{2, 3})));

        OnnxUnsupportedException ex = assertThrows(OnnxUnsupportedException.class, () -> Onnx.importModel(model));

        assertTrue(ex.getMessage().contains("add"));
    }

    @Test
    void invalidBoolMaskBroadcastShapesAreRejectedAsOnnxUnsupported() {
        OnnxProto.ModelProto model = model("invalid_bool_mask_broadcast", graph -> graph
                .addInput(OnnxTensorProtoUtil.valueInfo("x", DataType.FLOAT32, new int[]{2, 3}))
                .addInput(OnnxTensorProtoUtil.valueInfo("fallback", DataType.FLOAT32, new int[]{2, 3}))
                .addNode(nodeBuilder("mask", "Constant", "mask")
                        .addAttribute(tensorAttr("value", new Tensor(new byte[]{1, 0}, new int[]{2}, null, "mask", DataType.BOOL)))
                        .build())
                .addNode(nodeBuilder("where", "Where", "y", "mask", "x", "fallback").build())
                .addOutput(OnnxTensorProtoUtil.valueInfo("y", DataType.FLOAT32, new int[]{2, 3})));

        OnnxUnsupportedException ex = assertThrows(OnnxUnsupportedException.class, () -> Onnx.importModel(model));

        assertTrue(ex.getMessage().contains("where"));
    }

    private static void assertStaticParameterRejected(OnnxProto.ModelProto model) {
        OnnxUnsupportedException ex = assertThrows(OnnxUnsupportedException.class, () -> Onnx.importModel(model));
        assertTrue(ex.getMessage().contains("constant initializer or Constant node"));
    }

    private static OnnxProto.ModelProto model(String graphName, Consumer<OnnxProto.GraphProto.Builder> configure) {
        OnnxProto.GraphProto.Builder graph = OnnxProto.GraphProto.newBuilder().setName(graphName);
        configure.accept(graph);
        return OnnxProto.ModelProto.newBuilder()
                .setIrVersion(OnnxProto.Version.IR_VERSION_2023_5_5.getNumber())
                .addOpsetImport(OnnxProto.OperatorSetIdProto.newBuilder()
                        .setDomain("")
                        .setVersion(OnnxExportOptions.DEFAULT_OPSET))
                .setGraph(graph)
                .build();
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

    private static OnnxProto.AttributeProto tensorAttr(String name, Tensor value) {
        return OnnxProto.AttributeProto.newBuilder()
                .setName(name)
                .setT(OnnxTensorProtoUtil.tensorInitializer("", value))
                .build();
    }
}
