package onnx;

import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import java.util.function.Consumer;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class OnnxStaticParameterHardeningTest {
    @Test
    void runtimeStaticParameterInputsAreRejectedClearly() {
        assertConstantRejected(model("dynamic_slice_starts", graph -> graph
                .addInput(OnnxTensorProtoUtil.valueInfo("x", DataType.FLOAT32, new int[]{4}))
                .addInput(OnnxTensorProtoUtil.valueInfo("starts", DataType.INT32, new int[]{1}))
                .addInitializer(OnnxTensorProtoUtil.int64Initializer("ends", new long[]{3}))
                .addNode(node("slice", "Slice", "y", "x", "starts", "ends"))
                .addOutput(OnnxTensorProtoUtil.valueInfo("y", DataType.FLOAT32, new int[]{3}))));
        assertConstantRejected(model("dynamic_pad_pads", graph -> graph
                .addInput(OnnxTensorProtoUtil.valueInfo("x", DataType.FLOAT32, new int[]{2}))
                .addInput(OnnxTensorProtoUtil.valueInfo("pads", DataType.INT32, new int[]{2}))
                .addNode(node("pad", "Pad", "y", "x", "pads"))
                .addOutput(OnnxTensorProtoUtil.valueInfo("y", DataType.FLOAT32, new int[]{4}))));
        assertConstantRejected(model("dynamic_constant_of_shape", graph -> graph
                .addInput(OnnxTensorProtoUtil.valueInfo("shape", DataType.INT32, new int[]{1}))
                .addNode(node("constant_of_shape", "ConstantOfShape", "y", "shape"))
                .addOutput(OnnxTensorProtoUtil.valueInfo("y", DataType.FLOAT32, new int[]{3}))));
        assertConstantRejected(model("dynamic_cumsum_axis", graph -> graph
                .addInput(OnnxTensorProtoUtil.valueInfo("x", DataType.FLOAT32, new int[]{3}))
                .addInput(OnnxTensorProtoUtil.valueInfo("axis", DataType.INT32, new int[]{1}))
                .addNode(node("cumsum", "CumSum", "y", "x", "axis"))
                .addOutput(OnnxTensorProtoUtil.valueInfo("y", DataType.FLOAT32, new int[]{3}))));
        assertConstantRejected(model("dynamic_split_sizes", graph -> graph
                .addInput(OnnxTensorProtoUtil.valueInfo("x", DataType.FLOAT32, new int[]{4}))
                .addInput(OnnxTensorProtoUtil.valueInfo("split", DataType.INT32, new int[]{2}))
                .addNode(nodeBuilder("split_node", "Split", "a", "x", "split")
                        .addOutput("b")
                        .addAttribute(intAttr("axis", 0))
                        .build())
                .addOutput(OnnxTensorProtoUtil.valueInfo("a", DataType.FLOAT32, new int[]{2}))
                .addOutput(OnnxTensorProtoUtil.valueInfo("b", DataType.FLOAT32, new int[]{2}))));
    }

    @Test
    void runtimeScalarParametersAreRejectedClearly() {
        assertScalarRejected(model("dynamic_clip_min", graph -> graph
                .addInput(OnnxTensorProtoUtil.valueInfo("x", DataType.FLOAT32, new int[]{2}))
                .addInput(OnnxTensorProtoUtil.valueInfo("min", DataType.FLOAT32, new int[]{1}))
                .addNode(node("clip", "Clip", "y", "x", "min"))
                .addOutput(OnnxTensorProtoUtil.valueInfo("y", DataType.FLOAT32, new int[]{2}))));
        assertScalarRejected(model("dynamic_pow_exponent", graph -> graph
                .addInput(OnnxTensorProtoUtil.valueInfo("x", DataType.FLOAT32, new int[]{2}))
                .addInput(OnnxTensorProtoUtil.valueInfo("exponent", DataType.FLOAT32, new int[]{1}))
                .addNode(node("pow", "Pow", "y", "x", "exponent"))
                .addOutput(OnnxTensorProtoUtil.valueInfo("y", DataType.FLOAT32, new int[]{2}))));
    }

    @Test
    void staticShapeDslCanFeedReshapeExpandAndConstantOfShape() {
        OnnxProto.ModelProto reshapeModel = model("shape_gather_concat_reshape", graph -> graph
                .addInput(OnnxTensorProtoUtil.valueInfo("x", DataType.FLOAT32, new int[]{2, 3}))
                .addInitializer(OnnxTensorProtoUtil.int64Initializer("idx0", new long[]{0}))
                .addInitializer(OnnxTensorProtoUtil.int64Initializer("axes0", new long[]{0}))
                .addInitializer(OnnxTensorProtoUtil.int64Initializer("three", new long[]{3}))
                .addNode(node("shape", "Shape", "shape", "x"))
                .addNode(node("gather", "Gather", "dim0_scalar", "shape", "idx0"))
                .addNode(node("unsqueeze", "Unsqueeze", "dim0", "dim0_scalar", "axes0"))
                .addNode(nodeBuilder("concat", "Concat", "target_shape", "dim0", "three")
                        .addAttribute(intAttr("axis", 0))
                        .build())
                .addNode(node("reshape", "Reshape", "y", "x", "target_shape"))
                .addOutput(OnnxTensorProtoUtil.valueInfo("y", DataType.FLOAT32, new int[]{2, 3})));
        OnnxProto.ModelProto expandModel = model("shape_slice_concat_expand", graph -> graph
                .addInput(OnnxTensorProtoUtil.valueInfo("x", DataType.FLOAT32, new int[]{2, 1}))
                .addInitializer(OnnxTensorProtoUtil.int64Initializer("starts", new long[]{0}))
                .addInitializer(OnnxTensorProtoUtil.int64Initializer("ends", new long[]{1}))
                .addInitializer(OnnxTensorProtoUtil.int64Initializer("axes", new long[]{0}))
                .addInitializer(OnnxTensorProtoUtil.int64Initializer("three", new long[]{3}))
                .addNode(node("shape", "Shape", "shape", "x"))
                .addNode(node("slice", "Slice", "head", "shape", "starts", "ends", "axes"))
                .addNode(nodeBuilder("concat", "Concat", "target_shape", "head", "three")
                        .addAttribute(intAttr("axis", 0))
                        .build())
                .addNode(node("expand", "Expand", "y", "x", "target_shape"))
                .addOutput(OnnxTensorProtoUtil.valueInfo("y", DataType.FLOAT32, new int[]{2, 3})));
        OnnxProto.ModelProto constantModel = model("shape_size_constant_of_shape", graph -> graph
                .addInput(OnnxTensorProtoUtil.valueInfo("x", DataType.FLOAT32, new int[]{2, 3}))
                .addNode(node("size", "Size", "shape", "x"))
                .addNode(nodeBuilder("constant_of_shape", "ConstantOfShape", "y", "shape")
                        .addAttribute(tensorAttr("value", Tensor.scalar(2.0d, DataType.FLOAT32)))
                        .build())
                .addOutput(OnnxTensorProtoUtil.valueInfo("y", DataType.FLOAT32, new int[]{6})));

        ImportedOnnxModel reshape = Onnx.importModel(reshapeModel);
        reshape.input("x").setData(new float[]{1f, 2f, 3f, 4f, 5f, 6f});
        assertArrayEquals(new double[]{1, 2, 3, 4, 5, 6},
                OnnxRoundTripTestSupport.executeImported(reshape, "y"), 1e-6);

        ImportedOnnxModel expand = Onnx.importModel(expandModel);
        expand.input("x").setData(new float[]{10f, 20f});
        assertArrayEquals(new double[]{10, 10, 10, 20, 20, 20},
                OnnxRoundTripTestSupport.executeImported(expand, "y"), 1e-6);

        ImportedOnnxModel constant = Onnx.importModel(constantModel);
        constant.input("x").setData(new float[]{0f, 0f, 0f, 0f, 0f, 0f});
        assertArrayEquals(new double[]{2, 2, 2, 2, 2, 2},
                OnnxRoundTripTestSupport.executeImported(constant, "y"), 1e-6);
    }

    private static void assertConstantRejected(OnnxProto.ModelProto model) {
        OnnxUnsupportedException ex = assertThrows(OnnxUnsupportedException.class, () -> Onnx.importModel(model));
        assertTrue(ex.getMessage().contains("constant initializer or Constant node"));
    }

    private static void assertScalarRejected(OnnxProto.ModelProto model) {
        OnnxUnsupportedException ex = assertThrows(OnnxUnsupportedException.class, () -> Onnx.importModel(model));
        assertTrue(ex.getMessage().contains("scalar initializer or Constant node"));
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

    private static OnnxProto.NodeProto node(String name, String opType, String output, String... inputs) {
        return nodeBuilder(name, opType, output, inputs).build();
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
