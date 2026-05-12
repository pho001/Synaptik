package onnx;

import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import java.util.Map;
import java.util.function.Consumer;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class OnnxWave4ActivationExecutionTest {
    @Test
    void importComposedActivationSubsetExecutes() {
        OnnxProto.ModelProto model = model("wave4_activations", graph -> graph
                .addInput(OnnxTensorProtoUtil.valueInfo("x", DataType.FLOAT32, new int[]{5}))
                .addNode(nodeBuilder("reciprocal", "Reciprocal", "reciprocal", "x").build())
                .addNode(nodeBuilder("leaky", "LeakyRelu", "leaky", "x")
                        .addAttribute(floatAttr("alpha", 0.2f))
                        .build())
                .addNode(nodeBuilder("elu", "Elu", "elu", "x")
                        .addAttribute(floatAttr("alpha", 1.5f))
                        .build())
                .addNode(nodeBuilder("hard_sigmoid", "HardSigmoid", "hard", "x")
                        .addAttribute(floatAttr("alpha", 0.25f))
                        .addAttribute(floatAttr("beta", 0.5f))
                        .build())
                .addNode(nodeBuilder("softplus", "Softplus", "softplus", "x").build())
                .addOutput(OnnxTensorProtoUtil.valueInfo("reciprocal", DataType.FLOAT32, new int[]{5}))
                .addOutput(OnnxTensorProtoUtil.valueInfo("leaky", DataType.FLOAT32, new int[]{5}))
                .addOutput(OnnxTensorProtoUtil.valueInfo("elu", DataType.FLOAT32, new int[]{5}))
                .addOutput(OnnxTensorProtoUtil.valueInfo("hard", DataType.FLOAT32, new int[]{5}))
                .addOutput(OnnxTensorProtoUtil.valueInfo("softplus", DataType.FLOAT32, new int[]{5})));

        ImportedOnnxModel imported = Onnx.importModel(model);
        imported.input("x").setData(new float[]{-2f, -1f, 0.5f, 1f, 2f});

        assertArrayEquals(new double[]{-0.5, -1.0, 2.0, 1.0, 0.5},
                OnnxRoundTripTestSupport.executeImported(imported, "reciprocal"), 1e-6);
        assertArrayEquals(new double[]{-0.4, -0.2, 0.5, 1.0, 2.0},
                OnnxRoundTripTestSupport.executeImported(imported, "leaky"), 1e-6);
        assertArrayEquals(new double[]{
                        1.5 * (Math.exp(-2.0) - 1.0),
                        1.5 * (Math.exp(-1.0) - 1.0),
                        0.5,
                        1.0,
                        2.0
                },
                OnnxRoundTripTestSupport.executeImported(imported, "elu"), 1e-6);
        assertArrayEquals(new double[]{0.0, 0.25, 0.625, 0.75, 1.0},
                OnnxRoundTripTestSupport.executeImported(imported, "hard"), 1e-6);
        assertArrayEquals(new double[]{
                        Math.log(Math.exp(-2.0) + 1.0),
                        Math.log(Math.exp(-1.0) + 1.0),
                        Math.log(Math.exp(0.5) + 1.0),
                        Math.log(Math.E + 1.0),
                        Math.log(Math.exp(2.0) + 1.0)
                },
                OnnxRoundTripTestSupport.executeImported(imported, "softplus"), 1e-6);
    }

    @Test
    void firstClassUnaryMathImportsExportsAndRoundTrips() {
        Tensor x = new Tensor(new float[5], new int[]{5}, null, "x", DataType.FLOAT32);

        assertEquals("Reciprocal", singleNode(x.inv()).getOpType());
        assertEquals("Erf", singleNode(x.erf()).getOpType());
        assertEquals("Floor", singleNode(x.floor()).getOpType());
        assertEquals("Ceil", singleNode(x.ceil()).getOpType());
        assertEquals("Sign", singleNode(x.sign()).getOpType());

        Tensor y = x.erf().add(x.floor()).add(x.ceil()).add(x.sign()).add(x.inv());
        y.setLabel("y");
        double[] actual = OnnxRoundTripTestSupport.executeRoundTrip(
                y,
                "y",
                Map.of("x", new float[]{-2f, -0.25f, 0.5f, 1.25f, 2f})
        );
        double[] expected = new double[5];
        double[] values = {-2.0, -0.25, 0.5, 1.25, 2.0};
        for (int i = 0; i < values.length; i++) {
            double value = values[i];
            expected[i] = erfApprox(value) + Math.floor(value) + Math.ceil(value) + Math.signum(value) + 1.0 / value;
        }
        assertArrayEquals(expected, actual, 1e-5);
    }

    @Test
    void importNonZeroFailsWithDynamicShapeReason() {
        OnnxProto.ModelProto model = model("nonzero", graph -> graph
                .addInput(OnnxTensorProtoUtil.valueInfo("x", DataType.FLOAT32, new int[]{3}))
                .addNode(nodeBuilder("nonzero", "NonZero", "idx", "x").build())
                .addOutput(OnnxTensorProtoUtil.valueInfo("idx", DataType.INT32, new int[]{1, 3})));

        OnnxUnsupportedException ex = assertThrows(OnnxUnsupportedException.class, () -> Onnx.importModel(model));

        assertTrue(ex.getMessage().contains("dynamic-shape"));
    }

    private static double erfApprox(double value) {
        return utils.SpecialFunctions.erf(value);
    }

    private static OnnxProto.NodeProto singleNode(Tensor output) {
        OnnxProto.GraphProto graph = Onnx.exportModel(
                output,
                OnnxExportOptions.defaults().withLeafTensorPolicy(OnnxLeafTensorPolicy.INPUTS)
        ).proto().getGraph();
        assertEquals(1, graph.getNodeCount());
        return graph.getNode(0);
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

    private static OnnxProto.AttributeProto floatAttr(String name, float value) {
        return OnnxProto.AttributeProto.newBuilder().setName(name).setF(value).build();
    }
}
