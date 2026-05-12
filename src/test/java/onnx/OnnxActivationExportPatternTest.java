package onnx;

import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

class OnnxActivationExportPatternTest {
    @Test
    void leakyReluCompositionExportsCanonicalNodeAndRoundTrips() {
        Tensor x = input();
        Tensor y = Tensor.where(x.greaterOrEqual(zero()), x, x.mul(0.2d));
        y.setLabel("y");

        OnnxProto.GraphProto graph = exportGraph(y);

        assertEquals(List.of("LeakyRelu"), opTypes(graph));
        assertEquals(0.2f, floatAttr(graph.getNode(0), "alpha"), 1e-6f);
        assertArrayEquals(new double[]{-0.4, -0.2, 0.0, 1.0, 2.0},
                roundTrip(y), 1e-6);
    }

    @Test
    void eluCompositionExportsCanonicalNodeAndRoundTrips() {
        Tensor x = input();
        Tensor negative = x.exp().sub(Tensor.scalar(1.0d, DataType.FLOAT32)).mul(1.5d);
        Tensor y = Tensor.where(x.greaterOrEqual(zero()), x, negative);
        y.setLabel("y");

        OnnxProto.GraphProto graph = exportGraph(y);

        assertEquals(List.of("Elu"), opTypes(graph));
        assertEquals(1.5f, floatAttr(graph.getNode(0), "alpha"), 1e-6f);
        assertArrayEquals(new double[]{
                        1.5d * (Math.exp(-2.0d) - 1.0d),
                        1.5d * (Math.exp(-1.0d) - 1.0d),
                        0.0d,
                        1.0d,
                        2.0d
                },
                roundTrip(y), 1e-6);
    }

    @Test
    void hardSigmoidCompositionExportsCanonicalNodeAndRoundTrips() {
        Tensor x = input();
        Tensor y = x.mul(0.25d).add(Tensor.scalar(0.5d, DataType.FLOAT32)).clamp(0.0d, 1.0d);
        y.setLabel("y");

        OnnxProto.GraphProto graph = exportGraph(y);

        assertEquals(List.of("HardSigmoid"), opTypes(graph));
        assertEquals(0.25f, floatAttr(graph.getNode(0), "alpha"), 1e-6f);
        assertEquals(0.5f, floatAttr(graph.getNode(0), "beta"), 1e-6f);
        assertArrayEquals(new double[]{0.0, 0.25, 0.5, 0.75, 1.0},
                roundTrip(y), 1e-6);
    }

    @Test
    void softplusCompositionExportsCanonicalNodeAndRoundTrips() {
        Tensor x = input();
        Tensor y = x.exp().add(Tensor.scalar(1.0d, DataType.FLOAT32)).log();
        y.setLabel("y");

        OnnxProto.GraphProto graph = exportGraph(y);

        assertEquals(List.of("Softplus"), opTypes(graph));
        assertArrayEquals(new double[]{
                        Math.log(Math.exp(-2.0d) + 1.0d),
                        Math.log(Math.exp(-1.0d) + 1.0d),
                        Math.log(2.0d),
                        Math.log(Math.E + 1.0d),
                        Math.log(Math.exp(2.0d) + 1.0d)
                },
                roundTrip(y), 1e-6);
    }

    @Test
    void nearMissLeakyReluShapeRemainsPrimitiveGraph() {
        Tensor x = input();
        Tensor y = Tensor.where(x.greaterThan(zero()), x, x.mul(0.2d));
        y.setLabel("y");

        List<String> opTypes = opTypes(exportGraph(y));

        assertFalse(opTypes.contains("LeakyRelu"));
        assertEquals(List.of("Greater", "Mul", "Where"), opTypes);
    }

    private static Tensor input() {
        return new Tensor(new float[5], new int[]{5}, null, "x", DataType.FLOAT32);
    }

    private static Tensor zero() {
        return Tensor.scalar(0.0d, DataType.FLOAT32);
    }

    private static OnnxProto.GraphProto exportGraph(Tensor output) {
        return Onnx.exportModel(
                output,
                OnnxExportOptions.defaults().withLeafTensorPolicy(OnnxLeafTensorPolicy.INPUTS)
        ).proto().getGraph();
    }

    private static List<String> opTypes(OnnxProto.GraphProto graph) {
        return graph.getNodeList().stream()
                .map(OnnxProto.NodeProto::getOpType)
                .toList();
    }

    private static float floatAttr(OnnxProto.NodeProto node, String name) {
        return node.getAttributeList().stream()
                .filter(attr -> attr.getName().equals(name))
                .findFirst()
                .orElseThrow()
                .getF();
    }

    private static double[] roundTrip(Tensor output) {
        return OnnxRoundTripTestSupport.executeRoundTrip(
                output,
                "y",
                Map.of("x", new float[]{-2f, -1f, 0f, 1f, 2f})
        );
    }
}
