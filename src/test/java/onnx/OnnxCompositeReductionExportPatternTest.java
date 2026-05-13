package onnx;

import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

class OnnxCompositeReductionExportPatternTest {
    @Test
    void reduceL1CompositionExportsCanonicalNodeAndRoundTrips() {
        Tensor x = input2d();
        Tensor y = x.abs().sum(1, false);
        y.setLabel("y");

        OnnxProto.GraphProto graph = exportGraph(y);

        assertEquals(List.of("ReduceL1"), opTypes(graph));
        assertEquals(1, int64Initializer(graph, graph.getNode(0).getInput(1))[0]);
        assertArrayEquals(new double[]{6.0, 15.0}, roundTrip(y), 1e-6);
    }

    @Test
    void reduceL2CompositionExportsCanonicalNodeAndRoundTrips() {
        Tensor x = input2d();
        Tensor y = x.mul(x).sum(1, false).sqrt();
        y.setLabel("y");

        OnnxProto.GraphProto graph = exportGraph(y);

        assertEquals(List.of("ReduceL2"), opTypes(graph));
        assertEquals(1, int64Initializer(graph, graph.getNode(0).getInput(1))[0]);
        assertArrayEquals(new double[]{Math.sqrt(14.0), Math.sqrt(77.0)}, roundTrip(y), 1e-6);
    }

    @Test
    void reduceLogSumCompositionExportsCanonicalNodeAndRoundTrips() {
        Tensor x = positiveInput2d();
        Tensor y = x.sum(1, false).log();
        y.setLabel("y");

        OnnxProto.GraphProto graph = exportGraph(y);

        assertEquals(List.of("ReduceLogSum"), opTypes(graph));
        assertEquals(1, int64Initializer(graph, graph.getNode(0).getInput(1))[0]);
        assertArrayEquals(new double[]{Math.log(6.0), Math.log(15.0)}, roundTrip(y), 1e-6);
    }

    @Test
    void reduceLogSumExpCompositionExportsCanonicalNodeAndRoundTrips() {
        Tensor x = input2d();
        Tensor y = x.exp().sum(1, false).log();
        y.setLabel("y");

        OnnxProto.GraphProto graph = exportGraph(y);

        assertEquals(List.of("ReduceLogSumExp"), opTypes(graph));
        assertEquals(1, int64Initializer(graph, graph.getNode(0).getInput(1))[0]);
        assertArrayEquals(new double[]{
                        Math.log(Math.exp(1.0) + Math.exp(2.0) + Math.exp(3.0)),
                        Math.log(Math.exp(4.0) + Math.exp(5.0) + Math.exp(6.0))
                },
                roundTrip(y), 1e-6);
    }

    @Test
    void globalAveragePoolCompositionExportsCanonicalNodeAndRoundTrips() {
        Tensor x = new Tensor(new float[6], new int[]{1, 1, 2, 3}, null, "x", DataType.FLOAT32);
        Tensor y = x.mean(2, true).mean(3, true);
        y.setLabel("y");

        OnnxProto.GraphProto graph = exportGraph(y);

        assertEquals(List.of("GlobalAveragePool"), opTypes(graph));
        assertArrayEquals(new double[]{3.5},
                OnnxRoundTripTestSupport.executeRoundTrip(
                        y,
                        "y",
                        Map.of("x", new float[]{1f, 2f, 3f, 4f, 5f, 6f})
                ),
                1e-6);
    }

    @Test
    void sharedCompositeIntermediateRemainsPrimitiveGraph() {
        Tensor x = input2d();
        Tensor abs = x.abs();
        Tensor y = abs.sum(1, false).add(abs.max(1, false));
        y.setLabel("y");

        List<String> opTypes = opTypes(exportGraph(y));

        assertFalse(opTypes.contains("ReduceL1"));
        assertEquals(List.of("Abs", "ReduceSum", "ReduceMax", "Add"), opTypes);
    }

    private static Tensor input2d() {
        return new Tensor(new float[6], new int[]{2, 3}, null, "x", DataType.FLOAT32);
    }

    private static Tensor positiveInput2d() {
        return input2d();
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

    private static long[] int64Initializer(OnnxProto.GraphProto graph, String name) {
        return graph.getInitializerList().stream()
                .filter(initializer -> initializer.getName().equals(name))
                .findFirst()
                .map(initializer -> OnnxTensorProtoUtil.parseConstant(initializer, name).int64Values())
                .orElseThrow();
    }

    private static double[] roundTrip(Tensor output) {
        return OnnxRoundTripTestSupport.executeRoundTrip(
                output,
                "y",
                Map.of("x", new float[]{1f, 2f, 3f, 4f, 5f, 6f})
        );
    }
}
