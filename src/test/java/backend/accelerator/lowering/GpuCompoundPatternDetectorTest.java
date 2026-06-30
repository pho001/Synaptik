package backend.accelerator.lowering;

import backend.contract.ComputeBackend;
import backend.accelerator.dag.AcceleratorDagInput;
import backend.accelerator.dag.AcceleratorDagNode;
import backend.accelerator.dag.AcceleratorDagNodeType;
import backend.accelerator.dag.AcceleratorDagSpec;
import backend.accelerator.dag.AcceleratorDagValueRef;
import backend.accelerator.dag.AcceleratorPostOp;
import backend.accelerator.dag.AcceleratorPostOpType;
import backend.accelerator.dag.AcceleratorSubgraphOp;
import backend.accelerator.dag.AcceleratorSubgraphSpec;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GpuCompoundPatternDetectorTest {
    @Test
    void detectsLinearBiasActivationSummaryFromExistingMatmulSpec() {
        AcceleratorMatMulSpec matMulSpec = new AcceleratorMatMulSpec(
                0,
                1,
                2,
                4,
                2,
                4,
                3,
                true,
                List.of(AcceleratorPostOp.unary(AcceleratorPostOpType.RELU))
        );
        AcceleratorSubgraphSpec subgraph = new AcceleratorSubgraphSpec(
                3,
                List.of(3, 4),
                List.of(
                        new AcceleratorSubgraphOp(3, Operation.OpType.LINEAR),
                        new AcceleratorSubgraphOp(4, Operation.OpType.RELU)
                ),
                List.of(0, 1, 2),
                List.of(4)
        );

        GpuCompoundRegionSummary summary = GpuCompoundPatternDetector.detect(
                ComputeBackend.GPU_METAL,
                subgraph,
                linearReluDag(),
                matMulSpec
        );

        assertEquals(GpuCompoundPatternType.LINEAR_BIAS_ACTIVATION, summary.patternType());
        assertEquals("LINEAR_BIAS_ACTIVATION", summary.patternType().name());
        assertTrue(summary.supported());
        assertEquals(GpuLoweringUnsupportedReason.SUPPORTED, summary.reason());
        assertEquals(List.of("RELU"), summary.postOps());
    }

    @Test
    void detectsElementwiseChainSummaryFromDagNodes() {
        AcceleratorSubgraphSpec subgraph = new AcceleratorSubgraphSpec(
                2,
                List.of(2, 3, 4),
                List.of(
                        new AcceleratorSubgraphOp(2, Operation.OpType.ADD),
                        new AcceleratorSubgraphOp(3, Operation.OpType.RELU),
                        new AcceleratorSubgraphOp(4, Operation.OpType.EXP)
                ),
                List.of(0, 1),
                List.of(4)
        );

        GpuCompoundRegionSummary summary = GpuCompoundPatternDetector.detect(
                ComputeBackend.GPU_CUDA,
                subgraph,
                elementwiseDag(),
                null
        );

        assertEquals(GpuCompoundPatternType.ELEMENTWISE_CHAIN, summary.patternType());
        assertEquals("ELEMENTWISE_CHAIN", summary.patternType().name());
        assertTrue(summary.supported());
        assertEquals(List.of("ADD", "RELU", "EXP"), summary.dagNodeTypes());
    }

    @Test
    void classifiesReductionAdjacentCandidateAsUnsupportedWhenNoSafeSubsetExists() {
        AcceleratorSubgraphSpec subgraph = new AcceleratorSubgraphSpec(
                1,
                List.of(1),
                List.of(new AcceleratorSubgraphOp(1, Operation.OpType.LAYER_NORM)),
                List.of(0),
                List.of(1)
        );

        GpuCompoundRegionSummary summary = GpuCompoundPatternDetector.detect(
                ComputeBackend.GPU_METAL,
                subgraph,
                singleNodeDag(1, AcceleratorDagNodeType.ADD),
                null
        );

        assertEquals(GpuCompoundPatternType.REDUCTION_ADJACENT, summary.patternType());
        assertEquals("REDUCTION_ADJACENT", summary.patternType().name());
        assertFalse(summary.supported());
        assertEquals(GpuLoweringUnsupportedReason.COMPOUND_PATTERN_UNSUPPORTED, summary.reason());
        assertTrue(summary.detail().contains("REDUCTION_ADJACENT"));
    }

    @Test
    void reductionAdjacentLayerNormRejectsWithStableCompoundReason() {
        for (Operation.OpType opType : List.of(Operation.OpType.LAYER_NORM, Operation.OpType.RMS_NORM)) {
            AcceleratorSubgraphSpec subgraph = new AcceleratorSubgraphSpec(
                    1,
                    List.of(1),
                    List.of(new AcceleratorSubgraphOp(1, opType)),
                    List.of(0),
                    List.of(1)
            );

            GpuCompoundRegionSummary summary = GpuCompoundPatternDetector.detect(
                    ComputeBackend.GPU_CUDA,
                    subgraph,
                    singleNodeDag(1, AcceleratorDagNodeType.ADD),
                    null
            );

            assertEquals(GpuCompoundPatternType.REDUCTION_ADJACENT, summary.patternType());
            assertFalse(summary.supported());
            assertEquals(GpuLoweringUnsupportedReason.COMPOUND_PATTERN_UNSUPPORTED, summary.reason());
            assertTrue(summary.detail().contains("REDUCTION_ADJACENT"));
        }
    }

    @Test
    void rejectsCpuFusedOpTypeForGpuCompoundLowering() {
        AcceleratorSubgraphSpec subgraph = new AcceleratorSubgraphSpec(
                1,
                List.of(1),
                List.of(new AcceleratorSubgraphOp(1, Operation.OpType.FUSED)),
                List.of(0),
                List.of(1)
        );

        GpuCompoundRegionSummary summary = GpuCompoundPatternDetector.detect(
                ComputeBackend.GPU_CUDA,
                subgraph,
                null,
                null
        );

        assertEquals(GpuCompoundPatternType.CPU_FUSED_UNSUPPORTED, summary.patternType());
        assertEquals("CPU_FUSED_UNSUPPORTED", summary.patternType().name());
        assertFalse(summary.supported());
        assertEquals(GpuLoweringUnsupportedReason.CPU_FUSED_OPERATION_UNSUPPORTED, summary.reason());
    }

    @Test
    void recordsElementwiseChainFusionSubpatternSpanAndPrimitiveCount() {
        GpuFusionSubpatternSummary subpattern = GpuFusionSubpatternSummary.supported(
                GpuCompoundPatternType.ELEMENTWISE_CHAIN,
                List.of(20, 21, 22),
                List.of("p0", "p1", "p2"),
                "elementwise chain"
        );

        assertEquals(GpuCompoundPatternType.ELEMENTWISE_CHAIN, subpattern.patternType());
        assertTrue(subpattern.supported());
        assertEquals(List.of(20, 21, 22), subpattern.originalOperationNodeIds());
        assertEquals(List.of("p0", "p1", "p2"), subpattern.loweredPrimitiveIds());
        assertEquals(3, subpattern.loweredPrimitiveCount());
        assertEquals(GpuLoweringUnsupportedReason.SUPPORTED, subpattern.reason());
    }

    @Test
    void recordsMatmulBiasActivationEpilogueFusionSubpattern() {
        GpuFusionSubpatternSummary subpattern = GpuFusionSubpatternSummary.supported(
                GpuCompoundPatternType.LINEAR_BIAS_ACTIVATION,
                List.of(30, 31, 32),
                List.of("p0", "p1", "p2"),
                "matmul epilogue"
        );

        assertEquals(GpuCompoundPatternType.LINEAR_BIAS_ACTIVATION, subpattern.patternType());
        assertTrue(subpattern.supported());
        assertEquals(List.of(30, 31, 32), subpattern.originalOperationNodeIds());
        assertEquals(3, subpattern.loweredPrimitiveCount());
        assertTrue(subpattern.detail().contains("epilogue"));
    }

    @Test
    void rejectsCpuFusedOpTypeAsGpuFusionSubpattern() {
        AcceleratorSubgraphSpec subgraph = new AcceleratorSubgraphSpec(
                1,
                List.of(1),
                List.of(new AcceleratorSubgraphOp(1, Operation.OpType.FUSED)),
                List.of(0),
                List.of(1)
        );

        GpuCompoundRegionSummary summary = GpuCompoundPatternDetector.detect(
                ComputeBackend.GPU_METAL,
                subgraph,
                null,
                null
        );
        GpuFusionSubpatternSummary subpattern = GpuFusionSubpatternSummary.unsupported(
                summary.patternType(),
                summary.reason(),
                summary.orderedNodeIds(),
                summary.detail()
        );

        assertEquals(GpuCompoundPatternType.CPU_FUSED_UNSUPPORTED, subpattern.patternType());
        assertFalse(subpattern.supported());
        assertEquals(GpuLoweringUnsupportedReason.CPU_FUSED_OPERATION_UNSUPPORTED, subpattern.reason());
    }

    private static AcceleratorDagSpec linearReluDag() {
        return new AcceleratorDagSpec(
                List.of(
                        new AcceleratorDagInput(0, List.of(2, 3), DataType.FLOAT32),
                        new AcceleratorDagInput(1, List.of(3, 4), DataType.FLOAT32),
                        new AcceleratorDagInput(2, List.of(4), DataType.FLOAT32)
                ),
                List.of(
                        new AcceleratorDagNode(3, AcceleratorDagNodeType.LINEAR, AcceleratorDagValueRef.externalInput(0), AcceleratorDagValueRef.externalInput(1), AcceleratorDagValueRef.externalInput(2), AcceleratorDagValueRef.none(), 0, 2, 2, 4, 1, 1),
                        new AcceleratorDagNode(4, AcceleratorDagNodeType.RELU, AcceleratorDagValueRef.nodeOutput(0), AcceleratorDagValueRef.none(), AcceleratorDagValueRef.none(), AcceleratorDagValueRef.none(), 0, 2, 2, 4, 1, 1)
                ),
                List.of(1),
                List.of(4)
        );
    }

    private static AcceleratorDagSpec elementwiseDag() {
        return new AcceleratorDagSpec(
                List.of(
                        new AcceleratorDagInput(0, List.of(4), DataType.FLOAT32),
                        new AcceleratorDagInput(1, List.of(4), DataType.FLOAT32)
                ),
                List.of(
                        new AcceleratorDagNode(2, AcceleratorDagNodeType.ADD, AcceleratorDagValueRef.externalInput(0), AcceleratorDagValueRef.externalInput(1), AcceleratorDagValueRef.none(), AcceleratorDagValueRef.none(), 0, 1, 4, 1, 1, 1),
                        new AcceleratorDagNode(3, AcceleratorDagNodeType.RELU, AcceleratorDagValueRef.nodeOutput(0), AcceleratorDagValueRef.none(), AcceleratorDagValueRef.none(), AcceleratorDagValueRef.none(), 0, 1, 4, 1, 1, 1),
                        new AcceleratorDagNode(4, AcceleratorDagNodeType.EXP, AcceleratorDagValueRef.nodeOutput(1), AcceleratorDagValueRef.none(), AcceleratorDagValueRef.none(), AcceleratorDagValueRef.none(), 0, 1, 4, 1, 1, 1)
                ),
                List.of(2),
                List.of(4)
        );
    }

    private static AcceleratorDagSpec singleNodeDag(int nodeId, AcceleratorDagNodeType type) {
        return new AcceleratorDagSpec(
                List.of(new AcceleratorDagInput(0, List.of(4), DataType.FLOAT32)),
                List.of(new AcceleratorDagNode(nodeId, type, AcceleratorDagValueRef.externalInput(0), AcceleratorDagValueRef.none(), AcceleratorDagValueRef.none(), AcceleratorDagValueRef.none(), 0, 1, 4, 1, 1, 1)),
                List.of(0),
                List.of(nodeId)
        );
    }
}
