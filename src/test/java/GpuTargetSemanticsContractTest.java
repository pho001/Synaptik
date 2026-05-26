import backend.accelerator.lowering.GpuLoweringOperationFamily;
import backend.accelerator.lowering.GpuTargetSemanticsContract;
import operations.Operation;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class GpuTargetSemanticsContractTest {
    @Test
    void reductionContractsLockAxisKeepDimsAndOutputShapeSemantics() {
        GpuTargetSemanticsContract sum = GpuTargetSemanticsContract.forOp(Operation.OpType.SUM);

        assertNotNull(sum);
        assertEquals(GpuLoweringOperationFamily.REDUCTION, sum.family());
        assertTrue(sum.shapeContract().contains("keepDims"));
        assertTrue(sum.parameterContract().contains("axis"));
        assertFalse(sum.plannerAdmissionBlocked());
    }

    @Test
    void sdpaContractDefinesNarrowAdmissionAndRemainingCapabilityGates() {
        GpuTargetSemanticsContract sdpa = GpuTargetSemanticsContract.forOp(Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION);

        assertNotNull(sdpa);
        assertEquals(GpuLoweringOperationFamily.ATTENTION, sdpa.family());
        assertFalse(sdpa.plannerAdmissionBlocked());
        assertTrue(sdpa.dtypeContract().contains("FLOAT32"));
        assertTrue(sdpa.rankContract().contains("rank 3 or 4"));
        assertTrue(sdpa.shapeContract().contains("query/key"));
        assertTrue(sdpa.blockerReason().contains("scale"));
        assertTrue(sdpa.blockerReason().contains("capability-gated"));
        assertTrue(sdpa.parameterContract().contains("unmasked"));
        assertTrue(sdpa.parameterContract().contains("causal"));
        assertTrue(sdpa.parameterContract().contains("masked"));
    }

    @Test
    void matmulContractCoversBatchedShapeBroadcastAndGradientReductionSemantics() {
        GpuTargetSemanticsContract matmul = GpuTargetSemanticsContract.forOp(Operation.OpType.MATMUL);

        assertNotNull(matmul);
        assertEquals(GpuLoweringOperationFamily.MATMUL_LINEAR, matmul.family());
        assertFalse(matmul.plannerAdmissionBlocked());
        assertTrue(matmul.dtypeContract().contains("floating"));
        assertTrue(matmul.rankContract().contains("rank >= 2"));
        assertTrue(matmul.rankContract().contains("last-two dimensions"));
        assertTrue(matmul.shapeContract().contains("[..., M, K]"));
        assertTrue(matmul.shapeContract().contains("[..., K, N]"));
        assertTrue(matmul.shapeContract().contains("broadcast"));
        assertTrue(matmul.parameterContract().contains("rank-2 MATMUL"));
        assertTrue(matmul.parameterContract().contains("general DAG"));
        assertTrue(matmul.numericalContract().contains("CPU parity"));
        assertTrue(matmul.numericalContract().contains("gradient reductions"));
        assertTrue(matmul.numericalContract().contains("broadcast leading dimensions"));
    }

    @Test
    void boolCompareContractDistinguishesBoolOutputResidency() {
        GpuTargetSemanticsContract gt = GpuTargetSemanticsContract.forOp(Operation.OpType.GT);

        assertNotNull(gt);
        assertEquals(GpuLoweringOperationFamily.COMPARE_BOOL, gt.family());
        assertTrue(gt.dtypeContract().contains("BOOL output"));
        assertTrue(gt.parameterContract().contains("WHERE"));
    }

    @Test
    void phaseTwentySevenContractsCoverBoolLogicalAndBoolReductions() {
        for (Operation.OpType opType : new Operation.OpType[]{
                Operation.OpType.GT,
                Operation.OpType.GE,
                Operation.OpType.LT,
                Operation.OpType.LE,
                Operation.OpType.EQ,
                Operation.OpType.NE,
                Operation.OpType.LOGICAL_AND,
                Operation.OpType.LOGICAL_OR,
                Operation.OpType.LOGICAL_NOT,
                Operation.OpType.REDUCE_ALL,
                Operation.OpType.REDUCE_ANY
        }) {
            GpuTargetSemanticsContract contract = GpuTargetSemanticsContract.forOp(opType);

            assertNotNull(contract, () -> "missing Phase 27 BOOL contract for " + opType);
            assertEquals(GpuLoweringOperationFamily.COMPARE_BOOL, contract.family());
            assertTrue(contract.dtypeContract().contains("BOOL"));
            assertTrue(contract.numericalContract().contains("CPU parity"));
        }

        GpuTargetSemanticsContract all = GpuTargetSemanticsContract.forOp(Operation.OpType.REDUCE_ALL);
        assertTrue(all.rankContract().contains("keepDims"));
        assertTrue(all.blockerReason().contains("native BOOL output support"));
    }

    @Test
    void phaseTwentySevenContractsCoverConvPoolVariants() {
        for (Operation.OpType opType : new Operation.OpType[]{
                Operation.OpType.CONV2D,
                Operation.OpType.MAX_POOL2D,
                Operation.OpType.AVG_POOL2D
        }) {
            GpuTargetSemanticsContract contract = GpuTargetSemanticsContract.forOp(opType);

            assertNotNull(contract, () -> "missing Phase 27 conv/pool contract for " + opType);
            assertEquals(GpuLoweringOperationFamily.CONV_POOL, contract.family());
            assertTrue(contract.dtypeContract().contains("floating"));
            assertTrue(contract.rankContract().contains("rank 4"));
            assertTrue(contract.shapeContract().contains("stride"));
            assertTrue(contract.layoutContract().contains("layout"));
        }
    }

    @Test
    void windowLayoutContractsCoverUnfoldAndFoldGeometry() {
        GpuTargetSemanticsContract unfold = GpuTargetSemanticsContract.forOp(Operation.OpType.UNFOLD2D);
        GpuTargetSemanticsContract fold = GpuTargetSemanticsContract.forOp(Operation.OpType.FOLD2D);
        GpuTargetSemanticsContract unfoldAxis = GpuTargetSemanticsContract.forOp(Operation.OpType.UNFOLD_AXIS);

        assertNotNull(unfold);
        assertNotNull(fold);
        assertNotNull(unfoldAxis);
        assertEquals(GpuLoweringOperationFamily.LAYOUT_VIEW_ADJACENT, unfold.family());
        assertEquals(GpuLoweringOperationFamily.LAYOUT_VIEW_ADJACENT, fold.family());
        assertEquals(GpuLoweringOperationFamily.LAYOUT_VIEW_ADJACENT, unfoldAxis.family());
        assertTrue(unfold.dtypeContract().contains("floating value dtype"));
        assertTrue(fold.dtypeContract().contains("floating value dtype"));
        assertTrue(unfoldAxis.dtypeContract().contains("INT32"));
        assertTrue(unfoldAxis.dtypeContract().contains("BOOL"));
        assertTrue(unfold.rankContract().contains("rank-4 NCHW input"));
        assertTrue(unfold.rankContract().contains("rank-3"));
        assertTrue(fold.rankContract().contains("rank-3 column input"));
        assertTrue(fold.rankContract().contains("rank-4 NCHW outputShape"));
        assertTrue(unfoldAxis.rankContract().contains("input rank + 1"));
        assertTrue(unfold.shapeContract().contains("kernel"));
        assertTrue(unfold.shapeContract().contains("stride"));
        assertTrue(unfold.shapeContract().contains("padding"));
        assertTrue(unfold.shapeContract().contains("dilation"));
        assertTrue(unfold.shapeContract().contains("ceilMode"));
        assertTrue(fold.shapeContract().contains("outputShape"));
        assertTrue(fold.shapeContract().contains("column count"));
        assertTrue(unfoldAxis.shapeContract().contains("trailing size dimension"));
        assertTrue(unfoldAxis.parameterContract().contains("no padding"));
        assertTrue(unfoldAxis.parameterContract().contains("no dilation"));
        assertTrue(fold.numericalContract().contains("duplicate window accumulation"));
        assertFalse(unfold.plannerAdmissionBlocked());
        assertFalse(fold.plannerAdmissionBlocked());
        assertFalse(unfoldAxis.plannerAdmissionBlocked());
        assertTrue(unfold.blockerReason().contains("Scoped native GPU support"));
        assertTrue(fold.blockerReason().contains("visible planner rejections"));
        assertTrue(unfoldAxis.blockerReason().contains("axis-window materialization"));
    }

    @Test
    void phaseTwentySixContractsDistinguishResidencyFromNativeIndexCompute() {
        GpuTargetSemanticsContract loss = GpuTargetSemanticsContract.forOp(Operation.OpType.CROSS_ENTROPY_LOSS_INDICES);
        GpuTargetSemanticsContract gather = GpuTargetSemanticsContract.forOp(Operation.OpType.GATHER);
        GpuTargetSemanticsContract gatherNd = GpuTargetSemanticsContract.forOp(Operation.OpType.GATHER_ND);
        GpuTargetSemanticsContract gatherNdGrad = GpuTargetSemanticsContract.forOp(Operation.OpType.GATHER_ND_GRAD);
        GpuTargetSemanticsContract scatter = GpuTargetSemanticsContract.forOp(Operation.OpType.SCATTER_ADD);
        GpuTargetSemanticsContract scatterElements = GpuTargetSemanticsContract.forOp(Operation.OpType.SCATTER_ELEMENTS);
        GpuTargetSemanticsContract scatterNd = GpuTargetSemanticsContract.forOp(Operation.OpType.SCATTER_ND);
        GpuTargetSemanticsContract gatherGrad = GpuTargetSemanticsContract.forOp(Operation.OpType.GATHER_GRAD);
        GpuTargetSemanticsContract takeGrad = GpuTargetSemanticsContract.forOp(Operation.OpType.TAKE_ALONG_AXIS_GRAD);

        assertNotNull(loss);
        assertNotNull(gather);
        assertNotNull(gatherNd);
        assertNotNull(gatherNdGrad);
        assertNotNull(scatter);
        assertNotNull(scatterElements);
        assertNotNull(scatterNd);
        assertNotNull(gatherGrad);
        assertNotNull(takeGrad);
        assertEquals(GpuLoweringOperationFamily.LOSS_ADJACENT, loss.family());
        assertEquals(GpuLoweringOperationFamily.INDEX_SCATTER_GATHER, gather.family());
        assertTrue(loss.dtypeContract().contains("INT32 index targets"));
        assertTrue(loss.dtypeContract().contains("admitted Metal loss candidates"));
        assertTrue(loss.numericalContract().contains("ignore-index"));
        assertTrue(gather.dtypeContract().contains("native compute support is operation-specific"));
        assertFalse(gatherNd.plannerAdmissionBlocked());
        assertEquals("", gatherNd.blockerReason());
        assertTrue(gatherNdGrad.plannerAdmissionBlocked());
        assertTrue(gatherNdGrad.blockerReason().contains("UNSUPPORTED_INDEX_SEMANTICS"));
        assertTrue(scatter.numericalContract().contains("duplicate indices"));
        assertTrue(scatter.plannerAdmissionBlocked());
        assertTrue(scatterElements.plannerAdmissionBlocked());
        assertTrue(scatterNd.plannerAdmissionBlocked());
        assertTrue(gatherGrad.plannerAdmissionBlocked());
        assertTrue(takeGrad.plannerAdmissionBlocked());
        assertTrue(scatter.blockerReason().contains("UNSUPPORTED_DUPLICATE_INDEX"));
        assertTrue(scatterElements.blockerReason().contains("UNSUPPORTED_INDEX_SEMANTICS"));
        assertTrue(scatterNd.blockerReason().contains("UNSUPPORTED_INDEX_SEMANTICS"));
        assertTrue(gatherGrad.shapeContract().contains("original input shape"));
        assertTrue(takeGrad.shapeContract().contains("original input shape"));
        assertTrue(takeGrad.numericalContract().contains("logical-index accumulation order"));
    }

    @Test
    void phaseThirtySevenDenseLossContractsLockScopeWithoutPromotingIndexTargets() {
        GpuTargetSemanticsContract nll = GpuTargetSemanticsContract.forOp(Operation.OpType.NLL_LOSS);
        GpuTargetSemanticsContract denseCe = GpuTargetSemanticsContract.forOp(Operation.OpType.CROSS_ENTROPY_LOSS);
        GpuTargetSemanticsContract indexCe = GpuTargetSemanticsContract.forOp(Operation.OpType.CROSS_ENTROPY_LOSS_INDICES);

        assertNotNull(nll);
        assertNotNull(denseCe);
        assertNotNull(indexCe);
        assertEquals(GpuLoweringOperationFamily.LOSS_ADJACENT, nll.family());
        assertEquals(GpuLoweringOperationFamily.LOSS_ADJACENT, denseCe.family());
        assertTrue(nll.dtypeContract().contains("dense dtype-matched FLOAT32"));
        assertTrue(denseCe.shapeContract().contains("output shape is [1]"));
        assertTrue(denseCe.numericalContract().contains("sample-count denominator"));
        assertTrue(denseCe.blockerReason().contains("Metal admits"));
        assertFalse(denseCe.plannerAdmissionBlocked());

        assertTrue(indexCe.dtypeContract().contains("INT32 index targets"));
        assertTrue(indexCe.blockerReason().contains("UNSUPPORTED_INDEX_SEMANTICS"));
        assertTrue(indexCe.blockerReason().contains("scatter/index-gradient blockers"));
    }
}
