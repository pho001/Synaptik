package backend.accelerator.lowering;

import backend.ComputeBackend;
import backend.accelerator.dag.AcceleratorDagNode;
import backend.accelerator.dag.AcceleratorDagNodeType;
import backend.accelerator.dag.AcceleratorDagSpec;
import backend.accelerator.dag.AcceleratorDagValueRef;
import backend.accelerator.dag.AcceleratorDagValueRefKind;
import backend.accelerator.dag.AcceleratorPostOp;
import backend.accelerator.dag.AcceleratorPostOpType;
import backend.accelerator.dag.AcceleratorSubgraphOp;
import backend.accelerator.dag.AcceleratorSubgraphSpec;
import graph.CompiledNode;
import graph.optimizer.partition.PartitionPlanningContext;
import operations.Operation;

import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

/**
 * Classifies lowered accelerator DAGs into stable compound GPU pattern summaries.
 */
public final class GpuCompoundPatternDetector {
    private static final Set<AcceleratorDagNodeType> ELEMENTWISE_CHAIN_TYPES = EnumSet.of(
            AcceleratorDagNodeType.ADD,
            AcceleratorDagNodeType.SUB,
            AcceleratorDagNodeType.MUL,
            AcceleratorDagNodeType.DIV,
            AcceleratorDagNodeType.MIN,
            AcceleratorDagNodeType.MAX,
            AcceleratorDagNodeType.RELU,
            AcceleratorDagNodeType.TANH,
            AcceleratorDagNodeType.SIGMOID,
            AcceleratorDagNodeType.ABS,
            AcceleratorDagNodeType.EXP,
            AcceleratorDagNodeType.LOG,
            AcceleratorDagNodeType.NEG,
            AcceleratorDagNodeType.SQRT,
            AcceleratorDagNodeType.INV,
            AcceleratorDagNodeType.MUL_SCALAR,
            AcceleratorDagNodeType.POW_SCALAR,
            AcceleratorDagNodeType.ADD_SCALAR,
            AcceleratorDagNodeType.CLAMP_MIN,
            AcceleratorDagNodeType.CLAMP_MAX
    );

    private static final Set<AcceleratorPostOpType> ACTIVATION_POST_OPS = EnumSet.of(
            AcceleratorPostOpType.RELU,
            AcceleratorPostOpType.SIGMOID,
            AcceleratorPostOpType.TANH
    );

    private GpuCompoundPatternDetector() {
    }

    /**
     * Detects the compound pattern represented by a lowered accelerator subgraph.
     */
    public static GpuCompoundRegionSummary detect(
            ComputeBackend backend,
            AcceleratorSubgraphSpec subgraph,
            PartitionPlanningContext context,
            AcceleratorDagSpec dagSpec,
            AcceleratorMatMulSpec matMulSpec
    ) {
        if (subgraph == null) {
            return GpuCompoundRegionSummary.none(backend, List.of());
        }
        if (containsOpType(subgraph, Operation.OpType.FUSED)) {
            return unsupported(
                    backend,
                    subgraph,
                    dagSpec,
                    GpuCompoundPatternType.CPU_FUSED_UNSUPPORTED,
                    GpuLoweringUnsupportedReason.CPU_FUSED_OPERATION_UNSUPPORTED,
                    "CPU Operation.OpType.FUSED remains CPU-only for Phase 12"
            );
        }
        if (isLinearBiasActivation(matMulSpec)) {
            return GpuCompoundRegionSummary.supported(
                    backend,
                    GpuCompoundPatternType.LINEAR_BIAS_ACTIVATION,
                    subgraph.orderedNodeIds(),
                    subgraph.externalInputNodeIds(),
                    subgraph.outputNodeIds(),
                    dagNodeTypes(dagSpec),
                    postOps(matMulSpec),
                    "linear or matmul with bias and activation lowered through accelerator DAG"
            );
        }
        if (isElementwiseChain(dagSpec)) {
            return GpuCompoundRegionSummary.supported(
                    backend,
                    GpuCompoundPatternType.ELEMENTWISE_CHAIN,
                    subgraph.orderedNodeIds(),
                    subgraph.externalInputNodeIds(),
                    subgraph.outputNodeIds(),
                    dagNodeTypes(dagSpec),
                    List.of(),
                    "representative elementwise chain lowered through accelerator DAG"
            );
        }
        if (isNormalizationSubdag(subgraph, dagSpec)) {
            return GpuCompoundRegionSummary.supported(
                    backend,
                    GpuCompoundPatternType.NORMALIZATION,
                    subgraph.orderedNodeIds(),
                    subgraph.externalInputNodeIds(),
                    subgraph.outputNodeIds(),
                    dagNodeTypes(dagSpec),
                    List.of(),
                    "normalization lowered as region-internal reduction and elementwise DAG"
            );
        }
        if (containsReductionAdjacent(subgraph, context)) {
            return unsupported(
                    backend,
                    subgraph,
                    dagSpec,
                    GpuCompoundPatternType.REDUCTION_ADJACENT,
                    GpuLoweringUnsupportedReason.COMPOUND_PATTERN_UNSUPPORTED,
                    "REDUCTION_ADJACENT compound candidate is not supported by the Phase 12 minimal subset"
            );
        }
        return GpuCompoundRegionSummary.none(backend, subgraph.orderedNodeIds());
    }

    /**
     * Detects using source operation metadata when no planning context is needed.
     */
    public static GpuCompoundRegionSummary detect(
            ComputeBackend backend,
            AcceleratorSubgraphSpec subgraph,
            AcceleratorDagSpec dagSpec,
            AcceleratorMatMulSpec matMulSpec
    ) {
        return detect(backend, subgraph, null, dagSpec, matMulSpec);
    }

    /**
     * Detects maximal region-internal elementwise primitive chains in a lowered DAG.
     *
     * <p>The returned spans are original compiled node ids, not primitive indexes, so trace metadata stays tied to the
     * public graph even when a backend lowers one operation into multiple primitives.</p>
     */
    public static List<List<Integer>> detectElementwiseSubchains(AcceleratorDagSpec dagSpec) {
        if (dagSpec == null || dagSpec.nodes().size() < 2) {
            return List.of();
        }
        ArrayList<List<Integer>> out = new ArrayList<>();
        int index = 0;
        while (index < dagSpec.nodes().size()) {
            if (!isElementwiseDagNodeType(dagSpec.nodes().get(index).type())) {
                index++;
                continue;
            }
            ArrayList<Integer> chain = new ArrayList<>();
            chain.add(dagSpec.nodes().get(index).nodeId());
            int cursor = index + 1;
            while (cursor < dagSpec.nodes().size()
                    && isElementwiseDagNodeType(dagSpec.nodes().get(cursor).type())
                    && consumesPrimitiveOutput(dagSpec.nodes().get(cursor), cursor - 1)) {
                chain.add(dagSpec.nodes().get(cursor).nodeId());
                cursor++;
            }
            if (chain.size() >= 2) {
                out.add(List.copyOf(chain));
            }
            index = cursor;
        }
        return List.copyOf(out);
    }

    static boolean isElementwiseDagNodeType(AcceleratorDagNodeType type) {
        return ELEMENTWISE_CHAIN_TYPES.contains(type);
    }

    private static boolean isLinearBiasActivation(AcceleratorMatMulSpec matMulSpec) {
        return matMulSpec != null
                && matMulSpec.biasInputNodeId() >= 0
                && matMulSpec.postOps().stream().map(AcceleratorPostOp::type).anyMatch(ACTIVATION_POST_OPS::contains);
    }

    private static boolean isElementwiseChain(AcceleratorDagSpec dagSpec) {
        return dagSpec != null
                && dagSpec.nodes().size() >= 3
                && dagSpec.nodes().stream().allMatch(node -> ELEMENTWISE_CHAIN_TYPES.contains(node.type()));
    }

    private static boolean isNormalizationSubdag(AcceleratorSubgraphSpec subgraph, AcceleratorDagSpec dagSpec) {
        if (subgraph == null || dagSpec == null) {
            return false;
        }
        boolean normalizationOp = subgraph.ops().stream()
                .map(AcceleratorSubgraphOp::opType)
                .anyMatch(opType -> opType == Operation.OpType.LAYER_NORM || opType == Operation.OpType.RMS_NORM);
        if (!normalizationOp) {
            return false;
        }
        boolean hasMean = dagSpec.nodes().stream().anyMatch(node -> node.type() == AcceleratorDagNodeType.MEAN);
        boolean hasEpsilon = dagSpec.nodes().stream().anyMatch(node -> node.type() == AcceleratorDagNodeType.ADD_SCALAR);
        boolean hasInv = dagSpec.nodes().stream().anyMatch(node -> node.type() == AcceleratorDagNodeType.INV);
        return hasMean && hasEpsilon && hasInv;
    }

    private static boolean consumesPrimitiveOutput(AcceleratorDagNode node, int primitiveIndex) {
        return consumesPrimitiveOutput(node.input0(), primitiveIndex)
                || consumesPrimitiveOutput(node.input1(), primitiveIndex)
                || consumesPrimitiveOutput(node.input2(), primitiveIndex)
                || consumesPrimitiveOutput(node.input3(), primitiveIndex)
                || consumesPrimitiveOutput(node.input4(), primitiveIndex);
    }

    private static boolean consumesPrimitiveOutput(AcceleratorDagValueRef ref, int primitiveIndex) {
        return ref != null && ref.kind() == AcceleratorDagValueRefKind.NODE_OUTPUT && ref.index() == primitiveIndex;
    }

    private static boolean containsOpType(AcceleratorSubgraphSpec subgraph, Operation.OpType opType) {
        return subgraph.ops().stream().map(AcceleratorSubgraphOp::opType).anyMatch(opType::equals);
    }

    private static boolean containsReductionAdjacent(AcceleratorSubgraphSpec subgraph, PartitionPlanningContext context) {
        for (AcceleratorSubgraphOp op : subgraph.ops()) {
            if (isReductionAdjacent(op.opType())) {
                return true;
            }
        }
        if (context == null) {
            return false;
        }
        for (int nodeId : subgraph.orderedNodeIds()) {
            CompiledNode node = context.compiledNode(nodeId);
            if (node != null && node.operation() != null && isReductionAdjacent(node.operation().opType())) {
                return true;
            }
        }
        return false;
    }

    private static boolean isReductionAdjacent(Operation.OpType opType) {
        return switch (opType) {
            case SUM, MEAN, REDUCE_MIN, REDUCE_MAX, LAYER_NORM, RMS_NORM -> true;
            default -> false;
        };
    }

    private static GpuCompoundRegionSummary unsupported(
            ComputeBackend backend,
            AcceleratorSubgraphSpec subgraph,
            AcceleratorDagSpec dagSpec,
            GpuCompoundPatternType patternType,
            GpuLoweringUnsupportedReason reason,
            String detail
    ) {
        return GpuCompoundRegionSummary.unsupported(
                backend,
                patternType,
                reason,
                detail,
                subgraph.orderedNodeIds(),
                subgraph.externalInputNodeIds(),
                subgraph.outputNodeIds(),
                dagNodeTypes(dagSpec),
                List.of()
        );
    }

    private static List<String> dagNodeTypes(AcceleratorDagSpec dagSpec) {
        if (dagSpec == null) {
            return List.of();
        }
        return dagSpec.nodes().stream().map(node -> node.type().name()).toList();
    }

    private static List<String> postOps(AcceleratorMatMulSpec matMulSpec) {
        if (matMulSpec == null) {
            return List.of();
        }
        return matMulSpec.postOps().stream().map(postOp -> postOp.type().name()).toList();
    }
}
