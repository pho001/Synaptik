package graph.optimizer.region.lowering;

import graph.CompiledNode;
import graph.optimizer.partition.PartitionTarget;
import operations.Operation;

/**
 * Conservative backend-aware lowering policy used for traceable region unit decisions.
 */
public final class DefaultRegionLoweringPolicy implements RegionLoweringPolicy {
    @Override
    public RegionLoweringDecision decide(RegionLoweringPolicyContext context, CompiledNode node) {
        Operation operation = node == null ? null : node.operation();
        OperationSemanticLevel level = OperationSemanticClassifier.classify(operation);
        if (operation == null || operation.opType() == null) {
            return RegionLoweringDecision.reject(level, "missing operation");
        }
        if (context == null || context.target() == PartitionTarget.NONE) {
            return RegionLoweringDecision.reject(level, "missing backend owner");
        }
        if (context.target() == PartitionTarget.CPU) {
            return decideCpu(level, operation.opType());
        }
        return decideGpu(level, operation.opType());
    }

    private static RegionLoweringDecision decideCpu(OperationSemanticLevel level, Operation.OpType opType) {
        if (opType == Operation.OpType.FUSED) {
            return RegionLoweringDecision.fuse(level, RegionLoweringForm.CPU_FUSED_LOOP, "CPU fused loop");
        }
        return RegionLoweringDecision.keep(level, RegionLoweringForm.BACKEND_PRIMITIVE, "CPU backend primitive/kernel");
    }

    private static RegionLoweringDecision decideGpu(OperationSemanticLevel level, Operation.OpType opType) {
        return switch (opType) {
            case ADD, SUB, MUL, DIV, MIN, MAX, WHERE,
                 NEG, INV, LOG, EXP, FAST_EXP, TANH, FAST_TANH,
                 POW, SQRT, ABS, MUL_SCALAR, RELU, CLAMP_MIN, CLAMP_MAX, SIGMOID ->
                    RegionLoweringDecision.fuse(level, RegionLoweringForm.FUSED_ELEMENTWISE, "GPU region-internal elementwise fusion candidate");
            case MATMUL, LINEAR ->
                    RegionLoweringDecision.keep(level, RegionLoweringForm.BACKEND_PRIMITIVE, "GPU matmul/linear primitive");
            case SOFTMAX, LOG_SOFTMAX, SUM, MEAN, REDUCE_MIN, REDUCE_MAX, LAYER_NORM, RMS_NORM ->
                    RegionLoweringDecision.lower(level, RegionLoweringForm.BACKEND_DAG, "GPU reduction/normalization DAG");
            case SCALED_DOT_PRODUCT_ATTENTION ->
                    RegionLoweringDecision.keep(level, RegionLoweringForm.SDPA, "GPU SDPA region primitive");
            case CONV2D, CONV2D_GEMM, MAX_POOL2D, AVG_POOL2D ->
                    RegionLoweringDecision.keep(level, RegionLoweringForm.CONV_POOL, "GPU conv/pool region primitive");
            case RESHAPE, EXPAND, SELECT, PERMUTE, EXPAND_DIMS, SQUEEZE, CONTIGUOUS, NOOP ->
                    RegionLoweringDecision.lower(level, RegionLoweringForm.LAYOUT_REPAIR, "GPU layout/view handling");
            case NLL_LOSS, CROSS_ENTROPY_LOSS, CROSS_ENTROPY_LOSS_INDICES ->
                    RegionLoweringDecision.lower(level, RegionLoweringForm.LOSS_SUBDAG, "GPU loss-adjacent DAG");
            case SOFTMAX_GRAD, LOG_SOFTMAX_GRAD, MIN_GRAD, MAX_GRAD,
                 REDUCE_MIN_GRAD, REDUCE_MAX_GRAD,
                 CROSS_ENTROPY_LOSS_INDICES_GRAD,
                 SCALED_DOT_PRODUCT_ATTENTION_BACKWARD,
                 CONV2D_BACKWARD_INPUT, CONV2D_BACKWARD_WEIGHT,
                 CONV2D_BACKWARD_INPUT_GEMM, CONV2D_BACKWARD_WEIGHT_GEMM,
                 MAX_POOL2D_BACKWARD_INPUT, AVG_POOL2D_BACKWARD_INPUT ->
                    RegionLoweringDecision.lower(level, RegionLoweringForm.BACKEND_DAG, "GPU training/backward DAG");
            case FUSED ->
                    RegionLoweringDecision.reject(level, "CPU FUSED node is not valid inside GPU region");
            default -> RegionLoweringDecision.keep(level, RegionLoweringForm.BACKEND_PRIMITIVE, "GPU backend primitive/kernel");
        };
    }
}
