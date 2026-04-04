package backend;

import backend.kernels.cpu.CpuExecutionPlanner;
import backend.kernels.cpu.CpuKernel;
import backend.kernels.cpu.CpuKernelContext;
import backend.kernels.cpu.CpuStridedElementWise;
import backend.kernels.cpu.CpuNodeExecutionPlan;
import backend.kernels.cpu.ResolvedBroadcastPlan;
import backend.kernels.cpu.ResolvedDispatchHints;
import backend.kernels.cpu.ResolvedMatMulHints;
import backend.kernels.cpu.ResolvedReductionHints;
import backend.kernels.cpu.ResolvedWhereBroadcastPlan;
import backend.runtime.BlasConfig;
import backend.runtime.ExecutionContext;
import graph.execution.CompiledNodeExecutionMetadata;
import graph.codegen.FusedExternalInputPlan;
import operations.Operation;
import operations.FusedOperation;
import tensor.BroadcastPlan;
import tensor.BroadcastPlanner;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorRemap;
import tensor.WhereBroadcastPlan;
import tensor.WhereBroadcastPlanner;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

public final class CPUBackend {
    private record PreparedTypeContract(
            DataType outputType,
            List<DataType> expectedInputTypes
    ) {}

    private record PreparedInputsResult(
            List<backend.CpuPreparedInput> preparedInputs,
            List<Tensor> runtimeInputs
    ) {}

    public void execute(
            Tensor node,
            CompiledNodeExecutionMetadata metadata,
            ExecutionContext executionContext
    ) {
        Operation op = node.getOperation();
        if (op == null) {
            return;
        }

        CpuKernel kernel = metadata.cpuKernel();
        if (kernel == null) {
            throw new UnsupportedOperationException(
                    "Missing CPU kernel for opType=" + op.opType() +
                            " (operation class: " + op.getClass().getName() + ")"
            );
        }

        CpuExecutionPlanner planner = executionContext.cpuPlanner();

        CpuNodeExecutionPlan executionPlan = metadata.cpuPlan();
        if (executionPlan == null) {
            throw new IllegalStateException("Missing CpuNodeExecutionPlan for node " + node.getLabel());
        }

        List<Tensor> inputs = executionPlan.apply(node.getPrevTensors());
        if (executionPlan.stridedPath()) {
            CpuStridedElementWise.forward(op, inputs, node, new CpuKernelContext(planner, executionPlan, executionContext, metadata));
            return;
        }

        CpuKernelContext kernelContext = new CpuKernelContext(planner, executionPlan, executionContext, metadata);

        switch (node.getDataType()) {
            case FLOAT64 -> kernel.forwardF64(op, inputs, node, kernelContext);
            case FLOAT32 -> kernel.forwardF32(op, inputs, node, kernelContext);
            case FLOAT16 -> kernel.forwardF16(op, inputs, node, kernelContext);
            case INT32 -> kernel.forwardI32(op, inputs, node, kernelContext);
            case BOOL -> kernel.forwardBOOL(op, inputs, node, kernelContext);
        }

        if (node.getDataType() != DataType.FLOAT64) {
            node.markDataViewStale();
        }
    }

    public static CpuNodeExecutionPlan buildExecutionPlan(
            Operation op,
            List<Tensor> inputs,
            Tensor node,
            CpuExecutionPlanner planner,
            BlasConfig blasConfig
    ) {
        Objects.requireNonNull(node, "node cannot be null");
        Objects.requireNonNull(planner, "planner cannot be null");
        Objects.requireNonNull(blasConfig, "blasConfig cannot be null");

        List<Tensor> safeInputs = inputs == null ? List.of() : List.copyOf(inputs);
        PreparedTypeContract typeContract = resolveTypeContract(op, node, safeInputs);
        DataType targetType = typeContract.outputType();

        boolean stridedPath = canUseStridedPath(op, safeInputs, node, targetType, planner);
        PreparedInputsResult prepared = prepareInputs(op, safeInputs, node, typeContract, planner, stridedPath);

        ResolvedBroadcastPlan broadcastPlan = resolveBroadcastPlan(op, prepared.runtimeInputs(), node);
        ResolvedWhereBroadcastPlan whereBroadcastPlan = resolveWhereBroadcastPlan(op, prepared.runtimeInputs(), node);

        backend.CpuLayoutPlan layoutPlan = new backend.CpuLayoutPlan(
                stridedPath,
                targetType,
                planner.contiguousMaterializeThreshold(),
                broadcastPlan,
                whereBroadcastPlan,
                prepared.preparedInputs(),
                prepared.runtimeInputs()
        );

        ResolvedDispatchHints dispatchHints =
                (op != null && (op.opType().category() == Operation.OpArityClass.ELEMENT_WISE || op.opType() == Operation.OpType.FUSED))
                        ? planner.resolveDispatchHints(op, node)
                        : null;

        ResolvedReductionHints reductionHints =
                (op != null && switch (op.opType()) {
                    case SUM, MEAN, REDUCE_MIN, REDUCE_MAX, REDUCE_ALL, REDUCE_ANY, SOFTMAX, LOG_SOFTMAX, NLL_LOSS, CROSS_ENTROPY_LOSS -> true;
                    default -> false;
                })
                        ? planner.resolveReductionHints(estimateReductionLogicalSize(prepared.runtimeInputs(), node), targetType)
                        : null;

        ResolvedMatMulHints matMulHints =
                (op != null && op.opType() == Operation.OpType.MATMUL && prepared.runtimeInputs().size() >= 2)
                        ? planner.resolveMatMulHints(
                        prepared.runtimeInputs().get(0),
                        prepared.runtimeInputs().get(1),
                        node,
                        blasConfig
                )
                        : null;

        return new CpuNodeExecutionPlan(layoutPlan, dispatchHints, reductionHints, matMulHints);
    }

    private static PreparedInputsResult prepareInputs(
            Operation op,
            List<Tensor> inputs,
            Tensor node,
            PreparedTypeContract typeContract,
            CpuExecutionPlanner planner,
            boolean stridedPath
    ) {
        if (inputs.isEmpty()) {
            return new PreparedInputsResult(List.of(), List.of());
        }

        if (bypassPreparation(op) && !requiresPreparedInputs(op, inputs, node, typeContract, planner)) {
            return new PreparedInputsResult(List.of(), inputs);
        }

        if (stridedPath) {
            return new PreparedInputsResult(List.of(), inputs);
        }

        List<backend.CpuPreparedInput> preparedInputs = new ArrayList<>();
        List<Tensor> runtimeInputs = new ArrayList<>(inputs.size());

        for (int i = 0; i < inputs.size(); i++) {
            Tensor input = inputs.get(i);
            if (input == null) {
                throw new IllegalArgumentException("Input tensor at index " + i + " is null");
            }

            DataType expectedInputType = typeContract.expectedInputTypes().get(i);

            if (!requiresPreparedInput(op, input, node, expectedInputType, planner)) {
                runtimeInputs.add(input);
                continue;
            }

            if (!canConvertPreparedInput(input.getDataType(), expectedInputType)) {
                throw new IllegalArgumentException("Unsupported prepared input conversion for op="
                        + (op == null ? "null" : op.opType())
                        + ", inputIndex=" + i
                        + ", sourceType=" + input.getDataType()
                        + ", expectedType=" + expectedInputType);
            }

            Tensor preparedTensor = createPreparedTensor(input, expectedInputType, node, i);
            TensorRemap.RemapPlan remapPlan = TensorRemap.buildPlan(input, preparedTensor);
            preparedInputs.add(new backend.CpuPreparedInput(i, preparedTensor, remapPlan));
            runtimeInputs.add(preparedTensor);
        }

        return new PreparedInputsResult(preparedInputs, runtimeInputs);
    }

    private static boolean requiresPreparedInput(
            Operation op,
            Tensor input,
            Tensor node,
            DataType expectedInputType,
            CpuExecutionPlanner planner
    ) {
        if (input.getDataType() != expectedInputType) {
            return true;
        }

        if (input.hasStorageOffset()) {
            if (op == null) {
                return true;
            }
            return switch (op.opType()) {
                case RESHAPE, EXPAND, SELECT, PERMUTE, EXPAND_DIMS, SQUEEZE,
                        GATHER, GATHER_GRAD, TAKE_ALONG_AXIS, TAKE_ALONG_AXIS_GRAD, SCATTER_ADD,
                        SUM, MEAN, REDUCE_MIN, REDUCE_MAX, REDUCE_ALL, REDUCE_ANY,
                        SOFTMAX, LOG_SOFTMAX, NLL_LOSS, CROSS_ENTROPY_LOSS,
                        MIN_GRAD, MAX_GRAD, REDUCE_MIN_GRAD, REDUCE_MAX_GRAD,
                        NOOP -> false;
                default -> true;
            };
        }

        if (input.isContiguous()) {
            return false;
        }

        if (op == null) {
            return false;
        }

        return switch (op.opType()) {
            case CONTIGUOUS, RESHAPE, EXPAND, SELECT, PERMUTE, EXPAND_DIMS, SQUEEZE, SUM, MEAN, REDUCE_MIN, REDUCE_MAX, REDUCE_ALL, REDUCE_ANY, SOFTMAX, LOG_SOFTMAX, NLL_LOSS, CROSS_ENTROPY_LOSS, NOOP -> false;
            case MIN_GRAD, MAX_GRAD, REDUCE_MIN_GRAD, REDUCE_MAX_GRAD -> !input.isContiguous();
            case MATMUL, CONV2D, CONV2D_BACKWARD_INPUT, CONV2D_BACKWARD_WEIGHT -> true;
            case GT, GE, LT, LE, EQ, NE, WHERE, LOGICAL_AND, LOGICAL_OR, LOGICAL_NOT -> !input.isContiguous();
            default -> op.opType().category() == Operation.OpArityClass.ELEMENT_WISE
                    && planner.shouldMaterializeNonContiguous(node.getFlatDataSize());
        };
    }

    private static boolean canUseStridedPath(
            Operation op,
            List<Tensor> inputs,
            Tensor node,
            DataType targetType,
            CpuExecutionPlanner planner
    ) {
        if (op == null || node == null || inputs.isEmpty()) {
            return false;
        }
        if (op.opType() == Operation.OpType.CONTIGUOUS) {
            return false;
        }
        if (op.opType().category() != Operation.OpArityClass.ELEMENT_WISE || !CpuStridedElementWise.supports(op)) {
            return false;
        }
        if (requiresBinaryBroadcast(op, inputs, node)) {
            return false;
        }

        boolean hasOffsetInput = false;
        boolean hasNonContiguousInput = false;
        int[] outShape = node.getShapeUnsafe();

        for (int i = 0; i < inputs.size(); i++) {
            Tensor input = inputs.get(i);
            if (input == null) {
                return false;
            }
            if (!isStridedPathInputTypeCompatible(op, input, targetType, i)) {
                return false;
            }
            if (!Arrays.equals(input.getShapeUnsafe(), outShape)) {
                return false;
            }
            if (input.hasStorageOffset()) {
                hasOffsetInput = true;
            }
            if (!input.isContiguous()) {
                hasNonContiguousInput = true;
            }
        }

        if (!hasOffsetInput && !hasNonContiguousInput) {
            return false;
        }

        if (hasOffsetInput && !hasNonContiguousInput) {
            return true;
        }

        return !planner.shouldMaterializeNonContiguous(node.getFlatDataSize());
    }

    private static boolean isStridedPathInputTypeCompatible(Operation op, Tensor input, DataType targetType, int inputIndex) {
        if (op == null || input == null) {
            return false;
        }
        return switch (op.opType()) {
            case GT, GE, LT, LE, EQ, NE ->
                    input.getDataType() == DataType.FLOAT64
                            || input.getDataType() == DataType.FLOAT32
                            || input.getDataType() == DataType.FLOAT16;
            case WHERE -> inputIndex == 0
                    ? input.getDataType() == DataType.BOOL
                    : input.getDataType() == targetType;
            case LOGICAL_AND, LOGICAL_OR, LOGICAL_NOT -> input.getDataType() == DataType.BOOL;
            default -> input.getDataType() == targetType;
        };
    }

    private static boolean requiresBinaryBroadcast(Operation op, List<Tensor> inputs, Tensor node) {
        if (!supportsBinaryBroadcast(op) || inputs.size() != 2) {
            return false;
        }
        return !Arrays.equals(inputs.get(0).getShapeUnsafe(), node.getShapeUnsafe())
                || !Arrays.equals(inputs.get(1).getShapeUnsafe(), node.getShapeUnsafe());
    }

    private static boolean supportsBinaryBroadcast(Operation op) {
        if (op == null || op.opType() == null) {
            return false;
        }
        return switch (op.opType()) {
            case ADD, SUB, MUL, DIV, MIN, MAX, GT, GE, LT, LE, EQ, NE, LOGICAL_AND, LOGICAL_OR -> true;
            default -> false;
        };
    }

    private static ResolvedBroadcastPlan resolveBroadcastPlan(
            Operation op,
            List<Tensor> runtimeInputs,
            Tensor node
    ) {
        if (!supportsBinaryBroadcast(op) || runtimeInputs.size() != 2) {
            return null;
        }

        BroadcastPlan plan = BroadcastPlanner.plan(runtimeInputs.get(0), runtimeInputs.get(1));
        if (!Arrays.equals(plan.outShape(), node.getShapeUnsafe())) {
            throw new IllegalStateException(
                    "Resolved broadcast output shape " + Arrays.toString(plan.outShape()) +
                            " does not match node shape " + Arrays.toString(node.getShapeUnsafe())
            );
        }
        return ResolvedBroadcastPlan.from(plan);
    }

    private static ResolvedWhereBroadcastPlan resolveWhereBroadcastPlan(
            Operation op,
            List<Tensor> runtimeInputs,
            Tensor node
    ) {
        if (op == null || op.opType() != Operation.OpType.WHERE || runtimeInputs.size() != 3) {
            return null;
        }
        WhereBroadcastPlan plan = WhereBroadcastPlanner.plan(runtimeInputs.get(0), runtimeInputs.get(1), runtimeInputs.get(2));
        if (!Arrays.equals(plan.outShape(), node.getShapeUnsafe())) {
            throw new IllegalStateException(
                    "Resolved where output shape " + Arrays.toString(plan.outShape()) +
                            " does not match node shape " + Arrays.toString(node.getShapeUnsafe())
            );
        }
        return ResolvedWhereBroadcastPlan.from(plan);
    }

    private static int estimateReductionLogicalSize(List<Tensor> runtimeInputs, Tensor node) {
        if (runtimeInputs != null && !runtimeInputs.isEmpty() && runtimeInputs.get(0) != null) {
            return runtimeInputs.get(0).getFlatDataSize();
        }
        return node.getFlatDataSize();
    }

    private static boolean bypassPreparation(Operation op) {
        if (op == null || op.opType() == null) {
            return false;
        }
        return switch (op.opType()) {
            case CONTIGUOUS, RESHAPE, EXPAND, SELECT, PERMUTE, EXPAND_DIMS, SQUEEZE, SUM, MEAN, REDUCE_MIN, REDUCE_MAX, SOFTMAX, LOG_SOFTMAX, NLL_LOSS, CROSS_ENTROPY_LOSS,
                    REDUCE_MIN_GRAD, REDUCE_MAX_GRAD, NOOP -> true;
            default -> false;
        };
    }

    private static boolean requiresPreparedInputs(
            Operation op,
            List<Tensor> inputs,
            Tensor node,
            PreparedTypeContract typeContract,
            CpuExecutionPlanner planner
    ) {
        if (inputs == null || inputs.isEmpty()) {
            return false;
        }
        for (int i = 0; i < inputs.size(); i++) {
            Tensor input = inputs.get(i);
            if (input == null) {
                return true;
            }
            DataType expectedInputType = typeContract.expectedInputTypes().get(i);
            if (requiresPreparedInput(op, input, node, expectedInputType, planner)) {
                return true;
            }
        }
        return false;
    }

    private static PreparedTypeContract resolveTypeContract(Operation op, Tensor node, List<Tensor> inputs) {
        if (op == null || op.opType() == null) {
            DataType outputType = resolveTargetType(node, inputs);
            return uniformTypeContract(outputType, inputs == null ? 0 : inputs.size());
        }
        return switch (op.opType()) {
            case GT, GE, LT, LE, EQ, NE -> resolveCompareContract(inputs);
            case WHERE -> resolveWhereContract(inputs);
            case LOGICAL_AND, LOGICAL_OR -> resolveLogicalBinaryContract(inputs);
            case LOGICAL_NOT -> resolveLogicalUnaryContract(inputs);
            case REDUCE_ALL, REDUCE_ANY -> resolveBoolReductionContract(node, inputs);
            case GATHER -> resolveGatherContract(inputs);
            case GATHER_GRAD -> resolveGatherGradContract(inputs);
            case TAKE_ALONG_AXIS -> resolveTakeAlongAxisContract(inputs);
            case TAKE_ALONG_AXIS_GRAD -> resolveTakeAlongAxisGradContract(inputs);
            case SCATTER_ADD -> resolveScatterAddContract(inputs);
            case FUSED -> resolveFusedContract(op, inputs);
            default -> {
                DataType outputType = resolveTargetType(node, inputs);
                yield uniformTypeContract(outputType, inputs == null ? 0 : inputs.size());
            }
        };
    }

    private static PreparedTypeContract resolveFusedContract(Operation op, List<Tensor> inputs) {
        if (!(op instanceof FusedOperation fused)) {
            throw new IllegalArgumentException("FUSED contract resolution requires FusedOperation descriptor.");
        }
        List<FusedExternalInputPlan> inputPlans = fused.getPlan().inputs();
        if (inputs == null || inputs.size() != inputPlans.size()) {
            throw new IllegalArgumentException("FUSED input contract size mismatch.");
        }
        java.util.List<DataType> expectedInputTypes = new java.util.ArrayList<>(inputPlans.size());
        for (FusedExternalInputPlan inputPlan : inputPlans) {
            expectedInputTypes.add(inputPlan.dataType());
        }
        DataType outputType = fused.getPlan().outputNode().outputType();
        return new PreparedTypeContract(outputType, java.util.List.copyOf(expectedInputTypes));
    }

    private static PreparedTypeContract uniformTypeContract(DataType outputType, int arity) {
        List<DataType> inputTypes = new ArrayList<>(arity);
        for (int i = 0; i < arity; i++) {
            inputTypes.add(outputType);
        }
        return new PreparedTypeContract(outputType, List.copyOf(inputTypes));
    }

    private static PreparedTypeContract resolveCompareContract(List<Tensor> inputs) {
        if (inputs == null || inputs.size() != 2) {
            throw new IllegalArgumentException("Comparison ops expect exactly two inputs.");
        }
        DataType left = inputs.get(0).getDataType();
        DataType right = inputs.get(1).getDataType();
        if (left == DataType.BOOL || right == DataType.BOOL) {
            throw new IllegalArgumentException("Comparison ops require numeric inputs.");
        }
        DataType numericType = promote(left, right);
        return new PreparedTypeContract(DataType.BOOL, List.of(numericType, numericType));
    }

    private static PreparedTypeContract resolveWhereContract(List<Tensor> inputs) {
        if (inputs == null || inputs.size() != 3) {
            throw new IllegalArgumentException("where expects exactly three inputs.");
        }
        if (inputs.get(0).getDataType() != DataType.BOOL) {
            throw new IllegalArgumentException("where condition must have BOOL dtype.");
        }
        DataType trueType = inputs.get(1).getDataType();
        DataType falseType = inputs.get(2).getDataType();
        if (trueType == DataType.BOOL || falseType == DataType.BOOL) {
            throw new IllegalArgumentException("where branches must have numeric dtypes.");
        }
        DataType branchType = promote(trueType, falseType);
        return new PreparedTypeContract(branchType, List.of(DataType.BOOL, branchType, branchType));
    }

    private static PreparedTypeContract resolveLogicalBinaryContract(List<Tensor> inputs) {
        if (inputs == null || inputs.size() != 2) {
            throw new IllegalArgumentException("logical binary ops expect exactly two inputs.");
        }
        if (inputs.get(0).getDataType() != DataType.BOOL || inputs.get(1).getDataType() != DataType.BOOL) {
            throw new IllegalArgumentException("logical binary ops require BOOL inputs.");
        }
        return new PreparedTypeContract(DataType.BOOL, List.of(DataType.BOOL, DataType.BOOL));
    }

    private static PreparedTypeContract resolveLogicalUnaryContract(List<Tensor> inputs) {
        if (inputs == null || inputs.size() != 1) {
            throw new IllegalArgumentException("logicalNot expects exactly one input.");
        }
        if (inputs.get(0).getDataType() != DataType.BOOL) {
            throw new IllegalArgumentException("logicalNot requires BOOL input.");
        }
        return new PreparedTypeContract(DataType.BOOL, List.of(DataType.BOOL));
    }

    private static PreparedTypeContract resolveBoolReductionContract(Tensor node, List<Tensor> inputs) {
        if (inputs == null || inputs.size() != 1) {
            throw new IllegalArgumentException("BOOL reductions expect exactly one input.");
        }
        if (inputs.get(0).getDataType() != DataType.BOOL) {
            throw new IllegalArgumentException("BOOL reductions require BOOL input.");
        }
        return new PreparedTypeContract(DataType.BOOL, List.of(DataType.BOOL));
    }

    private static PreparedTypeContract resolveGatherContract(List<Tensor> inputs) {
        if (inputs == null || inputs.size() != 2) {
            throw new IllegalArgumentException("gather expects exactly two inputs.");
        }
        DataType sourceType = inputs.get(0).getDataType();
        DataType indexType = inputs.get(1).getDataType();
        if (indexType == DataType.BOOL) {
            throw new IllegalArgumentException("gather indices must be numeric integral values.");
        }
        return new PreparedTypeContract(sourceType, List.of(sourceType, indexType));
    }

    private static PreparedTypeContract resolveGatherGradContract(List<Tensor> inputs) {
        if (inputs == null || inputs.size() != 2) {
            throw new IllegalArgumentException("gatherGrad expects exactly two inputs.");
        }
        DataType indexType = inputs.get(0).getDataType();
        DataType gradType = inputs.get(1).getDataType();
        if (indexType == DataType.BOOL || gradType == DataType.BOOL) {
            throw new IllegalArgumentException("gatherGrad requires numeric indices and numeric gradient input.");
        }
        return new PreparedTypeContract(gradType, List.of(indexType, gradType));
    }

    private static PreparedTypeContract resolveScatterAddContract(List<Tensor> inputs) {
        if (inputs == null || inputs.size() != 3) {
            throw new IllegalArgumentException("scatterAdd expects exactly three inputs.");
        }
        DataType baseType = inputs.get(0).getDataType();
        DataType indexType = inputs.get(1).getDataType();
        DataType srcType = inputs.get(2).getDataType();
        if (baseType == DataType.BOOL || srcType == DataType.BOOL || indexType == DataType.BOOL) {
            throw new IllegalArgumentException("scatterAdd requires numeric tensors and numeric indices.");
        }
        if (baseType != srcType) {
            throw new IllegalArgumentException("scatterAdd requires base and src to have matching dtypes.");
        }
        return new PreparedTypeContract(baseType, List.of(baseType, indexType, srcType));
    }

    private static PreparedTypeContract resolveTakeAlongAxisContract(List<Tensor> inputs) {
        if (inputs == null || inputs.size() != 2) {
            throw new IllegalArgumentException("takeAlongAxis expects exactly two inputs.");
        }
        DataType sourceType = inputs.get(0).getDataType();
        DataType indexType = inputs.get(1).getDataType();
        if (indexType == DataType.BOOL) {
            throw new IllegalArgumentException("takeAlongAxis indices must be numeric integral values.");
        }
        return new PreparedTypeContract(sourceType, List.of(sourceType, indexType));
    }

    private static PreparedTypeContract resolveTakeAlongAxisGradContract(List<Tensor> inputs) {
        if (inputs == null || inputs.size() != 2) {
            throw new IllegalArgumentException("takeAlongAxisGrad expects exactly two inputs.");
        }
        DataType indexType = inputs.get(0).getDataType();
        DataType gradType = inputs.get(1).getDataType();
        if (indexType == DataType.BOOL || gradType == DataType.BOOL) {
            throw new IllegalArgumentException("takeAlongAxisGrad requires numeric indices and numeric gradient input.");
        }
        return new PreparedTypeContract(gradType, List.of(indexType, gradType));
    }

    private static DataType resolveTargetType(Tensor node, List<Tensor> inputs) {
        if (node != null && node.getDataType() != null) {
            return node.getDataType();
        }
        if (inputs == null || inputs.isEmpty()) {
            return DataType.FLOAT32;
        }

        DataType target = inputs.get(0).getDataType();
        for (int i = 1; i < inputs.size(); i++) {
            target = promote(target, inputs.get(i).getDataType());
        }
        return target == null ? DataType.FLOAT32 : target;
    }

    private static DataType promote(DataType left, DataType right) {
        if (left == DataType.INT32 || right == DataType.INT32) {
            throw new IllegalArgumentException("INT32 is not supported by generic floating numeric promotion. left=" + left + ", right=" + right);
        }
        if (left == DataType.FLOAT64 || right == DataType.FLOAT64) {
            return DataType.FLOAT64;
        }
        if (left == DataType.FLOAT32 || right == DataType.FLOAT32) {
            return DataType.FLOAT32;
        }
        if (left == DataType.FLOAT16 || right == DataType.FLOAT16) {
            return DataType.FLOAT16;
        }
        throw new IllegalArgumentException("BOOL is not supported by generic numeric promotion. left=" + left + ", right=" + right);
    }

    private static boolean canConvertPreparedInput(DataType sourceType, DataType expectedType) {
        if (sourceType == expectedType) {
            return true;
        }
        if (sourceType == DataType.BOOL || expectedType == DataType.BOOL || sourceType == DataType.INT32 || expectedType == DataType.INT32) {
            return false;
        }
        return true;
    }

    private static Tensor createPreparedTensor(Tensor source, DataType targetType, Tensor node, int inputIndex) {
        String baseLabel = node != null && node.getLabel() != null ? node.getLabel() : "node";
        String label = baseLabel + "_prepared_input_" + inputIndex;
        return new Tensor(source.getShape().clone(), new ArrayList<>(), label, targetType);
    }
}
