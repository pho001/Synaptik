package Backend;

import Config.backend.CpuKernelConfig;
import Backend.kernels.cpu.CpuKernel;
import Backend.kernels.cpu.CpuExecutionConfig;
import Backend.kernels.cpu.CpuStridedElementWise;
import Backend.kernels.cpu.CpuExecutionMode;
import Backend.kernels.cpu.ResolvedBroadcastPlan;
import Backend.registry.CpuKernelRegistry;
import Graph.codegen.FusedVectorOps;
import Operations.FusedOperation;
import Tensor.DataType;
import Tensor.BroadcastPlan;
import Tensor.Tensor;
import Tensor.TensorRemap;
import Operations.Operation;
import Operations.add;
import Operations.div;
import Operations.max;
import Operations.min;
import Operations.mul;
import Operations.sub;

import java.util.ArrayList;
import java.util.List;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;

public class CPUBackend {
    private CpuExecutionConfig executionConfig = CpuExecutionConfig.defaults();
    private static final boolean DISABLE_PRE_RESOLVED_EXECUTION_PLAN =
            Boolean.getBoolean("cg.cpu.disablePreResolvedExecutionPlan");

    public static final class CpuPreparedInput {
        private final int inputIndex;
        private final Tensor tensor;
        private final TensorRemap.RemapPlan remapPlan;

        private CpuPreparedInput(int inputIndex, Tensor tensor, TensorRemap.RemapPlan remapPlan) {
            this.inputIndex = inputIndex;
            this.tensor = tensor;
            this.remapPlan = remapPlan;
        }
    }

    public static final class CpuNodeExecutionPlan {
        private final boolean stridedPath;
        private final DataType targetType;
        private final List<CpuPreparedInput> remapInputs;
        private final List<Tensor> runtimeInputs;
        private final int materializeThreshold;
        private final ResolvedBroadcastPlan broadcastPlan;
        private final CpuExecutionConfig.ResolvedDispatchHints dispatchHints;

        private CpuNodeExecutionPlan(
                boolean stridedPath,
                DataType targetType,
                List<CpuPreparedInput> remapInputs,
                List<Tensor> runtimeInputs,
                int materializeThreshold,
                ResolvedBroadcastPlan broadcastPlan,
                CpuExecutionConfig.ResolvedDispatchHints dispatchHints
        ) {
            this.stridedPath = stridedPath;
            this.targetType = targetType;
            this.remapInputs = remapInputs;
            this.runtimeInputs = runtimeInputs;
            this.materializeThreshold = materializeThreshold;
            this.broadcastPlan = broadcastPlan;
            this.dispatchHints = dispatchHints;
        }

        public boolean stridedPath() {
            return stridedPath;
        }

        public DataType targetType() {
            return targetType;
        }

        public ResolvedBroadcastPlan broadcastPlan() {
            return broadcastPlan;
        }

        public CpuExecutionConfig.ResolvedDispatchHints dispatchHints() {
            return dispatchHints;
        }

        public List<Tensor> apply(List<Tensor> originalInputs) {
            if (runtimeInputs == null || runtimeInputs.isEmpty()) {
                return originalInputs;
            }
            if (remapInputs == null || remapInputs.isEmpty()) {
                return runtimeInputs;
            }
            for (CpuPreparedInput preparedInput : remapInputs) {
                Tensor source = originalInputs.get(preparedInput.inputIndex);
                TensorRemap.apply(source, preparedInput.tensor, preparedInput.remapPlan, materializeThreshold);
            }
            return runtimeInputs;
        }
    }

    public void execute(Operation op, List<Tensor> inputs,Tensor node) {
        if (op == null) {
            return;
        }
        long cpuConfigEpoch = ComputeEngine.getCpuConfigEpoch();
        CpuKernel kernel = node.getResolvedCpuKernel();
        if (kernel == null) {
            kernel = CpuKernelRegistry.resolve(op.opType());
            node.setResolvedCpuKernel(kernel);
        }
        if (kernel == null) {
            throw new UnsupportedOperationException(
                    "Missing CPU kernel for opType=" + op.opType() +
                            " (operation class: " + op.getClass().getName() + ")"
            );
        }

        CpuNodeExecutionPlan executionPlan = null;
        if (!DISABLE_PRE_RESOLVED_EXECUTION_PLAN) {
            if (node.getResolvedCpuConfigEpoch() == cpuConfigEpoch) {
                executionPlan = node.getResolvedCpuExecutionPlan();
            }
        }
        if (executionPlan == null) {
            executionPlan = buildExecutionPlan(op, inputs, node, executionConfig);
            if (!DISABLE_PRE_RESOLVED_EXECUTION_PLAN) {
                node.setResolvedCpuExecutionPlan(executionPlan);
                node.setResolvedCpuConfigEpoch(cpuConfigEpoch);
            }
        }
        if (executionPlan != null) {
            node.setResolvedBroadcastPlan(executionPlan.broadcastPlan());
        }

        if (executionPlan != null && executionPlan.stridedPath()) {
            CpuStridedElementWise.forward(op, inputs, node);
            return;
        }

        DataType dataType = executionPlan != null ? executionPlan.targetType() : node.getDataType();
        if (dataType == null) dataType = DataType.FLOAT32;
        List<Tensor> preparedInputs = executionPlan != null ? executionPlan.apply(inputs) : inputs;
        CpuExecutionConfig.pushResolvedHints(executionPlan != null ? executionPlan.dispatchHints() : null);
        try {
            switch (dataType) {
                case FLOAT64 -> kernel.forwardF64(op, preparedInputs, node, executionConfig);
                case FLOAT32 -> kernel.forwardF32(op, preparedInputs, node, executionConfig);
                case FLOAT16 -> kernel.forwardF16(op, preparedInputs, node, executionConfig);
            }
        } finally {
            CpuExecutionConfig.clearResolvedHints();
        }
        if (dataType != DataType.FLOAT64) {
            node.markDataViewStale();
        }
    }

    public static CpuNodeExecutionPlan buildExecutionPlan(
            Operation op,
            List<Tensor> inputs,
            Tensor node,
            CpuExecutionConfig config
    ) {
        DataType targetType = node.getDataType();
        if (targetType == null) {
            targetType = DataType.FLOAT32;
        }
        int materializeThreshold = Math.max(0, config.contiguousMaterializeThreshold());
        ResolvedBroadcastPlan resolvedBroadcastPlan = resolveBroadcastPlan(op);
        CpuExecutionConfig.ResolvedDispatchHints dispatchHints = resolveDispatchHints(op, node, targetType, config);

        if (canUseStridedPath(op, inputs, node, targetType, materializeThreshold)) {
            return new CpuNodeExecutionPlan(
                    true,
                    targetType,
                    List.of(),
                    List.of(),
                    materializeThreshold,
                    resolvedBroadcastPlan,
                    dispatchHints
            );
        }
        PreparedInputs prepared = prepareInputs(op, inputs, targetType, materializeThreshold);
        if (prepared == null) {
            return new CpuNodeExecutionPlan(
                    false,
                    targetType,
                    List.of(),
                    List.of(),
                    materializeThreshold,
                    resolvedBroadcastPlan,
                    dispatchHints
            );
        }
        List<Tensor> runtimeInputs = new ArrayList<>(prepared.runtimeInputs.size());
        for (CpuPreparedInput p : prepared.runtimeInputs) {
            runtimeInputs.add(p.tensor);
        }
        return new CpuNodeExecutionPlan(
                false,
                targetType,
                prepared.remapInputs,
                runtimeInputs,
                materializeThreshold,
                resolvedBroadcastPlan,
                dispatchHints
        );
    }

    private static final class PreparedInputs {
        private final List<CpuPreparedInput> runtimeInputs;
        private final List<CpuPreparedInput> remapInputs;

        private PreparedInputs(List<CpuPreparedInput> runtimeInputs, List<CpuPreparedInput> remapInputs) {
            this.runtimeInputs = runtimeInputs;
            this.remapInputs = remapInputs;
        }
    }

    private static boolean canUseStridedPath(
            Operation op,
            List<Tensor> inputs,
            Tensor node,
            DataType targetType,
            int materializeThreshold
    ) {
        if (op == null || node == null || inputs == null || inputs.isEmpty()) {
            return false;
        }
        if (op.opType() == Operation.OpType.CONTIGUOUS) {
            return false;
        }
        if (!op.isElementWise() || !CpuStridedElementWise.supports(op)) {
            return false;
        }

        boolean hasNonContiguousInput = false;
        int[] outShape = node.getShape();
        for (Tensor input : inputs) {
            if (input == null) {
                return false;
            }
            int[] inShape = input.getShape();
            if (inShape.length != outShape.length) {
                return false;
            }
            for (int d = 0; d < outShape.length; d++) {
                if (inShape[d] != outShape[d]) {
                    return false;
                }
            }
            if (input.getDataType() != targetType) {
                return false;
            }
            if (targetType == DataType.FLOAT32 && input.getFloat32Data() == null) {
                return false;
            }
            if (targetType == DataType.FLOAT16 && input.getFloat16Data() == null) {
                return false;
            }
            if (!input.isContiguous()) {
                hasNonContiguousInput = true;
            }
        }
        if (!hasNonContiguousInput) {
            return false;
        }

        int size = node.getFlatDataSize();
        return size < materializeThreshold;
    }

    private static PreparedInputs prepareInputs(
            Operation op,
            List<Tensor> inputs,
            DataType targetType,
            int materializeThreshold
    ) {
        if (inputs == null || inputs.isEmpty()) {
            return null;
        }
        boolean preserveBroadcastStrides = isBroadcastOpWithPlan(op);
        // Layout/reduction kernels are responsible for their own input layout strategy.
        if (op != null && (
                op.opType() == Operation.OpType.CONTIGUOUS
                        || op.opType() == Operation.OpType.SUM
                        || op.opType() == Operation.OpType.RESHAPE
                        || op.opType() == Operation.OpType.PERMUTE
                        || op.opType() == Operation.OpType.EXPAND_DIMS
                        || op.opType() == Operation.OpType.SQUEEZE
        )) {
            return null;
        }

        List<CpuPreparedInput> runtimeInputs = null;
        List<CpuPreparedInput> remapInputs = null;
        for (int i = 0; i < inputs.size(); i++) {
            Tensor input = inputs.get(i);
            boolean needsMaterialization = input != null
                    && !input.isContiguous()
                    && !preserveBroadcastStrides;
            boolean needsTypeConversion = input != null
                    && targetType != null
                    && input.getDataType() != targetType;

            if (!needsMaterialization && !needsTypeConversion) {
                if (runtimeInputs != null) {
                    runtimeInputs.add(new CpuPreparedInput(i, input, null));
                }
                continue;
            }

            if (runtimeInputs == null) {
                runtimeInputs = new ArrayList<>(inputs.size());
                for (int j = 0; j < i; j++) {
                    runtimeInputs.add(new CpuPreparedInput(j, inputs.get(j), null));
                }
                remapInputs = new ArrayList<>();
            }

            DataType tmpType = needsTypeConversion ? targetType : input.getDataType();
            Tensor remappedInput;
            if (preserveBroadcastStrides && input != null && !input.isContiguous()) {
                int size = input.getFlatDataSize();
                remappedInput = switch (tmpType) {
                    case FLOAT64 -> new Tensor(new double[size], input.getShape(), input.getStrides(), null, "_tmp", tmpType);
                    case FLOAT32 -> new Tensor(new float[size], input.getShape(), input.getStrides(), null, "_tmp", tmpType);
                    case FLOAT16 -> new Tensor(new short[size], input.getShape(), input.getStrides(), null, "_tmp", tmpType);
                };
            } else {
                remappedInput = new Tensor(input.getShape(), null, "_tmp", tmpType);
            }
            TensorRemap.RemapPlan remapPlan = TensorRemap.buildPlan(input, remappedInput);
            CpuPreparedInput preparedInput = new CpuPreparedInput(i, remappedInput, remapPlan);
            runtimeInputs.add(preparedInput);
            remapInputs.add(preparedInput);
        }

        if (runtimeInputs == null) {
            return null;
        }
        return new PreparedInputs(runtimeInputs, remapInputs);
    }

    private static boolean isBroadcastOpWithPlan(Operation op) {
        BroadcastPlan plan = extractBroadcastPlan(op);
        return plan != null && !plan.isNoBroadcast();
    }

    private static BroadcastPlan extractBroadcastPlan(Operation op) {
        if (op == null) {
            return null;
        }
        if (op instanceof add a) return a.getBroadcastPlan();
        if (op instanceof sub s) return s.getBroadcastPlan();
        if (op instanceof mul m) return m.getBroadcastPlan();
        if (op instanceof div d) return d.getBroadcastPlan();
        if (op instanceof min mi) return mi.getBroadcastPlan();
        if (op instanceof max ma) return ma.getBroadcastPlan();
        return null;
    }

    private static ResolvedBroadcastPlan resolveBroadcastPlan(Operation op) {
        return ResolvedBroadcastPlan.from(extractBroadcastPlan(op));
    }

    private static CpuExecutionConfig.ResolvedDispatchHints resolveDispatchHints(
            Operation op,
            Tensor node,
            DataType targetType,
            CpuExecutionConfig config
    ) {
        if (op == null || node == null || config == null) {
            return null;
        }
        int totalLength = Math.max(1, node.getFlatDataSize());
        CpuExecutionMode mode = config.modeFor(op, node);
        int scalarChunkSize = config.computeChunkSize(totalLength, 1);
        int vectorWidth = resolveVectorWidth(op, targetType);
        int vectorChunkSize = config.computeChunkSize(totalLength, Math.max(1, vectorWidth));
        return new CpuExecutionConfig.ResolvedDispatchHints(totalLength, mode, scalarChunkSize, vectorChunkSize);
    }

    private static int resolveVectorWidth(Operation op, DataType targetType) {
        if (op instanceof FusedOperation fused) {
            return Math.max(1, FusedVectorOps.width(fused.getPrecisionMode()));
        }
        if (targetType == null) {
            return 1;
        }
        return switch (targetType) {
            case FLOAT64 -> Math.max(1, DoubleVector.SPECIES_PREFERRED.length());
            case FLOAT32, FLOAT16 -> Math.max(1, FloatVector.SPECIES_PREFERRED.length());
        };
    }

    public void setExecutionConfig(CpuExecutionConfig executionConfig) {
        if (executionConfig == null) {
            throw new IllegalArgumentException("executionConfig cannot be null");
        }
        this.executionConfig = executionConfig;
    }

    public CpuExecutionConfig getExecutionConfig() {
        return executionConfig;
    }

    public void setKernelConfig(CpuKernelConfig cpuKernelConfig) {
        if (cpuKernelConfig == null) {
            throw new IllegalArgumentException("cpuKernelConfig cannot be null");
        }
        this.executionConfig = CpuExecutionConfig.fromKernelConfig(cpuKernelConfig);
    }


}
