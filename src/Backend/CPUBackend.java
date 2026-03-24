package Backend;

import Config.backend.CpuKernelConfig;
import Backend.kernels.cpu.CpuKernel;
import Backend.kernels.cpu.CpuExecutionConfig;
import Backend.kernels.cpu.CpuStridedElementWise;
import Backend.registry.CpuKernelRegistry;
import Tensor.DataType;
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

public class CPUBackend {
    private CpuExecutionConfig executionConfig = CpuExecutionConfig.defaults();


    public void execute(Operation op, List<Tensor> inputs,Tensor node) {
        if (op == null) {
            return;
        }
        CpuKernel kernel = node.getResolvedCpuKernel();
        if (kernel == null) {
            kernel = CpuKernelRegistry.resolve(op.opType());
        }
        if (kernel == null) {
            throw new UnsupportedOperationException(
                    "Missing CPU kernel for opType=" + op.opType() +
                            " (operation class: " + op.getClass().getName() + ")"
            );
        }

        if (canUseStridedPath(op, inputs, node)) {
            CpuStridedElementWise.forward(op, inputs, node);
            return;
        }

        DataType dataType = node.getDataType();
        if (dataType == null) {
            dataType = DataType.FLOAT32;
        }
        List<Tensor> preparedInputs = prepareInputs(op, inputs, dataType);
        switch (dataType) {
            case FLOAT64 -> kernel.forwardF64(op, preparedInputs, node, executionConfig);
            case FLOAT32 -> kernel.forwardF32(op, preparedInputs, node, executionConfig);
            case FLOAT16 -> kernel.forwardF16(op, preparedInputs, node, executionConfig);
        }
        if (dataType != DataType.FLOAT64) {
            node.markDataViewStale();
        }
    }

    private boolean canUseStridedPath(Operation op, List<Tensor> inputs, Tensor node) {
        if (op == null || node == null || inputs == null || inputs.isEmpty()) {
            return false;
        }
        if (op.opType() == Operation.OpType.CONTIGUOUS) {
            return false;
        }
        if (!op.isElementWise() || !CpuStridedElementWise.supports(op)) {
            return false;
        }

        DataType targetType = node.getDataType() == null ? DataType.FLOAT32 : node.getDataType();
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

        int threshold = Math.max(0, executionConfig.contiguousMaterializeThreshold());
        int size = node.getFlatDataSize();
        return size < threshold;
    }

    private List<Tensor> prepareInputs(Operation op, List<Tensor> inputs, DataType targetType) {
        if (inputs == null || inputs.isEmpty()) {
            return inputs;
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
            return inputs;
        }

        List<Tensor> prepared = null;
        int materializeThreshold = Math.max(0, executionConfig.contiguousMaterializeThreshold());
        for (int i = 0; i < inputs.size(); i++) {
            Tensor input = inputs.get(i);
            boolean needsMaterialization = input != null
                    && !input.isContiguous()
                    && !preserveBroadcastStrides;
            boolean needsTypeConversion = input != null
                    && targetType != null
                    && input.getDataType() != targetType;

            if (!needsMaterialization && !needsTypeConversion) {
                if (prepared != null) {
                    prepared.add(input);
                }
                continue;
            }

            if (prepared == null) {
                prepared = new ArrayList<>(inputs.size());
                for (int j = 0; j < i; j++) {
                    prepared.add(inputs.get(j));
                }
            }

            String tmpSuffix = needsMaterialization ? "_contiguous_tmp" : "_dtype_tmp";
            DataType tmpType = needsTypeConversion ? targetType : input.getDataType();
            Tensor remappedInput;
            if (preserveBroadcastStrides && input != null && !input.isContiguous()) {
                int size = input.getFlatDataSize();
                remappedInput = switch (tmpType) {
                    case FLOAT64 -> new Tensor(new double[size], input.getShape(), input.getStrides(), null, input.getLabel() + tmpSuffix, tmpType);
                    case FLOAT32 -> new Tensor(new float[size], input.getShape(), input.getStrides(), null, input.getLabel() + tmpSuffix, tmpType);
                    case FLOAT16 -> new Tensor(new short[size], input.getShape(), input.getStrides(), null, input.getLabel() + tmpSuffix, tmpType);
                };
            } else {
                remappedInput = new Tensor(input.getShape(), null, input.getLabel() + tmpSuffix, tmpType);
            }
            TensorRemap.apply(input, remappedInput, materializeThreshold);
            prepared.add(remappedInput);
        }

        return prepared == null ? inputs : prepared;
    }

    private static boolean isBroadcastOpWithPlan(Operation op) {
        if (op == null) {
            return false;
        }
        return (op instanceof add a && a.getBroadcastPlan() != null && !a.getBroadcastPlan().isNoBroadcast())
                || (op instanceof sub s && s.getBroadcastPlan() != null && !s.getBroadcastPlan().isNoBroadcast())
                || (op instanceof mul m && m.getBroadcastPlan() != null && !m.getBroadcastPlan().isNoBroadcast())
                || (op instanceof div d && d.getBroadcastPlan() != null && !d.getBroadcastPlan().isNoBroadcast())
                || (op instanceof min mi && mi.getBroadcastPlan() != null && !mi.getBroadcastPlan().isNoBroadcast())
                || (op instanceof max ma && ma.getBroadcastPlan() != null && !ma.getBroadcastPlan().isNoBroadcast());
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
