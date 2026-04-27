package backend.cpu.kernels.elementwise.strided;

import backend.cpu.kernels.CpuKernelContext;
import operations.Operation;
import tensor.Tensor;

import java.util.List;

public final class CpuStridedElementWise {
    private CpuStridedElementWise() {
    }

    public static boolean supports(Operation op) {
        return StridedElementWiseSemantics.supports(op);
    }

    public static void forward(Operation op, List<Tensor> inputs, Tensor node) {
        forward(op, inputs, node, null);
    }

    public static void forward(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (op == null) {
            return;
        }
        boolean useFastExpApprox = context != null && context.useFastExpApprox();
        boolean useFastTanhApprox = context != null && context.useFastTanhApprox();
        switch (node.getDataType()) {
            case FLOAT64 -> forwardF64(op, inputs, node, useFastExpApprox, useFastTanhApprox);
            case FLOAT32 -> forwardF32(op, inputs, node, useFastExpApprox, useFastTanhApprox);
            case BFLOAT16 -> forwardBF16(op, inputs, node, useFastExpApprox, useFastTanhApprox);
            case BOOL -> StridedBooleanLoops.forward(op, inputs, node);
            case INT32 -> throw new UnsupportedOperationException("INT32 is not supported by CpuStridedElementWise.");
        }
    }

    private static void forwardF64(
            Operation op,
            List<Tensor> inputs,
            Tensor node,
            boolean useFastExpApprox,
            boolean useFastTanhApprox
    ) {
        if (op.opType() == Operation.OpType.WHERE) {
            StridedWhereLoops.forwardF64(inputs, node);
            return;
        }
        StridedNumericInputs.F64 prepared = StridedNumericInputs.prepareF64(inputs, node);
        if (prepared == null) {
            return;
        }
        StridedNumericLoops.forwardF64(op, prepared, useFastExpApprox, useFastTanhApprox);
    }

    private static void forwardF32(
            Operation op,
            List<Tensor> inputs,
            Tensor node,
            boolean useFastExpApprox,
            boolean useFastTanhApprox
    ) {
        if (op.opType() == Operation.OpType.WHERE) {
            StridedWhereLoops.forwardF32(inputs, node, node.getFloat32Data());
            return;
        }
        StridedNumericInputs.F32 prepared = StridedNumericInputs.prepareF32(inputs, node);
        if (prepared == null) {
            return;
        }
        StridedNumericLoops.forwardF32(op, prepared, useFastExpApprox, useFastTanhApprox);
    }

    private static void forwardBF16(
            Operation op,
            List<Tensor> inputs,
            Tensor node,
            boolean useFastExpApprox,
            boolean useFastTanhApprox
    ) {
        if (op.opType() == Operation.OpType.WHERE) {
            StridedWhereLoops.forwardBF16(inputs, node, node.getBFloat16Data());
            return;
        }
        StridedNumericInputs.BF16 prepared = StridedNumericInputs.prepareBF16(inputs, node);
        if (prepared == null) {
            return;
        }
        StridedNumericLoops.forwardBF16(op, prepared, useFastExpApprox, useFastTanhApprox);
    }
}
