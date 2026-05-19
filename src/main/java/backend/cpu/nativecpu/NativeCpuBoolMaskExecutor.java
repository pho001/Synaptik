package backend.cpu.nativecpu;

import backend.cpu.kernels.CpuKernelContext;
import backend.cpu.kernels.elementwise.ElementwiseLoops;
import backend.cpu.kernels.elementwise.logical.LogicalBinaryElementwiseKernel;
import backend.cpu.kernels.elementwise.logical.LogicalUnaryElementwiseKernel;
import backend.cpu.nativecpu.layout.NativeCpuStorageFamily;
import backend.cpu.nativecpu.layout.NativeSegmentStridedKernels;
import backend.cpu.nativecpu.layout.NativeSegmentView;
import backend.cpu.nativecpu.layout.TensorPhysicalView;
import backend.memory.CpuMaterializationReason;
import config.runtime.NativeCpuFailurePolicy;
import operations.Operation;
import operations.reduction.reduceAll;
import operations.reduction.reduceAny;
import tensor.DataType;
import tensor.storage.NativeBoolStorage;
import tensor.storage.NativeTensorStorage;
import tensor.Tensor;

import java.util.List;

/**
 * Standalone native CPU path for dense BOOL mask operations.
 */
public final class NativeCpuBoolMaskExecutor {
    private NativeCpuBoolMaskExecutor() {
    }

    public static boolean tryRunLogicalUnary(
            LogicalUnaryElementwiseKernel kernel,
            List<Tensor> inputs,
            Tensor node,
            CpuKernelContext context
    ) {
        if (!nativeRequested(context)) {
            return false;
        }
        Operation op = context.executionOperation();
        NativeCpuKernelFact fact = NativeCpuKernelFacts.factFor(opType(op), DataType.BOOL);
        if (op == null || op.opType() != Operation.OpType.LOGICAL_NOT
                || inputs == null || inputs.size() != 1
                || inputs.getFirst().getDataType() != DataType.BOOL
                || node.getDataType() != DataType.BOOL
                || context.nodePlan().stridedPath()
                || !denseTensor(inputs.getFirst())
                || !sameFlatSize(inputs.getFirst(), node)) {
            fallbackLogicalUnary(kernel, inputs, node, context, fact, ineligibleReason(op, node, context));
            return true;
        }
        try {
            NativeBoolStorage input = requireBoolNativeInput(context, 0, "LOGICAL_NOT");
            NativeBoolStorage out = allocateBool(node, context, "logical_not");
            NativeSegmentStridedKernels.runUnary(
                    op,
                    denseView(context.inputNodeIds().getFirst(), inputs.getFirst(), input),
                    denseView(context.nodeId(), node, out),
                    false,
                    false
            );
            out.markModified();
            context.executionContext().attachNativeStorage(context.nodeId(), out, "native CPU LOGICAL_NOT wrote BOOL_MASK_NATIVE output");
            publishTrace(context, fact, "CPU_NATIVE", "");
        } catch (Throwable t) {
            fallbackLogicalUnary(kernel, inputs, node, context, fact, "native-kernel-failed:logical_not:" + safeMessage(t));
        }
        return true;
    }

    public static boolean tryRunLogicalBinary(
            LogicalBinaryElementwiseKernel kernel,
            List<Tensor> inputs,
            Tensor node,
            CpuKernelContext context
    ) {
        if (!nativeRequested(context)) {
            return false;
        }
        Operation op = context.executionOperation();
        NativeCpuKernelFact fact = NativeCpuKernelFacts.factFor(opType(op), DataType.BOOL);
        if (op == null || !isLogicalBinary(op.opType())
                || inputs == null || inputs.size() != 2
                || inputs.get(0).getDataType() != DataType.BOOL
                || inputs.get(1).getDataType() != DataType.BOOL
                || node.getDataType() != DataType.BOOL
                || context.nodePlan().stridedPath()
                || !denseTensor(inputs.get(0))
                || !denseTensor(inputs.get(1))
                || !sameFlatSize(inputs.get(0), node)
                || !sameFlatSize(inputs.get(1), node)) {
            fallbackLogicalBinary(kernel, inputs, node, context, fact, ineligibleReason(op, node, context));
            return true;
        }
        try {
            NativeBoolStorage left = requireBoolNativeInput(context, 0, opLabel(op).toUpperCase());
            NativeBoolStorage right = requireBoolNativeInput(context, 1, opLabel(op).toUpperCase());
            NativeBoolStorage out = allocateBool(node, context, opLabel(op));
            NativeSegmentStridedKernels.runBinary(
                    op,
                    denseView(context.inputNodeIds().get(0), inputs.get(0), left),
                    denseView(context.inputNodeIds().get(1), inputs.get(1), right),
                    denseView(context.nodeId(), node, out)
            );
            out.markModified();
            context.executionContext().attachNativeStorage(
                    context.nodeId(),
                    out,
                    "native CPU " + opLabel(op).toUpperCase() + " wrote BOOL_MASK_NATIVE output"
            );
            publishTrace(context, fact, "CPU_NATIVE", "");
        } catch (Throwable t) {
            fallbackLogicalBinary(kernel, inputs, node, context, fact,
                    "native-kernel-failed:" + opLabel(op) + ":" + safeMessage(t));
        }
        return true;
    }

    public static boolean tryRunReduction(Object reduction, Tensor input, Tensor node, CpuKernelContext context) {
        if (!nativeRequested(context)) {
            return false;
        }
        Operation.OpType opType = boolReductionType(reduction);
        NativeCpuKernelFact fact = NativeCpuKernelFacts.factFor(opType, DataType.BOOL);
        int dimension = boolReductionDimension(reduction);
        if (opType == Operation.OpType.UNKNOWN
                || input == null
                || node == null
                || input.getDataType() != DataType.BOOL
                || node.getDataType() != DataType.BOOL
                || context.nodePlan().stridedPath()
                || !denseTensor(input)
                || !validReductionDimension(input, dimension)) {
            return fallbackReduction(context, fact, "native-kernel-ineligible:" + opLabel(opType));
        }
        try {
            NativeBoolStorage nativeInput = requireBoolNativeInput(context, 0, opLabel(opType).toUpperCase());
            NativeBoolStorage out = allocateBool(node, context, opLabel(opType));
            NativeSegmentStridedKernels.runReduction(
                    opType,
                    denseView(context.inputNodeIds().getFirst(), input, nativeInput),
                    denseView(context.nodeId(), node, out),
                    dimension
            );
            out.markModified();
            context.executionContext().attachNativeStorage(
                    context.nodeId(),
                    out,
                    "native CPU " + opLabel(opType).toUpperCase() + " wrote BOOL_MASK_NATIVE output"
            );
            publishTrace(context, fact, "CPU_NATIVE", "");
        } catch (Throwable t) {
            return fallbackReduction(
                    context,
                    fact,
                    "native-kernel-failed:" + opLabel(opType) + ":" + safeMessage(t)
            );
        }
        return true;
    }

    private static void fallbackLogicalUnary(
            LogicalUnaryElementwiseKernel kernel,
            List<Tensor> inputs,
            Tensor node,
            CpuKernelContext context,
            NativeCpuKernelFact fact,
            String reason
    ) {
        handleRequireNative(context, "BOOL logical unary", reason);
        requireCpuReadableInputs(context);
        publishTrace(context, fact, "CPU_ARRAY", reason);
        ElementwiseLoops.runLogicalUnary(kernel, inputs.getFirst(), node, context);
    }

    private static void fallbackLogicalBinary(
            LogicalBinaryElementwiseKernel kernel,
            List<Tensor> inputs,
            Tensor node,
            CpuKernelContext context,
            NativeCpuKernelFact fact,
            String reason
    ) {
        handleRequireNative(context, "BOOL logical binary", reason);
        requireCpuReadableInputs(context);
        publishTrace(context, fact, "CPU_ARRAY", reason);
        ElementwiseLoops.runLogicalBinary(kernel, inputs.get(0), inputs.get(1), node, context);
    }

    private static boolean fallbackReduction(
            CpuKernelContext context,
            NativeCpuKernelFact fact,
            String reason
    ) {
        handleRequireNative(context, "BOOL reduction", reason);
        requireCpuReadableInputs(context);
        publishTrace(context, fact, "CPU_ARRAY", reason);
        return false;
    }

    private static NativeBoolStorage requireBoolNativeInput(CpuKernelContext context, int inputIndex, String op) {
        int inputNodeId = context.inputNodeIds().get(inputIndex);
        NativeTensorStorage storage = context.executionContext().requireNativeReadable(inputNodeId, CpuMaterializationReason.CPU_CONSUMER);
        if (storage instanceof NativeBoolStorage boolStorage) {
            return boolStorage;
        }
        throw new IllegalStateException("native " + op + " requires BOOL native input storage");
    }

    private static NativeBoolStorage allocateBool(Tensor node, CpuKernelContext context, String label) {
        return (NativeBoolStorage) context.executionContext().allocateNativeStorage(
                DataType.BOOL,
                node.getFlatDataSize(),
                "node-" + context.nodeId() + ":" + node.getLabel() + ":native-bool-" + label
        );
    }

    private static NativeSegmentView denseView(int nodeId, Tensor tensor, NativeBoolStorage storage) {
        return NativeSegmentView.from(
                TensorPhysicalView.of(
                        nodeId,
                        DataType.BOOL,
                        tensor.getShapeUnsafe(),
                        denseStrides(tensor.getShapeUnsafe()),
                        0,
                        NativeCpuStorageFamily.CPU_NATIVE
                ),
                storage
        );
    }

    private static int[] denseStrides(int[] shape) {
        int[] strides = new int[shape.length];
        int stride = 1;
        for (int dim = shape.length - 1; dim >= 0; dim--) {
            strides[dim] = stride;
            stride = Math.multiplyExact(stride, shape[dim]);
        }
        return strides;
    }

    private static boolean denseTensor(Tensor tensor) {
        return tensor != null && tensor.isContiguous() && !tensor.hasStorageOffset();
    }

    private static boolean sameFlatSize(Tensor input, Tensor node) {
        return input != null && node != null && input.getFlatDataSize() == node.getFlatDataSize();
    }

    private static boolean validReductionDimension(Tensor input, int dimension) {
        int rank = input.getShapeUnsafe().length;
        return dimension >= -1 && dimension < rank;
    }

    private static Operation.OpType boolReductionType(Object reduction) {
        if (reduction instanceof reduceAll) {
            return Operation.OpType.REDUCE_ALL;
        }
        if (reduction instanceof reduceAny) {
            return Operation.OpType.REDUCE_ANY;
        }
        return Operation.OpType.UNKNOWN;
    }

    private static int boolReductionDimension(Object reduction) {
        if (reduction instanceof reduceAll all) {
            return all.getDimension();
        }
        if (reduction instanceof reduceAny any) {
            return any.getDimension();
        }
        return Integer.MIN_VALUE;
    }

    private static boolean isLogicalBinary(Operation.OpType opType) {
        return opType == Operation.OpType.LOGICAL_AND || opType == Operation.OpType.LOGICAL_OR;
    }

    private static String ineligibleReason(Operation op, Tensor node, CpuKernelContext context) {
        Operation.OpType opType = opType(op);
        if (node == null || node.getDataType() != DataType.BOOL) {
            return "native-storage-dtype-unsupported:bool";
        }
        if (context.nodePlan().stridedPath()) {
            return "native-kernel-ineligible:" + opLabel(opType) + "-strided";
        }
        return "native-kernel-ineligible:" + opLabel(opType);
    }

    private static void handleRequireNative(CpuKernelContext context, String family, String reason) {
        if (context.executionContext().runtimeConfig().nativeCpuFailurePolicy() == NativeCpuFailurePolicy.REQUIRE_NATIVE) {
            throw new IllegalStateException("Native CPU execution required but " + family + " fell back to Java: " + reason);
        }
    }

    private static void requireCpuReadableInputs(CpuKernelContext context) {
        for (int inputNodeId : context.inputNodeIds()) {
            context.executionContext().requireCpuReadable(inputNodeId, CpuMaterializationReason.CPU_CONSUMER);
        }
    }

    private static void publishTrace(CpuKernelContext context, NativeCpuKernelFact fact, String actualCpuStorage, String fallbackReason) {
        var runtime = context.executionContext().runtimeConfig();
        context.putRuntimeState(
                context.executionContext().runtimeTensorForNodeId(context.nodeId()),
                new NativeCpuTraceState(
                        runtime.cpuStorageProfile().name(),
                        runtime.nativeCpuFailurePolicy().name(),
                        "CPU_NATIVE",
                        actualCpuStorage,
                        fact.status().name(),
                        fact.family().name(),
                        fallbackReason
                )
        );
    }

    private static boolean nativeRequested(CpuKernelContext context) {
        return NativeCpuRuntimePolicy.nativeRequested(context);
    }

    private static Operation.OpType opType(Operation op) {
        return op == null ? Operation.OpType.UNKNOWN : op.opType();
    }

    private static String opLabel(Operation op) {
        return opLabel(opType(op));
    }

    private static String opLabel(Operation.OpType opType) {
        return opType == null ? Operation.OpType.UNKNOWN.name().toLowerCase() : opType.name().toLowerCase();
    }

    private static String safeMessage(Throwable t) {
        String message = t.getMessage();
        return message == null || message.isBlank() ? t.getClass().getSimpleName() : message;
    }
}
