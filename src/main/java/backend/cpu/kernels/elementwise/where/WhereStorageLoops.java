package backend.cpu.kernels.elementwise.where;

import tensor.TensorInternalAccess;

import backend.cpu.kernels.CpuDTypeOps;
import backend.cpu.kernels.CpuKernelContext;
import backend.cpu.kernels.elementwise.ElementwiseLoops;
import backend.cpu.kernels.elementwise.ElementwiseNativeSupport;
import backend.cpu.kernels.storage.CpuStorageBindings;
import backend.cpu.kernels.storage.CpuStorageView;
import backend.cpu.nativecpu.NativeCpuKernelFact;
import backend.cpu.nativecpu.NativeCpuKernelFacts;
import operations.Operation;
import tensor.DataType;
import tensor.Tensor;
import tensor.storage.NativeTensorStorage;

import java.lang.foreign.MemorySegment;
import java.util.List;

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

final class WhereStorageLoops {
    private WhereStorageLoops() {
    }

    static void execute(WhereElementwiseKernel kernel, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!ElementwiseNativeSupport.nativeRequested(context)) {
            runArray(kernel, inputs, node, context);
            return;
        }
        Operation op = context.executionOperation();
        Operation.OpType opType = opType(op);
        NativeCpuKernelFact fact = NativeCpuKernelFacts.factFor(opType, node.getDataType());
        String ineligibleReason = nativeIneligibleReason(opType, inputs, node, context);
        if (!ineligibleReason.isBlank()) {
            fallbackToArray(kernel, inputs, node, context, fact, ineligibleReason);
            return;
        }
        try {
            byte[] condition = TensorInternalAccess.boolData(inputs.get(0));
            NativeTensorStorage trueStorage = ElementwiseNativeSupport.requireNativeInput(context, 1, node.getDataType(), "WHERE");
            NativeTensorStorage falseStorage = ElementwiseNativeSupport.requireNativeInput(context, 2, node.getDataType(), "WHERE");
            NativeTensorStorage outputStorage = ElementwiseNativeSupport.allocateNativeOutput(node, context, "where-storage-loop");
            CpuStorageBindings bindings = new CpuStorageBindings(
                    List.of(
                            ElementwiseNativeSupport.segmentView(inputs.get(1), trueStorage),
                            ElementwiseNativeSupport.segmentView(inputs.get(2), falseStorage)
                    ),
                    ElementwiseNativeSupport.segmentView(node, outputStorage)
            );
            runSegmentDense(kernel, condition, bindings);
            outputStorage.markModified();
            context.executionContext().attachNativeStorage(
                    context.nodeId(),
                    outputStorage,
                    "WHERE storage loop wrote " + node.getDataType() + " native output"
            );
            ElementwiseNativeSupport.publishTrace(context, fact, "CPU_NATIVE", "");
        } catch (Throwable t) {
            fallbackToArray(kernel, inputs, node, context, fact,
                    "native-kernel-failed:where:" + ElementwiseNativeSupport.safeMessage(t));
        }
    }

    static void runSegmentDense(WhereElementwiseKernel kernel, byte[] condition, CpuStorageBindings bindings) {
        validateDenseWhere(condition, bindings);
        DataType dtype = bindings.output().dtype();
        switch (dtype) {
            case FLOAT64 -> runSegmentDenseF64(
                    kernel,
                    condition,
                    bindings.input(0).requireSegment(),
                    bindings.input(1).requireSegment(),
                    bindings.output().requireSegment(),
                    bindings.output().logicalSize()
            );
            case FLOAT32 -> runSegmentDenseF32(
                    kernel,
                    condition,
                    bindings.input(0).requireSegment(),
                    bindings.input(1).requireSegment(),
                    bindings.output().requireSegment(),
                    bindings.output().logicalSize()
            );
            case BFLOAT16 -> runSegmentDenseBF16(
                    kernel,
                    condition,
                    bindings.input(0).requireSegment(),
                    bindings.input(1).requireSegment(),
                    bindings.output().requireSegment(),
                    bindings.output().logicalSize()
            );
            case INT32, INT64, BOOL -> throw new UnsupportedOperationException("Where storage loop does not support dtype: " + dtype);
        }
    }

    private static void runSegmentDenseF32(
            WhereElementwiseKernel kernel,
            byte[] condition,
            MemorySegment ifTrue,
            MemorySegment ifFalse,
            MemorySegment output,
            int size
    ) {
        for (int i = 0; i < size; i++) {
            long offset = (long) i * Float.BYTES;
            output.set(JAVA_FLOAT, offset, kernel.applyF32(
                    condition[i],
                    ifTrue.get(JAVA_FLOAT, offset),
                    ifFalse.get(JAVA_FLOAT, offset)
            ));
        }
    }

    private static void runSegmentDenseF64(
            WhereElementwiseKernel kernel,
            byte[] condition,
            MemorySegment ifTrue,
            MemorySegment ifFalse,
            MemorySegment output,
            int size
    ) {
        for (int i = 0; i < size; i++) {
            long offset = (long) i * Double.BYTES;
            output.set(JAVA_DOUBLE, offset, kernel.applyF64(
                    condition[i],
                    ifTrue.get(JAVA_DOUBLE, offset),
                    ifFalse.get(JAVA_DOUBLE, offset)
            ));
        }
    }

    private static void runSegmentDenseBF16(
            WhereElementwiseKernel kernel,
            byte[] condition,
            MemorySegment ifTrue,
            MemorySegment ifFalse,
            MemorySegment output,
            int size
    ) {
        for (int i = 0; i < size; i++) {
            long offset = (long) i * Short.BYTES;
            float trueValue = CpuDTypeOps.fromBFloat16Bits(ifTrue.get(JAVA_SHORT, offset));
            float falseValue = CpuDTypeOps.fromBFloat16Bits(ifFalse.get(JAVA_SHORT, offset));
            output.set(JAVA_SHORT, offset, CpuDTypeOps.toBFloat16Bits(kernel.applyBF16(condition[i], trueValue, falseValue)));
        }
    }

    private static void fallbackToArray(
            WhereElementwiseKernel kernel,
            List<Tensor> inputs,
            Tensor node,
            CpuKernelContext context,
            NativeCpuKernelFact fact,
            String reason
    ) {
        ElementwiseNativeSupport.requireFallbackAllowed(context, "where elementwise", reason);
        ElementwiseNativeSupport.requireCpuReadableInputs(context);
        ElementwiseNativeSupport.publishTrace(context, fact, "CPU_ARRAY", reason);
        runArray(kernel, inputs, node, context);
    }

    private static void runArray(WhereElementwiseKernel kernel, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        switch (node.getDataType()) {
            case FLOAT64, FLOAT32, BFLOAT16 -> ElementwiseLoops.runWhere(kernel, inputs.get(0), inputs.get(1), inputs.get(2), node, context);
            case INT32, INT64, BOOL -> throw new UnsupportedOperationException("Where only supports floating output tensors");
        }
    }

    private static String nativeIneligibleReason(
            Operation.OpType opType,
            List<Tensor> inputs,
            Tensor node,
            CpuKernelContext context
    ) {
        if (context == null || context.nodePlan() == null) {
            return "native-kernel-ineligible:where-plan";
        }
        if (opType != Operation.OpType.WHERE) {
            return "native-kernel-unsupported:" + opLabel(opType);
        }
        if (context.nodePlan().stridedPath()) {
            return "native-kernel-ineligible:where-strided";
        }
        if (!supportsNativeWhereDType(node.getDataType())) {
            return "native-storage-dtype-unsupported:" + node.getDataType().name().toLowerCase();
        }
        if (inputs == null || inputs.size() != 3) {
            return "native-kernel-ineligible:where-input-count";
        }
        DataType branchDataType = node.getDataType();
        if (inputs.get(0).getDataType() != DataType.BOOL
                || inputs.get(1).getDataType() != branchDataType
                || inputs.get(2).getDataType() != branchDataType) {
            return "native-kernel-ineligible:where-dtype";
        }
        if (context.whereBroadcastPlan() != null && !context.whereBroadcastPlan().isNoBroadcast()) {
            return "native-kernel-ineligible:where-broadcast";
        }
        int size = node.getFlatDataSize();
        if (inputs.get(0).getFlatDataSize() != size
                || inputs.get(1).getFlatDataSize() != size
                || inputs.get(2).getFlatDataSize() != size) {
            return "native-kernel-ineligible:where-shape";
        }
        if (!ElementwiseNativeSupport.isDenseView(inputs.get(0))
                || !ElementwiseNativeSupport.isDenseView(inputs.get(1))
                || !ElementwiseNativeSupport.isDenseView(inputs.get(2))
                || !ElementwiseNativeSupport.isDenseView(node)) {
            return "native-kernel-ineligible:where-layout";
        }
        return "";
    }

    private static void validateDenseWhere(byte[] condition, CpuStorageBindings bindings) {
        if (bindings.inputs().size() != 2) {
            throw new IllegalArgumentException("Where storage loop requires exactly 2 branch inputs.");
        }
        CpuStorageView ifTrue = bindings.input(0);
        CpuStorageView ifFalse = bindings.input(1);
        CpuStorageView output = bindings.output();
        if (ifTrue.dtype() != output.dtype() || ifFalse.dtype() != output.dtype()) {
            throw new IllegalArgumentException("Where storage loop dtype mismatch.");
        }
        if (ifTrue.kind() != output.kind() || ifFalse.kind() != output.kind()) {
            throw new IllegalArgumentException("Where storage loop requires matching branch/output storage kinds.");
        }
        if (condition.length < output.logicalSize()
                || ifTrue.logicalSize() != output.logicalSize()
                || ifFalse.logicalSize() != output.logicalSize()) {
            throw new IllegalArgumentException("Where storage loop requires same-shape dense inputs.");
        }
        if (!ElementwiseNativeSupport.isDenseView(ifTrue)
                || !ElementwiseNativeSupport.isDenseView(ifFalse)
                || !ElementwiseNativeSupport.isDenseView(output)) {
            throw new IllegalArgumentException("Where storage loop requires dense zero-offset branch/output views.");
        }
    }

    private static boolean supportsNativeWhereDType(DataType dataType) {
        return dataType == DataType.FLOAT32 || dataType == DataType.FLOAT64 || dataType == DataType.BFLOAT16;
    }

    private static Operation.OpType opType(Operation op) {
        return op == null ? Operation.OpType.UNKNOWN : op.opType();
    }

    private static String opLabel(Operation.OpType opType) {
        return opType.name().toLowerCase();
    }
}
