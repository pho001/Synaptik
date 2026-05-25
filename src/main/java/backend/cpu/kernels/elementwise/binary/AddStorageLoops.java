package backend.cpu.kernels.elementwise.binary;

import backend.cpu.kernels.CpuDTypeOps;
import backend.cpu.kernels.CpuKernelContext;
import backend.cpu.kernels.elementwise.ElementwiseLoops;
import backend.cpu.kernels.elementwise.binary.bf16.AddBF16;
import backend.cpu.kernels.elementwise.binary.f32.AddF32;
import backend.cpu.kernels.elementwise.binary.f64.AddF64;
import backend.cpu.kernels.elementwise.plan.ResolvedDispatchHints;
import backend.cpu.kernels.layout.plan.ResolvedBroadcastPlan;
import backend.cpu.kernels.storage.CpuStorageBindings;
import backend.cpu.kernels.storage.CpuStorageView;
import backend.memory.CpuMaterializationReason;
import backend.cpu.nativecpu.NativeCpuKernelFact;
import backend.cpu.nativecpu.NativeCpuKernelFacts;
import backend.cpu.nativecpu.NativeCpuTraceState;
import config.runtime.CpuStorageProfile;
import config.runtime.NativeCpuFailurePolicy;
import operations.Operation;
import tensor.DataType;
import tensor.Tensor;
import tensor.storage.NativeTensorStorage;

import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.List;

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

final class AddStorageLoops {
    private AddStorageLoops() {
    }

    static void execute(CpuAddKernel kernel, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (!nativeRequested(context)) {
            ElementwiseLoops.runBinary(kernel, inputs.get(0), inputs.get(1), node, context);
            return;
        }
        NativeCpuKernelFact fact = NativeCpuKernelFacts.factFor(Operation.OpType.ADD, node.getDataType());
        BiasBroadcastSpec biasSpec = biasBroadcastSpec(
                context.broadcastPlan(),
                inputs.get(0).getFlatDataSize(),
                inputs.get(1).getFlatDataSize(),
                node.getFlatDataSize()
        );
        String ineligibleReason = nativeIneligibleReason(inputs, node, context, biasSpec);
        if (!ineligibleReason.isBlank()) {
            fallbackToArray(kernel, inputs, node, context, fact, ineligibleReason);
            return;
        }
        try {
            NativeTensorStorage leftStorage = requireNativeInput(context, 0, node.getDataType());
            NativeTensorStorage rightStorage = requireNativeInput(context, 1, node.getDataType());
            NativeTensorStorage outputStorage = context.executionContext().allocateNativeStorage(
                    node.getDataType(),
                    node.getFlatDataSize(),
                    "node-" + context.nodeId() + ":" + node.getLabel() + ":add-storage-loop"
            );
            if (biasSpec == null) {
                CpuStorageBindings bindings = new CpuStorageBindings(
                        List.of(segmentView(inputs.get(0), leftStorage), segmentView(inputs.get(1), rightStorage)),
                        segmentView(node, outputStorage)
                );
                runSegmentDense(bindings);
            } else {
                runLastDimBiasAddF32(leftStorage.segment(), rightStorage.segment(), outputStorage.segment(), node.getFlatDataSize(), biasSpec);
            }
            outputStorage.markModified();
            context.executionContext().attachNativeStorage(
                    context.nodeId(),
                    outputStorage,
                    "ADD storage loop wrote " + node.getDataType() + " native output"
            );
            publishTrace(context, fact, "CPU_NATIVE", "");
        } catch (Throwable t) {
            fallbackToArray(kernel, inputs, node, context, fact, "native-kernel-failed:add:" + safeMessage(t));
        }
    }

    static void runArrayDense(
            CpuStorageBindings bindings,
            ResolvedDispatchHints hints,
            float[] leftContinuation,
            float[] rightContinuation
    ) {
        validateDenseAdd(bindings);
        DataType dtype = bindings.output().dtype();
        switch (dtype) {
            case FLOAT64 -> AddF64.run(
                    bindings.input(0).requireF64Array(),
                    bindings.input(1).requireF64Array(),
                    bindings.output().requireF64Array(),
                    hints
            );
            case FLOAT32 -> AddF32.run(
                    bindings.input(0).requireF32Array(),
                    bindings.input(1).requireF32Array(),
                    bindings.output().requireF32Array(),
                    hints
            );
            case BFLOAT16 -> runArrayDenseBF16(bindings, hints, leftContinuation, rightContinuation);
            case INT32, INT64, BOOL -> throw new UnsupportedOperationException("ADD storage loop does not support dtype: " + dtype);
        }
    }

    static void runSegmentDense(CpuStorageBindings bindings) {
        validateDenseAdd(bindings);
        DataType dtype = bindings.output().dtype();
        switch (dtype) {
            case FLOAT64 -> runSegmentDenseF64(
                    bindings.input(0).requireSegment(),
                    bindings.input(1).requireSegment(),
                    bindings.output().requireSegment(),
                    bindings.output().logicalSize()
            );
            case FLOAT32 -> runSegmentDenseF32(
                    bindings.input(0).requireSegment(),
                    bindings.input(1).requireSegment(),
                    bindings.output().requireSegment(),
                    bindings.output().logicalSize()
            );
            case BFLOAT16 -> runSegmentDenseBF16(
                    bindings.input(0).requireSegment(),
                    bindings.input(1).requireSegment(),
                    bindings.output().requireSegment(),
                    bindings.output().logicalSize()
            );
            case INT32, INT64, BOOL -> throw new UnsupportedOperationException("ADD storage loop does not support dtype: " + dtype);
        }
    }

    private static void runArrayDenseBF16(
            CpuStorageBindings bindings,
            ResolvedDispatchHints hints,
            float[] leftContinuation,
            float[] rightContinuation
    ) {
        short[] left = bindings.input(0).requireBF16Array();
        short[] right = bindings.input(1).requireBF16Array();
        short[] output = bindings.output().requireBF16Array();
        if (leftContinuation != null && rightContinuation != null) {
            AddBF16.run(leftContinuation, rightContinuation, output, hints);
        } else if (leftContinuation != null) {
            AddBF16.run(leftContinuation, right, output, hints);
        } else if (rightContinuation != null) {
            AddBF16.run(left, rightContinuation, output, hints);
        } else {
            AddBF16.run(left, right, output, hints);
        }
    }

    private static void runSegmentDenseF32(MemorySegment left, MemorySegment right, MemorySegment output, int size) {
        for (int i = 0; i < size; i++) {
            long offset = (long) i * Float.BYTES;
            output.set(JAVA_FLOAT, offset, left.get(JAVA_FLOAT, offset) + right.get(JAVA_FLOAT, offset));
        }
    }

    private static void runSegmentDenseF64(MemorySegment left, MemorySegment right, MemorySegment output, int size) {
        for (int i = 0; i < size; i++) {
            long offset = (long) i * Double.BYTES;
            output.set(JAVA_DOUBLE, offset, left.get(JAVA_DOUBLE, offset) + right.get(JAVA_DOUBLE, offset));
        }
    }

    private static void runSegmentDenseBF16(MemorySegment left, MemorySegment right, MemorySegment output, int size) {
        for (int i = 0; i < size; i++) {
            long offset = (long) i * Short.BYTES;
            float leftValue = CpuDTypeOps.fromBFloat16Bits(left.get(JAVA_SHORT, offset));
            float rightValue = CpuDTypeOps.fromBFloat16Bits(right.get(JAVA_SHORT, offset));
            output.set(JAVA_SHORT, offset, CpuDTypeOps.toBFloat16Bits(leftValue + rightValue));
        }
    }

    private static void runLastDimBiasAddF32(
            MemorySegment left,
            MemorySegment right,
            MemorySegment output,
            int size,
            BiasBroadcastSpec spec
    ) {
        for (int i = 0; i < size; i++) {
            long outputOffset = (long) i * Float.BYTES;
            long biasOffset = (long) (i % spec.lastDim()) * Float.BYTES;
            float leftValue = left.get(JAVA_FLOAT, spec.leftBias() ? biasOffset : outputOffset);
            float rightValue = right.get(JAVA_FLOAT, spec.leftBias() ? outputOffset : biasOffset);
            output.set(JAVA_FLOAT, outputOffset, leftValue + rightValue);
        }
    }

    private static void fallbackToArray(
            CpuAddKernel kernel,
            List<Tensor> inputs,
            Tensor node,
            CpuKernelContext context,
            NativeCpuKernelFact fact,
            String reason
    ) {
        if (context.executionContext().runtimeConfig().nativeCpuFailurePolicy() == NativeCpuFailurePolicy.REQUIRE_NATIVE) {
            throw new IllegalStateException("Native CPU execution required but ADD fell back to Java: " + reason);
        }
        for (int inputNodeId : context.inputNodeIds()) {
            context.executionContext().requireCpuReadable(inputNodeId, CpuMaterializationReason.CPU_CONSUMER);
        }
        publishTrace(context, fact, "CPU_ARRAY", reason);
        ElementwiseLoops.runBinary(kernel, inputs.get(0), inputs.get(1), node, context);
    }

    private static String nativeIneligibleReason(
            List<Tensor> inputs,
            Tensor node,
            CpuKernelContext context,
            BiasBroadcastSpec biasSpec
    ) {
        if (context == null || context.nodePlan() == null) {
            return "native-kernel-ineligible:add-plan";
        }
        if (node.getDataType() != DataType.FLOAT32
                && node.getDataType() != DataType.FLOAT64
                && node.getDataType() != DataType.BFLOAT16) {
            return "native-storage-dtype-unsupported:" + node.getDataType().name().toLowerCase();
        }
        if (context.nodePlan().stridedPath()) {
            return "native-kernel-ineligible:add-strided";
        }
        if (inputs.size() != 2
                || inputs.get(0).getDataType() != node.getDataType()
                || inputs.get(1).getDataType() != node.getDataType()) {
            return "native-kernel-ineligible:add-dtype";
        }
        if (!isDenseView(inputs.get(0)) || !isDenseView(inputs.get(1)) || !isDenseView(node)) {
            return "native-kernel-ineligible:add-layout";
        }
        if (context.broadcastPlan() != null && !context.broadcastPlan().isNoBroadcast()) {
            if (node.getDataType() == DataType.FLOAT32 && biasSpec != null) {
                return "";
            }
            return "native-kernel-ineligible:add-broadcast";
        }
        int size = node.getFlatDataSize();
        if (inputs.get(0).getFlatDataSize() != size || inputs.get(1).getFlatDataSize() != size) {
            return "native-kernel-ineligible:add-shape";
        }
        return "";
    }

    private static NativeTensorStorage requireNativeInput(CpuKernelContext context, int inputIndex, DataType dtype) {
        NativeTensorStorage storage = context.executionContext().requireNativeReadable(
                context.inputNodeIds().get(inputIndex),
                CpuMaterializationReason.CPU_CONSUMER
        );
        if (storage.getType() != dtype) {
            throw new IllegalStateException("ADD native input dtype mismatch. expected=" + dtype + ", actual=" + storage.getType());
        }
        return storage;
    }

    private static CpuStorageView segmentView(Tensor tensor, NativeTensorStorage storage) {
        return CpuStorageView.segment(
                tensor.getDataType(),
                storage.segment(),
                tensor.getShapeUnsafe(),
                tensor.getStridesUnsafe(),
                tensor.getStorageOffsetUnsafe(),
                tensor.getFlatDataSize()
        );
    }

    private static void validateDenseAdd(CpuStorageBindings bindings) {
        if (bindings.inputs().size() != 2) {
            throw new IllegalArgumentException("ADD storage loop requires exactly 2 inputs.");
        }
        CpuStorageView left = bindings.input(0);
        CpuStorageView right = bindings.input(1);
        CpuStorageView output = bindings.output();
        if (left.dtype() != output.dtype() || right.dtype() != output.dtype()) {
            throw new IllegalArgumentException("ADD storage loop dtype mismatch.");
        }
        if (left.kind() != output.kind() || right.kind() != output.kind()) {
            throw new IllegalArgumentException("ADD storage loop requires matching storage kinds.");
        }
        if (left.logicalSize() != output.logicalSize() || right.logicalSize() != output.logicalSize()) {
            throw new IllegalArgumentException("ADD storage loop requires same-shape dense inputs.");
        }
        if (!isDenseView(left) || !isDenseView(right) || !isDenseView(output)) {
            throw new IllegalArgumentException("ADD storage loop requires dense zero-offset views.");
        }
    }

    private static boolean isDenseView(Tensor tensor) {
        return tensor.isContiguous() && tensor.getStorageOffsetUnsafe() == 0;
    }

    private static boolean isDenseView(CpuStorageView view) {
        if (view.storageOffset() != 0) {
            return false;
        }
        int[] shape = view.shape();
        int[] strides = view.strides();
        int expected = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            if (strides[i] != expected) {
                return false;
            }
            expected = Math.multiplyExact(expected, shape[i]);
        }
        return true;
    }

    private static boolean nativeRequested(CpuKernelContext context) {
        return context != null
                && context.executionContext().runtimeConfig() != null
                && context.executionContext().runtimeConfig().cpuStorageProfile() == CpuStorageProfile.CPU_NATIVE;
    }

    private static BiasBroadcastSpec biasBroadcastSpec(ResolvedBroadcastPlan plan, int leftSize, int rightSize, int outputSize) {
        if (plan == null || plan.isNoBroadcast() || product(plan.outShape()) != outputSize) {
            return null;
        }
        int[] shape = plan.outShape();
        int lastDim = shape[shape.length - 1];
        boolean leftFull = isFullOutputSide(plan.aEffStrides(), plan.outStrides());
        boolean rightFull = isFullOutputSide(plan.bEffStrides(), plan.outStrides());
        boolean leftBias = isLastDimBiasSide(plan.aEffStrides(), shape) && leftSize == lastDim;
        boolean rightBias = isLastDimBiasSide(plan.bEffStrides(), shape) && rightSize == lastDim;
        if (leftFull && leftSize == outputSize && rightBias) {
            return new BiasBroadcastSpec(false, lastDim);
        }
        if (leftBias && rightFull && rightSize == outputSize) {
            return new BiasBroadcastSpec(true, lastDim);
        }
        return null;
    }

    private static boolean isFullOutputSide(int[] effectiveStrides, int[] outputStrides) {
        return Arrays.equals(effectiveStrides, outputStrides);
    }

    private static boolean isLastDimBiasSide(int[] effectiveStrides, int[] outputShape) {
        if (effectiveStrides == null || outputShape == null || effectiveStrides.length != outputShape.length || outputShape.length < 2) {
            return false;
        }
        int last = effectiveStrides.length - 1;
        if (outputShape[last] <= 0 || effectiveStrides[last] != 1) {
            return false;
        }
        for (int dim = 0; dim < last; dim++) {
            if (effectiveStrides[dim] != 0) {
                return false;
            }
        }
        return true;
    }

    private static int product(int[] values) {
        int product = 1;
        for (int value : values) {
            product *= value;
        }
        return product;
    }

    private static void publishTrace(CpuKernelContext context, NativeCpuKernelFact fact, String actualCpuStorage, String fallbackReason) {
        var runtime = context.executionContext().runtimeConfig();
        Tensor runtimeTensor = context.executionContext().runtimeTensorForNodeId(context.nodeId());
        boolean bf16Promoted = runtimeTensor.getDataType() == DataType.BFLOAT16
                && "CPU_NATIVE".equals(actualCpuStorage)
                && (fallbackReason == null || fallbackReason.isBlank());
        context.putRuntimeState(
                runtimeTensor,
                new NativeCpuTraceState(
                        runtime.cpuStorageProfile().name(),
                        runtime.nativeCpuFailurePolicy().name(),
                        "CPU_NATIVE",
                        actualCpuStorage,
                        fact.status().name(),
                        fact.family().name(),
                        fallbackReason,
                        bf16Promoted ? "BF16" : "",
                        bf16Promoted ? "F32_PROMOTED" : ""
                )
        );
    }

    private static String safeMessage(Throwable t) {
        String message = t.getMessage();
        return message == null || message.isBlank() ? t.getClass().getSimpleName() : message;
    }

    private record BiasBroadcastSpec(boolean leftBias, int lastDim) {
    }
}
