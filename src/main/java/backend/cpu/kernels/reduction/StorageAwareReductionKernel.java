package backend.cpu.kernels.reduction;

import backend.cpu.execution.CpuKernelContext;
import backend.cpu.kernels.CpuKernelCall;
import backend.cpu.kernels.CpuKernelResult;
import backend.cpu.kernels.CpuStorageAwareKernel;
import backend.cpu.storage.CpuStorageView;
import backend.cpu.nativecpu.CpuNativeTraceSupport;
import backend.cpu.nativecpu.layout.NativeCpuStorageFamily;
import backend.cpu.nativecpu.layout.NativeSegmentStridedKernels;
import backend.cpu.nativecpu.layout.NativeSegmentView;
import backend.cpu.nativecpu.layout.TensorPhysicalView;
import runtime.contract.CpuMaterializationReason;
import config.runtime.CpuStorageProfile;
import config.runtime.NativeCpuFailurePolicy;
import operations.Operation;
import tensor.DataType;
import tensor.Tensor;
import tensor.storage.NativeTensorStorage;

abstract class StorageAwareReductionKernel implements CpuStorageAwareKernel {
    @Override
    public final CpuKernelResult execute(CpuKernelCall call) {
        Tensor input = requireSingleInput(call);
        int dimension = dimension(call.operation());
        if (nativeRequested(call.context())) {
            CpuKernelResult result = executeNative(call, input, dimension);
            if (!CpuNativeTraceSupport.CPU_NATIVE.equals(result.route())) {
                executeArray(
                        call.operation(),
                        input,
                        call.outputTensor(),
                        requireSingleInputView(call),
                        requireOutputView(call),
                        call.context(),
                        dimension
                );
            }
            return result;
        }
        executeArray(
                call.operation(),
                input,
                call.outputTensor(),
                requireSingleInputView(call),
                requireOutputView(call),
                call.context(),
                dimension
        );
        return CpuKernelResult.completed();
    }

    protected abstract Operation.OpType opType();

    protected abstract int dimension(Operation operation);

    protected abstract void executeArray(
            Operation operation,
            Tensor input,
            Tensor output,
            CpuStorageView inputView,
            CpuStorageView outputView,
            CpuKernelContext context,
            int dimension
    );

    protected final String opLabel() {
        return opType().name().toLowerCase();
    }

    private CpuKernelResult executeNative(CpuKernelCall call, Tensor input, int dimension) {
        Tensor output = call.outputTensor();
        CpuKernelContext context = call.context();
        String reason = nativeIneligibleReason(call, input, output, dimension);
        if (!reason.isBlank()) {
            return fallbackToArray(context, reason);
        }
        try {
            NativeTensorStorage inputStorage = requireNativeInput(context, input.getDataType(), opLabel().toUpperCase());
            NativeTensorStorage outputStorage = allocateNativeOutput(output, context, opLabel());
            NativeSegmentStridedKernels.runReduction(
                    opType(),
                    nativeView(context.inputNodeIds().getFirst(), input, inputStorage),
                    nativeView(context.nodeId(), output, outputStorage),
                    dimension
            );
            context.executionContext().attachNativeStorage(
                    context.nodeId(),
                    outputStorage,
                    "reduction storage-aware kernel " + opLabel().toUpperCase()
                            + " wrote " + output.getDataType() + " native output"
            );
            CpuNativeTraceSupport.publishSegmentScalar(context, CpuNativeTraceSupport.CPU_NATIVE, "");
            return CpuKernelResult.route(CpuNativeTraceSupport.CPU_NATIVE);
        } catch (Throwable t) {
            return fallbackToArray(context, "native-kernel-failed:" + opLabel() + ":" + safeMessage(t));
        }
    }

    private String nativeIneligibleReason(CpuKernelCall call, Tensor input, Tensor output, int dimension) {
        CpuKernelContext context = call.context();
        if (context == null || context.nodePlan() == null) {
            return "native-kernel-ineligible:" + opLabel() + "-plan";
        }
        if (input == null || output == null || context.inputNodeIds().size() != 1 || call.inputs().size() != 1) {
            return "native-kernel-ineligible:" + opLabel() + "-input-count";
        }
        if (!NativeSegmentStridedKernels.supportsReduction(opType(), output.getDataType())) {
            return unsupportedReductionReason(output.getDataType());
        }
        if (input.getDataType() != output.getDataType()) {
            return "native-storage-dtype-unsupported:" + output.getDataType().name().toLowerCase();
        }
        if (context.nodePlan().stridedPath() || !denseTensor(input) || !denseTensor(output)) {
            return "native-kernel-ineligible:" + opLabel() + "-strided";
        }
        int[] shape = input.getShapeUnsafe();
        if (shape == null || shape.length == 0 || dimension < -1 || dimension >= shape.length) {
            return "native-kernel-ineligible:" + opLabel() + "-axis";
        }
        if (input.getFlatDataSize() <= 0 || expectedOutputSize(shape, dimension) != output.getFlatDataSize()) {
            return "native-kernel-ineligible:" + opLabel() + "-shape";
        }
        return "";
    }

    private String unsupportedReductionReason(DataType dataType) {
        if (dataType == DataType.BFLOAT16
                && (opType() == Operation.OpType.REDUCE_MIN || opType() == Operation.OpType.REDUCE_MAX)) {
            return "native-bf16-reduce-minmax-output-policy-unsupported";
        }
        if (!supportsNativeReductionDType(dataType)) {
            return "native-storage-dtype-unsupported:" + dataType.name().toLowerCase();
        }
        return "native-kernel-unsupported:" + opLabel();
    }

    private CpuKernelResult fallbackToArray(CpuKernelContext context, String reason) {
        requireFallbackAllowed(context, opLabel() + " reduction", reason);
        requireCpuReadableInputs(context);
        CpuNativeTraceSupport.publishSegmentScalar(context, CpuNativeTraceSupport.CPU_ARRAY, reason);
        return CpuKernelResult.fallback(CpuNativeTraceSupport.CPU_ARRAY, reason);
    }

    private NativeTensorStorage requireNativeInput(CpuKernelContext context, DataType dtype, String op) {
        NativeTensorStorage storage = context.executionContext().requireNativeReadable(
                context.inputNodeIds().getFirst(),
                CpuMaterializationReason.CPU_CONSUMER
        );
        if (storage.getType() != dtype) {
            throw new IllegalStateException("native " + op + " requires " + dtype + " native input storage");
        }
        return storage;
    }

    private NativeTensorStorage allocateNativeOutput(Tensor output, CpuKernelContext context, String label) {
        return context.executionContext().allocateNativeStorage(
                output.getDataType(),
                output.getFlatDataSize(),
                "node-" + context.nodeId() + ":" + output.getLabel() + ":native-" + label
        );
    }

    private NativeSegmentView nativeView(int nodeId, Tensor tensor, NativeTensorStorage storage) {
        return NativeSegmentView.from(
                TensorPhysicalView.of(
                        nodeId,
                        tensor.getDataType(),
                        tensor.getShapeUnsafe(),
                        tensor.getStridesUnsafe(),
                        tensor.getStorageOffsetUnsafe(),
                        NativeCpuStorageFamily.CPU_NATIVE
                ),
                storage
        );
    }

    private Tensor requireSingleInput(CpuKernelCall call) {
        if (call.inputTensors().size() != 1) {
            throw new IllegalArgumentException(opLabel().toUpperCase() + " expects exactly one input tensor");
        }
        return call.inputTensors().getFirst();
    }

    private CpuStorageView requireSingleInputView(CpuKernelCall call) {
        if (call.inputs().size() != 1) {
            throw new IllegalArgumentException(opLabel().toUpperCase() + " expects exactly one input storage view");
        }
        return call.inputs().getFirst();
    }

    private CpuStorageView requireOutputView(CpuKernelCall call) {
        if (call.output() == null) {
            throw new IllegalArgumentException(opLabel().toUpperCase() + " requires an output storage view");
        }
        return call.output();
    }

    private static boolean denseTensor(Tensor tensor) {
        return tensor != null && tensor.isContiguous() && !tensor.hasStorageOffset();
    }

    private static int expectedOutputSize(int[] shape, int dimension) {
        if (dimension == -1) {
            return 1;
        }
        int size = 1;
        for (int dim = 0; dim < shape.length; dim++) {
            if (dim != dimension) {
                size = Math.multiplyExact(size, shape[dim]);
            }
        }
        return size;
    }

    private static void requireFallbackAllowed(CpuKernelContext context, String family, String reason) {
        if (context.executionContext().runtimeConfig().nativeCpuFailurePolicy() == NativeCpuFailurePolicy.REQUIRE_NATIVE) {
            throw new IllegalStateException("Native CPU execution required but " + family + " fell back to Java: " + reason);
        }
    }

    private static void requireCpuReadableInputs(CpuKernelContext context) {
        for (int inputNodeId : context.inputNodeIds()) {
            context.executionContext().requireCpuReadable(inputNodeId, CpuMaterializationReason.CPU_CONSUMER);
        }
    }

    private static boolean nativeRequested(CpuKernelContext context) {
        return context != null
                && context.executionContext().runtimeConfig() != null
                && context.executionContext().runtimeConfig().cpuStorageProfile() == CpuStorageProfile.CPU_NATIVE;
    }

    private static boolean supportsNativeReductionDType(DataType dataType) {
        return dataType == DataType.FLOAT32
                || dataType == DataType.FLOAT64
                || dataType == DataType.BFLOAT16
                || dataType == DataType.BOOL;
    }

    private static String safeMessage(Throwable t) {
        String message = t.getMessage();
        return message == null || message.isBlank() ? t.getClass().getSimpleName() : message;
    }
}
