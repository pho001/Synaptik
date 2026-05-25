package backend.cpu.nativecpu;

import backend.cpu.execution.CpuKernelContext;
import tensor.DataType;
import tensor.Tensor;

public final class CpuNativeTraceSupport {
    public static final String CPU_ARRAY = "CPU_ARRAY";
    public static final String CPU_NATIVE = "CPU_NATIVE";

    private static final String ARRAY_ONLY = "ARRAY_ONLY";
    private static final String NATIVE_CORRECT_BUT_SLOW = "NATIVE_CORRECT_BUT_SLOW";
    private static final String NATIVE_MICROKERNEL = "NATIVE_MICROKERNEL";
    private static final String NATIVE_UNSUPPORTED = "NATIVE_UNSUPPORTED";
    private static final String SEGMENT_SCALAR = "SEGMENT_SCALAR";
    private static final String VIEW_ONLY = "VIEW_ONLY";

    private CpuNativeTraceSupport() {
    }

    public static void publishSegmentScalar(
            CpuKernelContext context,
            String actualCpuStorage,
            String fallbackReason
    ) {
        publishWithFallbackClassification(context, NATIVE_CORRECT_BUT_SLOW, SEGMENT_SCALAR, actualCpuStorage, fallbackReason);
    }

    public static void publishNativeMicrokernel(
            CpuKernelContext context,
            String actualCpuStorage,
            String fallbackReason
    ) {
        publishWithFallbackClassification(context, NATIVE_CORRECT_BUT_SLOW, NATIVE_MICROKERNEL, actualCpuStorage, fallbackReason);
    }

    public static void publishViewOnly(
            CpuKernelContext context,
            String actualCpuStorage,
            String fallbackReason
    ) {
        publish(context, VIEW_ONLY, VIEW_ONLY, actualCpuStorage, fallbackReason);
    }

    public static void publish(
            CpuKernelContext context,
            String status,
            String family,
            String actualCpuStorage,
            String fallbackReason
    ) {
        var runtime = context.executionContext().runtimeConfig();
        Tensor runtimeTensor = context.executionContext().runtimeTensorForNodeId(context.nodeId());
        boolean bf16Promoted = runtimeTensor.getDataType() == DataType.BFLOAT16
                && CPU_NATIVE.equals(actualCpuStorage)
                && (fallbackReason == null || fallbackReason.isBlank());
        context.putRuntimeState(
                runtimeTensor,
                new NativeCpuTraceState(
                        runtime.cpuStorageProfile().name(),
                        runtime.nativeCpuFailurePolicy().name(),
                        CPU_NATIVE,
                        actualCpuStorage,
                        status,
                        family,
                        fallbackReason,
                        bf16Promoted ? "BF16" : "",
                        bf16Promoted ? "F32_PROMOTED" : ""
                )
        );
    }

    private static void publishWithFallbackClassification(
            CpuKernelContext context,
            String defaultStatus,
            String defaultFamily,
            String actualCpuStorage,
            String fallbackReason
    ) {
        String status = statusFor(defaultStatus, fallbackReason);
        publish(context, status, familyFor(status, defaultFamily), actualCpuStorage, fallbackReason);
    }

    private static String statusFor(String defaultStatus, String fallbackReason) {
        if (fallbackReason == null || fallbackReason.isBlank()) {
            return defaultStatus;
        }
        if (fallbackReason.startsWith("native-storage-dtype-unsupported:")) {
            return ARRAY_ONLY;
        }
        if (fallbackReason.startsWith("native-kernel-unsupported:")
                || fallbackReason.startsWith("native-bf16-reduce-minmax-output-policy-unsupported")
                || fallbackReason.startsWith("native-argmax-index-output-unsupported")
                || fallbackReason.startsWith("native-softmax-scalar-loop-slower-than-array")) {
            return NATIVE_UNSUPPORTED;
        }
        return defaultStatus;
    }

    private static String familyFor(String status, String defaultFamily) {
        return ARRAY_ONLY.equals(status) || NATIVE_UNSUPPORTED.equals(status) ? ARRAY_ONLY : defaultFamily;
    }
}
