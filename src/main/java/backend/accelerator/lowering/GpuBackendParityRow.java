package backend.accelerator.lowering;

import operations.Operation;

import java.util.Objects;

/**
 * Derived Metal-vs-CUDA parity row for one GPU lowering operation entry.
 */
public record GpuBackendParityRow(
        Operation.OpType opType,
        GpuLoweringOperationFamily family,
        GpuLoweringCoverageStatus metalStatus,
        GpuLoweringUnsupportedReason metalReason,
        String metalNote,
        GpuLoweringCoverageStatus cudaStatus,
        GpuLoweringUnsupportedReason cudaReason,
        String cudaNote,
        boolean metalSupported,
        boolean cudaSupported,
        boolean parityGap,
        String requiredEvidence
) {
    public static final String NONE = "NONE";
    public static final String CUDA_NATIVE_EXECUTION_REQUIRED = "CUDA_NATIVE_EXECUTION_REQUIRED";
    public static final String CUDA_DTYPE_OR_LAYOUT_CONTRACT_REQUIRED = "CUDA_DTYPE_OR_LAYOUT_CONTRACT_REQUIRED";
    public static final String CUDA_INDEX_SEMANTICS_REQUIRED = "CUDA_INDEX_SEMANTICS_REQUIRED";
    public static final String CUDA_EXPLICIT_REJECTION_OK = "CUDA_EXPLICIT_REJECTION_OK";

    public GpuBackendParityRow {
        Objects.requireNonNull(opType, "opType");
        Objects.requireNonNull(family, "family");
        Objects.requireNonNull(metalStatus, "metalStatus");
        Objects.requireNonNull(metalReason, "metalReason");
        Objects.requireNonNull(cudaStatus, "cudaStatus");
        Objects.requireNonNull(cudaReason, "cudaReason");
        metalNote = metalNote == null ? "" : metalNote.strip();
        cudaNote = cudaNote == null ? "" : cudaNote.strip();
        requiredEvidence = requiredEvidence == null || requiredEvidence.isBlank()
                ? requiredEvidence(metalStatus, metalReason, cudaStatus, cudaReason)
                : requiredEvidence.strip();
    }

    /**
     * Creates a parity row from matching Metal and CUDA coverage matrix entries.
     */
    public static GpuBackendParityRow from(GpuLoweringCoverageEntry metal, GpuLoweringCoverageEntry cuda) {
        Objects.requireNonNull(metal, "metal");
        Objects.requireNonNull(cuda, "cuda");
        boolean metalSupported = metal.status() == GpuLoweringCoverageStatus.SUPPORTED;
        boolean cudaSupported = cuda.status() == GpuLoweringCoverageStatus.SUPPORTED;
        boolean parityGap = metalSupported && !cudaSupported;
        return new GpuBackendParityRow(
                metal.opType(),
                metal.family(),
                metal.status(),
                metal.reason(),
                metal.note(),
                cuda.status(),
                cuda.reason(),
                cuda.note(),
                metalSupported,
                cudaSupported,
                parityGap,
                requiredEvidence(metal.status(), metal.reason(), cuda.status(), cuda.reason())
        );
    }

    private static String requiredEvidence(
            GpuLoweringCoverageStatus metalStatus,
            GpuLoweringUnsupportedReason metalReason,
            GpuLoweringCoverageStatus cudaStatus,
            GpuLoweringUnsupportedReason cudaReason
    ) {
        if (metalStatus == GpuLoweringCoverageStatus.SUPPORTED
                && cudaStatus == GpuLoweringCoverageStatus.SUPPORTED) {
            return NONE;
        }
        if (metalStatus != GpuLoweringCoverageStatus.SUPPORTED
                && cudaStatus != GpuLoweringCoverageStatus.SUPPORTED
                && metalReason == cudaReason) {
            return NONE;
        }
        return switch (cudaReason) {
            case CAPABILITY_MISSING, DAG_PRIMITIVE_UNSUPPORTED -> CUDA_NATIVE_EXECUTION_REQUIRED;
            case UNSUPPORTED_DTYPE, UNSUPPORTED_LAYOUT -> CUDA_DTYPE_OR_LAYOUT_CONTRACT_REQUIRED;
            case UNSUPPORTED_INDEX_SEMANTICS, UNSUPPORTED_DUPLICATE_INDEX, UNSUPPORTED_BOUNDS_CHECK ->
                    CUDA_INDEX_SEMANTICS_REQUIRED;
            default -> metalStatus == GpuLoweringCoverageStatus.SUPPORTED
                    ? CUDA_NATIVE_EXECUTION_REQUIRED
                    : CUDA_EXPLICIT_REJECTION_OK;
        };
    }
}
