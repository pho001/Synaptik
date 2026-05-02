package backend.cuda.bridge;

import backend.accelerator.lowering.GpuBackendParityReport;
import backend.accelerator.lowering.GpuBackendParityReporter;
import backend.accelerator.lowering.GpuBackendParityRow;

import java.util.ArrayList;
import java.util.List;

/**
 * Layered CUDA bridge capability state.
 *
 * @param nativeLibraryAvailable whether a native library lookup was resolved
 * @param cudaRuntimeAvailable whether the shim reports CUDA runtime/device availability
 * @param contextAvailable whether a CUDA context can be created in principle
 * @param graphExecutionAvailable whether graph compile/execute ABI symbols are present
 * @param bufferExecutionSupported whether native buffer execution can be used
 * @param layoutAbiV2Supported whether layout ABI v2 metadata symbols can be used
 * @param layoutAbiV2Version native layout ABI version, or 0 when unavailable
 * @param code stable capability code
 * @param reason human-readable diagnostic reason
 */
public record CudaBridgeCapabilities(
        boolean nativeLibraryAvailable,
        boolean cudaRuntimeAvailable,
        boolean contextAvailable,
        boolean graphExecutionAvailable,
        boolean bufferExecutionSupported,
        boolean layoutAbiV2Supported,
        int layoutAbiV2Version,
        CudaBridgeCapabilityCode code,
        String reason
) {
    public CudaBridgeCapabilities {
        layoutAbiV2Version = Math.max(0, layoutAbiV2Version);
        code = code == null ? CudaBridgeCapabilityCode.NATIVE_LIBRARY_UNAVAILABLE : code;
        reason = reason == null ? "" : reason;
    }

    /**
     * Returns fully available graph capability state.
     */
    public static CudaBridgeCapabilities available(boolean bufferExecutionSupported) {
        return available(bufferExecutionSupported, false, 0);
    }

    /**
     * Returns fully available graph capability state with layout ABI detail.
     */
    public static CudaBridgeCapabilities available(
            boolean bufferExecutionSupported,
            boolean layoutAbiV2Supported,
            int layoutAbiV2Version
    ) {
        return new CudaBridgeCapabilities(
                true,
                true,
                true,
                true,
                bufferExecutionSupported,
                layoutAbiV2Supported,
                layoutAbiV2Version,
                CudaBridgeCapabilityCode.AVAILABLE,
                ""
        );
    }

    /**
     * Returns unavailable capability state with a stable code.
     */
    public static CudaBridgeCapabilities unavailable(CudaBridgeCapabilityCode code, String reason) {
        return new CudaBridgeCapabilities(false, false, false, false, false, false, 0, code, reason);
    }

    /**
     * Returns a dimension-by-dimension capability report. The report is conservative:
     * capability skips are diagnostic evidence and never support evidence.
     */
    public CudaCapabilityReport report() {
        List<CudaCapabilityReport.Entry> entries = new ArrayList<>();
        entries.add(entry(
                CudaCapabilityDimension.NATIVE_LIBRARY,
                nativeLibraryAvailable ? CudaCapabilityDimensionStatus.AVAILABLE : CudaCapabilityDimensionStatus.UNAVAILABLE,
                nativeLibraryAvailable ? "CUDA native library lookup succeeded." : reason
        ));
        entries.add(entry(
                CudaCapabilityDimension.CUDA_RUNTIME,
                cudaRuntimeAvailable ? CudaCapabilityDimensionStatus.AVAILABLE : CudaCapabilityDimensionStatus.UNAVAILABLE,
                cudaRuntimeAvailable ? "CUDA runtime/device reported available." : reason
        ));
        entries.add(entry(
                CudaCapabilityDimension.HARDWARE_DEVICE,
                cudaRuntimeAvailable ? CudaCapabilityDimensionStatus.AVAILABLE : CudaCapabilityDimensionStatus.UNAVAILABLE,
                cudaRuntimeAvailable ? "CUDA hardware device reported available by native shim." : reason
        ));
        entries.add(entry(
                CudaCapabilityDimension.CONTEXT,
                contextAvailable ? CudaCapabilityDimensionStatus.AVAILABLE : CudaCapabilityDimensionStatus.UNAVAILABLE,
                contextAvailable ? "CUDA context can be created." : reason
        ));
        entries.add(entry(
                CudaCapabilityDimension.GRAPH_EXECUTION_ABI,
                graphExecutionAvailable ? CudaCapabilityDimensionStatus.AVAILABLE : CudaCapabilityDimensionStatus.UNAVAILABLE,
                graphExecutionAvailable ? "CUDA graph compile/execute ABI is available." : reason
        ));
        entries.add(entry(
                CudaCapabilityDimension.BUFFER_BINDING_ABI,
                bufferExecutionSupported ? CudaCapabilityDimensionStatus.AVAILABLE : CudaCapabilityDimensionStatus.UNAVAILABLE,
                bufferExecutionSupported ? "CUDA native buffer binding ABI is available." : "CUDA native buffer binding ABI is unavailable."
        ));
        entries.add(entry(
                CudaCapabilityDimension.LAYOUT_ABI_V2,
                layoutAbiV2Supported
                        ? CudaCapabilityDimensionStatus.AVAILABLE
                        : layoutAbiV2Version > 0
                        ? CudaCapabilityDimensionStatus.VERSION_MISMATCH
                        : CudaCapabilityDimensionStatus.UNAVAILABLE,
                "layoutAbiV2Version=" + layoutAbiV2Version
        ));
        addDTypeRoleEntries(entries);
        addDagPrimitiveEntries(entries);
        entries.add(entry(
                CudaCapabilityDimension.VENDOR_LIBRARY_ROUTE,
                CudaCapabilityDimensionStatus.NOT_INTEGRATED,
                "cuBLAS/cuDNN routing is not integrated in the CUDA graph bridge"
        ));
        entries.add(entry(
                CudaCapabilityDimension.TOOLCHAIN,
                CudaCapabilityDimensionStatus.UNKNOWN,
                nativeLibraryAvailable
                        ? "Native shim is loaded; build toolchain availability is not probed at runtime."
                        : "CUDA build toolchain availability is not probed at runtime."
        ));
        return new CudaCapabilityReport(entries);
    }

    private static void addDTypeRoleEntries(List<CudaCapabilityReport.Entry> entries) {
        entries.add(entry(CudaCapabilityDimension.DTYPE_ROLE, CudaCapabilityDimensionStatus.AVAILABLE,
                "FLOAT32 compute/output is the current CUDA native dense graph contract."));
        entries.add(entry(CudaCapabilityDimension.DTYPE_ROLE, CudaCapabilityDimensionStatus.UNAVAILABLE,
                "BFLOAT16 compute/output is not supported by the CUDA graph bridge."));
        entries.add(entry(CudaCapabilityDimension.DTYPE_ROLE, CudaCapabilityDimensionStatus.UNAVAILABLE,
                "BOOL output is not supported by the CUDA graph bridge."));
        entries.add(entry(CudaCapabilityDimension.DTYPE_ROLE, CudaCapabilityDimensionStatus.UNAVAILABLE,
                "INT32 compute/output is not supported by the CUDA graph bridge."));
    }

    private static void addDagPrimitiveEntries(List<CudaCapabilityReport.Entry> entries) {
        GpuBackendParityReport parity = GpuBackendParityReporter.cudaAgainstMetal();
        long supported = parity.rows().stream().filter(GpuBackendParityRow::cudaSupported).count();
        long gaps = parity.gapRows().size();
        entries.add(entry(CudaCapabilityDimension.DAG_PRIMITIVE, CudaCapabilityDimensionStatus.AVAILABLE,
                "CUDA supported DAG primitive rows=" + supported));
        entries.add(entry(CudaCapabilityDimension.DAG_PRIMITIVE, CudaCapabilityDimensionStatus.UNAVAILABLE,
                "CUDA parity gap rows requiring evidence=" + gaps));
    }

    private static CudaCapabilityReport.Entry entry(
            CudaCapabilityDimension dimension,
            CudaCapabilityDimensionStatus status,
            String detail
    ) {
        return new CudaCapabilityReport.Entry(dimension, status, detail);
    }
}
