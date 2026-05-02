package backend.accelerator.residency;

import backend.ComputeBackend;
import backend.accelerator.lowering.GpuLoweringUnsupportedReason;
import backend.metal.MetalDTypeCapabilityDecision;
import backend.metal.MetalMpsCapabilities;
import tensor.DataType;

import java.util.Objects;

/**
 * Backend-neutral dtype residency policy for accelerator diagnostics.
 */
public final class AcceleratorDTypeResidencyPolicy {
    private static final String ROLE_EXTERNAL_INPUT = "externalInput";
    private static final String ROLE_INTERNAL_VALUE = "internalValue";
    private static final String ROLE_OUTPUT = "output";
    private static final String ROLE_COMPUTE = "compute";

    private AcceleratorDTypeResidencyPolicy() {
    }

    public static AcceleratorDTypeResidencyDecision forExternalInput(ComputeBackend backend, DataType dataType) {
        return decide(backend, dataType, ROLE_EXTERNAL_INPUT);
    }

    public static AcceleratorDTypeResidencyDecision forInternalValue(ComputeBackend backend, DataType dataType) {
        return decide(backend, dataType, ROLE_INTERNAL_VALUE);
    }

    public static AcceleratorDTypeResidencyDecision forOutput(ComputeBackend backend, DataType dataType) {
        return decide(backend, dataType, ROLE_OUTPUT);
    }

    public static AcceleratorDTypeResidencyDecision forCompute(ComputeBackend backend, DataType dataType) {
        return decide(backend, dataType, ROLE_COMPUTE);
    }

    private static AcceleratorDTypeResidencyDecision decide(ComputeBackend backend, DataType dataType, String role) {
        Objects.requireNonNull(backend, "backend cannot be null");
        Objects.requireNonNull(dataType, "dataType cannot be null");
        return switch (backend) {
            case CPU -> cpuDecision(backend, dataType, role);
            case GPU_METAL -> metalDecision(dataType, role);
            case GPU_CUDA -> cudaDecision(dataType, role);
            case GPU_OPENCL -> rejected(backend, dataType, role, true,
                    "backend=GPU_OPENCL role=" + role + " dtype=" + dataType + " has no native residency policy");
        };
    }

    private static AcceleratorDTypeResidencyDecision cpuDecision(ComputeBackend backend, DataType dataType, String role) {
        return new AcceleratorDTypeResidencyDecision(
                backend,
                dataType,
                role,
                true,
                false,
                false,
                false,
                null,
                "backend=CPU role=" + role + " dtype=" + dataType + " residentRepresentable=true nativeGpu=false"
        );
    }

    private static AcceleratorDTypeResidencyDecision metalDecision(DataType dataType, String role) {
        MetalDTypeCapabilityDecision decision = switch (role) {
            case ROLE_EXTERNAL_INPUT -> MetalMpsCapabilities.externalInputDecision(dataType);
            case ROLE_OUTPUT -> MetalMpsCapabilities.outputDecision(dataType);
            case ROLE_COMPUTE -> MetalMpsCapabilities.computeDecision(dataType);
            case ROLE_INTERNAL_VALUE -> MetalMpsCapabilities.storageDecision(dataType);
            default -> MetalMpsCapabilities.computeDecision(dataType);
        };
        boolean legal = switch (role) {
            case ROLE_EXTERNAL_INPUT, ROLE_OUTPUT, ROLE_COMPUTE -> decision.supported();
            case ROLE_INTERNAL_VALUE -> dataType == DataType.FLOAT32 || dataType == DataType.BFLOAT16;
            default -> false;
        };
        if (legal) {
            return supported(
                    ComputeBackend.GPU_METAL,
                    dataType,
                    role,
                    ROLE_EXTERNAL_INPUT.equals(role),
                    ROLE_OUTPUT.equals(role),
                    ROLE_COMPUTE.equals(role)
            );
        }
        return rejected(ComputeBackend.GPU_METAL, dataType, role, true, decision.detail());
    }

    private static AcceleratorDTypeResidencyDecision cudaDecision(DataType dataType, String role) {
        boolean legal = dataType == DataType.FLOAT32;
        if (legal) {
            return supported(
                    ComputeBackend.GPU_CUDA,
                    dataType,
                    role,
                    ROLE_EXTERNAL_INPUT.equals(role) || ROLE_INTERNAL_VALUE.equals(role),
                    ROLE_OUTPUT.equals(role),
                    ROLE_COMPUTE.equals(role)
            );
        }
        return rejected(ComputeBackend.GPU_CUDA, dataType, role, true,
                "backend=GPU_CUDA role=" + role + " dtype=" + dataType
                        + " unsupported; CUDA native dense buffer execution supports FLOAT32 only");
    }

    private static AcceleratorDTypeResidencyDecision supported(
            ComputeBackend backend,
            DataType dataType,
            String role,
            boolean inputLegal,
            boolean outputLegal,
            boolean computeLegal
    ) {
        return new AcceleratorDTypeResidencyDecision(
                backend,
                dataType,
                role,
                true,
                inputLegal,
                outputLegal,
                computeLegal,
                null,
                "backend=" + backend + " role=" + role + " dtype=" + dataType + " residentRepresentable=true"
        );
    }

    private static AcceleratorDTypeResidencyDecision rejected(
            ComputeBackend backend,
            DataType dataType,
            String role,
            boolean residentRepresentable,
            String detail
    ) {
        return new AcceleratorDTypeResidencyDecision(
                backend,
                dataType,
                role,
                residentRepresentable,
                false,
                false,
                false,
                GpuLoweringUnsupportedReason.UNSUPPORTED_DTYPE,
                detail
        );
    }
}
