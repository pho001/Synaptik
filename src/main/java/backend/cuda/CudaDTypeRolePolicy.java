package backend.cuda;

import tensor.DataType;

import java.util.Objects;

/**
 * Conservative CUDA dtype role policy.
 *
 * <p>dtype residency is not native dtype compute: residency-only support is report evidence,
 * not permission to execute arithmetic or produce native outputs for that dtype.</p>
 */
public final class CudaDTypeRolePolicy {
    public static final String SUPPORTED = "SUPPORTED";
    public static final String UNSUPPORTED_DTYPE = "UNSUPPORTED_DTYPE";
    public static final String UNSUPPORTED_ROLE = "UNSUPPORTED_ROLE";
    public static final String RESIDENCY_ONLY_NOT_COMPUTE = "RESIDENCY_ONLY_NOT_COMPUTE";

    private CudaDTypeRolePolicy() {
    }

    public static CudaDTypeRoleDecision decide(DataType dataType, CudaDTypeRole role) {
        Objects.requireNonNull(dataType, "dataType");
        Objects.requireNonNull(role, "role");
        return switch (role) {
            case COMPUTE_INPUT, COMPUTE_OUTPUT -> computeDecision(dataType, role);
            case INDEX_INPUT -> indexInputDecision(dataType);
            case PREDICATE_INPUT -> predicateInputDecision(dataType);
            case RESIDENCY_ONLY -> residencyDecision(dataType);
        };
    }

    public static CudaDTypeRoleDecision computeInput(DataType dataType) {
        return decide(dataType, CudaDTypeRole.COMPUTE_INPUT);
    }

    public static CudaDTypeRoleDecision computeOutput(DataType dataType) {
        return decide(dataType, CudaDTypeRole.COMPUTE_OUTPUT);
    }

    public static CudaDTypeRoleDecision indexInput(DataType dataType) {
        return decide(dataType, CudaDTypeRole.INDEX_INPUT);
    }

    public static CudaDTypeRoleDecision predicateInput(DataType dataType) {
        return decide(dataType, CudaDTypeRole.PREDICATE_INPUT);
    }

    public static CudaDTypeRoleDecision residencyOnly(DataType dataType) {
        return decide(dataType, CudaDTypeRole.RESIDENCY_ONLY);
    }

    private static CudaDTypeRoleDecision computeDecision(DataType dataType, CudaDTypeRole role) {
        if (dataType == DataType.FLOAT32) {
            return supported(dataType, role, "CUDA dense FLOAT32 native compute/output role is supported.");
        }
        if (dataType == DataType.BFLOAT16 || dataType == DataType.BOOL || dataType == DataType.INT32) {
            return rejected(dataType, role, RESIDENCY_ONLY_NOT_COMPUTE,
                    "backend=GPU_CUDA role=" + role + " dtype=" + dataType
                            + " code=" + RESIDENCY_ONLY_NOT_COMPUTE
                            + "; dtype residency is not native dtype compute.");
        }
        return rejected(dataType, role, UNSUPPORTED_DTYPE,
                "backend=GPU_CUDA role=" + role + " dtype=" + dataType
                        + " code=" + UNSUPPORTED_DTYPE);
    }

    private static CudaDTypeRoleDecision indexInputDecision(DataType dataType) {
        if (dataType == DataType.INT32) {
            return supported(dataType, CudaDTypeRole.INDEX_INPUT,
                    "CUDA INT32 index-input role is supported as residency/legality evidence, not generic INT32 compute.");
        }
        return rejected(dataType, CudaDTypeRole.INDEX_INPUT, UNSUPPORTED_DTYPE,
                "backend=GPU_CUDA role=INDEX_INPUT dtype=" + dataType + " code=" + UNSUPPORTED_DTYPE
                        + "; CUDA index inputs require INT32.");
    }

    private static CudaDTypeRoleDecision predicateInputDecision(DataType dataType) {
        if (dataType == DataType.BOOL) {
            return supported(dataType, CudaDTypeRole.PREDICATE_INPUT,
                    "CUDA BOOL predicate-input role is supported as residency/legality evidence, not BOOL-producing compute.");
        }
        return rejected(dataType, CudaDTypeRole.PREDICATE_INPUT, UNSUPPORTED_DTYPE,
                "backend=GPU_CUDA role=PREDICATE_INPUT dtype=" + dataType + " code=" + UNSUPPORTED_DTYPE
                        + "; CUDA predicate inputs require BOOL.");
    }

    private static CudaDTypeRoleDecision residencyDecision(DataType dataType) {
        if (dataType == DataType.FLOAT32
                || dataType == DataType.BFLOAT16
                || dataType == DataType.BOOL
                || dataType == DataType.INT32) {
            return supported(dataType, CudaDTypeRole.RESIDENCY_ONLY,
                    "CUDA can represent " + dataType + " residency evidence; dtype residency is not native dtype compute.");
        }
        return rejected(dataType, CudaDTypeRole.RESIDENCY_ONLY, UNSUPPORTED_DTYPE,
                "backend=GPU_CUDA role=RESIDENCY_ONLY dtype=" + dataType + " code=" + UNSUPPORTED_DTYPE);
    }

    private static CudaDTypeRoleDecision supported(DataType dataType, CudaDTypeRole role, String detail) {
        return new CudaDTypeRoleDecision(dataType, role, true, SUPPORTED,
                "backend=GPU_CUDA role=" + role + " dtype=" + dataType + " code=" + SUPPORTED + "; " + detail);
    }

    private static CudaDTypeRoleDecision rejected(
            DataType dataType,
            CudaDTypeRole role,
            String reasonCode,
            String detail
    ) {
        return new CudaDTypeRoleDecision(dataType, role, false, reasonCode, detail);
    }
}
