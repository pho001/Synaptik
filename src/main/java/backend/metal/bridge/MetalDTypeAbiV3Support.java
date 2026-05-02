package backend.metal.bridge;

import tensor.DataType;

/**
 * Constants for the optional Metal dtype ABI v3 descriptor.
 *
 * <p>The current execution symbols remain the FLOAT32-oriented {@code *_f32} ABI.
 * ABI v3 discovery only proves that a dylib can describe wider dtype roles without
 * silently treating them as supported compute/output paths.</p>
 */
public final class MetalDTypeAbiV3Support {
    public static final int REQUIRED_VERSION = 3;

    public static final int ROLE_STORAGE = 1;
    public static final int ROLE_EXTERNAL_INPUT = 2;
    public static final int ROLE_PREDICATE_INPUT = 3;
    public static final int ROLE_COMPUTE = 4;
    public static final int ROLE_OUTPUT = 5;

    public static final int DTYPE_FLOAT32 = 1;
    public static final int DTYPE_BOOL = 2;
    public static final int DTYPE_BFLOAT16 = 3;
    public static final int DTYPE_INT32 = 4;
    public static final int DTYPE_FLOAT64 = 5;

    private MetalDTypeAbiV3Support() {
    }

    public static int descriptorCode(DataType dataType) {
        return switch (dataType) {
            case FLOAT32 -> DTYPE_FLOAT32;
            case BOOL -> DTYPE_BOOL;
            case BFLOAT16 -> DTYPE_BFLOAT16;
            case INT32 -> DTYPE_INT32;
            case FLOAT64 -> DTYPE_FLOAT64;
        };
    }
}
