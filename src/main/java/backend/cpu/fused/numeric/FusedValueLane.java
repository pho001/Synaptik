package backend.cpu.fused.numeric;

import tensor.DataType;

/**
 * Stored numeric lane for fused inputs or outputs.
 */
public enum FusedValueLane {
    F32,
    F64,
    BF16,
    BOOL;

    public static FusedValueLane fromDataType(DataType dataType) {
        return switch (dataType) {
            case FLOAT64 -> F64;
            case FLOAT32 -> F32;
            case BFLOAT16 -> BF16;
            case BOOL -> BOOL;
            case INT32, INT64 -> throw new UnsupportedOperationException(
                    "Fused numeric lanes support FLOAT32, FLOAT64, BFLOAT16, and BOOL only: " + dataType
            );
        };
    }
}
