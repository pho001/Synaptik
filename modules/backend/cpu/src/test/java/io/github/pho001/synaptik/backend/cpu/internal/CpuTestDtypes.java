package io.github.pho001.synaptik.backend.cpu.internal;

import io.github.pho001.synaptik.model.datatype.DataType;
import java.util.List;

/** Finite six-type basis implemented by the historical clean-Java oracle generators. */
public final class CpuTestDtypes {
    private CpuTestDtypes() {}

    /** @return historical oracle types, not a claim about the current CPU capability surface */
    public static List<DataType> legacyOracleTypes() {
        return List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16,
                DataType.INT32, DataType.INT64, DataType.BOOL);
    }
}
