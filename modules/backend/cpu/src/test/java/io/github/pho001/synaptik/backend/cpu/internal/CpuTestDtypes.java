package io.github.pho001.synaptik.backend.cpu.internal;

import io.github.pho001.synaptik.model.datatype.DataType;
import java.util.List;

/** Current CPU types and the explicit historical clean-Java oracle fixture basis. */
public final class CpuTestDtypes {
    private CpuTestDtypes() {}

    /** @return every currently executable CPU data type */
    public static List<DataType> currentExecutable() {
        return List.of(DataType.values());
    }

    /** @return historical oracle types, not a claim about the current CPU capability surface */
    public static List<DataType> legacyOracleTypes() {
        return List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16,
                DataType.INT32, DataType.INT64, DataType.BOOL);
    }
}
