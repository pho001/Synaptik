package io.github.pho001.synaptik.backend.cpu.internal;

import io.github.pho001.synaptik.model.datatype.DataType;
import java.util.List;

/** Current CPU executable dtype surface; unsupported model dtypes remain tested by capability suites. */
public final class CpuTestDtypes {
    private CpuTestDtypes() {}

    public static List<DataType> currentExecutable() {
        return List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16,
                DataType.INT32, DataType.INT64, DataType.BOOL);
    }
}
