package backend.cpu.kernels.plan;

import tensor.DataType;

import java.util.List;
import java.util.Objects;

public record PreparedTypeContract(
        DataType outputType,
        List<DataType> expectedInputTypes
) {
    public PreparedTypeContract {
        outputType = Objects.requireNonNull(outputType, "outputType cannot be null");
        expectedInputTypes = List.copyOf(expectedInputTypes == null ? List.of() : expectedInputTypes);
    }
}
