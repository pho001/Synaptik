package graph.optimizer.region;

import tensor.DataType;

public record ValueTypeContract(
        DataType logicalType,
        DataType storageType,
        DataType computeType,
        DataType transportType
) {
    public ValueTypeContract {
        if (logicalType == null || storageType == null || computeType == null || transportType == null) {
            throw new IllegalArgumentException("ValueTypeContract dtypes cannot be null");
        }
    }

    public static ValueTypeContract same(DataType dataType) {
        return new ValueTypeContract(dataType, dataType, dataType, dataType);
    }
}
