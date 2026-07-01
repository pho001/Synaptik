package planning.partition.execution;

import tensor.DataType;

/**
 * Data type contract for a value as it moves through partition planning.
 *
 * @param logicalType logical tensor type
 * @param storageType type used for stored buffers
 * @param computeType type used for computation
 * @param transportType type used when crossing partition boundaries
 */
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

    /**
     * Creates a contract where all type roles use the same dtype.
     *
     * @param dataType dtype to use for all roles
     * @return type contract
     */
    public static ValueTypeContract same(DataType dataType) {
        return new ValueTypeContract(dataType, dataType, dataType, dataType);
    }
}
