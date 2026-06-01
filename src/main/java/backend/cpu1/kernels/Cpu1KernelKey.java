package backend.cpu1.kernels;

import backend.cpu1.storage.Cpu1StorageKind;
import operations.Operation;
import tensor.DataType;

import java.util.List;
import java.util.Objects;

/**
 * Prepare-time key for resolving a concrete cpu1 kernel variant.
 */
public record Cpu1KernelKey(
        Operation.OpType opType,
        DataType dataType,
        List<DataType> inputDataTypes,
        Cpu1LayoutKind layoutKind,
        Cpu1StorageKind storageKind,
        Cpu1VectorizationKind vectorizationKind
) {
    public Cpu1KernelKey {
        Objects.requireNonNull(opType, "opType cannot be null");
        Objects.requireNonNull(dataType, "dataType cannot be null");
        inputDataTypes = List.copyOf(Objects.requireNonNull(inputDataTypes, "inputDataTypes cannot be null"));
        Objects.requireNonNull(layoutKind, "layoutKind cannot be null");
        Objects.requireNonNull(storageKind, "storageKind cannot be null");
        Objects.requireNonNull(vectorizationKind, "vectorizationKind cannot be null");
    }

    public static Cpu1KernelKey of(
            Operation.OpType opType,
            DataType dataType,
            List<DataType> inputDataTypes,
            Cpu1LayoutKind layoutKind,
            Cpu1StorageKind storageKind,
            Cpu1VectorizationKind vectorizationKind
    ) {
        return new Cpu1KernelKey(opType, dataType, inputDataTypes, layoutKind, storageKind, vectorizationKind);
    }

    public static Cpu1KernelKey of(
            Operation.OpType opType,
            DataType dataType,
            Cpu1LayoutKind layoutKind,
            Cpu1StorageKind storageKind,
            Cpu1VectorizationKind vectorizationKind
    ) {
        return of(
                opType,
                dataType,
                defaultInputDataTypes(opType, dataType),
                layoutKind,
                storageKind,
                vectorizationKind
        );
    }

    public static Cpu1KernelKey of(
            Operation.OpType opType,
            DataType dataType,
            Cpu1LayoutKind layoutKind,
            Cpu1VectorizationKind vectorizationKind
    ) {
        return of(opType, dataType, layoutKind, Cpu1StorageKind.JAVA_ARRAY, vectorizationKind);
    }

    public static Cpu1KernelKey scalar(
            Operation.OpType opType,
            DataType dataType,
            Cpu1LayoutKind layoutKind
    ) {
        return of(opType, dataType, layoutKind, Cpu1StorageKind.JAVA_ARRAY, Cpu1VectorizationKind.SCALAR);
    }

    public static Cpu1KernelKey scalar(
            Operation.OpType opType,
            DataType dataType,
            Cpu1LayoutKind layoutKind,
            Cpu1StorageKind storageKind
    ) {
        return of(opType, dataType, layoutKind, storageKind, Cpu1VectorizationKind.SCALAR);
    }

    private static List<DataType> defaultInputDataTypes(Operation.OpType opType, DataType dataType) {
        return switch (opType) {
            case LOGICAL_NOT -> List.of(DataType.BOOL);
            case LOGICAL_AND, LOGICAL_OR -> List.of(DataType.BOOL, DataType.BOOL);
            case WHERE -> List.of(DataType.BOOL, dataType, dataType);
            case ADD, SUB, MUL, DIV, MIN, MAX, POW_TENSOR, GT, GE, LT, LE, EQ, NE -> List.of(dataType, dataType);
            default -> List.of(dataType);
        };
    }
}
