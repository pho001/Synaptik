package backend.cpu1.kernels.elementwise;

import backend.cpu1.kernels.Cpu1LayoutKind;
import backend.cpu1.kernels.Cpu1VectorizationKind;
import backend.cpu1.storage.Cpu1StorageKind;
import operations.Operation;
import tensor.DataType;

import java.util.List;

/**
 * Prepare-time key for resolving a concrete cpu1 kernel variant.
 */
public record Cpu1ElementwiseKernelKey(
        Operation.OpType opType,
        DataType dataType,
        List<DataType> inputDataTypes,
        Cpu1LayoutKind layoutKind,
        Cpu1StorageKind storageKind,
        Cpu1VectorizationKind vectorizationKind
) {
    public Cpu1ElementwiseKernelKey {
        if (opType == null) {
            throw new IllegalArgumentException("opType cannot be null");
        }
        if (dataType == null) {
            throw new IllegalArgumentException("dataType cannot be null");
        }
        if (inputDataTypes == null) {
            throw new IllegalArgumentException("inputDataTypes cannot be null");
        }
        if (layoutKind == null) {
            throw new IllegalArgumentException("layoutKind cannot be null");
        }
        if (storageKind == null) {
            throw new IllegalArgumentException("storageKind cannot be null");
        }
        if (vectorizationKind == null) {
            throw new IllegalArgumentException("vectorizationKind cannot be null");
        }
        inputDataTypes = List.copyOf(inputDataTypes);
    }

    public static Cpu1ElementwiseKernelKey of(
            Operation.OpType opType,
            DataType dataType,
            List<DataType> inputDataTypes,
            Cpu1LayoutKind layoutKind,
            Cpu1StorageKind storageKind,
            Cpu1VectorizationKind vectorizationKind
    ) {
        return new Cpu1ElementwiseKernelKey(opType, dataType, inputDataTypes, layoutKind, storageKind, vectorizationKind);
    }

    public static Cpu1ElementwiseKernelKey of(
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

    public static Cpu1ElementwiseKernelKey of(
            Operation.OpType opType,
            DataType dataType,
            Cpu1LayoutKind layoutKind,
            Cpu1VectorizationKind vectorizationKind
    ) {
        return of(opType, dataType, layoutKind, Cpu1StorageKind.JAVA_ARRAY, vectorizationKind);
    }

    public static Cpu1ElementwiseKernelKey scalar(
            Operation.OpType opType,
            DataType dataType,
            Cpu1LayoutKind layoutKind
    ) {
        return of(opType, dataType, layoutKind, Cpu1StorageKind.JAVA_ARRAY, Cpu1VectorizationKind.SCALAR);
    }

    public static Cpu1ElementwiseKernelKey scalar(
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
