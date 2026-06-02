package backend.cpu1.kernels;

import backend.cpu1.kernels.elementwise.Cpu1ElementwiseKernelId;
import backend.cpu1.kernels.elementwise.Cpu1ElementwiseKernelKey;
import backend.cpu1.storage.Cpu1StorageKind;
import operations.Operation;
import tensor.DataType;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Registry of concrete cpu1 kernel ids.
 */
public final class Cpu1KernelRegistry {
    private final Map<Cpu1ElementwiseKernelKey, Cpu1ElementwiseKernelId> kernels;

    public Cpu1KernelRegistry() {
        Map<Cpu1ElementwiseKernelKey, Cpu1ElementwiseKernelId> registered = new HashMap<>();
        for (Cpu1ElementwiseKernelId kernelId : Cpu1ElementwiseKernelId.values()) {
            registered.put(kernelId.key(), kernelId);
        }
        this.kernels = Map.copyOf(registered);
    }

    public Cpu1ElementwiseKernelId resolve(Operation.OpType opType, DataType dataType) {
        return resolve(opType, dataType, Cpu1LayoutKind.CONTIGUOUS, Cpu1VectorizationKind.SCALAR);
    }

    public Cpu1ElementwiseKernelId resolve(Operation.OpType opType, DataType dataType, Cpu1LayoutKind layoutKind) {
        return resolve(opType, dataType, layoutKind, Cpu1VectorizationKind.SCALAR);
    }

    public Cpu1ElementwiseKernelId resolve(
            Operation.OpType opType,
            DataType dataType,
            Cpu1LayoutKind layoutKind,
            Cpu1VectorizationKind vectorizationKind
    ) {
        return resolve(opType, dataType, layoutKind, Cpu1StorageKind.JAVA_ARRAY, vectorizationKind);
    }

    public Cpu1ElementwiseKernelId resolve(
            Operation.OpType opType,
            DataType dataType,
            Cpu1LayoutKind layoutKind,
            Cpu1StorageKind storageKind,
            Cpu1VectorizationKind vectorizationKind
    ) {
        return resolve(
                opType,
                dataType,
                Cpu1ElementwiseKernelKey.of(opType, dataType, layoutKind, storageKind, vectorizationKind).inputDataTypes(),
                layoutKind,
                storageKind,
                vectorizationKind
        );
    }

    public Cpu1ElementwiseKernelId resolve(
            Operation.OpType opType,
            DataType dataType,
            List<DataType> inputDataTypes,
            Cpu1LayoutKind layoutKind,
            Cpu1StorageKind storageKind,
            Cpu1VectorizationKind vectorizationKind
    ) {
        Cpu1ElementwiseKernelKey key = Cpu1ElementwiseKernelKey.of(
                Objects.requireNonNull(opType, "opType cannot be null"),
                Objects.requireNonNull(dataType, "dataType cannot be null"),
                Objects.requireNonNull(inputDataTypes, "inputDataTypes cannot be null"),
                Objects.requireNonNull(layoutKind, "layoutKind cannot be null"),
                Objects.requireNonNull(storageKind, "storageKind cannot be null"),
                Objects.requireNonNull(vectorizationKind, "vectorizationKind cannot be null")
        );
        Cpu1ElementwiseKernelId kernelId = kernels.get(key);
        if (kernelId == null) {
            throw new UnsupportedOperationException("No cpu1 " + layoutKind + " " + storageKind + " " + vectorizationKind
                    + " kernel for " + opType + ", dtype " + dataType + ", inputs " + inputDataTypes);
        }
        return kernelId;
    }
}
