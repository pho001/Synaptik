package backend.cpu1.prepare.dispatch;

import backend.cpu1.kernels.Cpu1VectorizationKind;
import backend.cpu1.launch.Cpu1LaunchConfig;
import backend.cpu1.storage.Cpu1StorageKind;
import operations.Operation;

import java.util.Objects;

/**
 * Prepare-time cpu1 dispatch result for one executable unit.
 */
public record Cpu1DispatchDecision(
        Operation.OpType kernelOpType,
        Cpu1CostClass costClass,
        Cpu1VectorizationKind requestedVectorizationKind,
        Cpu1LaunchConfig launchConfig,
        Cpu1StorageKind storageKind,
        int scalarChunkSize,
        int vectorChunkSize,
        int plannedWorkers
) {
    public Cpu1DispatchDecision {
        Objects.requireNonNull(kernelOpType, "kernelOpType cannot be null");
        Objects.requireNonNull(costClass, "costClass cannot be null");
        Objects.requireNonNull(requestedVectorizationKind, "requestedVectorizationKind cannot be null");
        Objects.requireNonNull(launchConfig, "launchConfig cannot be null");
        Objects.requireNonNull(storageKind, "storageKind cannot be null");
        scalarChunkSize = Math.max(1, scalarChunkSize);
        vectorChunkSize = Math.max(1, vectorChunkSize);
        plannedWorkers = Math.max(1, plannedWorkers);
    }
}
