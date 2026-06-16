package backend.cpu1.prepare.dispatch;

import backend.cpu1.kernels.Cpu1VectorizationKind;
import backend.cpu1.launch.Cpu1LaunchConfig;
import backend.cpu1.storage.Cpu1StorageKind;
import operations.Operation;

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
        if (kernelOpType == null) {
            throw new IllegalArgumentException("kernelOpType cannot be null");
        }
        if (costClass == null) {
            throw new IllegalArgumentException("costClass cannot be null");
        }
        if (requestedVectorizationKind == null) {
            throw new IllegalArgumentException("requestedVectorizationKind cannot be null");
        }
        if (launchConfig == null) {
            throw new IllegalArgumentException("launchConfig cannot be null");
        }
        if (storageKind == null) {
            throw new IllegalArgumentException("storageKind cannot be null");
        }
        scalarChunkSize = Math.max(1, scalarChunkSize);
        vectorChunkSize = Math.max(1, vectorChunkSize);
        plannedWorkers = Math.max(1, plannedWorkers);
    }
}
