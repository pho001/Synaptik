package backend.cpu1.prepare.dispatch;

import backend.cpu1.kernels.Cpu1VectorizationKind;
import backend.cpu1.launch.Cpu1LaunchConfig;
import backend.cpu1.storage.Cpu1StorageKind;

/**
 * Prepare-time dispatch result for a fused elementwise unit.
 *
 * <p>Unlike single-op dispatch this record intentionally has no kernel
 * {@code Operation.OpType}: a fused partition has no representative operation
 * identity. The concrete fused IR and ASM/codegen eligibility describe what
 * can run; this record only captures prepare-time launch and vectorization
 * tuning.</p>
 */
public record Cpu1FusedDispatchDecision(
        Cpu1CostClass costClass,
        Cpu1VectorizationKind requestedVectorizationKind,
        Cpu1LaunchConfig launchConfig,
        Cpu1StorageKind storageKind,
        int scalarChunkSize,
        int vectorChunkSize,
        int plannedWorkers
) {
    public Cpu1FusedDispatchDecision {
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
