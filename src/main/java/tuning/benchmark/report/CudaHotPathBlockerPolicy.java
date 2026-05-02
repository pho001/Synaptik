package tuning.benchmark.report;

import java.util.Comparator;
import java.util.List;
import java.util.Set;

/**
 * Deterministic CUDA hot-path blocker policy for v1.6 parity planning.
 */
public final class CudaHotPathBlockerPolicy {
    private static final Set<String> V16_BLOCKERS = Set.of(
            "transformer_block_hot_path",
            "masked_sdpa_small",
            "conv2d_resnet_3x3",
            "max_pool2d_small",
            "avg_pool2d_small",
            "dense_loss_small",
            "gather_take_small",
            "bool_compare_where_small"
    );
    private static final Set<String> REQUIRES_NATIVE_EVIDENCE = Set.of(
            "scatter_index_gradient_small",
            "training_transformer_block_hot_path",
            "training_dense_loss_small",
            "training_layer_norm_small",
            "training_reduction_chain_small",
            "training_cross_entropy_small"
    );
    private static final Set<String> ACCEPTED_CAPABILITY_GAPS = Set.of(
            "mlp_classifier_small_bf16",
            "layer_norm_small_bf16",
            "rms_norm_small_bf16",
            "reduction_chain_small_bf16"
    );

    private CudaHotPathBlockerPolicy() {
    }

    public static CudaHotPathBlockerClass classify(String workloadName) {
        String name = normalize(workloadName);
        if (V16_BLOCKERS.contains(name)) {
            return CudaHotPathBlockerClass.V16_BLOCKER;
        }
        if (REQUIRES_NATIVE_EVIDENCE.contains(name)) {
            return CudaHotPathBlockerClass.REQUIRES_NATIVE_EVIDENCE;
        }
        if (ACCEPTED_CAPABILITY_GAPS.contains(name)) {
            return CudaHotPathBlockerClass.ACCEPTED_CAPABILITY_GAP;
        }
        if (GpuHotPathCoverageTargets.defaultWorkloadNames().contains(name)) {
            return CudaHotPathBlockerClass.ACCEPTED_CAPABILITY_GAP;
        }
        return CudaHotPathBlockerClass.FUTURE_SCOPE;
    }

    public static String detail(String workloadName) {
        return switch (classify(workloadName)) {
            case V16_BLOCKER -> "CUDA CPU exit is a v1.6 blocker for this hot path.";
            case ACCEPTED_CAPABILITY_GAP -> "CUDA gap is accepted in Phase 40 baseline until a scoped dtype or capability phase targets it.";
            case FUTURE_SCOPE -> "Target is not part of the checked v1.6 hot-path set.";
            case REQUIRES_NATIVE_EVIDENCE -> "CUDA support requires native execution, parity, and trace/report evidence before promotion.";
        };
    }

    public static List<String> v16BlockerTargets() {
        return V16_BLOCKERS.stream()
                .sorted()
                .toList();
    }

    public static List<GpuCoverageTriageReport.CudaHotPathBlockerEntry> entriesForDefaultTargets() {
        return GpuHotPathCoverageTargets.defaultWorkloadNames().stream()
                .map(name -> new GpuCoverageTriageReport.CudaHotPathBlockerEntry(name, classify(name), detail(name)))
                .sorted(Comparator.comparing(GpuCoverageTriageReport.CudaHotPathBlockerEntry::workloadName))
                .toList();
    }

    private static String normalize(String workloadName) {
        return workloadName == null ? "" : workloadName.strip();
    }
}
