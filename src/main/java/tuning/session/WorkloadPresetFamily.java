package tuning.session;

import tuning.workload.WorkloadKind;
import tuning.workload.WorkloadSpec;

import java.util.List;

public enum WorkloadPresetFamily {
    GENERIC(WorkloadKind.GENERIC, TuningPreset.BALANCED, TuningPreset.BALANCED),
    MATMUL(WorkloadKind.MATMUL, TuningPreset.BALANCED, TuningPreset.BALANCED),
    MLP_CLASSIFICATION(WorkloadKind.MLP_CLASSIFICATION, TuningPreset.BALANCED, TuningPreset.THOROUGH),
    CONV2D(WorkloadKind.CONV2D, TuningPreset.BALANCED, TuningPreset.BALANCED),
    NORMALIZATION(WorkloadKind.NORMALIZATION, TuningPreset.BALANCED, TuningPreset.THOROUGH),
    TRANSFORMER_HOT_PATH(WorkloadKind.TRANSFORMER_HOT_PATH, TuningPreset.BALANCED, TuningPreset.THOROUGH),
    POOL2D(WorkloadKind.POOL2D, TuningPreset.BALANCED, TuningPreset.BALANCED),
    LOSS(WorkloadKind.LOSS, TuningPreset.THOROUGH, TuningPreset.THOROUGH);

    private final WorkloadKind workloadKind;
    private final TuningPreset benchmarkPreset;
    private final TuningPreset autotunePreset;

    WorkloadPresetFamily(WorkloadKind workloadKind, TuningPreset benchmarkPreset, TuningPreset autotunePreset) {
        this.workloadKind = workloadKind;
        this.benchmarkPreset = benchmarkPreset;
        this.autotunePreset = autotunePreset;
    }

    public WorkloadKind workloadKind() {
        return workloadKind;
    }

    public TuningPreset benchmarkPreset() {
        return benchmarkPreset;
    }

    public TuningPreset autotunePreset() {
        return autotunePreset;
    }

    public static WorkloadPresetFamily resolve(WorkloadKind kind) {
        if (kind == null) {
            return GENERIC;
        }
        for (WorkloadPresetFamily family : values()) {
            if (family.workloadKind == kind) {
                return family;
            }
        }
        return GENERIC;
    }

    public static WorkloadPresetFamily resolve(WorkloadSpec workload) {
        return workload == null ? GENERIC : resolve(workload.kind());
    }

    public static TuningPreset benchmarkPresetFor(WorkloadSpec workload) {
        return resolve(workload).benchmarkPreset();
    }

    public static TuningPreset autotunePresetFor(WorkloadSpec workload) {
        return resolve(workload).autotunePreset();
    }

    public static TuningPreset benchmarkPresetForSuite(List<WorkloadSpec> workloads) {
        if (workloads == null || workloads.isEmpty()) {
            return TuningPreset.BALANCED;
        }
        TuningPreset selected = TuningPreset.QUICK;
        for (WorkloadSpec workload : workloads) {
            TuningPreset candidate = benchmarkPresetFor(workload);
            if (candidate.ordinal() > selected.ordinal()) {
                selected = candidate;
            }
        }
        return selected;
    }
}
