package tuning.ownership;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

public final class TuningKnobOwnership {
    private static final Map<String, TuningKnobOwner> OWNERS = buildOwners();

    private TuningKnobOwnership() {
    }

    public static TuningKnobOwner ownerOf(String knobKey) {
        TuningKnobOwner owner = OWNERS.get(knobKey);
        if (owner == null) {
            throw new IllegalArgumentException("Unknown tuning knob: " + knobKey);
        }
        return owner;
    }

    public static void validateGraphWorkload(Map<String, String> knobAssignments, String source) {
        validate(knobAssignments, source, TuningKnobOwner.GRAPH_WORKLOAD);
    }

    public static void validatePlatformDtype(Map<String, String> knobAssignments, String source) {
        validate(knobAssignments, source, TuningKnobOwner.PLATFORM_DTYPE);
    }

    public static Set<String> knownKnobs() {
        return Set.copyOf(OWNERS.keySet());
    }

    private static void validate(Map<String, String> knobAssignments, String source, TuningKnobOwner expected) {
        if (knobAssignments == null || knobAssignments.isEmpty()) {
            return;
        }
        String label = source == null || source.isBlank() ? "candidate" : source;
        TreeSet<String> unknown = new TreeSet<>();
        TreeSet<String> wrongOwner = new TreeSet<>();
        for (String key : knobAssignments.keySet()) {
            TuningKnobOwner owner = OWNERS.get(key);
            if (owner == null) {
                unknown.add(key);
            } else if (owner != expected) {
                wrongOwner.add(key + "=" + owner);
            }
        }
        if (!unknown.isEmpty() || !wrongOwner.isEmpty()) {
            throw new IllegalStateException(
                    label + " changed knobs outside " + expected + " ownership; unknown=" + unknown
                            + ", wrongOwner=" + wrongOwner
            );
        }
    }

    private static Map<String, TuningKnobOwner> buildOwners() {
        LinkedHashMap<String, TuningKnobOwner> owners = new LinkedHashMap<>();
        add(owners, TuningKnobOwner.GRAPH_WORKLOAD,
                "compile.backendPlanning.discoveryMode",
                "compile.backendPlanning.failurePolicy",
                "compile.backendPlanning.ownershipPlanner",
                "compile.backendPlanning.cost.metalTransferModel",
                "compile.backendPlanning.cpuRegion.policy",
                "compile.regionOptimization.cpuFusion.mode"
        );
        add(owners, TuningKnobOwner.PLATFORM_DTYPE,
                "cpu.lowCostTargetChunksPerWorker",
                "cpu.mediumCostTargetChunksPerWorker",
                "cpu.highCostTargetChunksPerWorker",
                "cpu.minScalarChunkSize",
                "cpu.minVectorChunkSize",
                "cpu.minReductionChunkSize",
                "cpu.commonPoolLowCostMaxWorkPerWorker",
                "runtime.blas.provider",
                "runtime.blas.matmulMinWork",
                "runtime.blas.f32RequireMgeK",
                "runtime.blas.f32MaxNOverK",
                "runtime.blas.f32WideRequireMgeK",
                "runtime.blas.f32WideMaxNOverK",
                "cpu.matMulParallelMinSize",
                "cpu.matMulTileM",
                "cpu.matMulTileN",
                "cpu.matMulTileK",
                "cpu.matMulMicroKernel",
                "cpu.attentionMatMulTileM",
                "cpu.attentionMatMulTileN",
                "cpu.attentionMatMulTileK",
                "cpu.attentionMatMulMicroKernel",
                "runtime.conv2d.blasProvider",
                "runtime.conv2d.f64MinWork",
                "runtime.conv2d.f32MinWork",
                "runtime.conv2d.f32RequireMgeK",
                "runtime.conv2d.f32MaxNOverK",
                "runtime.conv2d.bf16MinWork",
                "runtime.conv2d.bf16RequireMgeK",
                "runtime.conv2d.bf16MaxNOverK",
                "cpu.cheapVectorMinSize",
                "cpu.transcendentalVectorMinSize",
                "cpu.cheapParallelMinSize",
                "cpu.transcendentalParallelMinSize",
                "cpu.fusedCheapVectorMinSize",
                "cpu.fusedTranscendentalVectorMinSize",
                "cpu.fusedCheapParallelMinSize",
                "cpu.fusedTranscendentalParallelMinSize",
                "cpu.fusedCheapContiguousAsmVectorWidth",
                "cpu.fusedCheapStridedAsmVectorWidth",
                "cpu.fusedNonCheapContiguousAsmVectorWidth",
                "cpu.fusedNonCheapStridedAsmVectorWidth",
                "cpu.reductionVectorMinSize",
                "cpu.reductionParallelMinSize",
                "cpu.attentionVectorMinSize",
                "cpu.attentionParallelMinSize",
                "cpu.contiguousMaterializeThreshold",
                "cpu.cheapF64MaterializeThreshold",
                "cpu.cheapF32MaterializeThreshold",
                "cpu.cheapBF16MaterializeThreshold",
                "cpu.whereMaterializeThreshold",
                "runtime.accelerator.metal.enabled",
                "runtime.accelerator.metal.requireRuntimeAvailability",
                "runtime.accelerator.metal.minimumEstimatedWork",
                "runtime.accelerator.cuda.buffer.bindingMode",
                "runtime.accelerator.opencl.buffer.bindingMode",
                "runtime.accelerator.metal.buffer.bindingMode"
        );
        return Map.copyOf(owners);
    }

    private static void add(
            LinkedHashMap<String, TuningKnobOwner> owners,
            TuningKnobOwner owner,
            String... keys
    ) {
        for (String key : keys) {
            TuningKnobOwner previous = owners.put(key, owner);
            if (previous != null && previous != owner) {
                throw new IllegalStateException("Duplicate tuning knob owner for " + key);
            }
        }
    }
}
