package tuning.calibration.family;

import tensor.DataType;
import tuning.ownership.TuningKnobOwnership;

import java.util.EnumMap;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

public final class CalibrationFamilyRegistry {
    private static final String VERSION = "calibration-family-v2";
    private static final Map<CalibrationFamilyId, CalibrationFamilySpec> SPECS = buildSpecs();

    private CalibrationFamilyRegistry() {
    }

    public static String version() {
        return VERSION;
    }

    public static List<CalibrationFamilyId> standardSuite() {
        return List.of(
                CalibrationFamilyId.SCHEDULER,
                CalibrationFamilyId.MATMUL,
                CalibrationFamilyId.ATTENTION_MATMUL,
                CalibrationFamilyId.ELEMENTWISE_DISPATCH,
                CalibrationFamilyId.FUSED_DISPATCH,
                CalibrationFamilyId.FUSED_ASM_WIDTH,
                CalibrationFamilyId.REDUCTION,
                CalibrationFamilyId.ATTENTION_THRESHOLDS,
                CalibrationFamilyId.MATERIALIZATION
        );
    }

    public static List<CalibrationFamilyId> fullSuite(boolean includeAccelerators) {
        if (!includeAccelerators) {
            return standardSuite();
        }
        java.util.ArrayList<CalibrationFamilyId> families = new java.util.ArrayList<>(standardSuite());
        families.add(CalibrationFamilyId.METAL_SELECTION);
        return List.copyOf(families);
    }

    public static CalibrationFamilySpec spec(CalibrationFamilyId id) {
        CalibrationFamilySpec spec = SPECS.get(Objects.requireNonNull(id, "id cannot be null"));
        if (spec == null) {
            throw new IllegalArgumentException("Unknown calibration family: " + id);
        }
        return spec;
    }

    public static CalibrationFamilyId parse(String value) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException("Calibration family cannot be blank.");
        }
        String normalized = value.trim().toLowerCase(java.util.Locale.ROOT);
        for (CalibrationFamilySpec spec : SPECS.values()) {
            if (spec.cliName().equals(normalized)) {
                return spec.id();
            }
        }
        throw new IllegalArgumentException("Unknown calibration family `" + value + "`. Supported families: " + supportedCliNames());
    }

    public static String supportedCliNames() {
        return String.join(", ", SPECS.values().stream()
                .map(CalibrationFamilySpec::cliName)
                .sorted()
                .toList());
    }

    public static boolean supportsDType(CalibrationFamilyId id, DataType dataType) {
        return spec(id).supportedDTypes().contains(dataType);
    }

    public static void validateCandidateChanges(CalibrationFamilyId id, Map<String, String> changes) {
        if (changes == null || changes.isEmpty()) {
            return;
        }
        Set<String> owned = spec(id).ownedKnobs();
        List<String> offenders = changes.keySet().stream()
                .filter(key -> !owned.contains(key))
                .sorted()
                .toList();
        if (!offenders.isEmpty()) {
            throw new IllegalStateException(
                    "Calibration family " + id + " candidate changed knobs outside its ownership: " + offenders
            );
        }
        TuningKnobOwnership.validatePlatformDtype(changes, id.name());
    }

    private static Map<CalibrationFamilyId, CalibrationFamilySpec> buildSpecs() {
        Map<CalibrationFamilyId, CalibrationFamilySpec> specs = new EnumMap<>(CalibrationFamilyId.class);
        put(specs, CalibrationFamilyId.SCHEDULER, "scheduler", false, Set.of(
                "cpu.lowCostTargetChunksPerWorker",
                "cpu.mediumCostTargetChunksPerWorker",
                "cpu.highCostTargetChunksPerWorker",
                "cpu.minScalarChunkSize",
                "cpu.minVectorChunkSize",
                "cpu.minReductionChunkSize",
                "cpu.commonPoolLowCostMaxWorkPerWorker"
        ));
        put(specs, CalibrationFamilyId.MATMUL, "matmul", false, Set.of(
                "runtime.blas.provider",
                "runtime.blas.matmulMinWork",
                "runtime.blas.f32RequireMgeK",
                "runtime.blas.f32MaxNOverK",
                "runtime.blas.f32WideRequireMgeK",
                "runtime.blas.f32WideMaxNOverK",
                "runtime.blas.threads",
                "runtime.blas.openBlasArrayCopyThreads",
                "runtime.blas.openBlasNativeSegmentThreads",
                "cpu.matMulParallelMinSize",
                "cpu.matMulTileM",
                "cpu.matMulTileN",
                "cpu.matMulTileK",
                "cpu.matMulMicroKernel"
        ));
        put(specs, CalibrationFamilyId.ATTENTION_MATMUL, "attention-matmul", false, Set.of(
                "cpu.attentionMatMulTileM",
                "cpu.attentionMatMulTileN",
                "cpu.attentionMatMulTileK",
                "cpu.attentionMatMulMicroKernel"
        ));
        put(specs, CalibrationFamilyId.ELEMENTWISE_DISPATCH, "elementwise-dispatch", false, Set.of(
                "cpu.cheapVectorMinSize",
                "cpu.nativeF32CheapVectorMinSize",
                "cpu.nativeF64CheapVectorMinSize",
                "cpu.transcendentalVectorMinSize",
                "cpu.cheapParallelMinSize",
                "cpu.transcendentalParallelMinSize"
        ));
        put(specs, CalibrationFamilyId.FUSED_DISPATCH, "fused-dispatch", false, Set.of(
                "cpu.fusedCheapVectorMinSize",
                "cpu.fusedTranscendentalVectorMinSize",
                "cpu.fusedCheapParallelMinSize",
                "cpu.fusedTranscendentalParallelMinSize"
        ));
        put(specs, CalibrationFamilyId.FUSED_ASM_WIDTH, "fused-asm-width", false,
                Set.of("cpu.fusedAsmVectorWidth"));
        put(specs, CalibrationFamilyId.REDUCTION, "reduction", false, Set.of(
                "cpu.reductionVectorMinSize",
                "cpu.reductionParallelMinSize"
        ));
        put(specs, CalibrationFamilyId.ATTENTION_THRESHOLDS, "attention-thresholds", false, Set.of(
                "cpu.attentionVectorMinSize",
                "cpu.attentionParallelMinSize"
        ));
        put(specs, CalibrationFamilyId.MATERIALIZATION, "materialization", false, Set.of(
                "cpu.contiguousMaterializeThreshold",
                "cpu.cheapF64MaterializeThreshold",
                "cpu.cheapF32MaterializeThreshold",
                "cpu.cheapBF16MaterializeThreshold",
                "cpu.whereMaterializeThreshold"
        ));
        put(specs, CalibrationFamilyId.METAL_SELECTION, "metal-selection", true, EnumSet.of(DataType.FLOAT32), Set.of(
                "runtime.accelerator.metal.enabled",
                "runtime.accelerator.metal.requireRuntimeAvailability",
                "runtime.accelerator.metal.minimumEstimatedWork"
        ));
        return Map.copyOf(specs);
    }

    private static void put(
            Map<CalibrationFamilyId, CalibrationFamilySpec> specs,
            CalibrationFamilyId id,
            String cliName,
            boolean acceleratorOptIn,
            Set<String> ownedKnobs
    ) {
        put(specs, id, cliName, acceleratorOptIn, EnumSet.of(DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16), ownedKnobs);
    }

    private static void put(
            Map<CalibrationFamilyId, CalibrationFamilySpec> specs,
            CalibrationFamilyId id,
            String cliName,
            boolean acceleratorOptIn,
            Set<DataType> supportedDTypes,
            Set<String> ownedKnobs
    ) {
        specs.put(id, new CalibrationFamilySpec(
                id,
                cliName,
                supportedDTypes,
                acceleratorOptIn,
                Set.copyOf(ownedKnobs)
        ));
    }
}
