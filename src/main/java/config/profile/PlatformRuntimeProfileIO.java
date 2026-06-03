package config.profile;

import backend.ApproxMode;
import backend.blas.BlasProvider;
import backend.runtime.ExecutionMode;
import config.backend.CpuMatMulMicroKernel;
import config.backend.SumAccuracyMode;
import config.runtime.AcceleratorBufferBindingMode;
import config.runtime.AcceleratorBufferConfig;
import config.runtime.BlasStorageMode;
import config.runtime.CpuStorageProfile;
import config.runtime.DeviceTransferPolicy;
import config.runtime.NativeCpuFailurePolicy;
import tensor.DataType;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

public final class PlatformRuntimeProfileIO {
    public static final String SUPPORTED_PLANNER_SCHEMA_VERSION = "1";
    public static final String SUPPORTED_PERSISTENCE_SCHEMA_VERSION = "1";

    private PlatformRuntimeProfileIO() {
    }

    public static void save(Path path, PlatformRuntimeProfile profile) {
        if (path == null) {
            throw new IllegalArgumentException("path cannot be null");
        }
        if (profile == null) {
            throw new IllegalArgumentException("profile cannot be null");
        }
        try {
            if (path.getParent() != null) {
                Files.createDirectories(path.getParent());
            }
            Files.writeString(path, toJson(profile), StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new IllegalStateException("Failed to save platform runtime profile to " + path, e);
        }
    }

    public static String toJson(PlatformRuntimeProfile profile) {
        PlatformProfileMetadata m = profile.metadata();
        return "{\n" +
                "  \"metadata\": {\n" +
                "    \"platformProfileId\": \"" + escape(m.platformProfileId()) + "\",\n" +
                "    \"hardwareKey\": \"" + escape(m.hardwareKey()) + "\",\n" +
                "    \"frameworkVersion\": \"" + escape(m.frameworkVersion()) + "\",\n" +
                "    \"plannerSchemaVersion\": \"" + escape(m.plannerSchemaVersion()) + "\",\n" +
                "    \"persistenceSchemaVersion\": \"" + escape(m.persistenceSchemaVersion()) + "\",\n" +
                "    \"createdAtIso\": \"" + escape(m.createdAtIso()) + "\",\n" +
                "    \"calibrationPreset\": \"" + escape(m.calibrationPreset()) + "\",\n" +
                "    \"dataType\": \"" + m.dataType().name() + "\",\n" +
                "    \"executionMode\": \"" + m.executionMode().name() + "\"\n" +
                "  },\n" +
                "  \"matmul\": {\n" +
                "    \"blasProvider\": \"" + profile.matmul().blasProvider().name() + "\",\n" +
                "    \"blasMatmulMinWork\": " + profile.matmul().blasMatmulMinWork() + ",\n" +
                "    \"blasThreads\": " + profile.matmul().blasThreads() + ",\n" +
                "    \"openBlasArrayCopyThreads\": " + profile.matmul().openBlasArrayCopyThreads() + ",\n" +
                "    \"openBlasNativeSegmentThreads\": " + profile.matmul().openBlasNativeSegmentThreads() + ",\n" +
                "    \"f32RequireMgeK\": " + profile.matmul().f32RequireMgeK() + ",\n" +
                "    \"f32MaxNOverK\": " + profile.matmul().f32MaxNOverK() + ",\n" +
                "    \"f32WideRequireMgeK\": " + profile.matmul().f32WideRequireMgeK() + ",\n" +
                "    \"f32WideMaxNOverK\": " + profile.matmul().f32WideMaxNOverK() + ",\n" +
                "    \"blasStorageMode\": \"" + profile.matmul().blasStorageMode().name() + "\",\n" +
                "    \"loopUnrollFactor\": " + profile.matmul().loopUnrollFactor() + ",\n" +
                "    \"matMulTileM\": " + profile.matmul().matMulTileM() + ",\n" +
                "    \"matMulTileN\": " + profile.matmul().matMulTileN() + ",\n" +
                "    \"matMulTileK\": " + profile.matmul().matMulTileK() + ",\n" +
                "    \"attentionMatMulTileM\": " + profile.matmul().attentionMatMulTileM() + ",\n" +
                "    \"attentionMatMulTileN\": " + profile.matmul().attentionMatMulTileN() + ",\n" +
                "    \"attentionMatMulTileK\": " + profile.matmul().attentionMatMulTileK() + ",\n" +
                "    \"matMulParallelMinSize\": " + profile.matmul().matMulParallelMinSize() + ",\n" +
                "    \"matMulMicroKernel\": \"" + profile.matmul().matMulMicroKernel().name() + "\",\n" +
                "    \"attentionMatMulMicroKernel\": \"" + profile.matmul().attentionMatMulMicroKernel().name() + "\"\n" +
                "  },\n" +
                "  \"conv2d\": {\n" +
                "    \"conv2dBlasProvider\": \"" + profile.conv2d().blasProvider().name() + "\",\n" +
                "    \"conv2dF64BlasMinWork\": " + profile.conv2d().f64BlasMinWork() + ",\n" +
                "    \"conv2dF32BlasMinWork\": " + profile.conv2d().f32BlasMinWork() + ",\n" +
                "    \"conv2dF32RequireMgeK\": " + profile.conv2d().f32RequireMgeK() + ",\n" +
                "    \"conv2dF32MaxNOverK\": " + profile.conv2d().f32MaxNOverK() + ",\n" +
                "    \"conv2dBf16BlasMinWork\": " + profile.conv2d().bf16BlasMinWork() + ",\n" +
                "    \"conv2dBf16RequireMgeK\": " + profile.conv2d().bf16RequireMgeK() + ",\n" +
                "    \"conv2dBf16MaxNOverK\": " + profile.conv2d().bf16MaxNOverK() + "\n" +
                "  },\n" +
                "  \"fused\": {\n" +
                "    \"fusedCheapVectorMinSize\": " + profile.fused().fusedCheapVectorMinSize() + ",\n" +
                "    \"fusedTranscendentalVectorMinSize\": " + profile.fused().fusedTranscendentalVectorMinSize() + ",\n" +
                "    \"fusedCheapParallelMinSize\": " + profile.fused().fusedCheapParallelMinSize() + ",\n" +
                "    \"fusedTranscendentalParallelMinSize\": " + profile.fused().fusedTranscendentalParallelMinSize() + ",\n" +
                "    \"fusedAsmVectorWidth\": " + profile.fused().fusedAsmVectorWidth() + "\n" +
                "  },\n" +
                "  \"elementwiseDispatch\": {\n" +
                "    \"cheapVectorMinSize\": " + profile.elementwiseDispatch().cheapVectorMinSize() + ",\n" +
                "    \"nativeF32CheapVectorMinSize\": " + profile.elementwiseDispatch().nativeF32CheapVectorMinSize() + ",\n" +
                "    \"nativeF64CheapVectorMinSize\": " + profile.elementwiseDispatch().nativeF64CheapVectorMinSize() + ",\n" +
                "    \"transcendentalVectorMinSize\": " + profile.elementwiseDispatch().transcendentalVectorMinSize() + ",\n" +
                "    \"cheapParallelMinSize\": " + profile.elementwiseDispatch().cheapParallelMinSize() + ",\n" +
                "    \"transcendentalParallelMinSize\": " + profile.elementwiseDispatch().transcendentalParallelMinSize() + "\n" +
                "  },\n" +
                "  \"reduction\": {\n" +
                "    \"reductionVectorMinSize\": " + profile.reduction().reductionVectorMinSize() + ",\n" +
                "    \"reductionParallelMinSize\": " + profile.reduction().reductionParallelMinSize() + ",\n" +
                "    \"attentionVectorMinSize\": " + profile.reduction().attentionVectorMinSize() + ",\n" +
                "    \"attentionParallelMinSize\": " + profile.reduction().attentionParallelMinSize() + ",\n" +
                "    \"sumAccuracyMode\": \"" + profile.reduction().sumAccuracyMode().name() + "\"\n" +
                "  },\n" +
                "  \"scheduler\": {\n" +
                "    \"lowCostTargetChunksPerWorker\": " + profile.scheduler().lowCostTargetChunksPerWorker() + ",\n" +
                "    \"mediumCostTargetChunksPerWorker\": " + profile.scheduler().mediumCostTargetChunksPerWorker() + ",\n" +
                "    \"highCostTargetChunksPerWorker\": " + profile.scheduler().highCostTargetChunksPerWorker() + ",\n" +
                "    \"minScalarChunkSize\": " + profile.scheduler().minScalarChunkSize() + ",\n" +
                "    \"minVectorChunkSize\": " + profile.scheduler().minVectorChunkSize() + ",\n" +
                "    \"minReductionChunkSize\": " + profile.scheduler().minReductionChunkSize() + ",\n" +
                "    \"commonPoolLowCostMaxWorkPerWorker\": " + profile.scheduler().commonPoolLowCostMaxWorkPerWorker() + "\n" +
                "  },\n" +
                "  \"materialization\": {\n" +
                "    \"contiguousMaterializeThreshold\": " + profile.materialization().contiguousMaterializeThreshold() + ",\n" +
                "    \"cheapF64MaterializeThreshold\": " + profile.materialization().cheapF64MaterializeThreshold() + ",\n" +
                "    \"cheapF32MaterializeThreshold\": " + profile.materialization().cheapF32MaterializeThreshold() + ",\n" +
                "    \"cheapBF16MaterializeThreshold\": " + profile.materialization().cheapBF16MaterializeThreshold() + ",\n" +
                "    \"whereMaterializeThreshold\": " + profile.materialization().whereMaterializeThreshold() + "\n" +
                "  },\n" +
                "  \"numerics\": {\n" +
                "    \"approxMode\": \"" + profile.numerics().approxMode().name() + "\",\n" +
                "    \"forceExactTranscendentals\": " + profile.numerics().forceExactTranscendentals() + "\n" +
                "  },\n" +
                "  \"accelerator\": {\n" +
                "    \"cudaEnabled\": " + profile.accelerator().cuda().enabled() + ",\n" +
                "    \"cudaRequireRuntimeAvailability\": " + profile.accelerator().cuda().requireRuntimeAvailability() + ",\n" +
                "    \"cudaMinimumEstimatedWork\": " + profile.accelerator().cuda().minimumEstimatedWork() + ",\n" +
                "    \"cudaBufferBindingMode\": \"" + profile.accelerator().cuda().buffer().bindingMode().name() + "\",\n" +
                "    \"cudaAllowPreparedInputMaterialization\": " + profile.accelerator().cuda().buffer().allowPreparedInputMaterialization() + ",\n" +
                "    \"cudaBufferMinimumEstimatedWork\": " + profile.accelerator().cuda().buffer().minimumEstimatedWork() + ",\n" +
                "    \"openclEnabled\": " + profile.accelerator().opencl().enabled() + ",\n" +
                "    \"openclRequireRuntimeAvailability\": " + profile.accelerator().opencl().requireRuntimeAvailability() + ",\n" +
                "    \"openclMinimumEstimatedWork\": " + profile.accelerator().opencl().minimumEstimatedWork() + ",\n" +
                "    \"openclBufferBindingMode\": \"" + profile.accelerator().opencl().buffer().bindingMode().name() + "\",\n" +
                "    \"openclAllowPreparedInputMaterialization\": " + profile.accelerator().opencl().buffer().allowPreparedInputMaterialization() + ",\n" +
                "    \"openclBufferMinimumEstimatedWork\": " + profile.accelerator().opencl().buffer().minimumEstimatedWork() + ",\n" +
                "    \"metalEnabled\": " + profile.accelerator().metal().enabled() + ",\n" +
                "    \"metalRequireRuntimeAvailability\": " + profile.accelerator().metal().requireRuntimeAvailability() + ",\n" +
                "    \"metalMinimumEstimatedWork\": " + profile.accelerator().metal().minimumEstimatedWork() + ",\n" +
                "    \"metalBufferBindingMode\": \"" + profile.accelerator().metal().buffer().bindingMode().name() + "\",\n" +
                "    \"metalAllowPreparedInputMaterialization\": " + profile.accelerator().metal().buffer().allowPreparedInputMaterialization() + ",\n" +
                "    \"metalBufferMinimumEstimatedWork\": " + profile.accelerator().metal().buffer().minimumEstimatedWork() + "\n" +
                "  },\n" +
                "  \"runtimePolicy\": {\n" +
                "    \"cpuStorageProfile\": \"" + profile.cpuStorageProfile().name() + "\",\n" +
                "    \"nativeCpuFailurePolicy\": \"" + profile.nativeCpuFailurePolicy().name() + "\",\n" +
                "    \"deviceTransferPolicy\": \"" + profile.deviceTransferPolicy().name() + "\"\n" +
                "  }\n" +
                "}\n";
    }

    public static PlatformRuntimeProfile loadOrDefault(Path path, PlatformRuntimeProfile fallback) {
        if (path == null || !Files.exists(path)) {
            return fallback;
        }
        try {
            String json = Files.readString(path, StandardCharsets.UTF_8);
            return fromJsonOrDefault(json, fallback);
        } catch (IOException e) {
            return fallback;
        }
    }

    public static PlatformRuntimeProfile loadStrict(Path path, PlatformRuntimeProfile fallback) {
        if (path == null) {
            throw new IllegalArgumentException("path cannot be null");
        }
        if (!Files.exists(path)) {
            throw new IllegalArgumentException("Platform runtime profile does not exist: " + path);
        }
        try {
            return fromJsonStrict(Files.readString(path, StandardCharsets.UTF_8), fallback);
        } catch (IOException e) {
            throw new IllegalStateException("Invalid platform runtime profile at " + path + ": " + e.getMessage(), e);
        } catch (RuntimeException e) {
            throw new IllegalArgumentException("Invalid platform runtime profile at " + path + ": " + e.getMessage(), e);
        }
    }

    public static PlatformRuntimeProfile fromJsonOrDefault(String json, PlatformRuntimeProfile fallback) {
        if (json == null || json.isBlank() || fallback == null) {
            return fallback;
        }
        try {
            PlatformProfileMetadata m = fallback.metadata();
            CpuMatMulMicroKernel loadedMatMulMicroKernel = findEnum(
                    json,
                    "matMulMicroKernel",
                    fallback.matmul().matMulMicroKernel(),
                    CpuMatMulMicroKernel.class
            );
            int loadedMatMulTileM = findInt(json, "matMulTileM", fallback.matmul().matMulTileM());
            int loadedMatMulTileN = findInt(json, "matMulTileN", fallback.matmul().matMulTileN());
            int loadedMatMulTileK = findInt(json, "matMulTileK", fallback.matmul().matMulTileK());
            MatmulPlatformProfile loadedMatmul = new MatmulPlatformProfile(
                    findEnum(json, "blasProvider", fallback.matmul().blasProvider(), BlasProvider.class),
                    findLong(json, "blasMatmulMinWork", fallback.matmul().blasMatmulMinWork()),
                    findInt(json, "blasThreads", fallback.matmul().blasThreads()),
                    findInt(json, "openBlasArrayCopyThreads", fallback.matmul().openBlasArrayCopyThreads()),
                    findInt(json, "openBlasNativeSegmentThreads", fallback.matmul().openBlasNativeSegmentThreads()),
                    findBoolean(json, "f32RequireMgeK", fallback.matmul().f32RequireMgeK()),
                    findDouble(json, "f32MaxNOverK", fallback.matmul().f32MaxNOverK()),
                    findBoolean(json, "f32WideRequireMgeK", fallback.matmul().f32WideRequireMgeK()),
                    findDouble(json, "f32WideMaxNOverK", fallback.matmul().f32WideMaxNOverK()),
                    findEnum(json, "blasStorageMode", fallback.matmul().blasStorageMode(), BlasStorageMode.class),
                    findInt(json, "loopUnrollFactor", fallback.matmul().loopUnrollFactor()),
                    loadedMatMulTileM,
                    loadedMatMulTileN,
                    loadedMatMulTileK,
                    findInt(json, "attentionMatMulTileM", loadedMatMulTileM),
                    findInt(json, "attentionMatMulTileN", loadedMatMulTileN),
                    findInt(json, "attentionMatMulTileK", loadedMatMulTileK),
                    findInt(json, "matMulParallelMinSize", fallback.matmul().matMulParallelMinSize()),
                    loadedMatMulMicroKernel,
                    findEnum(
                            json,
                            "attentionMatMulMicroKernel",
                            loadedMatMulMicroKernel,
                            CpuMatMulMicroKernel.class
                    )
            );
            Conv2dPlatformProfile loadedConv2d = new Conv2dPlatformProfile(
                    findEnum(json, "conv2dBlasProvider", loadedMatmul.blasProvider(), BlasProvider.class),
                    findLong(json, "conv2dF64BlasMinWork", loadedMatmul.blasMatmulMinWork()),
                    findLong(json, "conv2dF32BlasMinWork", loadedMatmul.blasMatmulMinWork()),
                    findBoolean(json, "conv2dF32RequireMgeK", loadedMatmul.f32RequireMgeK()),
                    findDouble(json, "conv2dF32MaxNOverK", loadedMatmul.f32MaxNOverK()),
                    findLong(json, "conv2dBf16BlasMinWork", loadedMatmul.blasMatmulMinWork()),
                    findBoolean(json, "conv2dBf16RequireMgeK", loadedMatmul.f32RequireMgeK()),
                    findDouble(json, "conv2dBf16MaxNOverK", loadedMatmul.f32MaxNOverK())
            );
            int loadedCheapVectorMinSize = findInt(
                    json,
                    "cheapVectorMinSize",
                    fallback.elementwiseDispatch().cheapVectorMinSize()
            );
            return new PlatformRuntimeProfile(
                    new PlatformProfileMetadata(
                            findString(json, "platformProfileId", m.platformProfileId()),
                            findString(json, "hardwareKey", m.hardwareKey()),
                            findString(json, "frameworkVersion", m.frameworkVersion()),
                            findString(json, "plannerSchemaVersion", m.plannerSchemaVersion()),
                            findString(json, "persistenceSchemaVersion", m.persistenceSchemaVersion()),
                            findString(json, "createdAtIso", m.createdAtIso()),
                            findString(json, "calibrationPreset", m.calibrationPreset()),
                            findEnum(json, "dataType", m.dataType(), DataType.class),
                            findEnum(json, "executionMode", m.executionMode(), ExecutionMode.class)
                    ),
                    loadedMatmul,
                    loadedConv2d,
                    new FusedPlatformProfile(
                            findInt(json, "fusedCheapVectorMinSize", fallback.fused().fusedCheapVectorMinSize()),
                            findInt(json, "fusedTranscendentalVectorMinSize", fallback.fused().fusedTranscendentalVectorMinSize()),
                            findInt(json, "fusedCheapParallelMinSize", fallback.fused().fusedCheapParallelMinSize()),
                            findInt(json, "fusedTranscendentalParallelMinSize", fallback.fused().fusedTranscendentalParallelMinSize()),
                            findInt(json, "fusedAsmVectorWidth", fallback.fused().fusedAsmVectorWidth())
                    ),
                    new ElementwiseDispatchPlatformProfile(
                            loadedCheapVectorMinSize,
                            findInt(json, "nativeF32CheapVectorMinSize", loadedCheapVectorMinSize),
                            findInt(json, "nativeF64CheapVectorMinSize", loadedCheapVectorMinSize),
                            findInt(json, "transcendentalVectorMinSize", fallback.elementwiseDispatch().transcendentalVectorMinSize()),
                            findInt(json, "cheapParallelMinSize", fallback.elementwiseDispatch().cheapParallelMinSize()),
                            findInt(json, "transcendentalParallelMinSize", fallback.elementwiseDispatch().transcendentalParallelMinSize())
                    ),
                    new ReductionPlatformProfile(
                            findInt(json, "reductionVectorMinSize", fallback.reduction().reductionVectorMinSize()),
                            findInt(json, "reductionParallelMinSize", fallback.reduction().reductionParallelMinSize()),
                            findInt(json, "attentionVectorMinSize", fallback.reduction().attentionVectorMinSize()),
                            findInt(json, "attentionParallelMinSize", fallback.reduction().attentionParallelMinSize()),
                            findEnum(json, "sumAccuracyMode", fallback.reduction().sumAccuracyMode(), SumAccuracyMode.class)
                    ),
                    new SchedulerPlatformProfile(
                            findInt(json, "lowCostTargetChunksPerWorker", fallback.scheduler().lowCostTargetChunksPerWorker()),
                            findInt(json, "mediumCostTargetChunksPerWorker", fallback.scheduler().mediumCostTargetChunksPerWorker()),
                            findInt(json, "highCostTargetChunksPerWorker", fallback.scheduler().highCostTargetChunksPerWorker()),
                            findInt(json, "minScalarChunkSize", fallback.scheduler().minScalarChunkSize()),
                            findInt(json, "minVectorChunkSize", fallback.scheduler().minVectorChunkSize()),
                            findInt(json, "minReductionChunkSize", fallback.scheduler().minReductionChunkSize()),
                            findInt(json, "commonPoolLowCostMaxWorkPerWorker", fallback.scheduler().commonPoolLowCostMaxWorkPerWorker())
                    ),
                    new MaterializationPlatformProfile(
                            findInt(json, "contiguousMaterializeThreshold", fallback.materialization().contiguousMaterializeThreshold()),
                            findInt(
                                    json,
                                    "cheapF64MaterializeThreshold",
                                    fallback.materialization().cheapF64MaterializeThreshold()
                            ),
                            findInt(
                                    json,
                                    "cheapF32MaterializeThreshold",
                                    fallback.materialization().cheapF32MaterializeThreshold()
                            ),
                            findInt(
                                    json,
                                    "cheapBF16MaterializeThreshold",
                                    fallback.materialization().cheapBF16MaterializeThreshold()
                            ),
                            findInt(
                                    json,
                                    "whereMaterializeThreshold",
                                    fallback.materialization().whereMaterializeThreshold()
                            )
                    ),
                    new NumericsPlatformProfile(
                            findEnum(json, "approxMode", fallback.numerics().approxMode(), ApproxMode.class),
                            findBoolean(json, "forceExactTranscendentals", fallback.numerics().forceExactTranscendentals())
                    ),
                    new AcceleratorPlatformProfile(
                            new AcceleratorBackendPlatformProfile(
                                    findBoolean(json, "cudaEnabled", fallback.accelerator().cuda().enabled()),
                                    findBoolean(
                                            json,
                                            "cudaRequireRuntimeAvailability",
                                            fallback.accelerator().cuda().requireRuntimeAvailability()
                                    ),
                                    findLong(
                                            json,
                                            "cudaMinimumEstimatedWork",
                                            fallback.accelerator().cuda().minimumEstimatedWork()
                                    ),
                                    acceleratorBufferConfig(json, "cuda", fallback.accelerator().cuda().buffer())
                            ),
                            new AcceleratorBackendPlatformProfile(
                                    findBoolean(json, "openclEnabled", fallback.accelerator().opencl().enabled()),
                                    findBoolean(
                                            json,
                                            "openclRequireRuntimeAvailability",
                                            fallback.accelerator().opencl().requireRuntimeAvailability()
                                    ),
                                    findLong(
                                            json,
                                            "openclMinimumEstimatedWork",
                                            fallback.accelerator().opencl().minimumEstimatedWork()
                                    ),
                                    acceleratorBufferConfig(json, "opencl", fallback.accelerator().opencl().buffer())
                            ),
                            new AcceleratorBackendPlatformProfile(
                                    findBoolean(json, "metalEnabled", fallback.accelerator().metal().enabled()),
                                    findBoolean(
                                            json,
                                            "metalRequireRuntimeAvailability",
                                            fallback.accelerator().metal().requireRuntimeAvailability()
                                    ),
                                    findLong(
                                            json,
                                            "metalMinimumEstimatedWork",
                                            fallback.accelerator().metal().minimumEstimatedWork()
                                    ),
                                    acceleratorBufferConfig(json, "metal", fallback.accelerator().metal().buffer())
                            )
                    ),
                    findEnum(json, "cpuStorageProfile", fallback.cpuStorageProfile(), CpuStorageProfile.class),
                    findEnum(
                            json,
                            "nativeCpuFailurePolicy",
                            fallback.nativeCpuFailurePolicy(),
                            NativeCpuFailurePolicy.class
                    ),
                    findEnum(
                            json,
                            "deviceTransferPolicy",
                            fallback.deviceTransferPolicy(),
                            DeviceTransferPolicy.class
                    )
            );
        } catch (Exception e) {
            return fallback;
        }
    }

    public static PlatformRuntimeProfile fromJsonStrict(String json, PlatformRuntimeProfile fallback) {
        if (json == null || json.isBlank()) {
            throw new IllegalArgumentException("Platform runtime profile JSON is blank");
        }
        if (fallback == null) {
            throw new IllegalArgumentException("fallback cannot be null");
        }
        validateSupportedSchemaVersions(json, fallback.metadata());
        validateBufferBindingMode(json, "cudaBufferBindingMode");
        validateBufferBindingMode(json, "openclBufferBindingMode");
        validateBufferBindingMode(json, "metalBufferBindingMode");
        return fromJsonOrDefault(json, fallback);
    }

    private static String findString(String json, String key, String fallback) {
        int idx = json.indexOf("\"" + key + "\"");
        if (idx < 0) {
            return fallback;
        }
        int firstQuote = json.indexOf('"', json.indexOf(':', idx) + 1);
        int secondQuote = json.indexOf('"', firstQuote + 1);
        return firstQuote < 0 || secondQuote < 0 ? fallback : json.substring(firstQuote + 1, secondQuote);
    }

    private static AcceleratorBufferConfig acceleratorBufferConfig(
            String json,
            String backendPrefix,
            AcceleratorBufferConfig fallback
    ) {
        AcceleratorBufferConfig resolved = fallback == null ? AcceleratorBufferConfig.defaults() : fallback;
        String prefix = backendPrefix == null || backendPrefix.isBlank()
                ? ""
                : backendPrefix.substring(0, 1).toLowerCase(java.util.Locale.ROOT) + backendPrefix.substring(1);
        return new AcceleratorBufferConfig(
                findEnum(json, prefix + "BufferBindingMode", resolved.bindingMode(), AcceleratorBufferBindingMode.class),
                findBoolean(
                        json,
                        prefix + "AllowPreparedInputMaterialization",
                        resolved.allowPreparedInputMaterialization()
                ),
                findLong(json, prefix + "BufferMinimumEstimatedWork", resolved.minimumEstimatedWork())
        );
    }

    private static void validateSupportedSchemaVersions(String json, PlatformProfileMetadata fallback) {
        String plannerSchemaVersion = findString(json, "plannerSchemaVersion", fallback.plannerSchemaVersion());
        if (!SUPPORTED_PLANNER_SCHEMA_VERSION.equals(plannerSchemaVersion)) {
            throw new IllegalArgumentException(
                    "Unsupported plannerSchemaVersion: " + plannerSchemaVersion
            );
        }
        String persistenceSchemaVersion = findString(
                json,
                "persistenceSchemaVersion",
                fallback.persistenceSchemaVersion()
        );
        if (!SUPPORTED_PERSISTENCE_SCHEMA_VERSION.equals(persistenceSchemaVersion)) {
            throw new IllegalArgumentException(
                    "Unsupported persistenceSchemaVersion: " + persistenceSchemaVersion
            );
        }
    }

    private static void validateBufferBindingMode(String json, String key) {
        if (!hasKey(json, key)) {
            return;
        }
        String value = findString(json, key, null);
        try {
            AcceleratorBufferBindingMode.valueOf(value);
        } catch (RuntimeException e) {
            throw new IllegalArgumentException("Invalid " + key + ": " + value, e);
        }
    }

    private static boolean hasKey(String json, String key) {
        return json.indexOf("\"" + key + "\"") >= 0;
    }

    private static int findInt(String json, String key, int fallback) {
        try {
            return (int) Math.round(findDouble(json, key, fallback));
        } catch (Exception e) {
            return fallback;
        }
    }

    private static long findLong(String json, String key, long fallback) {
        try {
            return Math.round(findDouble(json, key, fallback));
        } catch (Exception e) {
            return fallback;
        }
    }

    private static double findDouble(String json, String key, double fallback) {
        int idx = json.indexOf("\"" + key + "\"");
        if (idx < 0) {
            return fallback;
        }
        int colon = json.indexOf(':', idx);
        if (colon < 0) {
            return fallback;
        }
        int end = colon + 1;
        while (end < json.length() && " \n\r\t".indexOf(json.charAt(end)) >= 0) {
            end++;
        }
        int stop = end;
        while (stop < json.length() && ",}\n\r\t ".indexOf(json.charAt(stop)) < 0) {
            stop++;
        }
        return Double.parseDouble(json.substring(end, stop));
    }

    private static boolean findBoolean(String json, String key, boolean fallback) {
        int idx = json.indexOf("\"" + key + "\"");
        if (idx < 0) {
            return fallback;
        }
        int colon = json.indexOf(':', idx);
        if (colon < 0) {
            return fallback;
        }
        String tail = json.substring(colon + 1).trim();
        if (tail.startsWith("true")) {
            return true;
        }
        if (tail.startsWith("false")) {
            return false;
        }
        return fallback;
    }

    private static <E extends Enum<E>> E findEnum(String json, String key, E fallback, Class<E> type) {
        try {
            return Enum.valueOf(type, findString(json, key, fallback.name()));
        } catch (Exception e) {
            return fallback;
        }
    }

    private static String escape(String value) {
        return value == null ? "" : value.replace("\\", "\\\\").replace("\"", "\\\"");
    }
}
