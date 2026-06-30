package tuning.tensor;

import config.compile.CompileConfig;
import config.profile.ExecutionProfile;
import config.profile.ExecutionProfileIO;
import config.profile.GraphExecutionPolicy;
import config.profile.PlatformRuntimeProfile;
import config.profile.WorkloadProfile;
import config.runtime.RuntimeConfig;
import runtime.contract.ExecutionMode;
import tensor.AutotunePolicy;
import tensor.CompileMode;
import tensor.ComputeOptions;
import tensor.DataType;
import tensor.Tensor;
import tuning.autotune.AutotuneSession;
import tuning.autotune.GraphAutotuneMode;
import tuning.autotune.GraphAutotuneRequest;
import tuning.calibration.store.PlatformCalibrationPaths;
import tuning.preset.TuningPreset;
import tuning.search.SearchPolicy;
import tuning.store.FileBestProfileResolver;
import tuning.store.HardwareFingerprint;
import tuning.store.JsonFileBestProfileStore;
import tuning.store.PersistencePolicy;
import tuning.store.WorkloadFingerprint;
import tuning.workload.TensorRootWorkloadSpec;
import tuning.workload.WorkloadEnvironment;
import tuning.workload.WorkloadKind;
import tuning.workload.WorkloadMetadata;

import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;

public final class TensorComputeProfileResolver {
    private TensorComputeProfileResolver() {
    }

    public static ExecutionProfile resolve(Tensor tensor, ComputeOptions options) {
        if (options.autotunePolicy() == null || options.autotunePolicy() == AutotunePolicy.NEVER) {
            return defaultProfile(tensor, options);
        }
        return resolveAutotunedProfile(tensor, options);
    }

    public static CompileConfig defaultCompile(Tensor tensor, CompileMode compileMode) {
        return CompileConfig.defaultsForCompileMode(
                compileMode == null ? CompileMode.INFERENCE_ONLY : compileMode,
                hasTrainableLeafInputs(tensor)
        );
    }

    private static ExecutionProfile resolveAutotunedProfile(Tensor tensor, ComputeOptions options) {
        ExecutionProfile seed = defaultProfile(tensor, options);
        String graphSignature = graphSignature(tensor);
        String seedSignature = profileSignature(seed);
        String workloadName = genericWorkloadName(tensor);
        HardwareFingerprint hardware = HardwareFingerprint.capture();
        Path bestProfilePath = tensorBestProfilePath(seed, graphSignature, seedSignature, hardware);
        Path historyPath = tensorHistoryPath(seed, graphSignature, seedSignature, hardware);

        TensorRootWorkloadSpec workload = genericTensorWorkload(tensor, workloadName, graphSignature, seedSignature);
        WorkloadFingerprint fingerprint = workloadFingerprint(workload, seed);
        FileBestProfileResolver resolver = new FileBestProfileResolver(new JsonFileBestProfileStore());

        if (options.autotunePolicy() == AutotunePolicy.IF_MISSING) {
            var existing = resolver.resolve(bestProfilePath, hardware, fingerprint);
            if (existing.isPresent()) {
                return existing.get();
            }
        }

        PersistencePolicy persistence = new PersistencePolicy(true, true, bestProfilePath, historyPath);
        var request = new GraphAutotuneRequest(
                workload,
                seed.profileName(),
                seed.dataType(),
                seed.mode(),
                GraphExecutionPolicy.fromExecutionProfile(seed),
                PlatformRuntimeProfile.fromExecutionProfile(seed.profileName(), hardware.key(), "TENSOR_SEED", seed),
                GraphAutotuneMode.STANDARD,
                TuningPreset.BALANCED.autotuneMeasurement(),
                TuningPreset.BALANCED.autotuneValidation(),
                new SearchPolicy(1, 1, 1, false),
                persistence,
                null
        ).toAutotuneRequest();
        var result = AutotuneSession.create(request).run();
        if (result.bestProfile() != null) {
            return result.bestProfile();
        }
        return resolver.resolve(bestProfilePath, hardware, fingerprint).orElse(seed);
    }

    private static WorkloadFingerprint workloadFingerprint(TensorRootWorkloadSpec workload, ExecutionProfile seed) {
        var instance = workload.instantiate(new WorkloadEnvironment(seed));
        return WorkloadFingerprint.of(workload, instance.metadata(), seed);
    }

    private static TensorRootWorkloadSpec genericTensorWorkload(
            Tensor tensor,
            String workloadName,
            String graphSignature,
            String seedSignature
    ) {
        return new TensorRootWorkloadSpec(
                workloadName,
                WorkloadKind.GENERIC,
                environment -> tensor,
                environment -> tuning.validate.ValidationReference.none(),
                environment -> genericWorkloadMetadata(tensor, workloadName, graphSignature, seedSignature)
        );
    }

    private static WorkloadMetadata genericWorkloadMetadata(
            Tensor tensor,
            String workloadName,
            String graphSignature,
            String seedSignature
    ) {
        Map<String, Object> attributes = new LinkedHashMap<>();
        attributes.put("graphSignature", graphSignature);
        attributes.put("seedSignature", seedSignature);
        attributes.put("rootLabel", tensor.getLabel() == null ? "" : tensor.getLabel());
        attributes.put("shape", Arrays.toString(tensor.getShapeUnsafe()));
        attributes.put("nodeCount", tensor.topologicalSort().size());
        attributes.put("requiresGradLeaves", hasTrainableLeafInputs(tensor));
        return new WorkloadMetadata(workloadName, WorkloadKind.GENERIC, attributes);
    }

    private static Path tensorBestProfilePath(
            ExecutionProfile seed,
            String graphSignature,
            String seedSignature,
            HardwareFingerprint hardware
    ) {
        String platformId = PlatformCalibrationPaths.platformId(hardware);
        String variant = dtypeId(seed.dataType()) + "-" + modeId(seed.mode());
        return Path.of("build", "tuning", "tensor", platformId, graphSignature, seedSignature)
                .resolve(variant + "-best-profile.json");
    }

    private static Path tensorHistoryPath(
            ExecutionProfile seed,
            String graphSignature,
            String seedSignature,
            HardwareFingerprint hardware
    ) {
        String platformId = PlatformCalibrationPaths.platformId(hardware);
        String variant = dtypeId(seed.dataType()) + "-" + modeId(seed.mode());
        return Path.of("build", "tuning", "tensor", platformId, graphSignature, seedSignature)
                .resolve(variant + "-history.jsonl");
    }

    private static ExecutionProfile defaultProfile(Tensor tensor, ComputeOptions options) {
        CompileMode compileMode = options.compileMode() == null ? CompileMode.INFERENCE_ONLY : options.compileMode();
        CompileConfig compile = options.compile() != null ? options.compile() : defaultCompile(tensor, compileMode);
        RuntimeConfig runtime = options.runtime() == null ? defaultRuntime(tensor, compileMode) : options.runtime();
        ExecutionMode executionMode = defaultExecutionMode(tensor, compileMode);
        String name = "tensor-compute-" + dtypeId(tensor.getDataType()) + "-" + modeId(executionMode);
        return new ExecutionProfile(
                name,
                name,
                tensor.getDataType(),
                executionMode,
                compile,
                runtime,
                WorkloadProfile.none()
        );
    }

    private static RuntimeConfig defaultRuntime(Tensor tensor, CompileMode compileMode) {
        return switch (compileMode == null ? CompileMode.INFERENCE_ONLY : compileMode) {
            case INFERENCE_ONLY -> RuntimeConfig.inferenceDefaults(tensor.getDataType());
            case TRAINING -> RuntimeConfig.trainingDefaults(tensor.getDataType());
            case AUTO -> hasTrainableLeafInputs(tensor)
                    ? RuntimeConfig.trainingDefaults(tensor.getDataType())
                    : RuntimeConfig.inferenceDefaults(tensor.getDataType());
        };
    }

    private static ExecutionMode defaultExecutionMode(Tensor tensor, CompileMode compileMode) {
        return switch (compileMode == null ? CompileMode.INFERENCE_ONLY : compileMode) {
            case INFERENCE_ONLY -> ExecutionMode.FORWARD;
            case TRAINING -> hasTrainableLeafInputs(tensor) ? ExecutionMode.FORWARD_BACKWARD : ExecutionMode.FORWARD;
            case AUTO -> hasTrainableLeafInputs(tensor) ? ExecutionMode.FORWARD_BACKWARD : ExecutionMode.FORWARD;
        };
    }

    private static boolean hasTrainableLeafInputs(Tensor tensor) {
        for (Tensor node : tensor.forwardOutput().topologicalSort()) {
            if (node.getOperation() == null && node.getRequiresGrad()) {
                return true;
            }
        }
        return false;
    }

    private static String genericWorkloadName(Tensor tensor) {
        String label = tensor.getLabel();
        if (label == null || label.isBlank()) {
            return "tensor_root";
        }
        return "tensor_" + label.trim().toLowerCase(Locale.ROOT).replaceAll("[^a-z0-9]+", "_");
    }

    private static String graphSignature(Tensor tensor) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            Map<Tensor, Integer> nodeIds = new java.util.IdentityHashMap<>();
            var nodes = tensor.forwardOutput().topologicalSort();
            for (int i = 0; i < nodes.size(); i++) {
                nodeIds.put(nodes.get(i), i);
            }
            for (int i = 0; i < nodes.size(); i++) {
                Tensor node = nodes.get(i);
                updateDigest(digest, Integer.toString(i));
                updateDigest(digest, node.getOperation() == null ? "LEAF" : node.getOperation().opType().name());
                updateDigest(digest, node.getOperation() == null ? "" : node.getOperation().getClass().getName());
                updateDigest(digest, node.getOperation() == null ? "" : node.getOperation().getExpression());
                updateDigest(digest, node.getDataType().name());
                updateDigest(digest, Arrays.toString(node.getShapeUnsafe()));
                updateDigest(digest, Integer.toString(node.getPrevTensors().size()));
                for (Tensor input : node.getPrevTensors()) {
                    Integer inputId = nodeIds.get(input);
                    updateDigest(digest, inputId == null ? "?" : inputId.toString());
                }
            }
            return hex(digest.digest()).substring(0, 24);
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 is not available", e);
        }
    }

    private static String profileSignature(ExecutionProfile profile) {
        String json = ExecutionProfileIO.toJson(profile);
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            updateDigest(digest, json);
            return hex(digest.digest()).substring(0, 16);
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 is not available", e);
        }
    }

    private static void updateDigest(MessageDigest digest, String value) {
        digest.update((value == null ? "" : value).getBytes(java.nio.charset.StandardCharsets.UTF_8));
        digest.update((byte) '\n');
    }

    private static String hex(byte[] data) {
        StringBuilder sb = new StringBuilder(data.length * 2);
        for (byte b : data) {
            sb.append(Character.forDigit((b >>> 4) & 0xF, 16));
            sb.append(Character.forDigit(b & 0xF, 16));
        }
        return sb.toString();
    }

    private static String dtypeId(DataType dataType) {
        return switch (dataType) {
            case FLOAT64 -> "f64";
            case FLOAT32 -> "f32";
            case BFLOAT16 -> "bf16";
            case INT32 -> "i32";
            case INT64 -> "i64";
            case BOOL -> "bool";
        };
    }

    private static String modeId(ExecutionMode mode) {
        return mode.name().toLowerCase(Locale.ROOT).replace('_', '-');
    }
}
