package tensor;

import backend.ComputeBackend;
import backend.runtime.ExecutionMode;
import config.optimizer.OptimizerConfig;
import config.profile.ExecutionProfile;
import config.profile.ExecutionProfileIO;
import config.profile.WorkloadProfile;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import graph.execution.PreparedExecution;
import tuning.candidate.ProfileGridCandidateSpace;
import tuning.candidate.ProfileMutators;
import tuning.session.AutotuneSession;
import tuning.session.TuningDefaults;
import tuning.store.FileBestProfileResolver;
import tuning.store.HardwareFingerprint;
import tuning.store.JsonFileBestProfileStore;
import tuning.store.PersistencePolicy;
import tuning.store.PlatformCalibrationPaths;
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
import java.util.Objects;

final class TensorExecutionSupport {
    private TensorExecutionSupport() {
    }

    static ComputeBackend resolveBackend(ComputeBackend forcedBackend) {
        return forcedBackend != null ? forcedBackend : ComputeBackend.CPU;
    }

    static PreparedExecution prepare(Tensor tensor, ExecutionProfile profile) {
        if (profile == null) {
            throw new IllegalArgumentException("profile cannot be null");
        }
        return CompiledGraph.compile(tensor, profile.optimizer(), compileModeForProfile(profile)).prepare(profile.runtime());
    }

    static void compute(Tensor tensor, ExecutionProfile profile) {
        if (profile == null) {
            throw new IllegalArgumentException("profile cannot be null");
        }
        compute(prepare(tensor, profile), profile.mode());
    }

    static CompiledGraph compile(Tensor tensor, CompileMode compileMode) {
        Objects.requireNonNull(tensor, "tensor cannot be null");
        CompileMode effectiveMode = compileMode == null ? CompileMode.INFERENCE_ONLY : compileMode;
        return CompiledGraph.compile(tensor, defaultOptimizer(tensor, effectiveMode), effectiveMode);
    }

    static Tensor compute(Tensor tensor) {
        return compute(tensor, CompileMode.INFERENCE_ONLY);
    }

    static Tensor compute(Tensor tensor, CompileMode compileMode) {
        return compute(tensor, new ComputeOptions().compileMode(compileMode));
    }

    static Tensor compute(Tensor tensor, ComputeOptions options) {
        Objects.requireNonNull(tensor, "tensor cannot be null");
        ComputeOptions effectiveOptions = options == null ? new ComputeOptions() : options;
        ExecutionProfile profile = resolveProfile(tensor, effectiveOptions);
        PreparedExecution prepared = prepare(tensor, profile);
        compute(prepared, profile.mode());
        return tensor;
    }

    static void compute(PreparedExecution execution, ExecutionMode mode) {
        if (execution == null) {
            throw new IllegalArgumentException("execution cannot be null");
        }
        execution.execute(mode);
    }

    private static ExecutionProfile resolveProfile(Tensor tensor, ComputeOptions options) {
        if (options.autotunePolicy() == null || options.autotunePolicy() == AutotunePolicy.NEVER) {
            return defaultProfile(tensor, options);
        }
        return resolveAutotunedProfile(tensor, options);
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
        ProfileGridCandidateSpace candidateSpace = new ProfileGridCandidateSpace(
                seed,
                java.util.List.of(ProfileMutators.constrainedStageOrderSpace())
        );
        var request = TuningDefaults.balancedAutotune(workload, seed, candidateSpace, persistence);
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
        OptimizerConfig optimizer = options.optimizer() == null ? defaultOptimizer(tensor, compileMode) : options.optimizer();
        RuntimeConfig runtime = options.runtime() == null ? defaultRuntime(tensor, compileMode) : options.runtime();
        ExecutionMode executionMode = defaultExecutionMode(tensor, compileMode);
        String name = "tensor-compute-" + dtypeId(tensor.getDataType()) + "-" + modeId(executionMode);
        return new ExecutionProfile(
                name,
                name,
                tensor.getDataType(),
                executionMode,
                optimizer,
                runtime,
                WorkloadProfile.none()
        );
    }

    private static OptimizerConfig defaultOptimizer(Tensor tensor, CompileMode compileMode) {
        return switch (compileMode == null ? CompileMode.INFERENCE_ONLY : compileMode) {
            case INFERENCE_ONLY -> OptimizerConfig.inferenceDefaults();
            case TRAINING -> OptimizerConfig.trainingDefaults();
            case AUTO -> hasTrainableLeafInputs(tensor) ? OptimizerConfig.trainingDefaults() : OptimizerConfig.inferenceDefaults();
        };
    }

    private static RuntimeConfig defaultRuntime(Tensor tensor, CompileMode compileMode) {
        return switch (compileMode == null ? CompileMode.INFERENCE_ONLY : compileMode) {
            case INFERENCE_ONLY -> RuntimeConfig.inferenceDefaults();
            case TRAINING -> RuntimeConfig.trainingDefaults();
            case AUTO -> hasTrainableLeafInputs(tensor) ? RuntimeConfig.trainingDefaults() : RuntimeConfig.inferenceDefaults();
        };
    }

    private static ExecutionMode defaultExecutionMode(Tensor tensor, CompileMode compileMode) {
        return switch (compileMode == null ? CompileMode.INFERENCE_ONLY : compileMode) {
            case INFERENCE_ONLY -> ExecutionMode.FORWARD;
            case TRAINING -> hasTrainableLeafInputs(tensor) ? ExecutionMode.FORWARD_BACKWARD : ExecutionMode.FORWARD;
            case AUTO -> hasTrainableLeafInputs(tensor) ? ExecutionMode.FORWARD_BACKWARD : ExecutionMode.FORWARD;
        };
    }

    private static CompileMode compileModeForProfile(ExecutionProfile profile) {
        return profile != null && profile.mode() == ExecutionMode.FORWARD_BACKWARD
                ? CompileMode.TRAINING
                : CompileMode.INFERENCE_ONLY;
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
            case BOOL -> "bool";
        };
    }

    private static String modeId(ExecutionMode mode) {
        return mode.name().toLowerCase(Locale.ROOT).replace('_', '-');
    }
}
