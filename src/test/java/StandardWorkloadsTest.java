import backend.runtime.ExecutionMode;
import config.profile.ExecutionProfile;
import config.profile.WorkloadProfile;
import org.junit.jupiter.api.Test;
import tuning.autotune.TuningDefaults;
import tuning.benchmark.BenchmarkEntry;
import tuning.preset.TuningPreset;
import tuning.validate.ValidationReferenceKind;
import tuning.validate.ValidationTargetKind;
import tuning.workload.StandardWorkloads;
import tuning.workload.WorkloadCatalog;
import tuning.workload.WorkloadEnvironment;
import tuning.workload.WorkloadInstance;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class StandardWorkloadsTest {
    @Test
    void defaultCatalogExposesExpectedNamedWorkloads() {
        WorkloadCatalog catalog = StandardWorkloads.defaultCatalog();
        assertTrue(catalog.names().contains("matmul_small"));
        assertTrue(catalog.names().contains("matmul_batched_attention_like"));
        assertTrue(catalog.names().contains("abc_sequence_matmul_small"));
        assertTrue(catalog.names().contains("mlp_classifier_small"));
        assertTrue(catalog.names().contains("mlp_classifier_blas_heavy"));
        assertTrue(catalog.names().contains("conv2d_resnet_3x3"));
        assertTrue(catalog.names().contains("layer_norm_small"));
        assertTrue(catalog.names().contains("max_pool2d_small"));
        assertTrue(catalog.names().contains("cross_entropy_small"));
        assertTrue(catalog.names().contains("transformer_hot_path"));
    }

    @Test
    void transformerHotPathWorkloadUsesProfileWorkloadMetadata() {
        ExecutionProfile profile = new ExecutionProfile(
                "transformer-profile",
                "transformer-profile",
                tensor.DataType.FLOAT32,
                ExecutionMode.FORWARD,
                config.optimizer.OptimizerConfig.inferenceDefaults(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                StandardWorkloads.transformerHotPathDefaults()
        );

        WorkloadInstance instance = StandardWorkloads.transformerHotPath("transformer_hot_path")
                .instantiate(new WorkloadEnvironment(profile));

        assertEquals("transformer_hot_path", instance.metadata().name());
        assertEquals(tuning.workload.WorkloadKind.TRANSFORMER_HOT_PATH, instance.metadata().kind());
        assertEquals(1, instance.root().getShapeUnsafe().length);
        assertEquals(ValidationTargetKind.LABEL, instance.validationTarget().kind());
        assertEquals("rmsNorm", instance.validationTarget().label());
        assertEquals(ValidationReferenceKind.BASELINE_PROFILE, instance.reference().kind());
    }

    @Test
    void matmulAndConv2dWorkloadsProvideBaselineValidationReference() {
        ExecutionProfile profile = new ExecutionProfile(
                "workload-validation",
                "workload-validation",
                tensor.DataType.FLOAT64,
                ExecutionMode.FORWARD,
                config.optimizer.OptimizerConfig.inferenceDefaults(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );

        var matmul = StandardWorkloads.matmul("matmul_small_test", 1, 8, 8, 8)
                .instantiate(new WorkloadEnvironment(profile));
        var conv = StandardWorkloads.conv2d(
                        "conv_small_test",
                        1, 4, 4, 8, 8, 3, 3,
                        tensor.options.Conv2dOptions.defaults().withPadding(1, 1),
                        true
                )
                .instantiate(new WorkloadEnvironment(profile));
        var abc = StandardWorkloads.abcSequenceMatmul("abc_test", 16, 32)
                .instantiate(new WorkloadEnvironment(profile));
        var norm = StandardWorkloads.normalization(
                        "layer_norm_test",
                        tuning.workload.NormalizationWorkloadSpec.NormalizationKind.LAYER_NORM,
                        2, 16, 4, 1, 1e-5
                )
                .instantiate(new WorkloadEnvironment(profile));
        var mlp = StandardWorkloads.mlpClassification(
                        "mlp_test",
                        8, 16, 24, 12, 4,
                        tensor.loss.LossReduction.MEAN
                )
                .instantiate(new WorkloadEnvironment(profile));
        var pool = StandardWorkloads.pool2d(
                        "pool_test",
                        tuning.workload.Pool2dWorkloadSpec.PoolKind.MAX,
                        1, 4, 8, 8,
                        tensor.options.Pool2dOptions.square(2)
                )
                .instantiate(new WorkloadEnvironment(profile));
        var loss = StandardWorkloads.indexedLoss(
                        "loss_test",
                        tuning.workload.LossWorkloadSpec.LossKind.CROSS_ENTROPY_FROM_INDICES,
                        4, 8,
                        tensor.loss.LossReduction.MEAN
                )
                .instantiate(new WorkloadEnvironment(profile));

        assertEquals(ValidationReferenceKind.BASELINE_PROFILE, matmul.reference().kind());
        assertEquals(ValidationReferenceKind.BASELINE_PROFILE, conv.reference().kind());
        assertEquals(ValidationReferenceKind.BASELINE_PROFILE, abc.reference().kind());
        assertEquals(ValidationReferenceKind.BASELINE_PROFILE, mlp.reference().kind());
        assertEquals(ValidationReferenceKind.BASELINE_PROFILE, norm.reference().kind());
        assertEquals(ValidationReferenceKind.BASELINE_PROFILE, pool.reference().kind());
        assertEquals(ValidationReferenceKind.BASELINE_PROFILE, loss.reference().kind());

        assertEquals(ValidationTargetKind.LABEL, matmul.validationTarget().kind());
        assertEquals("matmul", matmul.validationTarget().label());
        assertEquals(ValidationTargetKind.LABEL, conv.validationTarget().kind());
        assertEquals("conv2d", conv.validationTarget().label());
        assertEquals(ValidationTargetKind.LABEL, norm.validationTarget().kind());
        assertEquals("layerNorm", norm.validationTarget().label());
        assertEquals(ValidationTargetKind.LABEL, pool.validationTarget().kind());
        assertEquals("maxPool2d", pool.validationTarget().label());
        assertEquals(ValidationTargetKind.ROOT, loss.validationTarget().kind());
    }

    @Test
    void standardWorkloadsSupportsPresetDrivenBenchmarkRequests() {
        BenchmarkEntry candidate = BenchmarkEntry.candidate("preset", new ExecutionProfile(
                "preset",
                "preset",
                tensor.DataType.FLOAT64,
                ExecutionMode.FORWARD,
                config.optimizer.OptimizerConfig.noOptimization(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        ));

        var request = StandardWorkloads.benchmark("matmul_small", java.util.List.of(candidate), TuningPreset.BALANCED);

        assertEquals("matmul_small", request.workload().name());
        assertEquals(TuningDefaults.balancedMeasurement().warmupIters(), request.measurement().warmupIters());
        assertEquals(1, request.entries().size());
    }

    @Test
    void standardWorkloadsSupportsRecommendedWorkloadAwarePresets() {
        BenchmarkEntry candidate = BenchmarkEntry.candidate("preset", new ExecutionProfile(
                "preset",
                "preset",
                tensor.DataType.FLOAT64,
                ExecutionMode.FORWARD,
                config.optimizer.OptimizerConfig.noOptimization(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        ));

        var benchmarkRequest = StandardWorkloads.benchmark("cross_entropy_small", java.util.List.of(candidate));
        assertEquals("cross_entropy_small", benchmarkRequest.workload().name());
        assertTrue(benchmarkRequest.validation().requireGradientMatch());
    }
}
