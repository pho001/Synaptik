import backend.runtime.ExecutionMode;
import config.profile.ExecutionProfile;
import config.profile.WorkloadProfile;
import org.junit.jupiter.api.Test;
import tuning.candidate.Candidate;
import tuning.session.TuningPreset;
import tuning.validate.ValidationReferenceKind;
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
                        tensor.Conv2dOptions.defaults().withPadding(1, 1),
                        true
                )
                .instantiate(new WorkloadEnvironment(profile));
        var norm = StandardWorkloads.normalization(
                        "layer_norm_test",
                        tuning.workload.NormalizationWorkloadSpec.NormalizationKind.LAYER_NORM,
                        2, 16, 4, 1, 1e-5
                )
                .instantiate(new WorkloadEnvironment(profile));
        var pool = StandardWorkloads.pool2d(
                        "pool_test",
                        tuning.workload.Pool2dWorkloadSpec.PoolKind.MAX,
                        1, 4, 8, 8,
                        tensor.Pool2dOptions.square(2)
                )
                .instantiate(new WorkloadEnvironment(profile));
        var loss = StandardWorkloads.indexedLoss(
                        "loss_test",
                        tuning.workload.LossWorkloadSpec.LossKind.CROSS_ENTROPY_FROM_INDICES,
                        4, 8,
                        tensor.LossReduction.MEAN
                )
                .instantiate(new WorkloadEnvironment(profile));

        assertEquals(ValidationReferenceKind.BASELINE_PROFILE, matmul.reference().kind());
        assertEquals(ValidationReferenceKind.BASELINE_PROFILE, conv.reference().kind());
        assertEquals(ValidationReferenceKind.BASELINE_PROFILE, norm.reference().kind());
        assertEquals(ValidationReferenceKind.BASELINE_PROFILE, pool.reference().kind());
        assertEquals(ValidationReferenceKind.BASELINE_PROFILE, loss.reference().kind());
    }

    @Test
    void standardWorkloadsSupportsPresetDrivenBenchmarkRequests() {
        Candidate candidate = new Candidate("preset", new ExecutionProfile(
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
        assertEquals(2, request.measurement().warmupIters());
        assertTrue(request.baselines().includeNoOptBaseline());
    }
}
