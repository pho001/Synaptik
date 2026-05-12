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

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

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
        assertTrue(catalog.names().contains("mlp_classifier_small_bf16"));
        assertTrue(catalog.names().contains("mlp_classifier_blas_heavy"));
        assertTrue(catalog.names().contains("reduction_chain_small"));
        assertTrue(catalog.names().contains("reduction_chain_small_bf16"));
        assertTrue(catalog.names().contains("conv2d_resnet_3x3"));
        assertTrue(catalog.names().contains("layer_norm_small"));
        assertTrue(catalog.names().contains("layer_norm_small_bf16"));
        assertTrue(catalog.names().contains("rms_norm_small"));
        assertTrue(catalog.names().contains("rms_norm_small_bf16"));
        assertTrue(catalog.names().contains("max_pool2d_small"));
        assertTrue(catalog.names().contains("avg_pool2d_small"));
        assertTrue(catalog.names().contains("dense_loss_small"));
        assertTrue(catalog.names().contains("cross_entropy_small"));
        assertTrue(catalog.names().contains("training_transformer_block_hot_path"));
        assertTrue(catalog.names().contains("training_dense_loss_small"));
        assertTrue(catalog.names().contains("training_reduction_chain_small"));
        assertTrue(catalog.names().contains("training_layer_norm_small"));
        assertTrue(catalog.names().contains("training_cross_entropy_small"));
        assertTrue(catalog.names().contains("bool_compare_where_small"));
        assertTrue(catalog.names().contains("gather_take_small"));
        assertTrue(catalog.names().contains("scatter_index_gradient_small"));
        assertTrue(catalog.names().contains("layout_broadcast_repair_small"));
        assertTrue(catalog.names().contains("transformer_hot_path"));
        assertTrue(catalog.names().contains("transformer_block_hot_path"));
    }

    @Test
    void layoutRepairWorkloadInstantiatesBroadcastContiguousGraph() {
        ExecutionProfile profile = new ExecutionProfile(
                "layout-repair-profile",
                "layout-repair-profile",
                tensor.DataType.FLOAT32,
                ExecutionMode.FORWARD,
                config.compile.CompileConfig.inference(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );

        WorkloadInstance instance = StandardWorkloads.layoutBroadcastRepair("layout_broadcast_repair_small", 4, 8)
                .instantiate(new WorkloadEnvironment(profile));

        assertEquals("layout_broadcast_repair_small", instance.metadata().name());
        assertEquals(tuning.workload.WorkloadKind.GENERIC, instance.metadata().kind());
        assertEquals(2, instance.root().getShapeUnsafe().length);
        assertEquals(ValidationTargetKind.ROOT, instance.validationTarget().kind());
        assertEquals(ValidationReferenceKind.BASELINE_PROFILE, instance.reference().kind());
    }

    @Test
    void gatherTakeWorkloadInstantiatesIndexForwardGraph() {
        ExecutionProfile profile = new ExecutionProfile(
                "gather-take-profile",
                "gather-take-profile",
                tensor.DataType.FLOAT32,
                ExecutionMode.FORWARD,
                config.compile.CompileConfig.inference(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );

        WorkloadInstance instance = StandardWorkloads.gatherTake("gather_take_small", 4, 8, 2)
                .instantiate(new WorkloadEnvironment(profile));

        assertEquals("gather_take_small", instance.metadata().name());
        assertEquals(tuning.workload.WorkloadKind.GENERIC, instance.metadata().kind());
        assertEquals(1, instance.root().getShapeUnsafe().length);
        assertEquals(ValidationTargetKind.ROOT, instance.validationTarget().kind());
        assertEquals(ValidationReferenceKind.BASELINE_PROFILE, instance.reference().kind());
    }

    @Test
    void scatterIndexGradientWorkloadInstantiatesExplicitIndexWriteAndGradientGraph() {
        ExecutionProfile profile = new ExecutionProfile(
                "scatter-index-gradient-profile",
                "scatter-index-gradient-profile",
                tensor.DataType.FLOAT32,
                ExecutionMode.FORWARD,
                config.compile.CompileConfig.inference(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );

        WorkloadInstance instance = StandardWorkloads.scatterIndexGradient("scatter_index_gradient_small", 4, 8, 2)
                .instantiate(new WorkloadEnvironment(profile));

        assertEquals("scatter_index_gradient_small", instance.metadata().name());
        assertEquals(tuning.workload.WorkloadKind.GENERIC, instance.metadata().kind());
        assertEquals(1, instance.root().getShapeUnsafe().length);
        assertEquals(ValidationTargetKind.ROOT, instance.validationTarget().kind());
        assertEquals(ValidationReferenceKind.BASELINE_PROFILE, instance.reference().kind());
        List<operations.Operation.OpType> opTypes = instance.root().topologicalSort().stream()
                .map(tensor.Tensor::getOperation)
                .filter(java.util.Objects::nonNull)
                .map(operations.Operation::opType)
                .toList();
        assertTrue(opTypes.contains(operations.Operation.OpType.SCATTER_ADD));
        assertTrue(opTypes.contains(operations.Operation.OpType.GATHER_GRAD));
        assertTrue(opTypes.contains(operations.Operation.OpType.TAKE_ALONG_AXIS_GRAD));
    }

    @Test
    void denseLossWorkloadInstantiatesDenseNllAndCrossEntropyGraph() {
        ExecutionProfile profile = new ExecutionProfile(
                "dense-loss-profile",
                "dense-loss-profile",
                tensor.DataType.FLOAT32,
                ExecutionMode.FORWARD,
                config.compile.CompileConfig.inference(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );

        WorkloadInstance instance = StandardWorkloads.denseLoss("dense_loss_small", 4, 8, tensor.loss.LossReduction.MEAN)
                .instantiate(new WorkloadEnvironment(profile));

        assertEquals("dense_loss_small", instance.metadata().name());
        assertEquals(tuning.workload.WorkloadKind.LOSS, instance.metadata().kind());
        assertEquals(1, instance.root().getShapeUnsafe().length);
        assertEquals(ValidationTargetKind.ROOT, instance.validationTarget().kind());
        assertEquals(ValidationReferenceKind.BASELINE_PROFILE, instance.reference().kind());
        List<operations.Operation.OpType> opTypes = instance.root().topologicalSort().stream()
                .map(tensor.Tensor::getOperation)
                .filter(java.util.Objects::nonNull)
                .map(operations.Operation::opType)
                .toList();
        assertTrue(opTypes.contains(operations.Operation.OpType.NLL_LOSS));
        assertTrue(opTypes.contains(operations.Operation.OpType.CROSS_ENTROPY_LOSS));
        assertTrue(!opTypes.contains(operations.Operation.OpType.CROSS_ENTROPY_LOSS_INDICES));
    }

    @Test
    void transformerBlockHotPathInstantiatesProjectionAttentionAndFeedForwardGraph() {
        ExecutionProfile profile = new ExecutionProfile(
                "transformer-block-profile",
                "transformer-block-profile",
                tensor.DataType.FLOAT32,
                ExecutionMode.FORWARD,
                config.compile.CompileConfig.inference(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                StandardWorkloads.transformerHotPathDefaults()
        );

        WorkloadInstance instance = StandardWorkloads.transformerBlockHotPath("transformer_block_hot_path")
                .instantiate(new WorkloadEnvironment(profile));

        assertEquals("transformer_block_hot_path", instance.metadata().name());
        assertEquals(tuning.workload.WorkloadKind.TRANSFORMER_HOT_PATH, instance.metadata().kind());
        assertEquals(1, instance.root().getShapeUnsafe().length);
        assertEquals(ValidationTargetKind.ROOT, instance.validationTarget().kind());
        assertEquals(ValidationReferenceKind.BASELINE_PROFILE, instance.reference().kind());
    }

    @Test
    void transformerBlockClosureWorkloadCoversAcceleratorEvidenceFamilies() throws Exception {
        ExecutionProfile profile = new ExecutionProfile(
                "accelerator-closure-transformer-block",
                "accelerator-closure-transformer-block",
                tensor.DataType.FLOAT32,
                ExecutionMode.FORWARD_BACKWARD,
                config.compile.CompileConfig.training(),
                config.runtime.RuntimeConfig.trainingDefaults(),
                StandardWorkloads.transformerHotPathDefaults()
        );

        WorkloadInstance instance = StandardWorkloads.transformerBlockHotPath("accelerator_closure_transformer_block")
                .instantiate(new WorkloadEnvironment(profile));

        assertEquals("accelerator_closure_transformer_block", instance.metadata().name());
        assertEquals(tuning.workload.WorkloadKind.TRANSFORMER_HOT_PATH, instance.metadata().kind());
        assertEquals(ValidationReferenceKind.BASELINE_PROFILE, instance.reference().kind());
        assertEquals(ValidationTargetKind.ROOT, instance.validationTarget().kind());
        assertEquals(1, instance.root().getShapeUnsafe().length);

        String source = Files.readString(Path.of("src/main/java/tuning/workload/TransformerBlockHotPathWorkloadSpec.java"));
        assertTrue(source.contains("linear"));
        assertTrue(source.contains("reshape"));
        assertTrue(source.contains("permute"));
        assertTrue(source.contains("scaledDotProductAttention"));
        assertTrue(source.contains("tanh"));
        assertTrue(source.contains("mean"));
        assertTrue(source.contains("gradientLabels"));
    }

    @Test
    void transformerHotPathWorkloadUsesProfileWorkloadMetadata() {
        ExecutionProfile profile = new ExecutionProfile(
                "transformer-profile",
                "transformer-profile",
                tensor.DataType.FLOAT32,
                ExecutionMode.FORWARD,
                config.compile.CompileConfig.inference(),
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
                config.compile.CompileConfig.inference(),
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
        var reduction = StandardWorkloads.reductionChain("reduction_test", 4, 8)
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
        var denseLoss = StandardWorkloads.denseLoss(
                        "dense_loss_test",
                        4, 8,
                        tensor.loss.LossReduction.MEAN
                )
                .instantiate(new WorkloadEnvironment(profile));
        var boolCompare = StandardWorkloads.boolCompareWhere("bool_compare_test", 4, 8)
                .instantiate(new WorkloadEnvironment(profile));
        var scatterIndexGradient = StandardWorkloads.scatterIndexGradient("scatter_index_gradient_test", 4, 8, 2)
                .instantiate(new WorkloadEnvironment(profile));

        assertEquals(ValidationReferenceKind.BASELINE_PROFILE, matmul.reference().kind());
        assertEquals(ValidationReferenceKind.BASELINE_PROFILE, conv.reference().kind());
        assertEquals(ValidationReferenceKind.BASELINE_PROFILE, abc.reference().kind());
        assertEquals(ValidationReferenceKind.BASELINE_PROFILE, mlp.reference().kind());
        assertEquals(ValidationReferenceKind.BASELINE_PROFILE, norm.reference().kind());
        assertEquals(ValidationReferenceKind.BASELINE_PROFILE, reduction.reference().kind());
        assertEquals(ValidationReferenceKind.BASELINE_PROFILE, pool.reference().kind());
        assertEquals(ValidationReferenceKind.BASELINE_PROFILE, loss.reference().kind());
        assertEquals(ValidationReferenceKind.BASELINE_PROFILE, denseLoss.reference().kind());
        assertEquals(ValidationReferenceKind.BASELINE_PROFILE, boolCompare.reference().kind());
        assertEquals(ValidationReferenceKind.BASELINE_PROFILE, scatterIndexGradient.reference().kind());

        assertEquals(ValidationTargetKind.LABEL, matmul.validationTarget().kind());
        assertEquals("matmul", matmul.validationTarget().label());
        assertEquals(ValidationTargetKind.LABEL, conv.validationTarget().kind());
        assertEquals("conv2d", conv.validationTarget().label());
        assertEquals(ValidationTargetKind.LABEL, norm.validationTarget().kind());
        assertEquals("layerNorm", norm.validationTarget().label());
        assertEquals(ValidationTargetKind.LABEL, pool.validationTarget().kind());
        assertEquals("maxPool2d", pool.validationTarget().label());
        assertEquals(ValidationTargetKind.ROOT, loss.validationTarget().kind());
        assertEquals(ValidationTargetKind.ROOT, denseLoss.validationTarget().kind());
        assertEquals(ValidationTargetKind.ROOT, reduction.validationTarget().kind());
        assertEquals(ValidationTargetKind.ROOT, boolCompare.validationTarget().kind());
        assertEquals(ValidationTargetKind.ROOT, scatterIndexGradient.validationTarget().kind());
    }

    @Test
    void standardWorkloadsSupportsPresetDrivenBenchmarkRequests() {
        BenchmarkEntry candidate = BenchmarkEntry.candidate("preset", new ExecutionProfile(
                "preset",
                "preset",
                tensor.DataType.FLOAT64,
                ExecutionMode.FORWARD,
                config.compile.CompileConfig.noGraphOptimizationBaseline(),
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
                config.compile.CompileConfig.noGraphOptimizationBaseline(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        ));

        var benchmarkRequest = StandardWorkloads.benchmark("cross_entropy_small", java.util.List.of(candidate));
        assertEquals("cross_entropy_small", benchmarkRequest.workload().name());
        assertTrue(benchmarkRequest.validation().requireGradientMatch());
    }
}
