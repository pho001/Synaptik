package io.github.pho001.synaptik.backend.cpu;

import static org.junit.jupiter.api.Assertions.*;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Set;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

class CpuInternalPackageInventoryTest {
    @Test void exposesOnlyTheAuthorizedSupportedAndInternalTypeInventory() throws Exception {
        Path root = Path.of("src/main/java/io/github/pho001/synaptik/backend/cpu");
        Set<String> packages;
        Set<String> javaFiles;
        try (var paths = Files.walk(root)) {
            packages = paths.filter(path -> path.getFileName().toString().equals("package-info.java"))
                    .map(path -> root.relativize(path.getParent()).toString().replace('\\', '/'))
                    .collect(Collectors.toSet());
        }
        try (var paths = Files.walk(root)) {
            javaFiles = paths.filter(path -> path.getFileName().toString().endsWith(".java"))
                    .map(path -> root.relativize(path).toString().replace('\\', '/'))
                    .collect(Collectors.toSet());
        }
        assertAll(
                () -> assertEquals(Set.of("", "internal", "internal/memory", "internal/prepare",
                        "internal/lowering", "internal/ir", "internal/codegen/emit",
                        "internal/route/portable", "internal/cache", "internal/executable",
                        "internal/reference"), packages),
                () -> assertEquals(Set.of(
                        "CpuCapabilityProvider.java", "package-info.java", "internal/package-info.java",
                        "internal/memory/CpuBorrowedBuffer.java", "internal/memory/CpuBufferArgument.java",
                        "internal/memory/CpuBufferRepresentation.java", "internal/memory/CpuNativeBuffer.java",
                        "internal/memory/CpuContiguousWorkspace.java",
                        "internal/memory/package-info.java", "internal/prepare/CpuPartitionAnalysisInputs.java",
                        "internal/prepare/CpuPartitionPreparationPlan.java",
                        "internal/prepare/CpuPartitionPreparer.java", "internal/prepare/CpuPartitionFinalizer.java",
                        "internal/prepare/package-info.java", "internal/lowering/CpuPartitionLowering.java",
                        "internal/lowering/CpuPartitionDagDecomposer.java",
                        "internal/lowering/CpuScalarPowerAnalysis.java",
                        "internal/lowering/CpuAffineLayoutLowering.java",
                        "internal/lowering/CpuNonAffineMovementLowering.java",
                        "internal/lowering/CpuIndexingLowering.java",
                        "internal/lowering/CpuScatterLowering.java",
                        "internal/lowering/CpuFoldLowering.java",
                        "internal/lowering/CpuOrderingLowering.java",
                        "internal/lowering/CpuRandomLowering.java",
                        "internal/lowering/CpuScanLowering.java",
                        "internal/lowering/CpuAggregateLowering.java",
                        "internal/lowering/CpuArgExtremaLowering.java",
                        "internal/lowering/CpuMaskedReductionLowering.java",
                        "internal/lowering/CpuAdvancedReductionLowering.java",
                        "internal/lowering/CpuSoftmaxLowering.java",
                        "internal/lowering/CpuBatchNormInferenceLowering.java",
                        "internal/lowering/CpuBatchNormTrainingLowering.java",
                        "internal/lowering/CpuConv2dLowering.java",
                        "internal/lowering/CpuConv1dCompositionLowering.java",
                        "internal/lowering/CpuConv3dLowering.java",
                        "internal/lowering/CpuTrailingNormalizationLowering.java",
                        "internal/lowering/CpuMaterializationPlan.java",
                        "internal/lowering/package-info.java", "internal/ir/CpuKernelIr.java",
                        "internal/ir/CpuPointwiseOpcode.java",
                        "internal/ir/CpuPortableKernelIr.java", "internal/ir/CpuAffineCopyIr.java",
                        "internal/ir/CpuDataMovementIr.java",
                        "internal/ir/CpuIndexingIr.java",
                        "internal/ir/CpuScatterIr.java",
                        "internal/ir/CpuFoldIr.java",
                        "internal/ir/CpuOrderingIr.java",
                        "internal/ir/CpuRandomIr.java",
                        "internal/ir/CpuScanIr.java",
                        "internal/ir/CpuAggregateIr.java",
                        "internal/ir/CpuArgExtremaIr.java",
                        "internal/ir/CpuMaskedReductionIr.java",
                        "internal/ir/CpuAdvancedReductionIr.java",
                        "internal/ir/CpuSoftmaxIr.java",
                        "internal/ir/CpuBatchNormInferenceIr.java",
                        "internal/ir/CpuBatchNormTrainingIr.java",
                        "internal/ir/CpuConv2dIr.java",
                        "internal/ir/CpuConv3dIr.java",
                        "internal/ir/CpuTrailingNormalizationIr.java",
                        "internal/ir/CpuAccessPlan.java", "internal/ir/package-info.java",
                        "internal/codegen/emit/CpuCarrierEmitter.java",
                        "internal/codegen/emit/CpuAffineCopyEmitter.java",
                        "internal/codegen/emit/CpuDataMovementEmitter.java",
                        "internal/codegen/emit/CpuIndexingEmitter.java",
                        "internal/codegen/emit/CpuScatterEmitter.java",
                        "internal/codegen/emit/CpuFoldEmitter.java",
                        "internal/codegen/emit/CpuOrderingEmitter.java",
                        "internal/codegen/emit/CpuRandomEmitter.java",
                        "internal/codegen/emit/CpuScanEmitter.java",
                        "internal/codegen/emit/CpuAggregateEmitter.java",
                        "internal/codegen/emit/CpuArgExtremaEmitter.java",
                        "internal/codegen/emit/CpuMaskedReductionEmitter.java",
                        "internal/codegen/emit/CpuLogSumExpEmitter.java",
                        "internal/codegen/emit/CpuStatisticalReductionEmitter.java",
                        "internal/codegen/emit/CpuNormEmitter.java",
                        "internal/codegen/emit/CpuSoftmaxEmitter.java",
                        "internal/codegen/emit/CpuBatchNormInferenceEmitter.java",
                        "internal/codegen/emit/CpuBatchNormTrainingEmitter.java",
                        "internal/codegen/emit/CpuConv2dEmitter.java",
                        "internal/codegen/emit/CpuConv3dEmitter.java",
                        "internal/codegen/emit/CpuLayerNormEmitter.java",
                        "internal/codegen/emit/CpuRmsNormEmitter.java",
                        "internal/codegen/emit/CpuExactSumEmitter.java",
                        "internal/codegen/emit/CpuExactProductEmitter.java",
                        "internal/codegen/emit/CpuClassFileKernelGenerator.java",
                        "internal/codegen/emit/CpuGeneratedKernel.java",
                        "internal/codegen/emit/CpuLoopEmitter.java",
                        "internal/codegen/emit/CpuScalarEmitter.java",
                        "internal/codegen/emit/CpuVectorInstructionEmitter.java",
                        "internal/codegen/emit/CpuVectorMath.java",
                        "internal/codegen/emit/package-info.java",
                        "internal/route/portable/CpuPortableRoutePlan.java",
                        "internal/route/portable/package-info.java",
                        "internal/cache/CpuGeneratedKernelArtifactStore.java",
                        "internal/cache/CpuGeneratorSchema.java", "internal/cache/CpuKernelSpecialization.java",
                        "internal/cache/CpuSpecializationBudget.java",
                        "internal/cache/CpuLoweringFingerprint.java", "internal/cache/package-info.java",
                        "internal/executable/CpuPreparedExecutable.java",
                        "internal/executable/CpuPreparedPartitionExecutable.java",
                        "internal/executable/CpuSoftmaxInputValidator.java",
                        "internal/executable/CpuWorkerGroup.java", "internal/executable/package-info.java",
                        "internal/reference/CpuScalarReferenceKernel.java",
                        "internal/reference/CpuAdvancedReductionReferenceKernel.java",
                        "internal/reference/CpuSoftmaxReferenceKernel.java",
                        "internal/reference/CpuBatchNormInferenceReferenceKernel.java",
                        "internal/reference/CpuBatchNormTrainingReferenceKernel.java",
                        "internal/reference/CpuConv2dReferenceKernel.java",
                        "internal/reference/CpuConv3dReferenceKernel.java",
                        "internal/reference/CpuTrailingNormalizationReferenceKernel.java",
                        "internal/reference/package-info.java"),
                        javaFiles),
                () -> {
                    try (var old = Files.exists(root.resolve("execution"))
                            ? Files.walk(root.resolve("execution")) : java.util.stream.Stream.<Path>empty()) {
                        assertEquals(0, old.filter(Files::isRegularFile).count());
                    }
                },
                () -> assertFalse(Files.exists(root.resolve("internal/route/nativeblas"))),
                () -> assertFalse(Files.exists(root.resolve("internal/route/nativeops"))));
    }
}
