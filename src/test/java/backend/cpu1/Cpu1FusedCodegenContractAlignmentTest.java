package backend.cpu1;

import backend.cpu1.exec.Cpu1FusedElementwiseExecutableUnit;
import backend.cpu1.fused.ir.Cpu1FusedAccessKind;
import backend.cpu1.fused.ir.Cpu1FusedExpressionPlan;
import backend.cpu1.fused.ir.Cpu1FusedInputPlan;
import backend.cpu1.fused.ir.Cpu1FusedNodePlan;
import backend.cpu1.fused.ir.Cpu1FusedScalarParameter;
import backend.cpu1.kernels.Cpu1LayoutKind;
import backend.cpu1.kernels.Cpu1VectorizationKind;
import backend.cpu1.kernels.fused.codegen.Cpu1FusedCodegenClassSignature;
import backend.cpu1.kernels.fused.codegen.Cpu1FusedCodegenKernel;
import backend.cpu1.kernels.fused.codegen.Cpu1FusedCodegenLoopKind;
import backend.cpu1.kernels.fused.codegen.Cpu1FusedCodegenPlan;
import backend.cpu1.kernels.fused.codegen.Cpu1FusedCodegenRejectionReason;
import backend.cpu1.launch.Cpu1LaunchConfig;
import backend.cpu1.launch.Cpu1SingleThreadLaunch;
import backend.cpu1.prepare.Cpu1PrepareConfig;
import backend.cpu1.prepare.Cpu1PreparedArtifact;
import backend.cpu1.prepare.Cpu1PreparedFusedElementwiseUnit;
import backend.cpu1.prepare.dispatch.Cpu1CostClass;
import backend.cpu1.prepare.dispatch.Cpu1FusedDispatchDecision;
import backend.cpu1.storage.Cpu1StorageKind;
import graph.execution.trace.StepTraceContribution;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class Cpu1FusedCodegenContractAlignmentTest {
    @Test
    void fusedExecutableOnlyCallsPreparedGeneratedKernel() throws IOException {
        String source = Files.readString(Path.of(
                "src/main/java/backend/cpu1/exec/Cpu1FusedElementwiseExecutableUnit.java"
        ));

        assertTrue(source.contains(
                "preparedUnit.generatedKernel().computeRange(args, startInclusive, endExclusive)"
        ));
        assertFalse(source.contains("Cpu1FusedCodegenKernelFactory"));
        assertFalse(source.contains("prepareKernel("));
        assertFalse(source.contains("rejectionReason"));
        assertFalse(source.contains(".nodes()"));
        assertFalse(source.contains("sourceOperations"));
        assertFalse(source.contains("operation()"));
        assertFalse(source.contains("switch"));
        assertFalse(source.contains("fallback"));
        assertFalse(source.contains("interpreter"));
        assertFalse(source.contains("evaluator"));
    }

    @Test
    void codegenFactoryIsOnlyCalledDuringPrepareInCpu1MainSources() throws IOException {
        List<Path> offenders;
        try (var stream = Files.walk(Path.of("src/main/java/backend/cpu1"))) {
            offenders = stream
                    .filter(path -> path.toString().endsWith(".java"))
                    .filter(path -> !path.endsWith("Cpu1FusedCodegenKernelFactory.java"))
                    .filter(path -> !path.endsWith("Cpu1FusedElementwisePreparer.java"))
                    .filter(path -> contains(path, "Cpu1FusedCodegenKernelFactory"))
                    .toList();
        }

        assertEquals(List.of(), offenders);
    }

    @Test
    void preparedFusedUnitRequiresGeneratedKernelForAcceptedCodegen() {
        IllegalArgumentException thrown = assertThrows(
                IllegalArgumentException.class,
                () -> new Cpu1PreparedFusedElementwiseUnit(
                        "fused",
                        List.of(1),
                        List.of(0),
                        1,
                        DataType.FLOAT32,
                        3,
                        new int[]{3},
                        expressionPlan(),
                        Cpu1LayoutKind.CONTIGUOUS,
                        Cpu1StorageKind.JAVA_ARRAY,
                        new Cpu1SingleThreadLaunch(),
                        Cpu1LaunchConfig.singleThread(),
                        dispatchDecision(),
                        Cpu1FusedCodegenRejectionReason.NONE,
                        null,
                        false,
                        false
                )
        );

        assertTrue(thrown.getMessage().contains("generatedKernel cannot be null"));
    }

    @Test
    void codegenKernelHandleDelegatesRangeWithoutInspectingPlan() {
        AtomicInteger start = new AtomicInteger(-1);
        AtomicInteger end = new AtomicInteger(-1);
        Cpu1FusedCodegenKernel kernel = new Cpu1FusedCodegenKernel(
                new Cpu1FusedCodegenClassSignature("test-signature"),
                "backend.cpu1.generated.TestKernel",
                (args, startInclusive, endExclusive) -> {
                    start.set(startInclusive);
                    end.set(endExclusive);
                }
        );

        kernel.computeRange(null, 4, 9);

        assertEquals(4, start.get());
        assertEquals(9, end.get());
    }

    @Test
    void classSignatureIncludesSupportAbiAndReluHelperTarget() {
        Cpu1FusedCodegenPlan plan = codegenPlan(expressionPlan());
        String signature = plan.classSignature().canonicalSignature();

        assertTrue(signature.contains("|supportAbi=1|"));
        assertTrue(signature.contains(
                "helperTargets=[backend/cpu1/kernels/fused/codegen/support/Cpu1FusedMathSupport.reluF32(F)F]"
        ));
    }

    @Test
    void classSignatureHasEmptyHelperTargetsForPureAdd() {
        Cpu1FusedCodegenPlan plan = codegenPlan(addExpressionPlan());
        String signature = plan.classSignature().canonicalSignature();

        assertTrue(signature.contains("|supportAbi=1|"));
        assertTrue(signature.contains("|helperTargets=[]|"));
        assertFalse(signature.contains("Cpu1FusedMathSupport."));
    }

    @Test
    void preparedArtifactWrapsFusedElementwiseUnitAsFirstClassExecutable() {
        Cpu1PreparedFusedElementwiseUnit preparedUnit = acceptedPreparedFusedUnit();
        Cpu1PreparedArtifact artifact = new Cpu1PreparedArtifact(preparedUnit);

        assertSame(preparedUnit, artifact.preparedFusedElementwiseUnit());
        Cpu1FusedElementwiseExecutableUnit executable = assertInstanceOf(
                Cpu1FusedElementwiseExecutableUnit.class,
                artifact.executableUnit()
        );
        assertSame(preparedUnit, executable.preparedUnit());
        assertTrue(artifact.scratchBufferSpec().isEmpty());
        assertThrows(IllegalStateException.class, artifact::preparedUnit);
    }

    @Test
    void fusedPreparedArtifactContributesCpu1FusedTraceMetadata() {
        Cpu1PreparedFusedElementwiseUnit preparedUnit = acceptedPreparedFusedUnit();
        Cpu1PreparedArtifact artifact = new Cpu1PreparedArtifact(preparedUnit);

        StepTraceContribution trace = artifact.traceContribution(null, null, null);

        assertEquals("CPU1_FUSED_ELEMENTWISE", trace.kernel());
        assertEquals("CPU1_FUSED_ELEMENTWISE", trace.attributes().get("cpu1KernelId"));
        assertEquals(1, trace.attributes().get("cpu1FusedNodeCount"));
        assertEquals(1, trace.attributes().get("cpu1FusedInputCount"));
        assertEquals("JAVA_ARRAY", trace.attributes().get("cpu1StorageKind"));
        assertEquals("CONTIGUOUS", trace.attributes().get("cpu1LayoutKind"));
        assertEquals(1, trace.attributes().get("cpu1FusedOutputNodeId"));
        assertEquals(3, trace.attributes().get("cpu1FusedElementCount"));
        assertEquals(1, trace.attributes().get("cpu1FusedLaunchWorkers"));
        assertEquals(0, trace.attributes().get("cpu1FusedLaunchChunkSize"));
        assertEquals("CHEAP_ELEMENTWISE", trace.attributes().get("cpu1FusedCostClass"));
        assertEquals("SCALAR", trace.attributes().get("cpu1FusedRequestedVectorization"));
        assertEquals(false, trace.attributes().get("cpu1FusedApproxExp"));
        assertEquals(false, trace.attributes().get("cpu1FusedApproxTanh"));
        assertEquals("NONE", trace.attributes().get("cpu1FusedCodegenRejectionReason"));
        assertEquals("test-signature", trace.attributes().get("cpu1FusedClassSignature"));
        assertEquals("backend.cpu1.generated.TestKernel", trace.attributes().get("cpu1FusedGeneratedClassName"));
        assertEquals("SCALAR", trace.dispatch().mode());
        assertEquals(1, trace.dispatch().vectorWidth());
        assertEquals(1, trace.dispatch().plannedWorkers());
        assertEquals("CPU1_FUSED_ELEMENTWISE", trace.fused().dispatchFamily());
        assertEquals("test-signature", trace.fused().schedulerSignature());
        assertEquals("CPU1", trace.fused().executionBackend());
        assertEquals(1, trace.fused().fusedNodeCount());
        assertEquals(1, trace.fused().fusedInputCount());
        assertEquals("NONE", trace.fused().vectorFallbackReason());
    }

    private static boolean contains(Path path, String needle) {
        try {
            return Files.readString(path).contains(needle);
        } catch (IOException e) {
            throw new IllegalStateException("Cannot read " + path, e);
        }
    }

    private static Cpu1FusedExpressionPlan expressionPlan() {
        return new Cpu1FusedExpressionPlan(
                List.of(new Cpu1FusedNodePlan(
                        0,
                        1,
                        Operation.OpType.RELU,
                        List.of(0),
                        1,
                        DataType.FLOAT32,
                        Cpu1FusedScalarParameter.NONE
                )),
                List.of(new Cpu1FusedInputPlan(
                        0,
                        0,
                        DataType.FLOAT32,
                        new int[]{3},
                        new int[]{1},
                        new int[]{3},
                        new int[]{1},
                        0,
                        new int[]{1},
                        Cpu1FusedAccessKind.DIRECT_CONTIGUOUS
                )),
                1
        );
    }

    private static Cpu1FusedExpressionPlan addExpressionPlan() {
        return new Cpu1FusedExpressionPlan(
                List.of(new Cpu1FusedNodePlan(
                        0,
                        2,
                        Operation.OpType.ADD,
                        List.of(0, 1),
                        2,
                        DataType.FLOAT32,
                        Cpu1FusedScalarParameter.NONE
                )),
                List.of(
                        new Cpu1FusedInputPlan(
                                0,
                                0,
                                DataType.FLOAT32,
                                new int[]{3},
                                new int[]{1},
                                new int[]{3},
                                new int[]{1},
                                0,
                                new int[]{1},
                                Cpu1FusedAccessKind.DIRECT_CONTIGUOUS
                        ),
                        new Cpu1FusedInputPlan(
                                1,
                                1,
                                DataType.FLOAT32,
                                new int[]{3},
                                new int[]{1},
                                new int[]{3},
                                new int[]{1},
                                0,
                                new int[]{1},
                                Cpu1FusedAccessKind.DIRECT_CONTIGUOUS
                        )
                ),
                2
        );
    }

    private static Cpu1FusedCodegenPlan codegenPlan(Cpu1FusedExpressionPlan expressionPlan) {
        return Cpu1FusedCodegenPlan.from(
                expressionPlan,
                DataType.FLOAT32,
                Cpu1LayoutKind.CONTIGUOUS,
                Cpu1StorageKind.JAVA_ARRAY,
                Cpu1FusedCodegenLoopKind.CONTIGUOUS_SCALAR,
                Cpu1PrepareConfig.scalarSingleThread()
        );
    }

    private static Cpu1PreparedFusedElementwiseUnit acceptedPreparedFusedUnit() {
        return new Cpu1PreparedFusedElementwiseUnit(
                "fused",
                List.of(1),
                List.of(0),
                1,
                DataType.FLOAT32,
                3,
                new int[]{3},
                expressionPlan(),
                Cpu1LayoutKind.CONTIGUOUS,
                Cpu1StorageKind.JAVA_ARRAY,
                new Cpu1SingleThreadLaunch(),
                Cpu1LaunchConfig.singleThread(),
                dispatchDecision(),
                Cpu1FusedCodegenRejectionReason.NONE,
                new Cpu1FusedCodegenKernel(
                        new Cpu1FusedCodegenClassSignature("test-signature"),
                        "backend.cpu1.generated.TestKernel",
                        (args, startInclusive, endExclusive) -> {
                        }
                ),
                false,
                false
        );
    }

    private static Cpu1FusedDispatchDecision dispatchDecision() {
        return new Cpu1FusedDispatchDecision(
                Cpu1CostClass.CHEAP_ELEMENTWISE,
                Cpu1VectorizationKind.SCALAR,
                Cpu1LaunchConfig.singleThread(),
                Cpu1StorageKind.JAVA_ARRAY,
                1024,
                1024,
                1
        );
    }
}
