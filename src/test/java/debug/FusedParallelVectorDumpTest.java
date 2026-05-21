package debug;

import backend.cpu.kernels.CpuExecutionMode;
import backend.ApproxMode;
import config.backend.AttentionMatMulPolicy;
import config.backend.CpuKernelConfig;
import config.backend.SumAccuracyMode;
import config.compile.CompileConfig;
import config.compile.GraphOptimizationConfig;
import config.profile.ExecutionProfile;
import config.profile.WorkloadProfile;
import config.runtime.ApproximationConfig;
import config.runtime.BlasConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import backend.cpu.fused.codegen.FusedKernelGeneratorRouter;
import graph.execution.PreparedExecution;
import graph.execution.PreparedExecutionStep;
import backend.cpu.fused.plan.FusedOperation;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.factory.TensorDataFactory;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

final class FusedParallelVectorDumpTest {
    private static final Path OUTPUT_DIR = Path.of("build", "fused-dump");

    @Test
    void dumpGeneratedParallelVectorFusedExecutables() throws IOException {
        Files.createDirectories(OUTPUT_DIR);

        dumpFor(DataType.FLOAT64, "debug.dump.AbcF64ParallelVectorFused");
        dumpFor(DataType.FLOAT32, "debug.dump.AbcF32ParallelVectorFused");
        dumpFor(DataType.BFLOAT16, "debug.dump.AbcBF16ParallelVectorRequested");
    }

    private static void dumpFor(DataType dataType, String binaryName) throws IOException {
        ExecutionProfile profile = new ExecutionProfile(
                "fused-dump-" + dataType.name().toLowerCase(),
                "fused-dump-" + dataType.name().toLowerCase(),
                dataType,
                backend.runtime.ExecutionMode.FORWARD,
                CompileConfig.inference().withGraphOptimization(config.compile.GraphOptimizationConfig.noGraphOptimization()),
                runtimeForDump(),
                WorkloadProfile.none()
        );

        Tensor root = fusedDumpRoot(dataType, 131_072);

        PreparedExecution prepared = CompiledGraph.compile(root, profile.compile()).prepare(profile.runtime());

        String fusedStepsSummary = prepared.forwardSteps().stream()
                .filter(step -> step.executionOperation() instanceof FusedOperation)
                .map(step -> {
                    var hints = testsupport.MetadataArtifacts.cpuPlan(step.metadata()) == null ? null : testsupport.MetadataArtifacts.cpuPlan(step.metadata()).dispatchHints();
                    return step.compiledNode().label()
                            + " mode=" + (hints == null ? "null" : hints.mode())
                            + " vectorWidth=" + (hints == null ? -1 : hints.vectorWidth())
                            + " workers=" + (hints == null ? -1 : hints.plannedWorkers())
                            + " expr=" + step.executionOperation().getExpression();
                })
                .collect(Collectors.joining("\n"));

        PreparedExecutionStep selected = prepared.forwardSteps().stream()
                .filter(step -> step.executionOperation() instanceof FusedOperation)
                .filter(step -> testsupport.MetadataArtifacts.cpuPlan(step.metadata()) != null)
                .filter(step -> testsupport.MetadataArtifacts.cpuPlan(step.metadata()).dispatchHints() != null)
                .filter(step -> testsupport.MetadataArtifacts.cpuPlan(step.metadata()).dispatchHints().parallel())
                .filter(step -> testsupport.MetadataArtifacts.cpuPlan(step.metadata()).dispatchHints().vectorized())
                .findFirst()
                .orElseThrow(() -> new IllegalStateException("Missing fused parallel-vector step for " + dataType + "\n" + fusedStepsSummary));

        assertNotNull(testsupport.MetadataArtifacts.fusedExecutable(selected.metadata()), "Prepared fused executable must be present");

        FusedOperation fused = (FusedOperation) selected.executionOperation();
        var hints = testsupport.MetadataArtifacts.cpuPlan(selected.metadata()).dispatchHints();
        assertTrue(hints.mode() == CpuExecutionMode.PARALLEL_VECTOR || hints.mode() == CpuExecutionMode.VECTOR,
                "Expected vector-capable fused dispatch mode");

        String internalName = binaryName.replace('.', '/');
        byte[] bytecode = FusedKernelGeneratorRouter.generate(
                internalName,
                fused.getPlan(),
                fused.getPrecisionMode(),
                hints.vectorWidth()
        );

        Path classPath = OUTPUT_DIR.resolve(binaryName.replace('.', '/') + ".class");
        Files.createDirectories(classPath.getParent());
        Files.write(classPath, bytecode);

        String metadata = """
                binaryName=%s
                dataType=%s
                selectedNodeLabel=%s
                fusedExpression=%s
                preparedExecutableClass=%s
                dispatchMode=%s
                vectorWidth=%d
                plannedWorkers=%d
                scalarChunkSize=%d
                vectorChunkSize=%d
                fusedPlanNodeCount=%d
                fusedPlanInputCount=%d
                note=%s
                """.formatted(
                binaryName,
                dataType,
                selected.compiledNode().label(),
                fused.getExpression(),
                testsupport.MetadataArtifacts.fusedExecutable(selected.metadata()).getClass().getName(),
                hints.mode(),
                hints.vectorWidth(),
                hints.plannedWorkers(),
                hints.scalarChunkSize(),
                hints.vectorChunkSize(),
                fused.getPlan().nodeCount(),
                fused.getPlan().inputCount(),
                dataType == DataType.BFLOAT16
                        ? "BF16 generated class now emits a real Vector API range loop using FloatVector lanes plus BF16 storage conversion helpers."
                        : "F32/F64 generated class contains a real Vector API range loop; parallel scheduling happens in backend.cpu.kernels.fused.FusedExecutor."
        );

        Path metadataPath = OUTPUT_DIR.resolve(binaryName.replace('.', '/') + ".metadata.txt");
        Files.writeString(metadataPath, metadata, StandardCharsets.UTF_8);
    }

    private static Tensor fusedDumpRoot(DataType dataType, int size) {
        Tensor a = TensorDataFactory.shapedTensor("dumpA", buildInput(size, 0.11), false, dataType, size);
        Tensor b = TensorDataFactory.shapedTensor("dumpB", buildInput(size, -0.07), false, dataType, size);
        Tensor c = TensorDataFactory.shapedTensor("dumpC", buildInput(size, 0.03), false, dataType, size);
        return a.add(b).mul(c).add(a.mul(0.25)).max(b).min(c).sigmoid();
    }

    private static double[] buildInput(int size, double phase) {
        double[] values = new double[size];
        for (int i = 0; i < values.length; i++) {
            values[i] = Math.sin(i * 0.013 + phase) * 0.75 + Math.cos(i * 0.007 - phase) * 0.25 + 1.25;
        }
        return values;
    }

    private static RuntimeConfig runtimeForDump() {
        CpuKernelConfig cpu = new CpuKernelConfig(
                1,
                16, 16, 16,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1_000_000_000,
                4,
                2,
                1,
                4_096,
                8_192,
                16_384,
                16_384,
                4,
                SumAccuracyMode.FAST,
                2_000_000,
                AttentionMatMulPolicy.AUTO
        );
        return new RuntimeConfig(
                cpu,
                new ApproximationConfig(ApproxMode.OFF, true),
                BlasConfig.disabled()
        );
    }
}
