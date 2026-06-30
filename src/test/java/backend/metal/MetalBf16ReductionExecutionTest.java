package backend.metal;

import runtime.contract.ExecutionMode;
import config.compile.BackendPlanningConfig;
import config.compile.CompileConfig;
import config.optimizer.CpuFusionConfig;
import config.optimizer.CpuRegionConfig;
import config.profile.ExecutionProfile;
import config.profile.WorkloadProfile;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import graph.execution.PublicationPolicy;
import trace.execution.ExecutionStepTrace;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tuning.workload.StandardWorkloads;
import tuning.workload.WorkloadEnvironment;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class MetalBf16ReductionExecutionTest {
    @Test
    void reductionChainExecutesOnMetalBfloat16() {
        String explicitLib = System.getProperty("synaptik.metal.mps.lib");
        assumeTrue(explicitLib != null && !explicitLib.isBlank(), "Set -Dsynaptik.metal.mps.lib to run Metal test.");

        ExecutionProfile profile = new ExecutionProfile(
                "metal-bf16-reduction-regression",
                "metal-bf16-reduction-regression",
                DataType.BFLOAT16,
                ExecutionMode.FORWARD,
                compileConfig(),
                RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );
        var workload = StandardWorkloads.reductionChain("metal_bf16_reduction_regression", 64, 1024)
                .instantiate(new WorkloadEnvironment(profile));

        var trace = CompiledGraph.compile(workload.root(), profile.compile())
                .prepare(profile.runtime())
                .executeTraced(profile.mode(), PublicationPolicy.NONE);

        assertEquals(1, trace.steps().size(), () -> "Expected one fused Metal reduction region, got:\n" + describe(trace.steps()));
        assertEquals(0, trace.cpuMaterializations().size(), () -> "Unexpected CPU materialization: " + trace.cpuMaterializations());

        ExecutionStepTrace step = trace.steps().getFirst();
        assertEquals("GPU_METAL", step.backend(), () -> "Expected Metal step, got:\n" + describe(trace.steps()));
        Map<String, Object> attrs = step.metadata().attributes();
        assertFalse(Boolean.TRUE.equals(attrs.get("metalUsedCpuFallback")), () -> "Metal BF16 reduction fell back to CPU: " + attrs);
        assertEquals("BUFFER_BINDING", attrs.get("metalExecutionPath"), () -> "Expected buffer binding execution: " + attrs);
    }

    private static CompileConfig compileConfig() {
        CompileConfig base = CompileConfig.inference();
        return base
                .withBackendPlanning(BackendPlanningConfig.autoAccelerator().withCpuRegions(CpuRegionConfig.defaults()))
                .withRegionOptimization(base.regionOptimization().withCpuFusion(CpuFusionConfig.defaults()));
    }

    private static String describe(java.util.List<ExecutionStepTrace> steps) {
        StringBuilder out = new StringBuilder();
        for (ExecutionStepTrace step : steps) {
            out.append('#').append(step.index())
                    .append(' ')
                    .append(step.backend())
                    .append(' ')
                    .append(step.opType())
                    .append(' ')
                    .append(step.shape())
                    .append(" attrs=")
                    .append(step.metadata().attributes())
                    .append('\n');
        }
        return out.toString();
    }
}
