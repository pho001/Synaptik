package backend.cpu1;

import backend.cpu1.prepare.Cpu1PreparedArtifact;
import backend.lowering.LoweredExecutionUnit;
import backend.lowering.LoweringFamily;
import backend.prepare.BackendPrepareContext;
import backend.prepare.BackendPrepareDispatcher;
import config.backend.CpuKernelConfig;
import config.runtime.ApproximationConfig;
import config.runtime.BlasConfig;
import config.runtime.FusedExecutionPolicy;
import config.runtime.RuntimeConfig;
import graph.CompiledNode;
import graph.compile.descriptor.CompiledTensorDescriptorBuilder;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;
import graph.compile.intent.BackendIntentPlan;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;

class BackendPrepareDispatcherCpu1FusedRouteTest {
    @Test
    void defaultFusedPolicyKeepsOldCpuFusedRoute() {
        Fixture fixture = fixture();
        LoweredExecutionUnit loweredUnit = new LoweredExecutionUnit(
                "relu",
                LoweringFamily.FUSED_NATIVE,
                List.of(fixture.reluNodeId())
        );
        RuntimeConfig runtimeConfig = runtimeConfig(FusedExecutionPolicy.defaultsInference());
        BackendPrepareDispatcher dispatcher = BackendPrepareDispatcher.from(runtimeConfig);

        IllegalStateException thrown = assertThrows(
                IllegalStateException.class,
                () -> dispatcher.prepareCpuFusedStep(
                        fixture.outputNode(),
                        loweredUnit,
                        fixture.context(runtimeConfig)
                )
        );

        assertTrue(thrown.getMessage().contains("requires artifact RegionExecutionPlan"));
    }

    @Test
    void explicitCpu1FusedPolicyRoutesToCpu1PreparerWithoutFallback() {
        Fixture fixture = fixture();
        LoweredExecutionUnit loweredUnit = new LoweredExecutionUnit(
                "relu",
                LoweringFamily.FUSED_NATIVE,
                List.of(fixture.reluNodeId())
        );
        RuntimeConfig runtimeConfig = runtimeConfig(new FusedExecutionPolicy(true, true));
        BackendPrepareDispatcher dispatcher = BackendPrepareDispatcher.from(runtimeConfig);

        var metadata = dispatcher.prepareCpuFusedStep(
                fixture.outputNode(),
                loweredUnit,
                fixture.context(runtimeConfig)
        );

        Cpu1PreparedArtifact artifact = assertInstanceOf(Cpu1PreparedArtifact.class, metadata.artifact());
        assertTrue(artifact.preparedFusedElementwiseUnit().generatedKernel()
                .generatedClassName()
                .contains("Cpu1GeneratedFusedKernel"));
    }

    private static Fixture fixture() {
        Tensor input = new Tensor(new float[]{-1.0f, 0.0f, 1.0f}, new int[]{3}, null, "input", DataType.FLOAT32);
        Tensor output = input.relu();
        List<CompiledNode> nodes = CompiledNode.snapshot(output.topologicalSort(), BackendIntentPlan.empty());
        CompiledTensorDescriptorIndex descriptorIndex = CompiledTensorDescriptorBuilder.build(nodes);
        int reluNodeId = nodes.stream()
                .filter(node -> node.operation() != null && node.operation().opType() == Operation.OpType.RELU)
                .findFirst()
                .orElseThrow()
                .id();
        return new Fixture(nodes, descriptorIndex, nodes.getLast(), reluNodeId);
    }

    private static RuntimeConfig runtimeConfig(FusedExecutionPolicy fusedExecutionPolicy) {
        return new RuntimeConfig(
                CpuKernelConfig.defaultsInference(),
                ApproximationConfig.defaults(),
                BlasConfig.disabled(),
                fusedExecutionPolicy
        );
    }

    private record Fixture(
            List<CompiledNode> nodes,
            CompiledTensorDescriptorIndex descriptorIndex,
            CompiledNode outputNode,
            int reluNodeId
    ) {
        BackendPrepareContext context(RuntimeConfig runtimeConfig) {
            return new BackendPrepareContext(runtimeConfig, false, nodes, descriptorIndex, Map.of());
        }
    }
}
