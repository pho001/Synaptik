package backend.cpu1;

import backend.cpu.CpuNodeExecutionArtifact;
import backend.cpu1.prepare.Cpu1PreparedArtifact;
import prepare.context.BackendPrepareContext;
import prepare.orchestration.BackendPrepareDispatcher;
import config.runtime.CpuExecutionPolicy;
import config.runtime.RuntimeConfig;
import graph.compile.CompiledNodeSnapshotter;
import graph.model.CompiledNode;
import planning.descriptor.CompiledTensorDescriptorBuilder;
import planning.descriptor.CompiledTensorDescriptorIndex;
import planning.intent.BackendIntentPlan;
import runtime.execution.InputResidencyRequirement;
import runtime.execution.OutputResidencyEffect;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;

class BackendPrepareDispatcherCpu1DirectRouteTest {
    @Test
    void defaultCpuExecutionPolicyKeepsLegacyCpuDirectPreparer() {
        Fixture fixture = reluFixture();
        RuntimeConfig runtimeConfig = RuntimeConfig.inferenceDefaults();

        var metadata = BackendPrepareDispatcher.from(runtimeConfig)
                .prepare(fixture.outputNode(), fixture.context(runtimeConfig));

        assertInstanceOf(CpuNodeExecutionArtifact.class, metadata.executable());
    }

    @Test
    void explicitCpu1DirectPolicyRoutesNormalCpuNodeToCpu1Preparer() {
        Fixture fixture = reluFixture();
        RuntimeConfig runtimeConfig = RuntimeConfig.inferenceDefaults()
                .withCpuExecutionPolicy(new CpuExecutionPolicy(true, false));

        var metadata = BackendPrepareDispatcher.from(runtimeConfig)
                .prepare(fixture.outputNode(), fixture.context(runtimeConfig));

        Cpu1PreparedArtifact artifact = assertInstanceOf(Cpu1PreparedArtifact.class, metadata.executable());
        assertEquals(Operation.OpType.RELU, artifact.preparedUnit().opType());
        assertEquals(fixture.outputNode().inputIds(), metadata.executionInputNodeIds());
        assertEquals(InputResidencyRequirement.Mode.CPU_READABLE_ALL, metadata.inputResidencyRequirement().mode());
        assertEquals(OutputResidencyEffect.Mode.CPU_CURRENT_PRESERVE_NATIVE, metadata.outputResidencyEffect().mode());
    }

    private static Fixture reluFixture() {
        Tensor input = new Tensor(new float[]{-1.0f, 0.0f, 2.0f}, new int[]{3}, null, "direct_input", DataType.FLOAT32);
        Tensor output = input.relu();
        List<CompiledNode> nodes = CompiledNodeSnapshotter.snapshot(output.topologicalSort(), BackendIntentPlan.empty());
        CompiledTensorDescriptorIndex descriptorIndex = CompiledTensorDescriptorBuilder.build(nodes);
        return new Fixture(nodes, descriptorIndex, nodes.getLast());
    }

    private record Fixture(
            List<CompiledNode> nodes,
            CompiledTensorDescriptorIndex descriptorIndex,
            CompiledNode outputNode
    ) {
        BackendPrepareContext context(RuntimeConfig runtimeConfig) {
            return new BackendPrepareContext(runtimeConfig, false, nodes, descriptorIndex, Map.of());
        }
    }
}
