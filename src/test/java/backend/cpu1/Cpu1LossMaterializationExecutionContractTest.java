package backend.cpu1;

import backend.cpu1.prepare.Cpu1NodePreparer;
import backend.cpu1.prepare.Cpu1PrepareConfig;
import graph.compile.CompiledNodeSnapshotter;
import graph.model.CompiledNode;
import planning.descriptor.CompiledTensorDescriptorBuilder;
import planning.descriptor.CompiledTensorDescriptorIndex;
import planning.intent.BackendIntentPlan;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.loss.LossReduction;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class Cpu1LossMaterializationExecutionContractTest {
    @Test
    void unplannedStridedCrossEntropyIndicesLogitsAreRejectedByCpu1LossPreparer() {
        Tensor base = new Tensor(new float[]{
                1.0f, 2.0f,
                3.0f, 4.0f,
                5.0f, 6.0f
        }, new int[]{3, 2}, null, "unplannedCeIndicesBaseLogits", DataType.FLOAT32);
        Tensor logitsView = base.permute(1, 0);
        Tensor targets = new Tensor(new int[]{2, 1}, new int[]{2}, null, "unplannedCeIndicesTargets", DataType.INT32);
        Fixture fixture = fixture(logitsView.crossEntropyLossFromIndices(targets, 1, LossReduction.MEAN));

        UnsupportedOperationException thrown = assertThrows(
                UnsupportedOperationException.class,
                () -> prepare(fixture)
        );

        assertTrue(thrown.getMessage().contains("requires dense contiguous"));
    }

    @Test
    void unplannedStridedDenseCrossEntropyLogitsAreRejectedByCpu1LossPreparer() {
        Tensor base = new Tensor(new double[]{
                1.0d, 2.0d,
                3.0d, 4.0d,
                5.0d, 6.0d
        }, new int[]{3, 2}, null, "unplannedDenseCeBaseLogits", DataType.FLOAT64);
        Tensor logitsView = base.permute(1, 0);
        Tensor targets = new Tensor(new double[]{
                0.0d, 0.0d, 1.0d,
                1.0d, 0.0d, 0.0d
        }, new int[]{2, 3}, null, "unplannedDenseCeTargets", DataType.FLOAT64);
        Fixture fixture = fixture(logitsView.crossEntropyLoss(targets, 1));

        UnsupportedOperationException thrown = assertThrows(
                UnsupportedOperationException.class,
                () -> prepare(fixture)
        );

        assertTrue(thrown.getMessage().contains("requires dense contiguous"));
    }

    @Test
    void unplannedStridedNllLogProbsAreRejectedByCpu1LossPreparer() {
        Tensor base = new Tensor(new float[]{
                -1.0f, -0.5f,
                -2.0f, -1.5f,
                -3.0f, -2.5f
        }, new int[]{3, 2}, null, "unplannedNllBaseLogProbs", DataType.FLOAT32);
        Tensor logProbsView = base.permute(1, 0);
        Tensor targets = new Tensor(new float[]{
                0.0f, 0.0f, 1.0f,
                1.0f, 0.0f, 0.0f
        }, new int[]{2, 3}, null, "unplannedNllTargets", DataType.FLOAT32);
        Fixture fixture = fixture(logProbsView.nllLoss(targets, 1));

        UnsupportedOperationException thrown = assertThrows(
                UnsupportedOperationException.class,
                () -> prepare(fixture)
        );

        assertTrue(thrown.getMessage().contains("requires dense contiguous"));
    }

    private static void prepare(Fixture fixture) {
        new Cpu1NodePreparer().prepare(
                fixture.node(),
                fixture.descriptorIndex(),
                Cpu1PrepareConfig.scalarSingleThread()
        );
    }

    private static Fixture fixture(Tensor out) {
        List<CompiledNode> nodes = CompiledNodeSnapshotter.snapshot(out.topologicalSort(), BackendIntentPlan.empty());
        CompiledTensorDescriptorIndex descriptorIndex = CompiledTensorDescriptorBuilder.build(nodes);
        return new Fixture(descriptorIndex, nodes.getLast());
    }

    private record Fixture(
            CompiledTensorDescriptorIndex descriptorIndex,
            CompiledNode node
    ) {
    }
}
