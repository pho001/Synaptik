package io.github.pho001.synaptik.compiler;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class FunctionalGradientRequestTest {
    @Test
    void snapshotsOneStageAndRetainsExactElements() {
        Tensor output = tensor(true);
        Tensor target = tensor(true);
        FunctionalGradientRequest.Stage stage =
                FunctionalGradientTestSupport.stage(output, List.of(target));
        FunctionalGradientRequest request =
                new FunctionalGradientRequest(List.of(stage));

        assertEquals(List.of(stage), request.stages());
        assertSame(output, ((FunctionalGradientRequest.ForwardTensorReference)
                request.stages().getFirst().outputs().getFirst()).tensor());
        assertSame(target, request.stages().getFirst().targets().getFirst());
    }

    @Test
    void rejectsUnboundedStageAndCreateGraphShapes() {
        Tensor tensor = tensor(true);
        FunctionalGradientRequest.Stage one =
                FunctionalGradientTestSupport.stage(tensor, List.of(tensor));
        assertThrows(IllegalArgumentException.class, () -> new FunctionalGradientRequest(List.of()));
        assertThrows(
                IllegalArgumentException.class,
                () -> new FunctionalGradientRequest(List.of(one, one, one)));
        assertThrows(
                IllegalArgumentException.class,
                () -> new FunctionalGradientRequest(List.of(
                        new FunctionalGradientRequest.Stage(
                                one.outputs(),
                                one.cotangentSeeds(),
                                one.targets(),
                        true,
                        FunctionalGradientRequest.DisconnectedPolicy.ERROR))));
    }

    @Test
    void snapshotsEveryStageListAndAcceptsOnlyTheExactTwoStageReferenceMatrix() {
        Tensor output = tensor(true);
        Tensor target = tensor(true);
        List<FunctionalGradientRequest.OutputReference> outputs =
                new ArrayList<>(List.of(
                        new FunctionalGradientRequest.ForwardTensorReference(output)));
        List<Optional<Tensor>> seeds = new ArrayList<>(List.of(Optional.empty()));
        List<Tensor> targets = new ArrayList<>(List.of(target));
        FunctionalGradientRequest.Stage first = new FunctionalGradientRequest.Stage(
                outputs,
                seeds,
                targets,
                true,
                FunctionalGradientRequest.DisconnectedPolicy.ERROR);
        FunctionalGradientRequest.Stage second = new FunctionalGradientRequest.Stage(
                List.of(new FunctionalGradientRequest.FirstStageGradientReference(0)),
                List.of(Optional.empty()),
                List.of(target),
                false,
                FunctionalGradientRequest.DisconnectedPolicy.ZERO);
        List<FunctionalGradientRequest.Stage> stages =
                new ArrayList<>(List.of(first, second));

        FunctionalGradientRequest request = new FunctionalGradientRequest(stages);
        outputs.clear();
        seeds.clear();
        targets.clear();
        stages.clear();

        assertEquals(2, request.stages().size());
        assertEquals(1, request.stages().getFirst().outputs().size());
        assertEquals(1, request.stages().getFirst().cotangentSeeds().size());
        assertEquals(1, request.stages().getFirst().targets().size());
        assertThrows(UnsupportedOperationException.class, () -> request.stages().clear());
        assertThrows(
                UnsupportedOperationException.class,
                () -> request.stages().getFirst().targets().clear());
    }

    @Test
    void rejectsEmptyMismatchedDuplicateAndWrongDirectionStageStructures() {
        Tensor output = tensor(true);
        Tensor target = tensor(true);
        assertThrows(
                IllegalArgumentException.class,
                () -> new FunctionalGradientRequest.Stage(
                        List.of(),
                        List.of(),
                        List.of(target),
                        false,
                        FunctionalGradientRequest.DisconnectedPolicy.ERROR));
        assertThrows(
                IllegalArgumentException.class,
                () -> new FunctionalGradientRequest.Stage(
                        List.of(new FunctionalGradientRequest.ForwardTensorReference(output)),
                        List.of(),
                        List.of(target),
                        false,
                        FunctionalGradientRequest.DisconnectedPolicy.ERROR));
        assertThrows(
                IllegalArgumentException.class,
                () -> new FunctionalGradientRequest.Stage(
                        List.of(new FunctionalGradientRequest.ForwardTensorReference(output)),
                        List.of(Optional.empty()),
                        List.of(),
                        false,
                        FunctionalGradientRequest.DisconnectedPolicy.ERROR));
        assertThrows(
                IllegalArgumentException.class,
                () -> new FunctionalGradientRequest.Stage(
                        List.of(new FunctionalGradientRequest.ForwardTensorReference(output)),
                        List.of(Optional.empty()),
                        List.of(target, target),
                        false,
                        FunctionalGradientRequest.DisconnectedPolicy.ERROR));

        FunctionalGradientRequest.Stage wrongFirst = new FunctionalGradientRequest.Stage(
                List.of(new FunctionalGradientRequest.FirstStageGradientReference(0)),
                List.of(Optional.empty()),
                List.of(target),
                false,
                FunctionalGradientRequest.DisconnectedPolicy.ERROR);
        assertThrows(
                IllegalArgumentException.class,
                () -> new FunctionalGradientRequest(List.of(wrongFirst)));

        FunctionalGradientRequest.Stage first = new FunctionalGradientRequest.Stage(
                List.of(new FunctionalGradientRequest.ForwardTensorReference(output)),
                List.of(Optional.empty()),
                List.of(target),
                true,
                FunctionalGradientRequest.DisconnectedPolicy.ERROR);
        FunctionalGradientRequest.Stage wrongSecond = new FunctionalGradientRequest.Stage(
                List.of(new FunctionalGradientRequest.ForwardTensorReference(output)),
                List.of(Optional.empty()),
                List.of(target),
                false,
                FunctionalGradientRequest.DisconnectedPolicy.ERROR);
        assertThrows(
                IllegalArgumentException.class,
                () -> new FunctionalGradientRequest(List.of(first, wrongSecond)));
        assertThrows(
                IllegalArgumentException.class,
                () -> new FunctionalGradientRequest.FirstStageGradientReference(-1));
    }

    private static Tensor tensor(boolean requiresGrad) {
        return TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, Shape.scalar(), Optional.empty(), requiresGrad));
    }
}

final class FunctionalGradientTestSupport {
    private FunctionalGradientTestSupport() {}

    static FunctionalGradientRequest.Stage stage(
            Tensor objective, List<Tensor> targets) {
        return new FunctionalGradientRequest.Stage(
                List.of(new FunctionalGradientRequest.ForwardTensorReference(objective)),
                List.of(Optional.empty()),
                targets,
                false,
                FunctionalGradientRequest.DisconnectedPolicy.ERROR);
    }

    static FunctionalGradientRequest request(
            Tensor objective, List<Tensor> targets) {
        return new FunctionalGradientRequest(List.of(stage(objective, targets)));
    }
}
