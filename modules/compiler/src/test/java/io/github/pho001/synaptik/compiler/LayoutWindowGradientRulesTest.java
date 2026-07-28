package io.github.pho001.synaptik.compiler;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;

import io.github.pho001.synaptik.config.compile.CompileMode;
import io.github.pho001.synaptik.config.compile.GraphOptimizationConfig;
import io.github.pho001.synaptik.model.graph.GraphPhase;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.layout.CropToShapeAttrs;
import io.github.pho001.synaptik.model.operation.layout.SliceAttrs;
import io.github.pho001.synaptik.model.operation.layout.SliceKind;
import io.github.pho001.synaptik.model.operation.layout.Window2dAttrs;
import io.github.pho001.synaptik.model.operation.layout.WindowTransformKind;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

final class LayoutWindowGradientRulesTest {
    @Test
    void sliceUpdateUsesLengthDefinedExtractionAndTargetRelativeForms() {
        Tensor dynamicBase = tensor(Shape.ofDimensions(new DynamicDimension("N")));
        Tensor update = tensor(Shape.of(2));
        Tensor finite = dynamicBase.sliceUpdate(
                update, new long[] {3}, new int[] {0}, new long[] {-1});
        Tensor updateGradient = gradient(finite.sum(), update);
        assertEquals(SliceKind.SLICE,
                updateGradient.provenance().orElseThrow().operation().kind());
        assertInstanceOf(
                SliceAttrs.class,
                updateGradient.provenance().orElseThrow().operation().attrs());

        Tensor base = tensor(Shape.of(5));
        Tensor placed = base.sliceUpdate(update, Shape.of(2));
        Tensor baseGradient = gradient(placed.sum(), base);
        assertEquals(SliceKind.SLICE_UPDATE,
                baseGradient.provenance().orElseThrow().operation().kind());
        assertInstanceOf(
                CropToShapeAttrs.class,
                baseGradient.provenance().orElseThrow().operation().attrs());
        Tensor placedUpdateGradient = gradient(placed.sum(), update);
        assertEquals(SliceKind.SLICE,
                placedUpdateGradient.provenance().orElseThrow().operation().kind());
        assertInstanceOf(
                CropToShapeAttrs.class,
                placedUpdateGradient.provenance().orElseThrow().operation().attrs());
        assertCompiles(placed.sum(), base);
        assertCompiles(finite.sum(), update);
    }

    @Test
    void generalAxisAndTwoDimensionalWindowsUseExactAdjoints() {
        Tensor axisInput = tensor(Shape.of(5));
        Tensor unfoldedGradient = gradient(axisInput.unfold(0, 3, 1).sum(), axisInput);
        assertEquals(WindowTransformKind.FOLD_AXIS,
                unfoldedGradient.provenance().orElseThrow().operation().kind());

        Tensor columns = tensor(Shape.of(3, 2));
        Tensor foldedGradient = gradient(columns.foldAxis(0, 4, 1).sum(), columns);
        assertEquals(WindowTransformKind.UNFOLD_AXIS,
                foldedGradient.provenance().orElseThrow().operation().kind());

        Window2dAttrs window = new Window2dAttrs(2, 2, 1, 1, 0, 0, 1, 1, false);
        Tensor image = tensor(Shape.of(1, 1, 3, 3));
        Tensor imageGradient = gradient(image.unfold2d(window).sum(), image);
        assertEquals(WindowTransformKind.FOLD2D,
                imageGradient.provenance().orElseThrow().operation().kind());

        Tensor imageColumns = tensor(Shape.of(1, 4, 4));
        Tensor columnsGradient =
                gradient(imageColumns.fold2d(Shape.of(1, 1, 3, 3), window).sum(), imageColumns);
        assertEquals(WindowTransformKind.UNFOLD2D,
                columnsGradient.provenance().orElseThrow().operation().kind());
        assertCompiles(image.unfold2d(window).sum(), image);
        assertCompiles(
                imageColumns.fold2d(Shape.of(1, 1, 3, 3), window).sum(),
                imageColumns);
    }

    private static Tensor gradient(Tensor objective, Tensor target) {
        var plan = AutogradPreflight.preflight(
                CompileMode.FORWARD_AND_BACKWARD,
                List.of(objective),
                new AutogradPreflight.FirstOrderRequest(objective, List.of(target)),
                CompileTimeConstantGraph.Ingress.empty());
        return FirstOrderAutograd.expand(plan, CompileTimeConstantGraph.Ingress.empty())
                .targetGradients().getFirst().gradient();
    }

    private static void assertCompiles(Tensor objective, Tensor target) {
        GraphCompilation compilation = GraphCompiler.compile(
                CompileMode.FORWARD_AND_BACKWARD,
                List.of(objective),
                Optional.of(new AutogradPreflight.FirstOrderRequest(
                        objective, List.of(target))),
                CompileTimeConstantGraph.Ingress.empty(),
                GraphOptimizationConfig.disabled());
        assertEquals(target.id(), compilation.gradientResults().getFirst().target());
        assertEquals(
                GraphPhase.BACKWARD,
                compilation.validatedGraph().graph().nodePhases().values().stream()
                        .filter(phase -> phase == GraphPhase.BACKWARD)
                        .findFirst()
                        .orElseThrow());
    }

    private static Tensor tensor(Shape shape) {
        return TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, shape, Optional.empty(), true));
    }
}
