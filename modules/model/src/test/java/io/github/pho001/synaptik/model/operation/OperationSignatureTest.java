package io.github.pho001.synaptik.model.operation;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.cast.CastAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.cast.CastKind;
import io.github.pho001.synaptik.model.operation.elementwise.classification.FloatingClassificationKind;
import io.github.pho001.synaptik.model.operation.elementwise.comparison.BinaryComparisonKind;
import io.github.pho001.synaptik.model.operation.elementwise.logical.BooleanLogicalKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ClampRangeAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarElementwiseKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarValueAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.selection.WhereSelectionKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.index.AxisGatherKind;
import io.github.pho001.synaptik.model.operation.index.AxisScatterKind;
import io.github.pho001.synaptik.model.operation.index.GatherNdAttrs;
import io.github.pho001.synaptik.model.operation.index.GatherNdKind;
import io.github.pho001.synaptik.model.operation.index.IndexAxisAttrs;
import io.github.pho001.synaptik.model.operation.index.OneHotAttrs;
import io.github.pho001.synaptik.model.operation.index.OneHotKind;
import io.github.pho001.synaptik.model.operation.index.ScatterElementsAttrs;
import io.github.pho001.synaptik.model.operation.index.ScatterNdAttrs;
import io.github.pho001.synaptik.model.operation.index.ScatterNdKind;
import io.github.pho001.synaptik.model.operation.index.ScatterReduction;
import io.github.pho001.synaptik.model.operation.index.SelectAttrs;
import io.github.pho001.synaptik.model.operation.index.SelectKind;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformAttrs;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformKind;
import io.github.pho001.synaptik.model.operation.layout.CompositionAxisAttrs;
import io.github.pho001.synaptik.model.operation.layout.ContiguousKind;
import io.github.pho001.synaptik.model.operation.layout.CropToShapeAttrs;
import io.github.pho001.synaptik.model.operation.layout.Fold2dAttrs;
import io.github.pho001.synaptik.model.operation.layout.FoldAxisAttrs;
import io.github.pho001.synaptik.model.operation.layout.PadAttrs;
import io.github.pho001.synaptik.model.operation.layout.PadKind;
import io.github.pho001.synaptik.model.operation.layout.PermutationAttrs;
import io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind;
import io.github.pho001.synaptik.model.operation.layout.SliceAttrs;
import io.github.pho001.synaptik.model.operation.layout.SliceKind;
import io.github.pho001.synaptik.model.operation.layout.TargetShapeAttrs;
import io.github.pho001.synaptik.model.operation.layout.TensorCompositionKind;
import io.github.pho001.synaptik.model.operation.layout.TileAttrs;
import io.github.pho001.synaptik.model.operation.layout.TileKind;
import io.github.pho001.synaptik.model.operation.layout.UnfoldAxisAttrs;
import io.github.pho001.synaptik.model.operation.layout.Unfold2dAttrs;
import io.github.pho001.synaptik.model.operation.layout.Window2dAttrs;
import io.github.pho001.synaptik.model.operation.layout.WindowTransformKind;
import io.github.pho001.synaptik.model.operation.normalization.SoftmaxAttrs;
import io.github.pho001.synaptik.model.operation.normalization.SoftmaxKind;
import io.github.pho001.synaptik.model.operation.ordering.OrderingKind;
import io.github.pho001.synaptik.model.operation.ordering.SortAttrs;
import io.github.pho001.synaptik.model.operation.ordering.TopKAttrs;
import io.github.pho001.synaptik.model.operation.ordering.TopKKind;
import io.github.pho001.synaptik.model.operation.random.DropoutAttrs;
import io.github.pho001.synaptik.model.operation.random.DropoutKind;
import io.github.pho001.synaptik.model.operation.random.GraphRngKind;
import io.github.pho001.synaptik.model.operation.random.GraphRngStateAttrs;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.ArgExtremaAttrs;
import io.github.pho001.synaptik.model.operation.reduction.AxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.MaskedReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.MultiAxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.StatisticalReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.SumToShapeAttrs;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanAttrs;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanKind;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

public final class OperationSignatureTest {
    @Test
    void validatesExactRecordShapeAndBounds() {
        var components = OperationSignature.class.getRecordComponents();

        assertAll(
                () -> assertTrue(OperationSignature.class.isRecord()),
                () -> assertEquals(
                        List.of(
                                "attributesType",
                                "minimumInputs",
                                "maximumInputs",
                                "minimumOutputs",
                                "maximumOutputs"),
                        Arrays.stream(components).map(component -> component.getName()).toList()),
                () -> assertEquals(
                        List.of(Class.class, int.class, int.class, int.class, int.class),
                        Arrays.stream(components).map(component -> component.getType()).toList()),
                () -> assertEquals(
                        "attributesType",
                        assertThrows(
                                        NullPointerException.class,
                                        () -> new OperationSignature(null, 0, 0, 1, 1))
                                .getMessage()),
                () -> assertTrue(assertThrows(
                                IllegalArgumentException.class,
                                () -> new OperationSignature(NoOperationAttrs.class, -1, 0, 1, 1))
                        .getMessage()
                        .contains("minimumInputs")),
                () -> assertTrue(assertThrows(
                                IllegalArgumentException.class,
                                () -> new OperationSignature(NoOperationAttrs.class, 2, 1, 1, 1))
                        .getMessage()
                        .contains("maximumInputs")),
                () -> assertTrue(assertThrows(
                                IllegalArgumentException.class,
                                () -> new OperationSignature(NoOperationAttrs.class, 0, 0, 0, 1))
                        .getMessage()
                        .contains("minimumOutputs")),
                () -> assertTrue(assertThrows(
                                IllegalArgumentException.class,
                                () -> new OperationSignature(NoOperationAttrs.class, 0, 0, 2, 1))
                        .getMessage()
                        .contains("maximumOutputs")));
    }

    @Test
    void supportsFixedBoundedVariadicZeroInputAndMultiOutputOccurrences() {
        OperationSignature fixed = OperationSignature.fixed(NoOperationAttrs.class, 2, 1);
        OperationSignature bounded = new OperationSignature(
                NoOperationAttrs.class, 1, 3, 1, 2);
        OperationSignature variadic = OperationSignature.inputRange(
                NoOperationAttrs.class, 1, Integer.MAX_VALUE, 1);
        OperationSignature source = OperationSignature.fixed(NoOperationAttrs.class, 0, 2);

        assertAll(
                () -> assertTrue(fixed.acceptsInputCount(2)),
                () -> assertFalse(fixed.acceptsInputCount(1)),
                () -> assertTrue(bounded.acceptsInputCount(1)),
                () -> assertTrue(bounded.acceptsInputCount(3)),
                () -> assertTrue(bounded.acceptsOutputCount(2)),
                () -> assertFalse(bounded.acceptsOutputCount(3)),
                () -> assertTrue(variadic.acceptsInputCount(Integer.MAX_VALUE)),
                () -> assertTrue(source.acceptsInputCount(0)),
                () -> assertTrue(source.acceptsOutputCount(2)),
                () -> assertTrue(assertThrows(
                                IllegalArgumentException.class,
                                () -> fixed.validateOccurrence(1, 1))
                        .getMessage()
                        .contains("input count")),
                () -> assertTrue(assertThrows(
                                IllegalArgumentException.class,
                                () -> fixed.validateOccurrence(2, 2))
                        .getMessage()
                        .contains("output count")));
    }

    @Test
    void matchesOnlyTheExactAttributesClassAndUsesRecordValueSemantics() {
        OperationSignature signature = OperationSignature.fixed(SampleAttrs.class, 1, 1);
        OperationSignature equal = OperationSignature.fixed(SampleAttrs.class, 1, 1);

        assertAll(
                () -> assertTrue(signature.acceptsAttributes(new SampleAttrs(3))),
                () -> assertFalse(signature.acceptsAttributes(NoOperationAttrs.INSTANCE)),
                () -> assertThrows(
                        NullPointerException.class, () -> signature.acceptsAttributes(null)),
                () -> assertEquals(signature, equal),
                () -> assertEquals(signature.hashCode(), equal.hashCode()),
                () -> assertSame(SampleAttrs.class, signature.attributesType()));
    }

    @Test
    void coversEveryCurrentProductionKindWithTheSelectedSignatureMatrix() {
        OperationSignature noAttrsUnary = fixed(NoOperationAttrs.class, 1);
        OperationSignature noAttrsBinary = fixed(NoOperationAttrs.class, 2);

        assertFamily(List.of(noAttrsBinary), BinaryArithmeticKind.values());
        assertFamily(List.of(fixed(CastAttrs.class, 1)), CastKind.values());
        assertFamily(List.of(noAttrsBinary), BinaryComparisonKind.values());
        assertKinds(List.of(noAttrsBinary), BooleanLogicalKind.AND, BooleanLogicalKind.OR);
        assertKinds(List.of(noAttrsUnary), BooleanLogicalKind.NOT);
        assertKinds(
                List.of(fixed(ScalarValueAttrs.class, 1)),
                ScalarElementwiseKind.ADD,
                ScalarElementwiseKind.SUB,
                ScalarElementwiseKind.MUL,
                ScalarElementwiseKind.DIV,
                ScalarElementwiseKind.MIN,
                ScalarElementwiseKind.MAX,
                ScalarElementwiseKind.POW);
        assertKinds(
                List.of(fixed(ClampRangeAttrs.class, 1)), ScalarElementwiseKind.CLAMP);
        assertFamily(List.of(fixed(NoOperationAttrs.class, 3)), WhereSelectionKind.values());
        assertFamily(List.of(noAttrsUnary), UnaryElementwiseKind.values());
        assertFamily(List.of(noAttrsUnary), FloatingClassificationKind.values());

        assertFamily(List.of(fixed(IndexAxisAttrs.class, 2)), AxisGatherKind.values());
        assertKinds(
                List.of(fixed(ScatterElementsAttrs.class, 3)),
                AxisScatterKind.SCATTER_ELEMENTS);
        assertKinds(
                List.of(fixed(IndexAxisAttrs.class, 3)),
                AxisScatterKind.SCATTER_ADD);
        assertFamily(List.of(fixed(GatherNdAttrs.class, 2)), GatherNdKind.values());
        assertFamily(List.of(fixed(ScatterNdAttrs.class, 3)), ScatterNdKind.values());
        assertFamily(List.of(fixed(SelectAttrs.class, 1)), SelectKind.values());
        assertFamily(List.of(fixed(OneHotAttrs.class, 1)), OneHotKind.values());

        assertKinds(
                List.of(fixed(PermutationAttrs.class, 1)), AxisTransformKind.PERMUTE);
        assertKinds(
                List.of(fixed(AxisTransformAttrs.class, 1)),
                AxisTransformKind.EXPAND_DIMS,
                AxisTransformKind.SQUEEZE);
        assertFamily(List.of(noAttrsUnary), ContiguousKind.values());
        assertFamily(List.of(fixed(PadAttrs.class, 1)), PadKind.values());
        assertFamily(List.of(fixed(TargetShapeAttrs.class, 1)), ShapeTransformKind.values());
        assertKinds(
                List.of(
                        fixed(SliceAttrs.class, 1),
                        fixed(CropToShapeAttrs.class, 1)),
                SliceKind.SLICE);
        assertKinds(
                List.of(fixed(SliceAttrs.class, 2)),
                SliceKind.SLICE_UPDATE);
        assertKinds(
                List.of(OperationSignature.inputRange(
                        CompositionAxisAttrs.class, 1, Integer.MAX_VALUE, 1)),
                TensorCompositionKind.CONCAT,
                TensorCompositionKind.STACK);
        assertFamily(List.of(fixed(TileAttrs.class, 1)), TileKind.values());
        assertKinds(
                List.of(fixed(UnfoldAxisAttrs.class, 1)),
                WindowTransformKind.UNFOLD_AXIS);
        assertKinds(
                List.of(fixed(FoldAxisAttrs.class, 1)), WindowTransformKind.FOLD_AXIS);
        assertKinds(
                List.of(fixed(Window2dAttrs.class, 1), fixed(Unfold2dAttrs.class, 1)),
                WindowTransformKind.UNFOLD2D);
        assertKinds(List.of(fixed(Fold2dAttrs.class, 1)), WindowTransformKind.FOLD2D);

        assertFamily(List.of(fixed(SoftmaxAttrs.class, 1)), SoftmaxKind.values());
        assertFamily(List.of(fixed(SortAttrs.class, 1)), OrderingKind.values());
        assertFamily(
                List.of(OperationSignature.fixed(TopKAttrs.class, 1, 2)),
                TopKKind.values());
        assertFamily(
                List.of(OperationSignature.fixed(DropoutAttrs.class, 2, 3)),
                DropoutKind.values());
        assertFamily(
                List.of(OperationSignature.fixed(GraphRngStateAttrs.class, 0, 1)),
                GraphRngKind.values());
        List<OperationSignature> maskedReduction = List.of(
                noAttrsUnary,
                fixed(AxisReductionAttrs.class, 1),
                fixed(MultiAxisReductionAttrs.class, 1),
                fixed(MaskedReductionAttrs.class, 2));
        List<OperationSignature> sumToShapeReduction = List.of(
                noAttrsUnary,
                fixed(AxisReductionAttrs.class, 1),
                fixed(MultiAxisReductionAttrs.class, 1),
                fixed(MaskedReductionAttrs.class, 2),
                fixed(SumToShapeAttrs.class, 1));
        List<OperationSignature> ordinaryReduction =
                List.of(
                        noAttrsUnary,
                        fixed(AxisReductionAttrs.class, 1),
                        fixed(MultiAxisReductionAttrs.class, 1));
        assertKinds(sumToShapeReduction, AggregateReductionKind.SUM);
        assertKinds(maskedReduction, AggregateReductionKind.MEAN);
        assertKinds(
                ordinaryReduction,
                AggregateReductionKind.PROD,
                AggregateReductionKind.MIN,
                AggregateReductionKind.MAX,
                AggregateReductionKind.ALL,
                AggregateReductionKind.ANY);
        assertKinds(
                List.of(fixed(ArgExtremaAttrs.class, 1)),
                AggregateReductionKind.ARG_MAX,
                AggregateReductionKind.ARG_MIN);
        assertKinds(
                List.of(fixed(MultiAxisReductionAttrs.class, 1)),
                AggregateReductionKind.LOG_SUM_EXP,
                AggregateReductionKind.L1_NORM,
                AggregateReductionKind.L2_NORM);
        assertKinds(
                List.of(fixed(StatisticalReductionAttrs.class, 1)),
                AggregateReductionKind.VARIANCE,
                AggregateReductionKind.STANDARD_DEVIATION);
        assertFamily(
                List.of(fixed(CumulativeScanAttrs.class, 1)), CumulativeScanKind.values());
    }

    @Test
    void rejectsRepresentativeCrossFamilyAttributePairingsDuringOperationConstruction() {
        assertAll(
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> new Operation(
                                BinaryArithmeticKind.ADD, new SelectAttrs(0, 0))),
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> new Operation(CastKind.CAST, NoOperationAttrs.INSTANCE)),
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> new Operation(
                                ScalarElementwiseKind.MUL, NoOperationAttrs.INSTANCE)),
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> new Operation(
                                FloatingClassificationKind.IS_FINITE,
                                new SelectAttrs(0, 0))),
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> new Operation(
                                AggregateReductionKind.SUM, new SelectAttrs(0, 0))),
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> new Operation(
                                ShapeTransformKind.RESHAPE, NoOperationAttrs.INSTANCE)),
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> new Operation(
                                AxisGatherKind.GATHER, NoOperationAttrs.INSTANCE)),
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> new Operation(
                                AxisScatterKind.SCATTER_ELEMENTS, new IndexAxisAttrs(0))),
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> new Operation(
                                AxisScatterKind.SCATTER_ADD,
                                new ScatterElementsAttrs(0, ScatterReduction.ADD))),
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> new Operation(
                                SliceKind.SLICE_UPDATE,
                                new CropToShapeAttrs(
                                        io.github.pho001.synaptik.model.shape.Shape.scalar(),
                                        io.github.pho001.synaptik.model.shape.Shape.scalar()))),
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> new Operation(
                                SliceKind.SLICE,
                                NoOperationAttrs.INSTANCE)),
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> new Operation(
                                WindowTransformKind.UNFOLD_AXIS,
                                NoOperationAttrs.INSTANCE)));
    }

    public static void assertSignatureEnumShape(Class<? extends Enum<?>> enumType) {
        var instanceFields = Arrays.stream(enumType.getDeclaredFields())
                .filter(field -> !Modifier.isStatic(field.getModifiers()))
                .toList();
        var signatureFields = Arrays.stream(enumType.getDeclaredFields())
                .filter(field -> !field.isEnumConstant())
                .filter(field -> !field.isSynthetic())
                .toList();
        var instanceMethods = Arrays.stream(enumType.getDeclaredMethods())
                .filter(method -> !Modifier.isStatic(method.getModifiers()))
                .filter(method -> !method.isSynthetic())
                .toList();

        assertAll(
                () -> assertTrue(Modifier.isPublic(enumType.getModifiers())),
                () -> assertTrue(Modifier.isFinal(enumType.getModifiers())),
                () -> assertTrue(enumType.isEnum()),
                () -> assertEquals(List.of(OperationKind.class), List.of(enumType.getInterfaces())),
                () -> assertTrue(instanceFields.isEmpty()),
                () -> assertTrue(signatureFields.stream().allMatch(field ->
                        Modifier.isPrivate(field.getModifiers())
                                && Modifier.isStatic(field.getModifiers())
                                && Modifier.isFinal(field.getModifiers())
                                && field.getType() == List.class)),
                () -> assertEquals(1, instanceMethods.size()),
                () -> assertEquals("signatures", instanceMethods.getFirst().getName()),
                () -> assertEquals(List.class, instanceMethods.getFirst().getReturnType()),
                () -> assertEquals(0, enumType.getDeclaredClasses().length));
    }

    private static OperationSignature fixed(
            Class<? extends OperationAttrs> attributesType, int inputCount) {
        return OperationSignature.fixed(attributesType, inputCount, 1);
    }

    private static void assertFamily(
            List<OperationSignature> expected, OperationKind[] kinds) {
        assertKinds(expected, kinds);
    }

    private static void assertKinds(
            List<OperationSignature> expected, OperationKind... kinds) {
        for (OperationKind kind : kinds) {
            assertEquals(expected, kind.signatures(), () -> "signatures for " + kind);
        }
    }

    private record SampleAttrs(int value) implements OperationAttrs {}
}
