package io.github.pho001.synaptik.backend.metal;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.config.compile.NumericalProfile;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.comparison.BinaryComparisonKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarElementwiseKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarValueAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.index.AxisGatherKind;
import io.github.pho001.synaptik.model.operation.index.IndexAxisAttrs;
import io.github.pho001.synaptik.model.operation.index.OneHotAttrs;
import io.github.pho001.synaptik.model.operation.index.OneHotKind;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformAttrs;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformKind;
import io.github.pho001.synaptik.model.operation.layout.ContiguousKind;
import io.github.pho001.synaptik.model.operation.layout.PermutationAttrs;
import io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind;
import io.github.pho001.synaptik.model.operation.layout.TargetShapeAttrs;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.linalg.MatmulKind;
import io.github.pho001.synaptik.model.operation.reduction.AxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.MaskedReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.MultiAxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.SumToShapeAttrs;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class MetalCapabilityProviderTest {
    private final MetalCapabilityProvider provider = new MetalCapabilityProvider();

    @Test
    void enforcesClosedStrictAndAcceleratorCapabilityMatrices() {
        assertSame(MetalCapabilityProvider.METAL_BACKEND_ID, provider.backendId());
        assertEquals("metal", provider.backendId().value());
        TensorDescriptor matrix = descriptor(Shape.of(2, 3), false);
        TensorDescriptor row = descriptor(Shape.of(3), false);
        assertTrue(provider.supports(unaryQuery(
                NumericalProfile.STRICT_IEEE, UnaryElementwiseKind.NEG, matrix, matrix)));
        assertTrue(provider.supports(unaryQuery(
                NumericalProfile.STRICT_IEEE, UnaryElementwiseKind.ABS, matrix, matrix)));
        assertTrue(provider.supports(unaryQuery(
                NumericalProfile.ACCELERATOR, UnaryElementwiseKind.ABS, matrix, matrix)));
        assertTrue(provider.supports(unaryQuery(
                NumericalProfile.ACCELERATOR, UnaryElementwiseKind.NEG, matrix, matrix)));
        Shape rank16 = Shape.of(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        assertTrue(provider.supports(unaryQuery(
                NumericalProfile.STRICT_IEEE,
                UnaryElementwiseKind.ABS,
                descriptor(rank16, true),
                descriptor(rank16, true))));

        for (BinaryArithmeticKind kind : BinaryArithmeticKind.values()) {
            boolean supported = kind == BinaryArithmeticKind.ADD
                    || kind == BinaryArithmeticKind.SUB
                    || kind == BinaryArithmeticKind.MUL
                    || kind == BinaryArithmeticKind.DIV;
            assertFalse(provider.supports(binaryQuery(
                    NumericalProfile.STRICT_IEEE, kind, matrix, row, matrix)));
            assertEquals(supported, provider.supports(binaryQuery(
                    NumericalProfile.ACCELERATOR, kind, matrix, row, matrix)));
            assertEquals(supported, provider.supports(binaryQuery(
                    NumericalProfile.ACCELERATOR, kind, row, matrix, matrix)));
        }
        for (UnaryElementwiseKind kind : UnaryElementwiseKind.values()) {
            assertEquals(kind == UnaryElementwiseKind.NEG || kind == UnaryElementwiseKind.ABS,
                    provider.supports(unaryQuery(
                            NumericalProfile.STRICT_IEEE, kind, matrix, matrix)),
                    "strict " + kind);
            assertEquals(kind == UnaryElementwiseKind.NEG || kind == UnaryElementwiseKind.ABS,
                    provider.supports(unaryQuery(
                            NumericalProfile.ACCELERATOR, kind, matrix, matrix)),
                    "accelerator " + kind);
        }
        TensorDescriptor reshapedView = new TensorDescriptor(
                DataType.FLOAT32,
                matrix.shape(),
                Optional.of(LayoutDescriptor.of(
                        matrix.shape(), new long[] {3, 1}, 0L, true)),
                false);
        Operation reshape = new Operation(
                ShapeTransformKind.RESHAPE,
                new TargetShapeAttrs(matrix.shape()));
        var strictReshape = new OperationCapabilityQuery(
                NumericalProfile.STRICT_IEEE,
                reshape,
                List.of(matrix),
                List.of(reshapedView));
        var acceleratorReshape = new OperationCapabilityQuery(
                NumericalProfile.ACCELERATOR,
                reshape,
                List.of(matrix),
                List.of(reshapedView));
        assertTrue(provider.supports(strictReshape));
        assertTrue(provider.supports(acceleratorReshape));
    }

    @Test
    void everyStrictExactRepresentativeIsAlsoAcceleratorSupported() {
        TensorDescriptor matrix = descriptor(Shape.of(2, 3), false);
        TensorDescriptor flat = descriptor(Shape.of(6), false);
        TensorDescriptor singletonRow = descriptor(Shape.of(1, 3), false);
        TensorDescriptor singletonMatrix = descriptor(Shape.of(2, 1, 3), false);
        TensorDescriptor reshaped = view(Shape.of(2, 3), 3, 1);
        TensorDescriptor expanded = view(Shape.of(2, 3), 0, 1);
        TensorDescriptor permuted = view(Shape.of(3, 2), 1, 3);
        TensorDescriptor rankExpanded = view(Shape.of(2, 1, 3), 3, 3, 1);
        TensorDescriptor squeezed = view(Shape.of(2, 3), 3, 1);
        Shape rank16 = Shape.of(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        TensorDescriptor rank16Value = descriptor(rank16, false);
        List<Occurrence> common = List.of(
                new Occurrence(
                        "NEG",
                        new Operation(UnaryElementwiseKind.NEG, NoOperationAttrs.INSTANCE),
                        matrix,
                        matrix),
                new Occurrence(
                        "ABS",
                        new Operation(UnaryElementwiseKind.ABS, NoOperationAttrs.INSTANCE),
                        matrix,
                        matrix),
                new Occurrence(
                        "RESHAPE",
                        new Operation(
                                ShapeTransformKind.RESHAPE,
                                new TargetShapeAttrs(Shape.of(2, 3))),
                        flat,
                        reshaped),
                new Occurrence(
                        "EXPAND",
                        new Operation(
                                ShapeTransformKind.EXPAND,
                                new TargetShapeAttrs(Shape.of(2, 3))),
                        singletonRow,
                        expanded),
                new Occurrence(
                        "PERMUTE",
                        new Operation(
                                AxisTransformKind.PERMUTE,
                                new PermutationAttrs(List.of(1, 0))),
                        matrix,
                        permuted),
                new Occurrence(
                        "EXPAND_DIMS",
                        new Operation(
                                AxisTransformKind.EXPAND_DIMS,
                                new AxisTransformAttrs(1)),
                        matrix,
                        rankExpanded),
                new Occurrence(
                        "SQUEEZE",
                        new Operation(
                                AxisTransformKind.SQUEEZE,
                                new AxisTransformAttrs(1)),
                        singletonMatrix,
                        squeezed),
                new Occurrence(
                        "CONTIGUOUS",
                        new Operation(ContiguousKind.CONTIGUOUS, NoOperationAttrs.INSTANCE),
                        expanded,
                        matrix),
                new Occurrence(
                        "rank-16 NEG boundary",
                        new Operation(UnaryElementwiseKind.NEG, NoOperationAttrs.INSTANCE),
                        rank16Value,
                        rank16Value));

        for (Occurrence occurrence : common) {
            assertTrue(provider.supports(new OperationCapabilityQuery(
                    NumericalProfile.STRICT_IEEE,
                    occurrence.operation(),
                    List.of(occurrence.input()),
                    List.of(occurrence.output()))), occurrence.name() + " strict");
            assertTrue(provider.supports(new OperationCapabilityQuery(
                    NumericalProfile.ACCELERATOR,
                    occurrence.operation(),
                    List.of(occurrence.input()),
                    List.of(occurrence.output()))), occurrence.name() + " accelerator");
        }
    }

    @Test
    void admitsOnlyExactInt32GatherAndOneHotOccurrencesInBothProfiles() {
        TensorDescriptor data = typed(DataType.FLOAT32, Shape.of(2, 3, 4), true);
        TensorDescriptor indices = typed(DataType.INT32, Shape.of(5, 6), false);
        TensorDescriptor gathered = typed(DataType.FLOAT32, Shape.of(2, 5, 6, 4), true);
        Operation gather = new Operation(AxisGatherKind.GATHER, new IndexAxisAttrs(1));
        TensorDescriptor oneHotIndices = typed(DataType.INT32, Shape.of(2, 3), false);
        TensorDescriptor oneHot = typed(DataType.BOOL, Shape.of(2, 3, 4), false);
        Operation encode = new Operation(OneHotKind.ONE_HOT, new OneHotAttrs(4));

        for (NumericalProfile profile : NumericalProfile.values()) {
            assertTrue(provider.supports(new OperationCapabilityQuery(
                    profile, gather, List.of(data, indices), List.of(gathered))));
            assertTrue(provider.supports(new OperationCapabilityQuery(
                    profile, encode, List.of(oneHotIndices), List.of(oneHot))));
        }

        assertFalse(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.STRICT_IEEE,
                new Operation(AxisGatherKind.GATHER_ELEMENTS, new IndexAxisAttrs(1)),
                List.of(data, indices),
                List.of(gathered))));
        assertFalse(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.ACCELERATOR,
                gather,
                List.of(data, typed(DataType.INT64, Shape.of(5, 6), false)),
                List.of(gathered))));
        assertFalse(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.ACCELERATOR,
                gather,
                List.of(data, indices),
                List.of(typed(DataType.FLOAT32, Shape.of(2, 5, 4), true)))));
        assertFalse(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.ACCELERATOR,
                gather,
                List.of(data, indices),
                List.of(typed(DataType.FLOAT32, gathered.shape(), false)))));
        assertFalse(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.STRICT_IEEE,
                encode,
                List.of(typed(DataType.INT64, Shape.of(2, 3), false)),
                List.of(oneHot))));
        assertFalse(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.STRICT_IEEE,
                encode,
                List.of(oneHotIndices),
                List.of(typed(DataType.BOOL, Shape.of(2, 3, 5), false)))));
    }

    @Test
    void rejectsEveryBoundaryOutsideTheExactDomain() {
        TensorDescriptor valid = descriptor(Shape.of(2, 3), false);
        Shape dynamic = Shape.ofDimensions(new DynamicDimension("N"));
        Shape rank17 = Shape.of(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        var noLayout = new TensorDescriptor(
                DataType.FLOAT32, Shape.of(2, 3), Optional.empty(), false);
        var offset = new TensorDescriptor(
                DataType.FLOAT32,
                Shape.of(2, 3),
                Optional.of(LayoutDescriptor.of(Shape.of(2, 3), new long[] {3, 1}, 1, true)),
                false);
        var view = new TensorDescriptor(
                DataType.FLOAT32,
                Shape.of(2, 3),
                Optional.of(LayoutDescriptor.of(Shape.of(2, 3), new long[] {3, 1}, 0, true)),
                false);
        var strided = new TensorDescriptor(
                DataType.FLOAT32,
                Shape.of(2, 3),
                Optional.of(LayoutDescriptor.of(Shape.of(2, 3), new long[] {1, 2}, 0, false)),
                false);

        assertTrue(provider.supports(unaryQuery(
                NumericalProfile.STRICT_IEEE, UnaryElementwiseKind.ABS, valid, valid)));
        assertTrue(provider.supports(unaryQuery(
                NumericalProfile.ACCELERATOR, UnaryElementwiseKind.ABS, valid, valid)));
        for (NumericalProfile profile : NumericalProfile.values()) {
            assertFalse(provider.supports(unaryQuery(
                    profile, UnaryElementwiseKind.ABS, typed(DataType.FLOAT64),
                    typed(DataType.FLOAT64))));
            assertFalse(provider.supports(unaryQuery(
                    profile, UnaryElementwiseKind.ABS, valid,
                    descriptor(Shape.of(3, 2), false))));
            assertFalse(provider.supports(unaryQuery(
                    profile, UnaryElementwiseKind.ABS, valid,
                    descriptor(Shape.of(2, 3), true))));
            assertFalse(provider.supports(unaryQuery(
                    profile, UnaryElementwiseKind.ABS, view, valid)));
            assertFalse(provider.supports(unaryQuery(
                    profile, UnaryElementwiseKind.ABS, descriptor(Shape.of(), false),
                    descriptor(Shape.of(), false))));
            assertFalse(provider.supports(unaryQuery(
                    profile, UnaryElementwiseKind.ABS, descriptor(rank17, false),
                    descriptor(rank17, false))));
        }
        assertFalse(provider.supports(new OperationCapabilityQuery(NumericalProfile.ACCELERATOR, new Operation(BinaryComparisonKind.GREATER_THAN, NoOperationAttrs.INSTANCE), List.of(valid, valid), List.of(valid))));
        assertFalse(provider.supports(new OperationCapabilityQuery(NumericalProfile.ACCELERATOR, new Operation(
                ScalarElementwiseKind.ADD,
                new ScalarValueAttrs(ScalarValue.float32(1.0f))), List.of(valid), List.of(valid))));
        assertFalse(provider.supports(query(
                NumericalProfile.STRICT_IEEE, typed(DataType.FLOAT64), typed(DataType.FLOAT64))));
        assertFalse(provider.supports(query(
                NumericalProfile.STRICT_IEEE, valid, descriptor(Shape.of(3, 2), false))));
        assertFalse(provider.supports(query(
                NumericalProfile.STRICT_IEEE, valid, descriptor(Shape.of(2, 3), true))));
        assertFalse(provider.supports(query(
                NumericalProfile.STRICT_IEEE,
                new TensorDescriptor(DataType.FLOAT32, dynamic, Optional.empty(), false),
                new TensorDescriptor(DataType.FLOAT32, dynamic, Optional.empty(), false))));
        assertFalse(provider.supports(query(
                NumericalProfile.STRICT_IEEE,
                descriptor(Shape.of(), false), descriptor(Shape.of(), false))));
        assertFalse(provider.supports(query(
                NumericalProfile.STRICT_IEEE,
                descriptor(Shape.of(2, 0), false), descriptor(Shape.of(2, 0), false))));
        assertFalse(provider.supports(query(
                NumericalProfile.STRICT_IEEE,
                descriptor(rank17, false), descriptor(rank17, false))));
        assertFalse(provider.supports(query(NumericalProfile.STRICT_IEEE, noLayout, noLayout)));
        assertFalse(provider.supports(query(NumericalProfile.STRICT_IEEE, offset, offset)));
        assertFalse(provider.supports(query(NumericalProfile.STRICT_IEEE, view, view)));
        assertFalse(provider.supports(query(NumericalProfile.STRICT_IEEE, strided, strided)));

        assertFalse(provider.supports(binaryQuery(
                NumericalProfile.ACCELERATOR, BinaryArithmeticKind.ADD, valid, valid,
                descriptor(Shape.of(3, 2), false))));
        assertFalse(provider.supports(binaryQuery(
                NumericalProfile.ACCELERATOR, BinaryArithmeticKind.ADD, valid, valid,
                descriptor(Shape.of(2, 3), true))));
        assertFalse(provider.supports(binaryQuery(
                NumericalProfile.ACCELERATOR, BinaryArithmeticKind.ADD, view, valid, valid)));
        TensorDescriptor float64 = typed(DataType.FLOAT64);
        assertFalse(provider.supports(binaryQuery(
                NumericalProfile.ACCELERATOR,
                BinaryArithmeticKind.ADD,
                float64,
                float64,
                float64)));
        TensorDescriptor scalarRank = descriptor(Shape.of(), false);
        assertFalse(provider.supports(binaryQuery(
                NumericalProfile.ACCELERATOR,
                BinaryArithmeticKind.ADD,
                scalarRank,
                scalarRank,
                scalarRank)));
        TensorDescriptor overRank = descriptor(rank17, false);
        assertFalse(provider.supports(binaryQuery(
                NumericalProfile.ACCELERATOR,
                BinaryArithmeticKind.ADD,
                overRank,
                overRank,
                overRank)));

    }

    @Test
    void acceleratorReductionMatrixIsExactAndStrictRemainsClosed() {
        TensorDescriptor input = descriptor(Shape.of(2, 3, 4), false);
        TensorDescriptor scalar = descriptor(Shape.scalar(), false);
        assertTrue(provider.supports(reductionQuery(
                NumericalProfile.ACCELERATOR,
                AggregateReductionKind.SUM,
                NoOperationAttrs.INSTANCE,
                input,
                scalar)));
        assertFalse(provider.supports(reductionQuery(
                NumericalProfile.STRICT_IEEE,
                AggregateReductionKind.SUM,
                NoOperationAttrs.INSTANCE,
                input,
                scalar)));
        assertTrue(provider.supports(reductionQuery(
                NumericalProfile.ACCELERATOR,
                AggregateReductionKind.MEAN,
                new AxisReductionAttrs(1, true),
                input,
                descriptor(Shape.of(2, 1, 4), false))));
        assertTrue(provider.supports(reductionQuery(
                NumericalProfile.ACCELERATOR,
                AggregateReductionKind.SUM,
                new MultiAxisReductionAttrs(List.of(2, 0), false),
                input,
                descriptor(Shape.of(3), false))));
        assertTrue(provider.supports(reductionQuery(
                NumericalProfile.ACCELERATOR,
                AggregateReductionKind.MEAN,
                new MultiAxisReductionAttrs(List.of(), true),
                input,
                input)));
        assertTrue(provider.supports(reductionQuery(
                NumericalProfile.ACCELERATOR,
                AggregateReductionKind.SUM,
                new SumToShapeAttrs(Shape.of(1, 4)),
                input,
                descriptor(Shape.of(1, 4), false))));

        TensorDescriptor view = new TensorDescriptor(
                DataType.FLOAT32,
                input.shape(),
                Optional.of(LayoutDescriptor.of(
                        input.shape(), new long[] {12, 4, 1}, 0L, true)),
                false);
        TensorDescriptor mask = new TensorDescriptor(
                DataType.BOOL,
                Shape.of(3),
                Optional.of(LayoutDescriptor.contiguous(Shape.of(3))),
                false);
        assertFalse(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.ACCELERATOR,
                new Operation(AggregateReductionKind.SUM, new MaskedReductionAttrs(1)),
                List.of(input, mask),
                List.of(descriptor(Shape.of(2, 4), false)))));
        assertFalse(provider.supports(reductionQuery(
                NumericalProfile.ACCELERATOR,
                AggregateReductionKind.SUM,
                new AxisReductionAttrs(3, false),
                input,
                descriptor(Shape.of(2, 3), false))));
        assertFalse(provider.supports(reductionQuery(
                NumericalProfile.ACCELERATOR,
                AggregateReductionKind.SUM,
                new SumToShapeAttrs(Shape.of(2, 2)),
                input,
                descriptor(Shape.of(2, 2), false))));
        assertFalse(provider.supports(reductionQuery(
                NumericalProfile.ACCELERATOR,
                AggregateReductionKind.SUM,
                NoOperationAttrs.INSTANCE,
                view,
                scalar)));
        assertFalse(provider.supports(reductionQuery(
                NumericalProfile.ACCELERATOR,
                AggregateReductionKind.SUM,
                NoOperationAttrs.INSTANCE,
                scalar,
                scalar)));
    }

    @Test
    void acceleratorMatmulAdmitsOnlyExactRankTwoShapesAndTransposeCandidates() {
        TensorDescriptor left = descriptor(Shape.of(2, 3), true);
        TensorDescriptor right = descriptor(Shape.of(3, 4), false);
        TensorDescriptor output = descriptor(Shape.of(2, 4), true);
        Operation matmul = new Operation(MatmulKind.MATMUL, NoOperationAttrs.INSTANCE);
        assertTrue(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.ACCELERATOR,
                matmul,
                List.of(left, right),
                List.of(output))));
        assertFalse(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.STRICT_IEEE,
                matmul,
                List.of(left, right),
                List.of(output))));

        TensorDescriptor leftTranspose = transposeCandidate(Shape.of(2, 3), true);
        TensorDescriptor rightTranspose = transposeCandidate(Shape.of(3, 4), false);
        assertTrue(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.ACCELERATOR,
                matmul,
                List.of(leftTranspose, rightTranspose),
                List.of(output))));
        assertTrue(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.ACCELERATOR,
                new Operation(
                        AxisTransformKind.PERMUTE,
                        new PermutationAttrs(List.of(1, 0))),
                List.of(descriptor(Shape.of(3, 2), true)),
                List.of(leftTranspose))));

        assertFalse(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.ACCELERATOR,
                matmul,
                List.of(left, descriptor(Shape.of(2, 4), false)),
                List.of(output))));
        assertFalse(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.ACCELERATOR,
                matmul,
                List.of(left, right),
                List.of(descriptor(Shape.of(2, 5), true)))));
        assertFalse(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.ACCELERATOR,
                matmul,
                List.of(left, right),
                List.of(descriptor(Shape.of(2, 4), false)))));
        assertFalse(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.ACCELERATOR,
                matmul,
                List.of(
                        new TensorDescriptor(
                                DataType.FLOAT32,
                                left.shape(),
                                Optional.of(LayoutDescriptor.of(
                                        left.shape(), new long[] {2, 1}, 0L, true)),
                                true),
                        right),
                List.of(output))));
        assertFalse(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.ACCELERATOR,
                matmul,
                List.of(
                        descriptor(Shape.of(2, 0), true),
                        descriptor(Shape.of(0, 4), false)),
                List.of(output))));
        TensorDescriptor vector = descriptor(Shape.of(3), true);
        assertFalse(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.ACCELERATOR,
                matmul,
                List.of(vector, right),
                List.of(output))));
        TensorDescriptor float64 = new TensorDescriptor(
                DataType.FLOAT64,
                Shape.of(2, 3),
                Optional.of(LayoutDescriptor.contiguous(Shape.of(2, 3))),
                true);
        assertFalse(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.ACCELERATOR,
                matmul,
                List.of(float64, right),
                List.of(output))));
    }

    @Test
    void rejectsNullWithTheContractMessage() {
        var failure = assertThrows(NullPointerException.class, () -> provider.supports(null));
        assertEquals("query", failure.getMessage());
    }

    private static OperationCapabilityQuery query(
            NumericalProfile profile, TensorDescriptor input, TensorDescriptor output) {
        return unaryQuery(profile, UnaryElementwiseKind.NEG, input, output);
    }

    private static OperationCapabilityQuery unaryQuery(
            NumericalProfile profile,
            UnaryElementwiseKind kind,
            TensorDescriptor input,
            TensorDescriptor output) {
        return new OperationCapabilityQuery(
                profile,
                new Operation(kind, NoOperationAttrs.INSTANCE),
                List.of(input),
                List.of(output));
    }

    private static OperationCapabilityQuery binaryQuery(
            NumericalProfile profile,
            BinaryArithmeticKind kind,
            TensorDescriptor left,
            TensorDescriptor right,
            TensorDescriptor output) {
        return new OperationCapabilityQuery(
                profile,
                new Operation(kind, NoOperationAttrs.INSTANCE),
                List.of(left, right),
                List.of(output));
    }

    private static OperationCapabilityQuery reductionQuery(
            NumericalProfile profile,
            AggregateReductionKind kind,
            io.github.pho001.synaptik.model.operation.OperationAttrs attrs,
            TensorDescriptor input,
            TensorDescriptor output) {
        return new OperationCapabilityQuery(
                profile,
                new Operation(kind, attrs),
                List.of(input),
                List.of(output));
    }

    private static Operation neg() {
        return new Operation(UnaryElementwiseKind.NEG, NoOperationAttrs.INSTANCE);
    }

    private static TensorDescriptor descriptor(Shape shape, boolean requiresGrad) {
        return new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), requiresGrad);
    }

    private static TensorDescriptor transposeCandidate(Shape shape, boolean requiresGrad) {
        long[] dimensions = shape.toLongArray();
        return new TensorDescriptor(
                DataType.FLOAT32,
                shape,
                Optional.of(LayoutDescriptor.of(
                        shape, new long[] {1L, dimensions[0]}, 0L, true)),
                requiresGrad);
    }

    private static TensorDescriptor view(Shape shape, long... strides) {
        return new TensorDescriptor(
                DataType.FLOAT32,
                shape,
                Optional.of(LayoutDescriptor.of(shape, strides, 0L, true)),
                false);
    }

    private record Occurrence(
            String name,
            Operation operation,
            TensorDescriptor input,
            TensorDescriptor output) { }

    private static TensorDescriptor typed(
            DataType type, Shape shape, boolean requiresGrad) {
        return new TensorDescriptor(
                type, shape, Optional.of(LayoutDescriptor.contiguous(shape)), requiresGrad);
    }

    private static TensorDescriptor typed(DataType type) {
        Shape shape = Shape.of(2, 3);
        return new TensorDescriptor(type, shape, Optional.of(LayoutDescriptor.contiguous(shape)),
                type.isDifferentiable());
    }
}
