package io.github.pho001.synaptik.backend.metal;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.ordering.OrderingKind;
import io.github.pho001.synaptik.model.operation.ordering.SortAttrs;
import io.github.pho001.synaptik.model.operation.ordering.TopKAttrs;
import io.github.pho001.synaptik.model.operation.ordering.TopKKind;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.ArgExtremaAttrs;
import io.github.pho001.synaptik.model.operation.reduction.ArgExtremaTiePolicy;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class MetalOrderingCapabilityTest {
    private final MetalCapabilityProvider provider = new MetalCapabilityProvider();

    @Test
    void allOrderingCarriersAndNumericArgCarriersAreSupported() {
        Shape inputShape = Shape.of(2, 4, 3);
        Shape topShape = Shape.of(2, 2, 3);
        Shape argShape = Shape.of(2, 3);
        for (DataType type : DataType.values()) {
            boolean gradient = type.isDifferentiable();
            TensorDescriptor input = descriptor(type, inputShape, gradient);
            assertTrue(supports(new Operation(OrderingKind.SORT, new SortAttrs(1, true)),
                            List.of(input), List.of(descriptor(type, inputShape, gradient))),
                    " SORT " + type);
            assertTrue(supports(new Operation(OrderingKind.ARGSORT, new SortAttrs(1, false)),
                            List.of(input), List.of(descriptor(DataType.INT64, inputShape, false))),
                    " ARGSORT " + type);
            assertTrue(supports(new Operation(TopKKind.TOP_K, new TopKAttrs(1, 2, true, false)),
                            List.of(input), List.of(
                                    descriptor(type, topShape, gradient),
                                    descriptor(DataType.INT64, topShape, false))),
                    " TOP_K " + type);
            for (AggregateReductionKind kind : List.of(
                    AggregateReductionKind.ARG_MAX, AggregateReductionKind.ARG_MIN)) {
                boolean expected = type != DataType.BOOL;
                boolean actual = supports(new Operation(kind, new ArgExtremaAttrs(
                                1, false, ArgExtremaTiePolicy.LAST_INDEX)),
                        List.of(input),
                        List.of(descriptor(DataType.INT64, argShape, false)));
                if (expected) {
                    assertTrue(actual, kind + " " + type);
                } else {
                    assertFalse(actual, kind + " " + type);
                }
            }
        }

    }

    @Test
    void orderingRejectsWrongRolesShapesAndMetadata() {
        Shape shape = Shape.of(2, 4);
        TensorDescriptor input = descriptor(DataType.FLOAT32, shape, true);
        Operation sort = new Operation(OrderingKind.SORT, new SortAttrs(1, false));
        assertFalse(supports(sort, List.of(input),
                List.of(descriptor(DataType.FLOAT32, shape, false))));
        assertFalse(supports(sort, List.of(input),
                List.of(descriptor(DataType.INT64, shape, false))));

        Operation top = new Operation(TopKKind.TOP_K, new TopKAttrs(1, 2, false, true));
        assertFalse(supports(top, List.of(input), List.of(
                descriptor(DataType.FLOAT32, Shape.of(2, 2), true),
                descriptor(DataType.INT32, Shape.of(2, 2), false))));
        assertFalse(supports(new Operation(TopKKind.TOP_K, new TopKAttrs(1, 5, true, true)),
                List.of(input), List.of(
                        descriptor(DataType.FLOAT32, Shape.of(2, 5), true),
                        descriptor(DataType.INT64, Shape.of(2, 5), false))));

        Operation arg = new Operation(AggregateReductionKind.ARG_MAX,
                new ArgExtremaAttrs(1, true, ArgExtremaTiePolicy.FIRST_INDEX));
        assertFalse(supports(arg, List.of(input),
                List.of(descriptor(DataType.INT64, Shape.of(2), false))));
        assertFalse(supports(arg, List.of(input),
                List.of(descriptor(DataType.INT32, Shape.of(2, 1), false))));
    }

    @Test
    void unsignedGeometryBoundaryIsExactAndAllocationFree() {
        long maximum = 0xffff_ffffL;
        Operation sort = new Operation(OrderingKind.SORT, new SortAttrs(0, false));
        Shape exact = Shape.of(maximum);
        assertTrue(supports(sort,
                List.of(descriptor(DataType.BOOL, exact, false)),
                List.of(descriptor(DataType.BOOL, exact, false))));

        Shape exactStride = Shape.of(1, maximum);
        Operation axisOne = new Operation(OrderingKind.SORT, new SortAttrs(1, false));
        assertTrue(supports(axisOne,
                List.of(descriptor(DataType.BOOL, exactStride, false)),
                List.of(descriptor(DataType.BOOL, exactStride, false))));

        Shape onePast = Shape.of(maximum + 1L);
        assertFalse(supports(sort,
                List.of(descriptor(DataType.BOOL, onePast, false)),
                List.of(descriptor(DataType.BOOL, onePast, false))));

        Shape productOverflow = Shape.of(65_536, 65_536);
        Operation axisZero = new Operation(OrderingKind.ARGSORT, new SortAttrs(0, false));
        assertFalse(supports(axisZero,
                List.of(descriptor(DataType.INT32, productOverflow, false)),
                List.of(descriptor(DataType.INT64, productOverflow, false))));
    }

    private boolean supports(
            Operation operation,
            List<TensorDescriptor> inputs,
            List<TensorDescriptor> outputs) {
        return provider.supports(new OperationCapabilityQuery(operation, inputs, outputs));
    }

    private static TensorDescriptor descriptor(
            DataType type, Shape shape, boolean requiresGrad) {
        return new TensorDescriptor(type, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), requiresGrad);
    }
}
