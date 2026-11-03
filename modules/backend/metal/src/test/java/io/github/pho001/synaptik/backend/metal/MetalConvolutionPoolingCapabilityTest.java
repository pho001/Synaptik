package io.github.pho001.synaptik.backend.metal;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dAttrs;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dKind;
import io.github.pho001.synaptik.model.operation.convolution.Conv3dAttrs;
import io.github.pho001.synaptik.model.operation.convolution.Conv3dKind;
import io.github.pho001.synaptik.model.operation.pooling.AveragePool2dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.AveragePool3dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.MaxPool2dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.MaxPool3dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.Pool2dKind;
import io.github.pho001.synaptik.model.operation.pooling.Pool3dKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class MetalConvolutionPoolingCapabilityTest {
    private final MetalCapabilityProvider provider = new MetalCapabilityProvider();

    @Test
    void convolutionIsAcceleratorOnlyWithExactCarrierAndGradientRows() {
        Operation conv2d = new Operation(Conv2dKind.CONV2D, Conv2dAttrs.defaults());
        TensorDescriptor input = descriptor(DataType.FLOAT32, Shape.of(1, 2, 3, 4), true);
        TensorDescriptor weight = descriptor(DataType.FLOAT32, Shape.of(4, 2, 1, 1), false);
        TensorDescriptor bias = descriptor(DataType.FLOAT32, Shape.of(4), false);
        TensorDescriptor output = descriptor(DataType.FLOAT32, Shape.of(1, 4, 3, 4), true);
        assertTrue(supports(conv2d,
                List.of(input, weight, bias), output));
        assertFalse(supports(conv2d,
                List.of(input, weight, bias), withGradient(output, false)));

        TensorDescriptor mixedInput = descriptor(DataType.BFLOAT16, input.shape(), false);
        assertTrue(supports(conv2d,
                List.of(mixedInput, withGradient(weight, false)),
                withGradient(output, false)));
        assertFalse(supports(conv2d,
                List.of(withGradient(mixedInput, true), withGradient(weight, false)),
                withGradient(output, true)));
        assertFalse(supports(conv2d,
                List.of(mixedInput,
                        descriptor(DataType.BFLOAT16, weight.shape(), false)),
                withGradient(output, false)));
    }

    @Test
    void groupedConvolutionAndThreeDimensionalGeometryAreExact() {
        Operation grouped = new Operation(Conv2dKind.CONV2D,
                new Conv2dAttrs(2, 1, 1, 0, 2, 1, 2));
        TensorDescriptor input = descriptor(DataType.FLOAT32, Shape.of(2, 4, 7, 5), false);
        TensorDescriptor weight = descriptor(DataType.FLOAT32, Shape.of(6, 2, 3, 2), false);
        TensorDescriptor bias = descriptor(DataType.BFLOAT16, Shape.of(6), false);
        TensorDescriptor output = descriptor(DataType.FLOAT32, Shape.of(2, 6, 3, 4), false);
        assertTrue(supports(grouped,
                List.of(input, weight, bias), output));
        assertFalse(supports(grouped,
                List.of(input, descriptor(DataType.FLOAT32, Shape.of(6, 3, 3, 2), false), bias),
                output));
        assertFalse(supports(grouped,
                List.of(input, weight, descriptor(DataType.FLOAT32, Shape.of(5), false)), output));

        Operation conv3d = new Operation(Conv3dKind.CONV3D,
                new Conv3dAttrs(1, 2, 1, 0, 1, 0, 1, 1, 2, 1));
        assertTrue(supports(conv3d,
                List.of(
                        descriptor(DataType.FLOAT32, Shape.of(1, 2, 4, 5, 6), false),
                        descriptor(DataType.FLOAT32, Shape.of(3, 2, 2, 2, 2), false)),
                descriptor(DataType.FLOAT32, Shape.of(1, 3, 3, 3, 4), false)));
    }

    @Test
    void groupedConv3dAdmitsEverySelectedFloat32GradientAndChecksTheExactFlagUnion() {
        Operation convolution = new Operation(Conv3dKind.CONV3D, Conv3dAttrs.defaults());
        for (boolean biased : List.of(false, true)) {
            int inputCount = biased ? 3 : 2;
            for (int mask = 0; mask < (1 << inputCount); mask++) {
                TensorDescriptor input = descriptor(
                        DataType.FLOAT32, Shape.of(2, 4, 3, 3, 3), (mask & 1) != 0);
                TensorDescriptor weight = descriptor(
                        DataType.FLOAT32, Shape.of(6, 4, 2, 2, 2), (mask & 2) != 0);
                TensorDescriptor bias = descriptor(DataType.FLOAT32, Shape.of(6), (mask & 4) != 0);
                List<TensorDescriptor> inputs = biased ? List.of(input, weight, bias) : List.of(input, weight);
                TensorDescriptor output = descriptor(DataType.FLOAT32, Shape.of(2, 6, 2, 2, 2), mask != 0);
                assertTrue(supports(convolution, inputs, output), "biased=" + biased + " mask=" + mask);
                assertFalse(supports(convolution, inputs, withGradient(output, mask == 0)));
            }
        }
        Operation grouped = new Operation(
                Conv3dKind.CONV3D, new Conv3dAttrs(1, 1, 1, 0, 0, 0, 1, 1, 1, 2));
        for (DataType type : List.of(DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16)) {
            boolean float32 = type == DataType.FLOAT32;
            assertTrue(supports(grouped,
                    List.of(descriptor(type, Shape.of(1, 4, 3, 3, 3), false),
                            descriptor(type, Shape.of(6, 2, 2, 2, 2), false)),
                    descriptor(type, Shape.of(1, 6, 2, 2, 2), false)));
            org.junit.jupiter.api.Assertions.assertEquals(float32, supports(grouped,
                    List.of(descriptor(type, Shape.of(1, 4, 3, 3, 3), true),
                            descriptor(type, Shape.of(6, 2, 2, 2, 2), true)),
                    descriptor(type, Shape.of(1, 6, 2, 2, 2), true)));
        }
    }

    @Test
    void maximumAndAveragePoolingKeepTheirQualifiedCarriers() {
        Operation max2d = new Operation(Pool2dKind.MAX_POOL2D,
                new MaxPool2dAttrs(2, 3, 2, 2, 1, 1, 1, 1, true));
        Shape inputShape = Shape.of(1, 2, 3, 4);
        Shape outputShape = Shape.of(1, 2, 3, 3);
        for (DataType type :
                List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16)) {
            assertEquals(type == DataType.FLOAT32, supports(max2d,
                    List.of(descriptor(type, inputShape, true)),
                    descriptor(type, outputShape, true)), "overlap " + type);
        }

        Operation nonOverlappingMax = new Operation(
                Pool2dKind.MAX_POOL2D,
                new MaxPool2dAttrs(2, 3, 2, 3, 0, 0, 1, 1, false));
        for (DataType type :
                List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16)) {
            assertTrue(supports(
                    nonOverlappingMax,
                    List.of(descriptor(type, Shape.of(1, 2, 4, 6), true)),
                    descriptor(type, Shape.of(1, 2, 2, 2), true)),
                    "non-overlap " + type);
        }

        assertFalse(supports(max2d,
                List.of(descriptor(DataType.INT32, inputShape, false)),
                descriptor(DataType.INT32, outputShape, false)));
        assertFalse(supports(max2d,
                List.of(descriptor(DataType.FLOAT32, inputShape, true)),
                descriptor(DataType.FLOAT32, outputShape, false)));

        Operation average2d = new Operation(Pool2dKind.AVERAGE_POOL2D,
                new AveragePool2dAttrs(2, 3, 2, 2, 1, 1, 1, 1, true));
        assertTrue(supports(average2d,
                List.of(descriptor(DataType.FLOAT32, inputShape, false)),
                descriptor(DataType.FLOAT32, outputShape, false)));
        for (DataType type : List.of(DataType.BFLOAT16, DataType.FLOAT16)) {
            assertTrue(supports(average2d,
                    List.of(descriptor(type, inputShape, false)),
                    descriptor(type, outputShape, false)));
        }
    }

    @Test
    void poolingGradientOverlapUsesTheSameFloat32ImageFoldBoundaryInBothRanks() {
        for (int spatial : List.of(2, 3)) {
            for (boolean maximum : List.of(false, true)) {
                for (boolean overlap : List.of(false, true)) {
                    int stride = overlap ? 1 : 5;
                    Operation pool = spatial == 2
                            ? maximum
                                    ? new Operation(Pool2dKind.MAX_POOL2D,
                                            new MaxPool2dAttrs(3, 3, stride, stride, 0, 0, 2, 2, false))
                                    : new Operation(Pool2dKind.AVERAGE_POOL2D,
                                            new AveragePool2dAttrs(3, 3, stride, stride, 0, 0, 2, 2, false))
                            : maximum
                                    ? new Operation(Pool3dKind.MAX_POOL3D,
                                            new MaxPool3dAttrs(3, 3, 3, stride, stride, stride,
                                                    0, 0, 0, 2, 2, 2, false))
                                    : new Operation(Pool3dKind.AVERAGE_POOL3D,
                                            new AveragePool3dAttrs(3, 3, 3, stride, stride, stride,
                                                    0, 0, 0, 2, 2, 2, false));
                    long out = overlap ? 3 : 1;
                    Shape source = spatial == 2 ? Shape.of(1, 2, 7, 7) : Shape.of(1, 2, 7, 7, 7);
                    Shape target = spatial == 2 ? Shape.of(1, 2, out, out) : Shape.of(1, 2, out, out, out);
                    for (DataType type : List.of(
                            DataType.FLOAT32, DataType.FLOAT64, DataType.BFLOAT16, DataType.FLOAT16)) {
                        for (boolean gradient : List.of(false, true)) {
                            boolean expected = (maximum || type != DataType.FLOAT64)
                                    && (!gradient || !overlap || type == DataType.FLOAT32);
                            TensorDescriptor input = descriptor(type, source, gradient);
                            TensorDescriptor output = descriptor(type, target, gradient);
                            assertEquals(expected, supports(pool, List.of(input), output),
                                    spatial + "d " + pool.kind() + " " + type + " grad=" + gradient
                                            + " overlap=" + overlap);
                            assertFalse(supports(pool, List.of(input), withGradient(output, !gradient)));
                        }
                    }
                }
            }
        }
    }

    @Test
    void threeDimensionalPoolFloorCeilDilationAndAllPaddingTerminalWindowsAreLiteral() {
        Shape source = Shape.of(1, 1, 3, 3, 3);
        Operation floor = new Operation(Pool3dKind.MAX_POOL3D,
                new MaxPool3dAttrs(2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 1, 1, false));
        Operation ceil = new Operation(Pool3dKind.MAX_POOL3D,
                new MaxPool3dAttrs(2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 1, 1, true));
        TensorDescriptor input = descriptor(DataType.FLOAT64, source, false);
        assertTrue(supports(floor, List.of(input),
                descriptor(DataType.FLOAT64, Shape.of(1, 1, 2, 2, 2), false)));
        assertTrue(supports(ceil, List.of(input),
                descriptor(DataType.FLOAT64, Shape.of(1, 1, 2, 3, 3), false)));
        assertFalse(supports(ceil, List.of(input),
                descriptor(DataType.FLOAT64, Shape.of(1, 1, 2, 2, 2), false)));

        Operation average3d = new Operation(Pool3dKind.AVERAGE_POOL3D,
                new AveragePool3dAttrs(1, 1, 2, 1, 1, 2, 0, 0, 2, 1, 1, 1, true));
        assertTrue(supports(average3d,
                List.of(descriptor(DataType.FLOAT32, Shape.of(1, 1, 1, 1, 1), false)),
                descriptor(DataType.FLOAT32, Shape.of(1, 1, 1, 1, 3), false)));
    }

    @Test
    void exactSingletonHeightViewAndUint32BoundsAreCapabilityFacts() {
        Shape sourceShape = Shape.of(2, 3, 1, 5);
        LayoutDescriptor exact = LayoutDescriptor.of(
                sourceShape, new long[] {15, 5, 5, 1}, 0, true);
        TensorDescriptor view = new TensorDescriptor(
                DataType.FLOAT32, sourceShape, Optional.of(exact), false);
        Operation max = new Operation(Pool2dKind.MAX_POOL2D,
                new MaxPool2dAttrs(1, 2, 1, 1, 0, 0, 1, 1, false));
        assertTrue(supports(max, List.of(view),
                descriptor(DataType.FLOAT32, Shape.of(2, 3, 1, 4), false)));
        LayoutDescriptor wrong = LayoutDescriptor.of(
                sourceShape, new long[] {30, 10, 5, 1}, 0, true);
        assertFalse(supports(max,
                List.of(new TensorDescriptor(
                        DataType.FLOAT32, sourceShape, Optional.of(wrong), false)),
                descriptor(DataType.FLOAT32, Shape.of(2, 3, 1, 4), false)));

        long onePast = 0x1_0000_0000L;
        Shape overLimit = Shape.of(1, 1, 1, onePast);
        assertFalse(supports(max,
                List.of(descriptor(DataType.FLOAT32, overLimit, false)),
                descriptor(DataType.FLOAT32, Shape.of(1, 1, 1, onePast - 1L), false)));

        long uint32Max = 0xffff_ffffL;
        Shape maximumWidth = Shape.of(1, 1, 1, uint32Max);
        Operation terminalOriginAtLimit = new Operation(
                Pool2dKind.MAX_POOL2D,
                new MaxPool2dAttrs(1, 1, 1, 3, 0, 0, 1, 1, true));
        assertTrue(supports(
                terminalOriginAtLimit,
                List.of(descriptor(DataType.FLOAT32, maximumWidth, false)),
                descriptor(
                        DataType.FLOAT32, Shape.of(1, 1, 1, 1_431_655_766L), false)));
        Operation terminalOriginOnePast = new Operation(
                Pool2dKind.MAX_POOL2D,
                new MaxPool2dAttrs(1, 1, 1, 4, 0, 0, 1, 1, true));
        assertFalse(supports(
                terminalOriginOnePast,
                List.of(descriptor(DataType.FLOAT32, maximumWidth, false)),
                descriptor(
                        DataType.FLOAT32, Shape.of(1, 1, 1, 1_073_741_825L), false)));
    }

    @Test
    void poolingKernelPositionWorkIsIndependentlyBounded() {
        Operation exactLimit = new Operation(
                Pool2dKind.MAX_POOL2D,
                new MaxPool2dAttrs(1, 65_536, 1, 1, 0, 32_767, 1, 1, false));
        assertTrue(supports(
                exactLimit,
                List.of(descriptor(DataType.FLOAT32, Shape.of(1, 1, 1, 2), false)),
                descriptor(DataType.FLOAT32, Shape.of(1, 1, 1, 1), false)));

        Operation onePast = new Operation(
                Pool2dKind.MAX_POOL2D,
                new MaxPool2dAttrs(1, 65_537, 1, 1, 0, 32_768, 1, 1, false));
        assertFalse(supports(
                onePast,
                List.of(descriptor(DataType.FLOAT32, Shape.of(1, 1, 1, 1), false)),
                descriptor(DataType.FLOAT32, Shape.of(1, 1, 1, 1), false)));

        Operation onePast3d = new Operation(
                Pool3dKind.AVERAGE_POOL3D,
                new AveragePool3dAttrs(
                        1, 1, 65_537,
                        1, 1, 1,
                        0, 0, 32_768,
                        1, 1, 1,
                        false));
        assertFalse(supports(
                onePast3d,
                List.of(descriptor(DataType.FLOAT32, Shape.of(1, 1, 1, 1, 1), false)),
                descriptor(DataType.FLOAT32, Shape.of(1, 1, 1, 1, 1), false)));
    }


    private boolean supports(
            Operation operation,
            List<TensorDescriptor> inputs,
            TensorDescriptor output) {
        return provider.supports(new OperationCapabilityQuery(
                operation, inputs, List.of(output)));
    }

    private static TensorDescriptor descriptor(DataType type, Shape shape, boolean requiresGrad) {
        return new TensorDescriptor(type, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), requiresGrad);
    }

    private static TensorDescriptor withGradient(TensorDescriptor descriptor, boolean requiresGrad) {
        return new TensorDescriptor(descriptor.dataType(), descriptor.shape(),
                descriptor.layout(), requiresGrad);
    }
}
