package io.github.pho001.synaptik.backend.metal;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.config.compile.NumericalProfile;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.attention.ScaledDotProductAttentionAttrs;
import io.github.pho001.synaptik.model.operation.attention.ScaledDotProductAttentionKind;
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
        assertTrue(supports(NumericalProfile.ACCELERATOR, conv2d,
                List.of(input, weight, bias), output));
        assertFalse(supports(NumericalProfile.STRICT_IEEE, conv2d,
                List.of(input, weight, bias), output));
        assertFalse(supports(NumericalProfile.ACCELERATOR, conv2d,
                List.of(input, weight, bias), withGradient(output, false)));

        TensorDescriptor mixedInput = descriptor(DataType.BFLOAT16, input.shape(), false);
        assertTrue(supports(NumericalProfile.ACCELERATOR, conv2d,
                List.of(mixedInput, withGradient(weight, false)),
                withGradient(output, false)));
        assertFalse(supports(NumericalProfile.ACCELERATOR, conv2d,
                List.of(withGradient(mixedInput, true), withGradient(weight, false)),
                withGradient(output, true)));
        assertFalse(supports(NumericalProfile.ACCELERATOR, conv2d,
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
        assertTrue(supports(NumericalProfile.ACCELERATOR, grouped,
                List.of(input, weight, bias), output));
        assertFalse(supports(NumericalProfile.ACCELERATOR, grouped,
                List.of(input, descriptor(DataType.FLOAT32, Shape.of(6, 3, 3, 2), false), bias),
                output));
        assertFalse(supports(NumericalProfile.ACCELERATOR, grouped,
                List.of(input, weight, descriptor(DataType.FLOAT32, Shape.of(5), false)), output));

        Operation conv3d = new Operation(Conv3dKind.CONV3D,
                new Conv3dAttrs(1, 2, 1, 0, 1, 0, 1, 1, 2, 1));
        assertTrue(supports(NumericalProfile.ACCELERATOR, conv3d,
                List.of(
                        descriptor(DataType.FLOAT32, Shape.of(1, 2, 4, 5, 6), false),
                        descriptor(DataType.FLOAT32, Shape.of(3, 2, 2, 2, 2), false)),
                descriptor(DataType.FLOAT32, Shape.of(1, 3, 3, 3, 4), false)));
    }

    @Test
    void maximumPoolingIsProfileCommonAndAveragePoolingIsAcceleratorFloat32() {
        Operation max2d = new Operation(Pool2dKind.MAX_POOL2D,
                new MaxPool2dAttrs(2, 3, 2, 2, 1, 1, 1, 1, true));
        Shape inputShape = Shape.of(1, 2, 3, 4);
        Shape outputShape = Shape.of(1, 2, 3, 3);
        for (NumericalProfile profile : NumericalProfile.values()) {
            for (DataType type :
                    List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16)) {
                assertFalse(supports(profile, max2d,
                        List.of(descriptor(type, inputShape, true)),
                        descriptor(type, outputShape, true)), "overlap " + profile + " " + type);
            }
        }
        Operation nonOverlappingMax = new Operation(
                Pool2dKind.MAX_POOL2D,
                new MaxPool2dAttrs(2, 3, 2, 3, 0, 0, 1, 1, false));
        for (NumericalProfile profile : NumericalProfile.values()) {
            for (DataType type :
                    List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16)) {
                assertTrue(supports(
                        profile,
                        nonOverlappingMax,
                        List.of(descriptor(type, Shape.of(1, 2, 4, 6), true)),
                        descriptor(type, Shape.of(1, 2, 2, 2), true)),
                        "non-overlap " + profile + " " + type);
            }
        }
        assertFalse(supports(NumericalProfile.ACCELERATOR, max2d,
                List.of(descriptor(DataType.INT32, inputShape, false)),
                descriptor(DataType.INT32, outputShape, false)));
        assertFalse(supports(NumericalProfile.ACCELERATOR, max2d,
                List.of(descriptor(DataType.FLOAT32, inputShape, true)),
                descriptor(DataType.FLOAT32, outputShape, false)));

        Operation average2d = new Operation(Pool2dKind.AVERAGE_POOL2D,
                new AveragePool2dAttrs(2, 3, 2, 2, 1, 1, 1, 1, true));
        assertTrue(supports(NumericalProfile.ACCELERATOR, average2d,
                List.of(descriptor(DataType.FLOAT32, inputShape, false)),
                descriptor(DataType.FLOAT32, outputShape, false)));
        assertFalse(supports(NumericalProfile.STRICT_IEEE, average2d,
                List.of(descriptor(DataType.FLOAT32, inputShape, false)),
                descriptor(DataType.FLOAT32, outputShape, false)));
        for (DataType type : List.of(DataType.BFLOAT16, DataType.FLOAT16)) {
            assertTrue(supports(NumericalProfile.ACCELERATOR, average2d,
                    List.of(descriptor(type, inputShape, false)),
                    descriptor(type, outputShape, false)));
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
        assertTrue(supports(NumericalProfile.STRICT_IEEE, floor, List.of(input),
                descriptor(DataType.FLOAT64, Shape.of(1, 1, 2, 2, 2), false)));
        assertTrue(supports(NumericalProfile.STRICT_IEEE, ceil, List.of(input),
                descriptor(DataType.FLOAT64, Shape.of(1, 1, 2, 3, 3), false)));
        assertFalse(supports(NumericalProfile.STRICT_IEEE, ceil, List.of(input),
                descriptor(DataType.FLOAT64, Shape.of(1, 1, 2, 2, 2), false)));

        Operation average3d = new Operation(Pool3dKind.AVERAGE_POOL3D,
                new AveragePool3dAttrs(1, 1, 2, 1, 1, 2, 0, 0, 2, 1, 1, 1, true));
        assertTrue(supports(NumericalProfile.ACCELERATOR, average3d,
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
        assertTrue(supports(NumericalProfile.STRICT_IEEE, max, List.of(view),
                descriptor(DataType.FLOAT32, Shape.of(2, 3, 1, 4), false)));
        LayoutDescriptor wrong = LayoutDescriptor.of(
                sourceShape, new long[] {30, 10, 5, 1}, 0, true);
        assertFalse(supports(NumericalProfile.STRICT_IEEE, max,
                List.of(new TensorDescriptor(
                        DataType.FLOAT32, sourceShape, Optional.of(wrong), false)),
                descriptor(DataType.FLOAT32, Shape.of(2, 3, 1, 4), false)));

        long onePast = 0x1_0000_0000L;
        Shape overLimit = Shape.of(1, 1, 1, onePast);
        assertFalse(supports(NumericalProfile.STRICT_IEEE, max,
                List.of(descriptor(DataType.FLOAT32, overLimit, false)),
                descriptor(DataType.FLOAT32, Shape.of(1, 1, 1, onePast - 1L), false)));

        long uint32Max = 0xffff_ffffL;
        Shape maximumWidth = Shape.of(1, 1, 1, uint32Max);
        Operation terminalOriginAtLimit = new Operation(
                Pool2dKind.MAX_POOL2D,
                new MaxPool2dAttrs(1, 1, 1, 3, 0, 0, 1, 1, true));
        assertTrue(supports(
                NumericalProfile.STRICT_IEEE,
                terminalOriginAtLimit,
                List.of(descriptor(DataType.FLOAT32, maximumWidth, false)),
                descriptor(
                        DataType.FLOAT32, Shape.of(1, 1, 1, 1_431_655_766L), false)));
        Operation terminalOriginOnePast = new Operation(
                Pool2dKind.MAX_POOL2D,
                new MaxPool2dAttrs(1, 1, 1, 4, 0, 0, 1, 1, true));
        assertFalse(supports(
                NumericalProfile.STRICT_IEEE,
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
                NumericalProfile.STRICT_IEEE,
                exactLimit,
                List.of(descriptor(DataType.FLOAT32, Shape.of(1, 1, 1, 2), false)),
                descriptor(DataType.FLOAT32, Shape.of(1, 1, 1, 1), false)));

        Operation onePast = new Operation(
                Pool2dKind.MAX_POOL2D,
                new MaxPool2dAttrs(1, 65_537, 1, 1, 0, 32_768, 1, 1, false));
        assertFalse(supports(
                NumericalProfile.STRICT_IEEE,
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
                NumericalProfile.ACCELERATOR,
                onePast3d,
                List.of(descriptor(DataType.FLOAT32, Shape.of(1, 1, 1, 1, 1), false)),
                descriptor(DataType.FLOAT32, Shape.of(1, 1, 1, 1, 1), false)));
    }

    @Test
    void attentionRemainsProductionFalse() {
        Operation attention = new Operation(
                ScaledDotProductAttentionKind.SCALED_DOT_PRODUCT_ATTENTION,
                new ScaledDotProductAttentionAttrs(Optional.empty(), false));
        TensorDescriptor value = descriptor(DataType.FLOAT32, Shape.of(1, 1, 1), false);
        assertFalse(supports(NumericalProfile.ACCELERATOR, attention,
                List.of(value, value, value), value));
    }

    private boolean supports(
            NumericalProfile profile,
            Operation operation,
            List<TensorDescriptor> inputs,
            TensorDescriptor output) {
        return provider.supports(new OperationCapabilityQuery(
                profile, operation, inputs, List.of(output)));
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
