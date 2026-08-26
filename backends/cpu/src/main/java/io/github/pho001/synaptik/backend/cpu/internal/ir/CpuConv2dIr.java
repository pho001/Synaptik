package io.github.pho001.synaptik.backend.cpu.internal.ir;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.DataTypePromotion;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * CPU-private generated identity for direct grouped NCHW two-dimensional cross-correlation.
 *
 * <p>The identity retains semantic types, intrinsic-bias presence, convolution attributes,
 * traversal version, and boundary access shapes. Concrete extents, offsets, strides, carriers,
 * slots, addresses, and range bounds remain cold invocation geometry, so one artifact is reusable
 * across compatible static Shapes.</p>
 *
 * @param inputTypes ordered input, weight, and optional intrinsic-bias represented types
 * @param resultType ordered-promotion accumulation and output type
 * @param strideHeight positive vertical output stride
 * @param strideWidth positive horizontal output stride
 * @param paddingHeight non-negative symmetric vertical padding
 * @param paddingWidth non-negative symmetric horizontal padding
 * @param dilationHeight positive vertical kernel dilation
 * @param dilationWidth positive horizontal kernel dilation
 * @param groups positive contiguous channel-group count
 * @param algorithmVersion direct traversal version, currently {@code 1}
 * @param intrinsicBias whether the third convolution input is the direct rank-one bias
 * @param epilogue non-null bounded external suffix retained in this generated body
 * @param inputAccesses ordered read access plans
 * @param outputAccess sole write access plan
 */
public record CpuConv2dIr(List<DataType> inputTypes, DataType resultType,
        long strideHeight, long strideWidth, long paddingHeight, long paddingWidth,
        long dilationHeight, long dilationWidth, long groups, int algorithmVersion,
        boolean intrinsicBias, Epilogue epilogue,
        List<CpuAccessPlan> inputAccesses, CpuAccessPlan outputAccess)
        implements CpuPortableKernelIr {

    /** Bounded family-local pointwise suffix retained by the direct generated body. */
    public enum Epilogue {
        /** No external pointwise suffix. */ NONE,
        /** One same-type right-broadcast external ADD. */ ADD,
        /** External ADD followed by exact same-type RELU. */ ADD_RELU
    }

    /**
     * Validates and snapshots one exact direct-convolution code-shaping identity.
     *
     * @param inputTypes ordered input, weight, optional intrinsic bias, and optional external-ADD
     *     types; copied defensively
     * @param resultType promoted accumulation and output type
     * @param strideHeight positive vertical output stride
     * @param strideWidth positive horizontal output stride
     * @param paddingHeight non-negative symmetric vertical padding
     * @param paddingWidth non-negative symmetric horizontal padding
     * @param dilationHeight positive vertical kernel dilation
     * @param dilationWidth positive horizontal kernel dilation
     * @param groups positive contiguous channel-group count
     * @param algorithmVersion direct traversal version, currently {@code 1}
     * @param intrinsicBias whether the direct convolution reads a rank-one bias
     * @param epilogue bounded external suffix generated in the same body
     * @param inputAccesses ordered read access plans; copied defensively
     * @param outputAccess sole write access plan
     * @throws NullPointerException if a required component or list element is {@code null}
     * @throws IllegalArgumentException if types, attributes, accesses, or algorithm disagree
     */
    public CpuConv2dIr {
        inputTypes = List.copyOf(inputTypes);
        Objects.requireNonNull(resultType, "resultType");
        inputAccesses = List.copyOf(inputAccesses);
        Objects.requireNonNull(epilogue, "epilogue");
        Objects.requireNonNull(outputAccess, "outputAccess");
        DataType promoted = inputTypes.isEmpty() ? null : inputTypes.getFirst();
        for (int i = 1; i < inputTypes.size(); i++) {
            promoted = DataTypePromotion.promoteFloating(promoted, inputTypes.get(i));
        }
        int convolutionInputs = intrinsicBias ? 3 : 2;
        int externalInputs = epilogue == Epilogue.NONE ? 0 : 1;
        if (inputTypes.size() != convolutionInputs + externalInputs
                || inputAccesses.size() != inputTypes.size()
                || inputTypes.stream().anyMatch(type -> !floating(type))
                || promoted != resultType || !floating(resultType)
                || strideHeight <= 0 || strideWidth <= 0
                || paddingHeight < 0 || paddingWidth < 0
                || dilationHeight <= 0 || dilationWidth <= 0 || groups <= 0
                || algorithmVersion != 1
                || inputAccesses.stream().anyMatch(access ->
                    access.accessKind() != CpuAccessPlan.AccessKind.READ)
                || outputAccess.accessKind() != CpuAccessPlan.AccessKind.WRITE) {
            throw new IllegalArgumentException("Conv2d IR facts disagree");
        }
    }

    /**
     * Encodes this instruction-free family at the generator/cache seam.
     *
     * @return a new immutable canonical kernel identity
     */
    public CpuKernelIr encodedKernelIr() {
        var values = new ArrayList<CpuKernelIr.Value>();
        for (int i = 0; i < inputTypes.size(); i++) {
            values.add(new CpuKernelIr.Value(i, inputTypes.get(i),
                    CpuKernelIr.Value.Kind.INPUT, inputAccesses.get(i)));
        }
        values.add(new CpuKernelIr.Value(values.size(), resultType,
                CpuKernelIr.Value.Kind.OUTPUT, outputAccess));
        String family = "conv2d:bias=" + intrinsicBias
                + ":epilogue=" + epilogue + ":types=" + inputTypes + ":result=" + resultType
                + ":strideH=" + strideHeight + ":strideW=" + strideWidth
                + ":padH=" + paddingHeight + ":padW=" + paddingWidth
                + ":dilationH=" + dilationHeight + ":dilationW=" + dilationWidth
                + ":groups=" + groups + ":algorithm=" + algorithmVersion;
        return new CpuKernelIr(values, List.of(), new CpuKernelIr.Loop("start", "end"),
                List.of(new CpuKernelIr.Store(values.size() - 1, 0)), family);
    }

    @Override public String structuralKey() { return encodedKernelIr().structuralKey(); }

    private static boolean floating(DataType type) {
        return type == DataType.FLOAT64 || type == DataType.FLOAT32 || type == DataType.BFLOAT16;
    }
}
