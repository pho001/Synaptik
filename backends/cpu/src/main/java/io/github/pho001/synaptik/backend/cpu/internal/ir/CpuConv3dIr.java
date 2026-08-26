package io.github.pho001.synaptik.backend.cpu.internal.ir;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.DataTypePromotion;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * CPU-private generated identity for direct grouped NCDHW three-dimensional cross-correlation.
 * Concrete extents, layouts, carrier bases, and output-cell ranges remain cold invocation facts.
 *
 * @param inputTypes ordered input, weight, and optional intrinsic-bias represented types
 * @param resultType exact ordered-promotion accumulation and output type
 * @param strideDepth positive depth stride
 * @param strideHeight positive height stride
 * @param strideWidth positive width stride
 * @param paddingDepth non-negative symmetric depth padding
 * @param paddingHeight non-negative symmetric height padding
 * @param paddingWidth non-negative symmetric width padding
 * @param dilationDepth positive depth dilation
 * @param dilationHeight positive height dilation
 * @param dilationWidth positive width dilation
 * @param groups positive contiguous channel-group count
 * @param algorithmVersion direct traversal version, currently {@code 1}
 * @param intrinsicBias whether the third input is the intrinsic rank-one bias
 * @param inputAccesses ordered read plans
 * @param outputAccess sole write plan
 */
public record CpuConv3dIr(List<DataType> inputTypes, DataType resultType,
        long strideDepth, long strideHeight, long strideWidth,
        long paddingDepth, long paddingHeight, long paddingWidth,
        long dilationDepth, long dilationHeight, long dilationWidth,
        long groups, int algorithmVersion, boolean intrinsicBias,
        List<CpuAccessPlan> inputAccesses, CpuAccessPlan outputAccess)
        implements CpuPortableKernelIr {

    /**
     * Validates and snapshots one rank-specific direct-convolution identity.
     *
     * @throws NullPointerException if a required component is {@code null}
     * @throws IllegalArgumentException if types, geometry, accesses, or version disagree
     */
    public CpuConv3dIr {
        inputTypes = List.copyOf(inputTypes);
        Objects.requireNonNull(resultType, "resultType");
        inputAccesses = List.copyOf(inputAccesses);
        Objects.requireNonNull(outputAccess, "outputAccess");
        DataType promoted = inputTypes.isEmpty() ? null : inputTypes.getFirst();
        for (int i = 1; i < inputTypes.size(); i++) {
            promoted = DataTypePromotion.promoteFloating(promoted, inputTypes.get(i));
        }
        if (inputTypes.size() != (intrinsicBias ? 3 : 2)
                || inputAccesses.size() != inputTypes.size()
                || inputTypes.stream().anyMatch(type -> !floating(type))
                || promoted != resultType || !floating(resultType)
                || strideDepth <= 0 || strideHeight <= 0 || strideWidth <= 0
                || paddingDepth < 0 || paddingHeight < 0 || paddingWidth < 0
                || dilationDepth <= 0 || dilationHeight <= 0 || dilationWidth <= 0
                || groups <= 0 || algorithmVersion != 1
                || inputAccesses.stream().anyMatch(access ->
                    access.accessKind() != CpuAccessPlan.AccessKind.READ)
                || outputAccess.accessKind() != CpuAccessPlan.AccessKind.WRITE) {
            throw new IllegalArgumentException("Conv3d IR facts disagree");
        }
    }

    /**
     * Encodes this rank-specific identity as the canonical instruction-free generator input.
     *
     * @return a new immutable kernel IR whose values preserve boundary order and whose family
     *     identity contains every code-shaping Conv3d fact; never {@code null}
     */
    public CpuKernelIr encodedKernelIr() {
        var values = new ArrayList<CpuKernelIr.Value>();
        for (int i = 0; i < inputTypes.size(); i++) {
            values.add(new CpuKernelIr.Value(i, inputTypes.get(i),
                    CpuKernelIr.Value.Kind.INPUT, inputAccesses.get(i)));
        }
        values.add(new CpuKernelIr.Value(values.size(), resultType,
                CpuKernelIr.Value.Kind.OUTPUT, outputAccess));
        String family = "conv3d:bias=" + intrinsicBias + ":types=" + inputTypes
                + ":result=" + resultType + ":strideD=" + strideDepth
                + ":strideH=" + strideHeight + ":strideW=" + strideWidth
                + ":padD=" + paddingDepth + ":padH=" + paddingHeight
                + ":padW=" + paddingWidth + ":dilationD=" + dilationDepth
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
