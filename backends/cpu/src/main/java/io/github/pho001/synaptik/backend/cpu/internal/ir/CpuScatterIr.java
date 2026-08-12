package io.github.pho001.synaptik.backend.cpu.internal.ir;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.index.ScatterReduction;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

/**
 * CPU-private structural identity for one functional scatter output writer.
 * Concrete axes, tuple depths, extents, offsets, strides, ranges, workspace sizes, and carrier
 * instances remain cold geometry. The retained facts are exactly those that alter generated
 * behavior or its direct signature.
 *
 * @param family non-null current scatter coordinate family
 * @param reduction non-null represented-value reduction; {@link Family#SCATTER_ADD} requires ADD
 * @param occurrenceToBoundary three semantic input occurrences mapped to unique boundaries
 * @param boundaryTypes unique input types followed by the output type
 * @param boundaryAccesses structural accesses aligned with {@code boundaryTypes}
 * @param scratchSignature zero when absent, or the positive exact-product scratch format version
 */
public record CpuScatterIr(Family family, ScatterReduction reduction,
        List<Integer> occurrenceToBoundary, List<DataType> boundaryTypes,
        List<CpuAccessPlan> boundaryAccesses, int scratchSignature)
        implements CpuPortableKernelIr {
    /** Current Model scatter coordinate families. */
    public enum Family {
        /** Same-rank selected-axis scatter. */ SCATTER_ELEMENTS,
        /** Gather-compatible selected-axis fixed addition. */ SCATTER_ADD,
        /** Tuple-indexed scatter. */ SCATTER_ND
    }

    /**
     * Validates and snapshots the structural form.
     *
     * @throws NullPointerException if a component or element is {@code null}
     * @throws IllegalArgumentException if arities, types, reduction, or scratch facts disagree
     */
    public CpuScatterIr {
        Objects.requireNonNull(family, "family");
        Objects.requireNonNull(reduction, "reduction");
        occurrenceToBoundary = List.copyOf(occurrenceToBoundary);
        boundaryTypes = List.copyOf(boundaryTypes);
        boundaryAccesses = List.copyOf(boundaryAccesses);
        if (occurrenceToBoundary.size() != 3 || boundaryTypes.size() < 2
                || boundaryTypes.size() > 4 || boundaryTypes.size() != boundaryAccesses.size()) {
            throw new IllegalArgumentException("scatter boundary facts must agree");
        }
        int inputBoundaryCount = boundaryTypes.size() - 1;
        var referencedInputs = new HashSet<Integer>();
        for (int boundary : occurrenceToBoundary) {
            if (boundary < 0 || boundary >= inputBoundaryCount) {
                throw new IllegalArgumentException("scatter occurrence mapping is out of range");
            }
            referencedInputs.add(boundary);
        }
        if (referencedInputs.size() != inputBoundaryCount) {
            throw new IllegalArgumentException("scatter occurrence mapping must cover every input");
        }
        for (int index = 0; index < inputBoundaryCount; index++) {
            if (boundaryAccesses.get(index).accessKind() != CpuAccessPlan.AccessKind.READ) {
                throw new IllegalArgumentException("scatter inputs must be read-only");
            }
        }
        if (boundaryAccesses.getLast().accessKind() != CpuAccessPlan.AccessKind.WRITE) {
            throw new IllegalArgumentException("scatter output must be write-only");
        }
        DataType dataType = boundaryTypes.get(occurrenceToBoundary.get(0));
        DataType indexType = boundaryTypes.get(occurrenceToBoundary.get(1));
        DataType updateType = boundaryTypes.get(occurrenceToBoundary.get(2));
        if (indexType != DataType.INT32 && indexType != DataType.INT64
                || updateType != dataType || boundaryTypes.getLast() != dataType
                || dataType == DataType.BOOL && reduction != ScatterReduction.NONE) {
            throw new IllegalArgumentException("scatter boundary types are inconsistent");
        }
        if (family == Family.SCATTER_ADD && reduction != ScatterReduction.ADD) {
            throw new IllegalArgumentException("SCATTER_ADD is fixed addition");
        }
        boolean floatingProduct = reduction == ScatterReduction.MUL
                && (dataType == DataType.FLOAT64 || dataType == DataType.FLOAT32
                    || dataType == DataType.BFLOAT16);
        if (scratchSignature < 0 || scratchSignature > 1
                || scratchSignature > 0 && !floatingProduct) {
            throw new IllegalArgumentException("scatter scratch signature is inconsistent");
        }
    }

    /**
     * Returns the instruction-free canonical generated form.
     *
     * @return a new non-null canonical kernel IR with one output store
     */
    public CpuKernelIr encodedKernelIr() {
        var values = new ArrayList<CpuKernelIr.Value>();
        for (int i = 0; i < boundaryTypes.size(); i++) {
            values.add(new CpuKernelIr.Value(i, boundaryTypes.get(i),
                    i + 1 == boundaryTypes.size() ? CpuKernelIr.Value.Kind.OUTPUT
                            : CpuKernelIr.Value.Kind.INPUT,
                    boundaryAccesses.get(i)));
        }
        String map = occurrenceToBoundary.stream().map(String::valueOf)
                .collect(Collectors.joining(","));
        return new CpuKernelIr(values, List.of(), new CpuKernelIr.Loop("start", "end"),
                List.of(new CpuKernelIr.Store(values.size() - 1, 0)),
                "scatter:" + family + ":reduction=" + reduction + ":map=" + map
                        + ":scratch=" + scratchSignature);
    }

    /** @return the deterministic canonical structural key */
    @Override public String structuralKey() { return encodedKernelIr().structuralKey(); }
}
