package io.github.pho001.synaptik.backend.cpu.internal.ir;

import io.github.pho001.synaptik.model.datatype.DataType;
import java.util.List;
import java.util.Objects;

/**
 * CPU-private structural form of one generated gather or one-hot output writer.
 * The form records only facts that can change generated code: the indexing family, semantic
 * input-occurrence mapping, ordered boundary types, and structural access plans. Concrete
 * extents, axes, tuple depth, layout magnitudes, carriers, and validated values remain cold
 * geometry and therefore do not enter this representation.
 *
 * @param family non-null closed indexing family
 * @param occurrenceToBoundary non-null semantic-input-occurrence to unique-boundary positions;
 *     copied defensively
 * @param boundaryTypes non-null ordered unique-input then output data types; copied defensively
 * @param boundaryAccesses non-null structural access plans in the same order as
 *     {@code boundaryTypes}; copied defensively
 */
public record CpuIndexingIr(Family family, List<Integer> occurrenceToBoundary,
        List<DataType> boundaryTypes, List<CpuAccessPlan> boundaryAccesses)
        implements CpuPortableKernelIr {
    /** Closed indexing families with distinct generated coordinate mappings. */
    public enum Family {
        /** Replaces one data axis with the complete indices shape. */ GATHER,
        /** Selects one data-axis coordinate at every aligned indices coordinate. */
        GATHER_ELEMENTS,
        /** Selects a data prefix through coordinate tuples in the final indices axis. */ GATHER_ND,
        /** Appends a dense canonical-BOOL indicator axis. */ ONE_HOT
    }

    /**
     * Validates and snapshots the structural indexing form.
     *
     * @throws NullPointerException if a component or element is {@code null}
     * @throws IllegalArgumentException if boundary cardinalities disagree or the occurrence map
     *     has the wrong family arity
     */
    public CpuIndexingIr {
        Objects.requireNonNull(family, "family");
        occurrenceToBoundary = List.copyOf(occurrenceToBoundary);
        boundaryTypes = List.copyOf(boundaryTypes);
        boundaryAccesses = List.copyOf(boundaryAccesses);
        if (boundaryTypes.size() < 2 || boundaryTypes.size() != boundaryAccesses.size()) {
            throw new IllegalArgumentException("indexing boundary facts must agree");
        }
        if (occurrenceToBoundary.size() != (family == Family.ONE_HOT ? 1 : 2)) {
            throw new IllegalArgumentException("indexing occurrence map has wrong arity");
        }
    }

    /**
     * Returns the instruction-free generated compatibility encoding.
     *
     * @return a new canonical kernel IR with ordered boundary values, one output store, and the
     *     family plus occurrence map in its identity; never {@code null}
     */
    public CpuKernelIr encodedKernelIr() {
        var values = new java.util.ArrayList<CpuKernelIr.Value>();
        for (int i = 0; i < boundaryTypes.size(); i++) values.add(new CpuKernelIr.Value(i,
                boundaryTypes.get(i), i + 1 == boundaryTypes.size()
                    ? CpuKernelIr.Value.Kind.OUTPUT : CpuKernelIr.Value.Kind.INPUT,
                boundaryAccesses.get(i)));
        String map = occurrenceToBoundary.stream().map(String::valueOf)
                .collect(java.util.stream.Collectors.joining(","));
        return new CpuKernelIr(values, List.of(), new CpuKernelIr.Loop("start", "end"),
                List.of(new CpuKernelIr.Store(values.size() - 1, 0)),
                "indexing:" + family + ":map=" + map);
    }

    @Override public String structuralKey() { return encodedKernelIr().structuralKey(); }
}
