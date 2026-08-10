package io.github.pho001.synaptik.backend.cpu.internal.ir;

import io.github.pho001.synaptik.model.datatype.DataType;
import java.util.List;
import java.util.Objects;

/**
 * Immutable represented-bit copy from one composed affine source to one materialized result
 * boundary.
 *
 * <p>The mapping steps describe the validated static view chain for structural identity. Exact
 * source and result addresses are composed during cold lowering and supplied separately to the
 * generated invocation. The copy performs no conversion or arithmetic; BFLOAT16 elements are
 * opaque 16-bit payloads.</p>
 *
 * @param dataType non-null logical type whose represented bits are copied unchanged
 * @param sourceAccess non-null read access form for the composed source address domain
 * @param resultAccess non-null write access form for the final result address domain
 * @param mappingSteps non-null ordered structural view transformations; copied defensively
 * @param writeDomain non-null logical-element or deduplicated distinct-address iteration form
 */
public record CpuAffineCopyIr(DataType dataType, CpuAccessPlan sourceAccess,
        CpuAccessPlan resultAccess, List<MappingStep> mappingSteps,
        WriteDomain writeDomain) implements CpuPortableKernelIr {
    /** Iteration domain used by the copy. */
    public enum WriteDomain {
        /** One disjoint write for every logical result coordinate. */ LOGICAL_ELEMENTS,
        /** One deterministic write for every distinct referenced result address. */ DISTINCT_ADDRESSES
    }
    /** Closed structural affine-mapping vocabulary. */
    public enum MappingKind {
        /** Explicit contiguous boundary semantics. */ CONTIGUOUS,
        /** Ordered-logical-element reshape. */ RESHAPE,
        /** Right-aligned singleton repetition. */ EXPAND,
        /** Axis permutation. */ PERMUTE,
        /** Singleton-axis insertion. */ EXPAND_DIMS,
        /** Singleton-axis removal. */ SQUEEZE,
        /** Fixed-coordinate axis removal. */ SELECT,
        /** Positive-step normalized slice. */ SLICE,
        /** Target-relative crop with an explicit prefix Shape. */ CROP_TO_SHAPE
    }
    /**
     * Structural coordinate transformation composed during cold lowering.
     *
     * @param kind non-null admitted affine transformation
     * @param inputRank non-negative input rank
     * @param outputRank non-negative output rank
     * @param axes non-null normalized axes relevant to the transformation; copied defensively
     */
    public record MappingStep(MappingKind kind, int inputRank, int outputRank,
            List<Integer> axes) {
        /**
         * Retains one non-null stable kind, non-negative ranks, and ordered structural axes.
         *
         * @throws NullPointerException if {@code kind}, {@code axes}, or an axis is {@code null}
         * @throws IllegalArgumentException if either rank is negative
         */
        public MappingStep {
            Objects.requireNonNull(kind, "kind");
            axes = List.copyOf(axes);
            if (inputRank < 0 || outputRank < 0) throw new IllegalArgumentException(
                    "mapping-step ranks must be non-negative");
        }
    }

    /**
     * Validates the closed one-source/one-result copy form.
     *
     * @throws NullPointerException if a reference component is {@code null}
     * @throws IllegalArgumentException if the access plans are not one read and one write over
     *     the same iteration rank
     */
    public CpuAffineCopyIr {
        Objects.requireNonNull(dataType, "dataType");
        Objects.requireNonNull(sourceAccess, "sourceAccess");
        Objects.requireNonNull(resultAccess, "resultAccess");
        mappingSteps = List.copyOf(mappingSteps);
        Objects.requireNonNull(writeDomain, "writeDomain");
        if (sourceAccess.accessKind() != CpuAccessPlan.AccessKind.READ
                || resultAccess.accessKind() != CpuAccessPlan.AccessKind.WRITE
                || sourceAccess.iterationRank() != resultAccess.iterationRank()) {
            throw new IllegalArgumentException("affine copy access plans must share one read/write domain");
        }
    }

    /**
     * Returns the cache-compatible instruction-free portable copy form.
     *
     * @return a new canonical two-boundary IR carrying one ordered store and no instructions
     */
    public CpuKernelIr encodedKernelIr() {
        String identity = "affine:" + dataType + ':' + mappingSteps + ':' + writeDomain;
        return new CpuKernelIr(List.of(
                new CpuKernelIr.Value(0, dataType, CpuKernelIr.Value.Kind.INPUT, sourceAccess),
                new CpuKernelIr.Value(1, dataType, CpuKernelIr.Value.Kind.OUTPUT, resultAccess)),
                List.of(), new CpuKernelIr.Loop("start", "end"),
                List.of(new CpuKernelIr.Store(1, 0)), identity);
    }

    /**
     * Returns the generated-code compatibility key of the encoded copy form.
     *
     * @return a non-null lowercase hexadecimal structural key
     */
    @Override public String structuralKey() { return encodedKernelIr().structuralKey(); }
}
