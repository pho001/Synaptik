package io.github.pho001.synaptik.backend.cpu.internal.ir;

import io.github.pho001.synaptik.model.datatype.DataType;
import java.util.List;
import java.util.Objects;

/**
 * CPU-private structural form for one static non-affine represented-bit movement.
 * Concrete extents, strides, offsets, axes, padding widths, repeats, and segment lengths are
 * deliberately retained in cold geometry rather than this generated-code identity.
 *
 * @param dataType represented-bit element type shared by every boundary
 * @param plan closed movement-family plan
 * @param inputAccesses unique input-boundary read forms in first-occurrence order
 * @param outputAccess sole injective output write form
 */
public record CpuDataMovementIr(DataType dataType, MovementPlan plan,
        List<CpuAccessPlan> inputAccesses, CpuAccessPlan outputAccess)
        implements CpuPortableKernelIr {
    /**
     * Closed structural movement-plan vocabulary. Concrete extents, layout magnitudes, window
     * starts, and other invocation facts remain in cold geometry rather than artifact identity.
     */
    public sealed interface MovementPlan permits PadPlan, TilePlan, ConcatPlan, StackPlan,
            UnfoldAxisPlan, Unfold2dPlan {
        /**
         * Returns the stable generated family name.
         *
         * @return non-null family name
         */
        String family();
        /**
         * Returns the output iteration rank.
         *
         * @return non-negative iteration rank
         */
        int outputRank();
        /**
         * Returns ordered semantic input roles.
         *
         * @return immutable occurrence-to-boundary map
         */
        List<Integer> occurrenceToBoundary();
        /**
         * Returns the structural immediate.
         *
         * @return exact pad bits, or zero when not padding
         */
        long immediateBits();
    }
    /**
     * Structural constant-padding plan.
     *
     * @param outputRank non-negative output iteration rank
     * @param immediateBits exact represented padding bits in the low-width portion appropriate
     *     to {@link CpuDataMovementIr#dataType()}
     */
    public record PadPlan(int outputRank, long immediateBits) implements MovementPlan {
        /** Validates the non-negative output rank. @throws IllegalArgumentException if negative */
        public PadPlan { if (outputRank < 0) throw new IllegalArgumentException("rank must be non-negative"); }
        @Override public String family() { return "PAD"; }
        @Override public List<Integer> occurrenceToBoundary() { return List.of(0); }
    }
    /**
     * Structural complete-pattern tiling plan.
     *
     * @param outputRank non-negative output iteration rank
     */
    public record TilePlan(int outputRank) implements MovementPlan {
        /** Validates the non-negative output rank. @throws IllegalArgumentException if negative */
        public TilePlan { if (outputRank < 0) throw new IllegalArgumentException("rank must be non-negative"); }
        @Override public String family() { return "TILE"; }
        @Override public List<Integer> occurrenceToBoundary() { return List.of(0); }
        @Override public long immediateBits() { return 0; }
    }
    /**
     * Structural ordered concatenation plan.
     *
     * @param outputRank positive output iteration rank
     * @param occurrenceToBoundary one-through-sixteen ordered semantic occurrences mapped to
     *     unique input-boundary positions; copied defensively
     */
    public record ConcatPlan(int outputRank, List<Integer> occurrenceToBoundary)
            implements MovementPlan {
        /**
         * Validates the positive output rank and snapshots the occurrence map.
         * @throws NullPointerException if {@code occurrenceToBoundary} is {@code null}
         * @throws IllegalArgumentException if the rank or occurrence count is outside its bound
         */
        public ConcatPlan {
            if (outputRank <= 0) throw new IllegalArgumentException("concat rank must be positive");
            occurrenceToBoundary = checkedMap(occurrenceToBoundary);
        }
        @Override public String family() { return "CONCAT"; }
        @Override public long immediateBits() { return 0; }
    }
    /**
     * Structural ordered stacking plan.
     *
     * @param outputRank positive output iteration rank
     * @param occurrenceToBoundary one-through-sixteen ordered semantic occurrences mapped to
     *     unique input-boundary positions; copied defensively
     */
    public record StackPlan(int outputRank, List<Integer> occurrenceToBoundary)
            implements MovementPlan {
        /**
         * Validates the positive output rank and snapshots the occurrence map.
         * @throws NullPointerException if {@code occurrenceToBoundary} is {@code null}
         * @throws IllegalArgumentException if the rank or occurrence count is outside its bound
         */
        public StackPlan {
            if (outputRank <= 0) throw new IllegalArgumentException("stack result rank must be positive");
            occurrenceToBoundary = checkedMap(occurrenceToBoundary);
        }
        @Override public String family() { return "STACK"; }
        @Override public long immediateBits() { return 0; }
    }
    /**
     * Structural general-axis window-extraction plan whose axis, window size, step, extents, and
     * layout magnitudes remain cold geometry. The occurrence map selects the sole input boundary.
     *
     * @param outputRank result iteration rank, exactly one greater than the input rank
     */
    public record UnfoldAxisPlan(int outputRank) implements MovementPlan {
        /**
         * Validates the result-rank boundary.
         * @throws IllegalArgumentException if {@code outputRank} is less than two
         */
        public UnfoldAxisPlan {
            if (outputRank < 2) throw new IllegalArgumentException("unfold-axis result rank must be at least two");
        }
        @Override public String family() { return "UNFOLD_AXIS"; }
        @Override public List<Integer> occurrenceToBoundary() { return List.of(0); }
        @Override public long immediateBits() { return 0; }
    }
    /**
     * Structural NCHW-to-columns window-extraction plan. Exact represented padding bits shape
     * generated compatibility while all spatial geometry remains cold. Direct conceptual zero
     * and exact typed positive zero therefore share structural identity.
     *
     * @param outputRank canonical rank-three result iteration rank
     * @param immediateBits exact represented padding bits in the type-appropriate low bits
     */
    public record Unfold2dPlan(int outputRank, long immediateBits) implements MovementPlan {
        /**
         * Validates the canonical result rank.
         * @throws IllegalArgumentException if {@code outputRank} is not three
         */
        public Unfold2dPlan {
            if (outputRank != 3) throw new IllegalArgumentException("unfold2d result rank must be three");
        }
        @Override public String family() { return "UNFOLD2D"; }
        @Override public List<Integer> occurrenceToBoundary() { return List.of(0); }
    }

    /**
     * Validates one structural movement identity and snapshots its unique input accesses.
     *
     * @throws NullPointerException if a required component or input access is {@code null}
     * @throws IllegalArgumentException if boundary counts, access roles, ranks, or occurrence
     *     positions are inconsistent
     */
    public CpuDataMovementIr {
        Objects.requireNonNull(dataType, "dataType");
        Objects.requireNonNull(plan, "plan");
        inputAccesses = List.copyOf(inputAccesses);
        Objects.requireNonNull(outputAccess, "outputAccess");
        if (inputAccesses.isEmpty() || inputAccesses.size() > 16
                || outputAccess.accessKind() != CpuAccessPlan.AccessKind.WRITE
                || outputAccess.iterationRank() != plan.outputRank()) {
            throw new IllegalArgumentException("movement boundary structure is inconsistent");
        }
        for (CpuAccessPlan access : inputAccesses) {
            if (access.accessKind() != CpuAccessPlan.AccessKind.READ) {
                throw new IllegalArgumentException("movement inputs must be read-only access forms");
            }
        }
        if (plan instanceof UnfoldAxisPlan && (inputAccesses.size() != 1
                || inputAccesses.getFirst().iterationRank() + 1 != plan.outputRank())) {
            throw new IllegalArgumentException("unfold-axis access ranks are inconsistent");
        }
        if (plan instanceof Unfold2dPlan && (inputAccesses.size() != 1
                || inputAccesses.getFirst().iterationRank() != 4)) {
            throw new IllegalArgumentException("unfold2d access ranks are inconsistent");
        }
        for (int boundary : plan.occurrenceToBoundary()) {
            if (boundary < 0 || boundary >= inputAccesses.size()) {
                throw new IllegalArgumentException("movement occurrence boundary is out of range");
            }
        }
    }

    /**
     * Returns the instruction-free cache/generator encoding.
     *
     * @return a new canonical encoding containing boundary values, one output store, and the
     *     complete structural movement identity; never {@code null}
     */
    public CpuKernelIr encodedKernelIr() {
        var values = new java.util.ArrayList<CpuKernelIr.Value>();
        for (int i = 0; i < inputAccesses.size(); i++) values.add(new CpuKernelIr.Value(i,
                dataType, CpuKernelIr.Value.Kind.INPUT, inputAccesses.get(i)));
        values.add(new CpuKernelIr.Value(values.size(), dataType, CpuKernelIr.Value.Kind.OUTPUT,
                outputAccess));
        String mapping = plan.occurrenceToBoundary().stream().map(String::valueOf)
                .collect(java.util.stream.Collectors.joining(","));
        String identity = "movement:" + plan.family() + ":rank=" + plan.outputRank()
                + ":map=" + mapping + ":bits=" + Long.toUnsignedString(plan.immediateBits(), 16);
        return new CpuKernelIr(values, List.of(), new CpuKernelIr.Loop("start", "end"),
                List.of(new CpuKernelIr.Store(values.size() - 1, 0)), identity);
    }

    @Override public String structuralKey() { return encodedKernelIr().structuralKey(); }

    private static List<Integer> checkedMap(List<Integer> source) {
        source = List.copyOf(source);
        if (source.isEmpty() || source.size() > 16) {
            throw new IllegalArgumentException("composition requires one through sixteen occurrences");
        }
        return source;
    }
}
