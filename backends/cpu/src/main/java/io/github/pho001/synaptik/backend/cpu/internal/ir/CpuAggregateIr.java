package io.github.pho001.synaptik.backend.cpu.internal.ir;

import io.github.pho001.synaptik.model.datatype.DataType;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * CPU-private structural identity for one ordinary numerical, extrema, Boolean, or binding-aware
 * SUM-to-Shape reduction.
 *
 * <p>Selected axes are retained as increasing normalized membership, so attribute-list order is
 * not an execution or artifact fact. Concrete extents, offsets, stride magnitudes, carriers,
 * ranges, and slots remain cold. Every output cell owns its complete canonical row-major domain;
 * no domain is split or combined. Floating numerical rows carry an exact run-owned state shape;
 * all other rows remain workspace-free.</p>
 *
 * @param kind non-null supported numerical, extrema, or Boolean fold
 * @param dataType non-null represented input and output type
 * @param form non-null ordinary axis form or right-aligned SUM-to-Shape form
 * @param selectedAxes non-null increasing normalized selected axes; copied defensively
 * @param keepDimensions whether ordinary selected axes remain as extent-one output dimensions;
 *     always false for SUM-to-Shape because retained target-one positions are encoded by form
 * @param inputAccess non-null structural read access
 * @param outputAccess non-null structural write access
 * @param sourceExtents exact bound source extents for SUM-to-Shape, otherwise empty; copied
 * @param targetExtents exact bound target extents for SUM-to-Shape, otherwise empty; copied
 * @param domainCount exact number of selected values in every output cell
 * @param stateLimbCount exact floating numerical limb count, otherwise zero
 * @param scratchSliceBytes exact floating numerical state bytes per worker range, otherwise zero
 * @param floatingPolicy {@link #FIRST_LOGICAL_NAN_AND_SIGNED_ZERO}; applies to floating extrema
 * @param rangePolicy {@link #COMPLETE_OUTPUT_CELLS}; fixes the meaning of generated bounds
 * @param workspacePolicy {@link #ZERO_WORKSPACE} or {@link #EXACT_FLOATING_STATE}
 */
public record CpuAggregateIr(Kind kind, DataType dataType, Form form, int[] selectedAxes,
        boolean keepDimensions, CpuAccessPlan inputAccess, CpuAccessPlan outputAccess,
        long[] sourceExtents, long[] targetExtents,
        long domainCount, int stateLimbCount, long scratchSliceBytes,
        int floatingPolicy, int rangePolicy, int workspacePolicy) implements CpuPortableKernelIr {
    /**
     * Creates a zero-workspace extrema/Boolean identity for existing schema-compatible callers.
     *
     * @param kind supported extrema or Boolean fold; not {@code null}
     * @param dataType represented input/output type; not {@code null}
     * @param form ordinary attribute form; not {@code null}
     * @param selectedAxes increasing normalized selected axes; copied defensively
     * @param keepDimensions whether selected axes remain as extent-one output dimensions
     * @param inputAccess structural input read access; not {@code null}
     * @param outputAccess structural output write access; not {@code null}
     * @param floatingPolicy fixed extrema floating selection policy
     * @param rangePolicy fixed complete-output-cell range policy
     * @param workspacePolicy required zero-workspace policy
     * @throws NullPointerException if a reference argument is {@code null}
     * @throws IllegalArgumentException if the supplied facts do not describe a supported
     *     zero-workspace aggregate
     */
    public CpuAggregateIr(Kind kind, DataType dataType, Form form, int[] selectedAxes,
            boolean keepDimensions, CpuAccessPlan inputAccess, CpuAccessPlan outputAccess,
            int floatingPolicy, int rangePolicy, int workspacePolicy) {
        this(kind, dataType, form, selectedAxes, keepDimensions, inputAccess, outputAccess,
                new long[0], new long[0], 0, 0, 0,
                floatingPolicy, rangePolicy, workspacePolicy);
    }

    /**
     * Creates an ordinary aggregate identity without bound SUM-to-Shape extent facts.
     *
     * @param kind supported aggregate kind; not {@code null}
     * @param dataType represented input/output type; not {@code null}
     * @param form ordinary attribute form; not {@code null}
     * @param selectedAxes increasing selected axes; copied defensively
     * @param keepDimensions whether selected axes remain as extent-one dimensions
     * @param inputAccess structural input read access; not {@code null}
     * @param outputAccess structural output write access; not {@code null}
     * @param domainCount exact selected-domain cardinality
     * @param stateLimbCount exact-state width, otherwise zero
     * @param scratchSliceBytes exact-state bytes, otherwise zero
     * @param floatingPolicy fixed floating policy
     * @param rangePolicy fixed complete-cell range policy
     * @param workspacePolicy fixed workspace policy
     * @throws NullPointerException if a reference argument is {@code null}
     * @throws IllegalArgumentException if the supplied facts do not describe a supported
     *     ordinary aggregate
     */
    public CpuAggregateIr(Kind kind, DataType dataType, Form form, int[] selectedAxes,
            boolean keepDimensions, CpuAccessPlan inputAccess, CpuAccessPlan outputAccess,
            long domainCount, int stateLimbCount, long scratchSliceBytes,
            int floatingPolicy, int rangePolicy, int workspacePolicy) {
        this(kind, dataType, form, selectedAxes, keepDimensions, inputAccess, outputAccess,
                new long[0], new long[0], domainCount, stateLimbCount, scratchSliceBytes,
                floatingPolicy, rangePolicy, workspacePolicy);
    }
    /**
     * Identifies the closed reduction semantics implemented by this CPU route.
     * Numeric kinds retain the represented input type; Boolean kinds consume and produce the
     * canonical one-byte Boolean representation.
     */
    public enum Kind {
        /** Computes the exact real sum before one represented rounding. */
        SUM,
        /** Divides the exact real sum by the exact domain cardinality before rounding. */
        MEAN,
        /** Computes the exact real product before one represented rounding. */
        PROD,
        /** Selects the numerically least value, with negative zero preferred over positive zero. */
        MIN,
        /** Selects the numerically greatest value, with positive zero preferred over negative zero. */
        MAX,
        /** Conjoins every canonical Boolean value in the selected domain. */
        ALL,
        /** Disjoins every canonical Boolean value in the selected domain. */
        ANY
    }

    /** Identifies which exact Model attribute shape supplied the selected axes. */
    public enum Form {
        /** Selects every input axis and produces the canonical scalar output. */
        FULL,
        /** Selects one already normalized input axis. */
        SINGLE_AXIS,
        /** Selects a canonical increasing membership set, including the valid empty set. */
        MULTI_AXIS,
        /** Reduces leading and unequal target-one axes under exact right alignment. */
        SUM_TO_SHAPE
    }

    /**
     * Selects the represented bits of the first NaN in canonical logical traversal order and
     * gives negative/positive zero the explicit minimum/maximum preference respectively.
     */
    public static final int FIRST_LOGICAL_NAN_AND_SIGNED_ZERO = 1;
    /** Generated bounds denote whole flattened output cells. */
    public static final int COMPLETE_OUTPUT_CELLS = 1;
    /** No workspace, partial value, or combine state is permitted. */
    public static final int ZERO_WORKSPACE = 0;
    /** One run-owned exact floating accumulator slice per simultaneously executing range. */
    public static final int EXACT_FLOATING_STATE = 1;

    /**
     * Validates and snapshots one structural aggregate identity.
     *
     * @param kind supported numerical, extrema, or Boolean-fold meaning; not {@code null}
     * @param dataType represented input/output type from the exact supported matrix; not
     *     {@code null}
     * @param form exact ordinary or SUM-to-Shape attribute form; not {@code null}
     * @param selectedAxes increasing normalized selected-axis membership; not {@code null} and
     *     copied defensively
     * @param keepDimensions whether ordinary selected axes remain as extent-one output
     *     dimensions; false for SUM-to-Shape
     * @param inputAccess structural read access whose rank bounds every selected axis; not
     *     {@code null}
     * @param outputAccess structural write access with the rank implied by the form and retention
     *     policy; not {@code null}
     * @param sourceExtents exact non-negative bound source extents for SUM-to-Shape, otherwise
     *     empty; copied defensively
     * @param targetExtents exact non-negative bound target extents for SUM-to-Shape, otherwise
     *     empty; copied defensively
     * @param domainCount exact non-negative selected-domain cardinality
     * @param stateLimbCount positive exact-state width for floating numerical rows, otherwise zero
     * @param scratchSliceBytes positive exact-state slice size for floating numerical rows,
     *     otherwise zero
     * @param floatingPolicy fixed first-logical-NaN and signed-zero policy
     * @param rangePolicy fixed complete-output-cell range policy
     * @param workspacePolicy fixed zero-workspace or exact-floating-state policy
     * @throws NullPointerException if a reference component is {@code null}
     * @throws IllegalArgumentException if type, extents, axes, access roles, or fixed policies
     *     disagree, including an invalid right-aligned SUM-to-Shape obligation
     */
    public CpuAggregateIr {
        Objects.requireNonNull(kind, "kind"); Objects.requireNonNull(dataType, "dataType");
        Objects.requireNonNull(form, "form"); Objects.requireNonNull(selectedAxes, "selectedAxes");
        Objects.requireNonNull(inputAccess, "inputAccess"); Objects.requireNonNull(outputAccess, "outputAccess");
        Objects.requireNonNull(sourceExtents, "sourceExtents");
        Objects.requireNonNull(targetExtents, "targetExtents");
        selectedAxes = selectedAxes.clone(); sourceExtents = sourceExtents.clone();
        targetExtents = targetExtents.clone();
        for (int i = 0; i < selectedAxes.length; i++) if (selectedAxes[i] < 0
                || selectedAxes[i] >= inputAccess.iterationRank()
                || i > 0 && selectedAxes[i - 1] >= selectedAxes[i])
            throw new IllegalArgumentException("aggregate axes must be increasing and distinct");
        boolean bool = kind == Kind.ALL || kind == Kind.ANY;
        boolean floating = dataType == DataType.FLOAT64 || dataType == DataType.FLOAT32
                || dataType == DataType.BFLOAT16;
        boolean reduction = form != Form.SUM_TO_SHAPE || selectedAxes.length != 0;
        boolean exact = reduction && (kind == Kind.SUM || kind == Kind.MEAN || kind == Kind.PROD);
        boolean numeric = dataType == DataType.FLOAT64 || dataType == DataType.FLOAT32
                || dataType == DataType.BFLOAT16 || dataType == DataType.INT32
                || dataType == DataType.INT64;
        if (bool != (dataType == DataType.BOOL) || !bool && !numeric
                || kind == Kind.MEAN && !floating || domainCount < 0
                || exact && floating && (stateLimbCount <= 0 || scratchSliceBytes <= 0
                    || workspacePolicy != EXACT_FLOATING_STATE)
                || (!exact || !floating) && (stateLimbCount != 0 || scratchSliceBytes != 0
                    || workspacePolicy != ZERO_WORKSPACE)
                || form == Form.SINGLE_AXIS && selectedAxes.length != 1
                || form == Form.FULL && (keepDimensions
                    || selectedAxes.length != inputAccess.iterationRank())
                || form == Form.FULL && outputAccess.iterationRank() != 0
                || form == Form.SUM_TO_SHAPE && (kind != Kind.SUM || keepDimensions
                    || outputAccess.iterationRank() > inputAccess.iterationRank()
                    || sourceExtents.length != inputAccess.iterationRank()
                    || targetExtents.length != outputAccess.iterationRank()
                    || selectedAxes.length == 0 && domainCount != 1)
                || form != Form.SUM_TO_SHAPE
                    && (sourceExtents.length != 0 || targetExtents.length != 0)
                || form != Form.FULL && form != Form.SUM_TO_SHAPE
                    && outputAccess.iterationRank() != (keepDimensions
                        ? inputAccess.iterationRank()
                        : inputAccess.iterationRank() - selectedAxes.length)
                || inputAccess.accessKind() != CpuAccessPlan.AccessKind.READ
                || outputAccess.accessKind() != CpuAccessPlan.AccessKind.WRITE
                || floatingPolicy != FIRST_LOGICAL_NAN_AND_SIGNED_ZERO
                || rangePolicy != COMPLETE_OUTPUT_CELLS)
            throw new IllegalArgumentException("aggregate structural facts disagree");
        if (form == Form.SUM_TO_SHAPE) validateSumToShape(
                sourceExtents, targetExtents, selectedAxes, domainCount);
    }

    /**
     * Returns canonical selected-axis membership without exposing retained record state.
     *
     * @return a new array containing the increasing normalized selected axes; never {@code null}
     */
    @Override public int[] selectedAxes() { return selectedAxes.clone(); }
    /**
     * Returns the bound source extents needed to validate and execute SUM-to-Shape.
     *
     * @return a defensive copy of the exact source extents, or an empty array for an ordinary
     *     aggregate; never {@code null}
     */
    @Override public long[] sourceExtents() { return sourceExtents.clone(); }
    /**
     * Returns the bound target extents needed to validate and execute SUM-to-Shape.
     *
     * @return a defensive copy of the exact target extents, or an empty array for an ordinary
     *     aggregate; never {@code null}
     */
    @Override public long[] targetExtents() { return targetExtents.clone(); }

    private static void validateSumToShape(long[] source, long[] target, int[] selected,
            long domainCount) {
        if (target.length > source.length || Arrays.stream(source).anyMatch(value -> value < 0)
                || Arrays.stream(target).anyMatch(value -> value < 0))
            throw new IllegalArgumentException("sum-to-Shape extents disagree");
        int selectedIndex = 0, leading = source.length - target.length;
        long expectedDomain = 1;
        for (int axis = 0; axis < source.length; axis++) {
            boolean reduce;
            if (axis < leading) reduce = true;
            else {
                long output = target[axis - leading];
                reduce = source[axis] != output && output == 1;
                if (source[axis] != output && !reduce)
                    throw new IllegalArgumentException("sum-to-Shape alignment disagrees");
            }
            if (reduce) {
                if (selectedIndex >= selected.length || selected[selectedIndex++] != axis)
                    throw new IllegalArgumentException("sum-to-Shape axes disagree");
                expectedDomain = expectedDomain == 0 || source[axis] == 0 ? 0
                        : Math.multiplyExact(expectedDomain, source[axis]);
            }
        }
        if (selectedIndex != selected.length || expectedDomain != domainCount)
            throw new IllegalArgumentException("sum-to-Shape domain disagrees");
    }

    /**
     * Encodes this aggregate as the two-boundary structural identity consumed by specialization.
     * Schema 29 includes domain cardinality and exact-state shape whenever they affect generated
     * bytes or resources, while concrete layouts, carrier instances, and assigned slots stay cold.
     *
     * @return a new canonical input/output kernel identity; never {@code null}
     */
    public CpuKernelIr encodedKernelIr() {
        var values = List.of(new CpuKernelIr.Value(0, dataType, CpuKernelIr.Value.Kind.INPUT, inputAccess),
                new CpuKernelIr.Value(1, dataType, CpuKernelIr.Value.Kind.OUTPUT, outputAccess));
        return new CpuKernelIr(values, List.of(), new CpuKernelIr.Loop("start", "end"),
                List.of(new CpuKernelIr.Store(1, 0)), "aggregate:" + kind + ":" + form
                        + ":axes=" + Arrays.toString(selectedAxes) + ":keep=" + keepDimensions
                        + ":source=" + Arrays.toString(sourceExtents)
                        + ":target=" + Arrays.toString(targetExtents)
                        + ":domain=" + domainCount + ":limbs=" + stateLimbCount
                        + ":slice=" + scratchSliceBytes
                        + ":floating=" + floatingPolicy + ":range=" + rangePolicy
                        + ":workspace=" + workspacePolicy);
    }

    /**
     * Returns compatibility identity while excluding extents, layouts, carriers, ranges, and
     * assigned slots that do not change generated bytes.
     *
     * @return a non-null lowercase hexadecimal structural key
     */
    @Override public String structuralKey() { return encodedKernelIr().structuralKey(); }
}
