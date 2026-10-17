package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import org.junit.jupiter.api.Test;

/**
 * Stage-A permission and carrier-fidelity gate for CPU task 0008O.
 *
 * <p>This is deliberately a test-only characterization.  It neither selects a CPU route nor
 * changes a generated schema. It is neither a generated-artifact assertion nor evidence that a
 * generated body uses a Vector API load. In particular, {@code DIRECT_SCALAR} in the reviewed IR
 * remains a scalar identity, not permission to reassociate a stable reduction. The table is kept
 * in executable form so every family/pass/candidate/type decision is checked rather than inferred
 * from Vector API availability.</p>
 */
final class CpuStableReductionVectorNumericalSpikeTest {
    // These are the five Stage-A semantic forms, not merely the three emitter owners.
    private enum Family { SOFTMAX, LOG_SOFTMAX, DENSE_CATEGORICAL, INDEX_CATEGORICAL, ATTENTION }
    private enum Pass { MAX, EXP_SUM, NORMALIZE_OR_LOG, CONTRIBUTION_OR_WEIGHT, WEIGHTED_VALUE }
    private enum Candidate { ORDERED_SCALAR, VECTOR_MAP_ORDERED_FOLD, LANE_LOCAL_FOLD, LANE_REORDERED }
    private enum Type { FLOAT32, FLOAT64 }
    private enum Access { CONTIGUOUS, INDEXED_OR_NONCONTIGUOUS_UNPROVED }
    private enum Decision {
        NOT_APPLICABLE,
        EXACT_CURRENT_CPU,
        SCALAR_ONLY_CONTROL,
        STOP_MODEL_OR_ARCHITECTURE_DECISION
    }
    private enum Carrier { ARRAY_ARRAY, SEGMENT_SEGMENT, ARRAY_SEGMENT, SEGMENT_ARRAY }

    /** One source-backed, type- and access-specific Stage-A applicability/permission decision. */
    private record Permission(Family family, Pass pass, Candidate candidate, Type type,
            Access access, Decision decision, String source) {
        private PermissionKey key() {
            return new PermissionKey(family, pass, candidate, type, access);
        }
    }

    /** The matrix coordinate; decision/source are intentionally excluded from uniqueness. */
    private record PermissionKey(Family family, Pass pass, Candidate candidate, Type type,
            Access access) { }

    /**
     * Ensures the Stage-A table is complete without inventing either a semantic pass or vector
     * access permission.
     *
     * <p>The table has one row for every family/pass/candidate/type/access combination. A pass
     * that is not part of the family is {@link Decision#NOT_APPLICABLE}, rather than a fabricated
     * successful pass. Candidate two is exact only for the contiguous access proven by this test;
     * indexed/non-contiguous access is an explicit scalar-only control until a legal Vector API
     * indexed load is proved. Candidates three and four stop: the Model contracts either defer
     * reduction order or make reassociation conditional on a later concrete conformance tolerance.
     * Candidate two itself never reassociates because it consumes vector-loaded lanes in increasing
     * logical order.</p>
     */
    @Test void stageAPermissionTableIsCompleteAndDoesNotManufactureSoftmaxPermission() {
        List<Permission> table = permissionTable();
        int expectedRows = Family.values().length * Pass.values().length * Candidate.values().length
                * Type.values().length * Access.values().length;
        assertEquals(expectedRows, table.size());
        assertEquals(table.size(), table.stream().distinct().count());
        assertEquals(expectedRows, table.stream().map(Permission::key).distinct().count(),
                "each family/pass/candidate/type/access coordinate must occur once");
        for (Family family : Family.values()) for (Pass pass : Pass.values())
            for (Candidate candidate : Candidate.values()) for (Type type : Type.values())
                for (Access access : Access.values()) {
                    PermissionKey key = new PermissionKey(family, pass, candidate, type, access);
                    assertEquals(1, table.stream().filter(row -> row.key().equals(key)).count(),
                            "missing or duplicate permission coordinate: " + key);
                }
        for (Permission row : table) {
            assertFalse(row.source().isBlank(), row.toString());
            if (!applies(row.family(), row.pass())) {
                assertEquals(Decision.NOT_APPLICABLE, row.decision(), row.toString());
            } else if (row.candidate() == Candidate.ORDERED_SCALAR) {
                assertEquals(Decision.EXACT_CURRENT_CPU, row.decision(), row.toString());
            } else if (row.candidate() == Candidate.VECTOR_MAP_ORDERED_FOLD
                    && row.access() == Access.CONTIGUOUS) {
                assertEquals(Decision.EXACT_CURRENT_CPU, row.decision(), row.toString());
            } else if (row.candidate() == Candidate.VECTOR_MAP_ORDERED_FOLD) {
                assertEquals(Decision.SCALAR_ONLY_CONTROL, row.decision(), row.toString());
            } else {
                assertEquals(Decision.STOP_MODEL_OR_ARCHITECTURE_DECISION, row.decision(),
                        row.toString());
            }
        }
        assertEquals(144, table.stream().filter(row -> row.decision() == Decision.NOT_APPLICABLE).count());
        assertEquals(96, table.stream().filter(row -> row.decision() == Decision.EXACT_CURRENT_CPU).count());
        assertEquals(32, table.stream().filter(row -> row.decision() == Decision.SCALAR_ONLY_CONTROL).count());
        assertEquals(128, table.stream().filter(row -> row.decision()
                == Decision.STOP_MODEL_OR_ARCHITECTURE_DECISION).count());
    }

    /**
     * Proves this test control's vector loading preserves each raw input-carrier value before any
     * reduction order is considered. It does not prove generated code or an output-carrier path.
     * Full blocks use the actual preferred species; tails are scalar controls, as required by the
     * task.
     */
    @Test void stageAVectorLoadsPreserveRawBitsForEveryCarrierAndRequiredWidth() {
        for (Type type : Type.values()) for (Carrier carrier : Carrier.values()) {
            int lanes = lanes(type);
            assertTrue(lanes > 1, type + " has no multi-lane preferred species");
            for (int width : List.of(1, lanes - 1, lanes, lanes + 1, 2 * lanes + 3)) {
                double[] stimulus = stimulus(width);
                try (Arena arena = Arena.ofConfined()) {
                    Object input = carrierInput(type, carrier, stimulus, arena);
                    for (int offset = 0; offset < width; ) {
                        int block = Math.min(lanes, width - offset);
                        if (block == lanes) assertRawBlock(type, input, offset, stimulus);
                        else assertScalarTail(type, input, offset, block, stimulus);
                        offset += block;
                    }
                }
            }
        }
    }

    /**
     * Exercises a source-level control for the permitted contiguous vector-map candidate through
     * the stable three-pass softmax shape; it is not generated-artifact evidence.
     * Each full preferred-species load is immediately consumed lane zero through lane {@code n-1};
     * maximum, shifted sum, and store passes therefore retain the scalar logical order and every
     * FLOAT32 narrowing point.  The outer/inner coordinates model first, middle, and last axes,
     * while disjoint ranges model caller-owned complete slices.
     */
    @Test void stageAOrderedVectorMapMatchesScalarSoftmaxRawBitsForAxesCarriersTailsAndRanges() {
        for (Type type : Type.values()) for (Carrier carrier : Carrier.values()) {
            int lanes = lanes(type);
            for (int width : List.of(1, lanes - 1, lanes, lanes + 1, 2 * lanes + 3)) {
                for (double[] values : stableStimuli(width)) {
                    double[] scalar = softmaxScalar(type, values);
                    double[] vector = softmaxVectorMapOrdered(type, carrier, values);
                    assertRawEquals(type, scalar, vector, type + "/" + carrier + "/" + width);
                    // The same complete-slice body is deterministic when callers partition only
                    // independent slices; no candidate owns a partial slice.
                    assertRawEquals(type, vector, softmaxVectorMapOrdered(type, carrier, values),
                            "deterministic caller-parallel control");
                }
            }
        }
    }

    /**
     * Locks the categorical special classes that make reassociation conditionally admissible.
     * Ignored indices are checked before any logits access; non-ignored indices are range checked
     * before the stable slice is evaluated.
     */
    @Test void stageACategoricalPreservesIgnoreIndexValidationEmptyAndSpecialValueClasses() {
        for (Type type : Type.values()) {
            assertEquals(0L, raw(type, indexLoss(type, new double[] {Double.NaN}, -1, -1)));
            assertThrows(IllegalArgumentException.class, () -> indexLoss(type, new double[] {1}, 1, -1));
            assertTrue(Double.isNaN(indexLoss(type, new double[] {Double.NaN, 0}, 1, -1)));
            assertTrue(Double.isNaN(indexLoss(type, new double[] {Double.POSITIVE_INFINITY, 0}, 1, -1)));
            assertTrue(Double.isNaN(indexLoss(type, new double[] {Double.NEGATIVE_INFINITY,
                    Double.NEGATIVE_INFINITY}, 0, -1)));
            assertEquals(Double.POSITIVE_INFINITY, indexLoss(type,
                    new double[] {0, Double.NEGATIVE_INFINITY}, 1, -1));
            assertEquals(0L, raw(type, indexLoss(type, new double[] {0}, 0, -1)));
        }
    }

    /**
     * Locks the attention row classifications independently of a finite reduction order.
     * The tested cases cover all-masked and {@code S == 0} positive zero, eligible NaN,
     * positive-infinity ties, all-negative-infinity zero, causal exclusion, and {@code Ev == 0}.
     */
    @Test void stageAAttentionPreservesEligibleSpecialClassesMasksCausalityAndEmptyValueWidth() {
        assertEquals(0.0, attentionWeight(new double[0], new boolean[0], 0, true));
        assertEquals(0.0, attentionWeight(new double[] {Double.NaN}, new boolean[] {false}, 0, false));
        assertTrue(Double.isNaN(attentionWeight(new double[] {Double.NaN}, new boolean[] {true}, 0, false)));
        assertEquals(0.5, attentionWeight(new double[] {Double.POSITIVE_INFINITY,
                Double.POSITIVE_INFINITY}, new boolean[] {true, true}, 0, false));
        assertEquals(0.0, attentionWeight(new double[] {Double.NEGATIVE_INFINITY,
                Double.NEGATIVE_INFINITY}, new boolean[] {true, true}, 0, false));
        assertEquals(0.0, attentionWeight(new double[] {1, 2}, new boolean[] {true, true}, 1, true));
    }

    private static List<Permission> permissionTable() {
        var result = new ArrayList<Permission>();
        for (Family family : Family.values()) for (Pass pass : Pass.values())
            for (Candidate candidate : Candidate.values()) for (Type type : Type.values())
                for (Access access : Access.values()) {
                    if (!applies(family, pass)) {
                        result.add(new Permission(family, pass, candidate, type, access,
                                Decision.NOT_APPLICABLE, notApplicableSource(family, pass)));
                    } else if (candidate == Candidate.ORDERED_SCALAR) {
                        result.add(new Permission(family, pass, candidate, type, access,
                                Decision.EXACT_CURRENT_CPU, orderedSource(family)));
                    } else if (candidate == Candidate.VECTOR_MAP_ORDERED_FOLD
                            && access == Access.CONTIGUOUS) {
                        result.add(new Permission(family, pass, candidate, type, access,
                                Decision.EXACT_CURRENT_CPU, contiguousVectorMapSource(family)));
                    } else if (candidate == Candidate.VECTOR_MAP_ORDERED_FOLD) {
                        result.add(new Permission(family, pass, candidate, type, access,
                                Decision.SCALAR_ONLY_CONTROL, indexedControlSource()));
                    } else {
                        result.add(new Permission(family, pass, candidate, type, access,
                                Decision.STOP_MODEL_OR_ARCHITECTURE_DECISION,
                                reassociationStopSource(family)));
                    }
                }
        return List.copyOf(result);
    }

    private static boolean applies(Family family, Pass pass) {
        return switch (family) {
            case SOFTMAX, LOG_SOFTMAX -> pass == Pass.MAX || pass == Pass.EXP_SUM
                    || pass == Pass.NORMALIZE_OR_LOG;
            case DENSE_CATEGORICAL, INDEX_CATEGORICAL -> pass == Pass.MAX || pass == Pass.EXP_SUM
                    || pass == Pass.CONTRIBUTION_OR_WEIGHT;
            case ATTENTION -> pass == Pass.MAX || pass == Pass.EXP_SUM
                    || pass == Pass.CONTRIBUTION_OR_WEIGHT || pass == Pass.WEIGHTED_VALUE;
        };
    }

    private static String notApplicableSource(Family family, Pass pass) {
        return "docs/planning/backends/cpu/tasks/0008o-stable-reduction-vector-numerical-spike.md"
                + " §Scope candidate-pass inventory: " + family + "/" + pass
                + " is not one of the named softmax normalize/log, categorical contribution, or"
                + " attention weight/weighted-value paths";
    }

    private static String orderedSource(Family family) {
        return switch (family) {
            case SOFTMAX, LOG_SOFTMAX -> "docs/planning/backends/cpu/tasks/"
                    + "0007e-portable-stable-softmax-and-log-softmax-coverage.md §Scope: selected axis"
                    + " traverses increasing logical coordinate order; CPU 0008O §Architecture constraints"
                    + " preserves that scalar order";
            case DENSE_CATEGORICAL, INDEX_CATEGORICAL -> "docs/planning/backends/cpu/tasks/"
                    + "0008i-portable-loss-family-execution.md §Scope: categorical execution freezes"
                    + " increasing non-class sample and class order; CPU 0008O §Architecture constraints"
                    + " preserves the current scalar accumulation order";
            case ATTENTION -> "docs/planning/backends/cpu/tasks/"
                    + "0008h-portable-scaled-dot-product-attention-execution.md §Scope: one complete row"
                    + " owns full domains; CPU 0008O §Architecture constraints preserves current scalar"
                    + " traversal and weighted-value order";
        };
    }

    private static String contiguousVectorMapSource(Family family) {
        return orderedSource(family) + "; CPU 0008O §Scope candidate 2 permits vector loads/map"
                + " with scalar logical-order folds for the dense-contiguous proven subset";
    }

    private static String indexedControlSource() {
        return "docs/planning/backends/cpu/tasks/0008o-stable-reduction-vector-numerical-spike.md"
                + " §Scope candidate matrix: non-contiguous access may use only proved legal indexed/gather"
                + " access; otherwise it is an explicit scalar-only control (no proof is recorded here)";
    }

    private static String reassociationStopSource(Family family) {
        return switch (family) {
            case SOFTMAX, LOG_SOFTMAX -> "docs/planning/modules/model/tasks/"
                    + "0016i-softmax-semantic-kinds-and-attributes.md §Out of scope excludes reduction"
                    + " order/precision; 0016j §Out of scope likewise excludes numerical stability and"
                    + " reduction order, so no reassociation permission exists";
            case DENSE_CATEGORICAL -> "docs/planning/modules/model/tasks/"
                    + "0022a-dense-target-categorical-cross-entropy-with-logits.md §Data types, result type,"
                    + " accumulation, and computation format permits reassociation subject to later"
                    + " conformance tolerances; no concrete tolerance exists, so this exact-current"
                    + " Stage-A gate stops rather than inventing one";
            case INDEX_CATEGORICAL -> "docs/planning/modules/model/tasks/"
                    + "0022b-index-target-categorical-cross-entropy-with-logits.md §Types, computation, and"
                    + " special values permits reassociation but supplies no concrete tolerance, so Stage A stops";
            case ATTENTION -> "docs/planning/modules/model/tasks/"
                    + "0019e-scaled-dot-product-attention.md §Numerical semantics permits reassociation but"
                    + " states later conformance sets tolerances; no concrete tolerance exists";
        };
    }

    /** Evaluates one index categorical stable-logits group with the Model-required class rules. */
    private static double indexLoss(Type type, double[] logits, int index, int ignore) {
        if (index == ignore) return 0.0;
        if (index < 0 || index >= logits.length) throw new IllegalArgumentException("index");
        boolean nan = false, positive = false, finite = false;
        for (double value : logits) { nan |= Double.isNaN(value); positive |= value == Double.POSITIVE_INFINITY;
            finite |= Double.isFinite(value); }
        if (nan || positive || !finite) return Double.NaN;
        double max = Double.NEGATIVE_INFINITY;
        for (double value : logits) max = Math.max(max, narrow(type, value));
        double sum = 0.0;
        for (double value : logits) sum = narrow(type, sum + narrow(type, StrictMath.exp(narrow(type, value) - max)));
        return narrow(type, max + narrow(type, StrictMath.log(sum)) - narrow(type, logits[index]));
    }

    /** Classifies one requested attention weight without evaluating excluded score/value arithmetic. */
    private static double attentionWeight(double[] score, boolean[] eligible, int requested, boolean causal) {
        if (requested >= score.length || !eligible[requested] || causal && requested > 0) return 0.0;
        int count = 0, positive = 0; boolean nan = false, allNegative = true;
        for (int i = 0; i < score.length; i++) if (eligible[i] && (!causal || i == 0)) {
            count++; nan |= Double.isNaN(score[i]); positive += score[i] == Double.POSITIVE_INFINITY ? 1 : 0;
            allNegative &= score[i] == Double.NEGATIVE_INFINITY;
        }
        if (count == 0 || allNegative) return 0.0;
        if (nan) return Double.NaN;
        if (positive != 0) return score[requested] == Double.POSITIVE_INFINITY ? 1.0 / positive : 0.0;
        double max = Double.NEGATIVE_INFINITY, sum = 0.0;
        for (int i = 0; i < score.length; i++) if (eligible[i] && (!causal || i == 0)) max = Math.max(max, score[i]);
        for (int i = 0; i < score.length; i++) if (eligible[i] && (!causal || i == 0)) sum += StrictMath.exp(score[i] - max);
        return StrictMath.exp(score[requested] - max) / sum;
    }

    /** Returns raw represented bits for the exact-positive-zero assertions. */
    private static long raw(Type type, double value) {
        return type == Type.FLOAT32 ? Float.floatToRawIntBits((float) value)
                : Double.doubleToRawLongBits(value);
    }

    private static int lanes(Type type) {
        return type == Type.FLOAT32 ? FloatVector.SPECIES_PREFERRED.length()
                : DoubleVector.SPECIES_PREFERRED.length();
    }

    private static double[] stimulus(int width) {
        double[] values = new double[width];
        double[] special = {Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY,
                -0.0d, Double.MIN_VALUE, -Double.MIN_VALUE, Double.MAX_VALUE, -Double.MAX_VALUE,
                0.0d, 1.0d, -1.0d};
        for (int index = 0; index < width; index++) values[index] = special[index % special.length];
        return values;
    }

    /** Returns finite and IEEE-special stable-reduction controls for one selected-axis width. */
    private static List<double[]> stableStimuli(int width) {
        var cases = new ArrayList<double[]>();
        cases.add(stimulus(width));
        double[] finite = new double[width];
        for (int i = 0; i < width; i++) finite[i] = i == 0 ? 1000.0 : -1000.0 - i;
        cases.add(finite);
        double[] zeros = new double[width];
        for (int i = 0; i < width; i++) zeros[i] = (i & 1) == 0 ? -0.0 : 0.0;
        cases.add(zeros);
        return List.copyOf(cases);
    }

    /**
     * Executes the current ordered three-pass FLOAT32/FLOAT64 stable softmax control.
     *
     * <p>This deliberately mirrors {@link CpuSoftmaxEmitter}: FLOAT32 arithmetic, including
     * Kahan's addend, temporary, compensation, and sum, is float arithmetic.  It is not a
     * binary64 oracle with a final float cast.  {@code Math}, rather than {@code StrictMath}, is
     * also intentional: that is the call owner in the generated CPU entry.</p>
     */
    private static double[] softmaxScalar(Type type, double[] values) {
        if (type == Type.FLOAT32) return softmaxScalarFloat(values);
        double maximum = Double.NEGATIVE_INFINITY, sum = 0.0, compensation = 0.0;
        for (double value : values) maximum = Math.max(maximum, value);
        for (double value : values) {
            double addend = Math.exp(value - maximum) - compensation;
            double temporary = sum + addend;
            compensation = (temporary - sum) - addend;
            sum = temporary;
        }
        double[] result = new double[values.length];
        for (int i = 0; i < result.length; i++) result[i] = Math.exp(values[i] - maximum) / sum;
        return result;
    }

    /**
     * Executes only full-block preferred-species vector loads/maps plus increasing-lane scalar
     * folds. A vector is loaded once per block per pass, mapped by an identity addition, then
     * consumed lane zero through lane {@code n - 1}; the tail uses the current scalar carrier
     * access.  This is candidate two, not a lane-local reduction disguised as a vector load.
     */
    private static double[] softmaxVectorMapOrdered(Type type, Carrier carrier, double[] values) {
        try (Arena arena = Arena.ofConfined()) {
            Object input = carrierInput(type, carrier, values, arena);
            if (type == Type.FLOAT32) return softmaxVectorMapOrderedFloat(input, values.length);
            double maximum = Double.NEGATIVE_INFINITY;
            int lanes = lanes(type), full = values.length / lanes * lanes;
            for (int base = 0; base < full; base += lanes) {
                DoubleVector block = loadDoubleBlock(input, base).add(0.0d);
                for (int lane = 0; lane < lanes; lane++) maximum = Math.max(maximum, block.lane(lane));
            }
            for (int i = full; i < values.length; i++) maximum = Math.max(maximum, readScalar(type, input, i));
            double sum = 0.0, compensation = 0.0;
            for (int base = 0; base < full; base += lanes) {
                DoubleVector block = loadDoubleBlock(input, base).add(0.0d);
                for (int lane = 0; lane < lanes; lane++) {
                    double addend = Math.exp(block.lane(lane) - maximum) - compensation;
                    double temporary = sum + addend;
                    compensation = (temporary - sum) - addend;
                    sum = temporary;
                }
            }
            for (int i = full; i < values.length; i++) {
                double addend = Math.exp(readScalar(type, input, i) - maximum) - compensation;
                double temporary = sum + addend;
                compensation = (temporary - sum) - addend;
                sum = temporary;
            }
            double[] result = new double[values.length];
            for (int base = 0; base < full; base += lanes) {
                DoubleVector block = loadDoubleBlock(input, base).add(0.0d);
                for (int lane = 0; lane < lanes; lane++) result[base + lane] = Math.exp(
                        block.lane(lane) - maximum) / sum;
            }
            for (int i = full; i < values.length; i++) result[i] = Math.exp(
                    readScalar(type, input, i) - maximum) / sum;
            return result;
        }
    }

    private static double[] softmaxScalarFloat(double[] values) {
        float maximum = Float.NEGATIVE_INFINITY;
        for (double value : values) maximum = Math.max(maximum, (float) value);
        float sum = 0.0f, compensation = 0.0f;
        for (double value : values) {
            float shifted = (float) value - maximum;
            float addend = (float) Math.exp(shifted) - compensation;
            float temporary = sum + addend;
            compensation = (temporary - sum) - addend;
            sum = temporary;
        }
        double[] result = new double[values.length];
        for (int i = 0; i < result.length; i++) result[i] = (float) Math.exp((float) values[i]
                - maximum) / sum;
        return result;
    }

    private static double[] softmaxVectorMapOrderedFloat(Object input, int length) {
        int lanes = lanes(Type.FLOAT32), full = length / lanes * lanes;
        float maximum = Float.NEGATIVE_INFINITY;
        for (int base = 0; base < full; base += lanes) {
            FloatVector block = loadFloatBlock(input, base).add(0.0f);
            for (int lane = 0; lane < lanes; lane++) maximum = Math.max(maximum, block.lane(lane));
        }
        for (int i = full; i < length; i++) maximum = Math.max(maximum, (float) readScalar(Type.FLOAT32, input, i));
        float sum = 0.0f, compensation = 0.0f;
        for (int base = 0; base < full; base += lanes) {
            FloatVector block = loadFloatBlock(input, base).add(0.0f);
            for (int lane = 0; lane < lanes; lane++) {
                float addend = (float) Math.exp(block.lane(lane) - maximum) - compensation;
                float temporary = sum + addend;
                compensation = (temporary - sum) - addend;
                sum = temporary;
            }
        }
        for (int i = full; i < length; i++) {
            float addend = (float) Math.exp((float) readScalar(Type.FLOAT32, input, i) - maximum) - compensation;
            float temporary = sum + addend;
            compensation = (temporary - sum) - addend;
            sum = temporary;
        }
        double[] result = new double[length];
        for (int base = 0; base < full; base += lanes) {
            FloatVector block = loadFloatBlock(input, base).add(0.0f);
            for (int lane = 0; lane < lanes; lane++) result[base + lane] = (float) Math.exp(
                    block.lane(lane) - maximum) / sum;
        }
        for (int i = full; i < length; i++) result[i] = (float) Math.exp(
                (float) readScalar(Type.FLOAT32, input, i) - maximum) / sum;
        return result;
    }

    private static FloatVector loadFloatBlock(Object input, int base) {
        return input instanceof float[] array ? FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, array, base)
                : FloatVector.fromMemorySegment(FloatVector.SPECIES_PREFERRED, (MemorySegment) input,
                        (long) base * Float.BYTES, ByteOrder.nativeOrder());
    }

    private static DoubleVector loadDoubleBlock(Object input, int base) {
        return input instanceof double[] array ? DoubleVector.fromArray(DoubleVector.SPECIES_PREFERRED, array, base)
                : DoubleVector.fromMemorySegment(DoubleVector.SPECIES_PREFERRED, (MemorySegment) input,
                        (long) base * Double.BYTES, ByteOrder.nativeOrder());
    }

    /** Reads a lane from a full vector block or the scalar tail without changing logical order. */
    private static double readVectorOrdered(Type type, Object input, int index, int length) {
        int lanes = lanes(type);
        int base = index / lanes * lanes;
        if (base + lanes <= length) {
            if (type == Type.FLOAT32) {
                FloatVector vector = input instanceof float[] array
                        ? FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, array, base)
                        : FloatVector.fromMemorySegment(FloatVector.SPECIES_PREFERRED,
                                (MemorySegment) input, (long) base * Float.BYTES,
                                ByteOrder.nativeOrder());
                return vector.lane(index - base);
            }
            DoubleVector vector = input instanceof double[] array
                    ? DoubleVector.fromArray(DoubleVector.SPECIES_PREFERRED, array, base)
                    : DoubleVector.fromMemorySegment(DoubleVector.SPECIES_PREFERRED,
                            (MemorySegment) input, (long) base * Double.BYTES,
                            ByteOrder.nativeOrder());
            return vector.lane(index - base);
        }
        return readScalar(type, input, index);
    }

    /** Returns one scalar tail element in the represented arithmetic format. */
    private static double readScalar(Type type, Object input, int index) {
        if (type == Type.FLOAT32) return input instanceof float[] array ? array[index]
                : ((MemorySegment) input).get(ValueLayout.JAVA_FLOAT_UNALIGNED
                        .withOrder(ByteOrder.nativeOrder()), (long) index * Float.BYTES);
        return input instanceof double[] array ? array[index] : ((MemorySegment) input).get(
                ValueLayout.JAVA_DOUBLE_UNALIGNED.withOrder(ByteOrder.nativeOrder()),
                (long) index * Double.BYTES);
    }

    /** Narrows exactly where the FLOAT32 scalar CPU algorithm stores an arithmetic intermediate. */
    private static double narrow(Type type, double value) {
        return type == Type.FLOAT32 ? (float) value : value;
    }

    /** Compares represented values without inventing a finite tolerance. */
    private static void assertRawEquals(Type type, double[] expected, double[] actual, String name) {
        assertEquals(expected.length, actual.length, name);
        for (int i = 0; i < expected.length; i++) {
            if (type == Type.FLOAT32) assertEquals(Float.floatToRawIntBits((float) expected[i]),
                    Float.floatToRawIntBits((float) actual[i]), name + "/" + i);
            else assertEquals(Double.doubleToRawLongBits(expected[i]),
                    Double.doubleToRawLongBits(actual[i]), name + "/" + i);
        }
    }

    private static Object carrierInput(Type type, Carrier carrier, double[] values, Arena arena) {
        boolean segment = carrier == Carrier.SEGMENT_SEGMENT || carrier == Carrier.SEGMENT_ARRAY;
        if (type == Type.FLOAT32) {
            float[] array = new float[values.length];
            for (int i = 0; i < array.length; i++) array[i] = (float) values[i];
            if (!segment) return array;
            MemorySegment segmentValue = arena.allocate((long) array.length * Float.BYTES, Float.BYTES);
            for (int i = 0; i < array.length; i++) segmentValue.set(ValueLayout.JAVA_FLOAT_UNALIGNED
                    .withOrder(ByteOrder.nativeOrder()), (long) i * Float.BYTES, array[i]);
            return segmentValue;
        }
        double[] array = values.clone();
        if (!segment) return array;
        MemorySegment segmentValue = arena.allocate((long) array.length * Double.BYTES, Double.BYTES);
        for (int i = 0; i < array.length; i++) segmentValue.set(ValueLayout.JAVA_DOUBLE_UNALIGNED
                .withOrder(ByteOrder.nativeOrder()), (long) i * Double.BYTES, array[i]);
        return segmentValue;
    }

    private static void assertRawBlock(Type type, Object input, int offset, double[] expected) {
        if (type == Type.FLOAT32) {
            FloatVector vector = input instanceof float[] array
                    ? FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, array, offset)
                    : FloatVector.fromMemorySegment(FloatVector.SPECIES_PREFERRED,
                            (MemorySegment) input, (long) offset * Float.BYTES, ByteOrder.nativeOrder());
            for (int lane = 0; lane < vector.length(); lane++) assertEquals(
                    Float.floatToRawIntBits((float) expected[offset + lane]),
                    Float.floatToRawIntBits(vector.lane(lane)));
        } else {
            DoubleVector vector = input instanceof double[] array
                    ? DoubleVector.fromArray(DoubleVector.SPECIES_PREFERRED, array, offset)
                    : DoubleVector.fromMemorySegment(DoubleVector.SPECIES_PREFERRED,
                            (MemorySegment) input, (long) offset * Double.BYTES, ByteOrder.nativeOrder());
            for (int lane = 0; lane < vector.length(); lane++) assertEquals(
                    Double.doubleToRawLongBits(expected[offset + lane]),
                    Double.doubleToRawLongBits(vector.lane(lane)));
        }
    }

    private static void assertScalarTail(Type type, Object input, int offset, int count,
            double[] expected) {
        for (int index = offset; index < offset + count; index++) {
            if (type == Type.FLOAT32) {
                float value = input instanceof float[] array ? array[index] : ((MemorySegment) input).get(
                        ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.nativeOrder()),
                        (long) index * Float.BYTES);
                assertEquals(Float.floatToRawIntBits((float) expected[index]), Float.floatToRawIntBits(value));
            } else {
                double value = input instanceof double[] array ? array[index] : ((MemorySegment) input).get(
                        ValueLayout.JAVA_DOUBLE_UNALIGNED.withOrder(ByteOrder.nativeOrder()),
                        (long) index * Double.BYTES);
                assertEquals(Double.doubleToRawLongBits(expected[index]), Double.doubleToRawLongBits(value));
            }
        }
    }
}
