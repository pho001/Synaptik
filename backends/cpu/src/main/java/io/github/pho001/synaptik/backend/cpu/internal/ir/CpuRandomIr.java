package io.github.pho001.synaptik.backend.cpu.internal.ir;

import io.github.pho001.synaptik.model.datatype.DataType;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * CPU-private structural identity for explicit graph RNG initialization and dropout.
 *
 * <p>The identity fixes the exact counter algorithm, raw initializer words or dropout
 * probability bits, represented value type, ordered boundary roles, and finite-precision policy.
 * Concrete extents, layout magnitudes, carriers, addresses, ranges, and state values consumed by
 * dropout remain cold invocation facts.</p>
 *
 * @param family non-null initializer or dropout family
 * @param valueType FLOAT64/FLOAT32 dropout value type, or INT64 for initialization
 * @param keyBits raw initializer key bits; ignored for dropout
 * @param counterBits raw initializer counter bits; ignored for dropout
 * @param probabilityBits raw validated binary64 probability bits; zero for initialization
 * @param boundaryAccess exact ordered boundary access roles; copied defensively
 */
public record CpuRandomIr(Family family, DataType valueType, long keyBits, long counterBits,
        long probabilityBits, List<CpuAccessPlan> boundaryAccess) implements CpuPortableKernelIr {
    /** Exact supported operation families. */
    public enum Family { /** Writes a raw two-word state. */ INITIAL_STATE,
        /** Consumes value/state and writes value/mask/next-state. */ DROPOUT }

    /** Exact CPU-private counter-to-word algorithm identity. */
    public static final String GENERATOR_ID = "SYNAPTIK_CPU_SPLITMIX64_COUNTER_V1";
    /** Exact V1 key-translation constant. */
    public static final long KEY_BIAS = 0x9e3779b97f4a7c15L;
    /** Exact V1 first mixing multiplier. */
    public static final long MIX_MULTIPLIER_1 = 0xbf58476d1ce4e5b9L;
    /** Exact V1 second mixing multiplier. */
    public static final long MIX_MULTIPLIER_2 = 0x94d049bb133111ebL;
    /** Exact V1 counter and logical-ordinal mapping identity. */
    public static final String COUNTER_MAPPING_ID =
            "MIX64_COUNTER_PLUS_LOGICAL_PLUS_MIX64_KEY_PLUS_BIAS_V1";
    /** Exact binary64 uniform conversion policy. */
    public static final String UNIFORM_ID = "UNIFORM53_TOP53_BINARY64_V1";
    /** Exact keep-threshold comparison policy. */
    public static final String THRESHOLD_ID =
            "KEEP_UNIFORM_GREATER_THAN_OR_EQUAL_BINARY64_PROBABILITY_V1";
    /** Exact dropout arithmetic and canonical-mask policy. */
    public static final String NUMERIC_ID = "DROPOUT_BINARY64_DIVIDE_NARROW_ONCE_V1";
    /** Exact represented keep-mask policy. */
    public static final String MASK_POLICY_ID = "CANONICAL_BOOL_BYTES_0_1_V1";
    /** Exact prologue and modulo state-advancement policy. */
    public static final String STATE_POLICY_ID =
            "PROLOGUE_ONCE_KEY_UNCHANGED_COUNTER_PLUS_N_MODULO_2_64_V1";

    /**
     * Validates and snapshots one exact random-family structural form.
     *
     * @param family non-null initializer or dropout family
     * @param valueType FLOAT64/FLOAT32 dropout value type, or INT64 for initialization
     * @param keyBits raw initializer key bits; ignored for dropout
     * @param counterBits raw initializer counter bits; ignored for dropout
     * @param probabilityBits raw validated binary64 probability bits; zero for initialization
     * @param boundaryAccess non-null exact ordered boundary access roles; copied defensively
     * @throws NullPointerException if {@code family}, {@code valueType},
     *     {@code boundaryAccess}, or one of its elements is null
     * @throws IllegalArgumentException if the family, type, probability, boundary count, or access
     *     roles are inconsistent
     */
    public CpuRandomIr {
        Objects.requireNonNull(family, "family");
        Objects.requireNonNull(valueType, "valueType");
        boundaryAccess = List.copyOf(boundaryAccess);
        if (family == Family.INITIAL_STATE) {
            if (valueType != DataType.INT64 || probabilityBits != 0 || boundaryAccess.size() != 1
                    || boundaryAccess.getFirst().accessKind() != CpuAccessPlan.AccessKind.WRITE) {
                throw new IllegalArgumentException("initial-state structural facts disagree");
            }
        } else if ((valueType != DataType.FLOAT64 && valueType != DataType.FLOAT32)
                || boundaryAccess.size() != 5
                || boundaryAccess.get(0).accessKind() != CpuAccessPlan.AccessKind.READ
                || boundaryAccess.get(1).accessKind() != CpuAccessPlan.AccessKind.READ
                || boundaryAccess.subList(2, 5).stream()
                    .anyMatch(plan -> plan.accessKind() != CpuAccessPlan.AccessKind.WRITE)) {
            throw new IllegalArgumentException("dropout structural facts disagree");
        }
    }

    /**
     * Returns an instruction-free generated-kernel encoding.
     *
     * @return fresh immutable encoding with exact boundary types and structural identity
     */
    public CpuKernelIr encodedKernelIr() {
        var values = new ArrayList<CpuKernelIr.Value>();
        if (family == Family.INITIAL_STATE) {
            values.add(new CpuKernelIr.Value(0, DataType.INT64, CpuKernelIr.Value.Kind.OUTPUT,
                    boundaryAccess.getFirst()));
        } else {
            values.add(new CpuKernelIr.Value(0, valueType, CpuKernelIr.Value.Kind.INPUT,
                    boundaryAccess.get(0)));
            values.add(new CpuKernelIr.Value(1, DataType.INT64, CpuKernelIr.Value.Kind.INPUT,
                    boundaryAccess.get(1)));
            values.add(new CpuKernelIr.Value(2, valueType, CpuKernelIr.Value.Kind.OUTPUT,
                    boundaryAccess.get(2)));
            values.add(new CpuKernelIr.Value(3, DataType.BOOL, CpuKernelIr.Value.Kind.OUTPUT,
                    boundaryAccess.get(3)));
            values.add(new CpuKernelIr.Value(4, DataType.INT64, CpuKernelIr.Value.Kind.OUTPUT,
                    boundaryAccess.get(4)));
        }
        String identity = "random:" + family + ":generator=" + GENERATOR_ID
                + ":keyBias=" + Long.toUnsignedString(KEY_BIAS, 16)
                + ":mixMultiplier1=" + Long.toUnsignedString(MIX_MULTIPLIER_1, 16)
                + ":mixMultiplier2=" + Long.toUnsignedString(MIX_MULTIPLIER_2, 16)
                + ":mapping=" + COUNTER_MAPPING_ID + ":uniform=" + UNIFORM_ID
                + ":threshold=" + THRESHOLD_ID + ":numeric=" + NUMERIC_ID
                + ":mask=" + MASK_POLICY_ID + ":state=" + STATE_POLICY_ID
                + ":type=" + valueType + ":key=" + Long.toUnsignedString(keyBits, 16)
                + ":counter=" + Long.toUnsignedString(counterBits, 16)
                + ":probability=" + Long.toUnsignedString(probabilityBits, 16);
        int marker = family == Family.INITIAL_STATE ? 0 : 2;
        return new CpuKernelIr(values, List.of(), new CpuKernelIr.Loop("start", "end"),
                List.of(new CpuKernelIr.Store(marker, 0)), identity);
    }

    /** @return deterministic hexadecimal structural key excluding cold geometry */
    @Override public String structuralKey() { return encodedKernelIr().structuralKey(); }
}
