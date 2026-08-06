package io.github.pho001.synaptik.backend.cpu.internal.ir;

import io.github.pho001.synaptik.model.datatype.DataType;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;
import java.util.Objects;

/**
 * Immutable route-independent canonical CPU kernel intermediate representation (IR).
 * Values use topology-local ordinals, so graph identities, extents, slots, routes, generator
 * versions, segment instances, and invocation bindings cannot enter canonical identity.
 * When analysis selects one contiguous input copy, it derives a second canonical consumer form
 * whose copied input has dense access; that structural change belongs to the lowering fingerprint,
 * while the source binding, workspace, costs, and concrete geometry remain outside this IR.
 *
 * @param values non-null dense ordered typed values; copied defensively
 * @param instructions non-null ordered exact computations; copied defensively
 * @param loop non-null universal primitive loop model
 * @param stores non-null ordered boundary stores; copied defensively
 */
public record CpuKernelIr(
        List<Value> values, List<Instruction> instructions, Loop loop, List<Store> stores) {
    /**
     * A typed topology-local boundary or virtual value.
     *
     * @param ordinal non-negative dense topology-local ordinal
     * @param dataType non-null logical data type
     * @param kind non-null materialization role
     * @param accessPlan non-null route-independent access form
     */
    public record Value(int ordinal, DataType dataType, Kind kind, CpuAccessPlan accessPlan) {
        /** Value materialization role. */
        public enum Kind {
            /** Materialized partition input. */ INPUT,
            /** Unit-private value with no physical declaration. */ VIRTUAL,
            /** Materialized partition output. */ OUTPUT
        }
        /**
         * Validates one topology-local value.
         *
         * @throws NullPointerException if a reference component is {@code null}
         * @throws IllegalArgumentException if {@code ordinal} is negative
         */
        public Value {
            if (ordinal < 0) throw new IllegalArgumentException("ordinal must be non-negative");
            Objects.requireNonNull(dataType, "dataType");
            Objects.requireNonNull(kind, "kind");
            Objects.requireNonNull(accessPlan, "accessPlan");
        }
    }

    /**
     * One exact ordered scalar computation.
     *
     * @param semantic non-null exact operation semantic
     * @param inputs non-null ordered input ordinals; copied defensively
     * @param output non-negative output ordinal
     */
    public record Instruction(Semantic semantic, List<Integer> inputs, int output) {
        /** Exact semantics implemented by task 0005A. */
        public enum Semantic {
            /** Fixed-order binary addition. */ ADD,
            /** Exact Model Gaussian error linear unit target. */ GELU_EXACT,
            /** Fixed-order binary multiplication. */ MUL
        }
        /**
         * Validates topology-local operands.
         *
         * @throws NullPointerException if {@code semantic}, {@code inputs}, or an input is null
         * @throws IllegalArgumentException if an ordinal is negative or the semantic's arity is
         *     not exact
         */
        public Instruction {
            Objects.requireNonNull(semantic, "semantic");
            Objects.requireNonNull(inputs, "inputs");
            inputs = List.copyOf(inputs);
            if (inputs.stream().anyMatch(i -> i == null || i < 0) || output < 0) {
                throw new IllegalArgumentException("instruction ordinals must be non-negative");
            }
            int expected = semantic == Semantic.GELU_EXACT ? 1 : 2;
            if (inputs.size() != expected) throw new IllegalArgumentException(
                    "instruction input count does not match semantic");
        }
    }

    /**
     * Universal primitive half-open loop model.
     *
     * @param startParameter exact stable name {@code start}
     * @param endParameter exact stable name {@code end}
     */
    public record Loop(String startParameter, String endParameter) {
        /**
         * Validates stable primitive-bound names.
         *
         * @throws IllegalArgumentException if the names are not exactly {@code start} and
         *     {@code end}, including when either name is {@code null}
         */
        public Loop {
            if (!"start".equals(startParameter) || !"end".equals(endParameter)) {
                throw new IllegalArgumentException("loop parameters must be start and end");
            }
        }
    }

    /**
     * Ordered write of a computed value to a boundary output.
     *
     * @param value non-negative computed-value ordinal
     * @param outputOrdinal non-negative boundary-output ordinal
     */
    public record Store(int value, int outputOrdinal) {
        /**
         * Validates non-negative topology-local ordinals.
         *
         * @throws IllegalArgumentException if either ordinal is negative
         */
        public Store {
            if (value < 0 || outputOrdinal < 0) throw new IllegalArgumentException(
                    "store ordinals must be non-negative");
        }
    }

    /**
     * Validates and snapshots one canonical IR.
     *
     * @throws NullPointerException if a collection, element, or loop is {@code null}
     * @throws IllegalArgumentException if value ordinals are not dense and ordered
     */
    public CpuKernelIr {
        values = copy(values, "values");
        instructions = copy(instructions, "instructions");
        Objects.requireNonNull(loop, "loop");
        stores = copy(stores, "stores");
        for (int i = 0; i < values.size(); i++) if (values.get(i).ordinal() != i) {
            throw new IllegalArgumentException("value ordinals must be dense and ordered");
        }
    }

    /**
     * Returns a deterministic structural key that excludes all instance facts.
     *
     * @return a lowercase hexadecimal SHA-256 structural key; never {@code null}
     */
    public String structuralKey() {
        StringBuilder text = new StringBuilder("cpu-ir-v2|");
        values.forEach(v -> text.append(v.ordinal()).append(':').append(v.dataType())
                .append(':').append(v.kind()).append(':').append(v.accessPlan().accessKind())
                .append(':').append(v.accessPlan().regime()).append(':')
                .append(v.accessPlan().iterationRank()).append(':')
                .append(v.accessPlan().axisRoles()).append(':')
                .append(v.accessPlan().contiguousSuffix()).append('|'));
        instructions.forEach(i -> text.append(i.semantic()).append(':').append(i.inputs())
                .append('>').append(i.output()).append('|'));
        stores.forEach(s -> text.append("store:").append(s.value()).append('>')
                .append(s.outputOrdinal()).append('|'));
        try {
            return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256")
                    .digest(text.toString().getBytes(StandardCharsets.US_ASCII)));
        } catch (NoSuchAlgorithmException impossible) { throw new AssertionError(impossible); }
    }

    private static <T> List<T> copy(List<T> values, String name) {
        Objects.requireNonNull(values, name);
        var copy = new ArrayList<T>(values.size());
        for (int i = 0; i < values.size(); i++) copy.add(Objects.requireNonNull(
                values.get(i), name + "[" + i + "]"));
        return List.copyOf(copy);
    }
}
