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
 * The instruction vocabulary includes the closed nineteen-kind unary matrix as distinct typed
 * opcodes; each unary Model occurrence remains one instruction with one same-typed FLOAT32 or
 * FLOAT64 result. Canonical BOOL values retain their materialized-boundary or unit-private virtual
 * role so generated code may represent eligible virtual floating masks without changing storage.
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
     * One exact ordered pointwise computation over topology-local values.
     *
     * @param opcode non-null family-oriented exact operation semantic
     * @param inputs non-null ordered input ordinals; copied defensively
     * @param output non-negative output ordinal
     * @param scalarImmediate exact typed primitive bits when required by {@code opcode}, otherwise
     *     {@code null}
     * @param powerRealization selected scalar-power realization for {@code SCALAR_POW}, otherwise
     *     {@code null}
     * @param clampImmediate exact ordered bounds for {@code SCALAR_CLAMP}, otherwise {@code null}
     */
    public record Instruction(CpuPointwiseOpcode opcode, List<Integer> inputs, int output,
            ScalarImmediate scalarImmediate, PowerRealization powerRealization,
            ClampImmediate clampImmediate) {
        /**
         * Validates topology-local operands.
         *
         * @throws NullPointerException if {@code opcode}, {@code inputs}, or an input is null
         * @throws IllegalArgumentException if an ordinal is negative, the opcode arity is not
         *     exact, or immediate presence disagrees with the opcode
         */
        public Instruction {
            Objects.requireNonNull(opcode, "opcode");
            Objects.requireNonNull(inputs, "inputs");
            inputs = List.copyOf(inputs);
            if (inputs.stream().anyMatch(i -> i == null || i < 0) || output < 0) {
                throw new IllegalArgumentException("instruction ordinals must be non-negative");
            }
            int expected = opcode.arity();
            if (inputs.size() != expected) throw new IllegalArgumentException(
                    "instruction input count does not match opcode");
            if (opcode.carriesScalarImmediate() != (scalarImmediate != null)) {
                throw new IllegalArgumentException("instruction scalar immediate does not match opcode");
            }
            if ((opcode == CpuPointwiseOpcode.SCALAR_POW) != (powerRealization != null)) {
                throw new IllegalArgumentException("instruction power realization does not match opcode");
            }
            if ((opcode == CpuPointwiseOpcode.SCALAR_CLAMP) != (clampImmediate != null)) {
                throw new IllegalArgumentException("instruction clamp immediate does not match opcode");
            }
        }

        /**
         * Creates an instruction with the existing scalar-power shape.
         * @param opcode non-null opcode
         * @param inputs non-null ordered input ordinals
         * @param output non-negative output ordinal
         * @param scalarImmediate exact scalar bits when required
         * @param powerRealization selected realization for scalar power
         * @throws NullPointerException if {@code opcode}, {@code inputs}, or an input is null
         * @throws IllegalArgumentException if ordinals, arity, immediate, or realization rules
         *     disagree
         */
        public Instruction(CpuPointwiseOpcode opcode, List<Integer> inputs, int output,
                ScalarImmediate scalarImmediate, PowerRealization powerRealization) {
            this(opcode, inputs, output, scalarImmediate, powerRealization, null);
        }

        /**
         * Creates one first-class range-clamp instruction.
         * @param opcode non-null {@code SCALAR_CLAMP} opcode
         * @param inputs non-null one-element input-ordinal list
         * @param output non-negative output ordinal
         * @param clampImmediate non-null exact ordered clamp bounds
         * @throws NullPointerException if {@code opcode}, {@code inputs}, an input, or
         *     {@code clampImmediate} is null
         * @throws IllegalArgumentException if ordinals, arity, or clamp-immediate rules disagree
         */
        public Instruction(CpuPointwiseOpcode opcode, List<Integer> inputs, int output,
                ClampImmediate clampImmediate) {
            this(opcode, inputs, output, null, null, clampImmediate);
        }

        /**
         * Creates an instruction without a scalar-power realization.
         *
         * @param opcode non-null opcode other than {@code SCALAR_POW}
         * @param inputs non-null ordered topology-local input ordinals; copied defensively
         * @param output non-negative topology-local output ordinal
         * @param scalarImmediate exact typed immediate when required, otherwise {@code null}
         * @throws NullPointerException if {@code opcode}, {@code inputs}, or an input is null
         * @throws IllegalArgumentException if ordinals, arity, immediate, or realization rules
         *     disagree
         */
        public Instruction(CpuPointwiseOpcode opcode, List<Integer> inputs, int output,
                ScalarImmediate scalarImmediate) {
            this(opcode, inputs, output, scalarImmediate, null, null);
        }

        /**
         * Creates a parameterless instruction.
         *
         * @param opcode non-null opcode that must not carry a scalar immediate
         * @param inputs non-null ordered topology-local input ordinals; copied defensively
         * @param output non-negative topology-local output ordinal
         * @throws NullPointerException if {@code opcode}, {@code inputs}, or an input is null
         * @throws IllegalArgumentException if ordinals, arity, or immediate requirements disagree
         */
        public Instruction(CpuPointwiseOpcode opcode, List<Integer> inputs, int output) {
            this(opcode, inputs, output, null, null, null);
        }
    }

    /** Closed exact/default realization selected for one semantic scalar-power instruction. */
    public enum PowerRealization {
        /** Invoke the direct typed power realization. */ DIRECT,
        /** Produce exact positive typed one without reading the base. */ POSITIVE_ONE,
        /** Forward the represented base for exact positive-one exponent. */ IDENTITY,
        /** Multiply the represented base by itself once in the result type. */ SQUARE,
        /** Divide exact positive typed one by the represented base once. */ RECIPROCAL
    }

    /**
     * Exact typed primitive bits retained for one scalar arithmetic instruction.
     *
     * @param dataType non-null exact scalar type; lowering admits FLOAT64, FLOAT32, INT32, or INT64
     * @param bits raw primitive bits in the low width of the selected type
     */
    public record ScalarImmediate(DataType dataType, long bits) {
        /**
         * Retains one non-null exact scalar type and its unmodified primitive bits.
         *
         * @param dataType non-null exact scalar type
         * @param bits raw primitive bits copied from the Model scalar value
         * @throws NullPointerException if {@code dataType} is {@code null}
         */
        public ScalarImmediate { Objects.requireNonNull(dataType, "dataType"); }
    }

    /**
     * Exact ordered typed primitive bounds retained by one first-class clamp instruction.
     *
     * @param lower non-null exact lower bound
     * @param upper non-null exact upper bound of the same FLOAT32 or FLOAT64 type
     */
    public record ClampImmediate(ScalarImmediate lower, ScalarImmediate upper) {
        /**
         * Validates two same-typed supported floating bounds.
         * @param lower non-null exact lower bound
         * @param upper non-null exact upper bound
         * @throws NullPointerException if either bound is {@code null}
         * @throws IllegalArgumentException if their types differ or are not FLOAT32/FLOAT64
         */
        public ClampImmediate {
            Objects.requireNonNull(lower, "lower");
            Objects.requireNonNull(upper, "upper");
            if (lower.dataType() != upper.dataType()
                    || lower.dataType() != DataType.FLOAT32 && lower.dataType() != DataType.FLOAT64) {
                throw new IllegalArgumentException("clamp bounds must have one floating data type");
            }
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
     * @throws IllegalArgumentException if value ordinals are not dense and ordered, instruction
     *     ordinals/types/immediates are inconsistent, or a store does not name an output value
     */
    public CpuKernelIr {
        values = copy(values, "values");
        instructions = copy(instructions, "instructions");
        Objects.requireNonNull(loop, "loop");
        stores = copy(stores, "stores");
        for (int i = 0; i < values.size(); i++) if (values.get(i).ordinal() != i) {
            throw new IllegalArgumentException("value ordinals must be dense and ordered");
        }
        List<Value> checkedValues = values;
        var produced = new java.util.HashSet<Integer>();
        for (Instruction instruction : instructions) {
            if (instruction.output() >= checkedValues.size() || instruction.inputs().stream()
                    .anyMatch(input -> input >= checkedValues.size()) || !produced.add(instruction.output())) {
                throw new IllegalArgumentException("instruction ordinals must reference unique IR values");
            }
            DataType outputType = checkedValues.get(instruction.output()).dataType();
            List<DataType> inputTypes = instruction.inputs().stream()
                    .map(input -> checkedValues.get(input).dataType()).toList();
            validateTypes(instruction, inputTypes, outputType);
        }
        for (Store store : stores) {
            if (store.value() >= values.size()
                    || values.get(store.value()).kind() != Value.Kind.OUTPUT) {
                throw new IllegalArgumentException("store must reference a materialized output value");
            }
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
        instructions.forEach(i -> text.append(i.opcode()).append(':').append(i.inputs())
                .append('>').append(i.output()).append(':').append(i.scalarImmediate()).append(':')
                .append(i.powerRealization()).append(':').append(i.clampImmediate()).append('|'));
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

    private static void validateTypes(Instruction instruction, List<DataType> inputs,
            DataType output) {
        CpuPointwiseOpcode opcode = instruction.opcode();
        boolean same = inputs.stream().allMatch(inputs.getFirst()::equals);
        boolean numeric = inputs.stream().allMatch(type -> type == DataType.FLOAT64
                || type == DataType.FLOAT32 || type == DataType.INT32 || type == DataType.INT64);
        boolean valid = switch (opcode.family()) {
            case BINARY_ARITHMETIC -> same && numeric && output == inputs.getFirst()
                    && (opcode != CpuPointwiseOpcode.DIV && opcode != CpuPointwiseOpcode.POW
                        || output == DataType.FLOAT64 || output == DataType.FLOAT32);
            case SCALAR_ARITHMETIC -> numeric && output == inputs.getFirst()
                    && instruction.scalarImmediate().dataType() == output
                    && ((opcode != CpuPointwiseOpcode.SCALAR_DIV
                            && opcode != CpuPointwiseOpcode.SCALAR_POW)
                        || output == DataType.FLOAT64 || output == DataType.FLOAT32)
                    && (opcode != CpuPointwiseOpcode.SCALAR_POW
                        || powerRealizationMatches(instruction));
            case SCALAR_RANGE -> output == inputs.getFirst()
                    && (output == DataType.FLOAT64 || output == DataType.FLOAT32)
                    && instruction.clampImmediate().lower().dataType() == output;
            case UNARY -> output == inputs.getFirst()
                    && (output == DataType.FLOAT64 || output == DataType.FLOAT32);
            case CLASSIFICATION -> (inputs.getFirst() == DataType.FLOAT64
                    || inputs.getFirst() == DataType.FLOAT32) && output == DataType.BOOL;
            case COMPARISON -> same && numeric && output == DataType.BOOL;
            case LOGICAL -> inputs.stream().allMatch(type -> type == DataType.BOOL)
                    && output == DataType.BOOL;
            case SELECTION -> inputs.get(0) == DataType.BOOL
                    && inputs.get(1) == inputs.get(2) && output == inputs.get(1)
                    && (output == DataType.FLOAT64 || output == DataType.FLOAT32);
            case CAST -> inputs.getFirst() == output && (output == DataType.FLOAT64
                    || output == DataType.FLOAT32 || output == DataType.INT32
                    || output == DataType.INT64 || output == DataType.BOOL);
        };
        if (!valid) throw new IllegalArgumentException("instruction data types do not match opcode");
    }

    private static boolean powerRealizationMatches(Instruction instruction) {
        ScalarImmediate immediate = instruction.scalarImmediate();
        long bits = immediate.bits();
        PowerRealization expected;
        if (immediate.dataType() == DataType.FLOAT32) {
            bits &= 0xffff_ffffL;
            expected = bits == 0L || bits == 0x8000_0000L ? PowerRealization.POSITIVE_ONE
                    : bits == 0x3f80_0000L ? PowerRealization.IDENTITY
                    : bits == 0x4000_0000L ? PowerRealization.SQUARE
                    : bits == 0xbf80_0000L ? PowerRealization.RECIPROCAL
                    : PowerRealization.DIRECT;
        } else if (immediate.dataType() == DataType.FLOAT64) {
            expected = bits == 0L || bits == 0x8000_0000_0000_0000L
                    ? PowerRealization.POSITIVE_ONE
                    : bits == 0x3ff0_0000_0000_0000L ? PowerRealization.IDENTITY
                    : bits == 0x4000_0000_0000_0000L ? PowerRealization.SQUARE
                    : bits == 0xbff0_0000_0000_0000L ? PowerRealization.RECIPROCAL
                    : PowerRealization.DIRECT;
        } else return false;
        return expected == instruction.powerRealization();
    }
}
