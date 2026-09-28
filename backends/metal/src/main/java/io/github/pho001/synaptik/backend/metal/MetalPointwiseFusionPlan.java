package io.github.pho001.synaptik.backend.metal;

import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.List;
import java.util.Objects;

/**
 * Immutable schema-17 execution extension for one {@code CUSTOM_PROGRAM} image.
 *
 * <p>The typed records are authoritative. The canonical manifest is a redundant cross-language
 * plan-agreement certificate. Generated Metal source bytes and hashes remain exclusively
 * native-owned; Java carries only the frozen integer byte-count table required for bounded
 * planning.</p>
 */
final class MetalPointwiseFusionPlan {
    static final int GENERATOR_SCHEMA = 1;
    static final int SOURCE_SIZE_TABLE_VERSION = 1;
    static final int MAX_GENERATED_UNITS = 32;
    static final int MAX_GENERATED_INSTRUCTIONS = 256;
    static final int MAX_FUNCTION_SOURCE_UTF8_BYTES = 16_384;
    static final int MAX_GENERATED_SOURCE_UTF8_BYTES = 262_144;
    static final int MAX_TOTAL_SOURCE_UTF8_BYTES = 1_048_576;
    static final int FIXED_CORPUS_UTF8_BYTES = 77_444;
    static final int NO_POSITION = -1;

    enum StepKind {
        FIXED_CUSTOM(1), MPSGRAPH_BOUNDARY(2), GENERATED_POINTWISE(3);

        private final int wire;

        StepKind(int wire) {
            this.wire = wire;
        }

        int wire() {
            return wire;
        }
    }

    enum Access {
        READ(1), WRITE(2);

        private final int wire;

        Access(int wire) {
            this.wire = wire;
        }

        int wire() {
            return wire;
        }
    }

    enum Opcode {
        FLOOR(1, 10), CEIL(2, 9), SIGN(3, 9), RELU(4, 9);

        private final int wire;
        private final int helperUtf8Bytes;

        Opcode(int wire, int helperUtf8Bytes) {
            this.wire = wire;
            this.helperUtf8Bytes = helperUtf8Bytes;
        }

        int wire() {
            return wire;
        }

        int helperUtf8Bytes() {
            return helperUtf8Bytes;
        }

        static Opcode from(MetalMpsGraphProgram.NodeKind kind) {
            return switch (Objects.requireNonNull(kind, "kind")) {
                case FLOOR -> FLOOR;
                case CEIL -> CEIL;
                case SIGN -> SIGN;
                case RELU -> RELU;
                default -> throw new IllegalArgumentException("node is not a schema-1 opcode");
            };
        }
    }

    enum CapReason {
        NONE(0), UNIT_COUNT(1), INSTRUCTION_COUNT(2), FUNCTION_UTF8_BYTES(3),
        GENERATED_UTF8_BYTES(4), TOTAL_UTF8_BYTES(5);

        private final int wire;

        CapReason(int wire) {
            this.wire = wire;
        }

        int wire() {
            return wire;
        }
    }

    record Step(
            StepKind kind,
            int memberStart,
            int memberCount,
            int bindingStart,
            int bindingCount,
            int instructionStart,
            int instructionCount,
            int expectedFunctionUtf8Bytes) {
        Step {
            Objects.requireNonNull(kind, "kind");
            if (memberStart < 0 || memberCount <= 0 || bindingStart < 0 || bindingCount <= 0
                    || instructionStart < 0 || instructionCount < 0
                    || expectedFunctionUtf8Bytes < 0
                    || (kind == StepKind.GENERATED_POINTWISE
                            ? instructionCount != memberCount || expectedFunctionUtf8Bytes == 0
                            : instructionCount != 0 || expectedFunctionUtf8Bytes != 0)) {
                throw new IllegalArgumentException("malformed Metal fusion step");
            }
        }
    }

    record Binding(
            int stepOrdinal,
            int argumentOrdinal,
            Access access,
            int materializedSlot,
            int programValueIndex) {
        Binding {
            Objects.requireNonNull(access, "access");
            if (stepOrdinal < 0 || argumentOrdinal < 0 || materializedSlot < 0
                    || programValueIndex < 0) {
                throw new IllegalArgumentException("malformed Metal fusion binding");
            }
        }
    }

    record Instruction(
            int stepOrdinal,
            int relativeNodePosition,
            Opcode opcode,
            int inputSsa,
            int outputSsa) {
        Instruction {
            Objects.requireNonNull(opcode, "opcode");
            if (stepOrdinal < 0 || relativeNodePosition < 0 || inputSsa < 0
                    || outputSsa != inputSsa + 1) {
                throw new IllegalArgumentException("malformed Metal fusion instruction");
            }
        }
    }

    private final List<Step> steps;
    private final int[] memberNodePositions;
    private final List<Binding> bindings;
    private final int[] materializedProgramValueIndices;
    private final int[] programToMaterializedSlot;
    private final int[] targetMaterializedSlots;
    private final List<Instruction> instructions;
    private final byte[] canonicalManifest;
    private final byte[] canonicalManifestDigest;
    private final int generatedUnitCount;
    private final int expectedGeneratedSourceUtf8Bytes;
    private final int expectedTotalSourceUtf8Bytes;
    private final int firstRejectedNodePosition;
    private final CapReason capReason;

    MetalPointwiseFusionPlan(
            List<Step> steps,
            int[] memberNodePositions,
            List<Binding> bindings,
            int[] materializedProgramValueIndices,
            int[] programToMaterializedSlot,
            int[] targetMaterializedSlots,
            List<Instruction> instructions,
            byte[] canonicalManifest,
            int generatedUnitCount,
            int expectedGeneratedSourceUtf8Bytes,
            int firstRejectedNodePosition,
            CapReason capReason) {
        this.steps = List.copyOf(steps);
        this.memberNodePositions = memberNodePositions.clone();
        this.bindings = List.copyOf(bindings);
        this.materializedProgramValueIndices = materializedProgramValueIndices.clone();
        this.programToMaterializedSlot = programToMaterializedSlot.clone();
        this.targetMaterializedSlots = targetMaterializedSlots.clone();
        this.instructions = List.copyOf(instructions);
        this.canonicalManifest = canonicalManifest.clone();
        this.canonicalManifestDigest = sha256(this.canonicalManifest);
        this.generatedUnitCount = generatedUnitCount;
        this.expectedGeneratedSourceUtf8Bytes = expectedGeneratedSourceUtf8Bytes;
        this.expectedTotalSourceUtf8Bytes = Math.addExact(
                FIXED_CORPUS_UTF8_BYTES, expectedGeneratedSourceUtf8Bytes);
        this.firstRejectedNodePosition = firstRejectedNodePosition;
        this.capReason = Objects.requireNonNull(capReason, "capReason");
        validate();
    }

    private void validate() {
        if (steps.isEmpty() || bindings.isEmpty() || materializedProgramValueIndices.length == 0
                || canonicalManifest.length == 0 || targetMaterializedSlots.length == 0
                || generatedUnitCount < 0 || generatedUnitCount > MAX_GENERATED_UNITS
                || instructions.size() > MAX_GENERATED_INSTRUCTIONS
                || expectedGeneratedSourceUtf8Bytes < 0
                || expectedGeneratedSourceUtf8Bytes > MAX_GENERATED_SOURCE_UTF8_BYTES
                || expectedTotalSourceUtf8Bytes > MAX_TOTAL_SOURCE_UTF8_BYTES
                || (capReason == CapReason.NONE) != (firstRejectedNodePosition == NO_POSITION)) {
            throw new IllegalArgumentException("malformed Metal fusion plan");
        }
        int generated = 0;
        int memberCursor = 0;
        int bindingCursor = 0;
        int instructionCursor = 0;
        for (Step step : steps) {
            if (step.memberStart() != memberCursor || step.bindingStart() != bindingCursor
                    || step.instructionStart() != instructionCursor
                    || step.memberCount() > memberNodePositions.length - memberCursor
                    || step.bindingCount() > bindings.size() - bindingCursor
                    || step.instructionCount() > instructions.size() - instructionCursor
                    || step.expectedFunctionUtf8Bytes() > MAX_FUNCTION_SOURCE_UTF8_BYTES) {
                throw new IllegalArgumentException("noncanonical Metal fusion step ranges");
            }
            if (step.kind() == StepKind.GENERATED_POINTWISE) generated++;
            memberCursor += step.memberCount();
            bindingCursor += step.bindingCount();
            instructionCursor += step.instructionCount();
        }
        if (memberCursor != memberNodePositions.length || bindingCursor != bindings.size()
                || instructionCursor != instructions.size() || generated != generatedUnitCount) {
            throw new IllegalArgumentException("incomplete Metal fusion step coverage");
        }
        int priorValue = -1;
        for (int slot = 0; slot < materializedProgramValueIndices.length; slot++) {
            int value = materializedProgramValueIndices[slot];
            if (value <= priorValue || value >= programToMaterializedSlot.length
                    || programToMaterializedSlot[value] != slot) {
                throw new IllegalArgumentException("malformed Metal materialized mapping");
            }
            priorValue = value;
        }
        for (int value = 0; value < programToMaterializedSlot.length; value++) {
            int slot = programToMaterializedSlot[value];
            if (slot >= materializedProgramValueIndices.length
                    || (slot >= 0 && materializedProgramValueIndices[slot] != value)) {
                throw new IllegalArgumentException("malformed inverse Metal materialized mapping");
            }
        }
        for (int slot : targetMaterializedSlots) {
            if (slot < 0 || slot >= materializedProgramValueIndices.length) {
                throw new IllegalArgumentException("Metal target has no materialized slot");
            }
        }
    }

    List<Step> steps() {
        return steps;
    }

    int[] memberNodePositions() {
        return memberNodePositions.clone();
    }

    List<Binding> bindings() {
        return bindings;
    }

    int[] materializedProgramValueIndices() {
        return materializedProgramValueIndices.clone();
    }

    int[] programToMaterializedSlot() {
        return programToMaterializedSlot.clone();
    }

    int[] targetMaterializedSlots() {
        return targetMaterializedSlots.clone();
    }

    List<Instruction> instructions() {
        return instructions;
    }

    byte[] canonicalManifest() {
        return canonicalManifest.clone();
    }

    byte[] canonicalManifestDigest() {
        return canonicalManifestDigest.clone();
    }

    int generatedUnitCount() {
        return generatedUnitCount;
    }

    int expectedGeneratedSourceUtf8Bytes() {
        return expectedGeneratedSourceUtf8Bytes;
    }

    int expectedTotalSourceUtf8Bytes() {
        return expectedTotalSourceUtf8Bytes;
    }

    int firstRejectedNodePosition() {
        return firstRejectedNodePosition;
    }

    CapReason capReason() {
        return capReason;
    }

    private static byte[] sha256(byte[] bytes) {
        try {
            return MessageDigest.getInstance("SHA-256").digest(bytes);
        } catch (NoSuchAlgorithmException impossible) {
            throw new AssertionError("SHA-256 is required by the Java platform", impossible);
        }
    }
}
