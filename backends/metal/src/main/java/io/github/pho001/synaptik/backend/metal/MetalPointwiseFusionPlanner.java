package io.github.pho001.synaptik.backend.metal;

import io.github.pho001.synaptik.config.compile.NumericalProfile;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutKind;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/** Deterministic schema-1 planner and integer source-size oracle for raw binary32 pointwise fusion. */
final class MetalPointwiseFusionPlanner {
    private static final int GENERATED_SOURCE_PREAMBLE_UTF8_BYTES = 49;

    private MetalPointwiseFusionPlanner() {}

    static MetalPointwiseFusionPlan plan(
            NumericalProfile numericalProfile,
            MetalMpsGraphProgram program,
            List<MetalMpsGraphProgram.ValueDescriptor> values,
            int[] feeds,
            int[] targets,
            MetalPreparedRoute route) {
        Objects.requireNonNull(numericalProfile, "numericalProfile");
        Objects.requireNonNull(program, "program");
        List<MetalMpsGraphProgram.ValueDescriptor> descriptors = List.copyOf(values);
        int[] feedValues = feeds.clone();
        int[] targetValues = targets.clone();
        if (route != MetalPreparedRoute.CUSTOM_PROGRAM) {
            throw new IllegalArgumentException("fusion extension requires CUSTOM_PROGRAM");
        }
        validateIndices(descriptors.size(), feedValues, "feed");
        validateIndices(descriptors.size(), targetValues, "target");

        List<MetalMpsGraphProgram.Node> nodes = program.nodes();
        int[] consumers = new int[descriptors.size()];
        for (MetalMpsGraphProgram.Node node : nodes) {
            for (int input : node.inputs()) consumers[input] = Math.incrementExact(consumers[input]);
        }
        boolean[] target = new boolean[descriptors.size()];
        for (int value : targetValues) target[value] = true;

        List<Candidate> candidates = discoverCandidates(nodes, descriptors, consumers, target);
        Candidate[] candidateAt = new Candidate[nodes.size()];
        int generatedUnits = 0;
        int generatedInstructions = 0;
        int generatedBytes = 0;
        int firstRejected = MetalPointwiseFusionPlan.NO_POSITION;
        MetalPointwiseFusionPlan.CapReason capReason = MetalPointwiseFusionPlan.CapReason.NONE;
        boolean stopped = false;
        int projectedNode = 0;
        int projectedStep = 0;
        for (Candidate candidate : candidates) {
            if (stopped) continue;
            projectedStep += candidate.start() - projectedNode;
            int functionBytes = functionUtf8Bytes(projectedStep, candidate.opcodes());
            MetalPointwiseFusionPlan.CapReason rejection = capRejection(
                    generatedUnits, generatedInstructions, generatedBytes,
                    candidate.opcodes().size(), functionBytes);
            if (rejection != MetalPointwiseFusionPlan.CapReason.NONE) {
                stopped = true;
                firstRejected = candidate.start();
                capReason = rejection;
                continue;
            }
            candidateAt[candidate.start()] = candidate;
            generatedUnits++;
            generatedInstructions += candidate.opcodes().size();
            generatedBytes = Math.addExact(generatedBytes,
                    functionBytes + (generatedUnits == 1 ? GENERATED_SOURCE_PREAMBLE_UTF8_BYTES : 0));
            projectedNode = candidate.end();
            projectedStep++;
        }

        var stepBuilders = new ArrayList<StepBuilder>();
        boolean[] localTranspose = new boolean[descriptors.size()];
        int nodePosition = 0;
        while (nodePosition < nodes.size()) {
            Candidate generated = candidateAt[nodePosition];
            if (generated != null) {
                stepBuilders.add(StepBuilder.generated(generated));
                nodePosition = generated.end();
            } else {
                MetalMpsGraphProgram.Node node = nodes.get(nodePosition);
                if (node.kind() != MetalMpsGraphProgram.NodeKind.MATMUL) {
                    for (int input : node.inputs()) localTranspose[input] = false;
                }
                boolean custom = fixedCustom(node, descriptors)
                        || (node.kind() == MetalMpsGraphProgram.NodeKind.MATMUL
                                && (localTranspose[node.firstInputIndex()]
                                        || localTranspose[node.secondInputIndex()]));
                stepBuilders.add(StepBuilder.single(
                        custom
                                ? MetalPointwiseFusionPlan.StepKind.FIXED_CUSTOM
                                : MetalPointwiseFusionPlan.StepKind.MPSGRAPH_BOUNDARY,
                        nodePosition));
                if (exactLocalTranspose(node, descriptors))
                    localTranspose[node.outputIndex()] = true;
                nodePosition++;
            }
        }

        boolean[] materialized = new boolean[descriptors.size()];
        for (int value : feedValues) materialized[value] = true;
        for (int value : targetValues) materialized[value] = true;
        for (StepBuilder builder : stepBuilders) {
            if (builder.kind == MetalPointwiseFusionPlan.StepKind.GENERATED_POINTWISE) {
                MetalMpsGraphProgram.Node first = nodes.get(builder.members[0]);
                MetalMpsGraphProgram.Node last = nodes.get(builder.members[builder.members.length - 1]);
                materialized[first.firstInputIndex()] = true;
                materialized[last.outputIndex()] = true;
            } else {
                MetalMpsGraphProgram.Node node = nodes.get(builder.members[0]);
                for (int input : node.inputs()) materialized[input] = true;
                for (int output : node.outputs()) materialized[output] = true;
            }
        }
        int materializedCount = 0;
        for (boolean present : materialized) if (present) materializedCount++;
        int[] materializedValues = new int[materializedCount];
        int[] programToSlot = new int[descriptors.size()];
        Arrays.fill(programToSlot, -1);
        int slot = 0;
        for (int value = 0; value < materialized.length; value++) {
            if (materialized[value]) {
                materializedValues[slot] = value;
                programToSlot[value] = slot++;
            }
        }
        int[] targetSlots = new int[targetValues.length];
        for (int position = 0; position < targetValues.length; position++) {
            targetSlots[position] = programToSlot[targetValues[position]];
        }

        var members = new ArrayList<Integer>();
        var bindings = new ArrayList<MetalPointwiseFusionPlan.Binding>();
        var instructions = new ArrayList<MetalPointwiseFusionPlan.Instruction>();
        var steps = new ArrayList<MetalPointwiseFusionPlan.Step>();
        for (int stepOrdinal = 0; stepOrdinal < stepBuilders.size(); stepOrdinal++) {
            StepBuilder builder = stepBuilders.get(stepOrdinal);
            int memberStart = members.size();
            for (int member : builder.members) members.add(member);
            int bindingStart = bindings.size();
            if (builder.kind == MetalPointwiseFusionPlan.StepKind.GENERATED_POINTWISE) {
                MetalMpsGraphProgram.Node first = nodes.get(builder.members[0]);
                MetalMpsGraphProgram.Node last = nodes.get(builder.members[builder.members.length - 1]);
                addBinding(bindings, stepOrdinal, 0, MetalPointwiseFusionPlan.Access.READ,
                        first.firstInputIndex(), programToSlot);
                addBinding(bindings, stepOrdinal, 1, MetalPointwiseFusionPlan.Access.WRITE,
                        last.outputIndex(), programToSlot);
            } else {
                MetalMpsGraphProgram.Node node = nodes.get(builder.members[0]);
                int argument = 0;
                for (int input : node.inputs()) {
                    addBinding(bindings, stepOrdinal, argument++, MetalPointwiseFusionPlan.Access.READ,
                            input, programToSlot);
                }
                for (int output : node.outputs()) {
                    addBinding(bindings, stepOrdinal, argument++, MetalPointwiseFusionPlan.Access.WRITE,
                            output, programToSlot);
                }
            }
            int instructionStart = instructions.size();
            if (builder.kind == MetalPointwiseFusionPlan.StepKind.GENERATED_POINTWISE) {
                for (int relative = 0; relative < builder.members.length; relative++) {
                    instructions.add(new MetalPointwiseFusionPlan.Instruction(
                            stepOrdinal,
                            relative,
                            MetalPointwiseFusionPlan.Opcode.from(
                                    nodes.get(builder.members[relative]).kind()),
                            relative,
                            relative + 1));
                }
            }
            int functionBytes = builder.kind == MetalPointwiseFusionPlan.StepKind.GENERATED_POINTWISE
                    ? functionUtf8Bytes(stepOrdinal, builder.opcodes(nodes)) : 0;
            steps.add(new MetalPointwiseFusionPlan.Step(
                    builder.kind,
                    memberStart,
                    builder.members.length,
                    bindingStart,
                    bindings.size() - bindingStart,
                    instructionStart,
                    instructions.size() - instructionStart,
                    functionBytes));
        }

        int[] memberArray = members.stream().mapToInt(Integer::intValue).toArray();
        byte[] manifest = manifest(
                numericalProfile,
                descriptors,
                nodes,
                feedValues,
                targetValues,
                steps,
                memberArray,
                bindings,
                materializedValues,
                targetSlots,
                instructions,
                generatedUnits,
                generatedBytes,
                firstRejected,
                capReason);
        return new MetalPointwiseFusionPlan(
                steps,
                memberArray,
                bindings,
                materializedValues,
                programToSlot,
                targetSlots,
                instructions,
                manifest,
                generatedUnits,
                generatedBytes,
                firstRejected,
                capReason);
    }

    private static List<Candidate> discoverCandidates(
            List<MetalMpsGraphProgram.Node> nodes,
            List<MetalMpsGraphProgram.ValueDescriptor> values,
            int[] consumers,
            boolean[] targets) {
        var result = new ArrayList<Candidate>();
        int position = 0;
        while (position < nodes.size()) {
            if (!eligible(nodes.get(position), values)) {
                position++;
                continue;
            }
            int end = position + 1;
            while (end < nodes.size()
                    && eligible(nodes.get(end), values)
                    && nodes.get(end - 1).outputIndex() == nodes.get(end).firstInputIndex()
                    && consumers[nodes.get(end - 1).outputIndex()] == 1
                    && !targets[nodes.get(end - 1).outputIndex()]) {
                end++;
            }
            int cursor = position;
            while (end - cursor >= 2) {
                int remaining = end - cursor;
                int length = remaining == 9 ? 7 : Math.min(8, remaining);
                if (length < 2) break;
                var opcodes = new ArrayList<MetalPointwiseFusionPlan.Opcode>(length);
                for (int index = cursor; index < cursor + length; index++) {
                    opcodes.add(MetalPointwiseFusionPlan.Opcode.from(nodes.get(index).kind()));
                }
                result.add(new Candidate(cursor, cursor + length, List.copyOf(opcodes)));
                cursor += length;
            }
            position = end;
        }
        return result;
    }

    private static boolean eligible(
            MetalMpsGraphProgram.Node node,
            List<MetalMpsGraphProgram.ValueDescriptor> values) {
        if (node.inputs().length != 1 || node.outputs().length != 1
                || node.attributeKind() != MetalMpsGraphProgram.AttributeKind.NONE) {
            return false;
        }
        int wire = node.kind().wireIdentity();
        if (wire < 60 || wire > 63) return false;
        MetalMpsGraphProgram.ValueDescriptor input = values.get(node.firstInputIndex());
        MetalMpsGraphProgram.ValueDescriptor output = values.get(node.outputIndex());
        return eligiblePointwiseGeometry(input)
                && eligiblePointwiseGeometry(output)
                && canonicalFloat32(input)
                && canonicalFloat32(output)
                && input.requiresGrad() == output.requiresGrad()
                && Arrays.equals(input.dimensions(), output.dimensions());
    }

    private static boolean eligiblePointwiseGeometry(
            MetalMpsGraphProgram.ValueDescriptor value) {
        if (value.rank() < 1 || value.rank() > MetalMpsGraphProgram.MAX_RANK) return false;
        long elements = 1L;
        for (long dimension : value.dimensions()) {
            if (dimension <= 0L || elements > 0xffff_ffffL / dimension) return false;
            elements *= dimension;
        }
        return elements >= 1L && elements <= 0xffff_ffffL;
    }

    private static boolean canonicalFloat32(MetalMpsGraphProgram.ValueDescriptor value) {
        return value.dataType() == DataType.FLOAT32 && canonicalStorage(value);
    }

    private static boolean canonicalStorage(MetalMpsGraphProgram.ValueDescriptor value) {
        return value.layout().filter(layout -> layout.kind() == LayoutKind.DENSE_CONTIGUOUS
                && layout.storageOffset() == 0L
                && !layout.isView()
                && layout.referencedElementSpan() == value.elementCount()).isPresent();
    }

    private static boolean exactLocalTranspose(
            MetalMpsGraphProgram.Node node,
            List<MetalMpsGraphProgram.ValueDescriptor> values) {
        if (node.kind() != MetalMpsGraphProgram.NodeKind.PERMUTE
                || node.inputs().length != 1
                || node.outputs().length != 1
                || node.attributeKind() != MetalMpsGraphProgram.AttributeKind.PERMUTATION) {
            return false;
        }
        MetalMpsGraphProgram.ValueDescriptor input = values.get(node.firstInputIndex());
        MetalMpsGraphProgram.ValueDescriptor output = values.get(node.outputIndex());
        int rank = input.rank();
        long[] words = node.attributeWords();
        if (rank < 2 || output.rank() != rank || words.length != rank + 1
                || words[0] != rank || !canonicalStorage(input)) {
            return false;
        }
        for (int axis = 0; axis < rank - 2; axis++) {
            if (words[axis + 1] != axis
                    || output.dimensions()[axis] != input.dimensions()[axis]) {
                return false;
            }
        }
        if (words[rank - 1] != rank - 1L
                || words[rank] != rank - 2L
                || output.dimensions()[rank - 2] != input.dimensions()[rank - 1]
                || output.dimensions()[rank - 1] != input.dimensions()[rank - 2]) {
            return false;
        }
        long[] expectedStrides = new long[rank];
        long stride = 1L;
        for (int axis = rank; axis-- > 0;) {
            expectedStrides[axis] = stride;
            stride = Math.multiplyExact(stride, input.dimensions()[axis]);
        }
        long swap = expectedStrides[rank - 2];
        expectedStrides[rank - 2] = expectedStrides[rank - 1];
        expectedStrides[rank - 1] = swap;
        return output.layout().filter(layout -> layout.storageOffset() == 0L
                && layout.isView()
                && layout.referencedElementSpan() == output.elementCount()
                && Arrays.equals(layout.strides(), expectedStrides)).isPresent();
    }

    private static boolean fixedCustom(
            MetalMpsGraphProgram.Node node,
            List<MetalMpsGraphProgram.ValueDescriptor> values) {
        if (node.kind() == MetalMpsGraphProgram.NodeKind.VARIANCE) {
            return task0069Variance(node, values);
        }
        if (node.kind() == MetalMpsGraphProgram.NodeKind.L1_NORM
                || node.kind() == MetalMpsGraphProgram.NodeKind.SCATTER_ADD) {
            return true;
        }
        if (node.kind().isCustomProgramOperation()) return true;
        if (node.kind() != MetalMpsGraphProgram.NodeKind.MATMUL) return false;
        MetalMpsGraphProgram.ValueDescriptor left = values.get(node.firstInputIndex());
        MetalMpsGraphProgram.ValueDescriptor right = values.get(node.secondInputIndex());
        MetalMpsGraphProgram.ValueDescriptor output = values.get(node.outputIndex());
        return left.dataType() != DataType.FLOAT32 || right.dataType() != DataType.FLOAT32
                || output.dataType() != DataType.FLOAT32 || left.rank() != 2 || right.rank() != 2
                || output.rank() != 2;
    }

    private static boolean task0069Variance(
            MetalMpsGraphProgram.Node node,
            List<MetalMpsGraphProgram.ValueDescriptor> values) {
        long[] words = node.attributeWords();
        MetalMpsGraphProgram.ValueDescriptor input = values.get(node.firstInputIndex());
        MetalMpsGraphProgram.ValueDescriptor output = values.get(node.outputIndex());
        return words.length == 4
                && words[0] == 1L
                && words[1] == 0L
                && (words[2] == 0L || words[2] == 1L)
                && words[3] == 0L
                && canonicalFloat32(input)
                && canonicalFloat32(output)
                && !input.requiresGrad()
                && !output.requiresGrad()
                && input.rank() == 1
                && input.dimensions()[0] == 1L
                && output.rank() == (int) words[2]
                && (words[2] == 0L || output.dimensions()[0] == 1L);
    }

    private static MetalPointwiseFusionPlan.CapReason capRejection(
            int unitCount,
            int instructionCount,
            int generatedBytes,
            int addedInstructions,
            int functionBytes) {
        if (unitCount == MetalPointwiseFusionPlan.MAX_GENERATED_UNITS) {
            return MetalPointwiseFusionPlan.CapReason.UNIT_COUNT;
        }
        if (instructionCount > MetalPointwiseFusionPlan.MAX_GENERATED_INSTRUCTIONS
                - addedInstructions) {
            return MetalPointwiseFusionPlan.CapReason.INSTRUCTION_COUNT;
        }
        if (functionBytes > MetalPointwiseFusionPlan.MAX_FUNCTION_SOURCE_UTF8_BYTES) {
            return MetalPointwiseFusionPlan.CapReason.FUNCTION_UTF8_BYTES;
        }
        int separator = unitCount == 0 ? GENERATED_SOURCE_PREAMBLE_UTF8_BYTES : 0;
        if (generatedBytes > MetalPointwiseFusionPlan.MAX_GENERATED_SOURCE_UTF8_BYTES
                - separator - functionBytes) {
            return MetalPointwiseFusionPlan.CapReason.GENERATED_UTF8_BYTES;
        }
        if (MetalPointwiseFusionPlan.FIXED_CORPUS_UTF8_BYTES + generatedBytes
                > MetalPointwiseFusionPlan.MAX_TOTAL_SOURCE_UTF8_BYTES
                        - separator - functionBytes) {
            return MetalPointwiseFusionPlan.CapReason.TOTAL_UTF8_BYTES;
        }
        return MetalPointwiseFusionPlan.CapReason.NONE;
    }

    static int functionUtf8Bytes(
            int stepOrdinal,
            List<MetalPointwiseFusionPlan.Opcode> opcodes) {
        if (opcodes.size() < 2 || opcodes.size() > 8) {
            throw new IllegalArgumentException("generated chain length must be in 2..8");
        }
        int bytes = 345 + unsignedDigits(stepOrdinal) + digits(opcodes.size());
        for (int index = 0; index < opcodes.size(); index++) {
            bytes = Math.addExact(bytes,
                    16 + digits(index) + digits(index + 1) + opcodes.get(index).helperUtf8Bytes());
        }
        return bytes;
    }

    private static int digits(int value) {
        if (value < 0) throw new IllegalArgumentException("decimal value must be non-negative");
        if (value < 10) return 1;
        if (value < 100) return 2;
        if (value < 1_000) return 3;
        if (value < 10_000) return 4;
        if (value < 100_000) return 5;
        if (value < 1_000_000) return 6;
        if (value < 10_000_000) return 7;
        if (value < 100_000_000) return 8;
        if (value < 1_000_000_000) return 9;
        return 10;
    }

    private static int unsignedDigits(int value) {
        long unsigned = Integer.toUnsignedLong(value);
        if (unsigned < 10L) return 1;
        if (unsigned < 100L) return 2;
        if (unsigned < 1_000L) return 3;
        if (unsigned < 10_000L) return 4;
        if (unsigned < 100_000L) return 5;
        if (unsigned < 1_000_000L) return 6;
        if (unsigned < 10_000_000L) return 7;
        if (unsigned < 100_000_000L) return 8;
        if (unsigned < 1_000_000_000L) return 9;
        return 10;
    }

    private static void addBinding(
            List<MetalPointwiseFusionPlan.Binding> bindings,
            int step,
            int argument,
            MetalPointwiseFusionPlan.Access access,
            int programValue,
            int[] programToSlot) {
        int slot = programToSlot[programValue];
        if (slot < 0) throw new IllegalStateException("step port is not materialized");
        bindings.add(new MetalPointwiseFusionPlan.Binding(
                step, argument, access, slot, programValue));
    }

    private static byte[] manifest(
            NumericalProfile profile,
            List<MetalMpsGraphProgram.ValueDescriptor> values,
            List<MetalMpsGraphProgram.Node> nodes,
            int[] feeds,
            int[] targets,
            List<MetalPointwiseFusionPlan.Step> steps,
            int[] members,
            List<MetalPointwiseFusionPlan.Binding> bindings,
            int[] materialized,
            int[] targetSlots,
            List<MetalPointwiseFusionPlan.Instruction> instructions,
            int generatedUnits,
            int generatedBytes,
            int rejected,
            MetalPointwiseFusionPlan.CapReason reason) {
        StringBuilder text = new StringBuilder(4096);
        text.append("format 1\n")
                .append("schema 17\n")
                .append("generator 1\n")
                .append("route 3\n")
                .append("profile ").append(Integer.toUnsignedString(
                        MetalMpsGraphProgram.numericalProfileWireValue(profile))).append('\n')
                .append("counts ").append(values.size()).append(' ').append(nodes.size()).append(' ')
                .append(feeds.length).append(' ').append(targets.length).append(' ')
                .append(steps.size()).append(' ').append(members.length).append(' ')
                .append(bindings.size()).append(' ').append(materialized.length).append(' ')
                .append(instructions.size()).append('\n')
                .append("stop ").append(Integer.toUnsignedString(rejected)).append(' ')
                .append(reason.wire()).append('\n')
                .append("source ").append(generatedBytes).append(' ')
                .append(MetalPointwiseFusionPlan.FIXED_CORPUS_UTF8_BYTES).append(' ')
                .append(MetalPointwiseFusionPlan.FIXED_CORPUS_UTF8_BYTES + generatedBytes)
                .append('\n')
                .append("caps 32 256 16384 262144 1048576\n")
                .append("source-size-table 1\n")
                .append("fixed-corpus-bytes 77444\n")
                .append("opcodes floor=1 ceil=2 sign=3 relu=4\n")
                .append("pointmeta-abi size=32 align=8 elementCount=u64@0 gridWidth=u64@8 gridHeight=u64@16 scalar=u32@24 reserved=u32@28\n");
        for (int value = 0; value < values.size(); value++) {
            MetalMpsGraphProgram.ValueDescriptor descriptor = values.get(value);
            text.append("value ").append(value).append(' ')
                    .append(MetalMpsGraphProgram.dataTypeWire(descriptor.dataType())).append(' ')
                    .append(descriptor.rank()).append(' ')
                    .append(descriptor.requiresGrad() ? 1 : 0).append(' ')
                    .append(descriptor.densePhysical() ? 1 : 0);
            for (long dimension : descriptor.dimensions()) text.append(' ').append(dimension);
            text.append('\n');
        }
        for (int slot = 0; slot < materialized.length; slot++) {
            text.append("materialized ").append(slot).append(' ')
                    .append(materialized[slot]).append('\n');
        }
        for (int index = 0; index < targets.length; index++) {
            text.append("target ").append(index).append(' ').append(targets[index]).append(' ')
                    .append(targetSlots[index]).append('\n');
        }
        for (int ordinal = 0; ordinal < steps.size(); ordinal++) {
            MetalPointwiseFusionPlan.Step step = steps.get(ordinal);
            text.append("step ").append(ordinal).append(' ').append(step.kind().wire()).append(' ')
                    .append(step.memberStart()).append(' ').append(step.memberCount()).append(' ')
                    .append(step.bindingStart()).append(' ').append(step.bindingCount()).append(' ')
                    .append(step.instructionStart()).append(' ').append(step.instructionCount()).append(' ')
                    .append(step.expectedFunctionUtf8Bytes()).append('\n');
            if (step.kind() == MetalPointwiseFusionPlan.StepKind.GENERATED_POINTWISE) {
                int terminalNode = members[step.memberStart() + step.memberCount() - 1];
                long elements = values.get(nodes.get(terminalNode).outputIndex()).elementCount();
                text.append("pointmeta ").append(ordinal).append(' ')
                        .append(elements).append(' ').append(elements).append(" 1 0 0\n");
            }
        }
        for (int ordinal = 0; ordinal < members.length; ordinal++) {
            text.append("member ").append(ordinal).append(' ').append(members[ordinal]).append('\n');
        }
        for (int ordinal = 0; ordinal < bindings.size(); ordinal++) {
            MetalPointwiseFusionPlan.Binding binding = bindings.get(ordinal);
            text.append("binding ").append(ordinal).append(' ').append(binding.stepOrdinal()).append(' ')
                    .append(binding.argumentOrdinal()).append(' ').append(binding.access().wire()).append(' ')
                    .append(binding.materializedSlot()).append(' ')
                    .append(binding.programValueIndex()).append('\n');
        }
        for (int ordinal = 0; ordinal < instructions.size(); ordinal++) {
            MetalPointwiseFusionPlan.Instruction instruction = instructions.get(ordinal);
            text.append("instruction ").append(ordinal).append(' ')
                    .append(instruction.stepOrdinal()).append(' ')
                    .append(instruction.relativeNodePosition()).append(' ')
                    .append(instruction.opcode().wire()).append(" 1 ")
                    .append(instruction.inputSsa()).append(" 4294967295 4294967295 ")
                    .append(instruction.outputSsa()).append(" 0 0 0 0000000000000000 0000000000000000\n");
        }
        text.append("generated-units ").append(generatedUnits).append('\n').append("end\n");
        return text.toString().getBytes(StandardCharsets.US_ASCII);
    }

    private static void validateIndices(int valueCount, int[] indices, String name) {
        boolean[] seen = new boolean[valueCount];
        for (int index : indices) {
            if (index < 0 || index >= valueCount || seen[index]) {
                throw new IllegalArgumentException("malformed " + name + " indices");
            }
            seen[index] = true;
        }
    }

    private record Candidate(
            int start, int end, List<MetalPointwiseFusionPlan.Opcode> opcodes) {}

    private static final class StepBuilder {
        private final MetalPointwiseFusionPlan.StepKind kind;
        private final int[] members;

        private StepBuilder(MetalPointwiseFusionPlan.StepKind kind, int[] members) {
            this.kind = kind;
            this.members = members;
        }

        static StepBuilder single(MetalPointwiseFusionPlan.StepKind kind, int member) {
            return new StepBuilder(kind, new int[] {member});
        }

        static StepBuilder generated(Candidate candidate) {
            int[] members = new int[candidate.end() - candidate.start()];
            for (int index = 0; index < members.length; index++) {
                members[index] = candidate.start() + index;
            }
            return new StepBuilder(MetalPointwiseFusionPlan.StepKind.GENERATED_POINTWISE, members);
        }

        List<MetalPointwiseFusionPlan.Opcode> opcodes(List<MetalMpsGraphProgram.Node> nodes) {
            var result = new ArrayList<MetalPointwiseFusionPlan.Opcode>(members.length);
            for (int member : members) {
                result.add(MetalPointwiseFusionPlan.Opcode.from(nodes.get(member).kind()));
            }
            return result;
        }
    }
}
