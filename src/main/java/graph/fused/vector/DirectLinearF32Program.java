package graph.fused.vector;

import backend.kernels.cpu.fused.FusedExecutionOptions;
import graph.codegen.FusedExpressionPlan;
import graph.codegen.FusedNodePlan;
import graph.codegen.ScalarDoubleAttribute;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import operations.Operation;
import tensor.Tensor;

import java.util.List;

final class DirectLinearF32Program {
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final FloatVector ZERO = FloatVector.zero(SPECIES);
    private static final FloatVector ONE = FloatVector.broadcast(SPECIES, 1.0f);

    private final int inputCount;
    private final int outputRef;
    private final Instruction[] instructions;

    private DirectLinearF32Program(int inputCount, int outputRef, Instruction[] instructions) {
        this.inputCount = inputCount;
        this.outputRef = outputRef;
        this.instructions = instructions;
    }

    static DirectLinearF32Program lower(FusedExpressionPlan plan) {
        Instruction[] instructions = new Instruction[plan.nodeCount()];
        int i = 0;
        for (FusedNodePlan node : plan.nodes()) {
            float scalar = node.attributes() instanceof ScalarDoubleAttribute attr ? (float) attr.value() : 0.0f;
            List<Integer> refs = node.inputRefs();
            int input0 = refs.isEmpty() ? -1 : refs.get(0);
            int input1 = refs.size() < 2 ? -1 : refs.get(1);
            int input2 = refs.size() < 3 ? -1 : refs.get(2);
            instructions[i++] = new Instruction(node.opType(), input0, input1, input2, scalar, node.outputRef());
        }
        return new DirectLinearF32Program(plan.inputCount(), plan.outputRef(), instructions);
    }

    void applyScalar(List<Tensor> inputs, Tensor out, int startInclusive, int endExclusive, FusedExecutionOptions options) {
        float[][] in = extractInputs(inputs);
        int[] offsets = extractOffsets(inputs);
        float[] outData = out.getFloat32Data();
        int outOffset = out.getStorageOffsetUnsafe();
        float[] slots = new float[instructions.length];
        for (int index = startInclusive; index < endExclusive; index++) {
            for (Instruction instruction : instructions) {
                slots[slotIndex(instruction.outputRef)] = evalScalar(instruction, in, offsets, slots, index, options);
            }
            outData[outOffset + index] = loadScalar(outputRef, in, offsets, slots, index);
        }
    }

    void applyVector(List<Tensor> inputs, Tensor out, int startInclusive, int endExclusive) {
        float[][] in = extractInputs(inputs);
        int[] offsets = extractOffsets(inputs);
        float[] outData = out.getFloat32Data();
        int outOffset = out.getStorageOffsetUnsafe();
        FloatVector[] slots = new FloatVector[instructions.length];
        int width = SPECIES.length();
        int upper = endExclusive - ((endExclusive - startInclusive) % width);
        for (int index = startInclusive; index < upper; index += width) {
            for (Instruction instruction : instructions) {
                slots[slotIndex(instruction.outputRef)] = evalVector(instruction, in, offsets, slots, index);
            }
            loadVector(outputRef, in, offsets, slots, index).intoArray(outData, outOffset + index);
        }
        if (upper < endExclusive) {
            applyScalar(inputs, out, upper, endExclusive, FusedExecutionOptions.exact());
        }
    }

    private float evalScalar(Instruction instruction, float[][] in, int[] offsets, float[] slots, int index, FusedExecutionOptions options) {
        float a = instruction.input0 >= 0 ? loadScalar(instruction.input0, in, offsets, slots, index) : 0.0f;
        float b = instruction.input1 >= 0 ? loadScalar(instruction.input1, in, offsets, slots, index) : 0.0f;
        return switch (instruction.opType) {
            case ADD -> a + b;
            case SUB -> a - b;
            case MUL -> a * b;
            case DIV -> a / b;
            case MIN -> Math.min(a, b);
            case MAX -> Math.max(a, b);
            case NEG -> -a;
            case INV -> 1.0f / a;
            case LOG -> (float) Math.log(a);
            case EXP -> options.useFastExpApprox() ? utils.FastExp.fastExpF32(a) : (float) Math.exp(a);
            case FAST_EXP -> utils.FastExp.fastExpF32(a);
            case TANH -> options.useFastTanhApprox() ? utils.FastExp.fastTanhF32(a) : (float) Math.tanh(a);
            case FAST_TANH -> utils.FastExp.fastTanhF32(a);
            case SQRT -> (float) Math.sqrt(a);
            case ABS -> Math.abs(a);
            case MUL_SCALAR -> a * instruction.scalarValue;
            case RELU -> Math.max(a, 0.0f);
            case CLAMP_MIN -> Math.max(a, instruction.scalarValue);
            case CLAMP_MAX -> Math.min(a, instruction.scalarValue);
            case SIGMOID -> 1.0f / (1.0f + (float) Math.exp(-a));
            case POW -> powScalar(a, instruction.scalarValue);
            case NOOP -> a;
            default -> throw new UnsupportedOperationException("Unsupported direct F32 fused op: " + instruction.opType);
        };
    }

    private FloatVector evalVector(Instruction instruction, float[][] in, int[] offsets, FloatVector[] slots, int index) {
        FloatVector a = instruction.input0 >= 0 ? loadVector(instruction.input0, in, offsets, slots, index) : ZERO;
        FloatVector b = instruction.input1 >= 0 ? loadVector(instruction.input1, in, offsets, slots, index) : ZERO;
        return switch (instruction.opType) {
            case ADD -> a.add(b);
            case SUB -> a.sub(b);
            case MUL -> a.mul(b);
            case DIV -> a.div(b);
            case MIN -> a.min(b);
            case MAX -> a.max(b);
            case NEG -> a.neg();
            case INV -> ONE.div(a);
            case LOG -> a.lanewise(VectorOperators.LOG);
            case EXP, FAST_EXP -> a.lanewise(VectorOperators.EXP);
            case TANH, FAST_TANH -> a.lanewise(VectorOperators.TANH);
            case SQRT -> a.lanewise(VectorOperators.SQRT);
            case ABS -> a.abs();
            case MUL_SCALAR -> a.mul(FloatVector.broadcast(SPECIES, instruction.scalarValue));
            case RELU -> a.max(ZERO);
            case CLAMP_MIN -> a.max(FloatVector.broadcast(SPECIES, instruction.scalarValue));
            case CLAMP_MAX -> a.min(FloatVector.broadcast(SPECIES, instruction.scalarValue));
            case SIGMOID -> {
                FloatVector half = FloatVector.broadcast(SPECIES, 0.5f);
                yield a.mul(half).lanewise(VectorOperators.TANH).add(ONE).mul(half);
            }
            case POW -> powVector(a, instruction.scalarValue);
            case NOOP -> a;
            default -> throw new UnsupportedOperationException("Unsupported direct F32 fused vector op: " + instruction.opType);
        };
    }

    private static float powScalar(float value, float exponent) {
        if (exponent == 0.0f) return 1.0f;
        if (exponent == 1.0f) return value;
        if (exponent == 2.0f) return value * value;
        if (exponent == 0.5f) return (float) Math.sqrt(value);
        if (exponent == -1.0f) return 1.0f / value;
        return (float) Math.pow(value, exponent);
    }

    private static FloatVector powVector(FloatVector value, float exponent) {
        if (exponent == 0.0f) return ONE;
        if (exponent == 1.0f) return value;
        if (exponent == 2.0f) return value.mul(value);
        if (exponent == 0.5f) return value.lanewise(VectorOperators.SQRT);
        if (exponent == -1.0f) return ONE.div(value);
        throw new UnsupportedOperationException("Unsupported direct F32 fused vector pow exponent: " + exponent);
    }

    private float loadScalar(int ref, float[][] in, int[] offsets, float[] slots, int index) {
        if (ref < inputCount) {
            return in[ref][offsets[ref] + index];
        }
        return slots[slotIndex(ref)];
    }

    private FloatVector loadVector(int ref, float[][] in, int[] offsets, FloatVector[] slots, int index) {
        if (ref < inputCount) {
            return FloatVector.fromArray(SPECIES, in[ref], offsets[ref] + index);
        }
        return slots[slotIndex(ref)];
    }

    private int slotIndex(int ref) {
        return ref - inputCount;
    }

    private float[][] extractInputs(List<Tensor> inputs) {
        float[][] in = new float[inputCount][];
        for (int i = 0; i < inputCount; i++) {
            in[i] = inputs.get(i).getFloat32Data();
        }
        return in;
    }

    private int[] extractOffsets(List<Tensor> inputs) {
        int[] offsets = new int[inputCount];
        for (int i = 0; i < inputCount; i++) {
            offsets[i] = inputs.get(i).getStorageOffsetUnsafe();
        }
        return offsets;
    }

    private record Instruction(Operation.OpType opType, int input0, int input1, int input2, float scalarValue, int outputRef) {
    }
}
