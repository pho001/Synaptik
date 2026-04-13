package graph.fused.vector;

import backend.kernels.cpu.fused.FusedExecutionOptions;
import graph.codegen.FusedExpressionPlan;
import graph.codegen.FusedNodePlan;
import graph.codegen.ScalarDoubleAttribute;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import operations.Operation;
import tensor.Tensor;

import java.util.List;

final class DirectLinearF64Program {
    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;
    private static final DoubleVector ZERO = DoubleVector.zero(SPECIES);
    private static final DoubleVector ONE = DoubleVector.broadcast(SPECIES, 1.0d);

    private final int inputCount;
    private final int outputRef;
    private final Instruction[] instructions;

    private DirectLinearF64Program(int inputCount, int outputRef, Instruction[] instructions) {
        this.inputCount = inputCount;
        this.outputRef = outputRef;
        this.instructions = instructions;
    }

    static DirectLinearF64Program lower(FusedExpressionPlan plan) {
        Instruction[] instructions = new Instruction[plan.nodeCount()];
        int i = 0;
        for (FusedNodePlan node : plan.nodes()) {
            double scalar = node.attributes() instanceof ScalarDoubleAttribute attr ? attr.value() : 0.0d;
            List<Integer> refs = node.inputRefs();
            int input0 = refs.isEmpty() ? -1 : refs.get(0);
            int input1 = refs.size() < 2 ? -1 : refs.get(1);
            int input2 = refs.size() < 3 ? -1 : refs.get(2);
            instructions[i++] = new Instruction(node.opType(), input0, input1, input2, scalar, node.outputRef());
        }
        return new DirectLinearF64Program(plan.inputCount(), plan.outputRef(), instructions);
    }

    void applyScalar(List<Tensor> inputs, Tensor out, int startInclusive, int endExclusive, FusedExecutionOptions options) {
        double[][] in = extractInputs(inputs);
        int[] offsets = extractOffsets(inputs);
        double[] outData = out.getFloat64Data();
        int outOffset = out.getStorageOffsetUnsafe();
        double[] slots = new double[instructions.length];
        for (int index = startInclusive; index < endExclusive; index++) {
            for (Instruction instruction : instructions) {
                slots[slotIndex(instruction.outputRef)] = evalScalar(instruction, in, offsets, slots, index, options);
            }
            outData[outOffset + index] = loadScalar(outputRef, in, offsets, slots, index);
        }
    }

    void applyVector(List<Tensor> inputs, Tensor out, int startInclusive, int endExclusive) {
        double[][] in = extractInputs(inputs);
        int[] offsets = extractOffsets(inputs);
        double[] outData = out.getFloat64Data();
        int outOffset = out.getStorageOffsetUnsafe();
        DoubleVector[] slots = new DoubleVector[instructions.length];
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

    private double evalScalar(Instruction instruction, double[][] in, int[] offsets, double[] slots, int index, FusedExecutionOptions options) {
        double a = instruction.input0 >= 0 ? loadScalar(instruction.input0, in, offsets, slots, index) : 0.0d;
        double b = instruction.input1 >= 0 ? loadScalar(instruction.input1, in, offsets, slots, index) : 0.0d;
        return switch (instruction.opType) {
            case ADD -> a + b;
            case SUB -> a - b;
            case MUL -> a * b;
            case DIV -> a / b;
            case MIN -> Math.min(a, b);
            case MAX -> Math.max(a, b);
            case NEG -> -a;
            case INV -> 1.0d / a;
            case LOG -> Math.log(a);
            case EXP -> options.useFastExpApprox() ? graph.codegen.FusedScalarOps.fastExpF64(a) : graph.codegen.FusedScalarOps.expF64(a, false);
            case FAST_EXP -> graph.codegen.FusedScalarOps.fastExpF64(a);
            case TANH -> options.useFastTanhApprox() ? graph.codegen.FusedScalarOps.fastTanhF64(a) : graph.codegen.FusedScalarOps.tanhF64(a, false);
            case FAST_TANH -> graph.codegen.FusedScalarOps.fastTanhF64(a);
            case SQRT -> Math.sqrt(a);
            case ABS -> Math.abs(a);
            case MUL_SCALAR -> a * instruction.scalarValue;
            case RELU -> Math.max(a, 0.0d);
            case CLAMP_MIN -> Math.max(a, instruction.scalarValue);
            case CLAMP_MAX -> Math.min(a, instruction.scalarValue);
            case SIGMOID -> 1.0d / (1.0d + Math.exp(-a));
            case POW -> powScalar(a, instruction.scalarValue);
            case NOOP -> a;
            default -> throw new UnsupportedOperationException("Unsupported direct F64 fused op: " + instruction.opType);
        };
    }

    private DoubleVector evalVector(Instruction instruction, double[][] in, int[] offsets, DoubleVector[] slots, int index) {
        DoubleVector a = instruction.input0 >= 0 ? loadVector(instruction.input0, in, offsets, slots, index) : ZERO;
        DoubleVector b = instruction.input1 >= 0 ? loadVector(instruction.input1, in, offsets, slots, index) : ZERO;
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
            case MUL_SCALAR -> a.mul(DoubleVector.broadcast(SPECIES, instruction.scalarValue));
            case RELU -> a.max(ZERO);
            case CLAMP_MIN -> a.max(DoubleVector.broadcast(SPECIES, instruction.scalarValue));
            case CLAMP_MAX -> a.min(DoubleVector.broadcast(SPECIES, instruction.scalarValue));
            case SIGMOID -> {
                DoubleVector half = DoubleVector.broadcast(SPECIES, 0.5d);
                yield a.mul(half).lanewise(VectorOperators.TANH).add(ONE).mul(half);
            }
            case POW -> powVector(a, instruction.scalarValue);
            case NOOP -> a;
            default -> throw new UnsupportedOperationException("Unsupported direct F64 fused vector op: " + instruction.opType);
        };
    }

    private static double powScalar(double value, double exponent) {
        if (exponent == 0.0d) return 1.0d;
        if (exponent == 1.0d) return value;
        if (exponent == 2.0d) return value * value;
        if (exponent == 0.5d) return Math.sqrt(value);
        if (exponent == -1.0d) return 1.0d / value;
        return Math.pow(value, exponent);
    }

    private static DoubleVector powVector(DoubleVector value, double exponent) {
        if (exponent == 0.0d) return ONE;
        if (exponent == 1.0d) return value;
        if (exponent == 2.0d) return value.mul(value);
        if (exponent == 0.5d) return value.lanewise(VectorOperators.SQRT);
        if (exponent == -1.0d) return ONE.div(value);
        throw new UnsupportedOperationException("Unsupported direct F64 fused vector pow exponent: " + exponent);
    }

    private double loadScalar(int ref, double[][] in, int[] offsets, double[] slots, int index) {
        if (ref < inputCount) {
            return in[ref][offsets[ref] + index];
        }
        return slots[slotIndex(ref)];
    }

    private DoubleVector loadVector(int ref, double[][] in, int[] offsets, DoubleVector[] slots, int index) {
        if (ref < inputCount) {
            return DoubleVector.fromArray(SPECIES, in[ref], offsets[ref] + index);
        }
        return slots[slotIndex(ref)];
    }

    private int slotIndex(int ref) {
        return ref - inputCount;
    }

    private double[][] extractInputs(List<Tensor> inputs) {
        double[][] in = new double[inputCount][];
        for (int i = 0; i < inputCount; i++) {
            in[i] = inputs.get(i).getFloat64Data();
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

    private record Instruction(Operation.OpType opType, int input0, int input1, int input2, double scalarValue, int outputRef) {
    }
}
