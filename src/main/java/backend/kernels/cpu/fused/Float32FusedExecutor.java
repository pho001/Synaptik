package backend.kernels.cpu.fused;

import backend.kernels.cpu.*;

import backend.kernels.cpu.fused.FusedExecutionOptions;
import graph.codegen.FusedBroadcastCursor;
import graph.codegen.FusedExpressionPlan;
import graph.codegen.FusedExternalInputPlan;
import graph.codegen.FusedNodeAttributes;
import graph.codegen.FusedNodePlan;
import graph.codegen.ScalarDoubleAttribute;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import operations.FusedOperation;
import operations.Operation;
import tensor.DataType;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

public final class Float32FusedExecutor {
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final FloatVector ZERO = FloatVector.zero(SPECIES);
    private static final FloatVector ONE = FloatVector.broadcast(SPECIES, 1.0f);

    private Float32FusedExecutor() {
    }

    public static void applyRangeScalar(
            FusedOperation fused,
            List<Tensor> inputs,
            Tensor node,
            CpuKernelContext context,
            FusedExecutionOptions options,
            int startInclusive,
            int endExclusive
    ) {
        validateArgs(fused, inputs, node, context, options);
        executeRangeScalar(fused.getPlan(), inputs, node, options, startInclusive, endExclusive);
    }

    public static void applyRangeVector(
            FusedOperation fused,
            List<Tensor> inputs,
            Tensor node,
            CpuKernelContext context,
            FusedExecutionOptions options,
            int startInclusive,
            int endExclusive
    ) {
        validateArgs(fused, inputs, node, context, options);
        FusedExpressionPlan plan = fused.getPlan();
        if (supportsVectorFastPath(plan, inputs, node)) {
            executeRangeVector(plan, inputs, node, startInclusive, endExclusive);
            return;
        }
        executeRangeScalar(plan, inputs, node, options, startInclusive, endExclusive);
    }

    private static void validateArgs(
            FusedOperation fused,
            List<Tensor> inputs,
            Tensor node,
            CpuKernelContext context,
            FusedExecutionOptions options
    ) {
        if (fused == null || inputs == null || node == null || context == null || options == null) {
            throw new IllegalArgumentException("F32 fused execution arguments cannot be null");
        }
    }

    private static void executeRangeScalar(
            FusedExpressionPlan plan,
            List<Tensor> inputs,
            Tensor out,
            FusedExecutionOptions options,
            int startInclusive,
            int endExclusive
    ) {
        List<InputAccessor> accessors = new ArrayList<>(plan.inputCount());
        for (int i = 0; i < plan.inputCount(); i++) {
            accessors.add(InputAccessor.create(plan.inputs().get(i), inputs.get(i), startInclusive));
        }

        float[] values = new float[plan.nodeCount()];
        boolean[] bools = new boolean[plan.nodeCount()];
        FusedNodePlan outputNode = plan.outputNode();
        int outBaseOffset = out.getStorageOffsetUnsafe();

        for (int index = startInclusive; index < endExclusive; index++) {
            for (FusedNodePlan current : plan.nodes()) {
                switch (current.opType()) {
                    case WHERE -> {
                        boolean cond = loadBoolRef(plan, current.inputRefs().get(0), accessors, bools);
                        values[current.index()] = cond
                                ? loadValueRef(plan, current.inputRefs().get(1), accessors, values)
                                : loadValueRef(plan, current.inputRefs().get(2), accessors, values);
                    }
                    case GT, GE, LT, LE, EQ, NE -> {
                        float a = loadValueRef(plan, current.inputRefs().get(0), accessors, values);
                        float b = loadValueRef(plan, current.inputRefs().get(1), accessors, values);
                        bools[current.index()] = evalCompare(current.opType(), a, b);
                    }
                    case LOGICAL_AND -> bools[current.index()] =
                            loadBoolRef(plan, current.inputRefs().get(0), accessors, bools)
                                    && loadBoolRef(plan, current.inputRefs().get(1), accessors, bools);
                    case LOGICAL_OR -> bools[current.index()] =
                            loadBoolRef(plan, current.inputRefs().get(0), accessors, bools)
                                    || loadBoolRef(plan, current.inputRefs().get(1), accessors, bools);
                    case LOGICAL_NOT -> bools[current.index()] =
                            !loadBoolRef(plan, current.inputRefs().get(0), accessors, bools);
                    default -> values[current.index()] = evalNumeric(
                            current,
                            loadInputValues(plan, current, accessors, values),
                            options
                    );
                }
            }

            if (outputNode.outputType() == DataType.BOOL) {
                out.getBoolData()[outBaseOffset + index] = bools[outputNode.index()] ? (byte) 1 : (byte) 0;
            } else {
                out.getFloat32Data()[outBaseOffset + index] = values[outputNode.index()];
            }
            for (InputAccessor accessor : accessors) {
                accessor.step();
            }
        }
    }

    private static void executeRangeVector(
            FusedExpressionPlan plan,
            List<Tensor> inputs,
            Tensor out,
            int startInclusive,
            int endExclusive
    ) {
        List<VectorInputAccessor> accessors = new ArrayList<>(plan.inputCount());
        for (int i = 0; i < plan.inputCount(); i++) {
            accessors.add(VectorInputAccessor.create(plan.inputs().get(i), inputs.get(i)));
        }

        Object[] values = new Object[plan.nodeCount()];
        FusedNodePlan outputNode = plan.outputNode();
        float[] outData = out.getFloat32Data();
        byte[] outBoolData = out.getBoolData();
        int outBaseOffset = out.getStorageOffsetUnsafe();
        int width = SPECIES.length();
        int upperBound = endExclusive - ((endExclusive - startInclusive) % width);

        for (int index = startInclusive; index < upperBound; index += width) {
            for (FusedNodePlan current : plan.nodes()) {
                values[current.index()] = evalNumericVector(current, plan, accessors, values, index);
            }
            if (outputNode.outputType() == DataType.BOOL) {
                VectorMask<Float> mask = (VectorMask<Float>) values[outputNode.index()];
                for (int lane = 0; lane < width; lane++) {
                    outBoolData[outBaseOffset + index + lane] = mask.laneIsSet(lane) ? (byte) 1 : (byte) 0;
                }
            } else {
                ((FloatVector) values[outputNode.index()]).intoArray(outData, outBaseOffset + index);
            }
        }

        if (upperBound < endExclusive) {
            executeRangeScalar(plan, inputs, out, FusedExecutionOptions.exact(), upperBound, endExclusive);
        }
    }

    private static float[] loadInputValues(
            FusedExpressionPlan plan,
            FusedNodePlan current,
            List<InputAccessor> accessors,
            float[] values
    ) {
        List<Integer> refs = current.inputRefs();
        float[] out = new float[refs.size()];
        for (int i = 0; i < refs.size(); i++) {
            out[i] = loadValueRef(plan, refs.get(i), accessors, values);
        }
        return out;
    }

    private static float loadValueRef(
            FusedExpressionPlan plan,
            int ref,
            List<InputAccessor> accessors,
            float[] values
    ) {
        if (ref < plan.inputCount()) {
            return accessors.get(ref).floatValue();
        }
        return values[ref - plan.inputCount()];
    }

    private static boolean loadBoolRef(
            FusedExpressionPlan plan,
            int ref,
            List<InputAccessor> accessors,
            boolean[] bools
    ) {
        if (ref < plan.inputCount()) {
            return accessors.get(ref).boolValue();
        }
        return bools[ref - plan.inputCount()];
    }

    private static Object loadVectorRef(
            FusedExpressionPlan plan,
            int ref,
            List<VectorInputAccessor> accessors,
            Object[] values,
            int logicalIndex
    ) {
        if (ref < plan.inputCount()) {
            return accessors.get(ref).load(logicalIndex);
        }
        return values[ref - plan.inputCount()];
    }

    private static Object evalNumericVector(
            FusedNodePlan node,
            FusedExpressionPlan plan,
            List<VectorInputAccessor> accessors,
            Object[] values,
            int logicalIndex
    ) {
        Object a = loadVectorRef(plan, node.inputRefs().get(0), accessors, values, logicalIndex);
        Object b = node.inputRefs().size() > 1
                ? loadVectorRef(plan, node.inputRefs().get(1), accessors, values, logicalIndex)
                : null;
        FusedNodeAttributes attrs = node.attributes();
        return switch (node.opType()) {
            case ADD -> ((FloatVector) a).add((FloatVector) b);
            case SUB -> ((FloatVector) a).sub((FloatVector) b);
            case MUL -> ((FloatVector) a).mul((FloatVector) b);
            case DIV -> ((FloatVector) a).div((FloatVector) b);
            case MIN -> ((FloatVector) a).min((FloatVector) b);
            case MAX -> ((FloatVector) a).max((FloatVector) b);
            case GT -> ((FloatVector) a).compare(VectorOperators.GT, (FloatVector) b);
            case GE -> ((FloatVector) a).compare(VectorOperators.GE, (FloatVector) b);
            case LT -> ((FloatVector) a).compare(VectorOperators.LT, (FloatVector) b);
            case LE -> ((FloatVector) a).compare(VectorOperators.LE, (FloatVector) b);
            case EQ -> ((FloatVector) a).compare(VectorOperators.EQ, (FloatVector) b);
            case NE -> ((FloatVector) a).compare(VectorOperators.NE, (FloatVector) b);
            case LOGICAL_AND -> ((VectorMask<Float>) a).and((VectorMask<Float>) b);
            case LOGICAL_OR -> ((VectorMask<Float>) a).or((VectorMask<Float>) b);
            case LOGICAL_NOT -> ((VectorMask<Float>) a).not();
            case WHERE -> ((FloatVector) loadVectorRef(plan, node.inputRefs().get(2), accessors, values, logicalIndex))
                    .blend((FloatVector) b, (VectorMask<Float>) a);
            case NEG -> ((FloatVector) a).neg();
            case INV -> ONE.div((FloatVector) a);
            case LOG -> ((FloatVector) a).lanewise(VectorOperators.LOG);
            case EXP, FAST_EXP -> ((FloatVector) a).lanewise(VectorOperators.EXP);
            case TANH, FAST_TANH -> ((FloatVector) a).lanewise(VectorOperators.TANH);
            case SQRT -> ((FloatVector) a).lanewise(VectorOperators.SQRT);
            case ABS -> ((FloatVector) a).abs();
            case MUL_SCALAR -> ((FloatVector) a).mul(FloatVector.broadcast(SPECIES, (float) ((ScalarDoubleAttribute) attrs).value()));
            case RELU -> ((FloatVector) a).max(ZERO);
            case CLAMP_MIN -> ((FloatVector) a).max(FloatVector.broadcast(SPECIES, (float) ((ScalarDoubleAttribute) attrs).value()));
            case CLAMP_MAX -> ((FloatVector) a).min(FloatVector.broadcast(SPECIES, (float) ((ScalarDoubleAttribute) attrs).value()));
            case SIGMOID -> {
                FloatVector half = FloatVector.broadcast(SPECIES, 0.5f);
                yield ((FloatVector) a).mul(half).lanewise(VectorOperators.TANH).add(ONE).mul(half);
            }
            case POW -> {
                float exponent = (float) ((ScalarDoubleAttribute) attrs).value();
                yield switch (Float.floatToIntBits(exponent)) {
                    case 0x00000000 -> ONE;
                    case 0x3f800000 -> a;
                    case 0x40000000 -> ((FloatVector) a).mul((FloatVector) a);
                    case 0x3f000000 -> ((FloatVector) a).lanewise(VectorOperators.SQRT);
                    case 0xbf800000 -> ONE.div((FloatVector) a);
                    default -> throw new UnsupportedOperationException("Unsupported F32 fused vector pow exponent: " + exponent);
                };
            }
            case NOOP -> a;
            default -> throw new UnsupportedOperationException("Unsupported F32 fused vector op: " + node.opType());
        };
    }

    private static boolean evalCompare(Operation.OpType opType, float a, float b) {
        return switch (opType) {
            case GT -> a > b;
            case GE -> a >= b;
            case LT -> a < b;
            case LE -> a <= b;
            case EQ -> a == b;
            case NE -> a != b;
            default -> throw new IllegalArgumentException("Unsupported compare op: " + opType);
        };
    }

    private static float evalNumeric(FusedNodePlan node, float[] in, FusedExecutionOptions options) {
        FusedNodeAttributes attrs = node.attributes();
        return switch (node.opType()) {
            case ADD -> in[0] + in[1];
            case SUB -> in[0] - in[1];
            case MUL -> in[0] * in[1];
            case DIV -> in[0] / in[1];
            case MIN -> Math.min(in[0], in[1]);
            case MAX -> Math.max(in[0], in[1]);
            case NEG -> -in[0];
            case INV -> 1.0f / in[0];
            case LOG -> (float) Math.log(in[0]);
            case EXP -> options.useFastExpApprox() ? graph.codegen.FusedScalarOps.fastExpF32(in[0]) : graph.codegen.FusedScalarOps.expF32(in[0], false);
            case FAST_EXP -> graph.codegen.FusedScalarOps.fastExpF32(in[0]);
            case TANH -> options.useFastTanhApprox() ? graph.codegen.FusedScalarOps.fastTanhF32(in[0]) : graph.codegen.FusedScalarOps.tanhF32(in[0], false);
            case FAST_TANH -> graph.codegen.FusedScalarOps.fastTanhF32(in[0]);
            case POW -> graph.codegen.FusedScalarOps.powF32(in[0], (float) ((ScalarDoubleAttribute) attrs).value());
            case SQRT -> (float) Math.sqrt(in[0]);
            case ABS -> Math.abs(in[0]);
            case MUL_SCALAR -> in[0] * (float) ((ScalarDoubleAttribute) attrs).value();
            case RELU -> Math.max(in[0], 0.0f);
            case CLAMP_MIN -> Math.max(in[0], (float) ((ScalarDoubleAttribute) attrs).value());
            case CLAMP_MAX -> Math.min(in[0], (float) ((ScalarDoubleAttribute) attrs).value());
            case SIGMOID -> 1.0f / (1.0f + (float) Math.exp(-in[0]));
            case NOOP -> in[0];
            default -> throw new UnsupportedOperationException("Unsupported F32 fused numeric op: " + node.opType());
        };
    }

    private static boolean supportsVectorFastPath(FusedExpressionPlan plan, List<Tensor> inputs, Tensor out) {
        if (out.getDataType() != DataType.FLOAT32 || out.getFloat32Data() == null || !out.isContiguous()) {
            return false;
        }
        if (plan.outputNode().outputType() == DataType.BOOL) {
            return false;
        }
        for (int i = 0; i < plan.inputCount(); i++) {
            FusedExternalInputPlan inputPlan = plan.inputs().get(i);
            Tensor input = inputs.get(i);
            if (!inputPlan.isLinearAccess()) {
                return false;
            }
            if (inputPlan.dataType() == DataType.BOOL) {
                if (input.getDataType() != DataType.BOOL || input.getBoolData() == null) {
                    return false;
                }
                continue;
            }
            if (inputPlan.dataType() != DataType.FLOAT32
                    || input.getDataType() != DataType.FLOAT32
                    || input.getFloat32Data() == null) {
                return false;
            }
        }
        for (FusedNodePlan node : plan.nodes()) {
            if (node.outputType() == DataType.BOOL || !supportsVectorFastPath(node.opType())) {
                return false;
            }
        }
        return true;
    }

    private static boolean supportsVectorFastPath(Operation.OpType opType) {
        return switch (opType) {
            case ADD, SUB, MUL, DIV, MIN, MAX, NEG, INV, LOG, EXP, FAST_EXP, TANH, FAST_TANH,
                    SQRT, ABS, MUL_SCALAR, RELU, CLAMP_MIN, CLAMP_MAX, SIGMOID, POW, NOOP,
                    GT, GE, LT, LE, EQ, NE, LOGICAL_AND, LOGICAL_OR, LOGICAL_NOT, WHERE -> true;
            default -> false;
        };
    }

    private sealed interface InputAccessor permits NumericInputAccessor, BoolInputAccessor {
        static InputAccessor create(FusedExternalInputPlan plan, Tensor tensor, int start) {
            if (plan.dataType() == DataType.BOOL) {
                return new BoolInputAccessor(plan, tensor.getBoolData(), start);
            }
            return new NumericInputAccessor(plan, tensor.getFloat32Data(), start);
        }

        default float floatValue() {
            throw new UnsupportedOperationException("Not a numeric accessor");
        }

        default boolean boolValue() {
            throw new UnsupportedOperationException("Not a bool accessor");
        }

        void step();
    }

    private static final class NumericInputAccessor implements InputAccessor {
        private final float[] data;
        private final boolean linear;
        private final int storageOffset;
        private final FusedBroadcastCursor cursor;
        private int logicalIndex;

        NumericInputAccessor(FusedExternalInputPlan plan, float[] data, int start) {
            this.data = data;
            this.linear = plan.isLinearAccess();
            this.storageOffset = plan.storageOffset();
            this.cursor = linear ? null : FusedBroadcastCursor.atStart(
                    start,
                    plan.logicalOutputShape(),
                    plan.logicalOutputDenseStrides(),
                    plan.effectiveStrides(),
                    plan.storageOffset()
            );
            this.logicalIndex = start;
        }

        @Override
        public float floatValue() {
            int idx = linear ? storageOffset + logicalIndex : cursor.idx();
            return data[idx];
        }

        @Override
        public void step() {
            if (linear) {
                logicalIndex++;
            } else {
                cursor.step();
            }
        }
    }

    private static final class BoolInputAccessor implements InputAccessor {
        private final byte[] data;
        private final boolean linear;
        private final int storageOffset;
        private final FusedBroadcastCursor cursor;
        private int logicalIndex;

        BoolInputAccessor(FusedExternalInputPlan plan, byte[] data, int start) {
            this.data = data;
            this.linear = plan.isLinearAccess();
            this.storageOffset = plan.storageOffset();
            this.cursor = linear ? null : FusedBroadcastCursor.atStart(
                    start,
                    plan.logicalOutputShape(),
                    plan.logicalOutputDenseStrides(),
                    plan.effectiveStrides(),
                    plan.storageOffset()
            );
            this.logicalIndex = start;
        }

        @Override
        public boolean boolValue() {
            int idx = linear ? storageOffset + logicalIndex : cursor.idx();
            return data[idx] != 0;
        }

        @Override
        public void step() {
            if (linear) {
                logicalIndex++;
            } else {
                cursor.step();
            }
        }
    }

    private static final class VectorInputAccessor {
        private final float[] data;
        private final byte[] boolData;
        private final boolean boolInput;
        private final int storageOffset;

        private VectorInputAccessor(float[] data, byte[] boolData, boolean boolInput, int storageOffset) {
            this.data = data;
            this.boolData = boolData;
            this.boolInput = boolInput;
            this.storageOffset = storageOffset;
        }

        static VectorInputAccessor create(FusedExternalInputPlan plan, Tensor tensor) {
            if (!plan.isLinearAccess()) {
                throw new IllegalArgumentException("Vector F32 accessor requires linear input");
            }
            if (plan.dataType() == DataType.BOOL) {
                return new VectorInputAccessor(null, tensor.getBoolData(), true, plan.storageOffset());
            }
            if (plan.dataType() != DataType.FLOAT32) {
                throw new IllegalArgumentException("Vector F32 accessor requires F32/BOOL input");
            }
            return new VectorInputAccessor(tensor.getFloat32Data(), null, false, plan.storageOffset());
        }

        Object load(int logicalIndex) {
            int base = storageOffset + logicalIndex;
            if (boolInput) {
                long bits = 0L;
                for (int lane = 0; lane < SPECIES.length(); lane++) {
                    if (boolData[base + lane] != 0) {
                        bits |= (1L << lane);
                    }
                }
                return VectorMask.fromLong(SPECIES, bits);
            }
            return FloatVector.fromArray(SPECIES, data, base);
        }
    }
}
