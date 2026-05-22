package backend.cpu.kernels.fused;

import tensor.TensorInternalAccess;

import backend.cpu.kernels.*;

import backend.cpu.kernels.fused.FusedExecutionOptions;
import backend.cpu.fused.runtime.FusedBroadcastCursor;
import backend.cpu.fused.ir.FusedExpressionPlan;
import backend.cpu.fused.ir.FusedExternalInputPlan;
import backend.cpu.fused.ir.FusedNodeAttributes;
import backend.cpu.fused.ir.FusedNodePlan;
import backend.cpu.fused.runtime.FusedScalarOps;
import backend.cpu.fused.ir.ScalarDoubleAttribute;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import backend.cpu.fused.plan.FusedOperation;
import operations.Operation;
import tensor.storage.BoolStorage;
import tensor.DataType;
import tensor.storage.Float64Storage;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

public final class Float64FusedExecutor {
    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;
    private static final DoubleVector ZERO = DoubleVector.zero(SPECIES);
    private static final DoubleVector ONE = DoubleVector.broadcast(SPECIES, 1.0d);

    private Float64FusedExecutor() {
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
            throw new IllegalArgumentException("F64 fused execution arguments cannot be null");
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

        double[] values = new double[plan.nodeCount()];
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
                        double a = loadValueRef(plan, current.inputRefs().get(0), accessors, values);
                        double b = loadValueRef(plan, current.inputRefs().get(1), accessors, values);
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
                TensorInternalAccess.boolData(out)[outBaseOffset + index] = bools[outputNode.index()] ? (byte) 1 : (byte) 0;
            } else {
                TensorInternalAccess.float64Data(out)[outBaseOffset + index] = values[outputNode.index()];
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
        double[] outData = TensorInternalAccess.float64Data(out);
        byte[] outBoolData = TensorInternalAccess.boolData(out);
        int outBaseOffset = out.getStorageOffsetUnsafe();
        int width = SPECIES.length();
        int upperBound = endExclusive - ((endExclusive - startInclusive) % width);

        for (int index = startInclusive; index < upperBound; index += width) {
            for (FusedNodePlan current : plan.nodes()) {
                values[current.index()] = evalNumericVector(current, plan, accessors, values, index);
            }
            if (outputNode.outputType() == DataType.BOOL) {
                VectorMask<Double> mask = (VectorMask<Double>) values[outputNode.index()];
                for (int lane = 0; lane < width; lane++) {
                    outBoolData[outBaseOffset + index + lane] = mask.laneIsSet(lane) ? (byte) 1 : (byte) 0;
                }
            } else {
                ((DoubleVector) values[outputNode.index()]).intoArray(outData, outBaseOffset + index);
            }
        }

        if (upperBound < endExclusive) {
            executeRangeScalar(plan, inputs, out, FusedExecutionOptions.exact(), upperBound, endExclusive);
        }
    }

    private static double[] loadInputValues(
            FusedExpressionPlan plan,
            FusedNodePlan current,
            List<InputAccessor> accessors,
            double[] values
    ) {
        List<Integer> refs = current.inputRefs();
        double[] out = new double[refs.size()];
        for (int i = 0; i < refs.size(); i++) {
            out[i] = loadValueRef(plan, refs.get(i), accessors, values);
        }
        return out;
    }

    private static double loadValueRef(
            FusedExpressionPlan plan,
            int ref,
            List<InputAccessor> accessors,
            double[] values
    ) {
        if (ref < plan.inputCount()) {
            return accessors.get(ref).doubleValue();
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
        List<Integer> refs = node.inputRefs();
        Object a = !refs.isEmpty()
                ? loadVectorRef(plan, refs.get(0), accessors, values, logicalIndex)
                : null;
        Object b = refs.size() > 1
                ? loadVectorRef(plan, refs.get(1), accessors, values, logicalIndex)
                : null;
        FusedNodeAttributes attrs = node.attributes();
        return switch (node.opType()) {
            case ADD -> ((DoubleVector) a).add((DoubleVector) b);
            case SUB -> ((DoubleVector) a).sub((DoubleVector) b);
            case MUL -> ((DoubleVector) a).mul((DoubleVector) b);
            case DIV -> ((DoubleVector) a).div((DoubleVector) b);
            case MIN -> ((DoubleVector) a).min((DoubleVector) b);
            case MAX -> ((DoubleVector) a).max((DoubleVector) b);
            case GT -> ((DoubleVector) a).compare(VectorOperators.GT, (DoubleVector) b);
            case GE -> ((DoubleVector) a).compare(VectorOperators.GE, (DoubleVector) b);
            case LT -> ((DoubleVector) a).compare(VectorOperators.LT, (DoubleVector) b);
            case LE -> ((DoubleVector) a).compare(VectorOperators.LE, (DoubleVector) b);
            case EQ -> ((DoubleVector) a).compare(VectorOperators.EQ, (DoubleVector) b);
            case NE -> ((DoubleVector) a).compare(VectorOperators.NE, (DoubleVector) b);
            case LOGICAL_AND -> ((VectorMask<Double>) a).and((VectorMask<Double>) b);
            case LOGICAL_OR -> ((VectorMask<Double>) a).or((VectorMask<Double>) b);
            case LOGICAL_NOT -> ((VectorMask<Double>) a).not();
            case WHERE -> ((DoubleVector) loadVectorRef(plan, node.inputRefs().get(2), accessors, values, logicalIndex)).blend((DoubleVector) b, (VectorMask<Double>) a);
            case NEG -> ((DoubleVector) a).neg();
            case INV -> ONE.div((DoubleVector) a);
            case LOG -> ((DoubleVector) a).lanewise(VectorOperators.LOG);
            case EXP, FAST_EXP -> ((DoubleVector) a).lanewise(VectorOperators.EXP);
            case TANH, FAST_TANH -> ((DoubleVector) a).lanewise(VectorOperators.TANH);
            case SQRT -> ((DoubleVector) a).lanewise(VectorOperators.SQRT);
            case ABS -> ((DoubleVector) a).abs();
            case CONST_SCALAR -> DoubleVector.broadcast(SPECIES, ((ScalarDoubleAttribute) attrs).value());
            case MUL_SCALAR -> ((DoubleVector) a).mul(DoubleVector.broadcast(SPECIES, ((ScalarDoubleAttribute) attrs).value()));
            case RELU -> ((DoubleVector) a).max(ZERO);
            case CLAMP_MIN -> ((DoubleVector) a).max(DoubleVector.broadcast(SPECIES, ((ScalarDoubleAttribute) attrs).value()));
            case CLAMP_MAX -> ((DoubleVector) a).min(DoubleVector.broadcast(SPECIES, ((ScalarDoubleAttribute) attrs).value()));
            case SIGMOID -> {
                DoubleVector half = DoubleVector.broadcast(SPECIES, 0.5d);
                yield ((DoubleVector) a).mul(half).lanewise(VectorOperators.TANH).add(ONE).mul(half);
            }
            case POW -> {
                double exponent = ((ScalarDoubleAttribute) attrs).value();
                if (exponent == 0.0d) yield ONE;
                if (exponent == 1.0d) yield a;
                if (exponent == 2.0d) yield ((DoubleVector) a).mul((DoubleVector) a);
                if (exponent == 0.5d) yield ((DoubleVector) a).lanewise(VectorOperators.SQRT);
                if (exponent == -1.0d) yield ONE.div((DoubleVector) a);
                throw new UnsupportedOperationException("Unsupported F64 fused vector pow exponent: " + exponent);
            }
            case POW_TENSOR -> throw new UnsupportedOperationException("F64 fused vector tensor pow is intentionally disabled.");
            case NOOP -> a;
            default -> throw new UnsupportedOperationException("Unsupported F64 fused vector op: " + node.opType());
        };
    }

    private static boolean evalCompare(Operation.OpType opType, double a, double b) {
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

    private static double evalNumeric(FusedNodePlan node, double[] in, FusedExecutionOptions options) {
        FusedNodeAttributes attrs = node.attributes();
        return switch (node.opType()) {
            case ADD -> in[0] + in[1];
            case SUB -> in[0] - in[1];
            case MUL -> in[0] * in[1];
            case DIV -> in[0] / in[1];
            case MIN -> Math.min(in[0], in[1]);
            case MAX -> Math.max(in[0], in[1]);
            case NEG -> -in[0];
            case INV -> 1.0d / in[0];
            case LOG -> Math.log(in[0]);
            case EXP -> options.useFastExpApprox() ? backend.cpu.fused.runtime.FusedScalarOps.fastExpF64(in[0]) : backend.cpu.fused.runtime.FusedScalarOps.expF64(in[0], false);
            case FAST_EXP -> backend.cpu.fused.runtime.FusedScalarOps.fastExpF64(in[0]);
            case TANH -> options.useFastTanhApprox() ? backend.cpu.fused.runtime.FusedScalarOps.fastTanhF64(in[0]) : backend.cpu.fused.runtime.FusedScalarOps.tanhF64(in[0], false);
            case FAST_TANH -> backend.cpu.fused.runtime.FusedScalarOps.fastTanhF64(in[0]);
            case POW -> FusedScalarOps.powF64(in[0], ((ScalarDoubleAttribute) attrs).value());
            case POW_TENSOR -> FusedScalarOps.powF64(in[0], in[1]);
            case SQRT -> Math.sqrt(in[0]);
            case ABS -> Math.abs(in[0]);
            case CONST_SCALAR -> ((ScalarDoubleAttribute) attrs).value();
            case MUL_SCALAR -> in[0] * ((ScalarDoubleAttribute) attrs).value();
            case RELU -> Math.max(in[0], 0.0d);
            case CLAMP_MIN -> Math.max(in[0], ((ScalarDoubleAttribute) attrs).value());
            case CLAMP_MAX -> Math.min(in[0], ((ScalarDoubleAttribute) attrs).value());
            case SIGMOID -> 1.0d / (1.0d + Math.exp(-in[0]));
            case NOOP -> in[0];
            default -> throw new UnsupportedOperationException("Unsupported F64 fused numeric op: " + node.opType());
        };
    }

    private static boolean supportsVectorFastPath(FusedExpressionPlan plan, List<Tensor> inputs, Tensor out) {
        if (out.getDataType() != DataType.FLOAT64 || !(TensorInternalAccess.storage(out) instanceof Float64Storage) || !out.isContiguous()) {
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
                if (input.getDataType() != DataType.BOOL || !(TensorInternalAccess.storage(input) instanceof BoolStorage)) {
                    return false;
                }
                continue;
            }
            if (inputPlan.dataType() != DataType.FLOAT64
                    || input.getDataType() != DataType.FLOAT64
                    || !(TensorInternalAccess.storage(input) instanceof Float64Storage)) {
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
                    SQRT, ABS, CONST_SCALAR, MUL_SCALAR, RELU, CLAMP_MIN, CLAMP_MAX, SIGMOID, POW, NOOP,
                    GT, GE, LT, LE, EQ, NE, LOGICAL_AND, LOGICAL_OR, LOGICAL_NOT, WHERE -> true;
            case POW_TENSOR -> false;
            default -> false;
        };
    }

    private sealed interface InputAccessor permits NumericInputAccessor, BoolInputAccessor {
        static InputAccessor create(FusedExternalInputPlan plan, Tensor tensor, int start) {
            if (plan.dataType() == DataType.BOOL) {
                return new BoolInputAccessor(plan, TensorInternalAccess.boolData(tensor), start);
            }
            return new NumericInputAccessor(plan, TensorInternalAccess.float64Data(tensor), start);
        }

        default double doubleValue() {
            throw new UnsupportedOperationException("Not a numeric accessor");
        }

        default boolean boolValue() {
            throw new UnsupportedOperationException("Not a bool accessor");
        }

        void step();
    }

    private static final class NumericInputAccessor implements InputAccessor {
        private final double[] data;
        private final boolean linear;
        private final int storageOffset;
        private final FusedBroadcastCursor cursor;
        private int logicalIndex;

        NumericInputAccessor(FusedExternalInputPlan plan, double[] data, int start) {
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
        public double doubleValue() {
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
        private final double[] data;
        private final byte[] boolData;
        private final boolean boolInput;
        private final int storageOffset;

        private VectorInputAccessor(double[] data, byte[] boolData, boolean boolInput, int storageOffset) {
            this.data = data;
            this.boolData = boolData;
            this.boolInput = boolInput;
            this.storageOffset = storageOffset;
        }

        static VectorInputAccessor create(FusedExternalInputPlan plan, Tensor tensor) {
            if (!plan.isLinearAccess()) {
                throw new IllegalArgumentException("Vector F64 accessor requires linear input");
            }
            if (plan.dataType() == DataType.BOOL) {
                return new VectorInputAccessor(null, TensorInternalAccess.boolData(tensor), true, plan.storageOffset());
            }
            if (plan.dataType() != DataType.FLOAT64) {
                throw new IllegalArgumentException("Vector F64 accessor requires F64/BOOL input");
            }
            return new VectorInputAccessor(TensorInternalAccess.float64Data(tensor), null, false, plan.storageOffset());
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
            return DoubleVector.fromArray(SPECIES, data, base);
        }
    }
}
