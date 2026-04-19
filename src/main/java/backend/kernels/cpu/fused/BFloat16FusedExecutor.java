package backend.kernels.cpu.fused;

import backend.kernels.cpu.*;
import backend.kernels.cpu.elementwise.plan.ResolvedDispatchHints;

import backend.kernels.cpu.fused.FusedExecutionOptions;
import graph.codegen.FusedExpressionPlan;
import graph.codegen.FusedExternalInputPlan;
import graph.codegen.FusedNodeAttributes;
import graph.codegen.FusedNodePlan;
import graph.codegen.FusedBroadcastCursor;
import graph.codegen.ScalarDoubleAttribute;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import operations.fused.FusedOperation;
import operations.Operation;
import tensor.DataType;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

public final class BFloat16FusedExecutor {
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final FloatVector ZERO = FloatVector.zero(SPECIES);
    private static final FloatVector ONE = FloatVector.broadcast(SPECIES, 1.0f);

    private BFloat16FusedExecutor() {
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
        executeRangeScalar(fused.getPlan(), inputs, node, context, options, startInclusive, endExclusive);
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
            executeRangeVector(plan, inputs, node, context, startInclusive, endExclusive);
            return;
        }
        executeRangeScalar(plan, inputs, node, context, options, startInclusive, endExclusive);
    }

    private static void validateArgs(
            FusedOperation fused,
            List<Tensor> inputs,
            Tensor node,
            CpuKernelContext context,
            FusedExecutionOptions options
    ) {
        if (fused == null || inputs == null || node == null || context == null || options == null) {
            throw new IllegalArgumentException("BF16 fused execution arguments cannot be null");
        }
    }

    private static void executeRangeScalar(
            FusedExpressionPlan plan,
            List<Tensor> inputs,
            Tensor out,
            CpuKernelContext context,
            FusedExecutionOptions options,
            int startInclusive,
            int endExclusive
    ) {
        List<InputAccessor> accessors = new ArrayList<>(plan.inputCount());
        for (int i = 0; i < plan.inputCount(); i++) {
            accessors.add(InputAccessor.create(plan.inputs().get(i), inputs.get(i), context.inputFloatContinuation(i, out.getFlatDataSize()), startInclusive));
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
                out.getBFloat16Data()[outBaseOffset + index] = CpuDTypeOps.toBFloat16Bits(values[outputNode.index()]);
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
            CpuKernelContext context,
            int startInclusive,
            int endExclusive
    ) {
        List<VectorInputAccessor> accessors = new ArrayList<>(plan.inputCount());
        for (int i = 0; i < plan.inputCount(); i++) {
            accessors.add(VectorInputAccessor.create(plan.inputs().get(i), inputs.get(i), context.inputFloatContinuation(i, out.getFlatDataSize())));
        }

        FloatVector[] values = new FloatVector[plan.nodeCount()];
        FusedNodePlan outputNode = plan.outputNode();
        short[] outData = out.getBFloat16Data();
        int outBaseOffset = out.getStorageOffsetUnsafe();
        int width = SPECIES.length();
        int upperBound = endExclusive - ((endExclusive - startInclusive) % width);
        float[] outScratch = new float[width];

        for (int index = startInclusive; index < upperBound; index += width) {
            for (FusedNodePlan current : plan.nodes()) {
                values[current.index()] = evalNumericVector(current, plan, accessors, values, index);
            }
            storeVector(outData, outBaseOffset + index, values[outputNode.index()], outScratch);
        }

        if (upperBound < endExclusive) {
            executeRangeScalar(plan, inputs, out, context, FusedExecutionOptions.exact(), upperBound, endExclusive);
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

    private static FloatVector loadVectorRef(
            FusedExpressionPlan plan,
            int ref,
            List<VectorInputAccessor> accessors,
            FloatVector[] values,
            int logicalIndex
    ) {
        if (ref < plan.inputCount()) {
            return accessors.get(ref).load(logicalIndex);
        }
        return values[ref - plan.inputCount()];
    }

    private static FloatVector evalNumericVector(
            FusedNodePlan node,
            FusedExpressionPlan plan,
            List<VectorInputAccessor> accessors,
            FloatVector[] values,
            int logicalIndex
    ) {
        List<Integer> refs = node.inputRefs();
        FloatVector a = !refs.isEmpty()
                ? loadVectorRef(plan, refs.get(0), accessors, values, logicalIndex)
                : null;
        FloatVector b = refs.size() > 1
                ? loadVectorRef(plan, refs.get(1), accessors, values, logicalIndex)
                : null;
        FusedNodeAttributes attrs = node.attributes();
        return switch (node.opType()) {
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
            case CONST_SCALAR -> FloatVector.broadcast(SPECIES, (float) ((ScalarDoubleAttribute) attrs).value());
            case MUL_SCALAR -> a.mul(FloatVector.broadcast(SPECIES, (float) ((ScalarDoubleAttribute) attrs).value()));
            case RELU -> a.max(ZERO);
            case CLAMP_MIN -> a.max(FloatVector.broadcast(SPECIES, (float) ((ScalarDoubleAttribute) attrs).value()));
            case CLAMP_MAX -> a.min(FloatVector.broadcast(SPECIES, (float) ((ScalarDoubleAttribute) attrs).value()));
            case SIGMOID -> {
                FloatVector half = FloatVector.broadcast(SPECIES, 0.5f);
                yield a.mul(half).lanewise(VectorOperators.TANH).add(ONE).mul(half);
            }
            case POW -> {
                float exponent = (float) ((ScalarDoubleAttribute) attrs).value();
                yield switch (Float.floatToIntBits(exponent)) {
                    case 0x00000000 -> ONE;
                    case 0x3f800000 -> a;
                    case 0x40000000 -> a.mul(a);
                    case 0x3f000000 -> a.lanewise(VectorOperators.SQRT);
                    case 0xbf800000 -> ONE.div(a);
                    default -> throw new UnsupportedOperationException("Unsupported BF16 fused vector pow exponent: " + exponent);
                };
            }
            case NOOP -> a;
            default -> throw new UnsupportedOperationException("Unsupported BF16 fused vector op: " + node.opType());
        };
    }

    private static void storeVector(short[] out, int baseIndex, FloatVector value, float[] scratch) {
        value.intoArray(scratch, 0);
        for (int lane = 0; lane < SPECIES.length(); lane++) {
            out[baseIndex + lane] = CpuDTypeOps.toBFloat16Bits(scratch[lane]);
        }
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
            case CONST_SCALAR -> (float) ((ScalarDoubleAttribute) attrs).value();
            case MUL_SCALAR -> in[0] * (float) ((ScalarDoubleAttribute) attrs).value();
            case RELU -> Math.max(in[0], 0.0f);
            case CLAMP_MIN -> Math.max(in[0], (float) ((ScalarDoubleAttribute) attrs).value());
            case CLAMP_MAX -> Math.min(in[0], (float) ((ScalarDoubleAttribute) attrs).value());
            case SIGMOID -> 1.0f / (1.0f + (float) Math.exp(-in[0]));
            case NOOP -> in[0];
            default -> throw new UnsupportedOperationException("Unsupported BF16 fused numeric op: " + node.opType());
        };
    }

    private static boolean supportsVectorFastPath(FusedExpressionPlan plan, List<Tensor> inputs, Tensor out) {
        if (out.getDataType() != DataType.BFLOAT16 || out.getBFloat16Data() == null || !out.isContiguous()) {
            return false;
        }
        if (plan.outputNode().outputType() == DataType.BOOL) {
            return false;
        }
        for (int i = 0; i < plan.inputCount(); i++) {
            FusedExternalInputPlan inputPlan = plan.inputs().get(i);
            Tensor input = inputs.get(i);
            if (inputPlan.dataType() != DataType.BFLOAT16
                    || !inputPlan.isLinearAccess()
                    || input.getDataType() != DataType.BFLOAT16
                    || input.getBFloat16Data() == null) {
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
                    SQRT, ABS, CONST_SCALAR, MUL_SCALAR, RELU, CLAMP_MIN, CLAMP_MAX, SIGMOID, POW, NOOP -> true;
            default -> false;
        };
    }

    private static void recordProfile(
            FusedOperation fused,
            ResolvedDispatchHints hints,
            boolean vectorFastPath,
            int length,
            int chunkSize,
            long startNs
    ) {
        if (!FusedExecutionProfiler.enabled()) {
            return;
        }
        int chunks = Math.max(1, (length + chunkSize - 1) / chunkSize);
        FusedExecutionProfiler.recordRun(
                fused.getSchedulerSignature(),
                hints.mode(),
                length,
                chunks,
                false,
                vectorFastPath,
                System.nanoTime() - startNs
        );
    }

    private sealed interface InputAccessor permits NumericInputAccessor, BoolInputAccessor, ContinuationInputAccessor {
        static InputAccessor create(FusedExternalInputPlan plan, Tensor tensor, float[] continuation, int start) {
            if (plan.dataType() == DataType.BOOL) {
                return new BoolInputAccessor(plan, tensor.getBoolData(), start);
            }
            if (continuation != null && plan.dataType() == DataType.BFLOAT16 && plan.isLinearAccess()) {
                return new ContinuationInputAccessor(plan, continuation, start);
            }
            return new NumericInputAccessor(plan, tensor.getBFloat16Data(), start);
        }

        default float floatValue() {
            throw new UnsupportedOperationException("Not a numeric accessor");
        }

        default boolean boolValue() {
            throw new UnsupportedOperationException("Not a bool accessor");
        }

        void step();
    }

    private static final class ContinuationInputAccessor implements InputAccessor {
        private final float[] data;
        private final int storageOffset;
        private int logicalIndex;

        ContinuationInputAccessor(FusedExternalInputPlan plan, float[] data, int start) {
            this.data = data;
            this.storageOffset = plan.storageOffset();
            this.logicalIndex = start;
        }

        @Override
        public float floatValue() {
            return data[storageOffset + logicalIndex];
        }

        @Override
        public void step() {
            logicalIndex++;
        }
    }

    private static final class NumericInputAccessor implements InputAccessor {
        private final short[] data;
        private final boolean linear;
        private final int storageOffset;
        private final FusedBroadcastCursor cursor;
        private int logicalIndex;

        NumericInputAccessor(FusedExternalInputPlan plan, short[] data, int start) {
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
            return CpuDTypeOps.fromBFloat16Bits(data[idx]);
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
        private final short[] data;
        private final float[] continuation;
        private final int storageOffset;
        private final float[] scratch;

        private VectorInputAccessor(short[] data, float[] continuation, int storageOffset) {
            this.data = data;
            this.continuation = continuation;
            this.storageOffset = storageOffset;
            this.scratch = new float[SPECIES.length()];
        }

        static VectorInputAccessor create(FusedExternalInputPlan plan, Tensor tensor, float[] continuation) {
            if (plan.dataType() != DataType.BFLOAT16 || !plan.isLinearAccess()) {
                throw new IllegalArgumentException("Vector BF16 accessor requires linear BF16 input");
            }
            return new VectorInputAccessor(tensor.getBFloat16Data(), continuation, plan.storageOffset());
        }

        FloatVector load(int logicalIndex) {
            int base = storageOffset + logicalIndex;
            if (continuation != null) {
                return FloatVector.fromArray(SPECIES, continuation, base);
            }
            for (int lane = 0; lane < SPECIES.length(); lane++) {
                scratch[lane] = CpuDTypeOps.fromBFloat16Bits(data[base + lane]);
            }
            return FloatVector.fromArray(SPECIES, scratch, 0);
        }
    }
}
