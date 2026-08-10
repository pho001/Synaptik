package io.github.pho001.synaptik.backend.cpu.internal.reference;

import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBufferArgument;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.List;
import java.util.Objects;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAffineCopyIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcode;
import io.github.pho001.synaptik.model.datatype.DataType;

/**
 * Scalar conformance realization for the bounded typed CPU pointwise semantics.
 * It evaluates already-lowered primitive arithmetic, exact extrema and clamp, direct Tensor
 * power, canonical-BOOL logic, the closed FLOAT32/FLOAT64 unary matrix, and the selected
 * scalar-power plan. Direct power uses {@link StrictMath#pow(double, double)} without
 * reclassifying an exponent. Unary evaluation preserves the specified exceptional-value
 * classifications, widens represented FLOAT32 values where required, and narrows once. It is an
 * unsupported cold-test/reference contract and is never a Runtime IR interpreter.
 */
public final class CpuScalarReferenceKernel {
    private static final ValueLayout.OfDouble DOUBLE = ValueLayout.JAVA_DOUBLE_UNALIGNED
            .withOrder(ByteOrder.nativeOrder());
    // Cephes erf/erfc rational coefficients, documented at
    // https://netlib.org/cephes/doubldoc.html and published in netlib cephes ndtr.c.
    private static final double[] ERF_T = {9.60497373987051638749E0,
            9.00260197203842689217E1, 2.23200534594684319226E3,
            7.00332514112805075473E3, 5.55923013010394962768E4};
    private static final double[] ERF_U = {3.35617141647503099647E1,
            5.21357949780152679795E2, 4.59432382970980127987E3,
            2.26290000613890934246E4, 4.92673942608635921086E4};
    private static final double[] ERFC_P = {2.46196981473530512524E-10,
            5.64189564831068821977E-1, 7.46321056442269912687E0,
            4.86371970985681366614E1, 1.96520832956077098242E2,
            5.26445194995477358631E2, 9.34528527171957607540E2,
            1.02755188689515710272E3, 5.57535335369399327526E2};
    private static final double[] ERFC_Q = {1.32281951154744992508E1,
            8.67072140885989742329E1, 3.54937778887819891062E2,
            9.75708501743205489753E2, 1.82390916687909736289E3,
            2.24633760818710981792E3, 1.65666309194161350182E3,
            5.57535340817727675546E2};
    private static final double[] ERFC_R = {5.64189583547755073984E-1,
            1.27536670759978104416E0, 5.01905042251180477414E0,
            6.16021097993053585195E0, 7.40974269950448939160E0,
            2.97886665372100240670E0};
    private static final double[] ERFC_S = {2.26052863220117276590E0,
            9.39603524938001434673E0, 1.20489539808096656605E1,
            1.70814450747565897222E1, 9.60896809063285878198E0,
            3.36907645100081516050E0};
    private CpuScalarReferenceKernel() { }

    /**
     * Evaluates the portable exact/default GELU target in fixed operation order.
     *
     * @param value input value, including IEEE 754 special values
     * @return {@code 0.5 * value * (1 + erf(value / sqrt(2)))} using the shared bounded error-
     *     function approximation, with negative infinity mapped to negative zero and other
     *     documented special-value classifications preserved
     */
    public static double gelu(double value) {
        if (value == Double.NEGATIVE_INFINITY) return -0.0d;
        return 0.5d * value * (1.0d + erf(value / Math.sqrt(2.0d)));
    }

    /**
     * Evaluates logistic sigmoid without avoidable exponential overflow.
     * @param value input value, including IEEE 754 special values
     * @return the stable two-branch sigmoid result; NaN remains NaN
     */
    public static double sigmoid(double value) {
        if (value >= 0.0d) return 1.0d / (1.0d + StrictMath.exp(-value));
        double exponential = StrictMath.exp(value);
        return exponential / (1.0d + exponential);
    }

    /**
     * Evaluates the fixed Model hyperbolic-tangent GELU approximation.
     * @param value input value, including IEEE 754 special values
     * @return the fixed-coefficient approximation, with negative infinity mapped to negative zero
     */
    public static double geluTanhApproximation(double value) {
        if (value == Double.NEGATIVE_INFINITY) return -0.0d;
        double cube = value * value * value;
        return 0.5d * value * (1.0d + StrictMath.tanh(Math.sqrt(2.0d / Math.PI)
                * (value + 0.044715d * cube)));
    }

    /**
     * Evaluates sigmoid linear unit without avoidable exponential overflow.
     * @param value input value, including IEEE 754 special values
     * @return {@code value * sigmoid(value)}, with negative infinity mapped to negative zero
     */
    public static double silu(double value) {
        if (value == Double.NEGATIVE_INFINITY) return -0.0d;
        if (value >= 0.0d) return value / (1.0d + StrictMath.exp(-value));
        double exponential = StrictMath.exp(value);
        return value * exponential / (1.0d + exponential);
    }

    /**
     * Evaluates a portable scalar approximation of the Gaussian error function.
     * The approximation is shared by generated and reference realizations and preserves NaN,
     * infinities, and signed zero classifications.
     *
     * @param value input value, including IEEE 754 special values
     * @return the finite approximation to the Gaussian error function, or the corresponding
     *     preserved NaN, infinity, or signed-zero classification
     */
    public static double erf(double value) {
        if (Double.isNaN(value)) return Double.NaN;
        if (value == 0.0d) return value;
        if (value == Double.POSITIVE_INFINITY) return 1.0d;
        if (value == Double.NEGATIVE_INFINITY) return -1.0d;
        double x = Math.abs(value);
        double result;
        if (x <= 1.0d) {
            double z = x * x;
            result = x * polevl(z, ERF_T) / p1evl(z, ERF_U);
        } else {
            double erfc = Math.exp(-x * x) * (x < 8.0d
                    ? polevl(x, ERFC_P) / p1evl(x, ERFC_Q)
                    : polevl(x, ERFC_R) / p1evl(x, ERFC_S));
            result = 1.0d - erfc;
        }
        return Math.copySign(result, value);
    }

    /**
     * Executes the fused reference calculation over one half-open range.
     *
     * @param a non-null first ADD input; not mutated
     * @param b non-null second ADD input; not mutated
     * @param c non-null MUL input; not mutated
     * @param output non-null destination mutated only in {@code [start, end)}
     * @param start non-negative inclusive element index
     * @param end exclusive element index no greater than any array length
     * @throws NullPointerException if an array is {@code null}
     * @throws IllegalArgumentException if the half-open range is negative, reversed, or exceeds an
     *     input or output array
     */
    public static void execute(double[] a, double[] b, double[] c, double[] output,
            long start, long end) {
        if (start < 0 || end < start || end > a.length || end > b.length || end > c.length
                || end > output.length) throw new IllegalArgumentException("invalid reference bounds");
        for (long index = start; index < end; index++) {
            double sum = a[(int) index] + b[(int) index];
            double activated = gelu(sum);
            output[(int) index] = activated * c[(int) index];
        }
    }

    /**
     * Executes the completed four-boundary FLOAT64 proving topology over the same normalized
     * bindings and direct carrier forms as generated scalar code.
     * This reference path may allocate coordinate arrays and use division/modulo because it is
     * conformance support, not the generated Runtime hot path.
     *
     * @param arguments non-null ordered direct inputs {@code a}, {@code b}, {@code c}, and output
     * @param bindings non-null matching normalized access bindings in the same order
     * @param start non-negative inclusive logical element bound
     * @param end exclusive logical element bound no greater than the first binding's count
     * @throws NullPointerException if an argument, binding, or list is {@code null}
     * @throws IllegalArgumentException if boundary counts or range are invalid
     * @throws ArithmeticException if exact address arithmetic overflows
     */
    public static void execute(List<CpuBufferArgument> arguments,
            List<CpuAccessPlan.Binding> bindings, long start, long end) {
        if (arguments.size() != 4 || bindings.size() != 4) throw new IllegalArgumentException(
                "reference execution requires four ordered boundaries");
        CpuAccessPlan.Binding first = bindings.getFirst();
        if (start < 0 || end < start || end > first.elementCount()) {
            throw new IllegalArgumentException("invalid reference bounds");
        }
        long[] extents = first.extents().stream().mapToLong(Long::longValue).toArray();
        for (long index = start; index < end; index++) {
            long[] coordinate = coordinates(index, extents);
            double sum = load(arguments.get(0), address(bindings.get(0), coordinate))
                    + load(arguments.get(1), address(bindings.get(1), coordinate));
            double result = gelu(sum)
                    * load(arguments.get(2), address(bindings.get(2), coordinate));
            store(arguments.get(3), address(bindings.get(3), coordinate), result);
        }
    }

    /**
     * Executes one already-lowered typed pointwise IR for differential conformance.
     *
     * @param ir non-null typed CPU pointwise IR
     * @param arguments non-null materialized boundary arguments in IR boundary order
     * @param bindings non-null normalized boundary bindings in the same order
     * @param start non-negative inclusive logical bound
     * @param end exclusive logical bound
     * @throws NullPointerException if {@code ir}, a list, argument, or binding is {@code null}
     * @throws IllegalArgumentException if boundary counts or the half-open range are invalid
     * @throws ArithmeticException if exact coordinate or address arithmetic overflows
     */
    public static void execute(CpuKernelIr ir, List<CpuBufferArgument> arguments,
            List<CpuAccessPlan.Binding> bindings, long start, long end) {
        List<CpuKernelIr.Value> boundaries = ir.values().stream()
                .filter(value -> value.kind() != CpuKernelIr.Value.Kind.VIRTUAL).toList();
        if (arguments.size() != boundaries.size() || bindings.size() != boundaries.size()
                || start < 0 || end < start || end > bindings.getFirst().elementCount()) {
            throw new IllegalArgumentException("invalid typed reference boundaries or range");
        }
        long[] extents = bindings.getFirst().extents().stream().mapToLong(Long::longValue).toArray();
        Object[] values = new Object[ir.values().size()];
        for (long index = start; index < end; index++) {
            long[] coordinate = coordinates(index, extents);
            for (int boundary = 0; boundary < boundaries.size(); boundary++) {
                CpuKernelIr.Value value = boundaries.get(boundary);
                if (value.kind() == CpuKernelIr.Value.Kind.INPUT
                        && requiresInputLoad(ir, value.ordinal())) values[value.ordinal()] =
                        load(arguments.get(boundary), value.dataType(),
                                address(bindings.get(boundary), coordinate));
            }
            for (CpuKernelIr.Instruction instruction : ir.instructions()) {
                values[instruction.output()] = evaluate(ir, instruction, values);
            }
            for (CpuKernelIr.Store store : ir.stores()) {
                CpuKernelIr.Value value = ir.values().get(store.value());
                int boundary = 0;
                while (boundaries.get(boundary).ordinal() != value.ordinal()) boundary++;
                store(arguments.get(boundary), value.dataType(),
                        address(bindings.get(boundary), coordinate), values[value.ordinal()]);
            }
        }
    }

    /**
     * Executes one cold-composed represented-bit affine address sequence for differential tests.
     *
     * @param ir non-null affine copy contract supplying the represented data type
     * @param addressPairs non-null alternating source and result element addresses
     * @param arguments non-null ordered source and writable result arguments
     * @param start non-negative inclusive address-pair index
     * @param end exclusive address-pair index no greater than the available pair count
     * @throws NullPointerException if {@code ir} or {@code addressPairs} is {@code null}
     * @throws IllegalArgumentException if the argument count, pair table, or range is invalid
     * @throws ArithmeticException if a requested pair index cannot be represented safely
     */
    public static void execute(CpuAffineCopyIr ir, long[] addressPairs,
            List<CpuBufferArgument> arguments, long start, long end) {
        Objects.requireNonNull(ir, "ir");
        Objects.requireNonNull(addressPairs, "addressPairs");
        if (arguments.size() != 2 || addressPairs.length % 2 != 0 || start < 0 || end < start
                || end > addressPairs.length / 2) {
            throw new IllegalArgumentException("invalid affine reference boundaries or range");
        }
        for (long index = start; index < end; index++) {
            int pair = Math.toIntExact(Math.multiplyExact(index, 2));
            Object represented = load(arguments.get(0), ir.dataType(), addressPairs[pair]);
            store(arguments.get(1), ir.dataType(), addressPairs[pair + 1], represented);
        }
    }

    private static Object evaluate(CpuKernelIr ir, CpuKernelIr.Instruction instruction,
            Object[] values) {
        DataType type = ir.values().get(instruction.inputs().getFirst()).dataType();
        Object left = values[instruction.inputs().getFirst()];
        Object right = instruction.inputs().size() > 1 ? values[instruction.inputs().get(1)] : null;
        Object scalar = instruction.scalarImmediate() == null ? null
                : immediate(instruction.scalarImmediate());
        return switch (instruction.opcode()) {
            case ADD -> arithmetic(type, left, right, 0);
            case SUB -> arithmetic(type, left, right, 1);
            case MUL -> arithmetic(type, left, right, 2);
            case DIV -> arithmetic(type, left, right, 3);
            case MIN -> extrema(type, left, right, true);
            case MAX -> extrema(type, left, right, false);
            case POW -> tensorPower(type, left, right);
            case SCALAR_ADD -> arithmetic(type, left, scalar, 0);
            case SCALAR_SUB -> arithmetic(type, left, scalar, 1);
            case SCALAR_MUL -> arithmetic(type, left, scalar, 2);
            case SCALAR_DIV -> arithmetic(type, left, scalar, 3);
            case SCALAR_POW -> power(type, left, instruction.scalarImmediate(),
                    instruction.powerRealization());
            case SCALAR_MIN -> extrema(type, left, scalar, true);
            case SCALAR_MAX -> extrema(type, left, scalar, false);
            case SCALAR_CLAMP -> extrema(type,
                    extrema(type, left, immediate(instruction.clampImmediate().lower()), false),
                    immediate(instruction.clampImmediate().upper()), true);
            case NEG -> { if (type == DataType.FLOAT64) yield Double.valueOf(-(double) left);
                yield Float.valueOf(-(float) left); }
            case ABS, RECIPROCAL, LOG, LOG1P, EXP, EXPM1, ERF, SQRT, RSQRT, FLOOR, CEIL,
                    SIGN, RELU, SIGMOID, TANH, GELU_EXACT, GELU_TANH_APPROXIMATION, SILU ->
                    unary(instruction.opcode(), type, left);
            case IS_FINITE -> (byte) ((type == DataType.FLOAT64
                    ? Double.isFinite((double) left) : Float.isFinite((float) left)) ? 1 : 0);
            case IS_NAN -> (byte) ((type == DataType.FLOAT64
                    ? Double.isNaN((double) left) : Float.isNaN((float) left)) ? 1 : 0);
            case IS_INF -> (byte) ((type == DataType.FLOAT64
                    ? Double.isInfinite((double) left) : Float.isInfinite((float) left)) ? 1 : 0);
            case GREATER_THAN, GREATER_OR_EQUAL, LESS_THAN, LESS_OR_EQUAL ->
                    bool(relation(instruction.opcode(), type, left, right));
            case EQUAL -> bool(equal(type, left, right));
            case NOT_EQUAL -> bool(!equal(type, left, right));
            case LOGICAL_AND -> bool((byte) left == 1 && (byte) right == 1);
            case LOGICAL_OR -> bool((byte) left == 1 || (byte) right == 1);
            case LOGICAL_NOT -> bool((byte) left == 0);
            case WHERE -> ((byte) left) == 1 ? values[instruction.inputs().get(1)]
                    : values[instruction.inputs().get(2)];
            case CAST -> left;
        };
    }

    private static Object unary(CpuPointwiseOpcode opcode, DataType type, Object input) {
        double value = type == DataType.FLOAT64 ? (double) input : (double) (float) input;
        double result = switch (opcode) {
            case ABS -> Math.abs(value); case RECIPROCAL -> 1.0d / value;
            case LOG -> StrictMath.log(value); case LOG1P -> StrictMath.log1p(value);
            case EXP -> StrictMath.exp(value); case EXPM1 -> StrictMath.expm1(value);
            case ERF -> erf(value); case SQRT -> StrictMath.sqrt(value);
            case RSQRT -> 1.0d / StrictMath.sqrt(value); case FLOOR -> StrictMath.floor(value);
            case CEIL -> StrictMath.ceil(value); case SIGN -> Math.signum(value);
            case RELU -> Math.max(value, +0.0d); case SIGMOID -> sigmoid(value);
            case TANH -> StrictMath.tanh(value); case GELU_EXACT -> gelu(value);
            case GELU_TANH_APPROXIMATION -> geluTanhApproximation(value); case SILU -> silu(value);
            default -> throw new AssertionError(opcode);
        };
        if (type == DataType.FLOAT64) return Double.valueOf(result);
        return Float.valueOf((float) result);
    }

    private static Object tensorPower(DataType type, Object base, Object exponent) {
        if (type == DataType.FLOAT64) return Double.valueOf(
                StrictMath.pow((double) base, (double) exponent));
        return Float.valueOf((float) StrictMath.pow((double) (float) base,
                (double) (float) exponent));
    }

    private static Object extrema(DataType type, Object left, Object right, boolean minimum) {
        return switch (type) {
            case FLOAT64 -> minimum ? Math.min((double) left, (double) right)
                    : Math.max((double) left, (double) right);
            case FLOAT32 -> minimum ? Math.min((float) left, (float) right)
                    : Math.max((float) left, (float) right);
            case INT32 -> minimum ? Math.min((int) left, (int) right)
                    : Math.max((int) left, (int) right);
            case INT64 -> minimum ? Math.min((long) left, (long) right)
                    : Math.max((long) left, (long) right);
            default -> throw new IllegalArgumentException("unsupported extrema type");
        };
    }

    private static Object arithmetic(DataType type, Object left, Object right, int operation) {
        return switch (type) {
            case FLOAT64 -> { double a = (double) left, b = (double) right;
                yield operation == 0 ? a + b : operation == 1 ? a - b
                        : operation == 2 ? a * b : a / b; }
            case FLOAT32 -> { float a = (float) left, b = (float) right;
                yield operation == 0 ? a + b : operation == 1 ? a - b
                        : operation == 2 ? a * b : a / b; }
            case INT32 -> { int a = (int) left, b = (int) right;
                if (operation == 3) throw new IllegalArgumentException("integral division unsupported");
                yield operation == 0 ? a + b : operation == 1 ? a - b : a * b; }
            case INT64 -> { long a = (long) left, b = (long) right;
                if (operation == 3) throw new IllegalArgumentException("integral division unsupported");
                yield operation == 0 ? a + b : operation == 1 ? a - b : a * b; }
            default -> throw new IllegalArgumentException("unsupported arithmetic type");
        };
    }

    private static Object power(DataType type, Object base, CpuKernelIr.ScalarImmediate exponent,
            CpuKernelIr.PowerRealization realization) {
        if (type == DataType.FLOAT64) {
            double value = base == null ? Double.NaN : (double) base;
            return switch (realization) {
                case DIRECT -> StrictMath.pow(value, Double.longBitsToDouble(exponent.bits()));
                case POSITIVE_ONE -> 1.0d;
                case IDENTITY -> value;
                case SQUARE -> value * value;
                case RECIPROCAL -> 1.0d / value;
            };
        }
        float value = base == null ? Float.NaN : (float) base;
        return switch (realization) {
            case DIRECT -> (float) StrictMath.pow((double) value,
                    (double) Float.intBitsToFloat((int) exponent.bits()));
            case POSITIVE_ONE -> 1.0f;
            case IDENTITY -> value;
            case SQUARE -> value * value;
            case RECIPROCAL -> 1.0f / value;
        };
    }

    private static boolean requiresInputLoad(CpuKernelIr ir, int ordinal) {
        return ir.instructions().stream().anyMatch(instruction -> {
            for (int input = 0; input < instruction.inputs().size(); input++) {
                if (instruction.inputs().get(input) != ordinal) continue;
                if (input == 0 && instruction.opcode()
                        == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcode.SCALAR_POW
                        && instruction.powerRealization()
                            == CpuKernelIr.PowerRealization.POSITIVE_ONE) continue;
                return true;
            }
            return false;
        });
    }

    private static boolean relation(io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcode opcode,
            DataType type, Object left, Object right) {
        int relation = switch (type) {
            case FLOAT64 -> (double) left > (double) right ? 1 : (double) left < (double) right ? -1
                    : (double) left == (double) right ? 0 : 2;
            case FLOAT32 -> (float) left > (float) right ? 1 : (float) left < (float) right ? -1
                    : (float) left == (float) right ? 0 : 2;
            case INT32 -> Integer.compare((int) left, (int) right);
            case INT64 -> Long.compare((long) left, (long) right);
            default -> throw new IllegalArgumentException("unsupported comparison type");
        };
        return switch (opcode) {
            case GREATER_THAN -> relation == 1; case GREATER_OR_EQUAL -> relation == 1 || relation == 0;
            case LESS_THAN -> relation == -1; case LESS_OR_EQUAL -> relation == -1 || relation == 0;
            default -> throw new AssertionError(opcode);
        };
    }

    private static boolean equal(DataType type, Object left, Object right) {
        return switch (type) {
            case FLOAT64 -> (double) left == (double) right;
            case FLOAT32 -> (float) left == (float) right;
            case INT32 -> (int) left == (int) right;
            case INT64 -> (long) left == (long) right;
            default -> throw new IllegalArgumentException("unsupported comparison type");
        };
    }

    private static byte bool(boolean value) { return (byte) (value ? 1 : 0); }

    private static Object immediate(CpuKernelIr.ScalarImmediate value) {
        return switch (value.dataType()) {
            case FLOAT64 -> Double.longBitsToDouble(value.bits());
            case FLOAT32 -> Float.intBitsToFloat((int) value.bits());
            case INT32 -> (int) value.bits(); case INT64 -> value.bits();
            default -> throw new IllegalArgumentException("unsupported immediate type");
        };
    }

    private static long[] coordinates(long index, long[] extents) {
        long[] result = new long[extents.length];
        for (int axis = extents.length - 1; axis >= 0; axis--) if (extents[axis] != 0) {
            result[axis] = index % extents[axis]; index /= extents[axis];
        }
        return result;
    }

    private static long address(CpuAccessPlan.Binding binding, long[] coordinates) {
        long address = binding.baseElementOffset();
        for (int axis = 0; axis < coordinates.length; axis++) address = Math.addExact(address,
                Math.multiplyExact(coordinates[axis], binding.effectiveStrides().get(axis)));
        return address;
    }

    private static double load(CpuBufferArgument argument, long address) {
        if (argument instanceof CpuBufferArgument.Doubles doubles) return doubles.carrier()[
                Math.toIntExact(doubles.byteOffset() / Double.BYTES + address)];
        return ((CpuBufferArgument.Segment) argument).segment().get(DOUBLE,
                Math.multiplyExact(address, Double.BYTES));
    }

    private static Object load(CpuBufferArgument argument, DataType type, long address) {
        long base = argument.byteOffset() / type.byteWidth() + address;
        if (argument instanceof CpuBufferArgument.Doubles value) return value.carrier()[Math.toIntExact(base)];
        if (argument instanceof CpuBufferArgument.Floats value) return value.carrier()[Math.toIntExact(base)];
        if (argument instanceof CpuBufferArgument.Shorts value) return value.carrier()[Math.toIntExact(base)];
        if (argument instanceof CpuBufferArgument.Ints value) return value.carrier()[Math.toIntExact(base)];
        if (argument instanceof CpuBufferArgument.Longs value) return value.carrier()[Math.toIntExact(base)];
        if (argument instanceof CpuBufferArgument.Bytes value) return value.carrier()[Math.toIntExact(base)];
        var segment = ((CpuBufferArgument.Segment) argument).segment();
        long offset = Math.multiplyExact(address, type.byteWidth());
        return switch (type) {
            case FLOAT64 -> segment.get(ValueLayout.JAVA_DOUBLE_UNALIGNED.withOrder(ByteOrder.nativeOrder()), offset);
            case FLOAT32 -> segment.get(ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.nativeOrder()), offset);
            case BFLOAT16 -> segment.get(ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.nativeOrder()), offset);
            case INT32 -> segment.get(ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.nativeOrder()), offset);
            case INT64 -> segment.get(ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.nativeOrder()), offset);
            case BOOL -> segment.get(ValueLayout.JAVA_BYTE, offset);
            default -> throw new IllegalArgumentException("unsupported reference type");
        };
    }

    private static void store(CpuBufferArgument argument, long address, double value) {
        if (argument instanceof CpuBufferArgument.Doubles doubles) doubles.carrier()[
                Math.toIntExact(doubles.byteOffset() / Double.BYTES + address)] = value;
        else ((CpuBufferArgument.Segment) argument).segment().set(DOUBLE,
                Math.multiplyExact(address, Double.BYTES), value);
    }

    private static void store(CpuBufferArgument argument, DataType type, long address, Object stored) {
        long base = argument.byteOffset() / type.byteWidth() + address;
        if (argument instanceof CpuBufferArgument.Doubles value) value.carrier()[Math.toIntExact(base)] = (double) stored;
        else if (argument instanceof CpuBufferArgument.Floats value) value.carrier()[Math.toIntExact(base)] = (float) stored;
        else if (argument instanceof CpuBufferArgument.Shorts value) value.carrier()[Math.toIntExact(base)] = (short) stored;
        else if (argument instanceof CpuBufferArgument.Ints value) value.carrier()[Math.toIntExact(base)] = (int) stored;
        else if (argument instanceof CpuBufferArgument.Longs value) value.carrier()[Math.toIntExact(base)] = (long) stored;
        else if (argument instanceof CpuBufferArgument.Bytes value) value.carrier()[Math.toIntExact(base)] = (byte) stored;
        else {
            var segment = ((CpuBufferArgument.Segment) argument).segment();
            long offset = Math.multiplyExact(address, type.byteWidth());
            switch (type) {
                case FLOAT64 -> segment.set(ValueLayout.JAVA_DOUBLE_UNALIGNED.withOrder(ByteOrder.nativeOrder()), offset, (double) stored);
                case FLOAT32 -> segment.set(ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.nativeOrder()), offset, (float) stored);
                case BFLOAT16 -> segment.set(ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.nativeOrder()), offset, (short) stored);
                case INT32 -> segment.set(ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.nativeOrder()), offset, (int) stored);
                case INT64 -> segment.set(ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.nativeOrder()), offset, (long) stored);
                case BOOL -> segment.set(ValueLayout.JAVA_BYTE, offset, (byte) stored);
                default -> throw new IllegalArgumentException("unsupported reference type");
            }
        }
    }

    private static double polevl(double x, double[] coefficients) {
        double result = coefficients[0];
        for (int i = 1; i < coefficients.length; i++) result = result * x + coefficients[i];
        return result;
    }

    private static double p1evl(double x, double[] coefficients) {
        double result = x + coefficients[0];
        for (int i = 1; i < coefficients.length; i++) result = result * x + coefficients[i];
        return result;
    }
}
