package backend.cpu1.kernels.layout;

import backend.cpu1.exec.Cpu1TensorView;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorSpecies;
import tensor.DataType;

import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

/**
 * Vector API loops for dense cpu1 layout copies and fills.
 */
public final class Cpu1LayoutVectorLoops {
    private static final VectorSpecies<Float> F32 = FloatVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Double> F64 = DoubleVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Short> BF16 = ShortVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Byte> BOOL = ByteVector.SPECIES_PREFERRED;
    private static final ByteOrder ORDER = ByteOrder.nativeOrder();

    private Cpu1LayoutVectorLoops() {
    }

    public static void copyDenseArray(
            Cpu1TensorView input,
            int inputOffset,
            Cpu1TensorView output,
            int outputOffset,
            int elements,
            DataType dataType
    ) {
        if (elements <= 0) {
            return;
        }
        switch (dataType) {
            case FLOAT32 -> copyF32Array(input.float32Array(), inputOffset, output.float32Array(), outputOffset, elements);
            case FLOAT64 -> copyF64Array(input.float64Array(), inputOffset, output.float64Array(), outputOffset, elements);
            case BFLOAT16 -> copyBF16Array(input.bfloat16Array(), inputOffset, output.bfloat16Array(), outputOffset, elements);
            case BOOL -> copyBoolArray(input.boolArray(), inputOffset, output.boolArray(), outputOffset, elements);
            case INT32, INT64 -> throw new UnsupportedOperationException("cpu1 layout vector dtype=" + dataType);
        }
    }

    public static void copyDenseSegment(
            Cpu1TensorView input,
            int inputOffset,
            Cpu1TensorView output,
            int outputOffset,
            int elements,
            DataType dataType
    ) {
        if (elements <= 0) {
            return;
        }
        switch (dataType) {
            case FLOAT32 -> copyF32Segment(input.segment(), inputOffset, output.segment(), outputOffset, elements);
            case FLOAT64 -> copyF64Segment(input.segment(), inputOffset, output.segment(), outputOffset, elements);
            case BFLOAT16 -> copyBF16Segment(input.segment(), inputOffset, output.segment(), outputOffset, elements);
            case BOOL -> copyBoolSegment(input.segment(), inputOffset, output.segment(), outputOffset, elements);
            case INT32, INT64 -> throw new UnsupportedOperationException("cpu1 layout vector dtype=" + dataType);
        }
    }

    public static void fillDenseArray(Cpu1TensorView output, int outputOffset, int elements, double value, DataType dataType) {
        if (elements <= 0) {
            return;
        }
        switch (dataType) {
            case FLOAT32 -> fillF32Array(output.float32Array(), outputOffset, elements, (float) value);
            case FLOAT64 -> fillF64Array(output.float64Array(), outputOffset, elements, value);
            case BFLOAT16 -> fillBF16Array(
                    output.bfloat16Array(),
                    outputOffset,
                    elements,
                    tensor.dtype.TensorDTypeOps.toBFloat16Bits((float) value)
            );
            case BOOL -> fillBoolArray(output.boolArray(), outputOffset, elements, value == 0.0d ? (byte) 0 : (byte) 1);
            case INT32, INT64 -> throw new UnsupportedOperationException("cpu1 layout vector dtype=" + dataType);
        }
    }

    public static void fillDenseSegment(
            Cpu1TensorView output,
            int outputOffset,
            int elements,
            double value,
            DataType dataType
    ) {
        if (elements <= 0) {
            return;
        }
        switch (dataType) {
            case FLOAT32 -> fillF32Segment(output.segment(), outputOffset, elements, (float) value);
            case FLOAT64 -> fillF64Segment(output.segment(), outputOffset, elements, value);
            case BFLOAT16 -> fillBF16Segment(
                    output.segment(),
                    outputOffset,
                    elements,
                    tensor.dtype.TensorDTypeOps.toBFloat16Bits((float) value)
            );
            case BOOL -> fillBoolSegment(output.segment(), outputOffset, elements, value == 0.0d ? (byte) 0 : (byte) 1);
            case INT32, INT64 -> throw new UnsupportedOperationException("cpu1 layout vector dtype=" + dataType);
        }
    }

    private static void copyF32Array(float[] input, int inputOffset, float[] output, int outputOffset, int elements) {
        int i = 0;
        int upper = F32.loopBound(elements);
        for (; i < upper; i += F32.length()) {
            FloatVector.fromArray(F32, input, inputOffset + i)
                    .intoArray(output, outputOffset + i);
        }
        for (; i < elements; i++) {
            output[outputOffset + i] = input[inputOffset + i];
        }
    }

    private static void copyF64Array(double[] input, int inputOffset, double[] output, int outputOffset, int elements) {
        int i = 0;
        int upper = F64.loopBound(elements);
        for (; i < upper; i += F64.length()) {
            DoubleVector.fromArray(F64, input, inputOffset + i)
                    .intoArray(output, outputOffset + i);
        }
        for (; i < elements; i++) {
            output[outputOffset + i] = input[inputOffset + i];
        }
    }

    private static void copyBF16Array(short[] input, int inputOffset, short[] output, int outputOffset, int elements) {
        int i = 0;
        int upper = BF16.loopBound(elements);
        for (; i < upper; i += BF16.length()) {
            ShortVector.fromArray(BF16, input, inputOffset + i)
                    .intoArray(output, outputOffset + i);
        }
        for (; i < elements; i++) {
            output[outputOffset + i] = input[inputOffset + i];
        }
    }

    private static void copyBoolArray(byte[] input, int inputOffset, byte[] output, int outputOffset, int elements) {
        int i = 0;
        int upper = BOOL.loopBound(elements);
        for (; i < upper; i += BOOL.length()) {
            ByteVector.fromArray(BOOL, input, inputOffset + i)
                    .intoArray(output, outputOffset + i);
        }
        for (; i < elements; i++) {
            output[outputOffset + i] = input[inputOffset + i];
        }
    }

    private static void copyF32Segment(MemorySegment input, int inputOffset, MemorySegment output, int outputOffset, int elements) {
        int i = 0;
        int upper = F32.loopBound(elements);
        for (; i < upper; i += F32.length()) {
            FloatVector.fromMemorySegment(F32, input, (long) (inputOffset + i) * Float.BYTES, ORDER)
                    .intoMemorySegment(output, (long) (outputOffset + i) * Float.BYTES, ORDER);
        }
        for (; i < elements; i++) {
            output.set(JAVA_FLOAT, (long) (outputOffset + i) * Float.BYTES,
                    input.get(JAVA_FLOAT, (long) (inputOffset + i) * Float.BYTES));
        }
    }

    private static void copyF64Segment(MemorySegment input, int inputOffset, MemorySegment output, int outputOffset, int elements) {
        int i = 0;
        int upper = F64.loopBound(elements);
        for (; i < upper; i += F64.length()) {
            DoubleVector.fromMemorySegment(F64, input, (long) (inputOffset + i) * Double.BYTES, ORDER)
                    .intoMemorySegment(output, (long) (outputOffset + i) * Double.BYTES, ORDER);
        }
        for (; i < elements; i++) {
            output.set(JAVA_DOUBLE, (long) (outputOffset + i) * Double.BYTES,
                    input.get(JAVA_DOUBLE, (long) (inputOffset + i) * Double.BYTES));
        }
    }

    private static void copyBF16Segment(MemorySegment input, int inputOffset, MemorySegment output, int outputOffset, int elements) {
        int i = 0;
        int upper = BF16.loopBound(elements);
        for (; i < upper; i += BF16.length()) {
            ShortVector.fromMemorySegment(BF16, input, (long) (inputOffset + i) * Short.BYTES, ORDER)
                    .intoMemorySegment(output, (long) (outputOffset + i) * Short.BYTES, ORDER);
        }
        for (; i < elements; i++) {
            output.set(JAVA_SHORT, (long) (outputOffset + i) * Short.BYTES,
                    input.get(JAVA_SHORT, (long) (inputOffset + i) * Short.BYTES));
        }
    }

    private static void copyBoolSegment(MemorySegment input, int inputOffset, MemorySegment output, int outputOffset, int elements) {
        int i = 0;
        int upper = BOOL.loopBound(elements);
        for (; i < upper; i += BOOL.length()) {
            ByteVector.fromMemorySegment(BOOL, input, inputOffset + i, ORDER)
                    .intoMemorySegment(output, outputOffset + i, ORDER);
        }
        for (; i < elements; i++) {
            output.set(JAVA_BYTE, outputOffset + i, input.get(JAVA_BYTE, inputOffset + i));
        }
    }

    private static void fillF32Array(float[] output, int outputOffset, int elements, float value) {
        FloatVector vector = FloatVector.broadcast(F32, value);
        int i = 0;
        int upper = F32.loopBound(elements);
        for (; i < upper; i += F32.length()) {
            vector.intoArray(output, outputOffset + i);
        }
        for (; i < elements; i++) {
            output[outputOffset + i] = value;
        }
    }

    private static void fillF64Array(double[] output, int outputOffset, int elements, double value) {
        DoubleVector vector = DoubleVector.broadcast(F64, value);
        int i = 0;
        int upper = F64.loopBound(elements);
        for (; i < upper; i += F64.length()) {
            vector.intoArray(output, outputOffset + i);
        }
        for (; i < elements; i++) {
            output[outputOffset + i] = value;
        }
    }

    private static void fillBF16Array(short[] output, int outputOffset, int elements, short value) {
        ShortVector vector = ShortVector.broadcast(BF16, value);
        int i = 0;
        int upper = BF16.loopBound(elements);
        for (; i < upper; i += BF16.length()) {
            vector.intoArray(output, outputOffset + i);
        }
        for (; i < elements; i++) {
            output[outputOffset + i] = value;
        }
    }

    private static void fillBoolArray(byte[] output, int outputOffset, int elements, byte value) {
        ByteVector vector = ByteVector.broadcast(BOOL, value);
        int i = 0;
        int upper = BOOL.loopBound(elements);
        for (; i < upper; i += BOOL.length()) {
            vector.intoArray(output, outputOffset + i);
        }
        for (; i < elements; i++) {
            output[outputOffset + i] = value;
        }
    }

    private static void fillF32Segment(MemorySegment output, int outputOffset, int elements, float value) {
        FloatVector vector = FloatVector.broadcast(F32, value);
        int i = 0;
        int upper = F32.loopBound(elements);
        for (; i < upper; i += F32.length()) {
            vector.intoMemorySegment(output, (long) (outputOffset + i) * Float.BYTES, ORDER);
        }
        for (; i < elements; i++) {
            output.set(JAVA_FLOAT, (long) (outputOffset + i) * Float.BYTES, value);
        }
    }

    private static void fillF64Segment(MemorySegment output, int outputOffset, int elements, double value) {
        DoubleVector vector = DoubleVector.broadcast(F64, value);
        int i = 0;
        int upper = F64.loopBound(elements);
        for (; i < upper; i += F64.length()) {
            vector.intoMemorySegment(output, (long) (outputOffset + i) * Double.BYTES, ORDER);
        }
        for (; i < elements; i++) {
            output.set(JAVA_DOUBLE, (long) (outputOffset + i) * Double.BYTES, value);
        }
    }

    private static void fillBF16Segment(MemorySegment output, int outputOffset, int elements, short value) {
        ShortVector vector = ShortVector.broadcast(BF16, value);
        int i = 0;
        int upper = BF16.loopBound(elements);
        for (; i < upper; i += BF16.length()) {
            vector.intoMemorySegment(output, (long) (outputOffset + i) * Short.BYTES, ORDER);
        }
        for (; i < elements; i++) {
            output.set(JAVA_SHORT, (long) (outputOffset + i) * Short.BYTES, value);
        }
    }

    private static void fillBoolSegment(MemorySegment output, int outputOffset, int elements, byte value) {
        ByteVector vector = ByteVector.broadcast(BOOL, value);
        int i = 0;
        int upper = BOOL.loopBound(elements);
        for (; i < upper; i += BOOL.length()) {
            vector.intoMemorySegment(output, outputOffset + i, ORDER);
        }
        for (; i < elements; i++) {
            output.set(JAVA_BYTE, outputOffset + i, value);
        }
    }
}
