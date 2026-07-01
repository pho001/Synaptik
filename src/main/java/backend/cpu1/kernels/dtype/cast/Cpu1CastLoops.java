package backend.cpu1.kernels.dtype.cast;

import backend.cpu1.exec.Cpu1TensorView;
import backend.cpu1.kernels.layout.Cpu1LayoutKernelSupport;
import backend.cpu1.prepare.Cpu1PreparedDTypeUnit;
import tensor.DataType;
import tensor.dtype.TensorDTypeOps;

import java.lang.foreign.MemorySegment;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

/**
 * Scalar cpu1 CAST loops over array and MemorySegment storage.
 */
public final class Cpu1CastLoops {
    private Cpu1CastLoops() {
    }

    public static void castArray(Cpu1PreparedDTypeUnit unit, Cpu1TensorView input, Cpu1TensorView output) {
        validate(unit, input, output);
        castArray(unit, CastLayout.from(unit, input, output), input, output);
    }

    public static void castSegment(Cpu1PreparedDTypeUnit unit, Cpu1TensorView input, Cpu1TensorView output) {
        validate(unit, input, output);
        castSegment(unit, CastLayout.from(unit, input, output), input, output);
    }

    private static void validate(Cpu1PreparedDTypeUnit unit, Cpu1TensorView input, Cpu1TensorView output) {
        if (unit == null) {
            throw new IllegalArgumentException("unit cannot be null");
        }
        if (input == null) {
            throw new IllegalArgumentException("input cannot be null");
        }
        if (output == null) {
            throw new IllegalArgumentException("output cannot be null");
        }
        if (input.dataType() != unit.inputDataType()) {
            throw new IllegalArgumentException("Input dtype " + input.dataType()
                    + " does not match prepared input dtype " + unit.inputDataType());
        }
        if (output.dataType() != unit.outputDataType()) {
            throw new IllegalArgumentException("Output dtype " + output.dataType()
                    + " does not match prepared output dtype " + unit.outputDataType());
        }
        if (input.elementCount() != unit.elementCount() || output.elementCount() != unit.elementCount()) {
            throw new IllegalArgumentException("cpu1 CAST requires input/output element count "
                    + unit.elementCount() + ", got input=" + input.elementCount()
                    + ", output=" + output.elementCount());
        }
    }

    private static void castArray(Cpu1PreparedDTypeUnit unit, CastLayout layout,
                                  Cpu1TensorView input, Cpu1TensorView output) {
        switch (input.dataType()) {
            case FLOAT64 -> castArrayF64(unit, layout, input.float64Array(), output);
            case FLOAT32 -> castArrayF32(unit, layout, input.float32Array(), output);
            case BFLOAT16 -> castArrayBF16(unit, layout, input.bfloat16Array(), output);
            case INT32 -> castArrayI32(unit, layout, input.int32Array(), output);
            case INT64 -> castArrayI64(unit, layout, input.int64Array(), output);
            case BOOL -> castArrayBOOL(unit, layout, input.boolArray(), output);
        }
    }

    private static void castSegment(Cpu1PreparedDTypeUnit unit, CastLayout layout,
                                    Cpu1TensorView input, Cpu1TensorView output) {
        MemorySegment inputSegment = input.segment();
        MemorySegment outputSegment = output.segment();
        DataType outputType = output.dataType();
        switch (input.dataType()) {
            case FLOAT64 -> castSegmentF64(unit, layout, inputSegment, outputType, outputSegment);
            case FLOAT32 -> castSegmentF32(unit, layout, inputSegment, outputType, outputSegment);
            case BFLOAT16 -> castSegmentBF16(unit, layout, inputSegment, outputType, outputSegment);
            case INT32 -> castSegmentI32(unit, layout, inputSegment, outputType, outputSegment);
            case INT64 -> castSegmentI64(unit, layout, inputSegment, outputType, outputSegment);
            case BOOL -> castSegmentBOOL(unit, layout, inputSegment, outputType, outputSegment);
        }
    }

    private static void castArrayF64(Cpu1PreparedDTypeUnit unit, CastLayout layout,
                                            double[] input, Cpu1TensorView output) {
        int inputBase = layout.inputBase();
        int outputBase = layout.outputBase();
        if (layout.contiguous()) {
            switch (output.dataType()) {
            case FLOAT64 -> {
                double[] outputValues = output.float64Array();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + logical;
                        int outputOffset = outputBase + logical;
                        double value = input[inputOffset];
                        outputValues[outputOffset] = value;
                    }
                });
            }
            case FLOAT32 -> {
                float[] outputValues = output.float32Array();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + logical;
                        int outputOffset = outputBase + logical;
                        double value = input[inputOffset];
                        outputValues[outputOffset] = (float) value;
                    }
                });
            }
            case BFLOAT16 -> {
                short[] outputValues = output.bfloat16Array();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + logical;
                        int outputOffset = outputBase + logical;
                        double value = input[inputOffset];
                        outputValues[outputOffset] = TensorDTypeOps.toBFloat16Bits((float) value);
                    }
                });
            }
            case INT32 -> {
                int[] outputValues = output.int32Array();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + logical;
                        int outputOffset = outputBase + logical;
                        double value = input[inputOffset];
                        outputValues[outputOffset] = (int) value;
                    }
                });
            }
            case INT64 -> {
                long[] outputValues = output.int64Array();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + logical;
                        int outputOffset = outputBase + logical;
                        double value = input[inputOffset];
                        outputValues[outputOffset] = (long) value;
                    }
                });
            }
            case BOOL -> {
                byte[] outputValues = output.boolArray();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + logical;
                        int outputOffset = outputBase + logical;
                        double value = input[inputOffset];
                        outputValues[outputOffset] = boolFromDouble(value);
                    }
                });
            }
            }
            return;
        }

        int[] inputShape = layout.inputShape();
        int[] inputStrides = layout.inputStrides();
        int[] inputDense = layout.inputDense();
        int[] outputShape = layout.outputShape();
        int[] outputStrides = layout.outputStrides();
        int[] outputDense = layout.outputDense();
        switch (output.dataType()) {
            case FLOAT64 -> {
                double[] outputValues = output.float64Array();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                inputShape,
                                inputStrides,
                                inputDense
                        );
                        int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                outputShape,
                                outputStrides,
                                outputDense
                        );
                        double value = input[inputOffset];
                        outputValues[outputOffset] = value;
                    }
                });
            }
            case FLOAT32 -> {
                float[] outputValues = output.float32Array();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                inputShape,
                                inputStrides,
                                inputDense
                        );
                        int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                outputShape,
                                outputStrides,
                                outputDense
                        );
                        double value = input[inputOffset];
                        outputValues[outputOffset] = (float) value;
                    }
                });
            }
            case BFLOAT16 -> {
                short[] outputValues = output.bfloat16Array();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                inputShape,
                                inputStrides,
                                inputDense
                        );
                        int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                outputShape,
                                outputStrides,
                                outputDense
                        );
                        double value = input[inputOffset];
                        outputValues[outputOffset] = TensorDTypeOps.toBFloat16Bits((float) value);
                    }
                });
            }
            case INT32 -> {
                int[] outputValues = output.int32Array();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                inputShape,
                                inputStrides,
                                inputDense
                        );
                        int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                outputShape,
                                outputStrides,
                                outputDense
                        );
                        double value = input[inputOffset];
                        outputValues[outputOffset] = (int) value;
                    }
                });
            }
            case INT64 -> {
                long[] outputValues = output.int64Array();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                inputShape,
                                inputStrides,
                                inputDense
                        );
                        int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                outputShape,
                                outputStrides,
                                outputDense
                        );
                        double value = input[inputOffset];
                        outputValues[outputOffset] = (long) value;
                    }
                });
            }
            case BOOL -> {
                byte[] outputValues = output.boolArray();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                inputShape,
                                inputStrides,
                                inputDense
                        );
                        int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                outputShape,
                                outputStrides,
                                outputDense
                        );
                        double value = input[inputOffset];
                        outputValues[outputOffset] = boolFromDouble(value);
                    }
                });
            }
        }
    }

    private static void castArrayF32(Cpu1PreparedDTypeUnit unit, CastLayout layout,
                                            float[] input, Cpu1TensorView output) {
        int inputBase = layout.inputBase();
        int outputBase = layout.outputBase();
        if (layout.contiguous()) {
            switch (output.dataType()) {
            case FLOAT64 -> {
                double[] outputValues = output.float64Array();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + logical;
                        int outputOffset = outputBase + logical;
                        double value = input[inputOffset];
                        outputValues[outputOffset] = value;
                    }
                });
            }
            case FLOAT32 -> {
                float[] outputValues = output.float32Array();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + logical;
                        int outputOffset = outputBase + logical;
                        double value = input[inputOffset];
                        outputValues[outputOffset] = (float) value;
                    }
                });
            }
            case BFLOAT16 -> {
                short[] outputValues = output.bfloat16Array();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + logical;
                        int outputOffset = outputBase + logical;
                        double value = input[inputOffset];
                        outputValues[outputOffset] = TensorDTypeOps.toBFloat16Bits((float) value);
                    }
                });
            }
            case INT32 -> {
                int[] outputValues = output.int32Array();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + logical;
                        int outputOffset = outputBase + logical;
                        double value = input[inputOffset];
                        outputValues[outputOffset] = (int) value;
                    }
                });
            }
            case INT64 -> {
                long[] outputValues = output.int64Array();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + logical;
                        int outputOffset = outputBase + logical;
                        double value = input[inputOffset];
                        outputValues[outputOffset] = (long) value;
                    }
                });
            }
            case BOOL -> {
                byte[] outputValues = output.boolArray();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + logical;
                        int outputOffset = outputBase + logical;
                        double value = input[inputOffset];
                        outputValues[outputOffset] = boolFromDouble(value);
                    }
                });
            }
            }
            return;
        }

        int[] inputShape = layout.inputShape();
        int[] inputStrides = layout.inputStrides();
        int[] inputDense = layout.inputDense();
        int[] outputShape = layout.outputShape();
        int[] outputStrides = layout.outputStrides();
        int[] outputDense = layout.outputDense();
        switch (output.dataType()) {
            case FLOAT64 -> {
                double[] outputValues = output.float64Array();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                inputShape,
                                inputStrides,
                                inputDense
                        );
                        int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                outputShape,
                                outputStrides,
                                outputDense
                        );
                        double value = input[inputOffset];
                        outputValues[outputOffset] = value;
                    }
                });
            }
            case FLOAT32 -> {
                float[] outputValues = output.float32Array();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                inputShape,
                                inputStrides,
                                inputDense
                        );
                        int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                outputShape,
                                outputStrides,
                                outputDense
                        );
                        double value = input[inputOffset];
                        outputValues[outputOffset] = (float) value;
                    }
                });
            }
            case BFLOAT16 -> {
                short[] outputValues = output.bfloat16Array();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                inputShape,
                                inputStrides,
                                inputDense
                        );
                        int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                outputShape,
                                outputStrides,
                                outputDense
                        );
                        double value = input[inputOffset];
                        outputValues[outputOffset] = TensorDTypeOps.toBFloat16Bits((float) value);
                    }
                });
            }
            case INT32 -> {
                int[] outputValues = output.int32Array();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                inputShape,
                                inputStrides,
                                inputDense
                        );
                        int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                outputShape,
                                outputStrides,
                                outputDense
                        );
                        double value = input[inputOffset];
                        outputValues[outputOffset] = (int) value;
                    }
                });
            }
            case INT64 -> {
                long[] outputValues = output.int64Array();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                inputShape,
                                inputStrides,
                                inputDense
                        );
                        int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                outputShape,
                                outputStrides,
                                outputDense
                        );
                        double value = input[inputOffset];
                        outputValues[outputOffset] = (long) value;
                    }
                });
            }
            case BOOL -> {
                byte[] outputValues = output.boolArray();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                inputShape,
                                inputStrides,
                                inputDense
                        );
                        int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                outputShape,
                                outputStrides,
                                outputDense
                        );
                        double value = input[inputOffset];
                        outputValues[outputOffset] = boolFromDouble(value);
                    }
                });
            }
        }
    }

    private static void castArrayBF16(Cpu1PreparedDTypeUnit unit, CastLayout layout,
                                            short[] input, Cpu1TensorView output) {
        int inputBase = layout.inputBase();
        int outputBase = layout.outputBase();
        if (layout.contiguous()) {
            switch (output.dataType()) {
            case FLOAT64 -> {
                double[] outputValues = output.float64Array();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + logical;
                        int outputOffset = outputBase + logical;
                        double value = TensorDTypeOps.fromBFloat16Bits(input[inputOffset]);
                        outputValues[outputOffset] = value;
                    }
                });
            }
            case FLOAT32 -> {
                float[] outputValues = output.float32Array();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + logical;
                        int outputOffset = outputBase + logical;
                        double value = TensorDTypeOps.fromBFloat16Bits(input[inputOffset]);
                        outputValues[outputOffset] = (float) value;
                    }
                });
            }
            case BFLOAT16 -> {
                short[] outputValues = output.bfloat16Array();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + logical;
                        int outputOffset = outputBase + logical;
                        double value = TensorDTypeOps.fromBFloat16Bits(input[inputOffset]);
                        outputValues[outputOffset] = TensorDTypeOps.toBFloat16Bits((float) value);
                    }
                });
            }
            case INT32 -> {
                int[] outputValues = output.int32Array();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + logical;
                        int outputOffset = outputBase + logical;
                        double value = TensorDTypeOps.fromBFloat16Bits(input[inputOffset]);
                        outputValues[outputOffset] = (int) value;
                    }
                });
            }
            case INT64 -> {
                long[] outputValues = output.int64Array();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + logical;
                        int outputOffset = outputBase + logical;
                        double value = TensorDTypeOps.fromBFloat16Bits(input[inputOffset]);
                        outputValues[outputOffset] = (long) value;
                    }
                });
            }
            case BOOL -> {
                byte[] outputValues = output.boolArray();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + logical;
                        int outputOffset = outputBase + logical;
                        double value = TensorDTypeOps.fromBFloat16Bits(input[inputOffset]);
                        outputValues[outputOffset] = boolFromDouble(value);
                    }
                });
            }
            }
            return;
        }

        int[] inputShape = layout.inputShape();
        int[] inputStrides = layout.inputStrides();
        int[] inputDense = layout.inputDense();
        int[] outputShape = layout.outputShape();
        int[] outputStrides = layout.outputStrides();
        int[] outputDense = layout.outputDense();
        switch (output.dataType()) {
            case FLOAT64 -> {
                double[] outputValues = output.float64Array();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                inputShape,
                                inputStrides,
                                inputDense
                        );
                        int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                outputShape,
                                outputStrides,
                                outputDense
                        );
                        double value = TensorDTypeOps.fromBFloat16Bits(input[inputOffset]);
                        outputValues[outputOffset] = value;
                    }
                });
            }
            case FLOAT32 -> {
                float[] outputValues = output.float32Array();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                inputShape,
                                inputStrides,
                                inputDense
                        );
                        int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                outputShape,
                                outputStrides,
                                outputDense
                        );
                        double value = TensorDTypeOps.fromBFloat16Bits(input[inputOffset]);
                        outputValues[outputOffset] = (float) value;
                    }
                });
            }
            case BFLOAT16 -> {
                short[] outputValues = output.bfloat16Array();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                inputShape,
                                inputStrides,
                                inputDense
                        );
                        int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                outputShape,
                                outputStrides,
                                outputDense
                        );
                        double value = TensorDTypeOps.fromBFloat16Bits(input[inputOffset]);
                        outputValues[outputOffset] = TensorDTypeOps.toBFloat16Bits((float) value);
                    }
                });
            }
            case INT32 -> {
                int[] outputValues = output.int32Array();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                inputShape,
                                inputStrides,
                                inputDense
                        );
                        int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                outputShape,
                                outputStrides,
                                outputDense
                        );
                        double value = TensorDTypeOps.fromBFloat16Bits(input[inputOffset]);
                        outputValues[outputOffset] = (int) value;
                    }
                });
            }
            case INT64 -> {
                long[] outputValues = output.int64Array();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                inputShape,
                                inputStrides,
                                inputDense
                        );
                        int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                outputShape,
                                outputStrides,
                                outputDense
                        );
                        double value = TensorDTypeOps.fromBFloat16Bits(input[inputOffset]);
                        outputValues[outputOffset] = (long) value;
                    }
                });
            }
            case BOOL -> {
                byte[] outputValues = output.boolArray();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                inputShape,
                                inputStrides,
                                inputDense
                        );
                        int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                outputShape,
                                outputStrides,
                                outputDense
                        );
                        double value = TensorDTypeOps.fromBFloat16Bits(input[inputOffset]);
                        outputValues[outputOffset] = boolFromDouble(value);
                    }
                });
            }
        }
    }

    private static void castArrayI32(Cpu1PreparedDTypeUnit unit, CastLayout layout,
                                            int[] input, Cpu1TensorView output) {
        int inputBase = layout.inputBase();
        int outputBase = layout.outputBase();
        if (layout.contiguous()) {
            switch (output.dataType()) {
            case FLOAT64 -> {
                double[] outputValues = output.float64Array();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + logical;
                        int outputOffset = outputBase + logical;
                        double value = input[inputOffset];
                        outputValues[outputOffset] = value;
                    }
                });
            }
            case FLOAT32 -> {
                float[] outputValues = output.float32Array();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + logical;
                        int outputOffset = outputBase + logical;
                        double value = input[inputOffset];
                        outputValues[outputOffset] = (float) value;
                    }
                });
            }
            case BFLOAT16 -> {
                short[] outputValues = output.bfloat16Array();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + logical;
                        int outputOffset = outputBase + logical;
                        double value = input[inputOffset];
                        outputValues[outputOffset] = TensorDTypeOps.toBFloat16Bits((float) value);
                    }
                });
            }
            case INT32 -> {
                int[] outputValues = output.int32Array();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + logical;
                        int outputOffset = outputBase + logical;
                        double value = input[inputOffset];
                        outputValues[outputOffset] = (int) value;
                    }
                });
            }
            case INT64 -> {
                long[] outputValues = output.int64Array();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + logical;
                        int outputOffset = outputBase + logical;
                        double value = input[inputOffset];
                        outputValues[outputOffset] = (long) value;
                    }
                });
            }
            case BOOL -> {
                byte[] outputValues = output.boolArray();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + logical;
                        int outputOffset = outputBase + logical;
                        double value = input[inputOffset];
                        outputValues[outputOffset] = boolFromDouble(value);
                    }
                });
            }
            }
            return;
        }

        int[] inputShape = layout.inputShape();
        int[] inputStrides = layout.inputStrides();
        int[] inputDense = layout.inputDense();
        int[] outputShape = layout.outputShape();
        int[] outputStrides = layout.outputStrides();
        int[] outputDense = layout.outputDense();
        switch (output.dataType()) {
            case FLOAT64 -> {
                double[] outputValues = output.float64Array();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                inputShape,
                                inputStrides,
                                inputDense
                        );
                        int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                outputShape,
                                outputStrides,
                                outputDense
                        );
                        double value = input[inputOffset];
                        outputValues[outputOffset] = value;
                    }
                });
            }
            case FLOAT32 -> {
                float[] outputValues = output.float32Array();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                inputShape,
                                inputStrides,
                                inputDense
                        );
                        int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                outputShape,
                                outputStrides,
                                outputDense
                        );
                        double value = input[inputOffset];
                        outputValues[outputOffset] = (float) value;
                    }
                });
            }
            case BFLOAT16 -> {
                short[] outputValues = output.bfloat16Array();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                inputShape,
                                inputStrides,
                                inputDense
                        );
                        int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                outputShape,
                                outputStrides,
                                outputDense
                        );
                        double value = input[inputOffset];
                        outputValues[outputOffset] = TensorDTypeOps.toBFloat16Bits((float) value);
                    }
                });
            }
            case INT32 -> {
                int[] outputValues = output.int32Array();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                inputShape,
                                inputStrides,
                                inputDense
                        );
                        int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                outputShape,
                                outputStrides,
                                outputDense
                        );
                        double value = input[inputOffset];
                        outputValues[outputOffset] = (int) value;
                    }
                });
            }
            case INT64 -> {
                long[] outputValues = output.int64Array();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                inputShape,
                                inputStrides,
                                inputDense
                        );
                        int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                outputShape,
                                outputStrides,
                                outputDense
                        );
                        double value = input[inputOffset];
                        outputValues[outputOffset] = (long) value;
                    }
                });
            }
            case BOOL -> {
                byte[] outputValues = output.boolArray();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                inputShape,
                                inputStrides,
                                inputDense
                        );
                        int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                outputShape,
                                outputStrides,
                                outputDense
                        );
                        double value = input[inputOffset];
                        outputValues[outputOffset] = boolFromDouble(value);
                    }
                });
            }
        }
    }

    private static void castArrayI64(Cpu1PreparedDTypeUnit unit, CastLayout layout,
                                            long[] input, Cpu1TensorView output) {
        int inputBase = layout.inputBase();
        int outputBase = layout.outputBase();
        if (layout.contiguous()) {
            switch (output.dataType()) {
            case FLOAT64 -> {
                double[] outputValues = output.float64Array();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + logical;
                        int outputOffset = outputBase + logical;
                        double value = (double) input[inputOffset];
                        outputValues[outputOffset] = value;
                    }
                });
            }
            case FLOAT32 -> {
                float[] outputValues = output.float32Array();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + logical;
                        int outputOffset = outputBase + logical;
                        double value = (double) input[inputOffset];
                        outputValues[outputOffset] = (float) value;
                    }
                });
            }
            case BFLOAT16 -> {
                short[] outputValues = output.bfloat16Array();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + logical;
                        int outputOffset = outputBase + logical;
                        double value = (double) input[inputOffset];
                        outputValues[outputOffset] = TensorDTypeOps.toBFloat16Bits((float) value);
                    }
                });
            }
            case INT32 -> {
                int[] outputValues = output.int32Array();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + logical;
                        int outputOffset = outputBase + logical;
                        double value = (double) input[inputOffset];
                        outputValues[outputOffset] = (int) value;
                    }
                });
            }
            case INT64 -> {
                long[] outputValues = output.int64Array();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + logical;
                        int outputOffset = outputBase + logical;
                        double value = (double) input[inputOffset];
                        outputValues[outputOffset] = (long) value;
                    }
                });
            }
            case BOOL -> {
                byte[] outputValues = output.boolArray();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + logical;
                        int outputOffset = outputBase + logical;
                        double value = (double) input[inputOffset];
                        outputValues[outputOffset] = boolFromDouble(value);
                    }
                });
            }
            }
            return;
        }

        int[] inputShape = layout.inputShape();
        int[] inputStrides = layout.inputStrides();
        int[] inputDense = layout.inputDense();
        int[] outputShape = layout.outputShape();
        int[] outputStrides = layout.outputStrides();
        int[] outputDense = layout.outputDense();
        switch (output.dataType()) {
            case FLOAT64 -> {
                double[] outputValues = output.float64Array();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                inputShape,
                                inputStrides,
                                inputDense
                        );
                        int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                outputShape,
                                outputStrides,
                                outputDense
                        );
                        double value = (double) input[inputOffset];
                        outputValues[outputOffset] = value;
                    }
                });
            }
            case FLOAT32 -> {
                float[] outputValues = output.float32Array();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                inputShape,
                                inputStrides,
                                inputDense
                        );
                        int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                outputShape,
                                outputStrides,
                                outputDense
                        );
                        double value = (double) input[inputOffset];
                        outputValues[outputOffset] = (float) value;
                    }
                });
            }
            case BFLOAT16 -> {
                short[] outputValues = output.bfloat16Array();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                inputShape,
                                inputStrides,
                                inputDense
                        );
                        int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                outputShape,
                                outputStrides,
                                outputDense
                        );
                        double value = (double) input[inputOffset];
                        outputValues[outputOffset] = TensorDTypeOps.toBFloat16Bits((float) value);
                    }
                });
            }
            case INT32 -> {
                int[] outputValues = output.int32Array();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                inputShape,
                                inputStrides,
                                inputDense
                        );
                        int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                outputShape,
                                outputStrides,
                                outputDense
                        );
                        double value = (double) input[inputOffset];
                        outputValues[outputOffset] = (int) value;
                    }
                });
            }
            case INT64 -> {
                long[] outputValues = output.int64Array();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                inputShape,
                                inputStrides,
                                inputDense
                        );
                        int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                outputShape,
                                outputStrides,
                                outputDense
                        );
                        double value = (double) input[inputOffset];
                        outputValues[outputOffset] = (long) value;
                    }
                });
            }
            case BOOL -> {
                byte[] outputValues = output.boolArray();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                inputShape,
                                inputStrides,
                                inputDense
                        );
                        int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                outputShape,
                                outputStrides,
                                outputDense
                        );
                        double value = (double) input[inputOffset];
                        outputValues[outputOffset] = boolFromDouble(value);
                    }
                });
            }
        }
    }

    private static void castArrayBOOL(Cpu1PreparedDTypeUnit unit, CastLayout layout,
                                            byte[] input, Cpu1TensorView output) {
        int inputBase = layout.inputBase();
        int outputBase = layout.outputBase();
        if (layout.contiguous()) {
            switch (output.dataType()) {
            case FLOAT64 -> {
                double[] outputValues = output.float64Array();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + logical;
                        int outputOffset = outputBase + logical;
                        double value = input[inputOffset] == 0 ? 0.0d : 1.0d;
                        outputValues[outputOffset] = value;
                    }
                });
            }
            case FLOAT32 -> {
                float[] outputValues = output.float32Array();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + logical;
                        int outputOffset = outputBase + logical;
                        double value = input[inputOffset] == 0 ? 0.0d : 1.0d;
                        outputValues[outputOffset] = (float) value;
                    }
                });
            }
            case BFLOAT16 -> {
                short[] outputValues = output.bfloat16Array();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + logical;
                        int outputOffset = outputBase + logical;
                        double value = input[inputOffset] == 0 ? 0.0d : 1.0d;
                        outputValues[outputOffset] = TensorDTypeOps.toBFloat16Bits((float) value);
                    }
                });
            }
            case INT32 -> {
                int[] outputValues = output.int32Array();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + logical;
                        int outputOffset = outputBase + logical;
                        double value = input[inputOffset] == 0 ? 0.0d : 1.0d;
                        outputValues[outputOffset] = (int) value;
                    }
                });
            }
            case INT64 -> {
                long[] outputValues = output.int64Array();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + logical;
                        int outputOffset = outputBase + logical;
                        double value = input[inputOffset] == 0 ? 0.0d : 1.0d;
                        outputValues[outputOffset] = (long) value;
                    }
                });
            }
            case BOOL -> {
                byte[] outputValues = output.boolArray();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + logical;
                        int outputOffset = outputBase + logical;
                        double value = input[inputOffset] == 0 ? 0.0d : 1.0d;
                        outputValues[outputOffset] = boolFromDouble(value);
                    }
                });
            }
            }
            return;
        }

        int[] inputShape = layout.inputShape();
        int[] inputStrides = layout.inputStrides();
        int[] inputDense = layout.inputDense();
        int[] outputShape = layout.outputShape();
        int[] outputStrides = layout.outputStrides();
        int[] outputDense = layout.outputDense();
        switch (output.dataType()) {
            case FLOAT64 -> {
                double[] outputValues = output.float64Array();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                inputShape,
                                inputStrides,
                                inputDense
                        );
                        int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                outputShape,
                                outputStrides,
                                outputDense
                        );
                        double value = input[inputOffset] == 0 ? 0.0d : 1.0d;
                        outputValues[outputOffset] = value;
                    }
                });
            }
            case FLOAT32 -> {
                float[] outputValues = output.float32Array();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                inputShape,
                                inputStrides,
                                inputDense
                        );
                        int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                outputShape,
                                outputStrides,
                                outputDense
                        );
                        double value = input[inputOffset] == 0 ? 0.0d : 1.0d;
                        outputValues[outputOffset] = (float) value;
                    }
                });
            }
            case BFLOAT16 -> {
                short[] outputValues = output.bfloat16Array();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                inputShape,
                                inputStrides,
                                inputDense
                        );
                        int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                outputShape,
                                outputStrides,
                                outputDense
                        );
                        double value = input[inputOffset] == 0 ? 0.0d : 1.0d;
                        outputValues[outputOffset] = TensorDTypeOps.toBFloat16Bits((float) value);
                    }
                });
            }
            case INT32 -> {
                int[] outputValues = output.int32Array();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                inputShape,
                                inputStrides,
                                inputDense
                        );
                        int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                outputShape,
                                outputStrides,
                                outputDense
                        );
                        double value = input[inputOffset] == 0 ? 0.0d : 1.0d;
                        outputValues[outputOffset] = (int) value;
                    }
                });
            }
            case INT64 -> {
                long[] outputValues = output.int64Array();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                inputShape,
                                inputStrides,
                                inputDense
                        );
                        int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                outputShape,
                                outputStrides,
                                outputDense
                        );
                        double value = input[inputOffset] == 0 ? 0.0d : 1.0d;
                        outputValues[outputOffset] = (long) value;
                    }
                });
            }
            case BOOL -> {
                byte[] outputValues = output.boolArray();
                unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                    for (int logical = start; logical < end; logical++) {
                        int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                inputShape,
                                inputStrides,
                                inputDense
                        );
                        int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                                logical,
                                outputShape,
                                outputStrides,
                                outputDense
                        );
                        double value = input[inputOffset] == 0 ? 0.0d : 1.0d;
                        outputValues[outputOffset] = boolFromDouble(value);
                    }
                });
            }
        }
    }

    private static void castSegmentF64(Cpu1PreparedDTypeUnit unit, CastLayout layout,
                                              MemorySegment input, DataType outputType,
                                              MemorySegment output) {
        int inputBase = layout.inputBase();
        int outputBase = layout.outputBase();
        if (layout.contiguous()) {
            switch (outputType) {
            case FLOAT64 -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + logical;
                    int outputOffset = outputBase + logical;
                    double value = input.get(JAVA_DOUBLE, (long) inputOffset * Double.BYTES);
                    output.set(JAVA_DOUBLE, (long) outputOffset * Double.BYTES, value);
                }
            });
            case FLOAT32 -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + logical;
                    int outputOffset = outputBase + logical;
                    double value = input.get(JAVA_DOUBLE, (long) inputOffset * Double.BYTES);
                    output.set(JAVA_FLOAT, (long) outputOffset * Float.BYTES, (float) value);
                }
            });
            case BFLOAT16 -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + logical;
                    int outputOffset = outputBase + logical;
                    double value = input.get(JAVA_DOUBLE, (long) inputOffset * Double.BYTES);
                    output.set(JAVA_SHORT, (long) outputOffset * Short.BYTES, TensorDTypeOps.toBFloat16Bits((float) value));
                }
            });
            case INT32 -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + logical;
                    int outputOffset = outputBase + logical;
                    double value = input.get(JAVA_DOUBLE, (long) inputOffset * Double.BYTES);
                    output.set(JAVA_INT, (long) outputOffset * Integer.BYTES, (int) value);
                }
            });
            case INT64 -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + logical;
                    int outputOffset = outputBase + logical;
                    double value = input.get(JAVA_DOUBLE, (long) inputOffset * Double.BYTES);
                    output.set(JAVA_LONG, (long) outputOffset * Long.BYTES, (long) value);
                }
            });
            case BOOL -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + logical;
                    int outputOffset = outputBase + logical;
                    double value = input.get(JAVA_DOUBLE, (long) inputOffset * Double.BYTES);
                    output.set(JAVA_BYTE, (long) outputOffset * Byte.BYTES, boolFromDouble(value));
                }
            });
            }
            return;
        }

        int[] inputShape = layout.inputShape();
        int[] inputStrides = layout.inputStrides();
        int[] inputDense = layout.inputDense();
        int[] outputShape = layout.outputShape();
        int[] outputStrides = layout.outputStrides();
        int[] outputDense = layout.outputDense();
        switch (outputType) {
            case FLOAT64 -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            inputShape,
                            inputStrides,
                            inputDense
                    );
                    int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            outputShape,
                            outputStrides,
                            outputDense
                    );
                    double value = input.get(JAVA_DOUBLE, (long) inputOffset * Double.BYTES);
                    output.set(JAVA_DOUBLE, (long) outputOffset * Double.BYTES, value);
                }
            });
            case FLOAT32 -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            inputShape,
                            inputStrides,
                            inputDense
                    );
                    int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            outputShape,
                            outputStrides,
                            outputDense
                    );
                    double value = input.get(JAVA_DOUBLE, (long) inputOffset * Double.BYTES);
                    output.set(JAVA_FLOAT, (long) outputOffset * Float.BYTES, (float) value);
                }
            });
            case BFLOAT16 -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            inputShape,
                            inputStrides,
                            inputDense
                    );
                    int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            outputShape,
                            outputStrides,
                            outputDense
                    );
                    double value = input.get(JAVA_DOUBLE, (long) inputOffset * Double.BYTES);
                    output.set(JAVA_SHORT, (long) outputOffset * Short.BYTES, TensorDTypeOps.toBFloat16Bits((float) value));
                }
            });
            case INT32 -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            inputShape,
                            inputStrides,
                            inputDense
                    );
                    int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            outputShape,
                            outputStrides,
                            outputDense
                    );
                    double value = input.get(JAVA_DOUBLE, (long) inputOffset * Double.BYTES);
                    output.set(JAVA_INT, (long) outputOffset * Integer.BYTES, (int) value);
                }
            });
            case INT64 -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            inputShape,
                            inputStrides,
                            inputDense
                    );
                    int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            outputShape,
                            outputStrides,
                            outputDense
                    );
                    double value = input.get(JAVA_DOUBLE, (long) inputOffset * Double.BYTES);
                    output.set(JAVA_LONG, (long) outputOffset * Long.BYTES, (long) value);
                }
            });
            case BOOL -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            inputShape,
                            inputStrides,
                            inputDense
                    );
                    int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            outputShape,
                            outputStrides,
                            outputDense
                    );
                    double value = input.get(JAVA_DOUBLE, (long) inputOffset * Double.BYTES);
                    output.set(JAVA_BYTE, (long) outputOffset * Byte.BYTES, boolFromDouble(value));
                }
            });
        }
    }

    private static void castSegmentF32(Cpu1PreparedDTypeUnit unit, CastLayout layout,
                                              MemorySegment input, DataType outputType,
                                              MemorySegment output) {
        int inputBase = layout.inputBase();
        int outputBase = layout.outputBase();
        if (layout.contiguous()) {
            switch (outputType) {
            case FLOAT64 -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + logical;
                    int outputOffset = outputBase + logical;
                    double value = input.get(JAVA_FLOAT, (long) inputOffset * Float.BYTES);
                    output.set(JAVA_DOUBLE, (long) outputOffset * Double.BYTES, value);
                }
            });
            case FLOAT32 -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + logical;
                    int outputOffset = outputBase + logical;
                    double value = input.get(JAVA_FLOAT, (long) inputOffset * Float.BYTES);
                    output.set(JAVA_FLOAT, (long) outputOffset * Float.BYTES, (float) value);
                }
            });
            case BFLOAT16 -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + logical;
                    int outputOffset = outputBase + logical;
                    double value = input.get(JAVA_FLOAT, (long) inputOffset * Float.BYTES);
                    output.set(JAVA_SHORT, (long) outputOffset * Short.BYTES, TensorDTypeOps.toBFloat16Bits((float) value));
                }
            });
            case INT32 -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + logical;
                    int outputOffset = outputBase + logical;
                    double value = input.get(JAVA_FLOAT, (long) inputOffset * Float.BYTES);
                    output.set(JAVA_INT, (long) outputOffset * Integer.BYTES, (int) value);
                }
            });
            case INT64 -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + logical;
                    int outputOffset = outputBase + logical;
                    double value = input.get(JAVA_FLOAT, (long) inputOffset * Float.BYTES);
                    output.set(JAVA_LONG, (long) outputOffset * Long.BYTES, (long) value);
                }
            });
            case BOOL -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + logical;
                    int outputOffset = outputBase + logical;
                    double value = input.get(JAVA_FLOAT, (long) inputOffset * Float.BYTES);
                    output.set(JAVA_BYTE, (long) outputOffset * Byte.BYTES, boolFromDouble(value));
                }
            });
            }
            return;
        }

        int[] inputShape = layout.inputShape();
        int[] inputStrides = layout.inputStrides();
        int[] inputDense = layout.inputDense();
        int[] outputShape = layout.outputShape();
        int[] outputStrides = layout.outputStrides();
        int[] outputDense = layout.outputDense();
        switch (outputType) {
            case FLOAT64 -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            inputShape,
                            inputStrides,
                            inputDense
                    );
                    int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            outputShape,
                            outputStrides,
                            outputDense
                    );
                    double value = input.get(JAVA_FLOAT, (long) inputOffset * Float.BYTES);
                    output.set(JAVA_DOUBLE, (long) outputOffset * Double.BYTES, value);
                }
            });
            case FLOAT32 -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            inputShape,
                            inputStrides,
                            inputDense
                    );
                    int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            outputShape,
                            outputStrides,
                            outputDense
                    );
                    double value = input.get(JAVA_FLOAT, (long) inputOffset * Float.BYTES);
                    output.set(JAVA_FLOAT, (long) outputOffset * Float.BYTES, (float) value);
                }
            });
            case BFLOAT16 -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            inputShape,
                            inputStrides,
                            inputDense
                    );
                    int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            outputShape,
                            outputStrides,
                            outputDense
                    );
                    double value = input.get(JAVA_FLOAT, (long) inputOffset * Float.BYTES);
                    output.set(JAVA_SHORT, (long) outputOffset * Short.BYTES, TensorDTypeOps.toBFloat16Bits((float) value));
                }
            });
            case INT32 -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            inputShape,
                            inputStrides,
                            inputDense
                    );
                    int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            outputShape,
                            outputStrides,
                            outputDense
                    );
                    double value = input.get(JAVA_FLOAT, (long) inputOffset * Float.BYTES);
                    output.set(JAVA_INT, (long) outputOffset * Integer.BYTES, (int) value);
                }
            });
            case INT64 -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            inputShape,
                            inputStrides,
                            inputDense
                    );
                    int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            outputShape,
                            outputStrides,
                            outputDense
                    );
                    double value = input.get(JAVA_FLOAT, (long) inputOffset * Float.BYTES);
                    output.set(JAVA_LONG, (long) outputOffset * Long.BYTES, (long) value);
                }
            });
            case BOOL -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            inputShape,
                            inputStrides,
                            inputDense
                    );
                    int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            outputShape,
                            outputStrides,
                            outputDense
                    );
                    double value = input.get(JAVA_FLOAT, (long) inputOffset * Float.BYTES);
                    output.set(JAVA_BYTE, (long) outputOffset * Byte.BYTES, boolFromDouble(value));
                }
            });
        }
    }

    private static void castSegmentBF16(Cpu1PreparedDTypeUnit unit, CastLayout layout,
                                              MemorySegment input, DataType outputType,
                                              MemorySegment output) {
        int inputBase = layout.inputBase();
        int outputBase = layout.outputBase();
        if (layout.contiguous()) {
            switch (outputType) {
            case FLOAT64 -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + logical;
                    int outputOffset = outputBase + logical;
                    double value = TensorDTypeOps.fromBFloat16Bits(input.get(JAVA_SHORT, (long) inputOffset * Short.BYTES));
                    output.set(JAVA_DOUBLE, (long) outputOffset * Double.BYTES, value);
                }
            });
            case FLOAT32 -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + logical;
                    int outputOffset = outputBase + logical;
                    double value = TensorDTypeOps.fromBFloat16Bits(input.get(JAVA_SHORT, (long) inputOffset * Short.BYTES));
                    output.set(JAVA_FLOAT, (long) outputOffset * Float.BYTES, (float) value);
                }
            });
            case BFLOAT16 -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + logical;
                    int outputOffset = outputBase + logical;
                    double value = TensorDTypeOps.fromBFloat16Bits(input.get(JAVA_SHORT, (long) inputOffset * Short.BYTES));
                    output.set(JAVA_SHORT, (long) outputOffset * Short.BYTES, TensorDTypeOps.toBFloat16Bits((float) value));
                }
            });
            case INT32 -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + logical;
                    int outputOffset = outputBase + logical;
                    double value = TensorDTypeOps.fromBFloat16Bits(input.get(JAVA_SHORT, (long) inputOffset * Short.BYTES));
                    output.set(JAVA_INT, (long) outputOffset * Integer.BYTES, (int) value);
                }
            });
            case INT64 -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + logical;
                    int outputOffset = outputBase + logical;
                    double value = TensorDTypeOps.fromBFloat16Bits(input.get(JAVA_SHORT, (long) inputOffset * Short.BYTES));
                    output.set(JAVA_LONG, (long) outputOffset * Long.BYTES, (long) value);
                }
            });
            case BOOL -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + logical;
                    int outputOffset = outputBase + logical;
                    double value = TensorDTypeOps.fromBFloat16Bits(input.get(JAVA_SHORT, (long) inputOffset * Short.BYTES));
                    output.set(JAVA_BYTE, (long) outputOffset * Byte.BYTES, boolFromDouble(value));
                }
            });
            }
            return;
        }

        int[] inputShape = layout.inputShape();
        int[] inputStrides = layout.inputStrides();
        int[] inputDense = layout.inputDense();
        int[] outputShape = layout.outputShape();
        int[] outputStrides = layout.outputStrides();
        int[] outputDense = layout.outputDense();
        switch (outputType) {
            case FLOAT64 -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            inputShape,
                            inputStrides,
                            inputDense
                    );
                    int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            outputShape,
                            outputStrides,
                            outputDense
                    );
                    double value = TensorDTypeOps.fromBFloat16Bits(input.get(JAVA_SHORT, (long) inputOffset * Short.BYTES));
                    output.set(JAVA_DOUBLE, (long) outputOffset * Double.BYTES, value);
                }
            });
            case FLOAT32 -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            inputShape,
                            inputStrides,
                            inputDense
                    );
                    int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            outputShape,
                            outputStrides,
                            outputDense
                    );
                    double value = TensorDTypeOps.fromBFloat16Bits(input.get(JAVA_SHORT, (long) inputOffset * Short.BYTES));
                    output.set(JAVA_FLOAT, (long) outputOffset * Float.BYTES, (float) value);
                }
            });
            case BFLOAT16 -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            inputShape,
                            inputStrides,
                            inputDense
                    );
                    int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            outputShape,
                            outputStrides,
                            outputDense
                    );
                    double value = TensorDTypeOps.fromBFloat16Bits(input.get(JAVA_SHORT, (long) inputOffset * Short.BYTES));
                    output.set(JAVA_SHORT, (long) outputOffset * Short.BYTES, TensorDTypeOps.toBFloat16Bits((float) value));
                }
            });
            case INT32 -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            inputShape,
                            inputStrides,
                            inputDense
                    );
                    int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            outputShape,
                            outputStrides,
                            outputDense
                    );
                    double value = TensorDTypeOps.fromBFloat16Bits(input.get(JAVA_SHORT, (long) inputOffset * Short.BYTES));
                    output.set(JAVA_INT, (long) outputOffset * Integer.BYTES, (int) value);
                }
            });
            case INT64 -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            inputShape,
                            inputStrides,
                            inputDense
                    );
                    int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            outputShape,
                            outputStrides,
                            outputDense
                    );
                    double value = TensorDTypeOps.fromBFloat16Bits(input.get(JAVA_SHORT, (long) inputOffset * Short.BYTES));
                    output.set(JAVA_LONG, (long) outputOffset * Long.BYTES, (long) value);
                }
            });
            case BOOL -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            inputShape,
                            inputStrides,
                            inputDense
                    );
                    int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            outputShape,
                            outputStrides,
                            outputDense
                    );
                    double value = TensorDTypeOps.fromBFloat16Bits(input.get(JAVA_SHORT, (long) inputOffset * Short.BYTES));
                    output.set(JAVA_BYTE, (long) outputOffset * Byte.BYTES, boolFromDouble(value));
                }
            });
        }
    }

    private static void castSegmentI32(Cpu1PreparedDTypeUnit unit, CastLayout layout,
                                              MemorySegment input, DataType outputType,
                                              MemorySegment output) {
        int inputBase = layout.inputBase();
        int outputBase = layout.outputBase();
        if (layout.contiguous()) {
            switch (outputType) {
            case FLOAT64 -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + logical;
                    int outputOffset = outputBase + logical;
                    double value = input.get(JAVA_INT, (long) inputOffset * Integer.BYTES);
                    output.set(JAVA_DOUBLE, (long) outputOffset * Double.BYTES, value);
                }
            });
            case FLOAT32 -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + logical;
                    int outputOffset = outputBase + logical;
                    double value = input.get(JAVA_INT, (long) inputOffset * Integer.BYTES);
                    output.set(JAVA_FLOAT, (long) outputOffset * Float.BYTES, (float) value);
                }
            });
            case BFLOAT16 -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + logical;
                    int outputOffset = outputBase + logical;
                    double value = input.get(JAVA_INT, (long) inputOffset * Integer.BYTES);
                    output.set(JAVA_SHORT, (long) outputOffset * Short.BYTES, TensorDTypeOps.toBFloat16Bits((float) value));
                }
            });
            case INT32 -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + logical;
                    int outputOffset = outputBase + logical;
                    double value = input.get(JAVA_INT, (long) inputOffset * Integer.BYTES);
                    output.set(JAVA_INT, (long) outputOffset * Integer.BYTES, (int) value);
                }
            });
            case INT64 -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + logical;
                    int outputOffset = outputBase + logical;
                    double value = input.get(JAVA_INT, (long) inputOffset * Integer.BYTES);
                    output.set(JAVA_LONG, (long) outputOffset * Long.BYTES, (long) value);
                }
            });
            case BOOL -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + logical;
                    int outputOffset = outputBase + logical;
                    double value = input.get(JAVA_INT, (long) inputOffset * Integer.BYTES);
                    output.set(JAVA_BYTE, (long) outputOffset * Byte.BYTES, boolFromDouble(value));
                }
            });
            }
            return;
        }

        int[] inputShape = layout.inputShape();
        int[] inputStrides = layout.inputStrides();
        int[] inputDense = layout.inputDense();
        int[] outputShape = layout.outputShape();
        int[] outputStrides = layout.outputStrides();
        int[] outputDense = layout.outputDense();
        switch (outputType) {
            case FLOAT64 -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            inputShape,
                            inputStrides,
                            inputDense
                    );
                    int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            outputShape,
                            outputStrides,
                            outputDense
                    );
                    double value = input.get(JAVA_INT, (long) inputOffset * Integer.BYTES);
                    output.set(JAVA_DOUBLE, (long) outputOffset * Double.BYTES, value);
                }
            });
            case FLOAT32 -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            inputShape,
                            inputStrides,
                            inputDense
                    );
                    int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            outputShape,
                            outputStrides,
                            outputDense
                    );
                    double value = input.get(JAVA_INT, (long) inputOffset * Integer.BYTES);
                    output.set(JAVA_FLOAT, (long) outputOffset * Float.BYTES, (float) value);
                }
            });
            case BFLOAT16 -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            inputShape,
                            inputStrides,
                            inputDense
                    );
                    int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            outputShape,
                            outputStrides,
                            outputDense
                    );
                    double value = input.get(JAVA_INT, (long) inputOffset * Integer.BYTES);
                    output.set(JAVA_SHORT, (long) outputOffset * Short.BYTES, TensorDTypeOps.toBFloat16Bits((float) value));
                }
            });
            case INT32 -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            inputShape,
                            inputStrides,
                            inputDense
                    );
                    int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            outputShape,
                            outputStrides,
                            outputDense
                    );
                    double value = input.get(JAVA_INT, (long) inputOffset * Integer.BYTES);
                    output.set(JAVA_INT, (long) outputOffset * Integer.BYTES, (int) value);
                }
            });
            case INT64 -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            inputShape,
                            inputStrides,
                            inputDense
                    );
                    int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            outputShape,
                            outputStrides,
                            outputDense
                    );
                    double value = input.get(JAVA_INT, (long) inputOffset * Integer.BYTES);
                    output.set(JAVA_LONG, (long) outputOffset * Long.BYTES, (long) value);
                }
            });
            case BOOL -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            inputShape,
                            inputStrides,
                            inputDense
                    );
                    int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            outputShape,
                            outputStrides,
                            outputDense
                    );
                    double value = input.get(JAVA_INT, (long) inputOffset * Integer.BYTES);
                    output.set(JAVA_BYTE, (long) outputOffset * Byte.BYTES, boolFromDouble(value));
                }
            });
        }
    }

    private static void castSegmentI64(Cpu1PreparedDTypeUnit unit, CastLayout layout,
                                              MemorySegment input, DataType outputType,
                                              MemorySegment output) {
        int inputBase = layout.inputBase();
        int outputBase = layout.outputBase();
        if (layout.contiguous()) {
            switch (outputType) {
            case FLOAT64 -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + logical;
                    int outputOffset = outputBase + logical;
                    double value = (double) input.get(JAVA_LONG, (long) inputOffset * Long.BYTES);
                    output.set(JAVA_DOUBLE, (long) outputOffset * Double.BYTES, value);
                }
            });
            case FLOAT32 -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + logical;
                    int outputOffset = outputBase + logical;
                    double value = (double) input.get(JAVA_LONG, (long) inputOffset * Long.BYTES);
                    output.set(JAVA_FLOAT, (long) outputOffset * Float.BYTES, (float) value);
                }
            });
            case BFLOAT16 -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + logical;
                    int outputOffset = outputBase + logical;
                    double value = (double) input.get(JAVA_LONG, (long) inputOffset * Long.BYTES);
                    output.set(JAVA_SHORT, (long) outputOffset * Short.BYTES, TensorDTypeOps.toBFloat16Bits((float) value));
                }
            });
            case INT32 -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + logical;
                    int outputOffset = outputBase + logical;
                    double value = (double) input.get(JAVA_LONG, (long) inputOffset * Long.BYTES);
                    output.set(JAVA_INT, (long) outputOffset * Integer.BYTES, (int) value);
                }
            });
            case INT64 -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + logical;
                    int outputOffset = outputBase + logical;
                    double value = (double) input.get(JAVA_LONG, (long) inputOffset * Long.BYTES);
                    output.set(JAVA_LONG, (long) outputOffset * Long.BYTES, (long) value);
                }
            });
            case BOOL -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + logical;
                    int outputOffset = outputBase + logical;
                    double value = (double) input.get(JAVA_LONG, (long) inputOffset * Long.BYTES);
                    output.set(JAVA_BYTE, (long) outputOffset * Byte.BYTES, boolFromDouble(value));
                }
            });
            }
            return;
        }

        int[] inputShape = layout.inputShape();
        int[] inputStrides = layout.inputStrides();
        int[] inputDense = layout.inputDense();
        int[] outputShape = layout.outputShape();
        int[] outputStrides = layout.outputStrides();
        int[] outputDense = layout.outputDense();
        switch (outputType) {
            case FLOAT64 -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            inputShape,
                            inputStrides,
                            inputDense
                    );
                    int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            outputShape,
                            outputStrides,
                            outputDense
                    );
                    double value = (double) input.get(JAVA_LONG, (long) inputOffset * Long.BYTES);
                    output.set(JAVA_DOUBLE, (long) outputOffset * Double.BYTES, value);
                }
            });
            case FLOAT32 -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            inputShape,
                            inputStrides,
                            inputDense
                    );
                    int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            outputShape,
                            outputStrides,
                            outputDense
                    );
                    double value = (double) input.get(JAVA_LONG, (long) inputOffset * Long.BYTES);
                    output.set(JAVA_FLOAT, (long) outputOffset * Float.BYTES, (float) value);
                }
            });
            case BFLOAT16 -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            inputShape,
                            inputStrides,
                            inputDense
                    );
                    int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            outputShape,
                            outputStrides,
                            outputDense
                    );
                    double value = (double) input.get(JAVA_LONG, (long) inputOffset * Long.BYTES);
                    output.set(JAVA_SHORT, (long) outputOffset * Short.BYTES, TensorDTypeOps.toBFloat16Bits((float) value));
                }
            });
            case INT32 -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            inputShape,
                            inputStrides,
                            inputDense
                    );
                    int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            outputShape,
                            outputStrides,
                            outputDense
                    );
                    double value = (double) input.get(JAVA_LONG, (long) inputOffset * Long.BYTES);
                    output.set(JAVA_INT, (long) outputOffset * Integer.BYTES, (int) value);
                }
            });
            case INT64 -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            inputShape,
                            inputStrides,
                            inputDense
                    );
                    int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            outputShape,
                            outputStrides,
                            outputDense
                    );
                    double value = (double) input.get(JAVA_LONG, (long) inputOffset * Long.BYTES);
                    output.set(JAVA_LONG, (long) outputOffset * Long.BYTES, (long) value);
                }
            });
            case BOOL -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            inputShape,
                            inputStrides,
                            inputDense
                    );
                    int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            outputShape,
                            outputStrides,
                            outputDense
                    );
                    double value = (double) input.get(JAVA_LONG, (long) inputOffset * Long.BYTES);
                    output.set(JAVA_BYTE, (long) outputOffset * Byte.BYTES, boolFromDouble(value));
                }
            });
        }
    }

    private static void castSegmentBOOL(Cpu1PreparedDTypeUnit unit, CastLayout layout,
                                              MemorySegment input, DataType outputType,
                                              MemorySegment output) {
        int inputBase = layout.inputBase();
        int outputBase = layout.outputBase();
        if (layout.contiguous()) {
            switch (outputType) {
            case FLOAT64 -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + logical;
                    int outputOffset = outputBase + logical;
                    double value = input.get(JAVA_BYTE, (long) inputOffset * Byte.BYTES) == 0 ? 0.0d : 1.0d;
                    output.set(JAVA_DOUBLE, (long) outputOffset * Double.BYTES, value);
                }
            });
            case FLOAT32 -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + logical;
                    int outputOffset = outputBase + logical;
                    double value = input.get(JAVA_BYTE, (long) inputOffset * Byte.BYTES) == 0 ? 0.0d : 1.0d;
                    output.set(JAVA_FLOAT, (long) outputOffset * Float.BYTES, (float) value);
                }
            });
            case BFLOAT16 -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + logical;
                    int outputOffset = outputBase + logical;
                    double value = input.get(JAVA_BYTE, (long) inputOffset * Byte.BYTES) == 0 ? 0.0d : 1.0d;
                    output.set(JAVA_SHORT, (long) outputOffset * Short.BYTES, TensorDTypeOps.toBFloat16Bits((float) value));
                }
            });
            case INT32 -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + logical;
                    int outputOffset = outputBase + logical;
                    double value = input.get(JAVA_BYTE, (long) inputOffset * Byte.BYTES) == 0 ? 0.0d : 1.0d;
                    output.set(JAVA_INT, (long) outputOffset * Integer.BYTES, (int) value);
                }
            });
            case INT64 -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + logical;
                    int outputOffset = outputBase + logical;
                    double value = input.get(JAVA_BYTE, (long) inputOffset * Byte.BYTES) == 0 ? 0.0d : 1.0d;
                    output.set(JAVA_LONG, (long) outputOffset * Long.BYTES, (long) value);
                }
            });
            case BOOL -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + logical;
                    int outputOffset = outputBase + logical;
                    double value = input.get(JAVA_BYTE, (long) inputOffset * Byte.BYTES) == 0 ? 0.0d : 1.0d;
                    output.set(JAVA_BYTE, (long) outputOffset * Byte.BYTES, boolFromDouble(value));
                }
            });
            }
            return;
        }

        int[] inputShape = layout.inputShape();
        int[] inputStrides = layout.inputStrides();
        int[] inputDense = layout.inputDense();
        int[] outputShape = layout.outputShape();
        int[] outputStrides = layout.outputStrides();
        int[] outputDense = layout.outputDense();
        switch (outputType) {
            case FLOAT64 -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            inputShape,
                            inputStrides,
                            inputDense
                    );
                    int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            outputShape,
                            outputStrides,
                            outputDense
                    );
                    double value = input.get(JAVA_BYTE, (long) inputOffset * Byte.BYTES) == 0 ? 0.0d : 1.0d;
                    output.set(JAVA_DOUBLE, (long) outputOffset * Double.BYTES, value);
                }
            });
            case FLOAT32 -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            inputShape,
                            inputStrides,
                            inputDense
                    );
                    int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            outputShape,
                            outputStrides,
                            outputDense
                    );
                    double value = input.get(JAVA_BYTE, (long) inputOffset * Byte.BYTES) == 0 ? 0.0d : 1.0d;
                    output.set(JAVA_FLOAT, (long) outputOffset * Float.BYTES, (float) value);
                }
            });
            case BFLOAT16 -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            inputShape,
                            inputStrides,
                            inputDense
                    );
                    int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            outputShape,
                            outputStrides,
                            outputDense
                    );
                    double value = input.get(JAVA_BYTE, (long) inputOffset * Byte.BYTES) == 0 ? 0.0d : 1.0d;
                    output.set(JAVA_SHORT, (long) outputOffset * Short.BYTES, TensorDTypeOps.toBFloat16Bits((float) value));
                }
            });
            case INT32 -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            inputShape,
                            inputStrides,
                            inputDense
                    );
                    int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            outputShape,
                            outputStrides,
                            outputDense
                    );
                    double value = input.get(JAVA_BYTE, (long) inputOffset * Byte.BYTES) == 0 ? 0.0d : 1.0d;
                    output.set(JAVA_INT, (long) outputOffset * Integer.BYTES, (int) value);
                }
            });
            case INT64 -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            inputShape,
                            inputStrides,
                            inputDense
                    );
                    int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            outputShape,
                            outputStrides,
                            outputDense
                    );
                    double value = input.get(JAVA_BYTE, (long) inputOffset * Byte.BYTES) == 0 ? 0.0d : 1.0d;
                    output.set(JAVA_LONG, (long) outputOffset * Long.BYTES, (long) value);
                }
            });
            case BOOL -> unit.launchPolicy().launch(layout.elementCount(), (start, end) -> {
                for (int logical = start; logical < end; logical++) {
                    int inputOffset = inputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            inputShape,
                            inputStrides,
                            inputDense
                    );
                    int outputOffset = outputBase + Cpu1LayoutKernelSupport.logicalOffset(
                            logical,
                            outputShape,
                            outputStrides,
                            outputDense
                    );
                    double value = input.get(JAVA_BYTE, (long) inputOffset * Byte.BYTES) == 0 ? 0.0d : 1.0d;
                    output.set(JAVA_BYTE, (long) outputOffset * Byte.BYTES, boolFromDouble(value));
                }
            });
        }
    }

    private static byte boolFromDouble(double value) {
        return value == 0.0d ? (byte) 0 : (byte) 1;
    }

    private record CastLayout(
            int elementCount,
            boolean contiguous,
            int inputBase,
            int outputBase,
            int[] inputShape,
            int[] inputStrides,
            int[] inputDense,
            int[] outputShape,
            int[] outputStrides,
            int[] outputDense
    ) {
        private static CastLayout from(Cpu1PreparedDTypeUnit unit, Cpu1TensorView input, Cpu1TensorView output) {
            boolean contiguous = input.contiguous() && output.contiguous();
            if (contiguous) {
                return new CastLayout(
                        unit.elementCount(),
                        true,
                        input.storageOffset(),
                        output.storageOffset(),
                        null,
                        null,
                        null,
                        null,
                        null,
                        null
                );
            }

            int[] inputShape = input.shape();
            int[] outputShape = output.shape();
            return new CastLayout(
                    unit.elementCount(),
                    false,
                    input.storageOffset(),
                    output.storageOffset(),
                    inputShape,
                    input.strides(),
                    Cpu1LayoutKernelSupport.denseStrides(inputShape),
                    outputShape,
                    output.strides(),
                    Cpu1LayoutKernelSupport.denseStrides(outputShape)
            );
        }
    }
}
