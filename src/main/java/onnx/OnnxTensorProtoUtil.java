package onnx;

import tensor.TensorInternalAccess;

import backend.cpu.kernels.CpuDTypeOps;
import com.google.protobuf.ByteString;
import tensor.DataType;
import tensor.Tensor;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;

final class OnnxTensorProtoUtil {
    private OnnxTensorProtoUtil() {
    }

    static OnnxProto.ValueInfoProto valueInfo(String name, DataType dataType, int[] shape) {
        OnnxProto.TensorShapeProto.Builder shapeBuilder = OnnxProto.TensorShapeProto.newBuilder();
        for (int dim : shape) {
            shapeBuilder.addDim(OnnxProto.TensorShapeProto.Dimension.newBuilder().setDimValue(dim));
        }
        OnnxProto.TypeProto.Tensor tensorType = OnnxProto.TypeProto.Tensor.newBuilder()
                .setElemType(OnnxDataTypes.toOnnx(dataType))
                .setShape(shapeBuilder)
                .build();
        return OnnxProto.ValueInfoProto.newBuilder()
                .setName(name)
                .setType(OnnxProto.TypeProto.newBuilder().setTensorType(tensorType))
                .build();
    }

    static OnnxProto.TensorProto tensorInitializer(String name, Tensor tensor) {
        OnnxProto.TensorProto.Builder builder = OnnxProto.TensorProto.newBuilder()
                .setName(name)
                .setDataType(OnnxDataTypes.toOnnx(tensor.getDataType()));
        for (int dim : tensor.getShapeUnsafe()) {
            builder.addDims(dim);
        }
        switch (tensor.getDataType()) {
            case FLOAT32 -> {
                double[] values = tensor.toDoubleArrayCopy();
                for (double value : values) {
                    builder.addFloatData((float) value);
                }
            }
            case FLOAT64 -> {
                double[] values = tensor.toDoubleArrayCopy();
                for (double value : values) {
                    builder.addDoubleData(value);
                }
            }
            case BFLOAT16 -> {
                short[] bits = logicalBfloat16(tensor);
                for (short bit : bits) {
                    builder.addInt32Data(bit & 0xFFFF);
                }
            }
            case INT32 -> {
                int[] values = logicalInt32(tensor);
                for (int value : values) {
                    builder.addInt32Data(value);
                }
            }
            case INT64 -> {
                long[] values = logicalInt64(tensor);
                for (long value : values) {
                    builder.addInt64Data(value);
                }
            }
            case BOOL -> {
                boolean[] values = tensor.toBooleanArrayCopy();
                for (boolean value : values) {
                    builder.addInt32Data(value ? 1 : 0);
                }
            }
        }
        return builder.build();
    }

    static OnnxProto.TensorProto int64Initializer(String name, long[] values) {
        OnnxProto.TensorProto.Builder builder = OnnxProto.TensorProto.newBuilder()
                .setName(name)
                .setDataType(OnnxProto.TensorProto.DataType.INT64.getNumber())
                .addDims(values.length);
        for (long value : values) {
            builder.addInt64Data(value);
        }
        return builder.build();
    }

    static ImportedConstant parseConstant(OnnxProto.TensorProto proto, String context) {
        if (proto.getDataLocation() == OnnxProto.TensorProto.DataLocation.EXTERNAL) {
            throw new OnnxUnsupportedException(context + " uses external tensor data, which is not supported by ONNX v0 import.");
        }
        int[] shape = dims(proto, context);
        int count = elementCount(shape);
        DataType dataType = OnnxDataTypes.toSynaptik(proto.getDataType(), context);
        return ImportedConstant.tensor(switch (dataType) {
            case FLOAT32 -> new Tensor(readFloat32(proto, count, context), shape, null, proto.getName(), DataType.FLOAT32);
            case FLOAT64 -> new Tensor(readFloat64(proto, count, context), shape, null, proto.getName(), DataType.FLOAT64);
            case BFLOAT16 -> new Tensor(readBfloat16(proto, count, context), shape, null, proto.getName(), DataType.BFLOAT16);
            case INT32 -> new Tensor(readInt32(proto, count, context), shape, null, proto.getName(), DataType.INT32);
            case INT64 -> new Tensor(readInt64(proto, count, context), shape, null, proto.getName(), DataType.INT64);
            case BOOL -> new Tensor(readBool(proto, count, context), shape, null, proto.getName(), DataType.BOOL);
        });
    }

    static int[] staticShape(OnnxProto.ValueInfoProto valueInfo) {
        String context = "value '" + valueInfo.getName() + "'";
        if (!valueInfo.hasType() || !valueInfo.getType().hasTensorType()) {
            throw new OnnxUnsupportedException(context + " is not a dense tensor value.");
        }
        OnnxProto.TensorShapeProto shape = valueInfo.getType().getTensorType().getShape();
        int[] out = new int[shape.getDimCount()];
        for (int i = 0; i < out.length; i++) {
            OnnxProto.TensorShapeProto.Dimension dim = shape.getDim(i);
            if (!dim.hasDimValue()) {
                throw new OnnxUnsupportedException(context + " uses dynamic or symbolic dimension " + i + ".");
            }
            out[i] = Math.toIntExact(dim.getDimValue());
        }
        return out;
    }

    static DataType valueInfoDataType(OnnxProto.ValueInfoProto valueInfo) {
        String context = "value '" + valueInfo.getName() + "'";
        if (!valueInfo.hasType() || !valueInfo.getType().hasTensorType()) {
            throw new OnnxUnsupportedException(context + " is not a dense tensor value.");
        }
        return OnnxDataTypes.toSynaptik(valueInfo.getType().getTensorType().getElemType(), context);
    }

    private static int[] dims(OnnxProto.TensorProto proto, String context) {
        int[] shape = new int[proto.getDimsCount()];
        for (int i = 0; i < shape.length; i++) {
            shape[i] = Math.toIntExact(proto.getDims(i));
        }
        return shape;
    }

    private static int elementCount(int[] shape) {
        int count = 1;
        for (int dim : shape) {
            count = Math.multiplyExact(count, dim);
        }
        return count;
    }

    private static float[] readFloat32(OnnxProto.TensorProto proto, int count, String context) {
        if (!proto.getRawData().isEmpty()) {
            ByteBuffer buffer = raw(proto, count * Float.BYTES, context);
            float[] out = new float[count];
            for (int i = 0; i < count; i++) {
                out[i] = buffer.getFloat();
            }
            return out;
        }
        if (proto.getFloatDataCount() != count) {
            throw wrongCount(context, count, proto.getFloatDataCount());
        }
        float[] out = new float[count];
        for (int i = 0; i < count; i++) {
            out[i] = proto.getFloatData(i);
        }
        return out;
    }

    private static double[] readFloat64(OnnxProto.TensorProto proto, int count, String context) {
        if (!proto.getRawData().isEmpty()) {
            ByteBuffer buffer = raw(proto, count * Double.BYTES, context);
            double[] out = new double[count];
            for (int i = 0; i < count; i++) {
                out[i] = buffer.getDouble();
            }
            return out;
        }
        if (proto.getDoubleDataCount() != count) {
            throw wrongCount(context, count, proto.getDoubleDataCount());
        }
        double[] out = new double[count];
        for (int i = 0; i < count; i++) {
            out[i] = proto.getDoubleData(i);
        }
        return out;
    }

    private static short[] readBfloat16(OnnxProto.TensorProto proto, int count, String context) {
        if (!proto.getRawData().isEmpty()) {
            ByteBuffer buffer = raw(proto, count * Short.BYTES, context);
            short[] out = new short[count];
            for (int i = 0; i < count; i++) {
                out[i] = buffer.getShort();
            }
            return out;
        }
        if (proto.getInt32DataCount() != count) {
            throw wrongCount(context, count, proto.getInt32DataCount());
        }
        short[] out = new short[count];
        for (int i = 0; i < count; i++) {
            out[i] = (short) proto.getInt32Data(i);
        }
        return out;
    }

    private static int[] readInt32(OnnxProto.TensorProto proto, int count, String context) {
        if (!proto.getRawData().isEmpty()) {
            ByteBuffer buffer = raw(proto, count * Integer.BYTES, context);
            int[] out = new int[count];
            for (int i = 0; i < count; i++) {
                out[i] = buffer.getInt();
            }
            return out;
        }
        if (proto.getInt32DataCount() != count) {
            throw wrongCount(context, count, proto.getInt32DataCount());
        }
        int[] out = new int[count];
        for (int i = 0; i < count; i++) {
            out[i] = proto.getInt32Data(i);
        }
        return out;
    }

    private static byte[] readBool(OnnxProto.TensorProto proto, int count, String context) {
        byte[] out = new byte[count];
        if (!proto.getRawData().isEmpty()) {
            ByteString raw = proto.getRawData();
            if (raw.size() != count) {
                throw wrongCount(context, count, raw.size());
            }
            for (int i = 0; i < count; i++) {
                out[i] = raw.byteAt(i) == 0 ? (byte) 0 : (byte) 1;
            }
            return out;
        }
        if (proto.getInt32DataCount() != count) {
            throw wrongCount(context, count, proto.getInt32DataCount());
        }
        for (int i = 0; i < count; i++) {
            out[i] = proto.getInt32Data(i) == 0 ? (byte) 0 : (byte) 1;
        }
        return out;
    }

    private static long[] readInt64(OnnxProto.TensorProto proto, int count, String context) {
        if (!proto.getRawData().isEmpty()) {
            ByteBuffer buffer = raw(proto, count * Long.BYTES, context);
            long[] out = new long[count];
            for (int i = 0; i < count; i++) {
                out[i] = buffer.getLong();
            }
            return out;
        }
        if (proto.getInt64DataCount() != count) {
            throw wrongCount(context, count, proto.getInt64DataCount());
        }
        long[] out = new long[count];
        for (int i = 0; i < count; i++) {
            out[i] = proto.getInt64Data(i);
        }
        return out;
    }

    private static ByteBuffer raw(OnnxProto.TensorProto proto, int expectedBytes, String context) {
        byte[] bytes = proto.getRawData().toByteArray();
        if (bytes.length != expectedBytes) {
            throw new OnnxUnsupportedException(context + " raw_data has " + bytes.length
                    + " bytes, expected " + expectedBytes + ".");
        }
        return ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
    }

    private static OnnxUnsupportedException wrongCount(String context, int expected, int actual) {
        return new OnnxUnsupportedException(context + " has " + actual + " values, expected " + expected + ".");
    }

    private static short[] logicalBfloat16(Tensor tensor) {
        if (tensor.isContiguous() && !tensor.hasStorageOffset()) {
            return Arrays.copyOf(TensorInternalAccess.bfloat16Data(tensor), tensor.getFlatDataSize());
        }
        double[] values = tensor.toDoubleArrayCopy();
        short[] out = new short[values.length];
        for (int i = 0; i < values.length; i++) {
            out[i] = CpuDTypeOps.toBFloat16Bits((float) values[i]);
        }
        return out;
    }

    private static int[] logicalInt32(Tensor tensor) {
        if (tensor.isContiguous() && !tensor.hasStorageOffset()) {
            return Arrays.copyOf(TensorInternalAccess.int32Data(tensor), tensor.getFlatDataSize());
        }
        double[] values = tensor.toDoubleArrayCopy();
        int[] out = new int[values.length];
        for (int i = 0; i < values.length; i++) {
            out[i] = (int) values[i];
        }
        return out;
    }

    private static long[] logicalInt64(Tensor tensor) {
        if (tensor.isContiguous() && !tensor.hasStorageOffset()) {
            return Arrays.copyOf(TensorInternalAccess.int64Data(tensor), tensor.getFlatDataSize());
        }
        long[] out = new long[tensor.getFlatDataSize()];
        for (int i = 0; i < out.length; i++) {
            out[i] = tensor.getInt64ByFlatIndex(i);
        }
        return out;
    }

    record ImportedConstant(Tensor tensor, long[] int64Values) {
        static ImportedConstant tensor(Tensor tensor) {
            long[] int64Values = tensor != null && tensor.getDataType() == DataType.INT64
                    ? logicalInt64(tensor)
                    : null;
            return new ImportedConstant(tensor, int64Values);
        }

        boolean isTensor() {
            return tensor != null;
        }
    }
}
