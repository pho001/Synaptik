package onnx;

import tensor.DataType;

final class OnnxDataTypes {
    private OnnxDataTypes() {
    }

    static int toOnnx(DataType dataType) {
        return switch (dataType) {
            case FLOAT32 -> OnnxProto.TensorProto.DataType.FLOAT.getNumber();
            case FLOAT64 -> OnnxProto.TensorProto.DataType.DOUBLE.getNumber();
            case BFLOAT16 -> OnnxProto.TensorProto.DataType.BFLOAT16.getNumber();
            case INT32 -> OnnxProto.TensorProto.DataType.INT32.getNumber();
            case INT64 -> OnnxProto.TensorProto.DataType.INT64.getNumber();
            case BOOL -> OnnxProto.TensorProto.DataType.BOOL.getNumber();
        };
    }

    static DataType toSynaptik(int onnxDataType, String context) {
        OnnxProto.TensorProto.DataType type = OnnxProto.TensorProto.DataType.forNumber(onnxDataType);
        if (type == null) {
            throw new OnnxUnsupportedException(context + " uses unknown ONNX dtype id " + onnxDataType + ".");
        }
        return switch (type) {
            case FLOAT -> DataType.FLOAT32;
            case DOUBLE -> DataType.FLOAT64;
            case BFLOAT16 -> DataType.BFLOAT16;
            case INT32 -> DataType.INT32;
            case INT64 -> DataType.INT64;
            case BOOL -> DataType.BOOL;
            default -> throw new OnnxUnsupportedException(context + " uses unsupported ONNX dtype " + type + ".");
        };
    }

    static boolean isInt64(int onnxDataType) {
        return onnxDataType == OnnxProto.TensorProto.DataType.INT64.getNumber();
    }
}
