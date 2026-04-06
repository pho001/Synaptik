package tensor;

public enum DataType {
    FLOAT64, // double v Javě
    FLOAT32, // float v Javě
    BFLOAT16, // v Javě se často reprezentuje jako short[]
    INT32,   // integer index tensor stored explicitly
    BOOL     // mask/boolean tensor stored explicitly
}
