package backend.kernels.cpu.elementwise;

public enum LogicalUnaryOp {
    NOT {
        @Override
        public byte apply(byte value) {
            return value == 0 ? (byte) 1 : (byte) 0;
        }
    };

    public abstract byte apply(byte value);
}
