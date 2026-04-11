package backend.kernels.cpu.elementwise;

public enum LogicalBinaryOp {
    AND {
        @Override
        public byte apply(byte left, byte right) {
            return (left != 0 && right != 0) ? (byte) 1 : (byte) 0;
        }
    },
    OR {
        @Override
        public byte apply(byte left, byte right) {
            return (left != 0 || right != 0) ? (byte) 1 : (byte) 0;
        }
    };

    public abstract byte apply(byte left, byte right);
}
