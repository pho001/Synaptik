package backend.kernels.cpu.elementwise;

public enum CompareOp {
    GT {
        @Override
        public boolean test(double left, double right) { return left > right; }

        @Override
        public boolean test(float left, float right) { return left > right; }
    },
    GE {
        @Override
        public boolean test(double left, double right) { return left >= right; }

        @Override
        public boolean test(float left, float right) { return left >= right; }
    },
    LT {
        @Override
        public boolean test(double left, double right) { return left < right; }

        @Override
        public boolean test(float left, float right) { return left < right; }
    },
    LE {
        @Override
        public boolean test(double left, double right) { return left <= right; }

        @Override
        public boolean test(float left, float right) { return left <= right; }
    },
    EQ {
        @Override
        public boolean test(double left, double right) { return left == right; }

        @Override
        public boolean test(float left, float right) { return left == right; }
    },
    NE {
        @Override
        public boolean test(double left, double right) { return left != right; }

        @Override
        public boolean test(float left, float right) { return left != right; }
    };

    public abstract boolean test(double left, double right);

    public abstract boolean test(float left, float right);
}
