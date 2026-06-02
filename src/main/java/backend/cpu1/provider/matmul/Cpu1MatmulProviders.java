package backend.cpu1.provider.matmul;

import java.util.Objects;

/**
 * Explicit provider selector for cpu1 matmul routes.
 */
public final class Cpu1MatmulProviders {
    private static final Cpu1MatmulProvider JAVA_SCALAR = new Cpu1JavaScalarMatmulProvider();
    private static final Cpu1MatmulProvider OPENBLAS_ARRAY = new Cpu1OpenBlasArrayMatmulProvider();

    private Cpu1MatmulProviders() {
    }

    public static Cpu1MatmulProvider forRoute(Cpu1MatmulRoute route) {
        Objects.requireNonNull(route, "route cannot be null");
        return switch (route) {
            case JAVA_SCALAR -> JAVA_SCALAR;
            case OPENBLAS_ARRAY_COPYING -> OPENBLAS_ARRAY;
            case AUTO -> throw new UnsupportedOperationException(
                    "cpu1 matmul route " + route + " must be resolved before provider selection."
            );
            case OPENBLAS_NATIVE_SEGMENT -> throw new UnsupportedOperationException(
                    "cpu1 matmul route " + route + " does not have a provider implementation yet."
            );
        };
    }
}
