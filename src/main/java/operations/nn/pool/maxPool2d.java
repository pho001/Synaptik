package operations.nn.pool;
import operations.Operation;

import tensor.options.Pool2dOptions;

/**
 * Describes NCHW 2-D maximum pooling.
 *
 * <p>{@link Pool2dOptions} supplies kernel, stride, padding, and average-count
 * semantics. The descriptor does not own tensor storage.</p>
 */
public final class maxPool2d implements Operation {
    private final Pool2dOptions options;

    /**
     * Creates a pooling descriptor.
     *
     * @param options non-null pooling window and stride options
     * @throws IllegalArgumentException if {@code options} is {@code null}
     */
    public maxPool2d(Pool2dOptions options) {
        if (options == null) {
            throw new IllegalArgumentException("options cannot be null");
        }
        this.options = options;
    }

    /**
     * Returns the pooling options.
     *
     * @return non-null kernel, stride, padding, and count settings
     */
    public Pool2dOptions getOptions() {
        return options;
    }

    @Override
    public OpType opType() {
        return OpType.MAX_POOL2D;
    }

    @Override
    public OpArityClass arityClass() {
        return OpArityClass.SPECIAL;
    }

    @Override
    public boolean isFusable() {
        return false;
    }

    @Override
    public OpSemanticFamily semanticFamily() {
        return OpSemanticFamily.SPECIAL;
    }

    @Override
    public OpComputationalCost computationalCost() {
        return OpComputationalCost.EXPENSIVE;
    }

    @Override
    public OpControlTrait controlTrait() {
        return OpControlTrait.NONE;
    }

    @Override
    public OpResultKind resultKind() {
        return OpResultKind.NUMERIC;
    }

    @Override
    public String getExpression() {
        return "maxPool2d";
    }
}
