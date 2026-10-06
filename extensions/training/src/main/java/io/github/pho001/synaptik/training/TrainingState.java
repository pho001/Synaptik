package io.github.pho001.synaptik.training;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.Shape;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Immutable detached in-memory snapshot of one training session's complete resumable state.
 *
 * <p>The snapshot contains exact SGD configuration, including dampening, successful execution and
 * optimizer-step counters, pending accumulation count, and one stable-path entry for every
 * parameter. Each entry owns canonical big-endian logical-parameter, master-parameter, momentum,
 * and accumulated-gradient bytes. Logical payloads use the parameter's declared type; master,
 * momentum, and accumulation payloads use {@code FLOAT64} for {@code FLOAT64} parameters and
 * {@code FLOAT32} otherwise. Successful session snapshots contain only finite represented payload
 * values; restore revalidates this invariant before installation. The snapshot retains no Module,
 * Parameter, Tensor, storage, Engine, prepared execution, backend resource, or closeable owner and
 * remains readable after the originating session closes.</p>
 *
 * <p>This is an in-memory validate-before-install handoff, not a durable checkpoint format. It
 * defines no codec, version migration, file publication, graph/random state, data cursor, epoch,
 * or scheduler state. {@link TrainingSession#restore(TrainingState)} accepts it only when the
 * complete optimizer configuration and ordered parameter path/schema match.</p>
 */
public final class TrainingState {
    private final Sgd optimizer;
    private final long executions;
    private final long optimizerSteps;
    private final long accumulatedGradientRuns;
    private final List<ParameterState> parameters;

    TrainingState(
            Sgd optimizer,
            long executions,
            long optimizerSteps,
            long accumulatedGradientRuns,
            List<ParameterState> parameters) {
        this.optimizer = Objects.requireNonNull(optimizer, "optimizer");
        if (executions < 0) {
            throw new IllegalArgumentException("executions must be non-negative");
        }
        if (optimizerSteps < 0 || optimizerSteps > executions) {
            throw new IllegalArgumentException(
                    "optimizerSteps must be in [0, executions]");
        }
        if (accumulatedGradientRuns < 0 || accumulatedGradientRuns > executions) {
            throw new IllegalArgumentException(
                    "accumulatedGradientRuns must be in [0, executions]");
        }
        this.executions = executions;
        this.optimizerSteps = optimizerSteps;
        this.accumulatedGradientRuns = accumulatedGradientRuns;
        Objects.requireNonNull(parameters, "parameters");
        if (parameters.isEmpty()) {
            throw new IllegalArgumentException("parameters must not be empty");
        }
        var snapshot = new ArrayList<ParameterState>(parameters.size());
        for (int index = 0; index < parameters.size(); index++) {
            snapshot.add(Objects.requireNonNull(
                    parameters.get(index), "parameters[" + index + "]"));
        }
        this.parameters = List.copyOf(snapshot);
    }

    /**
     * Returns the exact immutable SGD configuration captured with this state.
     *
     * @return the non-null configuration retained by exact reference
     */
    public Sgd optimizer() {
        return optimizer;
    }

    /**
     * Returns the number of successful forward/backward executions.
     *
     * @return a non-negative counter
     */
    public long executions() {
        return executions;
    }

    /**
     * Returns the number of successful parameter updates.
     *
     * @return a non-negative counter no greater than {@link #executions()}
     */
    public long optimizerSteps() {
        return optimizerSteps;
    }

    /**
     * Returns how many gradient runs are represented by each pending accumulated-gradient payload.
     *
     * @return a non-negative counter no greater than {@link #executions()}
     */
    public long accumulatedGradientRuns() {
        return accumulatedGradientRuns;
    }

    /**
     * Returns all parameter snapshots in deterministic recursive Module traversal order.
     *
     * @return the same non-null immutable non-empty list on every call
     */
    public List<ParameterState> parameters() {
        return parameters;
    }

    /**
     * Immutable detached state for one stable recursive parameter path.
     */
    public static final class ParameterState {
        private final String path;
        private final DataType dataType;
        private final Shape shape;
        private final byte[] parameterBytes;
        private final byte[] masterParameterBytes;
        private final byte[] momentumBytes;
        private final byte[] accumulatedGradientBytes;

        ParameterState(
                String path,
                DataType dataType,
                Shape shape,
                byte[] parameterBytes,
                byte[] masterParameterBytes,
                byte[] momentumBytes,
                byte[] accumulatedGradientBytes) {
            this.path = Objects.requireNonNull(path, "path");
            if (path.isBlank()) {
                throw new IllegalArgumentException("path must not be blank");
            }
            this.dataType = Objects.requireNonNull(dataType, "dataType");
            if (!dataType.isFloating()) {
                throw new IllegalArgumentException(
                        "training state requires floating data type: " + dataType);
            }
            this.shape = Objects.requireNonNull(shape, "shape");
            if (!shape.isFullyStatic()) {
                throw new IllegalArgumentException("training state shape must be fully static");
            }
            long elements = shape.knownElementCount().orElseThrow();
            long logicalBytes = Math.multiplyExact(elements, dataType.byteWidth());
            long optimizerBytes = Math.multiplyExact(
                    elements,
                    dataType == DataType.FLOAT64 ? Double.BYTES : Float.BYTES);
            if (logicalBytes > Integer.MAX_VALUE || optimizerBytes > Integer.MAX_VALUE) {
                throw new IllegalArgumentException(
                        "training state payload exceeds JVM byte[] limit");
            }
            this.parameterBytes =
                    copyExact(parameterBytes, "parameterBytes", (int) logicalBytes);
            this.masterParameterBytes = copyExact(
                    masterParameterBytes, "masterParameterBytes", (int) optimizerBytes);
            this.momentumBytes =
                    copyExact(momentumBytes, "momentumBytes", (int) optimizerBytes);
            this.accumulatedGradientBytes = copyExact(
                    accumulatedGradientBytes,
                    "accumulatedGradientBytes",
                    (int) optimizerBytes);
        }

        /**
         * Returns the stable recursive Module path associated with this entry.
         *
         * @return the non-blank path retained by this state
         */
        public String path() {
            return path;
        }

        /**
         * Returns the exact logical parameter element type.
         *
         * @return {@link DataType#FLOAT64}, {@link DataType#FLOAT32},
         *     {@link DataType#BFLOAT16}, or {@link DataType#FLOAT16}
         */
        public DataType dataType() {
            return dataType;
        }

        /**
         * Returns the exact immutable parameter Shape.
         *
         * @return the non-null fully static Shape retained by reference
         */
        public Shape shape() {
            return shape;
        }

        /**
         * Returns an independent read-only big-endian view of canonical logical parameter bytes.
         *
         * @return a fresh view with position zero and exact logical payload limit
         */
        public ByteBuffer parameterBytes() {
            return view(parameterBytes);
        }

        /**
         * Returns an independent read-only big-endian view of canonical master parameter bytes.
         *
         * <p>The payload uses {@code FLOAT64} elements for a {@code FLOAT64} parameter and
         * {@code FLOAT32} elements for every other supported floating parameter type.</p>
         *
         * @return a fresh view with position zero and exact optimizer-width payload limit
         */
        public ByteBuffer masterParameterBytes() {
            return view(masterParameterBytes);
        }

        byte[] masterParameterBytesInternal() {
            return masterParameterBytes;
        }

        /**
         * Returns an independent read-only big-endian view of the SGD momentum slot.
         *
         * @return a fresh view with position zero and exact payload limit
         */
        public ByteBuffer momentumBytes() {
            return view(momentumBytes);
        }

        /**
         * Returns an independent read-only big-endian view of pending accumulated gradients.
         *
         * @return a fresh view with position zero and exact payload limit
         */
        public ByteBuffer accumulatedGradientBytes() {
            return view(accumulatedGradientBytes);
        }

        byte[] parameterBytesInternal() {
            return parameterBytes;
        }

        byte[] momentumBytesInternal() {
            return momentumBytes;
        }

        byte[] accumulatedGradientBytesInternal() {
            return accumulatedGradientBytes;
        }

        private static byte[] copyExact(byte[] source, String name, int expected) {
            Objects.requireNonNull(source, name);
            if (source.length != expected) {
                throw new IllegalArgumentException(
                        name + " length mismatch: expected=" + expected
                                + ", actual=" + source.length);
            }
            return source.clone();
        }

        private static ByteBuffer view(byte[] bytes) {
            return ByteBuffer.wrap(bytes).asReadOnlyBuffer().order(ByteOrder.BIG_ENDIAN);
        }
    }
}
