package io.github.pho001.synaptik.training;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.Shape;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Immutable detached in-memory snapshot of one training session's complete resumable state.
 *
 * <p>The snapshot contains the exact default optimizer and ordered parameter-group configurations,
 * including dampening. It also contains successful execution and optimizer-step counters,
 * pending accumulation count, and one stable-path entry for every
 * parameter. Each entry owns canonical big-endian logical-parameter, master-parameter, first-slot,
 * optional second-moment, and accumulated-gradient bytes. The first slot is SGD momentum or
 * Adam/AdamW's first moment; only Adam/AdamW entries have a second moment. Logical payloads use
 * the parameter's declared type; optimizer slots, master, and accumulation use {@code FLOAT64}
 * for {@code FLOAT64} parameters and
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
    private final Optimizer optimizer;
    private final List<ParameterGroup> parameterGroups;
    private final long executions;
    private final long optimizerSteps;
    private final long accumulatedGradientRuns;
    private final List<ParameterState> parameters;

    /**
     * Creates an ungrouped detached state for the legacy session configuration.
     *
     * @param optimizer non-null immutable default optimizer configuration
     * @param executions non-negative successful execution count
     * @param optimizerSteps successful update count in {@code [0, executions]}
     * @param accumulatedGradientRuns pending run count in {@code [0, executions]}
     * @param parameters non-null, nonempty ordered list of non-null detached parameter states
     * @throws NullPointerException if {@code optimizer}, {@code parameters}, or an entry is null
     * @throws IllegalArgumentException if a counter is invalid or {@code parameters} is empty
     */
    TrainingState(
            Optimizer optimizer,
            long executions,
            long optimizerSteps,
            long accumulatedGradientRuns,
            List<ParameterState> parameters) {
        this(optimizer, List.of(), executions, optimizerSteps, accumulatedGradientRuns, parameters);
    }

    /**
     * Creates detached state with the exact ordered group configuration for restore matching.
     *
     * @param optimizer non-null immutable default optimizer configuration
     * @param parameterGroups non-null ordered list of non-null immutable groups, copied on capture
     * @param executions non-negative successful execution count
     * @param optimizerSteps successful update count in {@code [0, executions]}
     * @param accumulatedGradientRuns pending run count in {@code [0, executions]}
     * @param parameters non-null, nonempty ordered list of non-null detached parameter states
     * @throws NullPointerException if {@code optimizer}, {@code parameterGroups},
     *     {@code parameters}, or an entry is null
     * @throws IllegalArgumentException if a counter is invalid or {@code parameters} is empty
     */
    TrainingState(
            Optimizer optimizer,
            List<ParameterGroup> parameterGroups,
            long executions,
            long optimizerSteps,
            long accumulatedGradientRuns,
            List<ParameterState> parameters) {
        this.optimizer = Objects.requireNonNull(optimizer, "optimizer");
        this.parameterGroups = List.copyOf(Objects.requireNonNull(parameterGroups, "parameterGroups"));
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
     * Returns the exact immutable default optimizer configuration captured with this state.
     *
     * <p>The return type is {@link Optimizer}, including Adam and AdamW. This intentionally
     * replaces the earlier {@code Sgd} method signature; SGD-specific callers may cast after
     * checking the algorithm kind. There is no second default-optimizer accessor.</p>
     *
     * @return the non-null configuration retained by exact reference
     */
    public Optimizer optimizer() {
        return optimizer;
    }

    /**
     * Returns the detached ordered group configuration captured with this state.
     *
     * <p>Both group order and the path order within each group participate in restore
     * compatibility. An ungrouped snapshot has an empty list.</p>
     *
     * @return non-null immutable list of immutable groups, possibly empty
     */
    public List<ParameterGroup> parameterGroups() {
        return parameterGroups;
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
     *
     * <p>Canonical payload bytes are private, exact-width values exposed only through fresh
     * read-only views. This state owns its bytes independently of parameter storage and remains
     * readable after the originating session closes.</p>
     */
    public static final class ParameterState {
        private enum PayloadOwnership {
            COPY,
            ADOPT
        }

        private final String path;
        private final DataType dataType;
        private final Shape shape;
        private final byte[] parameterBytes;
        private final byte[] masterParameterBytes;
        private final byte[] momentumBytes;
        private final byte[] secondMomentBytes;
        private final byte[] accumulatedGradientBytes;

        /**
         * Defensively captures caller-supplied canonical payloads for one parameter.
         *
         * <p>Each array is checked against the declared logical or optimizer width and copied, so
         * later writes through a caller-retained reference cannot change this state.</p>
         *
         * @param path non-null, non-blank stable recursive parameter path
         * @param dataType non-null supported floating parameter type
         * @param shape non-null fully static parameter Shape
         * @param parameterBytes non-null logical-width canonical bytes, copied on construction
         * @param masterParameterBytes non-null optimizer-width canonical bytes, copied on construction
         * @param momentumBytes non-null first-slot optimizer-width canonical bytes, copied
         * @param accumulatedGradientBytes non-null optimizer-width canonical bytes, copied on construction
         * @throws NullPointerException if any argument is null
         * @throws IllegalArgumentException if the path, type, Shape, or a payload width is invalid
         */
        ParameterState(
                String path,
                DataType dataType,
                Shape shape,
                byte[] parameterBytes,
                byte[] masterParameterBytes,
                byte[] momentumBytes,
                byte[] accumulatedGradientBytes) {
            this(path, dataType, shape, parameterBytes, masterParameterBytes, momentumBytes,
                    null, accumulatedGradientBytes, PayloadOwnership.COPY);
        }

        /**
         * Defensively captures a state with an Adam/AdamW second-moment payload.
         *
         * @param path non-null stable recursive path
         * @param dataType non-null supported floating type
         * @param shape non-null fully static shape
         * @param parameterBytes non-null logical-width canonical bytes, copied
         * @param masterParameterBytes non-null optimizer-width canonical bytes, copied
         * @param firstMomentBytes non-null optimizer-width canonical bytes, copied
         * @param secondMomentBytes non-null optimizer-width canonical bytes, copied
         * @param accumulatedGradientBytes non-null optimizer-width canonical bytes, copied
         * @throws NullPointerException if a required argument is null
         * @throws IllegalArgumentException if schema or payload widths are invalid
         */
        ParameterState(String path, DataType dataType, Shape shape, byte[] parameterBytes,
                byte[] masterParameterBytes, byte[] firstMomentBytes, byte[] secondMomentBytes,
                byte[] accumulatedGradientBytes) {
            this(path, dataType, shape, parameterBytes, masterParameterBytes, firstMomentBytes,
                    Objects.requireNonNull(secondMomentBytes, "secondMomentBytes"),
                    accumulatedGradientBytes, PayloadOwnership.COPY);
        }

        /**
         * Takes exclusive ownership of fresh canonical arrays filled by a session snapshot.
         *
         * <p>Only a producer that retains no array reference may use this package-private path.
         * Payload widths and non-null inputs receive the same validation as the defensive
         * constructor; public views still cannot expose the mutable arrays.</p>
         *
         * @param path non-null, non-blank stable recursive parameter path
         * @param dataType non-null supported floating parameter type
         * @param shape non-null fully static parameter Shape
         * @param parameterBytes non-null exclusively owned logical-width canonical bytes
         * @param masterParameterBytes non-null exclusively owned optimizer-width canonical bytes
         * @param momentumBytes non-null exclusively owned optimizer-width canonical bytes
         * @param accumulatedGradientBytes non-null exclusively owned optimizer-width canonical bytes
         * @return a detached immutable state retaining the supplied arrays without copying
         * @throws NullPointerException if any argument is null
         * @throws IllegalArgumentException if the path, type, Shape, or a payload width is invalid
         */
        static ParameterState adoptFreshSnapshotArrays(
                String path,
                DataType dataType,
                Shape shape,
                byte[] parameterBytes,
                byte[] masterParameterBytes,
                byte[] momentumBytes,
                byte[] accumulatedGradientBytes) {
            return new ParameterState(path, dataType, shape, parameterBytes, masterParameterBytes,
                    momentumBytes, null, accumulatedGradientBytes, PayloadOwnership.ADOPT);
        }

        /**
         * Takes exclusive ownership of fresh canonical payload arrays for an adaptive snapshot.
         *
         * @param path non-null stable recursive path
         * @param dataType non-null supported floating type
         * @param shape non-null fully static shape
         * @param parameterBytes non-null logical-width bytes owned exclusively by the caller
         * @param masterParameterBytes non-null optimizer-width master bytes owned exclusively
         * @param firstMomentBytes non-null optimizer-width first-moment bytes owned exclusively
         * @param secondMomentBytes non-null optimizer-width second-moment bytes owned exclusively
         * @param accumulatedGradientBytes non-null optimizer-width pending-sum bytes owned exclusively
         * @return detached immutable state adopting the supplied arrays
         * @throws NullPointerException if a required argument is null
         * @throws IllegalArgumentException if schema or payload widths are invalid
         */
        static ParameterState adoptFreshSnapshotArrays(String path, DataType dataType, Shape shape,
                byte[] parameterBytes, byte[] masterParameterBytes, byte[] firstMomentBytes,
                byte[] secondMomentBytes, byte[] accumulatedGradientBytes) {
            Objects.requireNonNull(secondMomentBytes, "secondMomentBytes");
            return new ParameterState(path, dataType, shape, parameterBytes, masterParameterBytes,
                    firstMomentBytes, secondMomentBytes, accumulatedGradientBytes,
                    PayloadOwnership.ADOPT);
        }

        private ParameterState(
                String path,
                DataType dataType,
                Shape shape,
                byte[] parameterBytes,
                byte[] masterParameterBytes,
                byte[] momentumBytes,
                byte[] secondMomentBytes,
                byte[] accumulatedGradientBytes,
                PayloadOwnership ownership) {
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
            this.parameterBytes = exactPayload(
                    parameterBytes, "parameterBytes", (int) logicalBytes, ownership);
            this.masterParameterBytes = exactPayload(
                    masterParameterBytes, "masterParameterBytes", (int) optimizerBytes, ownership);
            this.momentumBytes = exactPayload(
                    momentumBytes, "momentumBytes", (int) optimizerBytes, ownership);
            this.secondMomentBytes = secondMomentBytes == null ? null : exactPayload(
                    secondMomentBytes, "secondMomentBytes", (int) optimizerBytes, ownership);
            this.accumulatedGradientBytes = exactPayload(
                    accumulatedGradientBytes,
                    "accumulatedGradientBytes",
                    (int) optimizerBytes,
                    ownership);
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
         * Returns an independent read-only big-endian view of the first optimizer slot.
         *
         * <p>This compatibility accessor is SGD momentum or Adam/AdamW's first moment.</p>
         *
         * @return a fresh view with position zero and exact payload limit
         */
        public ByteBuffer momentumBytes() {
            return view(momentumBytes);
        }

        /**
         * Returns an independent read-only view of the first optimizer slot.
         *
         * @return a fresh big-endian view of SGD momentum or Adam/AdamW's first moment
         */
        public ByteBuffer firstMomentBytes() {
            return view(momentumBytes);
        }

        /**
         * Returns an independent read-only view of an adaptive second-moment slot, when present.
         *
         * @return a fresh big-endian view for Adam/AdamW, or empty for SGD
         */
        public Optional<ByteBuffer> secondMomentBytes() {
            return secondMomentBytes == null ? Optional.empty() : Optional.of(view(secondMomentBytes));
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

        byte[] secondMomentBytesInternal() {
            return secondMomentBytes;
        }

        byte[] accumulatedGradientBytesInternal() {
            return accumulatedGradientBytes;
        }

        private static byte[] exactPayload(
                byte[] source, String name, int expected, PayloadOwnership ownership) {
            Objects.requireNonNull(source, name);
            if (source.length != expected) {
                throw new IllegalArgumentException(
                        name + " length mismatch: expected=" + expected
                                + ", actual=" + source.length);
            }
            return ownership == PayloadOwnership.COPY ? source.clone() : source;
        }

        private static ByteBuffer view(byte[] bytes) {
            return ByteBuffer.wrap(bytes).asReadOnlyBuffer().order(ByteOrder.BIG_ENDIAN);
        }
    }
}
