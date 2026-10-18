package io.github.pho001.synaptik.checkpoint.graph;

import io.github.pho001.synaptik.checkpoint.artifact.ArtifactEnvelope;
import io.github.pho001.synaptik.checkpoint.artifact.ArtifactFormatException;
import io.github.pho001.synaptik.checkpoint.artifact.ArtifactLimits;
import io.github.pho001.synaptik.compiler.GraphCompilationPort;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.CompiledGraphModel;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphPhase;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformKind;
import io.github.pho001.synaptik.model.operation.layout.PermutationAttrs;
import io.github.pho001.synaptik.model.operation.linalg.MatmulKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.nio.BufferUnderflowException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Optional;

/**
 * Closed graph-model.v1 wire on the existing opaque envelope. IDs, flags and field order are
 * explicit and independent of Java enum ordinals. All parsing uses bounded views; only final
 * candidate leaves and Compiler reconstruction create Tensors, after complete stored validation.
 */
final class GraphArtifactCodec {
    private static final long MAGIC = 0x53594e4752463031L;
    private static final String KIND = "graph-model.v1";

    private GraphArtifactCodec() { }

    /**
     * Dry-sizes all metadata and six encoded bounds before storage admission or bulk copying.
     * @param source fully metadata-validated normalized source
     * @param limits explicit encoded ceilings
     * @return complete detached canonical image, ready for unchanged ArtifactFile publication
     * @throws IllegalArgumentException if an encoded bound or BOOL lane is invalid
     * @throws ArithmeticException if checked sizing overflows
     * @throws IllegalStateException if named host state is absent/dead/inaccessible
     */
    static ArtifactEnvelope encode(GraphSource source, ArtifactLimits limits) {
        Sink dry = new Sink(null);
        write(dry, source);
        long stateBytes = 0;
        for (Tensor state : source.states) {
            int length = GraphValueStorage.payloadLength(state.descriptor());
            if (length > limits.maxPayloadBytes()) throw new IllegalArgumentException("State payload exceeds limits");
            stateBytes = Math.addExact(stateBytes, length);
        }
        preflight(source.states.size(), dry.size, stateBytes, limits);
        var segments = GraphValueStorage.admit(source.states);
        GraphValueStorage.validateBools(source.states, segments);
        ByteBuffer schema = ByteBuffer.allocate(Math.toIntExact(dry.size)).order(ByteOrder.BIG_ENDIAN);
        write(new Sink(schema), source);
        var entries = new ArrayList<ArtifactEnvelope.Entry>();
        entries.add(new ArtifactEnvelope.Entry("schema", schema.array()));
        for (int i = 0; i < source.states.size(); i++)
            entries.add(new ArtifactEnvelope.Entry(ordinal(i),
                    GraphValueStorage.encode(source.states.get(i).descriptor(), segments.get(i))));
        return new ArtifactEnvelope(KIND, entries);
    }

    /**
     * Independently enforces all six equivalent file limits, before schema encoding or value copies.
     * @param states unique state count, at most 100000000
     * @param schema exact schema bytes
     * @param stateBytes sum of unique logical state bytes
     * @param limits non-null encoded ceilings
     * @throws IllegalArgumentException if any limit is exceeded
     * @throws ArithmeticException if checked aggregate/manifest/file sizing overflows
     */
    private static void preflight(int states, long schema, long stateBytes, ArtifactLimits limits) {
        long manifest = Math.addExact(72L, Math.multiplyExact(58L, states));
        long aggregate = Math.addExact(schema, stateBytes);
        long file = Math.addExact(48L, Math.addExact(manifest, aggregate));
        if (states > 100_000_000 || states + 1L > limits.maxEntries() || limits.maxKeyBytes() < 14
                || schema > Integer.MAX_VALUE - 8 || schema > limits.maxPayloadBytes()
                || manifest > limits.maxManifestBytes() || aggregate > limits.maxAggregatePayloadBytes()
                || file > limits.maxFileBytes())
            throw new IllegalArgumentException("Graph artifact exceeds caller limits");
    }

    /**
     * Reads complete closed wire before candidate creation, host import or ordinary reconstruction.
     * @param envelope immutable detached image, not trusted as a validation credential
     * @param limits caller's explicit file-equivalent ceilings
     * @return independent final source without original storage, saved batches or a second copy
     * @throws ArtifactFormatException if wire, bounds, aliases, BOOL, topology or Compiler admission fails
     * @throws IllegalStateException if Tensor identifiers are exhausted; never translated
     */
    static GraphCheckpoint.Source decode(ArtifactEnvelope envelope, ArtifactLimits limits)
            throws ArtifactFormatException {
        try {
            if (!KIND.equals(envelope.kind()) || envelope.entries().isEmpty()
                    || !envelope.entries().getFirst().key().equals("schema"))
                throw invalid("Expected graph-model.v1 schema");
            long stateBytes = 0;
            for (int i = 1; i < envelope.entries().size(); i++) {
                var entry = envelope.entries().get(i);
                if (!entry.key().equals(ordinal(i - 1)) || entry.size() > limits.maxPayloadBytes())
                    throw invalid("Unexpected state key/length");
                stateBytes = Math.addExact(stateBytes, entry.size());
            }
            preflight(envelope.entries().size() - 1, envelope.entries().getFirst().size(), stateBytes, limits);
            Reader reader = new Reader(envelope.entries().getFirst().payload());
            ByteBuffer in = reader.bytes;
            if (in.getLong() != MAGIC || in.getInt() != 1) throw invalid("Unknown graph magic/version");
            int inputCount = reader.count(7);
            var values = new ArrayList<GraphValue>();
            var inputIds = new ArrayList<ValueId>();
            for (int i = 0; i < inputCount; i++) {
                TensorDescriptor descriptor = reader.descriptor();
                GraphValueStorage.leafBytes(descriptor);
                ValueId id = new ValueId(i);
                values.add(new GraphValue(id, descriptor));
                inputIds.add(id);
            }
            int stateCount = reader.count(16);
            if (stateCount > 100_000_000 || stateCount != envelope.entries().size() - 1)
                throw invalid("State count/payload mismatch");
            int boundary = Math.addExact(inputCount, stateCount);
            var roles = new ArrayList<GraphCheckpoint.Role>(stateCount);
            for (int i = 0; i < stateCount; i++) {
                TensorDescriptor descriptor = reader.descriptor();
                var role = switch (reader.u8()) {
                    case 1 -> GraphCheckpoint.Role.PARAMETER;
                    case 2 -> GraphCheckpoint.Role.BUFFER;
                    case 3 -> GraphCheckpoint.Role.CONSTANT;
                    default -> throw invalid("Unknown state role");
                };
                if (role == GraphCheckpoint.Role.PARAMETER
                        && (!descriptor.dataType().isFloating() || !descriptor.requiresGrad()))
                    throw invalid("PARAMETER requires floating requiresGrad=true");
                long length = in.getLong();
                if (length != GraphValueStorage.payloadLength(descriptor)
                        || length != envelope.entries().get(i + 1).size())
                    throw invalid("Noncanonical state payload length");
                if (descriptor.dataType() == DataType.BOOL) {
                    ByteBuffer payload = envelope.entries().get(i + 1).payload();
                    while (payload.hasRemaining()) {
                        byte value = payload.get();
                        if (value != 0 && value != 1) throw invalid("Noncanonical BOOL payload");
                    }
                }
                roles.add(role);
                ValueId id = new ValueId(Math.addExact(inputCount, i));
                values.add(new GraphValue(id, descriptor));
                inputIds.add(id);
            }
            int registrationCount = reader.count(9);
            var names = new ArrayList<String>(registrationCount);
            var registrations = new ArrayList<Integer>(registrationCount);
            var uniqueNames = new HashSet<String>();
            int nextState = 0;
            for (int i = 0; i < registrationCount; i++) {
                String name = reader.name();
                if (!uniqueNames.add(name)) throw invalid("Duplicate leaf name");
                int index = in.getInt();
                if (index < 0 || index >= stateCount || index > nextState)
                    throw invalid("Noncanonical alias index/order");
                if (index == nextState) nextState++;
                names.add(name);
                registrations.add(index);
            }
            if (nextState != stateCount) throw invalid("Unregistered state");
            for (String name : names)
                for (int dot = name.indexOf('.'); dot >= 0; dot = name.indexOf('.', dot + 1))
                    if (uniqueNames.contains(name.substring(0, dot))) throw invalid("Leaf prefix collision");
            int nodeCount = reader.count(13);
            Math.addExact(boundary, nodeCount);
            var nodes = new ArrayList<CompiledNode>(nodeCount);
            var phases = new HashMap<NodeId, GraphPhase>();
            for (int i = 0; i < nodeCount; i++) {
                if (reader.u8() != 1) throw invalid("Unknown source phase");
                int op = reader.u8();
                int arity = switch (op) {
                    case 1, 2, 4 -> 2;
                    case 3, 5 -> 1;
                    default -> throw invalid("Unknown source operation");
                };
                var args = new ArrayList<ValueId>(arity);
                int result = Math.addExact(boundary, i);
                for (int j = 0; j < arity; j++) args.add(reader.reference(result));
                Operation operation = switch (op) {
                    case 1 -> new Operation(BinaryArithmeticKind.ADD, NoOperationAttrs.INSTANCE);
                    case 2 -> new Operation(BinaryArithmeticKind.MUL, NoOperationAttrs.INSTANCE);
                    case 3 -> new Operation(UnaryElementwiseKind.NEG, NoOperationAttrs.INSTANCE);
                    case 4 -> new Operation(MatmulKind.MATMUL, NoOperationAttrs.INSTANCE);
                    case 5 -> {
                        if (in.getInt() != 2 || in.getInt() != 1 || in.getInt() != 0)
                            throw invalid("Only rank-two PERMUTE[1,0] is supported");
                        yield new Operation(AxisTransformKind.PERMUTE, new PermutationAttrs(List.of(1, 0)));
                    }
                    default -> throw new AssertionError();
                };
                ValueId id = new ValueId(result);
                values.add(new GraphValue(id, reader.descriptor()));
                NodeId nodeId = new NodeId(i);
                nodes.add(new CompiledNode(nodeId, operation, args, List.of(id)));
                phases.put(nodeId, GraphPhase.FORWARD);
            }
            int outputCount = reader.count(4);
            if (outputCount == 0) throw invalid("Empty output boundary");
            var outputs = new ArrayList<ValueId>(outputCount);
            for (int i = 0; i < outputCount; i++) outputs.add(reader.reference(values.size()));
            if (in.hasRemaining()) throw invalid("Trailing graph schema bytes");
            var graph = new CompiledGraphModel(values, nodes, inputIds, outputs, phases);
            canonicalTopology(graph, inputCount);
            // Candidate leaves are final objects, still storage-free during complete Compiler preflight.
            var leaves = new ArrayList<Tensor>(boundary);
            for (int i = 0; i < boundary; i++) leaves.add(TensorFactory.create(values.get(i).descriptor()));
            new GraphCompilationPort.SourceCapture(graph, leaves);
            for (int i = 0; i < stateCount; i++) {
                Tensor state = leaves.get(inputCount + i);
                state.replaceHostStorage(GraphValueStorage.decode(state.descriptor(),
                        envelope.entries().get(i + 1).payload()));
            }
            List<Tensor> reconstructed = GraphCompilationPort.reconstructSource(graph, leaves);
            var named = new ArrayList<GraphCheckpoint.Leaf>(registrationCount);
            for (int i = 0; i < registrationCount; i++) {
                int index = registrations.get(i);
                named.add(new GraphCheckpoint.Leaf(names.get(i), roles.get(index), leaves.get(inputCount + index)));
            }
            var source = new GraphCheckpoint.Source(leaves.subList(0, inputCount), reconstructed, named);
            if (!GraphSource.capture(source).graph.equals(graph)) throw invalid("Noncanonical source recapture");
            return source;
        } catch (IllegalArgumentException | ArithmeticException | BufferUnderflowException failure) {
            throw new ArtifactFormatException("Invalid graph source metadata or bounds", failure);
        }
    }

    /**
     * Checks reachability and capture's input-first depth-first postorder without constructing Tensors.
     * @param graph structurally closed normalized Model graph
     * @param declaredInputs number of explicit slots allowed to be unused
     * @throws IllegalArgumentException if a node/state is dead or node order is noncanonical
     */
    private static void canonicalTopology(CompiledGraphModel graph, int declaredInputs) {
        boolean[] seen = new boolean[graph.values().size()];
        var stack = new ArrayDeque<Integer>();
        int boundary = graph.inputs().size();
        int nextNode = 0;
        for (ValueId root : graph.outputs()) {
            stack.push(Math.toIntExact(root.value()));
            while (!stack.isEmpty()) {
                int value = stack.pop();
                if (value < 0) {
                    if (~value != nextNode++) throw invalid("Noncanonical source node order");
                    continue;
                }
                if (seen[value]) continue;
                seen[value] = true;
                if (value < boundary) continue;
                int node = value - boundary;
                stack.push(~node);
                var args = graph.nodes().get(node).inputs();
                for (int i = args.size() - 1; i >= 0; i--) stack.push(Math.toIntExact(args.get(i).value()));
            }
        }
        for (int i = declaredInputs; i < seen.length; i++)
            if (!seen[i]) throw invalid("Unreachable state or dead node");
    }

    /**
     * Uses the exact same bounded field traversal for dry sizing and encoding.
     * @param out dry or previously bounded encoding sink
     * @param source normalized metadata-validated association
     */
    private static void write(Sink out, GraphSource source) {
        out.i64(MAGIC); out.i32(1);
        out.i32(source.source.inputs().size());
        for (Tensor input : source.source.inputs()) out.descriptor(input.descriptor());
        out.i32(source.states.size());
        for (int i = 0; i < source.states.size(); i++) {
            var descriptor = source.states.get(i).descriptor();
            out.descriptor(descriptor);
            out.u8(switch (source.roles.get(i)) { case PARAMETER -> 1; case BUFFER -> 2; case CONSTANT -> 3; });
            out.i64(GraphValueStorage.leafBytes(descriptor));
        }
        out.i32(source.source.leaves().size());
        for (int i = 0; i < source.source.leaves().size(); i++) {
            out.name(source.source.leaves().get(i).name());
            out.i32(source.registrations.get(i));
        }
        out.i32(source.graph.nodes().size());
        for (var node : source.graph.nodes()) {
            out.u8(1);
            var kind = node.operation().kind();
            int op = kind == BinaryArithmeticKind.ADD ? 1 : kind == BinaryArithmeticKind.MUL ? 2
                    : kind == UnaryElementwiseKind.NEG ? 3 : kind == MatmulKind.MATMUL ? 4 : 5;
            out.u8(op);
            for (ValueId id : node.inputs()) out.i32(Math.toIntExact(id.value()));
            if (op == 5) { out.i32(2); out.i32(1); out.i32(0); }
            out.descriptor(source.graph.values().get(Math.toIntExact(node.outputs().getFirst().value())).descriptor());
        }
        out.i32(source.graph.outputs().size());
        for (ValueId id : source.graph.outputs()) out.i32(Math.toIntExact(id.value()));
    }

    /**
     * Formats only the fixed decimal wire ordinal, never a user name or Tensor ID.
     * @param index zero-based state ordinal in 0..99999999
     * @return exact fourteen-byte ASCII payload key
     * @throws IllegalArgumentException if the ordinal is out of range
     */
    private static String ordinal(int index) {
        if (index < 0 || index >= 100_000_000) throw invalid("State ordinal out of range");
        return "state/" + String.format(java.util.Locale.ROOT, "%08d", index);
    }

    /**
     * Creates a semantic failure translated by decode but left unchecked on source/save paths.
     * @param message precise non-null rejected-field diagnostic
     * @return new unchecked invalid-source failure
     */
    private static IllegalArgumentException invalid(String message) { return new IllegalArgumentException(message); }

    /** Bounded-view wire reader; no raw length controls allocation without a remaining-byte check. */
    private static final class Reader {
        private final ByteBuffer bytes;

        /**
         * Borrows schema bytes through a private cursor without copying or trusting them.
         * @param bytes borrowed schema view; an independent big-endian cursor is used
         */
        Reader(ByteBuffer bytes) { this.bytes = bytes.slice().order(ByteOrder.BIG_ENDIAN); }

        /**
         * Consumes one fixed-width unsigned field without sign extension.
         * @return unsigned next byte; truncation is translated by the outer decoder
         */
        int u8() { return Byte.toUnsignedInt(bytes.get()); }

        /**
         * Rejects noncanonical boolean representations rather than treating arbitrary bytes as true.
         * @return canonical zero-or-one flag
         * @throws IllegalArgumentException if the encoded flag is noncanonical
         */
        boolean flag() {
            int flag = u8();
            if (flag > 1) throw invalid("Noncanonical flag");
            return flag == 1;
        }

        /**
         * Bounds a raw collection length before any count-proportional allocation.
         * @param minimumBytes minimum encoded bytes per following item, positive
         * @return nonnegative byte-bounded count safe for the following collection allocation
         * @throws IllegalArgumentException if count is negative or exceeds remaining bytes
         */
        int count(int minimumBytes) {
            int count = bytes.getInt();
            if (count < 0 || count > bytes.remaining() / minimumBytes) throw invalid("Count exceeds remaining bytes");
            return count;
        }

        /**
         * Admits only prior file-local values, excluding cycles and forward references.
         * @param preceding number of preceding available values
         * @return preceding file-local value ID
         * @throws IllegalArgumentException if reference is negative, forward or dangling
         */
        ValueId reference(int preceding) {
            int id = bytes.getInt();
            if (id < 0 || id >= preceding) throw invalid("Dangling/forward value reference");
            return new ValueId(id);
        }

        /**
         * Bounds encoded text before strict decoding and validates the exact registration grammar.
         * @return exact strict UTF-8 name with valid dot segments, without normalization
         * @throws IllegalArgumentException if name length, Unicode or grammar is invalid
         */
        String name() {
            int length = bytes.getInt();
            if (length <= 0 || length > bytes.remaining() - 4) throw invalid("Invalid name length");
            ByteBuffer text = bytes.slice(); text.limit(length);
            bytes.position(bytes.position() + length);
            try {
                String name = StandardCharsets.UTF_8.newDecoder().onMalformedInput(CodingErrorAction.REPORT)
                        .onUnmappableCharacter(CodingErrorAction.REPORT).decode(text).toString();
                if (GraphSource.nameLength(name) != length) throw invalid("Noncanonical UTF-8 name");
                return name;
            } catch (CharacterCodingException failure) {
                throw new IllegalArgumentException("Malformed UTF-8 name", failure);
            }
        }

        /**
         * Decodes full metadata through Model's descriptor and layout constructors.
         * @return exact Model descriptor with checked static geometry, no backend rank ceiling
         * @throws IllegalArgumentException if IDs, flags, rank, extents, layout or gradients fail
         * @throws ArithmeticException if checked Model geometry overflows
         */
        TensorDescriptor descriptor() {
            DataType type = switch (u8()) {
                case 1 -> DataType.FLOAT64; case 2 -> DataType.FLOAT32; case 3 -> DataType.BFLOAT16;
                case 4 -> DataType.INT32; case 5 -> DataType.INT64; case 6 -> DataType.BOOL;
                case 7 -> DataType.FLOAT16; default -> throw invalid("Unknown dtype ID");
            };
            boolean grad = flag();
            int rank = bytes.getInt();
            if (rank < 0 || 8L * rank + 1 > bytes.remaining()) throw invalid("Rank exceeds remaining bytes");
            long[] extents = new long[rank];
            for (int i = 0; i < rank; i++) {
                extents[i] = bytes.getLong();
                if (extents[i] < 0) throw invalid("Negative extent");
            }
            Shape shape = Shape.of(extents);
            Optional<LayoutDescriptor> layout = Optional.empty();
            if (flag()) {
                if (9L + 8L * rank > bytes.remaining()) throw invalid("Truncated layout");
                long offset = bytes.getLong();
                boolean view = flag();
                long[] strides = new long[rank];
                for (int i = 0; i < rank; i++) strides[i] = bytes.getLong();
                layout = Optional.of(LayoutDescriptor.of(shape, strides, offset, view));
            }
            var descriptor = new TensorDescriptor(type, shape, layout, grad);
            GraphValueStorage.bytes(descriptor);
            return descriptor;
        }
    }

    /** One traversal counts with checked long arithmetic before any schema array is allocated. */
    private static final class Sink {
        private final ByteBuffer bytes;
        private long size;

        /**
         * Selects dry counting or writing to a destination already admitted by the same traversal.
         * @param bytes null for dry sizing, otherwise the already bounded exact destination
         */
        Sink(ByteBuffer bytes) { this.bytes = bytes; }

        /** Counts one byte and writes its low bits only during the bounded encoding pass.
         * @param value low eight bits to encode
         */
        void u8(int value) { size = Math.addExact(size, 1); if (bytes != null) bytes.put((byte) value); }

        /** Counts and optionally writes one fixed-width signed four-byte field.
         * @param value signed big-endian i32 value
         */
        void i32(int value) { size = Math.addExact(size, 4); if (bytes != null) bytes.putInt(value); }

        /** Counts and optionally writes one fixed-width signed eight-byte field.
         * @param value signed big-endian i64 value
         */
        void i64(long value) { size = Math.addExact(size, 8); if (bytes != null) bytes.putLong(value); }

        /** Counts strict UTF-8 before allocating its encoded bytes in the bounded writing pass.
         * @param name already grammar-validated exact UTF-8 registration name
         */
        void name(String name) {
            int length = GraphSource.nameLength(name);
            i32(length); size = Math.addExact(size, length);
            if (bytes != null) bytes.put(name.getBytes(StandardCharsets.UTF_8));
        }

        /** Preserves every descriptor field with explicit wire IDs and optional layout geometry.
         * @param descriptor exact checked Model metadata, including unresolved-layout presence
         */
        void descriptor(TensorDescriptor descriptor) {
            u8(switch (descriptor.dataType()) {
                case FLOAT64 -> 1; case FLOAT32 -> 2; case BFLOAT16 -> 3; case INT32 -> 4;
                case INT64 -> 5; case BOOL -> 6; case FLOAT16 -> 7;
            });
            u8(descriptor.requiresGrad() ? 1 : 0);
            i32(descriptor.shape().rank());
            for (long extent : descriptor.shape().toLongArray()) i64(extent);
            u8(descriptor.layout().isPresent() ? 1 : 0);
            descriptor.layout().ifPresent(layout -> {
                i64(layout.storageOffset()); u8(layout.isView() ? 1 : 0);
                for (long stride : layout.strides()) i64(stride);
            });
        }
    }
}
