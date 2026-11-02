package io.github.pho001.synaptik.checkpoint.graph;

import io.github.pho001.synaptik.checkpoint.artifact.ArtifactEnvelope;
import io.github.pho001.synaptik.checkpoint.artifact.ArtifactFormatException;
import io.github.pho001.synaptik.checkpoint.artifact.ArtifactLimits;
import io.github.pho001.synaptik.compiler.GraphCompilationPort;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.CompiledScanBody;
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
import io.github.pho001.synaptik.model.operation.convolution.*;
import io.github.pho001.synaptik.model.operation.index.*;
import io.github.pho001.synaptik.model.operation.layout.*;
import io.github.pho001.synaptik.model.operation.pooling.*;
import io.github.pho001.synaptik.model.operation.recurrent.*;
import io.github.pho001.synaptik.model.operation.scan.ScanKind;
import io.github.pho001.synaptik.model.operation.scan.ScanAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.*;
import io.github.pho001.synaptik.model.operation.elementwise.cast.*;
import io.github.pho001.synaptik.model.operation.reduction.*;
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
import java.util.OptionalInt;

/**
 * Selective closed graph-model.v1/v2/v3/v4/v5/v6 wire on the existing opaque envelope. IDs, flags and field order are
 * explicit and independent of Java enum ordinals. All parsing uses bounded views; only final
 * candidate leaves and Compiler reconstruction create Tensors. Complete metadata preflight precedes
 * host import and every body formal/producer allocation. Inline bodies retain independent local
 * IDs and latent residual metadata, never runtime histories or saved input storage.
 * V6 appends only 34=MAX_POOL3D and 35=AVERAGE_POOL3D, each with twelve i64 geometry fields
 * (kernel, stride, padding, dilation; depth/height/width within each), then one ceil-mode flag.
 * The smallest sufficient version is selected across outer and contained occurrences.
 */
final class GraphArtifactCodec {
    private static final long MAGIC = 0x53594e4752463031L;
    private static final String KIND = "graph-model.v1";
    private static final long MAGIC_V2 = 0x53594e4752463032L;
    private static final String KIND_V2 = "graph-model.v2";
    private static final long MAGIC_V3 = 0x53594e4752463033L;
    private static final String KIND_V3 = "graph-model.v3";
    private static final long MAGIC_V4 = 0x53594e4752463034L;
    private static final String KIND_V4 = "graph-model.v4";
    private static final long MAGIC_V5 = 0x53594e4752463035L;
    private static final String KIND_V5 = "graph-model.v5";
    private static final long MAGIC_V6 = 0x53594e4752463036L;
    private static final String KIND_V6 = "graph-model.v6";

    private GraphArtifactCodec() { }

    /**
     * Dry-sizes all metadata and six encoded bounds before storage admission or bulk copying.
     * Independently rechecks v5/v6 cold eligibility through Compiler before dry sizing;
     * a raw association is not a semantic admission credential.
     * @param source fully metadata-validated normalized source
     * @param limits explicit encoded ceilings
     * @return complete detached canonical image, ready for unchanged ArtifactFile publication
     * @throws IllegalArgumentException if source cold admission, an encoded bound or BOOL lane is invalid
     * @throws ArithmeticException if checked sizing overflows
     * @throws IllegalStateException if named host state is absent/dead/inaccessible
     */
    static ArtifactEnvelope encode(GraphSource source, ArtifactLimits limits) {
        int version = 1;
        for (CompiledNode node : source.graph.nodes()) {
            int op = opcode(node);
            version = Math.max(version, op >= 34 ? 6 : op == 33 ? 5 : op == 23 ? 4 : op >= 9 ? 3 : op >= 6 ? 2 : 1);
            if (node.scanBody().isPresent()) {
                for (CompiledNode local : node.scanBody().orElseThrow().graph().nodes())
                    if (opcode(local, true) >= 34) version = 6;
            }
        }
        if (version >= 5) {
            var leaves = new ArrayList<Tensor>(Math.addExact(source.source.inputs().size(), source.states.size()));
            leaves.addAll(source.source.inputs()); leaves.addAll(source.states);
            new GraphCompilationPort.SourceCapture(source.graph, leaves);
        }
        Sink dry = new Sink(null);
        write(dry, source, version);
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
        write(new Sink(schema), source, version);
        var entries = new ArrayList<ArtifactEnvelope.Entry>();
        entries.add(new ArtifactEnvelope.Entry("schema", schema.array()));
        for (int i = 0; i < source.states.size(); i++)
            entries.add(new ArtifactEnvelope.Entry(ordinal(i),
                    GraphValueStorage.encode(source.states.get(i).descriptor(), segments.get(i))));
        return new ArtifactEnvelope(version == 1 ? KIND : version == 2 ? KIND_V2 : version == 3 ? KIND_V3
                : version == 4 ? KIND_V4 : version == 5 ? KIND_V5 : KIND_V6, entries);
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
            int version = KIND.equals(envelope.kind()) ? 1 : KIND_V2.equals(envelope.kind()) ? 2
                    : KIND_V3.equals(envelope.kind()) ? 3 : KIND_V4.equals(envelope.kind()) ? 4
                    : KIND_V5.equals(envelope.kind()) ? 5 : KIND_V6.equals(envelope.kind()) ? 6 : 0;
            if (version == 0 || envelope.entries().isEmpty()
                    || !envelope.entries().getFirst().key().equals("schema"))
                throw invalid("Expected graph-model.v1/v2/v3/v4/v5/v6 schema");
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
            if (in.getLong() != (version == 1 ? MAGIC : version == 2 ? MAGIC_V2 : version == 3 ? MAGIC_V3
                    : version == 4 ? MAGIC_V4 : version == 5 ? MAGIC_V5 : MAGIC_V6)
                    || in.getInt() != version)
                throw invalid("Mismatched graph kind/magic/version");
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
            var nodes = reader.nodes(version, false, values);
            var phases = phases(nodes);
            int outputCount = reader.count(4);
            if (outputCount == 0) throw invalid("Empty output boundary");
            var outputs = new ArrayList<ValueId>(outputCount);
            for (int i = 0; i < outputCount; i++) outputs.add(reader.reference(values.size()));
            if (in.hasRemaining()) throw invalid("Trailing graph schema bytes");
            var graph = new CompiledGraphModel(values, nodes, inputIds, outputs, phases);
            canonicalTopology(graph, inputCount);
            GraphCompilationPort.validateSource(graph);
            // Candidate leaves are final storage-free objects; all graph metadata already passed.
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
        int[] producers = new int[seen.length];
        java.util.Arrays.fill(producers, -1);
        for (int n = 0; n < graph.nodes().size(); n++)
            for (ValueId id : graph.nodes().get(n).outputs()) producers[Math.toIntExact(id.value())] = n;
        boolean[] visited = new boolean[graph.nodes().size()];
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
                int node = producers[value];
                if (visited[node]) continue;
                visited[node] = true;
                for (ValueId sibling : graph.nodes().get(node).outputs()) seen[Math.toIntExact(sibling.value())] = true;
                stack.push(~node);
                var args = graph.nodes().get(node).inputs();
                for (int i = args.size() - 1; i >= 0; i--) stack.push(Math.toIntExact(args.get(i).value()));
            }
        }
        for (int i = declaredInputs; i < seen.length; i++)
            if (!seen[i]) throw invalid("Unreachable state or dead node");
    }

    /**
     * Assigns only the source phase to metadata already checked by the node reader.
     * @param nodes parsed occurrences in local canonical order
     * @return complete local phase map, never null
     */
    private static HashMap<NodeId, GraphPhase> phases(List<CompiledNode> nodes) {
        var phases = new HashMap<NodeId, GraphPhase>();
        for (var node : nodes) phases.put(node.id(), GraphPhase.FORWARD);
        return phases;
    }

    /**
     * Uses the exact same bounded field traversal for dry sizing and encoding.
     * @param out dry or previously bounded encoding sink
     * @param source normalized metadata-validated association
     * @param version smallest sufficient closed version, already selected from admitted opcodes
     * @throws IllegalArgumentException if an operation's kind, attributes, phase or arity is unsupported
     */
    private static void write(Sink out, GraphSource source, int version) {
        out.i64(version == 1 ? MAGIC : version == 2 ? MAGIC_V2 : version == 3 ? MAGIC_V3
                : version == 4 ? MAGIC_V4 : version == 5 ? MAGIC_V5 : MAGIC_V6); out.i32(version);
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
        writeNodes(out, source.graph, version, false);
        out.references(source.graph.outputs());
    }

    /**
     * Writes producer records in one scope; a general occurrence contains exactly one flat body.
     * @param out dry or admitted destination with checked aggregate body counters
     * @param graph normalized local graph, never flattened into its parent
     * @param version selected outer format; bodies use its v3-style result records
     * @param bodyScope whether only the bounded non-scan body inventory is permitted
     * @throws IllegalArgumentException if phase, opcode, attachment or topology is unsupported
     * @throws ArithmeticException if a contained aggregate budget overflows signed i32
     */
    private static void writeNodes(Sink out, CompiledGraphModel graph, int version, boolean bodyScope) {
        canonicalTopology(graph, graph.inputs().size());
        out.i32(graph.nodes().size());
        for (var node : graph.nodes()) {
            if (graph.nodePhases().get(node.id()) != GraphPhase.FORWARD)
                throw invalid("Only FORWARD source phase is supported");
            out.u8(1);
            int op = opcode(node, bodyScope);
            out.u8(op);
            if (op == 16 || op >= 20 && op <= 23 || op == 33) out.i32(node.inputs().size());
            for (ValueId id : node.inputs()) out.i32(Math.toIntExact(id.value()));
            attributes(out, node, op);
            if (op == 23 || op == 33) {
                var body = node.scanBody().orElseThrow();
                var local = body.graph();
                out.formals = Math.addExact(out.formals, local.inputs().size());
                out.nodes = Math.addExact(out.nodes, local.nodes().size());
                out.values = Math.addExact(out.values, local.values().size());
                out.roles = Math.addExact(out.roles, Math.addExact(local.outputs().size(),
                        ((ScanAttrs) node.operation().attrs()).outputCount()));
                out.i32(local.inputs().size());
                for (ValueId id : local.inputs())
                    out.descriptor(local.values().get(Math.toIntExact(id.value())).descriptor());
                writeNodes(out, local, version, true);
                out.references(local.outputs());
                out.references(body.nextCarry()); out.references(body.emissions()); out.references(body.residuals());
                if (!body.demandedOutputIndices().isEmpty()) throw invalid("Cold scan demand must be empty");
                out.i32(0);
            }
            if (version >= 3) out.i32(node.outputs().size());
            for (ValueId id : node.outputs())
                out.descriptor(graph.values().get(Math.toIntExact(id.value())).descriptor());
        }
    }

    /**
     * Admits the closed wire independently of Compiler admission, including during dry sizing.
     * @param node stored occurrence with exact Model attributes and admitted input/output counts
     * @return explicit stable outer opcode 1..23 or 33..35, never an enum ordinal or fallback opcode
     * @throws IllegalArgumentException if a kind, attachment, attributes or arity
     *     cannot be represented exactly by the closed wire, independently of cold copy admission
     */
    private static int opcode(CompiledNode node) {
        return opcode(node, false);
    }

    /**
     * Selects the explicit scope-specific inventory without widening outer admission.
     * Compiler retains family inference and dtype/descriptor validation. Length mode has its own
     * outer opcode and header; it never changes the all-valid seven-field record.
     * Both scan opcodes retain nonempty carry; length opcode 33 additionally retains positive
     * ROW carry, declared axes zero/one and rank-two ROW(1) declarations, including unused ones.
     * Compiler's independent cold preflight retains the constructor/type/coordinate intersection.
     * Broader source headers reject here before sizing or storage admission, and the reader
     * applies the same scalar restrictions before body/role allocation. Mode is opcode-derived.
     * @param node local occurrence with all canonical output ports
     * @param bodyScope true only inside one general scan attachment
     * @return stable wire opcode in 1..35
     * @throws IllegalArgumentException if the exact scoped signature is not representable
     */
    private static int opcode(CompiledNode node, boolean bodyScope) {
        var kind = node.operation().kind();
        if (node.scanBody().isPresent() != (kind == ScanKind.SCAN)
                || bodyScope && (kind == ScanKind.SCAN || kind instanceof RecurrentScanKind))
            throw invalid("Missing, extra or nested scan attachment");
        if (kind == ScanKind.SCAN && node.operation().attrs() instanceof ScanAttrs scan
                && (scan.carryCount() == 0 || scan.lengthAware() && scan.rowCarryCount() == 0)) {
            throw invalid("Wire scan requires nonempty carry and positive ROW length mode");
        }
        if (kind == ScanKind.SCAN && node.operation().attrs() instanceof ScanAttrs scan
                && scan.lengthAware()) {
            var local = node.scanBody().orElseThrow().graph();
            for (int i = 0; i < scan.sequenceCount(); i++) {
                int axis = scan.sequenceRowAxes().get(i).orElse(-1);
                if (axis > 1) throw invalid("Wire scan supports only sequence row axes zero/one");
                if (axis == 1) {
                    ValueId formal = local.inputs().get(Math.addExact(scan.carryCount(), i));
                    for (GraphValue value : local.values())
                        if (value.id().equals(formal) && value.descriptor().shape().rank() != 2)
                            throw invalid("Wire scan requires rank-two ROW(1) declarations");
                }
            }
        }
        int op;
        if (kind == ScanKind.SCAN) op = node.operation().attrs() instanceof ScanAttrs scan
                && scan.lengthAware() ? 33 : 23;
        else if (bodyScope && kind == BinaryArithmeticKind.DIV) op = 24;
        else if (bodyScope && kind == ScalarElementwiseKind.ADD) op = 25;
        else if (bodyScope && kind == ScalarElementwiseKind.SUB) op = 26;
        else if (bodyScope && kind == ScalarElementwiseKind.MUL) op = 27;
        else if (bodyScope && kind == ScalarElementwiseKind.DIV) op = 28;
        else if (bodyScope && kind == AggregateReductionKind.SUM) op = 29;
        else if (bodyScope && kind == ShapeTransformKind.EXPAND) op = 30;
        else if (bodyScope && kind == ContiguousKind.CONTIGUOUS) op = 31;
        else if (bodyScope && kind == CastKind.CAST) op = 32;
        else if (kind == BinaryArithmeticKind.ADD) op = 1;
        else if (kind == BinaryArithmeticKind.MUL) op = 2;
        else if (kind == UnaryElementwiseKind.NEG) op = 3;
        else if (kind == MatmulKind.MATMUL) op = 4;
        else if (kind == AxisTransformKind.PERMUTE) op = 5;
        else if (kind == UnaryElementwiseKind.RELU) op = 6;
        else if (kind == UnaryElementwiseKind.TANH) op = 7;
        else if (kind == UnaryElementwiseKind.SIGMOID) op = 8;
        else if (kind == BinaryArithmeticKind.SUB) op = 9;
        else if (kind == ShapeTransformKind.RESHAPE) op = 10;
        else if (kind == AxisTransformKind.EXPAND_DIMS) op = 11;
        else if (kind == AxisTransformKind.SQUEEZE) op = 12;
        else if (kind == SelectKind.SELECT) op = 13;
        else if (kind == AxisGatherKind.GATHER) op = 14;
        else if (kind == SliceKind.SLICE) op = 15;
        else if (kind == TensorCompositionKind.CONCAT) op = 16;
        else if (kind == Conv2dKind.CONV2D) op = 17;
        else if (kind == Conv3dKind.CONV3D) op = 18;
        else if (kind == Pool2dKind.MAX_POOL2D) op = 19;
        else if (kind == RecurrentScanKind.RNN_TANH) op = 20;
        else if (kind == RecurrentScanKind.GRU_RESET_AFTER) op = 21;
        else if (kind == RecurrentScanKind.LSTM) op = 22;
        else if (kind == Pool3dKind.MAX_POOL3D) op = 34;
        else if (kind == Pool3dKind.AVERAGE_POOL3D) op = 35;
        else throw invalid("Unknown source operation: " + kind);
        var a = node.operation().attrs();
        boolean attrs = switch (op) {
            case 5 -> a instanceof PermutationAttrs p && p.axes().equals(List.of(1, 0));
            case 10, 30 -> a instanceof TargetShapeAttrs t && t.targetShape().isFullyStatic()
                    && (bodyScope || java.util.Arrays.stream(t.targetShape().toLongArray()).allMatch(n -> n > 0));
            case 11, 12 -> a instanceof AxisTransformAttrs axis && (bodyScope || axis.axis() == 2);
            case 13 -> a instanceof SelectAttrs select && select.axis() == 0;
            case 14 -> a instanceof IndexAxisAttrs axis && axis.axis() == 0;
            case 15 -> a instanceof SliceAttrs s && s.axes().equals(List.of(1))
                    && s.lengths().getFirst() > 0 && s.steps().equals(List.of(1L));
            case 16 -> a instanceof CompositionAxisAttrs axis && axis.axis() == 0;
            case 17 -> a.equals(Conv2dAttrs.defaults());
            case 18 -> a.equals(Conv3dAttrs.defaults());
            case 19 -> a instanceof MaxPool2dAttrs p && p.strideHeight() == p.kernelHeight()
                    && p.strideWidth() == p.kernelWidth() && p.paddingHeight() == 0 && p.paddingWidth() == 0
                    && p.dilationHeight() == 1 && p.dilationWidth() == 1 && !p.ceilMode();
            case 20, 21, 22 -> a instanceof RecurrentDirection;
            case 23, 33 -> a instanceof ScanAttrs;
            case 25, 26, 27, 28 -> a instanceof ScalarValueAttrs;
            case 29 -> a == NoOperationAttrs.INSTANCE || a instanceof AxisReductionAttrs
                    || a instanceof MultiAxisReductionAttrs || a instanceof SumToShapeAttrs;
            case 32 -> a instanceof CastAttrs;
            case 34 -> a instanceof MaxPool3dAttrs;
            case 35 -> a instanceof AveragePool3dAttrs;
            default -> a == NoOperationAttrs.INSTANCE;
        };
        int arity = switch (op) {
            case 1, 2, 4, 9, 14, 24 -> 2;
            case 17, 18 -> 3;
            case 16, 20, 21, 22, 23, 33 -> node.inputs().size();
            default -> 1;
        };
        boolean variable = op == 23 || op == 33 ? a instanceof ScanAttrs scan && arity == scan.inputCount()
                : op == 16 ? arity > 0 : op >= 20 && op <= 22
                ? arity >= (op == 22 ? 6 : 5) && arity <= (op == 22 ? 7 : 6) : true;
        if (!attrs || !variable || node.inputs().size() != arity
                || node.outputs().size() != (op == 23 || op == 33 ? ((ScanAttrs) a).outputCount()
                        : op == 22 ? 3 : op == 20 || op == 21 ? 2 : 1))
            throw invalid("Unsupported source attributes/arity");
        return op;
    }

    /**
     * Traverses exact attributes identically for dry sizing and bounded writing.
     * @param out dry or bounded destination
     * @param node independently admitted operation occurrence
     * @param op explicit stable opcode matching the node
     */
    private static void attributes(Sink out, CompiledNode node, int op) {
        var attrs = node.operation().attrs();
        switch (op) {
            case 5 -> { out.i32(2); out.i32(1); out.i32(0); }
            case 10, 30 -> out.shape(((TargetShapeAttrs) attrs).targetShape());
            case 11, 12 -> out.i32(((AxisTransformAttrs) attrs).axis());
            case 13 -> { var a = (SelectAttrs) attrs; out.i32(a.axis()); out.i64(a.index()); }
            case 14 -> out.i32(((IndexAxisAttrs) attrs).axis());
            case 15 -> {
                var a = (SliceAttrs) attrs;
                out.i32(a.axes().size());
                for (int j = 0; j < a.axes().size(); j++) {
                    out.i64(a.starts().get(j)); out.i64(a.lengths().get(j));
                    out.i32(a.axes().get(j)); out.i64(a.steps().get(j));
                }
            }
            case 16 -> out.i32(((CompositionAxisAttrs) attrs).axis());
            case 17 -> {
                var a = (Conv2dAttrs) attrs;
                out.i64(a.strideHeight()); out.i64(a.strideWidth()); out.i64(a.paddingHeight());
                out.i64(a.paddingWidth()); out.i64(a.dilationHeight()); out.i64(a.dilationWidth()); out.i64(a.groups());
            }
            case 18 -> {
                var a = (Conv3dAttrs) attrs;
                out.i64(a.strideDepth()); out.i64(a.strideHeight()); out.i64(a.strideWidth());
                out.i64(a.paddingDepth()); out.i64(a.paddingHeight()); out.i64(a.paddingWidth());
                out.i64(a.dilationDepth()); out.i64(a.dilationHeight()); out.i64(a.dilationWidth()); out.i64(a.groups());
            }
            case 19 -> {
                var a = (MaxPool2dAttrs) attrs;
                out.i64(a.kernelHeight()); out.i64(a.kernelWidth()); out.i64(a.strideHeight()); out.i64(a.strideWidth());
                out.i64(a.paddingHeight()); out.i64(a.paddingWidth()); out.i64(a.dilationHeight()); out.i64(a.dilationWidth());
                out.u8(a.ceilMode() ? 1 : 0);
            }
            case 20, 21, 22 -> out.u8(attrs == RecurrentDirection.FORWARD ? 1 : 2);
            case 23, 33 -> {
                var a = (ScanAttrs) attrs;
                out.i64(a.timeCount()); out.i32(a.carryCount()); out.i32(a.sequenceCount());
                out.i32(a.invariantCount()); out.i32(a.emissionCount()); out.i32(a.residualCount());
                out.u8(a.reverse() ? 1 : 0);
                if (op == 33) {
                    out.i32(a.rowCarryCount()); out.i32(a.sequenceRowAxes().size());
                    for (OptionalInt axis : a.sequenceRowAxes()) {
                        out.u8(axis.isPresent() ? 1 : 0);
                        if (axis.isPresent()) out.i32(axis.getAsInt());
                    }
                }
            }
            case 25, 26, 27, 28 -> out.scalar(((ScalarValueAttrs) attrs).value());
            case 29 -> {
                if (attrs == NoOperationAttrs.INSTANCE) out.u8(0);
                else if (attrs instanceof AxisReductionAttrs a) {
                    out.u8(1); out.i32(a.axis()); out.u8(a.keepDimensions() ? 1 : 0);
                } else if (attrs instanceof MultiAxisReductionAttrs a) {
                    out.u8(2); out.i32(a.axes().size());
                    for (int axis : a.axes()) out.i32(axis);
                    out.u8(a.keepDimensions() ? 1 : 0);
                } else {
                    out.u8(3); out.shape(((SumToShapeAttrs) attrs).targetShape());
                }
            }
            case 32 -> out.dtype(((CastAttrs) attrs).targetDataType());
            case 34 -> {
                var a = (MaxPool3dAttrs) attrs;
                out.i64(a.kernelDepth()); out.i64(a.kernelHeight()); out.i64(a.kernelWidth());
                out.i64(a.strideDepth()); out.i64(a.strideHeight()); out.i64(a.strideWidth());
                out.i64(a.paddingDepth()); out.i64(a.paddingHeight()); out.i64(a.paddingWidth());
                out.i64(a.dilationDepth()); out.i64(a.dilationHeight()); out.i64(a.dilationWidth());
                out.u8(a.ceilMode() ? 1 : 0);
            }
            case 35 -> {
                var a = (AveragePool3dAttrs) attrs;
                out.i64(a.kernelDepth()); out.i64(a.kernelHeight()); out.i64(a.kernelWidth());
                out.i64(a.strideDepth()); out.i64(a.strideHeight()); out.i64(a.strideWidth());
                out.i64(a.paddingDepth()); out.i64(a.paddingHeight()); out.i64(a.paddingWidth());
                out.i64(a.dilationDepth()); out.i64(a.dilationHeight()); out.i64(a.dilationWidth());
                out.u8(a.ceilMode() ? 1 : 0);
            }
            default -> { /* parameterless admitted kind */ }
        }
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

        private int formals, nodes, values, roles;

        /**
         * Parses one scope as metadata, byte-bounding allocations and checking cumulative IDs.
         * @param version exact enclosing version
         * @param bodyScope whether only the non-scan body inventory is permitted
         * @param localValues boundary descriptors, extended by every canonical producer port
         * @return local occurrences without host values or Tensor reconstruction tables
         * @throws IllegalArgumentException if scoped wire, counts or attributes are invalid
         * @throws ArithmeticException if local or aggregate budgets overflow signed i32
         */
        List<CompiledNode> nodes(int version, boolean bodyScope, ArrayList<GraphValue> localValues) {
            int count = count(13);
            Math.addExact(localValues.size(), count);
            if (bodyScope) nodes = Math.addExact(nodes, count);
            var result = new ArrayList<CompiledNode>(count);
            for (int i = 0; i < count; i++) {
                if (u8() != 1) throw invalid("Unknown source phase");
                int op = u8();
                if (op < 1 || op > (version == 1 ? 5 : version == 2 ? 8 : version == 3 ? 22
                        : version == 4 ? 32 : version == 5 ? 33 : 35)
                        || bodyScope && (op >= 20 && op <= 23 || op == 33)
                        || !bodyScope && op >= 24 && op <= 32)
                    throw invalid("Operation exceeds graph version/scope");
                int arity = switch (op) {
                    case 1, 2, 4, 9, 14, 24 -> 2;
                    case 17, 18 -> 3;
                    case 16, 20, 21, 22, 23, 33 -> count(4);
                    default -> 1;
                };
                if (arity == 0 || arity > bytes.remaining() / 4
                        || op >= 20 && op <= 22 && (arity < (op == 22 ? 6 : 5)
                        || arity > (op == 22 ? 7 : 6))) throw invalid("Invalid source input count");
                var args = new ArrayList<ValueId>(arity);
                int first = localValues.size();
                for (int j = 0; j < arity; j++) args.add(reference(first));
                Operation operation = operation(op, arity);
                Optional<CompiledScanBody> body = Optional.empty();
                if (op == 23 || op == 33) {
                    var attrs = (ScanAttrs) operation.attrs();
                    if (arity != attrs.inputCount()) throw invalid("Scan actual count mismatch");
                    body = Optional.of(body(attrs, version));
                }
                int outputs = version >= 3 ? count(7) : 1;
                int expected = op == 23 || op == 33 ? ((ScanAttrs) operation.attrs()).outputCount()
                        : op == 22 ? 3 : op == 20 || op == 21 ? 2 : 1;
                if (outputs != expected) throw invalid("Invalid output count");
                Math.addExact(first, outputs);
                if (bodyScope) values = Math.addExact(values, outputs);
                var ids = new ArrayList<ValueId>(outputs);
                for (int j = 0; j < outputs; j++) {
                    var id = new ValueId(Math.addExact(first, j));
                    localValues.add(new GraphValue(id, descriptor()));
                    ids.add(id);
                }
                var node = new CompiledNode(new NodeId(i), operation, args, ids, body);
                opcode(node, bodyScope);
                result.add(node);
            }
            return result;
        }

        /**
         * Reads a depth-one attachment with local IDs and exact ordered role cardinalities.
         * Lengths is an outer actual only; formals always count carry, step and invariant positions.
         * Compiler subsequently verifies derived residual order and family descriptors.
         * @param attrs checked scalar header
         * @param version enclosing closed wire version, also bounding body opcodes
         * @return complete attachment with empty demand
         * @throws IllegalArgumentException if counts, topology, roots or demand are invalid
         * @throws ArithmeticException if contained aggregate counters overflow signed i32
         */
        CompiledScanBody body(ScanAttrs attrs, int version) {
            int count = count(7);
            if (count != Math.addExact(Math.addExact(attrs.carryCount(), attrs.sequenceCount()), attrs.invariantCount()))
                throw invalid("Scan formal count mismatch");
            formals = Math.addExact(formals, count);
            values = Math.addExact(values, count);
            var localValues = new ArrayList<GraphValue>(count);
            var inputs = new ArrayList<ValueId>(count);
            for (int i = 0; i < count; i++) {
                var id = new ValueId(i);
                inputs.add(id);
                localValues.add(new GraphValue(id, descriptor()));
            }
            var localNodes = nodes(version, true, localValues);
            var roots = references(localValues.size(), -1, Math.min(localValues.size(), attrs.outputCount()));
            var carry = references(localValues.size(), attrs.carryCount(), attrs.carryCount());
            var emissions = references(localValues.size(), attrs.emissionCount(), attrs.emissionCount());
            var residuals = references(localValues.size(), attrs.residualCount(), attrs.residualCount());
            if (count(4) != 0) throw invalid("Cold scan demand must be empty");
            var ordered = new java.util.LinkedHashSet<>(carry);
            ordered.addAll(emissions); ordered.addAll(residuals);
            if (!roots.equals(List.copyOf(ordered))) throw invalid("Noncanonical body role roots");
            var graph = new CompiledGraphModel(localValues, localNodes, inputs, roots, phases(localNodes));
            canonicalTopology(graph, count);
            return new CompiledScanBody(graph, carry, emissions, residuals, List.of());
        }

        /**
         * Bounds local root/role lists before allocation, retaining repeated roles.
         * @param preceding available local value count
         * @param expected exact count, or -1 for unique roots
         * @param maximum intrinsic header/value ceiling
         * @return ordered local references
         * @throws IllegalArgumentException if cardinality or a local ID is invalid
         * @throws ArithmeticException if aggregate role-reference count overflows
         */
        List<ValueId> references(int preceding, int expected, int maximum) {
            int count = count(4);
            if (count > maximum || expected >= 0 && count != expected)
                throw invalid("Invalid body role/root count");
            roles = Math.addExact(roles, count);
            var ids = new ArrayList<ValueId>(count);
            for (int i = 0; i < count; i++) ids.add(reference(preceding));
            return ids;
        }

        /**
         * Decodes wire attributes into metadata only; Compiler owns Tensor constructors.
         * @param op scope/version-admitted stable opcode
         * @param arity already parsed actual input count, checked against the length header before role allocation
         * @return exact immutable operation
         * @throws IllegalArgumentException if an attribute tag, flag or value is invalid
         * @throws ArithmeticException if checked scan header cardinalities overflow
         */
        Operation operation(int op, int arity) {
            return switch (op) {
                case 1 -> new Operation(BinaryArithmeticKind.ADD, NoOperationAttrs.INSTANCE);
                case 2 -> new Operation(BinaryArithmeticKind.MUL, NoOperationAttrs.INSTANCE);
                case 3 -> new Operation(UnaryElementwiseKind.NEG, NoOperationAttrs.INSTANCE);
                case 4 -> new Operation(MatmulKind.MATMUL, NoOperationAttrs.INSTANCE);
                case 5 -> {
                    if (bytes.getInt() != 2 || bytes.getInt() != 1 || bytes.getInt() != 0)
                        throw invalid("Only rank-two PERMUTE[1,0] is supported");
                    yield new Operation(AxisTransformKind.PERMUTE, new PermutationAttrs(List.of(1, 0)));
                }
                case 6 -> new Operation(UnaryElementwiseKind.RELU, NoOperationAttrs.INSTANCE);
                case 7 -> new Operation(UnaryElementwiseKind.TANH, NoOperationAttrs.INSTANCE);
                case 8 -> new Operation(UnaryElementwiseKind.SIGMOID, NoOperationAttrs.INSTANCE);
                case 9 -> new Operation(BinaryArithmeticKind.SUB, NoOperationAttrs.INSTANCE);
                case 10 -> new Operation(ShapeTransformKind.RESHAPE, new TargetShapeAttrs(shape()));
                case 11, 12 -> new Operation(op == 11 ? AxisTransformKind.EXPAND_DIMS : AxisTransformKind.SQUEEZE,
                        new AxisTransformAttrs(bytes.getInt()));
                case 13 -> new Operation(SelectKind.SELECT, new SelectAttrs(bytes.getInt(), bytes.getLong()));
                case 14 -> new Operation(AxisGatherKind.GATHER, new IndexAxisAttrs(bytes.getInt()));
                case 15 -> {
                    if (count(28) != 1) throw invalid("Only one slice entry is supported");
                    yield new Operation(SliceKind.SLICE, new SliceAttrs(List.of(bytes.getLong()),
                            List.of(bytes.getLong()), List.of(bytes.getInt()), List.of(bytes.getLong())));
                }
                case 16 -> new Operation(TensorCompositionKind.CONCAT, new CompositionAxisAttrs(bytes.getInt()));
                case 17 -> new Operation(Conv2dKind.CONV2D, new Conv2dAttrs(bytes.getLong(), bytes.getLong(),
                        bytes.getLong(), bytes.getLong(), bytes.getLong(), bytes.getLong(), bytes.getLong()));
                case 18 -> new Operation(Conv3dKind.CONV3D, new Conv3dAttrs(bytes.getLong(), bytes.getLong(),
                        bytes.getLong(), bytes.getLong(), bytes.getLong(), bytes.getLong(), bytes.getLong(),
                        bytes.getLong(), bytes.getLong(), bytes.getLong()));
                case 19 -> new Operation(Pool2dKind.MAX_POOL2D, new MaxPool2dAttrs(bytes.getLong(), bytes.getLong(),
                        bytes.getLong(), bytes.getLong(), bytes.getLong(), bytes.getLong(), bytes.getLong(),
                        bytes.getLong(), flag()));
                case 20, 21, 22 -> new Operation(op == 20 ? RecurrentScanKind.RNN_TANH
                        : op == 21 ? RecurrentScanKind.GRU_RESET_AFTER : RecurrentScanKind.LSTM,
                        switch (u8()) {
                            case 1 -> RecurrentDirection.FORWARD;
                            case 2 -> RecurrentDirection.REVERSE;
                            default -> throw invalid("Unknown recurrent direction");
                        });
                case 23, 33 -> {
                    var header = new ScanAttrs(bytes.getLong(), bytes.getInt(), bytes.getInt(), bytes.getInt(),
                            bytes.getInt(), bytes.getInt(), flag());
                    if (header.carryCount() == 0) throw invalid("Wire scan requires nonempty carry");
                    if (op == 23) yield new Operation(ScanKind.SCAN, header);
                    if (Math.addExact(header.inputCount(), 1) != arity) throw invalid("Scan actual count mismatch");
                    int rowCarry = bytes.getInt();
                    if (rowCarry <= 0 || rowCarry > header.carryCount()) throw invalid("Invalid row carry count");
                    int count = count(1);
                    if (count != header.sequenceCount()) throw invalid("Scan sequence axis count mismatch");
                    var axes = new ArrayList<OptionalInt>(count);
                    for (int i = 0; i < count; i++) {
                        if (!flag()) axes.add(OptionalInt.empty());
                        else {
                            int axis = bytes.getInt();
                            if (axis < 0 || axis > 1) throw invalid("Unsupported sequence row axis");
                            axes.add(OptionalInt.of(axis));
                        }
                    }
                    yield new Operation(ScanKind.SCAN, new ScanAttrs(header.timeCount(), header.carryCount(), header.sequenceCount(), header.invariantCount(), header.emissionCount(), header.residualCount(), header.reverse(), true, rowCarry, axes));
                }
                case 24 -> new Operation(BinaryArithmeticKind.DIV, NoOperationAttrs.INSTANCE);
                case 25, 26, 27, 28 -> new Operation(switch (op) {
                    case 25 -> ScalarElementwiseKind.ADD; case 26 -> ScalarElementwiseKind.SUB;
                    case 27 -> ScalarElementwiseKind.MUL; default -> ScalarElementwiseKind.DIV;
                }, new ScalarValueAttrs(scalar()));
                case 29 -> new Operation(AggregateReductionKind.SUM, switch (u8()) {
                    case 0 -> NoOperationAttrs.INSTANCE;
                    case 1 -> new AxisReductionAttrs(bytes.getInt(), flag());
                    case 2 -> {
                        int count = count(4);
                        var axes = new ArrayList<Integer>(count);
                        for (int i = 0; i < count; i++) axes.add(bytes.getInt());
                        yield new MultiAxisReductionAttrs(axes, flag());
                    }
                    case 3 -> new SumToShapeAttrs(shape());
                    default -> throw invalid("Unknown SUM tag");
                });
                case 30 -> new Operation(ShapeTransformKind.EXPAND, new TargetShapeAttrs(shape()));
                case 31 -> new Operation(ContiguousKind.CONTIGUOUS, NoOperationAttrs.INSTANCE);
                case 32 -> new Operation(CastKind.CAST, new CastAttrs(dtype()));
                case 34 -> new Operation(Pool3dKind.MAX_POOL3D, new MaxPool3dAttrs(
                        bytes.getLong(), bytes.getLong(), bytes.getLong(),
                        bytes.getLong(), bytes.getLong(), bytes.getLong(),
                        bytes.getLong(), bytes.getLong(), bytes.getLong(),
                        bytes.getLong(), bytes.getLong(), bytes.getLong(), flag()));
                case 35 -> new Operation(Pool3dKind.AVERAGE_POOL3D, new AveragePool3dAttrs(
                        bytes.getLong(), bytes.getLong(), bytes.getLong(),
                        bytes.getLong(), bytes.getLong(), bytes.getLong(),
                        bytes.getLong(), bytes.getLong(), bytes.getLong(),
                        bytes.getLong(), bytes.getLong(), bytes.getLong(), flag()));
                default -> throw invalid("Unknown source operation");
            };
        }

        /**
         * Reads a static shape with byte-bounded rank and checked element count.
         * @return exact static shape, including scalar and empty shapes
         * @throws IllegalArgumentException if rank or extents are invalid
         * @throws ArithmeticException if element count overflows
         */
        Shape shape() {
            int rank = count(8);
            long[] extents = new long[rank];
            for (int i = 0; i < rank; i++) {
                extents[i] = bytes.getLong();
                if (extents[i] < 0) throw invalid("Negative extent");
            }
            var shape = Shape.of(extents);
            shape.knownElementCount();
            return shape;
        }

        /**
         * Reads the explicit dtype namespace shared by descriptors and attributes.
         * @return exact Model dtype
         * @throws IllegalArgumentException if the wire ID is unknown
         */
        DataType dtype() {
            return switch (u8()) {
                case 1 -> DataType.FLOAT64; case 2 -> DataType.FLOAT32; case 3 -> DataType.BFLOAT16;
                case 4 -> DataType.INT32; case 5 -> DataType.INT64; case 6 -> DataType.BOOL;
                case 7 -> DataType.FLOAT16; default -> throw invalid("Unknown dtype ID");
            };
        }

        /**
         * Reads scalar raw bits without conversion or NaN canonicalization.
         * @return exact typed scalar
         * @throws IllegalArgumentException if dtype or BOOL bits are invalid
         */
        ScalarValue scalar() {
            return switch (dtype()) {
                case FLOAT64 -> ScalarValue.float64(Double.longBitsToDouble(bytes.getLong()));
                case FLOAT32 -> ScalarValue.float32(Float.intBitsToFloat(bytes.getInt()));
                case BFLOAT16 -> ScalarValue.bfloat16Bits(bytes.getShort());
                case FLOAT16 -> ScalarValue.float16Bits(bytes.getShort());
                case INT32 -> ScalarValue.int32(bytes.getInt());
                case INT64 -> ScalarValue.int64(bytes.getLong());
                case BOOL -> ScalarValue.bool(flag());
            };
        }

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
            DataType type = dtype();
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
        private int formals, nodes, values, roles;

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

        /**
         * Preserves a count-prefixed local reference list without deduplicating roles.
         * @param ids ordered canonical references in the current scope
         */
        void references(List<ValueId> ids) {
            i32(ids.size());
            for (ValueId id : ids) i32(Math.toIntExact(id.value()));
        }

        /**
         * Writes a fully static shape shared by layout and reduction attributes.
         * @param shape exact nonnegative static target, including scalar/empty
         * @throws IllegalArgumentException if the shape is unresolved
         * @throws ArithmeticException if its element count overflows
         */
        void shape(Shape shape) {
            if (!shape.isFullyStatic()) throw invalid("Unresolved attribute shape");
            shape.knownElementCount();
            i32(shape.rank());
            for (long extent : shape.toLongArray()) i64(extent);
        }

        /**
         * Uses stable explicit IDs, not enum ordinals, at every dtype site.
         * @param type exact Model dtype
         */
        void dtype(DataType type) {
            u8(switch (type) {
                case FLOAT64 -> 1; case FLOAT32 -> 2; case BFLOAT16 -> 3; case INT32 -> 4;
                case INT64 -> 5; case BOOL -> 6; case FLOAT16 -> 7;
            });
        }

        /**
         * Preserves typed scalar attribute bits, including signed zero and NaN payloads.
         * @param value exact immutable scalar, never converted through another dtype
         */
        void scalar(ScalarValue value) {
            dtype(value.dataType());
            switch (value.dataType()) {
                case FLOAT64 -> i64(Double.doubleToRawLongBits(value.float64Value()));
                case FLOAT32 -> i32(Float.floatToRawIntBits(value.float32Value()));
                case INT64 -> i64(value.int64Value());
                case INT32 -> i32(value.int32Value());
                case BFLOAT16, FLOAT16 -> {
                    short bits = value.dataType() == DataType.BFLOAT16 ? value.bfloat16Bits() : value.float16Bits();
                    u8(bits >>> 8); u8(bits);
                }
                case BOOL -> u8(value.booleanValue() ? 1 : 0);
            }
        }

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
         * @throws IllegalArgumentException if shape is not fully static
         * @throws ArithmeticException if logical byte/stride geometry overflows
         */
        void descriptor(TensorDescriptor descriptor) {
            GraphValueStorage.bytes(descriptor);
            dtype(descriptor.dataType());
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
