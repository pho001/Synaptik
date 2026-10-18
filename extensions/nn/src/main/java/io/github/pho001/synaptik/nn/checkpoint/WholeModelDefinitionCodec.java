package io.github.pho001.synaptik.nn.checkpoint;

import io.github.pho001.synaptik.checkpoint.artifact.ArtifactFormatException;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.convolution.Conv1dAttrs;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dAttrs;
import io.github.pho001.synaptik.model.operation.convolution.Conv3dAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.nn.module.ModelSpecification;
import io.github.pho001.synaptik.nn.module.ModelSpecification.*;
import java.nio.BufferUnderflowException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * Cold, metadata-only SYNDEF01 codec. Wire identities are explicit, independent of enum order.
 * Raw admission bounds recursion, counts, rank and strings before decoded allocation; NN owns
 * semantic geometry and canonical parameter-identity validation. No values or Tensors are created.
 */
final class WholeModelDefinitionCodec {
    static final int MAX_BYTES = 1048576;
    private static final long MAGIC = 0x53594e4445463031L;

    private WholeModelDefinitionCodec() { }

    /**
     * Measures canonical metadata without allocating a definition byte array.
     * @param specification non-null NN-validated closed configuration
     * @return exact encoded size, in bytes, at most one MiB
     * @throws IllegalArgumentException if the fixed definition ceiling is exceeded
     * @throws ArithmeticException if checked size arithmetic overflows
     */
    static int size(ModelSpecification specification) {
        Sink sink = new Sink(null);
        write(sink, specification);
        if (sink.size > MAX_BYTES) throw new IllegalArgumentException("Definition exceeds 1 MiB");
        return Math.toIntExact(sink.size);
    }

    /**
     * Encodes only after dry sizing has admitted the fixed metadata ceiling.
     * @param specification non-null NN-validated configuration, including current aliases
     * @return new exact canonical big-endian definition bytes
     * @throws IllegalArgumentException if the definition exceeds one MiB
     * @throws ArithmeticException if checked size arithmetic overflows
     */
    static byte[] encode(ModelSpecification specification) {
        ByteBuffer bytes = ByteBuffer.allocate(size(specification)).order(ByteOrder.BIG_ENDIAN);
        write(new Sink(bytes), specification);
        return bytes.array();
    }

    /**
     * Validates raw metadata and delegates semantic validation to NN, without reconstructing state.
     * @param bytes non-null read-only or writable definition view; its cursor is not modified
     * @return fully validated immutable specification with no live state
     * @throws ArtifactFormatException if wire bounds, canonical fields or NN semantics are invalid
     */
    static ModelSpecification decode(ByteBuffer bytes) throws ArtifactFormatException {
        if (bytes.remaining() > MAX_BYTES) throw malformed("Definition exceeds 1 MiB");
        try {
            Reader reader = new Reader(bytes.slice().order(ByteOrder.BIG_ENDIAN));
            if (reader.bytes.getLong() != MAGIC || reader.bytes.getInt() != 1)
                throw malformed("Unsupported definition magic/version");
            Step step = reader.node(1);
            if (!(step instanceof Sequence root)) throw malformed("Definition root must be Sequence");
            ModelSpecification unaliased = new ModelSpecification(root, List.of());
            int states = unaliased.stateDescriptors().size();
            int groups = reader.count(states / 2, 14, "alias groups");
            List<AliasGroup> aliases = new ArrayList<>(groups);
            int members = 0;
            for (int i = 0; i < groups; i++) {
                int count = reader.count(states - members, 5, "alias members");
                if (count < 2) throw malformed("Alias group requires two members");
                members = Math.addExact(members, count);
                List<String> paths = new ArrayList<>(count);
                for (int j = 0; j < count; j++) paths.add(reader.path());
                aliases.add(new AliasGroup(paths));
            }
            if (reader.bytes.hasRemaining()) throw malformed("Trailing definition bytes");
            return new ModelSpecification(root, aliases);
        } catch (BufferUnderflowException | IllegalArgumentException | ArithmeticException failure) {
            throw new ArtifactFormatException("Malformed whole-model definition", failure);
        }
    }

    /**
     * Traverses canonical fields identically for sizing and encoding, without state access.
     * @param sink dry or bounded-byte destination, never null
     * @param specification non-null NN-validated immutable definition
     * @throws ArithmeticException if checked byte counting overflows
     */
    private static void write(Sink sink, ModelSpecification specification) {
        sink.i64(MAGIC); sink.i32(1);
        writeNode(sink, specification.root());
        sink.i32(specification.aliases().size());
        for (AliasGroup group : specification.aliases()) {
            sink.i32(group.paths().size());
            for (String path : group.paths()) {
                // Derived NN paths are ASCII numeric indices and closed parameter names.
                sink.i32(path.length());
                for (int i = 0; i < path.length(); i++) sink.u8(path.charAt(i));
            }
        }
    }

    /**
     * Writes one admitted bounded subtree in prefix numeric encounter order.
     * @param s non-null sizing or encoding destination
     * @param node non-null closed step from an NN-validated tree
     * @throws ArithmeticException if checked byte counting overflows
     */
    private static void writeNode(Sink s, Step node) {
        switch (node) {
            case Sequence n -> {
                s.u8(1); s.i32(n.steps().size());
                for (Step child : n.steps()) writeNode(s, child);
            }
            case LinearStep n -> {
                s.u8(2); s.i64(n.inFeatures()); s.i64(n.outFeatures());
                s.u8(n.bias() ? 1 : 0); s.u8(typeId(n.dataType())); s.u8(n.automaticInputGuards() ? 1 : 0);
            }
            case LayerNormStep n -> {
                s.u8(3); s.u8(typeId(n.dataType())); s.i32(n.normalizedShape().rank());
                for (int i = 0; i < n.normalizedShape().rank(); i++)
                    s.i64(((StaticDimension) n.normalizedShape().dimension(i)).size());
                writeScalar(s, n.epsilon());
            }
            case EmbeddingStep n -> {
                s.u8(4); s.i64(n.vocabularySize()); s.i64(n.embeddingSize()); s.u8(typeId(n.dataType()));
            }
            case Conv1dStep n -> {
                s.u8(5); s.i64(n.inChannels()); s.i64(n.outChannels()); s.i64(n.kernelWidth());
                s.u8(n.bias() ? 1 : 0); s.u8(typeId(n.dataType()));
                var a = n.attrs();
                s.i64(a.stride()); s.i64(a.padding()); s.i64(a.dilation()); s.i64(a.groups());
            }
            case Conv2dStep n -> {
                s.u8(6); s.i64(n.inChannels()); s.i64(n.outChannels());
                s.i64(n.kernelHeight()); s.i64(n.kernelWidth());
                s.u8(n.bias() ? 1 : 0); s.u8(typeId(n.dataType()));
                var a = n.attrs();
                s.i64(a.strideHeight()); s.i64(a.strideWidth()); s.i64(a.paddingHeight());
                s.i64(a.paddingWidth()); s.i64(a.dilationHeight()); s.i64(a.dilationWidth()); s.i64(a.groups());
            }
            case Conv3dStep n -> {
                s.u8(7); s.i64(n.inChannels()); s.i64(n.outChannels()); s.i64(n.kernelDepth());
                s.i64(n.kernelHeight()); s.i64(n.kernelWidth());
                s.u8(n.bias() ? 1 : 0); s.u8(typeId(n.dataType()));
                var a = n.attrs();
                s.i64(a.strideDepth()); s.i64(a.strideHeight()); s.i64(a.strideWidth());
                s.i64(a.paddingDepth()); s.i64(a.paddingHeight()); s.i64(a.paddingWidth());
                s.i64(a.dilationDepth()); s.i64(a.dilationHeight()); s.i64(a.dilationWidth()); s.i64(a.groups());
            }
            case ActivationStep n -> {
                s.u8(8); s.u8(switch (n.kind()) { case RELU -> 1; case TANH -> 2; case SIGMOID -> 3; });
            }
        }
    }

    /**
     * Preserves scalar logical width and raw bits without numeric conversion or normalization.
     * @param s non-null sizing or encoding destination
     * @param value non-null exact typed scalar
     */
    private static void writeScalar(Sink s, ScalarValue value) {
        s.u8(typeId(value.dataType()));
        switch (value.dataType()) {
            case FLOAT64 -> s.i64(Double.doubleToRawLongBits(value.float64Value()));
            case FLOAT32 -> s.i32(Float.floatToRawIntBits(value.float32Value()));
            case BFLOAT16 -> s.i16(value.bfloat16Bits());
            case FLOAT16 -> s.i16(value.float16Bits());
            case INT32 -> s.i32(value.int32Value());
            case INT64 -> s.i64(value.int64Value());
            case BOOL -> s.u8(value.booleanValue() ? 1 : 0);
        }
    }

    /**
     * Assigns the frozen model.v1 wire IDs, deliberately unrelated to Java enum ordinals.
     * @param type non-null Model data type
     * @return stable positive unsigned-byte wire ID
     */
    private static int typeId(DataType type) {
        return switch (type) {
            case FLOAT64 -> 1; case FLOAT32 -> 2; case BFLOAT16 -> 3;
            case INT32 -> 4; case INT64 -> 5; case BOOL -> 6; case FLOAT16 -> 7;
        };
    }

    /**
     * Identifies direct wire rejection rather than an owner validation failure.
     * @param message non-null malformed-field diagnostic
     * @return checked format failure carrying this diagnostic
     */
    private static ArtifactFormatException malformed(String message) {
        return new ArtifactFormatException(message);
    }

    /** One writing algorithm serves both checked dry sizing and bounded encoding. */
    private static final class Sink {
        private final ByteBuffer bytes;
        private long size;
        /**
         * Selects dry counting or writing into a previously admitted exact-size destination.
         * @param bytes nullable destination; null performs counting only
         */
        Sink(ByteBuffer bytes) { this.bytes = bytes; }
        /**
         * Counts and optionally writes one raw byte.
         * @param value low eight bits to retain
         */
        void u8(int value) { size = Math.addExact(size, 1); if (bytes != null) bytes.put((byte) value); }
        /**
         * Counts and optionally writes a big-endian two-byte bit pattern.
         * @param value exact logical bits
         */
        void i16(short value) { size = Math.addExact(size, 2); if (bytes != null) bytes.putShort(value); }
        /**
         * Counts and optionally writes a big-endian four-byte value.
         * @param value exact signed value or raw logical bits
         */
        void i32(int value) { size = Math.addExact(size, 4); if (bytes != null) bytes.putInt(value); }
        /**
         * Counts and optionally writes a big-endian eight-byte value.
         * @param value exact signed value or raw logical bits
         */
        void i64(long value) { size = Math.addExact(size, 8); if (bytes != null) bytes.putLong(value); }
    }

    /** Recursion is admitted one frame at a time, never beyond depth 64 or 4096 nodes. */
    private static final class Reader {
        private final ByteBuffer bytes;
        private int nodes;
        /**
         * Borrows a private view already bounded by the fixed definition ceiling.
         * @param bytes non-null big-endian view with an independent cursor
         */
        Reader(ByteBuffer bytes) { this.bytes = bytes; }

        /**
         * Consumes one byte without sign extension.
         * @return unsigned value in 0..255
         * @throws BufferUnderflowException if truncated; the outer decoder translates it
         */
        int u8() { return Byte.toUnsignedInt(bytes.get()); }
        /**
         * Consumes one canonical zero-or-one flag.
         * @return true only for encoded one
         * @throws ArtifactFormatException if the flag is not canonical
         */
        boolean flag() throws ArtifactFormatException {
            int flag = u8();
            if (flag > 1) throw malformed("Noncanonical definition flag");
            return flag == 1;
        }
        /**
         * Decodes explicit frozen type IDs rather than enum ordinal or name.
         * @return non-null exact Model data type
         * @throws ArtifactFormatException if the type ID is unknown
         */
        DataType type() throws ArtifactFormatException {
            return switch (u8()) {
                case 1 -> DataType.FLOAT64; case 2 -> DataType.FLOAT32; case 3 -> DataType.BFLOAT16;
                case 4 -> DataType.INT32; case 5 -> DataType.INT64; case 6 -> DataType.BOOL;
                case 7 -> DataType.FLOAT16; default -> throw malformed("Unknown definition DataType ID");
            };
        }
        /**
         * Admits a raw count against both semantic capacity and remaining encoded bytes.
         * @param maximum remaining semantic capacity, nonnegative
         * @param minimumBytes positive minimum encoded bytes per item
         * @param role non-null diagnostic field name
         * @return nonnegative count safe for the immediately following bounded allocation
         * @throws ArtifactFormatException if count is negative or exceeds either bound
         */
        int count(int maximum, int minimumBytes, String role) throws ArtifactFormatException {
            int count = bytes.getInt();
            if (count < 0 || count > maximum || count > bytes.remaining() / minimumBytes)
                throw malformed("Definition " + role + " exceeds bounds");
            return count;
        }
        /**
         * Bounds the raw byte length before strict UTF-8 decoding; NN later checks exact paths.
         * @return non-null exact decoded path text without normalization
         * @throws ArtifactFormatException if length or UTF-8 is malformed
         */
        String path() throws ArtifactFormatException {
            int length = bytes.getInt();
            if (length <= 0 || length > bytes.remaining()) throw malformed("Invalid alias path length");
            ByteBuffer text = bytes.slice(); text.limit(length);
            bytes.position(bytes.position() + length);
            try {
                return StandardCharsets.UTF_8.newDecoder().onMalformedInput(CodingErrorAction.REPORT)
                        .onUnmappableCharacter(CodingErrorAction.REPORT).decode(text).toString();
            } catch (CharacterCodingException failure) {
                throw new ArtifactFormatException("Invalid UTF-8 alias path", failure);
            }
        }
        /**
         * Admits depth/count before recursion and length/rank before decoded collection allocation.
         * @param depth current occurrence depth, root one
         * @return validated closed semantic step, never null
         * @throws ArtifactFormatException if raw fields, IDs or bounded counts are invalid
         * @throws IllegalArgumentException if NN/Model constructors reject semantic attributes
         * @throws ArithmeticException if owner checked shape/geometry arithmetic overflows
         */
        Step node(int depth) throws ArtifactFormatException {
            if (depth > 64 || ++nodes > 4096) throw malformed("Definition node count/depth exceeded");
            return switch (u8()) {
                case 1 -> {
                    int count = count(4096 - nodes, 2, "children");
                    List<Step> children = new ArrayList<>(count);
                    for (int i = 0; i < count; i++) children.add(node(depth + 1));
                    yield new Sequence(children);
                }
                case 2 -> new LinearStep(bytes.getLong(), bytes.getLong(), flag(), type(), flag());
                case 3 -> {
                    DataType type = type();
                    int rank = bytes.getInt();
                    // Reserve the scalar tag and at least its smallest logical width before allocation.
                    if (rank <= 0 || 8L * rank + 2 > bytes.remaining())
                        throw malformed("Definition normalized rank exceeds remaining bytes");
                    int scalarPosition = bytes.position() + Math.toIntExact(8L * rank);
                    int scalarId = Byte.toUnsignedInt(bytes.get(scalarPosition));
                    int width = switch (scalarId) {
                        case 1, 5 -> 8; case 2, 4 -> 4; case 3, 7 -> 2; case 6 -> 1;
                        default -> throw malformed("Unknown scalar DataType ID");
                    };
                    if (8L * rank + 1 + width > bytes.remaining())
                        throw malformed("Truncated normalized Shape/scalar");
                    long[] extents = new long[rank];
                    for (int i = 0; i < rank; i++) extents[i] = bytes.getLong();
                    yield new LayerNormStep(Shape.of(extents), type, scalar());
                }
                case 4 -> new EmbeddingStep(bytes.getLong(), bytes.getLong(), type());
                case 5 -> new Conv1dStep(bytes.getLong(), bytes.getLong(), bytes.getLong(), flag(), type(),
                        new Conv1dAttrs(bytes.getLong(), bytes.getLong(), bytes.getLong(), bytes.getLong()));
                case 6 -> new Conv2dStep(bytes.getLong(), bytes.getLong(), bytes.getLong(), bytes.getLong(),
                        flag(), type(), new Conv2dAttrs(bytes.getLong(), bytes.getLong(), bytes.getLong(),
                        bytes.getLong(), bytes.getLong(), bytes.getLong(), bytes.getLong()));
                case 7 -> new Conv3dStep(bytes.getLong(), bytes.getLong(), bytes.getLong(), bytes.getLong(),
                        bytes.getLong(), flag(), type(), new Conv3dAttrs(bytes.getLong(), bytes.getLong(),
                        bytes.getLong(), bytes.getLong(), bytes.getLong(), bytes.getLong(), bytes.getLong(),
                        bytes.getLong(), bytes.getLong(), bytes.getLong()));
                case 8 -> new ActivationStep(switch (u8()) {
                    case 1 -> ActivationKind.RELU; case 2 -> ActivationKind.TANH; case 3 -> ActivationKind.SIGMOID;
                    default -> throw malformed("Unknown activation ID");
                });
                default -> throw malformed("Unknown definition node ID");
            };
        }
        /**
         * Restores exact typed bits; the consuming NN constructor retains epsilon validation.
         * @return non-null raw-preserving Model scalar, with no Tensor or storage
         * @throws ArtifactFormatException if type ID or BOOL flag is unknown/noncanonical
         */
        ScalarValue scalar() throws ArtifactFormatException {
            return switch (type()) {
                case FLOAT64 -> ScalarValue.float64(Double.longBitsToDouble(bytes.getLong()));
                case FLOAT32 -> ScalarValue.float32(Float.intBitsToFloat(bytes.getInt()));
                case BFLOAT16 -> ScalarValue.bfloat16Bits(bytes.getShort());
                case FLOAT16 -> ScalarValue.float16Bits(bytes.getShort());
                case INT32 -> ScalarValue.int32(bytes.getInt());
                case INT64 -> ScalarValue.int64(bytes.getLong());
                case BOOL -> ScalarValue.bool(flag());
            };
        }
    }
}
