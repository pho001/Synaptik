package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarElementwiseKind;
import java.io.*;
import java.lang.classfile.*;
import java.lang.classfile.instruction.InvokeInstruction;
import java.net.URI;
import java.util.*;
import javax.tools.*;

/**
 * Independently javac-compiled, typed counterparts for the finite scalar/clamp matrix.
 *
 * <p>This test-only oracle keeps selected-Code observations separate from the limited,
 * dependency-based equivalence facts used to compare an emitted specialization with an
 * independently written clean-Java counterpart.</p>
 */
final class CpuScalarImmediateClampCleanJavaOracle {
    private static final String PACKAGE = "io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.generated";
    private static final String NAME = "ScalarImmediateClampCleanJava";
    private CpuScalarImmediateClampCleanJavaOracle() { }

    /**
     * Observations from one exact selected Code attribute, split between evidence that remains
     * literal and facts that can be compared across independently optimal Java spellings.
     *
     * @param descriptor exact ordered member ABI
     * @param carrierRoles ordered input and selected-output carrier observations
     * @param controlFlow normalized loop and branch skeleton
     * @param rawAddressDataflow selected carrier-address events in their original Code order
     * @param addressDataflow paired canonical address facts; it normalizes only proven
     *     independent monotonic input/output cursor scheduling, including a direct ordinal or
     *     shared selected-address root, never selected role, range, carrier, conversion, or
     *     dependency-sensitive address facts
     * @param algorithm selected-store operation and operand dataflow
     * @param conversion carrier conversion placement and kind
     * @param outputStore selected output-store identity
     * @param provenance selected-Code provenance observations retained for diagnostics, not
     *     stamped into either paired Code projection as equivalence evidence
     * @param forbidden forbidden allocation, dispatch, reflection, boxing, or synchronization
     *     observations
     */
    record Projection(String descriptor, List<String> carrierRoles, String controlFlow,
                      String rawAddressDataflow, String addressDataflow, String algorithm,
                      String conversion, String outputStore, String provenance,
                      List<String> forbidden) { }

    /** Ordered prepared topology facts; these are not Code-projection facts. */
    record Topology(String inputShape, String inputLayout, String outputShape, String outputLayout,
                    String accessRegime, List<CarrierAccess> carrierRoles) { }

    /**
     * Exact association between two independently selected artifacts.  This remains deliberately
     * outside {@link Projection}: Code does not encode preparation candidates and clean javac
     * output does not encode preparation strategy.  Each field is independently reconstructible;
     * no opaque concatenated identity is permitted because later ledger integration joins rows on
     * these exact typed facts.
     */
    record PairBinding(String formId, Topology topology, String generatedClassSha256,
                       String selectedEntryDescriptor, String generatedMemberSchemaHash,
                       String preparedIrStructuralKey, String generatorSchema,
                       String classIdentitySchema, String selectedStrategy,
                       int materializationCandidateCount, int materializationSelectedCount,
                       String materializationPolicyIdentity, String cleanClassSha256,
                       String cleanSelectedMethodName, String cleanSelectedMethodDescriptor,
                       String cleanMemberSchemaHash, String cleanSourcePolicyIdentity,
                       String cleanSourcePolicyHash) { }
    private enum AddressProgram { DENSE, ZERO, LAST_AXIS, BLOCK_OUTER, GENERAL }
    private record SourceProgram(AddressProgram address, boolean vector) { }

    /** One class artifact plus the independently selected member bytes/schema fingerprints. */
    record Compilation(byte[] classBytes, Map<String, String> methodSchemaHashes) { }

    /** Exact generated/clean identity binding used by support and semantic evidence. */
    record PairedArtifact(String formId, PairBinding binding) { }

    static Compilation compile(List<CpuScalarImmediateClampMatrixOracle.Form> forms) {
        JavaCompiler compiler = ToolProvider.getSystemJavaCompiler();
        if (compiler == null) throw new AssertionError("JDK compiler is required for clean-Java proof");
        Map<String, ByteFile> output = new LinkedHashMap<>();
        Map<String, SourceProgram> programs = new LinkedHashMap<>();
        // This choice intentionally has no Class-File input.  A form already carries the semantic
        // operation, exact typed values, layouts/shapes, direct carriers, and requested compute
        // preference.  Dense/zero/bias/block/odometer are the five independent loop algorithms
        // for those layouts; the vector spelling is selected from the requested preference and
        // the Java Vector API element types, not from the generated artifact.
        for (var form : forms) programs.put(form.id(), semanticProgram(form));
        try (StandardJavaFileManager standard = compiler.getStandardFileManager(null, null, null)) {
            JavaFileManager manager = new ForwardingJavaFileManager<>(standard) {
                @Override public JavaFileObject getJavaFileForOutput(Location l, String n,
                        JavaFileObject.Kind k, FileObject s) {
                    ByteFile file = new ByteFile(n); output.put(n, file); return file;
                }
            };
            DiagnosticCollector<JavaFileObject> diagnostics = new DiagnosticCollector<>();
            boolean ok = compiler.getTask(null, manager, diagnostics, List.of("--release", "26"), null,
                    List.of(new Source(PACKAGE + '.' + NAME, source(forms, programs)))).call();
            assertTrue(ok, "clean-Java matrix compilation: " + diagnostics.getDiagnostics());
        } catch (IOException failure) { throw new AssertionError(failure); }
        byte[] all = output.get(PACKAGE + '.' + NAME).bytes();
        Map<String, String> schemas = new LinkedHashMap<>();
        for (var form : forms) {
            MethodModel selected = selectedMethod(all, method(form), expectedDescriptor(form));
            schemas.put(form.id(), CpuScalarImmediateClampMatrixOracle.sha256(
                    normalizedMethodSchema(selected).getBytes(java.nio.charset.StandardCharsets.UTF_8)));
        }
        return new Compilation(all, Map.copyOf(schemas));
    }

    /**
     * Compiles a deliberately supplied clean-Java mutation fixture.  The structural negative
     * controls use this seam so that they exercise the same Class-File selection, tokenization,
     * projection, and comparison path as a matrix counterpart; they never edit a projection.
     *
     * @param binaryName binary name of the one supplied compilation unit
     * @param source independent Java source for that unit
     * @return the actual javac-produced Class-File bytes for {@code binaryName}
     */
    static byte[] compileFixture(String binaryName, String source) {
        JavaCompiler compiler = ToolProvider.getSystemJavaCompiler();
        if (compiler == null) throw new AssertionError("JDK compiler is required for mutation proof");
        Map<String, ByteFile> output = new LinkedHashMap<>();
        try (StandardJavaFileManager standard = compiler.getStandardFileManager(null, null, null)) {
            JavaFileManager manager = new ForwardingJavaFileManager<>(standard) {
                @Override public JavaFileObject getJavaFileForOutput(Location location, String name,
                        JavaFileObject.Kind kind, FileObject sibling) {
                    ByteFile file = new ByteFile(name);
                    output.put(name, file);
                    return file;
                }
            };
            DiagnosticCollector<JavaFileObject> diagnostics = new DiagnosticCollector<>();
            boolean ok = compiler.getTask(null, manager, diagnostics, List.of("--release", "26"), null,
                    List.of(new Source(binaryName, source))).call();
            assertTrue(ok, "clean-Java mutation fixture compilation: " + diagnostics.getDiagnostics());
        } catch (IOException failure) {
            throw new AssertionError(failure);
        }
        ByteFile result = output.get(binaryName);
        assertNotNull(result, "fixture compiler did not produce " + binaryName);
        return result.bytes();
    }

    private static SourceProgram semanticProgram(CpuScalarImmediateClampMatrixOracle.Form form) {
        AddressProgram address = switch (form.accessRegime()) {
            case DENSE_LINEAR -> AddressProgram.DENSE;
            case SCALAR_ALL_ZERO -> AddressProgram.ZERO;
            case LAST_AXIS_BIAS -> AddressProgram.LAST_AXIS;
            case BLOCK_OUTER -> AddressProgram.BLOCK_OUTER;
            case GENERAL_ODOMETER -> AddressProgram.GENERAL;
        };
        boolean vector = form.fixture().preference()
                == io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.VECTOR_IF_ELIGIBLE
                && form.fixture().type() != DataType.BFLOAT16
                // The Vector API has no lane-wise pow.  Direct pow is therefore independently
                // optimal as a scalar StrictMath loop; the algebraic pow specializations below
                // are vectorizable because their operation is a primitive vector operation.
                && !(form.fixture().operation() == ScalarElementwiseKind.POW
                && (form.fixture().category() == CpuScalarImmediateClampMatrixOracle.Category.DIRECT_FRACTIONAL
                || form.fixture().category() == CpuScalarImmediateClampMatrixOracle.Category.DIRECT_INTEGRAL))
                // A direct contiguous range is the independently vectorizable specialization.
                // Broadcast/non-contiguous cursors retain their scalar cursor machines; that
                // avoids per-vector coordinate repair and preserves the one-element semantics.
                && vectorizableAddress(form)
                && form.inputShape().rank() > 0
                && form.outputShape().knownElementCount().orElseThrow() > 0;
        return new SourceProgram(address, vector);
    }
    private static boolean vectorizableAddress(CpuScalarImmediateClampMatrixOracle.Form form) {
        return switch (form.accessRegime()) {
            case DENSE_LINEAR, SCALAR_ALL_ZERO -> true;
            case LAST_AXIS_BIAS, BLOCK_OUTER -> form.fixture().type() == DataType.FLOAT64
                    || form.fixture().type() == DataType.INT64;
            case GENERAL_ODOMETER -> false;
        };
    }

    static PairedArtifact pair(CpuScalarImmediateClampMatrixOracle.Form form, Compilation clean) {
        var formArtifact = CpuScalarImmediateClampMatrixOracle.formArtifact(form);
        return new PairedArtifact(form.id(), pairBinding(formArtifact, clean));
    }
    private static PairBinding pairBinding(CpuScalarImmediateClampMatrixOracle.FormArtifact artifact,
            Compilation clean) {
        var form = artifact.form();
        var generated = artifact.artifact();
        String policy = cleanSourcePolicy(form);
        return new PairBinding(form.id(), topology(form), generated.hash(), generated.descriptor(),
                memberHash(generated.bytes(), null, generated.descriptor()), generated.structuralKey(),
                generated.generatorSchema(), generated.classIdentitySchema(), artifact.selectedStrategy(),
                artifact.materializationCandidates(), artifact.materializationSelected(),
                materializationPolicyIdentity(form), CpuScalarImmediateClampMatrixOracle.sha256(clean.classBytes()),
                method(form), expectedDescriptor(form), clean.methodSchemaHashes().get(form.id()), policy,
                CpuScalarImmediateClampMatrixOracle.sha256(policy.getBytes(java.nio.charset.StandardCharsets.UTF_8)));
    }

    private static Topology topology(CpuScalarImmediateClampMatrixOracle.Form form) {
        return new Topology(form.inputShape().toString(), form.inputLayout().toString(),
                form.outputShape().toString(), form.outputLayout().toString(), form.accessRegime().name(),
                List.copyOf(form.fixture().carriers()));
    }
    private static String materializationPolicyIdentity(CpuScalarImmediateClampMatrixOracle.Form form) {
        return form.materializationPolicy().toString();
    }
    private static String cleanSourcePolicy(CpuScalarImmediateClampMatrixOracle.Form form) {
        SourceProgram program = semanticProgram(form);
        return "semantic-address=" + program.address() + ";vector=" + program.vector();
    }

    static void validatePairBinding(CpuScalarImmediateClampMatrixOracle.Form form,
            PairedArtifact pair, Compilation freshlyCompiledClean) {
        PairBinding binding = pair.binding();
        var prepared = CpuScalarImmediateClampMatrixOracle.formArtifact(form);
        var generated = prepared.artifact();
        assertEquals(form.id(), pair.formId(), "paired artifact form identity");
        assertEquals(form.id(), binding.formId(), form.id() + " pair binding form identity");
        assertEquals(topology(form), binding.topology(), form.id() + " prepared topology");
        assertEquals(generated.hash(), binding.generatedClassSha256(), form.id() + " generated class hash");
        assertEquals(generated.descriptor(), binding.selectedEntryDescriptor(), form.id() + " generated entry ABI");
        assertEquals(memberHash(generated.bytes(), null, generated.descriptor()), binding.generatedMemberSchemaHash(),
                form.id() + " generated selected member schema");
        assertEquals(generated.structuralKey(), binding.preparedIrStructuralKey(), form.id() + " prepared IR key");
        assertEquals(generated.generatorSchema(), binding.generatorSchema(), form.id() + " generator schema");
        assertEquals(generated.classIdentitySchema(), binding.classIdentitySchema(), form.id() + " class identity schema");
        assertEquals(prepared.selectedStrategy(), binding.selectedStrategy(), form.id() + " selected strategy");
        assertEquals(prepared.materializationCandidates(), binding.materializationCandidateCount(), form.id() + " candidate count");
        assertEquals(prepared.materializationSelected(), binding.materializationSelectedCount(), form.id() + " selected count");
        assertEquals(materializationPolicyIdentity(form), binding.materializationPolicyIdentity(), form.id() + " policy identity");
        assertEquals(CpuScalarImmediateClampMatrixOracle.sha256(freshlyCompiledClean.classBytes()), binding.cleanClassSha256(),
                form.id() + " fresh clean class hash");
        assertEquals(method(form), binding.cleanSelectedMethodName(), form.id() + " clean method name");
        assertEquals(expectedDescriptor(form), binding.cleanSelectedMethodDescriptor(), form.id() + " clean method ABI");
        assertEquals(freshlyCompiledClean.methodSchemaHashes().get(form.id()), binding.cleanMemberSchemaHash(),
                form.id() + " fresh clean member schema");
        String policy = cleanSourcePolicy(form);
        assertEquals(policy, binding.cleanSourcePolicyIdentity(), form.id() + " clean source policy");
        assertEquals(CpuScalarImmediateClampMatrixOracle.sha256(policy.getBytes(java.nio.charset.StandardCharsets.UTF_8)),
                binding.cleanSourcePolicyHash(), form.id() + " clean source policy hash");
    }

    static Projection projection(byte[] bytes, String methodName, String descriptor) {
        MethodModel method = methodName == null
                ? selectedGeneratedMethod(bytes, descriptor)
                : selectedMethod(bytes, methodName, descriptor);
        List<String> tokens = tokens(method);
        assertFalse(tokens.isEmpty(), "selected compiled method has no body");
        AddressFacts addresses = address(method, tokens);
        return new Projection(method.methodType().stringValue(), carrierRoles(method, tokens),
                flow(method), addresses.raw(), addresses.canonical(), algorithm(method, tokens), conversion(tokens),
                store(tokens), provenance(bytes, method, tokens), forbidden(method, tokens));
    }

    private static String memberHash(byte[] bytes, String methodName, String descriptor) {
        MethodModel method = methodName == null ? selectedGeneratedMethod(bytes, descriptor)
                : selectedMethod(bytes, methodName, descriptor);
        return CpuScalarImmediateClampMatrixOracle.sha256(normalizedMethodSchema(method)
                .getBytes(java.nio.charset.StandardCharsets.UTF_8));
    }

    private static MethodModel selectedMethod(byte[] bytes, String name, String descriptor) {
        List<MethodModel> matches = ClassFile.of().parse(bytes).methods().stream()
                .filter(m -> m.methodName().stringValue().equals(name)
                        && m.methodType().stringValue().equals(descriptor)).toList();
        assertEquals(1, matches.size(), "exactly one selected method " + name + descriptor);
        return matches.getFirst();
    }
    private static MethodModel selectedGeneratedMethod(byte[] bytes, String descriptor) {
        List<MethodModel> methods = ClassFile.of().parse(bytes).methods();
        assertEquals(1, methods.size(), "generated artifact has one direct entry method");
        MethodModel method = methods.getFirst();
        assertEquals(descriptor, method.methodType().stringValue(), "generated exact entry descriptor");
        return method;
    }

    private static String expectedDescriptor(CpuScalarImmediateClampMatrixOracle.Form form) {
        var f = form.fixture();
        return "(" + descriptor(f.type(), f.carriers().getFirst()) + descriptor(f.type(), f.carriers().getLast())
                + "[JJJ)V";
    }
    private static String descriptor(DataType type, CarrierAccess carrier) {
        if (carrier == CarrierAccess.MEMORY_SEGMENT) return "Ljava/lang/foreign/MemorySegment;";
        return switch (type) { case BFLOAT16 -> "[S"; case FLOAT32 -> "[F"; case FLOAT64 -> "[D";
            case INT32 -> "[I"; case INT64 -> "[J"; default -> throw new AssertionError(type); };
    }
    private static String normalizedMethodSchema(MethodModel method) {
        return method.methodName().stringValue() + method.methodType().stringValue() + '|'
                + String.join(";", tokens(method).stream().map(t -> t.substring(0, t.indexOf('|'))).toList());
    }

    static void compare(CpuScalarImmediateClampMatrixOracle.Form form, Projection generated,
            Projection clean) {
        assertEquals(generated.descriptor(), clean.descriptor(), form.id() + " exact ordered typed ABI");
        assertEquals(generated.carrierRoles(), clean.carrierRoles(), form.id() + " carrier access roles");
        assertEquals(generated.controlFlow(), clean.controlFlow(), form.id() + " control flow/back edges");
        assertEquals(generated.addressDataflow(), clean.addressDataflow(), form.id() + " address/dataflow regime");
        assertEquals(generated.algorithm(), clean.algorithm(), form.id() + " semantic algorithm");
        assertEquals(generated.conversion(), clean.conversion(), form.id() + " conversion");
        assertEquals(generated.outputStore(), clean.outputStore(), form.id() + " output store");
        // Preparation/materialization provenance is deliberately PairBinding-only.  It is not
        // stamped into both independently compiled Code projections, where matching strings
        // would be circular evidence rather than a structural equivalence fact.
        assertNoForbiddenObservations(form, generated, clean);
    }

    /**
     * Enforces the closed selected-Code invocation and overhead allowlist for one exact form.
     *
     * <p>This deliberately checks each selected artifact independently.  It is not a
     * generated-versus-{@code javac} structural-equivalence assertion: the two valid clean-Java
     * spellings may differ in their control-flow shape.</p>
     *
     * @param form exact finite matrix form used solely for failure attribution
     * @param generated projection of the generated entry member
     * @param clean projection of the independently javac-compiled typed counterpart member
     */
    static void assertNoForbiddenObservations(CpuScalarImmediateClampMatrixOracle.Form form,
            Projection generated, Projection clean) {
        assertTrue(generated.forbidden().isEmpty(),
                form.id() + " generated forbidden overhead: " + generated.forbidden());
        assertTrue(clean.forbidden().isEmpty(),
                form.id() + " clean forbidden overhead: " + clean.forbidden());
    }

    /**
     * Inspects the two actual selected Code members with the closed invocation allowlist.
     *
     * <p>This narrower entry point intentionally avoids constructing a {@link Projection}:
     * projection also validates structural dataflow facts that are not required to decide
     * whether an invocation is allowed.  The inspection remains form-attributable and checks
     * both the generated member and the independently javac-compiled typed counterpart.</p>
     *
     * @param form exact finite matrix form used solely for failure attribution
     * @param generatedBytes generated Class-File bytes for {@code form}
     * @param clean compiled clean-Java matrix artifact
     */
    static void assertClosedInvocationAllowlist(CpuScalarImmediateClampMatrixOracle.Form form,
            byte[] generatedBytes, Compilation clean) {
        MethodModel generated = selectedGeneratedMethod(generatedBytes,
                CpuScalarImmediateClampMatrixOracle.formArtifact(form).artifact().descriptor());
        MethodModel cleanMethod = selectedMethod(clean.classBytes(), method(form), expectedDescriptor(form));
        List<String> generatedForbidden = forbidden(generated, tokens(generated));
        List<String> cleanForbidden = forbidden(cleanMethod, tokens(cleanMethod));
        assertTrue(generatedForbidden.isEmpty(),
                form.id() + " generated forbidden overhead: " + generatedForbidden);
        assertTrue(cleanForbidden.isEmpty(),
                form.id() + " clean forbidden overhead: " + cleanForbidden);
    }

    static String rejectedDimension(Projection a, Projection b) {
        if (!a.descriptor.equals(b.descriptor)) return "ABI order";
        if (!a.carrierRoles.equals(b.carrierRoles)) return "carrier access role";
        // Extra calls are forbidden independently of the semantic side effects that a hostile
        // call might also introduce.  Report that closed-world violation first.
        if (!a.forbidden.equals(b.forbidden)) return "forbidden overhead";
        if (!a.controlFlow.equals(b.controlFlow)) return "control-flow/back-edge";
        if (!a.addressDataflow.equals(b.addressDataflow)) return "address regime";
        if (!a.algorithm.equals(b.algorithm)) return "algorithm";
        if (!a.conversion.equals(b.conversion)) return "conversion";
        if (!a.outputStore.equals(b.outputStore)) return "output store";
        return null;
    }
    static boolean rejects(Projection a, Projection b) { return rejectedDimension(a, b) != null; }


    /**
     * Changes exactly one selected member Code byte.  The parser resolves the requested method
     * name/descriptor and its {@code Code} attribute before looking for the opcode; it therefore
     * cannot accidentally edit a constant-pool, another method, or a similarly valued class byte.
     * The caller normally uses a verifier-invalid replacement (for example {@code FASTORE} to
     * {@code DASTORE}) so execution is intentionally not attempted; Class-File parsing and the
     * selected-store extractor must still observe the changed instruction.
     */
    static byte[] mutateSelectedStoreOpcode(byte[] bytes, String name, String descriptor,
            int expectedOpcode, int replacementOpcode) {
        byte[] result = bytes.clone();
        CodeRange code = selectedCodeRange(result, name, descriptor);
        int found = -1;
        for (int offset = code.start(); offset < code.end(); offset++) {
            if (Byte.toUnsignedInt(result[offset]) == expectedOpcode) {
                assertEquals(-1, found, "selected Code has exactly one targeted store opcode");
                found = offset;
            }
        }
        assertTrue(found >= 0, "selected Code has targeted store opcode " + expectedOpcode);
        assertEquals(expectedOpcode, Byte.toUnsignedInt(result[found]), "exact selected pre-mutation opcode");
        result[found] = (byte) replacementOpcode;
        assertEquals(replacementOpcode, Byte.toUnsignedInt(result[found]), "exact selected post-mutation opcode");
        // Parseability is separate from verifier validity.  The latter is expected for a typed
        // store replacement with an incompatible stack value.
        ClassFile.of().parse(result);
        return result;
    }

    private record CodeRange(int start, int end) { }
    private static CodeRange selectedCodeRange(byte[] bytes, String wantedName, String wantedDescriptor) {
        int[] cursor = {8};
        String[] utf8 = constantPoolUtf8(bytes, cursor);
        int at = cursor[0] + 6; // class access, this_class, super_class
        int interfaces = u2(bytes, at); at += 2 + interfaces * 2;
        int fields = u2(bytes, at); at += 2;
        for (int index = 0; index < fields; index++) at = skipMember(bytes, at);
        int methods = u2(bytes, at); at += 2;
        CodeRange result = null;
        for (int index = 0; index < methods; index++) {
            int nameIndex = u2(bytes, at + 2), descriptorIndex = u2(bytes, at + 4);
            int attributes = u2(bytes, at + 6); at += 8;
            boolean selected = wantedName.equals(utf8[nameIndex]) && wantedDescriptor.equals(utf8[descriptorIndex]);
            for (int attribute = 0; attribute < attributes; attribute++) {
                String attributeName = utf8[u2(bytes, at)];
                int length = u4(bytes, at + 2), body = at + 6;
                if (selected && "Code".equals(attributeName)) {
                    assertNull(result, "selected member has one Code attribute");
                    int codeLength = u4(bytes, body + 4);
                    result = new CodeRange(body + 8, body + 8 + codeLength);
                }
                at = body + length;
            }
        }
        assertNotNull(result, "selected member Code " + wantedName + wantedDescriptor);
        return result;
    }
    private static int skipMember(byte[] bytes, int at) {
        int attributes = u2(bytes, at + 6); at += 8;
        for (int index = 0; index < attributes; index++) at += 6 + u4(bytes, at + 2);
        return at;
    }
    private static String[] constantPoolUtf8(byte[] bytes, int[] cursor) {
        int count = u2(bytes, cursor[0]); cursor[0] += 2;
        String[] values = new String[count];
        for (int index = 1; index < count; index++) {
            int tag = Byte.toUnsignedInt(bytes[cursor[0]++]);
            switch (tag) {
                case 1 -> { int length = u2(bytes, cursor[0]); cursor[0] += 2;
                    values[index] = new String(bytes, cursor[0], length, java.nio.charset.StandardCharsets.UTF_8);
                    cursor[0] += length; }
                case 3, 4, 9, 10, 11, 12, 17, 18 -> cursor[0] += tag <= 4 ? 4 : 4;
                case 5, 6 -> { cursor[0] += 8; index++; }
                case 7, 8, 16, 19, 20 -> cursor[0] += 2;
                case 15 -> cursor[0] += 3;
                default -> throw new AssertionError("unknown Class-File constant-pool tag " + tag);
            }
        }
        return values;
    }
    private static int u2(byte[] bytes, int at) { return (Byte.toUnsignedInt(bytes[at]) << 8) | Byte.toUnsignedInt(bytes[at + 1]); }
    private static int u4(byte[] bytes, int at) { return (u2(bytes, at) << 16) | u2(bytes, at + 2); }

    private static List<String> tokens(MethodModel method) {
        return method.code().orElseThrow().elementStream().filter(Instruction.class::isInstance)
                .map(Instruction.class::cast).map(i -> i.opcode().name() + "|" + i).toList();
    }
    private static List<String> carrierRoles(MethodModel method, List<String> tokens) {
        String descriptor = method.methodType().stringValue();
        String input = parameter(descriptor, 0), output = parameter(descriptor, 1);
        String load = input.equals("Ljava/lang/foreign/MemorySegment;") ? "input:segment:get"
                : "input:array:" + arrayOpcode(input, true);
        String store = output.equals("Ljava/lang/foreign/MemorySegment;") ? "output:segment:set"
                : "output:array:" + arrayOpcode(output, false);
        String inputProbe = load.endsWith("get") ? "MemorySegment.get" : load.substring(load.lastIndexOf(':') + 1);
        boolean inputObserved = tokens.stream().anyMatch(t -> t.contains(inputProbe));
        if (!inputObserved) load = "input:unused";
        // The descriptor supplies the declared carrier ABI. The selected instructions supply the
        // actual store provenance, so a byte-level mutation reaches extraction and comparison.
        if (store.endsWith("set")) {
            if (!tokens.stream().anyMatch(t -> t.contains("MemorySegment.set"))) store = "output:missing-store";
        } else {
            String observed = List.of("SASTORE", "FASTORE", "DASTORE", "IASTORE", "LASTORE").stream()
                    .filter(opcode -> tokens.stream().anyMatch(t -> t.startsWith(opcode + "|")))
                    .findFirst().orElse("missing-store");
            store = "output:array:" + observed;
        }
        return List.of(load, store);
    }
    private static String parameter(String descriptor, int number) {
        int at = 1;
        for (int index = 0; index < number; index++) at += descriptor.charAt(at) == '[' ? 2 : descriptor.indexOf(';', at) - at + 1;
        return descriptor.charAt(at) == '[' ? descriptor.substring(at, at + 2) : descriptor.substring(at, descriptor.indexOf(';', at) + 1);
    }
    private static String arrayOpcode(String type, boolean load) {
        if (type.equals("[F")) return load ? "FALOAD" : "FASTORE";
        if (type.equals("[D")) return load ? "DALOAD" : "DASTORE";
        if (type.equals("[I")) return load ? "IALOAD" : "IASTORE";
        if (type.equals("[J")) return load ? "LALOAD" : "LASTORE";
        return load ? "SALOAD" : "SASTORE";
    }
    private static String flow(MethodModel method) {
        var elements = method.code().orElseThrow().elementStream().toList();
        var labelInstruction = new IdentityHashMap<java.lang.classfile.Label, Integer>();
        var branches = new ArrayList<String>();
        int instruction = 0;
        for (var element : elements) {
            if (element instanceof java.lang.classfile.instruction.LabelTarget target) {
                labelInstruction.put(target.label(), instruction);
            }
            if (element instanceof Instruction) instruction++;
        }
        instruction = 0;
        for (var element : elements) {
            if (element instanceof java.lang.classfile.instruction.BranchInstruction branch) {
                int target = labelInstruction.get(branch.target());
                // This is intentionally an ordered edge observation.  The source and direct
                // emitter may split a semantic block around casts/rounding, but a changed branch
                // target or conditional fallthrough remains visible before phase normalization.
                branches.add(instruction + ":" + branch.opcode().name() + "->" + target
                        + (target <= instruction ? ":back" : ":forward"));
            }
            if (element instanceof Instruction) instruction++;
        }
        boolean vector = method.code().orElseThrow().elementStream().filter(InvokeInstruction.class::isInstance)
                .map(InvokeInstruction.class::cast).anyMatch(call -> call.owner().asInternalName().startsWith("jdk/incubator/vector/"));
        boolean guarded = branches.stream().anyMatch(edge -> edge.contains(":IF"));
        assertTrue(guarded, "selected method must expose a conditional range/loop guard: " + branches);
        assertTrue(branches.stream().anyMatch(edge -> edge.endsWith(":back")),
                "selected method must expose a loop-back edge: " + branches);
        boolean auxiliaryLoop = !vector && branches.stream().filter(edge -> edge.endsWith(":back"))
                .map(edge -> edge.substring(edge.indexOf("->") + 2, edge.indexOf(":back"))).distinct().count() > 1;
        String auxiliary = auxiliaryLoop ? "AUXILIARY_LOOP->AUXILIARY_LOOP,SCALAR_MAIN;" : "";
        return vector
                ? "PHASES[START_END_GUARD->VECTOR_MAIN,EXIT;VECTOR_MAIN->VECTOR_SPAN_LIMIT_GUARD,SCALAR_TAIL_HANDOFF;"
                + "VECTOR_SPAN_LIMIT_GUARD->VECTOR_MAIN,SCALAR_TAIL_HANDOFF;SCALAR_TAIL_HANDOFF->SCALAR_MAIN;"
                + "SCALAR_MAIN->CURSOR_RESET_OUTER_TRANSITION,EXIT;CURSOR_RESET_OUTER_TRANSITION->SCALAR_MAIN,EXIT;"
                + auxiliary + "EXIT->RETURN]"
                : "PHASES[START_END_GUARD->SCALAR_MAIN,EXIT;SCALAR_MAIN->CURSOR_RESET_OUTER_TRANSITION,EXIT;"
                + "CURSOR_RESET_OUTER_TRANSITION->SCALAR_MAIN,EXIT;" + auxiliary + "EXIT->RETURN]";
    }
    /**
     * Produces raw selected-address evidence and its paired canonical counterpart.  The raw
     * program deliberately retains every selected cursor update in Code order.  Canonicalization
     * is limited to self-contained, monotonic selected-store addressing: an explicit selected
     * output cursor, direct range-ordinal-plus-base addressing, or one shared selected-address
     * root may be paired with a distinct input cursor only when neither update reads the other.
     * Input resets, strides, general odometers, selected store identity, carrier-specific
     * accesses and conversions, and every dependency-sensitive order remain facts. Thus the
     * canonical form permits only the proved commutative scheduling spelling, while raw facts
     * remain available to the mutation controls.
     *
     * @param method exact selected member
     * @param tokens selected member instructions only
     * @return raw and paired-canonical selected address facts
     */
    private static AddressFacts address(MethodModel method, List<String> tokens) {
        CursorSites sites = cursorSites(method, tokens);
        assertFalse(sites.output().isEmpty(), "typed output store has no local cursor carrier");
        CursorProgram input = cursorProgram(tokens, sites.input());
        CursorProgram output = cursorProgram(tokens, sites.output());
        String scalar = input.fixed() ? "FIXED_INPUT_BROADCAST"
                : input.regime();
        boolean vector = tokens.stream().anyMatch(t -> t.contains("jdk/incubator/vector"));
        String inputProgram = scalar + "[" + input.events() + "]";
        // Output is always a monotonic selected-store cursor in this finite family.  Its local
        // spelling differs (the emitter may fold the vector store index into its loop update),
        // while the input cursor owns the meaningful broadcast/reset transitions.
        // A dense body may feed the selected store directly from ordinal-plus-base rather than
        // from an incremented local.  Both are monotonic selected-store addressing; the input
        // regime and protected non-ordinal cursor-update order retain the distinction needed to
        // reject an address-regime mutation without requiring avoidable output cursor state.
        String outputProgram = "MONOTONIC_SELECTED_STORE";
        String updateOrder = cursorUpdateOrder(tokens, sites);
        String prefix = vector ? "VECTOR_MAIN:" + inputProgram + ";SCALAR_TAIL:CONTINUOUS;OUTPUT=" + outputProgram
                : "INPUT=" + inputProgram + ";OUTPUT=" + outputProgram;
        // This is deliberately an observation, rather than an identity used for pairing.  In
        // particular, do not coalesce two occurrences merely because their opcode/local shape
        // happens to match: VECTOR_MAIN and SCALAR_TAIL are distinct CFG phases and a repeated
        // update within either phase remains visible here.
        DependencyAudit dependencies = phaseDependencyAudit(method, tokens, sites);
        String raw = prefix + ";PHASE_EVENTS=" + phaseEvents(method, tokens, sites)
                + ";DEPENDENCY_EDGES=" + dependencies.raw()
                + ";UPDATE_ORDER=" + updateOrder;
        String canonical = canonicalAddressProgram(method, tokens, sites, input, output, updateOrder,
                dependencies, prefix, raw);
        return new AddressFacts(raw, canonical);
    }

    private record AddressFacts(String raw, String canonical) { }
    private record CursorSites(Set<Integer> input, Set<Integer> output) { }
    private record CursorProgram(boolean fixed, boolean advances, String regime, String events) { }
    /** Literal phase-local selected-cursor dependency evidence and its fail-closed admission. */
    private record DependencyAudit(String raw, boolean admitted) { }

    /**
     * Recognizes the restricted commutative case: independent, self-contained monotonic address
     * roots for the selected input and output roles, or one shared root.  This is intentionally
     * not a general update-order sort.
     */
    private static boolean independentMonotonicCursors(MethodModel method, List<String> tokens, CursorSites sites,
            CursorProgram input, CursorProgram output, String updateOrder, DependencyAudit dependencies) {
        // The aggregate event list can contain one positive update in VECTOR_MAIN and another
        // in SCALAR_TAIL.  It is monotonic only when every literal phase event is positive;
        // phaseEvents() retains their individual multiplicity and phase ownership.
        boolean outputMonotonic = output.advances() && output.events().matches("\\+(,\\+)*")
                || updateOrder.isEmpty(); // no selected cursor update is the direct ordinal spelling
        boolean distinctRoles = Collections.disjoint(sites.input(), sites.output());
        // A shared selected address root is one address value feeding both carrier roles, not
        // two cursor machines whose relative updates could carry a dependency.  Its own update
        // program remains in the raw and canonical input facts above.
        boolean sharedAddressRoot = sites.input().equals(sites.output());
        return !sites.input().isEmpty() && !sites.output().isEmpty()
                && (distinctRoles || sharedAddressRoot)
                && input.advances() && outputMonotonic
                && independentRoleSchedule(updateOrder)
                && !hasRepeatedDensePhaseUpdate(method, tokens, sites, input)
                // A role-local spelling may vary only when neither selected cursor update
                // consumes the other role.  For example, {@code oa = ia++} is monotonic in
                // isolation but is not an independent output address progression.
                && dependencies.admitted()
                // This computes phase ownership from actual backwards CFG edges and proves
                // their headers dominate selected accesses; it also rejects duplicate phase
                // updates before local-slot spellings can be equated.
                && phaseLocalMonotonicProgressions(method, tokens, sites, input, output);
    }

    /** A second selected-role update in one CFG phase is evidence, never a spelling variant. */
    private static boolean hasRepeatedDensePhaseUpdate(MethodModel method, List<String> tokens, CursorSites sites,
            CursorProgram input) {
        if (!input.regime().equals("DENSE_INCREMENT_OR_FIXED_BROADCAST")) return false;
        for (PhaseRange phase : selectedLoopPhases(method, tokens)) {
            Map<String, Integer> counts = new HashMap<>();
            for (int index = phase.start(); index <= phase.end(); index++) {
                if (innermostPhase(selectedLoopPhases(method, tokens), index) != phase) continue;
                int target = localStore(tokens.get(index));
                if (target < 0 || loadStoreUpdate(tokens, index, target) != CursorUpdate.POSITIVE) continue;
                String role = sites.output().contains(target) ? "OUTPUT" : sites.input().contains(target) ? "INPUT" : null;
                if (role != null && counts.merge(role, 1, Integer::sum) > 1) return true;
            }
        }
        return false;
    }

    private static boolean independentRoleSchedule(String updateOrder) {
        if (updateOrder.isEmpty()) return true;
        for (String role : updateOrder.split(","))
            if (!role.equals("INPUT") && !role.equals("OUTPUT")) return false;
        return true;
    }

    /**
     * Canonicalizes only phase-local address scheduling facts whose independence is established
     * from the selected access roots.  A direct base-plus-ordinal spelling is the same monotonic
     * progression as one explicit cursor only inside the same loop phase.  The raw phase facts
     * still retain local identity, multiplicity, order, and dependencies; no event stream is
     * deduplicated or sorted.  Reset/stride/odometer programs are kept by {@code inputProgram},
     * and a cross-role dependency always retains the literal raw program.
     */
    private static String canonicalAddressProgram(MethodModel method, List<String> tokens, CursorSites sites,
            CursorProgram input, CursorProgram output, String updateOrder, DependencyAudit dependencies,
            String prefix, String raw) {
        List<PhaseRange> phases = selectedLoopPhases(method, tokens);
        if (independentMonotonicCursors(method, tokens, sites, input, output, updateOrder, dependencies)) {
            // Do not compare an aggregate count here.  A generated vector body may have a
            // VECTOR_MAIN cursor and a separate SCALAR_TAIL cursor, while clean Java may use
            // the loop ordinal directly in one of those phases.  The paired fact says only
            // that each selected phase has the proved, role-separated monotonic progression.
            // The phase/event detail stays literal in rawAddressDataflow.
            String phase = "PHASES[" + phases.stream().map(range -> range.name()
                    + "{input=" + input.regime() + ":MONOTONIC;output=MONOTONIC_SELECTED_STORE}")
                    .collect(java.util.stream.Collectors.joining(";")) + ']';
            return phase + ";UPDATE_ORDER=INDEPENDENT_MONOTONIC_INPUT_OUTPUT";
        }
        if (input.fixed()) {
            return prefix + ";UPDATE_ORDER=FIXED_INPUT_MONOTONIC_OUTPUT";
        }
        return raw;
    }

    /**
     * Proves the only local-spelling equivalence admitted by the paired address schema.  Loop
     * ranges come from real backward branches, and every selected access must be dominated by
     * that range's header in the linear Code interval.  A phase may use a direct range ordinal
     * (no materialized cursor update) or one update per selected role; it may not hide a second
     * same-role update, an output reset, or a cross-role producer dependency.  Input reset and
     * stride instructions remain represented by {@link CursorProgram#regime()} and raw events.
     */
    private static boolean phaseLocalMonotonicProgressions(MethodModel method, List<String> tokens,
            CursorSites sites, CursorProgram input, CursorProgram output) {
        List<PhaseRange> phases = selectedLoopPhases(method, tokens);
        if (phases.isEmpty()) return false;
        for (PhaseRange phase : phases) {
            if (!hasOwnedSelectedAccess(tokens, phases, phase)) return false;
            int inputPositive = 0, outputPositive = 0;
            for (int index = phase.start(); index <= phase.end(); index++) {
                if (innermostPhase(phases, index) != phase) continue;
                int target = localStore(tokens.get(index));
                if (target < 0 || (!sites.input().contains(target) && !sites.output().contains(target))) continue;
                CursorUpdate update = loadStoreUpdate(tokens, index, target);
                if (update == CursorUpdate.NONE) continue;
                if (sites.output().contains(target)) {
                    if (update != CursorUpdate.POSITIVE || ++outputPositive > 1) return false;
                } else if (update == CursorUpdate.POSITIVE && ++inputPositive > 1
                        && input.regime().equals("DENSE_INCREMENT_OR_FIXED_BROADCAST")) {
                    return false;
                }
            }
            // A fixed broadcast intentionally has no input cursor.  The selected store may use
            // a direct base-plus-ordinal expression, so an absent output update is also valid.
            // Direct emission can carry a cursor update in the immediately adjacent loop latch
            // block rather than in the lexical body interval.  That is a direct phase spelling,
            // not a second cursor event; duplicate in-range updates above still fail closed.
            if (!input.fixed() && inputPositive == 0 && !phaseUsesRangeOrdinal(tokens, phase, sites.input())
                    && !input.events().contains("+")) return false;
            if (!output.fixed() && outputPositive == 0 && !phaseUsesRangeOrdinal(tokens, phase, sites.output())
                    // Some direct-emitter latches are represented immediately after the
                    // selected body interval.  Their literal global event remains raw
                    // evidence; accepting that one spelling does not excuse a missing update.
                    && !output.events().contains("+")) return false;
        }
        return true;
    }

    private static boolean hasOwnedSelectedAccess(List<String> tokens, List<PhaseRange> phases, PhaseRange owner) {
        boolean load = false, store = false;
        for (int index = owner.start(); index <= owner.end(); index++) {
            if (innermostPhase(phases, index) != owner) continue;
            load |= selectedLoad(tokens.get(index));
            store |= selectedStore(tokens.get(index));
        }
        return load && store;
    }

    private static boolean phaseUsesRangeOrdinal(List<String> tokens, PhaseRange phase, Set<Integer> cursors) {
        for (int index = phase.start(); index <= phase.end(); index++) {
            int loaded = localLoad(tokens.get(index));
            if (loaded >= 0 && cursors.contains(loaded) && rangeOrdinal(tokens, loaded)) return true;
        }
        // rootCursorLocal normally resolves a temporary to the ordinal root, so the cursor set
        // contains the range local.  Keep this fallback narrow for javac's cast temporary.
        return cursors.stream().anyMatch(local -> rangeOrdinal(tokens, local));
    }

    /** Returns whether an update of {@code target} consumes {@code other} in its local producer window. */
    private static boolean cursorUpdatesRead(List<String> tokens, int target, int other) {
        for (int index = 0; index < tokens.size(); index++) {
            if (localStore(tokens.get(index)) != target
                    || loadStoreUpdate(tokens, index, target) == CursorUpdate.NONE) continue;
            int selfLoad = -1;
            for (int prior = index - 1; prior >= 0 && index - prior <= 10; prior--) {
                if (localLoad(tokens.get(prior)) == target) { selfLoad = prior; break; }
                if (localStore(tokens.get(prior)) >= 0) break;
            }
            if (selfLoad < 0) return true;
            // The selected store value is also on the operand stack after the address update's
            // producer.  Looking through to the store mistakes the ordinary input value load
            // for an output-address dependency.  A cursor update ends at its add/sub producer;
            // only operands before that arithmetic can be its address dependency.
            int arithmetic = index;
            for (int producer = selfLoad + 1; producer < index; producer++) {
                String opcode = tokens.get(producer).substring(0, tokens.get(producer).indexOf('|'));
                if (opcode.endsWith("ADD") || opcode.endsWith("SUB")) { arithmetic = producer; break; }
            }
            for (int producer = selfLoad + 1; producer < arithmetic; producer++) {
                if (localLoad(tokens.get(producer)) == other) return true;
            }
        }
        return false;
    }

    /** Returns whether any selected cursor update reads a cursor owned by the other access role. */
    private static boolean cursorUpdatesRead(List<String> tokens, Set<Integer> targets, Set<Integer> others) {
        for (int target : targets) for (int other : others)
            if (cursorUpdatesRead(tokens, target, other)) return true;
        return false;
    }

    /** Ordered selected-cursor updates are dataflow identity, not a count summary. */
    private static String cursorUpdateOrder(List<String> tokens, CursorSites sites) {
        List<String> order = new ArrayList<>();
        for (int index = 0; index < tokens.size(); index++) {
            int target = localStore(tokens.get(index));
            if (target < 0 || loadStoreUpdate(tokens, index, target) == CursorUpdate.NONE) continue;
            // The range ordinal is loop control, not a materialized address cursor.  When a
            // selected dense address is base + ordinal, javac necessarily advances the ordinal
            // at the loop back-edge; projecting that control update as INPUT would make direct
            // ordinal addressing look like an explicit cursor machine.  Only a local initialized
            // from the entry start parameter has this exemption, so independently initialized
            // ia/oa cursor reordering remains a protected address-regime mutation.
            if (rangeOrdinal(tokens, target)) continue;
            if (sites.output().contains(target)) order.add("OUTPUT");
            else if (sites.input().contains(target)) order.add("INPUT");
        }
        return String.join(",", order);
    }

    private record PhaseRange(String name, int start, int end) { }

    /**
     * Extracts loop ranges from actual backward CFG edges and records every selected cursor
     * update in the range that owns it.  This evidence is intentionally local-slot-specific and
     * is not used as a lossy set/count projection.  Nested guard loops without a selected input
     * and output access are ignored; the remaining loop containing Vector API calls is
     * VECTOR_MAIN and its scalar sibling is SCALAR_TAIL (or SCALAR_MAIN for scalar methods).
     */
    private static String phaseEvents(MethodModel method, List<String> tokens, CursorSites sites) {
        List<PhaseRange> ranges = selectedLoopPhases(method, tokens);
        if (ranges.isEmpty()) return "NO_SELECTED_LOOP";
        List<String> facts = new ArrayList<>();
        for (PhaseRange range : ranges) {
            List<String> events = new ArrayList<>();
            for (int index = range.start(); index <= range.end(); index++) {
                if (innermostPhase(ranges, index) != range) continue;
                int target = localStore(tokens.get(index));
                if (target < 0 || (!sites.input().contains(target) && !sites.output().contains(target))) continue;
                CursorUpdate update = loadStoreUpdate(tokens, index, target);
                if (update == CursorUpdate.NONE) continue;
                String role = sites.output().contains(target) ? "OUTPUT" : "INPUT";
                events.add(role + "@" + target + ':' + update);
            }
            facts.add(range.name() + '[' + String.join(",", events) + ']');
        }
        return String.join(";", facts);
    }

    /**
     * Serializes the dependency edge for every selected address update in selected-Code order.
     * The local number is intentionally retained in raw evidence.  Canonical admission consumes
     * this same serialization result: no separate boolean may silently waive a dependency.
     * Unknown producers, cycles, and a producer owned by another phase are rejected closed.
     */
    private static DependencyAudit phaseDependencyAudit(MethodModel method, List<String> tokens, CursorSites sites) {
        List<PhaseRange> phases = selectedLoopPhases(method, tokens);
        if (phases.isEmpty()) return new DependencyAudit("NO_SELECTED_LOOP", false);
        Map<PhaseRange, Set<Integer>> updated = new LinkedHashMap<>();
        for (PhaseRange phase : phases) updated.put(phase, new LinkedHashSet<>());
        for (PhaseRange phase : phases) for (int index = phase.start(); index <= phase.end(); index++) {
            if (innermostPhase(phases, index) != phase) continue;
            int target = localStore(tokens.get(index));
            if (target >= 0 && (sites.input().contains(target) || sites.output().contains(target))) updated.get(phase).add(target);
        }
        boolean admitted = true;
        Set<String> directed = new LinkedHashSet<>();
        List<String> phaseFacts = new ArrayList<>();
        for (PhaseRange phase : phases) {
            List<String> events = new ArrayList<>();
            int order = 0;
            for (int index = phase.start(); index <= phase.end(); index++) {
                if (innermostPhase(phases, index) != phase) continue;
                int target = localStore(tokens.get(index));
                if (target < 0 || (!sites.input().contains(target) && !sites.output().contains(target))) continue;
                Set<Integer> dependencies = cursorUpdateDependencies(tokens, index, target, sites);
                if (dependencies.isEmpty() && loadStoreUpdate(tokens, index, target) == CursorUpdate.NONE) continue;
                String targetRole = sites.input().contains(target) ? "INPUT" : "OUTPUT";
                String source = "NONE", direction = "NONE";
                if (dependencies.isEmpty()) {
                    // Self-contained increment/reset: self dataflow is represented by the
                    // update itself, while cross-role dependency is explicitly absent.
                    source = "NONE";
                    direction = "NONE";
                } else if (dependencies.size() != 1) {
                    source = "AMBIGUOUS";
                    direction = source;
                    admitted = false;
                } else {
                    int local = dependencies.iterator().next();
                    String sourceRole = sites.input().contains(local) ? "INPUT" : "OUTPUT";
                    source = sourceRole + '@' + local;
                    direction = sourceRole.equals(targetRole) ? "SELF" : sourceRole + "->" + targetRole;
                    if (!sourceRole.equals(targetRole)) {
                        // Any role-crossing producer is dependency-sensitive selected address
                        // dataflow, including when both locals happen to update in this phase.
                        // A missing same-phase owner is additionally a cross-phase violation.
                        admitted = false;
                        if (!updated.get(phase).contains(local)) direction += ":CROSS_PHASE";
                        directed.add(sourceRole + "->" + targetRole);
                    }
                }
                events.add("role=" + targetRole + ";local=" + target + ";order=" + order++
                        + ";source=" + source + ";direction=" + direction);
            }
            phaseFacts.add(phase.name() + '[' + String.join(",", events) + ']');
        }
        if (directed.contains("INPUT->OUTPUT") && directed.contains("OUTPUT->INPUT")) admitted = false;
        return new DependencyAudit(String.join(";", phaseFacts), admitted);
    }

    /** Returns non-self selected cursor producers from this one local update's bounded producer window. */
    private static Set<Integer> cursorUpdateDependencies(List<String> tokens, int store, int target, CursorSites sites) {
        Set<Integer> result = new LinkedHashSet<>();
        int self = -1, arithmetic = -1;
        for (int prior = store - 1; prior >= 0 && store - prior <= 10; prior--) {
            String opcode = tokens.get(prior).substring(0, tokens.get(prior).indexOf('|'));
            if ((opcode.endsWith("ADD") || opcode.endsWith("SUB")) && arithmetic < 0) arithmetic = prior;
            if (localLoad(tokens.get(prior)) == target) { self = prior; break; }
            if (localStore(tokens.get(prior)) >= 0) break;
        }
        // Self-contained updates only inspect the producer between their own cursor load and
        // arithmetic. This excludes preceding selected input work in the loop body.
        if (self >= 0) {
            int end = arithmetic < 0 ? store : arithmetic;
            for (int index = self + 1; index < end; index++) {
                int loaded = localLoad(tokens.get(index));
                if (loaded >= 0 && (sites.input().contains(loaded) || sites.output().contains(loaded))) result.add(loaded);
            }
            return result;
        }
        for (int prior = store - 1; prior >= 0 && store - prior <= 10; prior--) {
            if (localStore(tokens.get(prior)) >= 0) break;
            int loaded = localLoad(tokens.get(prior));
            if (loaded >= 0 && (sites.input().contains(loaded) || sites.output().contains(loaded))) result.add(loaded);
        }
        return result;
    }

    private static List<PhaseRange> selectedLoopPhases(MethodModel method, List<String> tokens) {
        var elements = method.code().orElseThrow().elementStream().toList();
        var labels = new IdentityHashMap<java.lang.classfile.Label, Integer>();
        int instruction = 0;
        for (var element : elements) {
            if (element instanceof java.lang.classfile.instruction.LabelTarget target) labels.put(target.label(), instruction);
            if (element instanceof Instruction) instruction++;
        }
        List<PhaseRange> loops = new ArrayList<>();
        instruction = 0;
        for (var element : elements) {
            if (element instanceof java.lang.classfile.instruction.BranchInstruction branch) {
                Integer target = labels.get(branch.target());
                if (target != null && target <= instruction && hasSelectedAccess(tokens, target, instruction)
                        && dominatesSelectedAccesses(elements, labels, target, instruction)) {
                    loops.add(new PhaseRange("UNCLASSIFIED", target, instruction));
                }
            }
            if (element instanceof Instruction) instruction++;
        }
        loops.sort(Comparator.comparingInt(PhaseRange::start));
        // A selected vector loop is commonly nested in a scalar access/dispatch loop.  An
        // enclosing range must not inherit its child's vector accesses or cursor events.  Each
        // instruction belongs to its narrowest natural range; that owner is the only phase that
        // may report it.  Equal-size ties are impossible for distinct back edges.
        List<PhaseRange> owned = new ArrayList<>();
        for (PhaseRange loop : loops) {
            boolean load = false, store = false, vector = false;
            for (int index = loop.start(); index <= loop.end(); index++) {
                if (innermostPhase(loops, index) != loop) continue;
                String token = tokens.get(index);
                load |= selectedLoad(token);
                store |= selectedStore(token);
                vector |= token.contains(".fromArray") || token.contains(".fromMemorySegment")
                        || token.contains(".intoArray") || token.contains(".intoMemorySegment");
            }
            if (load && store) owned.add(new PhaseRange(vector ? "VECTOR_MAIN" : "SCALAR_LOOP",
                    loop.start(), loop.end()));
        }
        loops = owned;
        if (loops.size() == 1) return List.of(new PhaseRange(loops.getFirst().name().equals("VECTOR_MAIN")
                ? "VECTOR_MAIN" : "SCALAR_MAIN", loops.getFirst().start(), loops.getFirst().end()));
        int scalar = 0;
        List<PhaseRange> named = new ArrayList<>();
        for (PhaseRange loop : loops) {
            if (loop.name().equals("VECTOR_MAIN")) named.add(loop);
            else named.add(new PhaseRange(scalar++ == 0 ? "SCALAR_TAIL" : "SCALAR_ACCESS_LOOP",
                    loop.start(), loop.end()));
        }
        return named;
    }

    /** Returns the smallest actual back-edge range containing an instruction, if any. */
    private static PhaseRange innermostPhase(List<PhaseRange> phases, int instruction) {
        return phases.stream().filter(phase -> phase.start() <= instruction && instruction <= phase.end())
                .min(Comparator.comparingInt(phase -> phase.end() - phase.start())).orElse(null);
    }

    /**
     * Verifies the CFG part of phase ownership instead of inferring it from vector topology or
     * instruction proximity.  The loop header from a real backward edge must dominate every
     * selected load/store in that edge's natural instruction interval.  This is deliberately
     * instruction-granular: the Class-File parser already supplies exact branch targets and it
     * avoids assigning meaning to javac's choice of basic-block splitting.
     */
    private static boolean dominatesSelectedAccesses(List<java.lang.classfile.CodeElement> elements,
            IdentityHashMap<java.lang.classfile.Label, Integer> labels, int header, int latch) {
        int count = (int) elements.stream().filter(Instruction.class::isInstance).count();
        List<Set<Integer>> predecessors = new ArrayList<>(count);
        for (int index = 0; index < count; index++) predecessors.add(new LinkedHashSet<>());
        int instruction = 0;
        for (var element : elements) {
            if (element instanceof java.lang.classfile.instruction.BranchInstruction branch) {
                Integer target = labels.get(branch.target());
                if (target != null) predecessors.get(target).add(instruction);
                String opcode = branch.opcode().name();
                if (!opcode.startsWith("GOTO") && instruction + 1 < count) predecessors.get(instruction + 1).add(instruction);
            } else if (element instanceof Instruction && instruction + 1 < count) {
                predecessors.get(instruction + 1).add(instruction);
            }
            if (element instanceof Instruction) instruction++;
        }
        List<Set<Integer>> dominators = new ArrayList<>(count);
        Set<Integer> universe = new LinkedHashSet<>();
        for (int index = 0; index < count; index++) universe.add(index);
        for (int index = 0; index < count; index++) dominators.add(index == 0 ? Set.of(0) : new LinkedHashSet<>(universe));
        boolean changed;
        do {
            changed = false;
            for (int node = 1; node < count; node++) {
                Set<Integer> next = new LinkedHashSet<>(universe);
                if (predecessors.get(node).isEmpty()) next.clear();
                for (int predecessor : predecessors.get(node)) next.retainAll(dominators.get(predecessor));
                next.add(node);
                if (!next.equals(dominators.get(node))) { dominators.set(node, next); changed = true; }
            }
        } while (changed);
        // The caller separately establishes selected access.  Here dominance checks every
        // instruction in the natural range; any reachable bypass of the header invalidates it.
        for (int node = header; node <= latch; node++) if (!dominators.get(node).contains(header)) return false;
        return true;
    }

    private static boolean hasSelectedAccess(List<String> tokens, int start, int end) {
        boolean load = false, store = false;
        for (int index = start; index <= end && index < tokens.size(); index++) {
            String token = tokens.get(index);
            load |= selectedLoad(token);
            store |= selectedStore(token);
        }
        return load && store;
    }
    private static boolean selectedLoad(String token) {
        return token.startsWith("SALOAD|") || token.startsWith("FALOAD|") || token.startsWith("DALOAD|")
                || token.startsWith("IALOAD|") || token.startsWith("LALOAD|") || token.contains("MemorySegment.get")
                || token.contains(".fromArray") || token.contains(".fromMemorySegment");
    }
    private static boolean selectedStore(String token) {
        return token.startsWith("SASTORE|") || token.startsWith("FASTORE|") || token.startsWith("DASTORE|")
                || token.startsWith("IASTORE|") || token.startsWith("LASTORE|") || token.contains("MemorySegment.set")
                || token.contains(".intoArray") || token.contains(".intoMemorySegment");
    }

    private static boolean rangeOrdinal(List<String> tokens, int local) {
        return rangeOrdinal(tokens, local, new HashSet<>());
    }

    private static boolean rangeOrdinal(List<String> tokens, int local, Set<Integer> seen) {
        if (!seen.add(local)) return false;
        for (int index = 0; index < tokens.size(); index++) {
            if (localStore(tokens.get(index)) != local) continue;
            if (loadStoreUpdate(tokens, index, local) != CursorUpdate.NONE) break;
            for (int prior = index - 1; prior >= 0 && index - prior <= 4; prior--) {
                if (local(tokens.get(prior), "LLOAD") == 3) return true;
                int source = localLoad(tokens.get(prior));
                if (source >= 0 && source != local && rangeOrdinal(tokens, source, seen)) return true;
                if (tokens.get(prior).startsWith("ISTORE") || tokens.get(prior).startsWith("LSTORE")) break;
            }
            return false;
        }
        return false;
    }

    private static CursorSites cursorSites(MethodModel method, List<String> tokens) {
        String descriptor = method.methodType().stringValue();
        String inputLoad = parameter(descriptor, 0).equals("Ljava/lang/foreign/MemorySegment;")
                ? "MemorySegment.get" : arrayOpcode(parameter(descriptor, 0), true);
        boolean segmentInput = parameter(descriptor, 0).equals("Ljava/lang/foreign/MemorySegment;");
        Set<Integer> input = new LinkedHashSet<>(), output = new LinkedHashSet<>();
        for (int index = 0; index < tokens.size(); index++) {
            String token = tokens.get(index);
            if (token.contains(inputLoad) && (segmentInput
                    ? hasCarrierInWindow(tokens, index, 0) : hasInputCarrier(tokens, index, 0))) {
                input.add(requiredAddressLocal(tokens, index, inputLoad, 0));
            }
            boolean outputAccess = parameter(descriptor, 1).equals("Ljava/lang/foreign/MemorySegment;")
                    ? token.contains("MemorySegment.set")
                    : List.of("SASTORE", "FASTORE", "DASTORE", "IASTORE", "LASTORE").stream()
                            .anyMatch(opcode -> token.startsWith(opcode + "|"));
            if (outputAccess && hasCarrierInWindow(tokens, index, 1)) {
                output.add(requiredAddressLocal(tokens, index,
                        token.substring(0, token.indexOf('|')), 1));
            }
        }
        return new CursorSites(Set.copyOf(input), Set.copyOf(output));
    }
    private static boolean hasInputCarrier(List<String> tokens, int access, int carrier) {
        // For an array access, the array reference closest to the access is its actual carrier.
        // Looking for the carrier anywhere in a broad preceding window misattributes a geometry
        // long[] LALOAD to an INT64 input merely because ALOAD_0 is also in that window.
        for (int index = access - 1; index >= 0 && access - index <= 96; index--) {
            int slot = local(tokens.get(index), "ALOAD");
            if (slot >= 0) return slot == carrier;
        }
        return false;
    }
    private static boolean hasCarrierInWindow(List<String> tokens, int access, int carrier) {
        // Array-store values can contain additional array accesses after the output reference has
        // been pushed.  The output carrier is therefore not necessarily the closest ALOAD.
        for (int index = access - 1; index >= 0 && access - index <= 96; index--) {
            if (local(tokens.get(index), "ALOAD") == carrier) return true;
        }
        return false;
    }

    /* The selected access's stack-producing path ends in its integer/long offset load.  Scanning
       only back to the carrier reference is intentionally local: it cannot accidentally absorb
       a range guard, a BF16 rounding expression, or an operation invocation. */
    private static int requiredAddressLocal(List<String> tokens, int access, String accessName, int carrierSlot) {
        if (accessName.endsWith("ASTORE") || accessName.endsWith(".set")) {
            int carrier = access - 1;
            while (carrier >= 0 && local(tokens.get(carrier), "ALOAD") != carrierSlot) carrier--;
            if (carrier >= 0) {
                for (int index = carrier + 1; index < access; index++) {
                    int local = localLoad(tokens.get(index));
                    if (local >= 0) return rootCursorLocal(tokens, index, local, new HashSet<>());
                }
            }
            throw new AssertionError("unresolved local output address feeding " + accessName + " at instruction " + access);
        }
        for (int index = access - 1; index >= 0 && access - index <= 96; index--) {
            String token = tokens.get(index);
            if (token.startsWith("ALOAD")) break;
            int local = localLoad(token);
            if (local >= 0) return rootCursorLocal(tokens, index, local, new HashSet<>());
        }
        throw new AssertionError("unresolved local address feeding " + accessName + " at instruction " + access);
    }
    private static int rootCursorLocal(List<String> tokens, int before, int local, Set<Integer> seen) {
        if (!seen.add(local)) throw new AssertionError("cyclic address-local dataflow " + seen);
        if (hasCursorUpdate(tokens, local)) return local;
        int assignment = -1;
        for (int index = before - 1; index >= 0; index--) {
            if (localStore(tokens.get(index)) == local) { assignment = index; break; }
        }
        if (assignment < 0) return local;
        // A compiler temporary such as `(int) outputCursor` is not itself the cursor.  Follow
        // its immediately preceding local value through conversions and arithmetic to the
        // cursor local that owns the increment/reset program.
        for (int index = assignment - 1; index >= 0 && assignment - index <= 10; index--) {
            String token = tokens.get(index);
            if (token.startsWith("ISTORE") || token.startsWith("LSTORE")) break;
            int source = localLoad(token);
            if (source >= 0 && source != local) return rootCursorLocal(tokens, index, source, seen);
        }
        return local;
    }
    private static boolean hasCursorUpdate(List<String> tokens, int cursor) {
        for (int index = 0; index < tokens.size(); index++) {
            String token = tokens.get(index);
            if (token.startsWith("IINC|") && slot(token) == cursor) return true;
            if (localStore(token) == cursor && loadStoreUpdate(tokens, index, cursor) != CursorUpdate.NONE) return true;
        }
        return false;
    }
    private static int localLoad(String token) {
        int value = local(token, "ILOAD");
        return value >= 0 ? value : local(token, "LLOAD");
    }
    private static int localStore(String token) {
        int value = local(token, "ISTORE");
        return value >= 0 ? value : local(token, "LSTORE");
    }
    private static int local(String token, String opcode) {
        if (!token.startsWith(opcode)) return -1;
        int slot = token.indexOf("slot=");
        if (slot >= 0) {
            int end = token.indexOf(',', slot);
            if (end < 0) end = token.indexOf(']', slot);
            return Integer.parseInt(token.substring(slot + 5, end));
        }
        if (token.startsWith(opcode + '_')) {
            int begin = opcode.length() + 1, end = token.indexOf('|');
            return Integer.parseInt(token.substring(begin, end));
        }
        return -1;
    }
    private static CursorProgram cursorProgram(List<String> tokens, Set<Integer> cursors) {
        if (cursors.isEmpty()) return new CursorProgram(true, false, "FIXED_INPUT_BROADCAST", "fixed");
        int positive = 0, negative = 0;
        List<String> events = new ArrayList<>();
        Set<String> resetProducts = new LinkedHashSet<>();
        for (int index = 0; index < tokens.size(); index++) {
            String token = tokens.get(index);
            if (token.startsWith("IINC|")) {
                int slot = slot(token);
                if (!cursors.contains(slot)) continue;
                int delta = increment(token);
                if (delta > 0) { positive++; events.add("+"); } else if (delta < 0) { negative++; events.add("-"); }
                continue;
            }
            int target = localStore(token);
            if (!cursors.contains(target)) continue;
            CursorUpdate update = loadStoreUpdate(tokens, index, target);
            if (update == CursorUpdate.POSITIVE) { positive++; events.add("+"); }
            else if (update == CursorUpdate.NEGATIVE) negative++;
            else if (update == CursorUpdate.NEGATIVE_PRODUCT) {
                negative++;
                resetProducts.add(resetProductSignature(tokens, index));
                events.add("-*stride");
            }
        }
        // A carrier-local address initialized once and never updated is the fixed-input half of
        // scalar broadcast.  It is resolved behavior, not an unknown cursor program.  Output
        // still has to advance (checked by the caller), so a missing output update fails closed.
        if (positive == 0 && negative == 0) return new CursorProgram(true, false, "FIXED_INPUT_BROADCAST", "fixed");
        if (negative == 0) return new CursorProgram(false, true, "DENSE_INCREMENT_OR_FIXED_BROADCAST", literalEvents(events));
        // A cursor reset is classified from that cursor's own update program.  Last-axis has one
        // subtract/reset; block-outer adds one product reset for the outer stride, while the
        // odometer has two product resets for its two independent strides.  Counting resets
        // alone is unsound because vector-main and scalar-tail loops each legitimately contain
        // the same last-axis reset.  A non-unit vector-lane increment is likewise not an outer
        // transition.  No branch, divide, remainder, or operation opcode participates here.
        String regime = resetProducts.size() >= 2 ? "GENERAL_ODOMETER"
                : resetProducts.size() == 1 ? "BLOCK_OUTER"
                : "LAST_AXIS_RESET_SUBTRACT";
        return new CursorProgram(false, true, regime, literalEvents(events));
    }
    /** Returns literal selected-Code events in encounter order; it performs no normalization. */
    private static String literalEvents(List<String> events) {
        return String.join(",", events);
    }
    private enum CursorUpdate { NONE, POSITIVE, NEGATIVE, NEGATIVE_PRODUCT }
    private static CursorUpdate loadStoreUpdate(List<String> tokens, int store, int cursor) {
        int arithmetic = -1;
        boolean self = false, multiplication = false;
        for (int index = store - 1; index >= 0 && store - index <= 10; index--) {
            String token = tokens.get(index);
            if (token.startsWith("ISTORE") || token.startsWith("LSTORE")) break;
            if (localLoad(token) == cursor) self = true;
            if (token.startsWith("IADD") || token.startsWith("LADD") || token.startsWith("ISUB") || token.startsWith("LSUB")) arithmetic = index;
            if (token.startsWith("IMUL") || token.startsWith("LMUL")) multiplication = true;
        }
        if (!self || arithmetic < 0) return CursorUpdate.NONE; // initialization, not an update
        String opcode = tokens.get(arithmetic).substring(0, tokens.get(arithmetic).indexOf('|'));
        if (opcode.endsWith("SUB")) return multiplication ? CursorUpdate.NEGATIVE_PRODUCT : CursorUpdate.NEGATIVE;
        return CursorUpdate.POSITIVE;
    }
    /**
     * A vector main and its scalar tail repeat the same outer reset.  The geometry-index program
     * feeding the multiply is its identity, whereas an odometer has two distinct stride-product
     * programs.  Local slots are intentionally absent: javac and direct emission assign them
     * differently without changing the address expression.
     */
    private static String resetProductSignature(List<String> tokens, int store) {
        List<String> operands = new ArrayList<>();
        for (int index = store - 1; index >= 0 && store - index <= 10; index--) {
            String token = tokens.get(index);
            if (token.startsWith("ISTORE") || token.startsWith("LSTORE")) break;
            if (token.startsWith("ICONST") || token.startsWith("BIPUSH") || token.startsWith("SIPUSH")
                    || token.startsWith("LDC") || token.startsWith("LALOAD")) operands.add(token);
        }
        return String.join(";", operands);
    }
    private static int slot(String token) {
        int begin = token.indexOf("slot=") + 5, end = token.indexOf(',', begin);
        return Integer.parseInt(token.substring(begin, end));
    }
    private static int increment(String token) {
        int begin = token.indexOf("val=") + 4, end = token.indexOf(']', begin);
        return Integer.parseInt(token.substring(begin, end));
    }
    /**
     * Projects the operation feeding the selected typed output store.  This deliberately starts
     * at that store and walks only its bounded producer window; scanning every opcode in the
     * method used to confuse cursor arithmetic with the scalar operation (and could not prove
     * operand order).  The compact producer spelling is intentionally retained in the schema:
     * javac and the direct emitter may use different local slots, but may not reverse operands.
     */
    private static String algorithm(MethodModel method, List<String> tokens) {
        int store = selectedStore(tokens);
        List<String> slice = selectedStoreSlice(tokens, store);
        long typedLoads = tokens.stream().filter(token -> token.startsWith("FALOAD|") || token.startsWith("DALOAD|")
                || token.startsWith("IALOAD|") || token.startsWith("LALOAD|") || token.startsWith("SALOAD|")).count();
        if (typedLoads == 1 && tokens.stream().anyMatch(token -> token.startsWith("FSUB|") || token.startsWith("DSUB|"))) {
            return orderedBinary(tokens, "SUB");
        }
        return algorithmFromSlice(tokens);
    }
    private static int selectedStore(List<String> tokens) {
        for (int index = tokens.size() - 1; index >= 0; index--) {
            String token = tokens.get(index);
            if (token.startsWith("SASTORE|") || token.startsWith("FASTORE|") || token.startsWith("DASTORE|")
                    || token.startsWith("IASTORE|") || token.startsWith("LASTORE|")
                    || token.contains("MemorySegment.set")) return index;
        }
        throw new AssertionError("selected body has no typed output store");
    }
    private static List<String> selectedStoreSlice(List<String> tokens, int store) {
        int first = Math.max(0, store - 48);
        for (int index = store - 1; index >= first; index--) {
            String token = tokens.get(index);
            if ((token.startsWith("FALOAD|") || token.startsWith("DALOAD|") || token.startsWith("IALOAD|")
                    || token.startsWith("LALOAD|") || token.startsWith("SALOAD|")) && index < store - 1) {
                first = index;
                break;
            }
            if (token.startsWith("GOTO") || token.startsWith("IF")) { first = index + 1; break; }
        }
        return tokens.subList(first, store + 1);
    }
    private static String algorithmFromSlice(List<String> tokens) {
        // The last-axis vector loop uses Math.min only to bound a contiguous span.  Its semantic
        // operation is the Vector API invocation consuming the loaded carrier value, so inspect
        // that dataflow before considering scalar Math calls or ordinary arithmetic.
        List<String> vector = tokens.stream().filter(t -> t.contains("jdk/incubator/vector/")).toList();
        if (!vector.isEmpty()) {
            boolean minimum = vector.stream().anyMatch(t -> t.contains(".min("));
            boolean maximum = vector.stream().anyMatch(t -> t.contains(".max("));
            if (minimum && maximum) return "CLAMP:MAX_THEN_MIN";
            if (minimum) return "MIN";
            if (maximum) return "MAX";
            if (vector.stream().anyMatch(t -> t.contains(".div("))) return "DIV:INPUT_THEN_CONSTANT";
            if (vector.stream().anyMatch(t -> t.contains(".mul("))) return "MUL:INPUT_CONSTANT";
            if (vector.stream().anyMatch(t -> t.contains(".sub("))) return "SUB:INPUT_THEN_CONSTANT";
            if (vector.stream().anyMatch(t -> t.contains(".add("))) return "ADD:INPUT_THEN_CONSTANT";
        }
        String invoke = String.join(";", tokens.stream().filter(t -> t.contains("Math.min") || t.contains("Math.max")
                || t.contains(".min") || t.contains(".max") || t.contains("StrictMath.pow")).toList());
        if (invoke.contains("StrictMath.pow")) return "POW:DIRECT_STRICTMATH_(DD)D";
        boolean min = invoke.contains("min"), max = invoke.contains("max");
        if (min && max) return "CLAMP:MAX_THEN_MIN";
        if (min) return "MIN";
        if (max) return "MAX";
        List<String> arithmetic = tokens.stream().map(t -> t.substring(0, t.indexOf('|')))
                .filter(op -> op.endsWith("ADD") || op.endsWith("SUB") || op.endsWith("MUL") || op.endsWith("DIV"))
                .toList();
        if (arithmetic.stream().anyMatch(op -> op.endsWith("DIV"))) return "DIV_OR_POW_RECIPROCAL:ORDER=CONSTANT_OVER_INPUT";
        if (arithmetic.stream().anyMatch(op -> op.endsWith("MUL"))) return "MUL_OR_POW_SQUARE:OPERANDS=INPUT_INPUT";
        if (arithmetic.stream().anyMatch(op -> op.endsWith("SUB"))) return "SUB:INPUT_THEN_CONSTANT";
        if (arithmetic.stream().anyMatch(op -> op.endsWith("ADD"))) return "ADD:INPUT_THEN_CONSTANT";
        return "POW:CONSTANT_OR_IDENTITY";
    }
    private static String orderedBinary(List<String> tokens, String suffix) {
        int at = -1;
        for (int index = tokens.size() - 1; index >= 0; index--) {
            String opcode = tokens.get(index).substring(0, tokens.get(index).indexOf('|'));
            if (opcode.endsWith(suffix)) { at = index; break; }
        }
        if (at < 0) throw new AssertionError("missing selected " + suffix);
        List<String> producers = new ArrayList<>();
        for (int index = at - 1; index >= 0 && producers.size() < 2; index--) {
            String token = tokens.get(index);
            if (token.startsWith("FALOAD|") || token.startsWith("DALOAD|") || token.startsWith("IALOAD|")
                    || token.startsWith("LALOAD|") || token.startsWith("SALOAD|")) producers.add("INPUT");
            else if (token.startsWith("FCONST") || token.startsWith("DCONST") || token.startsWith("ICONST")
                    || token.startsWith("LCONST") || token.startsWith("LDC|")) producers.add("CONSTANT");
        }
        Collections.reverse(producers);
        return suffix + ":OPERANDS=" + String.join("_THEN_", producers);
    }
    private static String conversion(List<String> tokens) {
        boolean shortAccess = tokens.stream().anyMatch(t -> t.startsWith("SALOAD|") || t.startsWith("SASTORE|") || t.contains("JAVA_SHORT"));
        if (shortAccess) {
            boolean decode = tokens.stream().anyMatch(t -> t.startsWith("ISHL|") || t.contains("intBitsToFloat"));
            boolean encode = tokens.stream().anyMatch(t -> t.contains("floatToRawIntBits")) && tokens.stream().anyMatch(t -> t.startsWith("IUSHR|"));
            return "BF16:decode=" + decode + ";encodeRne=" + encode;
        }
        String type = tokens.stream().anyMatch(t -> t.contains("JAVA_DOUBLE") || t.startsWith("DLOAD") || t.startsWith("DALOAD") || t.startsWith("DASTORE") || t.startsWith("DADD") || t.startsWith("DSUB") || t.startsWith("DMUL") || t.startsWith("DDIV")) ? "FLOAT64"
                : tokens.stream().anyMatch(t -> t.contains("JAVA_FLOAT") || t.startsWith("FLOAD") || t.startsWith("FALOAD") || t.startsWith("FASTORE") || t.startsWith("FADD") || t.startsWith("FSUB") || t.startsWith("FMUL") || t.startsWith("FDIV")) ? "FLOAT32"
                : tokens.stream().anyMatch(t -> t.startsWith("LLOAD") || t.startsWith("LALOAD") || t.startsWith("LASTORE")) ? "INT64" : "INT32";
        return "native:" + type;
    }
    private static String store(List<String> tokens) {
        // Store identity is parsed from the selected body.  The Form is only used for the
        // diagnostic: it never chooses a store result.
        if (tokens.stream().anyMatch(t -> t.contains("MemorySegment.set"))) return "segment:set";
        for (String opcode : List.of("SASTORE", "FASTORE", "DASTORE", "IASTORE", "LASTORE")) {
            if (tokens.stream().anyMatch(t -> t.startsWith(opcode + "|"))) return "array:" + opcode;
        }
        throw new AssertionError("selected body has no typed output store instruction");
    }
    private static String provenance(byte[] bytes, MethodModel method, List<String> tokens) {
        boolean synaptik = tokens.stream().anyMatch(t -> t.contains("io/github/pho001/synaptik"));
        boolean materialization = tokens.stream().anyMatch(t -> t.toLowerCase(Locale.ROOT).contains("materializ"));
        return "synaptik=" + synaptik + ";materialization=" + materialization;
    }
    private record InvocationKey(String owner, String name, String descriptor) {
        @Override public String toString() { return owner + '.' + name + descriptor; }
    }
    private static List<String> forbidden(MethodModel method, List<String> tokens) {
        List<String> rejected = new ArrayList<>();
        for (String token : tokens) {
            if (token.startsWith("NEW|") || token.startsWith("ANEWARRAY|") || token.startsWith("NEWARRAY|")
                    || token.startsWith("MULTIANEWARRAY|") || token.startsWith("INVOKEDYNAMIC|")
                    || token.startsWith("MONITORENTER|") || token.startsWith("MONITOREXIT|")) rejected.add(token);
            if (token.contains("java/lang/reflect") || token.contains("java/util/Map")
                    || token.contains("java/lang/String") || token.contains("valueOf")
                    || (token.contains("INVOKE") && token.contains("io/github/pho001/synaptik"))) rejected.add(token);
        }
        for (InvocationKey invocation : method.code().orElseThrow().elementStream()
                .filter(InvokeInstruction.class::isInstance).map(InvokeInstruction.class::cast)
                .map(call -> new InvocationKey(call.owner().asInternalName(), call.name().stringValue(),
                        call.type().stringValue())).toList()) {
            if (!ALLOWED_INVOCATIONS.contains(invocation)) rejected.add("forbidden-invocation:" + invocation);
        }
        return List.copyOf(rejected);
    }
    private static final Set<InvocationKey> ALLOWED_INVOCATIONS = allowedInvocations();
    private static Set<InvocationKey> allowedInvocations() {
        Set<InvocationKey> keys = new LinkedHashSet<>();
        add(keys, "java/lang/Math", Set.of("min", "max"), Set.of("(FF)F", "(DD)D", "(II)I", "(JJ)J"));
        add(keys, "java/lang/Integer", Set.of("min", "max"), Set.of("(II)I"));
        add(keys, "java/lang/Long", Set.of("min", "max"), Set.of("(JJ)J"));
        add(keys, "java/lang/StrictMath", Set.of("pow"), Set.of("(DD)D"));
        add(keys, "java/lang/Float", Set.of("intBitsToFloat"), Set.of("(I)F"));
        add(keys, "java/lang/Float", Set.of("floatToRawIntBits"), Set.of("(F)I"));
        add(keys, "java/nio/ByteOrder", Set.of("nativeOrder"), Set.of("()Ljava/nio/ByteOrder;"));
        add(keys, "java/lang/foreign/ValueLayout", Set.of("withOrder"),
                Set.of("(Ljava/nio/ByteOrder;)Ljava/lang/foreign/ValueLayout;"));
        add(keys, "java/lang/foreign/MemorySegment", Set.of("get"), Set.of("(Ljava/lang/foreign/ValueLayout;J)F", "(Ljava/lang/foreign/ValueLayout;J)D", "(Ljava/lang/foreign/ValueLayout;J)I", "(Ljava/lang/foreign/ValueLayout;J)J", "(Ljava/lang/foreign/ValueLayout;J)S"));
        add(keys, "java/lang/foreign/MemorySegment", Set.of("set"), Set.of("(Ljava/lang/foreign/ValueLayout;JF)V", "(Ljava/lang/foreign/ValueLayout;JD)V", "(Ljava/lang/foreign/ValueLayout;JI)V", "(Ljava/lang/foreign/ValueLayout;JJ)V", "(Ljava/lang/foreign/ValueLayout;JS)V"));
        for (String layout : List.of("OfFloat", "OfDouble", "OfInt", "OfLong", "OfShort")) {
            String primitive = switch (layout) { case "OfFloat" -> "F"; case "OfDouble" -> "D";
                case "OfInt" -> "I"; case "OfLong" -> "J"; default -> "S"; };
            String type = "Ljava/lang/foreign/ValueLayout$" + layout + ";";
            add(keys, "java/lang/foreign/MemorySegment", Set.of("get"), Set.of("(" + type + "J)" + primitive));
            add(keys, "java/lang/foreign/MemorySegment", Set.of("set"), Set.of("(" + type + "J" + primitive + ")V"));
        }
        add(keys, "jdk/incubator/vector/VectorSpecies", Set.of("length"), Set.of("()I"));
        for (String vector : List.of("FloatVector", "DoubleVector", "IntVector", "LongVector")) {
            String internal = "jdk/incubator/vector/" + vector;
            String descriptor = "Ljdk/incubator/vector/" + vector + ";";
            String primitive = switch (vector) { case "FloatVector" -> "F"; case "DoubleVector" -> "D"; case "IntVector" -> "I"; default -> "J"; };
            add(keys, internal, Set.of("fromArray"), Set.of("(Ljdk/incubator/vector/VectorSpecies;[" + primitive + "I)" + descriptor));
            add(keys, internal, Set.of("fromMemorySegment"), Set.of("(Ljdk/incubator/vector/VectorSpecies;Ljava/lang/foreign/MemorySegment;JLjava/nio/ByteOrder;)" + descriptor));
            add(keys, internal, Set.of("broadcast"), Set.of("(Ljdk/incubator/vector/VectorSpecies;" + primitive + ")" + descriptor));
            add(keys, internal, Set.of("add", "sub", "min", "max"), Set.of("(" + primitive + ")" + descriptor));
            add(keys, internal, Set.of("div"), Set.of("(" + primitive + ")" + descriptor,
                    "(Ljdk/incubator/vector/Vector;)" + descriptor));
            add(keys, internal, Set.of("mul"), Set.of("(" + primitive + ")" + descriptor,
                    "(Ljdk/incubator/vector/Vector;)" + descriptor));
            add(keys, internal, Set.of("intoArray"), Set.of("([" + primitive + "I)V"));
            add(keys, internal, Set.of("intoMemorySegment"), Set.of("(Ljava/lang/foreign/MemorySegment;JLjava/nio/ByteOrder;)V"));
        }
        return Set.copyOf(keys);
    }
    private static void add(Set<InvocationKey> keys, String owner, Set<String> names, Set<String> descriptors) {
        for (String name : names) for (String descriptor : descriptors) keys.add(new InvocationKey(owner, name, descriptor));
    }
    /** Canonical matrix order is the method identity; unlike a hash it is collision-free. */
    static String method(CpuScalarImmediateClampMatrixOracle.Form f) {
        int index = CpuScalarImmediateClampMatrixOracle.forms().indexOf(f);
        if (index < 0) throw new AssertionError("form is outside the canonical 0009A matrix: " + f.id());
        return "f" + index;
    }

    private static String source(List<CpuScalarImmediateClampMatrixOracle.Form> forms,
            Map<String, SourceProgram> programs) {
        StringBuilder s = new StringBuilder("package ").append(PACKAGE).append(";import java.lang.foreign.*;import jdk.incubator.vector.*;public final class ").append(NAME).append("{private ").append(NAME).append("(){}");
        for (var f : forms) append(s, f, programs.get(f.id()));
        return s.append('}').toString();
    }
    private static void append(StringBuilder s, CpuScalarImmediateClampMatrixOracle.Form f, SourceProgram program) {
        var x = f.fixture(); String in = type(x.type(), x.carriers().getFirst()), out = type(x.type(), x.carriers().getLast());
        s.append("public static void ").append(method(f)).append('(').append(in).append(" input,").append(out).append(" output,long[] g,long start,long end){");
        s.append("long ib=g[").append(2 * f.inputShape().rank()).append("],ob=g[").append(2 * f.outputShape().rank() + 1).append("]; ");
        if (program.vector()) appendVector(s, f, program);
        boolean directDenseArray = f.id().startsWith("FIXTURE-")
                && program.address() == AddressProgram.DENSE
                && x.carriers().getFirst() != CarrierAccess.MEMORY_SEGMENT
                && x.carriers().getLast() != CarrierAccess.MEMORY_SEGMENT;
        String advance = appendScalarLoop(s, f, program.address(), program.vector() ? "tail" : "start",
                directDenseArray);
        // A dense array range has direct base-plus-range-relative-ordinal access.  This is the clean-Java
        // choice because it avoids needless loop-carried address state; it is chosen from the
        // shape/layout semantics, not from emitted bytecode.
        String inputAddress = directDenseArray ? "ib+(tailOrdinal-start)" : "ia";
        String outputAddress = directDenseArray ? "ob+(tailOrdinal-start)" : "oa";
        String v = load("input", x.type(), x.carriers().getFirst(), inputAddress);
        String value = operation(f, v); s.append(store("output", x.type(), x.carriers().getLast(), outputAddress, value)).append(';')
                .append(advance).append("}}\n");
    }

    /*
     * These are clean-Java layout algorithms: mutable cursors avoid per-element division and
     * remainder for non-dense broadcast layouts.  They are selected from shape, layout, carrier
     * ABI, and the independent hot-loop policy; source generation never inspects generated
     * Class-File bytes.
     */
    private static String appendScalarLoop(StringBuilder s, CpuScalarImmediateClampMatrixOracle.Form f, AddressProgram program,
            String first, boolean directDenseArray) {
        int rank = f.inputShape().rank();
        int boundaries = 2;
        int inputBase = 2 * rank;
        int inputInner = 2 * rank + boundaries + boundaries * rank;
        // The vector loop owns the leading ordinal range.  Its scalar tail continues from the
        // same semantic range state so no processed element is revisited.
        if (directDenseArray) {
            // The direct scalar loop uses tailOrdinal in both selected addresses.  It needs no
            // mutable address cursor, whether it is the whole scalar body or a vector tail.
        } else if (first.equals("tail") && program == AddressProgram.DENSE) {
            s.append("long ia=ib+(tail-start),oa=ob+(tail-start);");
        } else if (first.equals("tail") && program == AddressProgram.ZERO) {
            s.append("long ia=ib,oa=ob+(tail-start);");
        } else {
            s.append("long ia=ib,oa=ob;");
        }
        switch (program) {
            case DENSE -> { s.append("for(long tailOrdinal=").append(first)
                    .append(";tailOrdinal<end;tailOrdinal++){"); return directDenseArray ? "" : "oa++;ia++;"; }
            case ZERO -> s.append("for(long tailOrdinal=").append(first)
                    .append(";tailOrdinal<end;tailOrdinal++){");
            case LAST_AXIS -> {
                if (!first.equals("tail")) s.append("long inner=g[").append(inputInner).append("]; ");
                s.append("for(long tailOrdinal=")
                        .append(first).append(";tailOrdinal<end;tailOrdinal++){");
                return lastAxisAdvance(rank);
            }
            case BLOCK_OUTER -> {
                int outer = 0;
                if (!first.equals("tail")) s.append("long c0=g[").append(rank).append("],inner=g[")
                        .append(inputInner).append("]; ");
                s.append("for(long tailOrdinal=").append(first)
                        .append(";tailOrdinal<end;tailOrdinal++){");
                return blockOuterAdvance(rank, boundaries, outer);
            }
            case GENERAL -> {
                s.append("long c0=g[").append(rank).append("],c1=g[").append(rank + 1)
                        .append("];for(long tailOrdinal=").append(first)
                        .append(";tailOrdinal<end;tailOrdinal++){");
                return generalAdvance(rank, boundaries);
            }
        }
        return "oa++;";
    }
    private static String lastAxisAdvance(int rank) {
        return "oa++;ia++;inner++;if(inner>=g[" + (rank - 1)
                + "]){inner=0;ia-=g[" + (rank - 1) + "];}";
    }
    private static String blockOuterAdvance(int rank, int boundaries,
            int outer) {
        int innerSize = 2 * rank + boundaries + boundaries * rank + boundaries;
        int stride = 2 * rank + boundaries;
        return "oa++;ia++;inner++;if(inner>=g[" + innerSize + "]){inner=0;ia-=g["
                + innerSize + "];c0++;ia+=g[" + stride + "];if(c0>=g[" + outer
                + "]){c0=0;ia-=g[" + stride + "]*g[" + outer + "];}}";
    }
    private static String generalAdvance(int rank, int boundaries) {
        int strides = 2 * rank + boundaries;
        return "oa++;c1++;ia+=g[" + (strides + 1) + "];if(c1>=g[1]){c1=0;ia-=g["
                + (strides + 1) + "]*g[1];c0++;ia+=g[" + strides
                + "];if(c0>=g[0]){c0=0;ia-=g[" + strides + "]*g[0];}}";
    }
    private static void appendVector(StringBuilder s, CpuScalarImmediateClampMatrixOracle.Form f, SourceProgram program) {
        var x = f.fixture(); boolean d = x.type() == DataType.FLOAT64;
        String vector = switch (x.type()) { case FLOAT32 -> "FloatVector"; case FLOAT64 -> "DoubleVector"; case INT32 -> "IntVector"; case INT64 -> "LongVector"; default -> throw new AssertionError(x.type()); };
        String boxed = switch (x.type()) { case FLOAT32 -> "Float"; case FLOAT64 -> "Double"; case INT32 -> "Integer"; case INT64 -> "Long"; default -> throw new AssertionError(x.type()); };
        String one = switch (x.type()) { case FLOAT32 -> "1.0f"; case FLOAT64 -> "1.0d"; case INT32 -> "1"; case INT64 -> "1L"; default -> throw new AssertionError(x.type()); };
        String immediate = literal(x.type(), x.immediate()), upper = x.upper() == null ? "" : literal(x.type(), x.upper());
        String byteOffset = "(ib+(ordinal-start))*" + (d || x.type() == DataType.INT64 ? "8L" : "4L");
        String scalarLoad = x.carriers().getFirst() == CarrierAccess.MEMORY_SEGMENT
                ? "input.get(ValueLayout.JAVA_" + (x.type() == DataType.FLOAT32 ? "FLOAT" : x.type() == DataType.FLOAT64 ? "DOUBLE" : x.type() == DataType.INT32 ? "INT" : "LONG")
                        + "_UNALIGNED,ib*" + (d || x.type() == DataType.INT64 ? "8L" : "4L") + ")"
                : "input[(int)ib]";
        String vectorLoad = (x.operation() == ScalarElementwiseKind.POW
                && (x.category() == CpuScalarImmediateClampMatrixOracle.Category.POW_POSITIVE_ZERO
                        || x.category() == CpuScalarImmediateClampMatrixOracle.Category.POW_NEGATIVE_ZERO))
                ? vector + ".broadcast(sp," + one + ")"
                : program.address() == AddressProgram.ZERO
                ? vector + ".broadcast(sp," + scalarLoad + ")"
                : x.carriers().getFirst() == CarrierAccess.MEMORY_SEGMENT
                ? vector + ".fromMemorySegment(sp,input," + byteOffset + ",java.nio.ByteOrder.nativeOrder())"
                : vector + ".fromArray(sp,input,(int)(ib+(ordinal-start)))";
        String vectorStore = x.carriers().getLast() == CarrierAccess.MEMORY_SEGMENT
                ? ".intoMemorySegment(output,(ob+(ordinal-start))*" + (d || x.type() == DataType.INT64 ? "8L" : "4L")
                        + ",java.nio.ByteOrder.nativeOrder())"
                : ".intoArray(output,(int)(ob+(ordinal-start)))";
        if (program.address() == AddressProgram.LAST_AXIS) {
            int rank = f.inputShape().rank();
            int inputInner = 2 * rank + 2 + 2 * rank;
            String cursorLoad = vectorLoad.replace("(ib+(ordinal-start))", "ib");
            String cursorStore = vectorStore.replace("(ob+(ordinal-start))", "ob");
            s.append("VectorSpecies<").append(boxed).append("> sp=").append(vector)
                    .append(".SPECIES_PREFERRED;long lanes=sp.length(),inner=g[").append(inputInner)
                    .append("],ordinal=start;for(;ordinal<end;){long span=Math.min(end-ordinal,g[")
                    .append(rank - 1).append("]-inner);if(span<lanes)break;")
                    .append(vector).append(" v=").append(cursorLoad).append(';')
                    .append(vectorOperation(x, vector, immediate, upper, one)).append(cursorStore)
                    .append(";ordinal+=lanes;ib+=lanes;inner+=lanes;if(inner>=g[")
                    .append(rank - 1).append("]){inner=0;ib-=g[").append(rank - 1)
                    .append("]; }ob+=lanes;}long tail=ordinal;");
            return;
        }
        if (program.address() == AddressProgram.BLOCK_OUTER) {
            int rank = f.inputShape().rank();
            int inputInner = 2 * rank + 2 + 2 * rank;
            int innerSize = inputInner + 2;
            int stride = 2 * rank + 2;
            String cursorLoad = vectorLoad.replace("(ib+(ordinal-start))", "ib");
            String cursorStore = vectorStore.replace("(ob+(ordinal-start))", "ob");
            s.append("VectorSpecies<").append(boxed).append("> sp=").append(vector)
                    .append(".SPECIES_PREFERRED;long lanes=sp.length(),c0=g[").append(rank)
                    .append("],inner=g[").append(inputInner).append("],ordinal=start;for(;ordinal<end;){long span=Math.min(end-ordinal,g[")
                    .append(innerSize).append("]-inner);if(span<lanes)break;")
                    .append(vector).append(" v=").append(cursorLoad).append(';')
                    .append(vectorOperation(x, vector, immediate, upper, one)).append(cursorStore)
                    .append(";ordinal+=lanes;ib+=lanes;inner+=lanes;if(inner>=g[").append(innerSize)
                    .append("]){inner=0;ib-=g[").append(innerSize).append("];c0++;ib+=g[")
                    .append(stride).append("];if(c0>=g[0]){c0=0;ib-=g[")
                    .append(stride).append("]*g[0];}}ob+=lanes;}long tail=ordinal;");
            return;
        }
        s.append("VectorSpecies<").append(boxed).append("> sp=").append(vector).append(".SPECIES_PREFERRED;long lanes=sp.length(),limit=end-(end-start)%lanes,ordinal=start;for(;ordinal<limit;ordinal+=lanes){")
                .append(vector).append(" v=").append(vectorLoad).append(';');
        s.append(vectorOperation(x, vector, immediate, upper, one)).append(vectorStore).append(";}long tail=ordinal;");
    }
    private static String vectorOperation(CpuScalarImmediateClampMatrixOracle.Fixture x,
            String vector, String immediate, String upper, String one) {
        return switch (x.operation()) {
            case ADD -> "v.add(" + immediate + ')'; case SUB -> "v.sub(" + immediate + ')'; case MUL -> "v.mul(" + immediate + ')';
            case DIV -> "v.div(" + immediate + ')'; case MIN -> "v.min(" + immediate + ')'; case MAX -> "v.max(" + immediate + ')';
            case CLAMP -> "v.max(" + immediate + ").min(" + upper + ')';
            case POW -> switch (x.category()) { case POW_POSITIVE_ZERO, POW_NEGATIVE_ZERO -> vector + ".broadcast(sp," + one + ')'; case IDENTITY -> "v"; case SQUARE -> "v.mul(v)"; case RECIPROCAL -> vector + ".broadcast(sp," + one + ").div(v)"; default -> throw new AssertionError(x.category()); };
        };
    }
    private static String address(CpuScalarImmediateClampMatrixOracle.Form f) { return switch (f.accessRegime()) {
        case DENSE_LINEAR -> "ib+ordinal"; case SCALAR_ALL_ZERO -> "ib";
        case LAST_AXIS_BIAS -> "ib+(ordinal%g[1])";
        case BLOCK_OUTER, GENERAL_ODOMETER -> "ib+(ordinal/g[1])*g[6]+(ordinal%g[1])*g[7]"; }; }
    private static String operation(CpuScalarImmediateClampMatrixOracle.Form f, String v) {
        var k=f.fixture().operation(); String q="("+literal(f.fixture().type(), f.fixture().immediate())+")";
        if (k==ScalarElementwiseKind.CLAMP) return "Math.min(Math.max("+v+','+q+"),"+literal(f.fixture().type(),f.fixture().upper())+")";
        if (k == ScalarElementwiseKind.POW) return switch (f.fixture().category()) {
            case POW_POSITIVE_ZERO, POW_NEGATIVE_ZERO -> f.fixture().type() == DataType.FLOAT64 ? "1.0d" : "1.0f";
            case IDENTITY -> v; case SQUARE -> v+"*"+v; case RECIPROCAL -> (f.fixture().type() == DataType.FLOAT64 ? "1.0d/" : "1.0f/")+v;
            case DIRECT_FRACTIONAL, DIRECT_INTEGRAL -> f.fixture().type() == DataType.FLOAT64
                    ? "StrictMath.pow("+v+','+q+")" : "(float)StrictMath.pow("+v+','+q+")";
            default -> throw new AssertionError(f.fixture().category()); };
        return switch(k) { case ADD -> v+"+"+q; case SUB -> v+"-"+q; case MUL -> v+"*"+q; case DIV -> v+"/"+q;
            case MIN -> "Math.min("+v+','+q+")"; case MAX -> "Math.max("+v+','+q+")";
            default -> throw new AssertionError(k); }; }
    private static String literal(DataType t, io.github.pho001.synaptik.model.datatype.ScalarValue v) { return switch(t) {
        case BFLOAT16, FLOAT32 -> Float.toString(t==DataType.BFLOAT16 ? CpuScalarImmediateClampEquivalenceOracle.floatValue(v.bfloat16Bits()) : v.float32Value())+"f";
        case FLOAT64 -> Double.toString(v.float64Value())+"d"; case INT32 -> Integer.toString(v.int32Value()); case INT64 -> Long.toString(v.int64Value())+"L"; default -> throw new AssertionError(t); }; }
    private static String type(DataType t, CarrierAccess c) { if(c==CarrierAccess.MEMORY_SEGMENT)return "MemorySegment";return switch(t){case BFLOAT16->"short[]";case FLOAT32->"float[]";case FLOAT64->"double[]";case INT32->"int[]";case INT64->"long[]";default->throw new AssertionError(t);}; }
    private static String load(String n,DataType t,CarrierAccess c,String i) { String raw=c==CarrierAccess.MEMORY_SEGMENT?n+".get(ValueLayout.JAVA_"+(t==DataType.FLOAT32?"FLOAT":t==DataType.FLOAT64?"DOUBLE":t==DataType.INT32?"INT":t==DataType.INT64?"LONG":"SHORT")+"_UNALIGNED,"+i+"*"+(t==DataType.FLOAT64||t==DataType.INT64?8:t==DataType.BFLOAT16?2:4)+"L)":n+"[(int)("+i+")]"; return t==DataType.BFLOAT16?"Float.intBitsToFloat(((int)"+raw+"&0xffff)<<16)":raw; }
    private static String store(String n,DataType t,CarrierAccess c,String i,String v) { String value=t==DataType.BFLOAT16?bfloat(v):v; if(c==CarrierAccess.MEMORY_SEGMENT)return n+".set(ValueLayout.JAVA_"+(t==DataType.FLOAT32?"FLOAT":t==DataType.FLOAT64?"DOUBLE":t==DataType.INT32?"INT":t==DataType.INT64?"LONG":"SHORT")+"_UNALIGNED,"+i+"*"+(t==DataType.FLOAT64||t==DataType.INT64?8:t==DataType.BFLOAT16?2:4)+"L,"+value+")";return n+"[(int)("+i+")]="+value; }
    private static String bfloat(String value) { String bits="Float.floatToRawIntBits("+value+")"; return "(short)((("+bits+"&0x7fffffff)>0x7f800000)?0x7fc0:("+bits+"+0x7fff+("+bits+">>>16&1))>>>16)"; }
    private static final class Source extends SimpleJavaFileObject { final String s; Source(String n,String s){super(URI.create("string:///"+n.replace('.','/')+Kind.SOURCE.extension),Kind.SOURCE);this.s=s;}public CharSequence getCharContent(boolean x){return s;} }
    private static final class ByteFile extends SimpleJavaFileObject { final ByteArrayOutputStream b=new ByteArrayOutputStream(); ByteFile(String n){super(URI.create("bytes:///"+n.replace('.','/')+Kind.CLASS.extension),Kind.CLASS);}public OutputStream openOutputStream(){return b;}byte[] bytes(){return b.toByteArray();} }
}
