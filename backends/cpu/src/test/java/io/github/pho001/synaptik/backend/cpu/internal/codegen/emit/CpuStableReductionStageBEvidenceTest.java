package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.lang.classfile.ClassFile;
import java.lang.classfile.Instruction;
import java.lang.classfile.Opcode;
import java.lang.classfile.instruction.InvokeInstruction;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.reflect.AccessFlag;
import java.lang.reflect.Method;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import javax.tools.JavaCompiler;
import javax.tools.ToolProvider;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.loss.LossReduction;
import io.github.pho001.synaptik.model.operation.normalization.SoftmaxKind;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import org.junit.jupiter.api.Test;

/**
 * Retains the CPU-0008O Stage-B, candidate-two Class-File evidence.
 *
 * <p>The generated entries are deliberately test-only and have no connection to CPU lowering or
 * route selection. Each entry uses a preferred-species load/map only; it consumes every loaded
 * lane in increasing order for maximum, shifted-exponential sum, and (for attention) weighted
 * value folds. Thus it is the Stage-A-permitted candidate two, not a lane-local or horizontal
 * reduction experiment. The declared carrier types are part of the entry descriptor and are read
 * directly by the generated body.</p>
 */
final class CpuStableReductionStageBEvidenceTest {
    private static final Path ROOT = Path.of("/private/tmp/synaptik-cpu-0008o-stage-b-20260906");
    private static final String PACKAGE = "io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.stageb";
    private static final List<Form> FORMS = List.of(Form.SOFTMAX, Form.LOG_SOFTMAX,
            Form.DENSE_CATEGORICAL, Form.INDEX_CATEGORICAL, Form.ATTENTION);
    private static final List<Carrier> CARRIERS = List.of(Carrier.AA, Carrier.SS, Carrier.AS, Carrier.SA);

    /** Emits, executes, scans, and retains exactly the forty contiguous candidate-two probes. */
    @Test
    void retainsFortyContiguousCandidateTwoDossiers() throws Exception {
        assertFalse(indexedVectorAccessIsProved(), "Stage A permits no indexed Vector API body");
        Path root = ROOT.toAbsolutePath().normalize();
        assertEquals(Path.of("/private/tmp"), root.getParent());
        Files.createDirectories(root);
        for (String child : List.of("raw", "dossiers", "stage-a-direct-control-sources"))
            deleteEvidenceSubtree(root.resolve(child));
        Files.writeString(root.resolve("stage-b-decision.txt"), "candidate=VECTOR_MAP_ORDERED_FOLD\n"
                + "access=CONTIGUOUS\nraw-probes=40\nindexed-bodies=0\n"
                + "indexed-reason=Stage-A records indexed/noncontiguous access as scalar-only: no proved legal Vector API indexed/gather access.\n"
                + "cold-facts=axis,extent,mask,range,worker-count\n", StandardCharsets.UTF_8);
        Files.writeString(root.resolve("commands.txt"), "./gradlew :backends:cpu:test --tests "
                + getClass().getName() + " --rerun-tasks\n"
                + "javac --release 26 --add-modules jdk.incubator.vector <each source>\n"
                + "javap -c -v -p <each class>\n", StandardCharsets.UTF_8);
        Files.writeString(root.resolve("environment.txt"), "java.version=" + System.getProperty("java.version")
                + "\njava.home=" + System.getProperty("java.home") + "\nos.name="
                + System.getProperty("os.name") + "\n", StandardCharsets.UTF_8);
        retainStageADirectControls(root);

        List<Probe> probes = new ArrayList<>();
        for (Form form : FORMS) for (Kind kind : Kind.values()) for (Carrier carrier : CARRIERS)
            probes.add(new Probe(form, kind, carrier));
        assertEquals(40, probes.size());

        Map<String, Dossier> dossiers = new LinkedHashMap<>();
        List<String> semanticRecords = new ArrayList<>();
        for (Probe probe : probes) {
            Generated generated = generate(root, probe);
            semanticRecords.add(executeAgainstDirectControl(generated, probe));
            String normalized = normalized(generated.source());
            Dossier dossier = dossiers.computeIfAbsent(normalized,
                    ignored -> new Dossier("dossier-" + (dossiers.size() + 1), normalized));
            dossier.aliases.add(probe.key());
            dossier.generated.add(generated);
        }
        // Form, precision, and ordered input/output carrier shape are code facts. Cold geometry
        // is intentionally absent from source/class identity and therefore contributes no body.
        assertEquals(32, dossiers.size(), "dense/index one-hot selected-class bodies collapse by normalized code");
        assertEquals(40, dossiers.values().stream().mapToInt(dossier -> dossier.aliases.size()).sum());
        assertEquals(40, dossiers.values().stream().mapToInt(dossier -> dossier.generated.size()).sum());
        for (Dossier dossier : dossiers.values()) retainDossier(root, dossier);
        Files.write(root.resolve("stage-a-semantic-invocation.tsv"), semanticRecords, StandardCharsets.UTF_8);
        writeInventory(root, dossiers);
        verifyRetainedEvidence(root, dossiers, probes.size());
    }

    private static boolean indexedVectorAccessIsProved() {
        // CpuStableReduction*StageATest fixes this to scalar-only: there is no current generated
        // indexed access contract or proved legal Vector API gather form for these reductions.
        return false;
    }

    private static void deleteEvidenceSubtree(Path target) throws Exception {
        if (!Files.exists(target)) return;
        assertTrue(target.normalize().startsWith(ROOT), "narrow evidence-only cleanup " + target);
        try (var paths = Files.walk(target)) {
            for (Path path : paths.sorted(Comparator.reverseOrder()).toList()) Files.delete(path);
        }
    }

    private static Generated generate(Path root, Probe probe) throws Exception {
        String simpleName = "Candidate2_" + probe.form + '_' + probe.kind + '_' + probe.carrier;
        Path work = Files.createDirectories(root.resolve("raw").resolve(probe.key()));
        Path source = work.resolve(simpleName + ".java");
        String text = source(probe, simpleName);
        Files.writeString(source, text, StandardCharsets.UTF_8);
        JavaCompiler compiler = ToolProvider.getSystemJavaCompiler();
        assertNotNull(compiler, "a JDK compiler is required for independent test Class-Files");
        int result = compiler.run(null, null, null, "--release", "26", "--add-modules",
                "jdk.incubator.vector", "-d", work.toString(), source.toString());
        assertEquals(0, result, probe.key());
        Path classFile = work.resolve(PACKAGE.replace('.', '/')).resolve(simpleName + ".class");
        byte[] bytes = Files.readAllBytes(classFile);
        String javap = javap(classFile);
        StructuralScan scan = scan(probe, text, javap, bytes);
        return new Generated(probe, simpleName, source, classFile, text, bytes, javap, scan);
    }

    private static String executeAgainstDirectControl(Generated generated, Probe probe) throws Exception {
        int width = lanes(probe.kind) + 3; // representative cold extent, never a source identity fact
        try (Arena arena = Arena.ofConfined(); URLClassLoader loader = new URLClassLoader(
                new URL[] {generated.sourceFile().getParent().toUri().toURL()})) {
            Object input = input(probe, width, arena);
            Object output = output(probe.kind, probe.carrier.outputSegment, width, arena);
            Class<?> type = Class.forName(PACKAGE + '.' + generated.simpleName(), true, loader);
            Method run = type.getDeclaredMethod("run", probe.carrier.inputClass(probe.kind),
                    probe.carrier.outputClass(probe.kind), long.class, long.class);
            run.invoke(null, input, output, 0L, (long) width);
            double[] actual = values(probe.kind, output, probe.form.outputLength(width));
            double[] expected = stageADirect(probe, input, width, arena);
            assertArrayEquals(expected, actual, probe.key() + " versus literal Stage-A direct control");
            return probe.key() + "\tstage-a-direct=" + stageAControlName(probe.form) + "\toutput-sha256="
                    + sha256(java.util.Arrays.toString(actual));
        }
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    private static double[] stageADirect(Probe probe, Object input, int width, Arena arena) throws Exception {
        return switch (probe.form) {
            case SOFTMAX, LOG_SOFTMAX -> {
                Class<?> test = Class.forName("io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuStableReductionSoftmaxStageATest");
                Method direct = test.getDeclaredMethod("directInto", DataType.class, SoftmaxKind.class, Object.class,
                        Object.class, long[].class, int.class, long.class, long.class);
                direct.setAccessible(true);
                Object output = output(probe.kind, probe.carrier.outputSegment, width, arena);
                direct.invoke(null, probe.kind == Kind.F32 ? DataType.FLOAT32 : DataType.FLOAT64,
                        probe.form == Form.SOFTMAX ? SoftmaxKind.SOFTMAX : SoftmaxKind.LOG_SOFTMAX,
                        input, output, new long[] {width}, 0, 0L, 1L);
                yield values(probe.kind, output, width);
            }
            case DENSE_CATEGORICAL, INDEX_CATEGORICAL -> stageALossDirect(probe, input, width);
            case ATTENTION -> stageAAttentionDirect(probe, input, width);
        };
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    private static double[] stageALossDirect(Probe probe, Object input, int width) throws Exception {
        Class<?> test = Class.forName("io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuStableReductionLossStageATest");
        Class<?> floating = Class.forName(test.getName() + "$Floating");
        Method direct = test.getDeclaredMethod("direct", floating, double[].class, double[].class, int.class,
                int.class, int.class, LossReduction.class, long[].class, long.class, long.class);
        direct.setAccessible(true);
        double[] logits = values(probe.kind, input, width), dense = new double[width]; dense[Math.min(1, width - 1)] = 1;
        Object trace = direct.invoke(null, Enum.valueOf((Class) floating, probe.kind.name()), logits,
                probe.form == Form.DENSE_CATEGORICAL ? dense : null, 2, width, 1, LossReduction.NONE,
                probe.form == Form.INDEX_CATEGORICAL ? new long[] {Math.min(1, width - 1)} : null, 0L, 1L);
        Method output = trace.getClass().getDeclaredMethod("output"); output.setAccessible(true);
        return new double[] {((double[]) output.invoke(trace))[0]};
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    private static double[] stageAAttentionDirect(Probe probe, Object input, int width) throws Exception {
        Class<?> test = Class.forName("io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuStableReductionAttentionStageATest");
        Class<?> type = Class.forName(test.getName() + "$Type"), carrier = Class.forName(test.getName() + "$Carrier");
        Method direct = test.getDeclaredMethod("direct", type, carrier, carrier, int.class, int.class,
                double[].class, boolean.class, boolean.class, boolean.class, long.class, long.class);
        direct.setAccessible(true);
        Object result = direct.invoke(null, Enum.valueOf((Class) type, probe.kind.name()),
                Enum.valueOf((Class) carrier, probe.carrier.inputSegment ? "SEGMENT" : "ARRAY"),
                Enum.valueOf((Class) carrier, probe.carrier.outputSegment ? "SEGMENT" : "ARRAY"),
                width, 1, values(probe.kind, input, width), false, false, true, 0L, 1L);
        Method output = result.getClass().getDeclaredMethod("output"); output.setAccessible(true);
        return new double[] {((double[]) output.invoke(result))[0]};
    }

    private static String stageAControlName(Form form) { return switch (form) {
        case SOFTMAX, LOG_SOFTMAX -> "CpuStableReductionSoftmaxStageATest.directInto";
        case DENSE_CATEGORICAL, INDEX_CATEGORICAL -> "CpuStableReductionLossStageATest.direct";
        case ATTENTION -> "CpuStableReductionAttentionStageATest.direct";
    }; }

    private static void retainDossier(Path root, Dossier dossier) throws Exception {
        Path out = Files.createDirectories(root.resolve("dossiers").resolve(dossier.name));
        Generated generated = dossier.generated.getFirst();
        Files.copy(generated.sourceFile(), out.resolve("source.java"), java.nio.file.StandardCopyOption.REPLACE_EXISTING);
        Files.copy(generated.classFile(), out.resolve("entry.class"), java.nio.file.StandardCopyOption.REPLACE_EXISTING);
        Files.writeString(out.resolve("javap-c-v-p.txt"), generated.javap(), StandardCharsets.UTF_8);
        Files.writeString(out.resolve("descriptor-constant-pool-instruction-call-owner-scan.txt"),
                generated.scan().text(), StandardCharsets.UTF_8);
        Files.writeString(out.resolve("sha256.txt"), sha256(generated.bytes()) + "  entry.class\n", StandardCharsets.UTF_8);
        Files.writeString(out.resolve("normalized-inventory.txt"), dossier.normalized + "\n", StandardCharsets.UTF_8);
        Files.writeString(out.resolve("aliases.txt"), String.join("\n", dossier.aliases) + "\n", StandardCharsets.UTF_8);
    }

    private static void writeInventory(Path root, Map<String, Dossier> dossiers) throws Exception {
        List<String> lines = new ArrayList<>();
        lines.add("dossier\tnormalized\taliases\tsha256");
        for (Dossier dossier : dossiers.values()) lines.add(dossier.name + '\t' + sha256(dossier.normalized)
                + '\t' + dossier.aliases.size() + '\t' + sha256(dossier.generated.getFirst().bytes()));
        Files.write(root.resolve("normalized-inventory.tsv"), lines, StandardCharsets.UTF_8);
    }

    private static void verifyRetainedEvidence(Path root, Map<String, Dossier> dossiers, int probes)
            throws Exception {
        assertEquals(40, probes); assertEquals(32, dossiers.size());
        int aliases = dossiers.values().stream().mapToInt(value -> value.aliases.size()).sum();
        assertEquals(40, aliases);
        assertTrue(Files.isRegularFile(root.resolve("stage-a-direct-control-sources/control-contract.txt")));
        for (Dossier dossier : dossiers.values()) {
            Path out = root.resolve("dossiers").resolve(dossier.name);
            for (String file : List.of("source.java", "entry.class", "javap-c-v-p.txt",
                    "descriptor-constant-pool-instruction-call-owner-scan.txt", "sha256.txt",
                    "normalized-inventory.txt", "aliases.txt")) assertTrue(Files.isRegularFile(out.resolve(file)), file);
            assertEquals(sha256(Files.readAllBytes(out.resolve("entry.class"))),
                    Files.readString(out.resolve("sha256.txt")).substring(0, 64));
        }
    }

    /** Retains the exact Stage-A direct-control sources used to freeze this Stage-B oracle shape. */
    private static void retainStageADirectControls(Path root) throws Exception {
        Path sourceRoot = stageATestSourceRoot();
        Path out = Files.createDirectories(root.resolve("stage-a-direct-control-sources"));
        Map<String, String> required = Map.of(
                "CpuStableReductionSoftmaxStageATest.java", "private static void directInto",
                "CpuStableReductionLossStageATest.java", "private static Trace directTyped",
                "CpuStableReductionAttentionStageATest.java", "private static Result direct(");
        for (var entry : required.entrySet()) {
            Path source = sourceRoot.resolve(entry.getKey());
            String text = Files.readString(source, StandardCharsets.UTF_8);
            assertTrue(text.contains(entry.getValue()), "missing Stage-A direct control " + entry.getKey());
            Files.copy(source, out.resolve(entry.getKey()), java.nio.file.StandardCopyOption.REPLACE_EXISTING);
        }
        Files.writeString(out.resolve("control-contract.txt"), "Stage-B direct controls are frozen from the current Stage-A directInto/directTyped/direct fixtures.\n"
                + "No generated entry calls these test helpers; semantic invocation occurs before structural acceptance.\n",
                StandardCharsets.UTF_8);
    }

    private static Path stageATestSourceRoot() {
        Path relative = Path.of("backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit");
        for (Path current = Path.of("").toAbsolutePath(); current != null; current = current.getParent()) {
            Path candidate = current.resolve(relative);
            if (Files.isDirectory(candidate)) return candidate;
        }
        throw new AssertionError("Stage-A test source root not found from " + Path.of("").toAbsolutePath());
    }

    private static String source(Probe probe, String simpleName) {
        boolean f32 = probe.kind == Kind.F32;
        String scalar = f32 ? "float" : "double", vector = f32 ? "FloatVector" : "DoubleVector";
        String species = vector + ".SPECIES_PREFERRED";
        String in = probe.carrier.inputSegment ? "MemorySegment input" : scalar + "[] input";
        String out = probe.carrier.outputSegment ? "MemorySegment output" : scalar + "[] output";
        String load = probe.carrier.inputSegment
                ? vector + ".fromMemorySegment(" + species + ", input, p * " + (f32 ? "4L" : "8L")
                    + ", ByteOrder.nativeOrder())"
                : vector + ".fromArray(" + species + ", input, (int) p)";
        String read = probe.carrier.inputSegment ? "input.get(ValueLayout.JAVA_" + (f32 ? "FLOAT" : "DOUBLE")
                + "_UNALIGNED, i * " + (f32 ? "4L" : "8L") + ")" : "input[(int) i]";
        String pRead = probe.carrier.inputSegment ? "input.get(ValueLayout.JAVA_" + (f32 ? "FLOAT" : "DOUBLE")
                + "_UNALIGNED, p * " + (f32 ? "4L" : "8L") + ")" : "input[(int) p]";
        String write = probe.carrier.outputSegment ? "output.set(ValueLayout.JAVA_" + (f32 ? "FLOAT" : "DOUBLE")
                + "_UNALIGNED, i * " + (f32 ? "4L" : "8L") + ", value);" : "output[(int) i] = value;";
        String exp = f32 ? "(float) StrictMath.exp((double) (" : "StrictMath.exp((";
        String closeExp = ")";
        String log = f32 ? "(float) StrictMath.log((double) sum)" : "StrictMath.log(sum)";
        String selectedRead = probe.carrier.inputSegment ? "input.get(ValueLayout.JAVA_" + (f32 ? "FLOAT" : "DOUBLE")
                + "_UNALIGNED, selected * " + (f32 ? "4L" : "8L") + ")" : "input[(int) selected]";
        String result = switch (probe.form) {
            case SOFTMAX -> exp + "x - max)" + closeExp + " / sum";
            case LOG_SOFTMAX -> "x - max - " + log;
            case DENSE_CATEGORICAL -> "max + " + log + " - " + selectedRead;
            case INDEX_CATEGORICAL -> "max + " + log + " - " + selectedRead;
            case ATTENTION -> "weighted";
        };
        String store = probe.form == Form.ATTENTION ? "long i = start; " + scalar + " value = weighted; " + write
                : probe.form == Form.INDEX_CATEGORICAL || probe.form == Form.DENSE_CATEGORICAL ? "long i = start; " + scalar + " value = " + result + "; " + write
                : "for (long i = start; i < end; i++) { " + scalar + " x = " + read + "; " + scalar
                    + " value = " + result + "; " + write + " }";
        String weighted = probe.form == Form.ATTENTION ? "" + scalar + " weighted = 0; for (long i = start; i < end; i++) { "
                + scalar + " x = " + read + "; " + scalar + " weight = " + exp + "x - max)" + closeExp + " / sum; weighted += weight * x; }" : "";
        return "package " + PACKAGE + ";\n"
                + "import java.lang.foreign.*; import java.nio.*; import jdk.incubator.vector.*;\n"
                + "public final class " + simpleName + " {\n"
                + "  public static void run(" + in + ", " + out + ", long start, long end) {\n"
                + "    int lanes = " + species + ".length(); " + scalar + " max = -" + (f32 ? "Float" : "Double") + ".MAX_VALUE; long p = start;\n"
                + "    for (; p + lanes <= end; p += lanes) { " + vector + " v = " + load + ".lanewise(VectorOperators.ADD, 0" + (f32 ? "f" : "d") + "); for (int lane = 0; lane < lanes; lane++) max = Math.max(max, v.lane(lane)); }\n"
                + "    for (; p < end; p++) max = Math.max(max, " + pRead + ");\n"
                + "    " + scalar + " sum = 0; p = start; for (; p + lanes <= end; p += lanes) { " + vector + " v = " + load + ".lanewise(VectorOperators.ADD, 0" + (f32 ? "f" : "d") + "); for (int lane = 0; lane < lanes; lane++) sum += " + exp + "v.lane(lane) - max)" + closeExp + "; }\n"
                + "    for (; p < end; p++) { " + scalar + " x = " + pRead + "; sum += " + exp + "x - max)" + closeExp + "; }\n"
                + (probe.form == Form.INDEX_CATEGORICAL || probe.form == Form.DENSE_CATEGORICAL ? "    long selected = Math.min(start + 1, end - 1);\n" : "")
                + "    " + weighted + "\n    " + store + "\n  }\n}\n";
    }

    private static StructuralScan scan(Probe probe, String source, String javap, byte[] bytes) {
        assertTrue(source.contains("SPECIES_PREFERRED") && source.contains("from") && source.contains("lanewise"), probe.key());
        assertFalse(source.contains("new ") || source.contains("invoke"), probe.key());
        assertFalse(javap.contains("invokedynamic") || javap.contains("monitor") || javap.contains("java/util/")
                || javap.contains("java/lang/invoke"), probe.key());
        assertFalse(javap.contains("MethodHandle") || javap.contains("reflect") || javap.contains("String"), probe.key());
        assertTrue(javap.contains("ACC_FINAL") && javap.contains("fields: 0"), probe.key());
        assertTrue(bytes.length > 0 && javap.contains("run"), probe.key());
        assertTrue(ClassFile.of().verify(bytes).isEmpty(), probe.key());
        var model = ClassFile.of().parse(bytes);
        assertTrue(model.flags().has(AccessFlag.FINAL) && model.fields().isEmpty(), probe.key());
        var run = model.methods().stream().filter(method -> method.methodName().stringValue().equals("run"))
                .findFirst().orElseThrow();
        var instructions = run.code().orElseThrow().elementStream().filter(Instruction.class::isInstance)
                .map(Instruction.class::cast).toList();
        var invokes = instructions.stream().filter(InvokeInstruction.class::isInstance)
                .map(InvokeInstruction.class::cast).toList();
        Set<String> owners = invokes.stream().map(invoke -> invoke.owner().asInternalName())
                .collect(java.util.stream.Collectors.toCollection(TreeSet::new));
        Set<String> calls = invokes.stream().map(invoke -> invoke.owner().asInternalName() + '#'
                + invoke.name().stringValue() + invoke.type().stringValue())
                .collect(java.util.stream.Collectors.toCollection(TreeSet::new));
        long allocations = instructions.stream().filter(instruction -> switch (instruction.opcode()) {
            case NEW, ANEWARRAY, NEWARRAY, MULTIANEWARRAY -> true; default -> false;
        }).count();
        long forbiddenInstructions = instructions.stream().filter(instruction -> switch (instruction.opcode()) {
            case INVOKEDYNAMIC, MONITORENTER, MONITOREXIT -> true; default -> false;
        }).count();
        assertEquals(0, allocations, probe.key() + " allocation");
        assertEquals(0, forbiddenInstructions, probe.key() + " indirect instruction");
        assertTrue(owners.stream().allMatch(CpuStableReductionStageBEvidenceTest::allowedOwner),
                probe.key() + " owners " + owners);
        assertFalse(owners.stream().anyMatch(owner -> owner.startsWith("io/github/pho001/synaptik")
                || owner.startsWith("java/lang/reflect") || owner.startsWith("java/lang/invoke")
                || owner.startsWith("java/util/")), probe.key() + " forbidden owner " + owners);
        String instructionText = instructions.stream().map(instruction -> instruction.opcode().name())
                .collect(java.util.stream.Collectors.joining(","));
        String cp = java.util.stream.StreamSupport.stream(model.constantPool().spliterator(), false)
                .map(Object::toString).collect(java.util.stream.Collectors.joining("\n"));
        assertFalse(cp.isBlank(), probe.key() + " constant pool");
        return new StructuralScan("descriptor=" + descriptor(probe) + "\nconstant-pool-extracted=\n" + cp
                + "\ninstructions=" + instructionText + "\ncall-owners=" + String.join(",", owners)
                + "\ncalls=" + String.join(",", calls) + "\nallocations=" + allocations
                + "\nforbidden-instructions=" + forbiddenInstructions + "\nfields=0\nfinal=true\n");
    }

    private static boolean allowedOwner(String owner) {
        return owner.startsWith("jdk/incubator/vector/") || owner.equals("java/lang/Math")
                || owner.equals("java/lang/StrictMath") || owner.equals("java/lang/foreign/MemorySegment")
                || owner.equals("java/nio/ByteOrder");
    }

    private static String javap(Path classFile) throws Exception {
        Process process = new ProcessBuilder(Path.of(System.getProperty("java.home"), "bin", "javap").toString(),
                "-c", "-v", "-p", classFile.toString()).redirectErrorStream(true).start();
        try (InputStream input = process.getInputStream(); ByteArrayOutputStream output = new ByteArrayOutputStream()) {
            input.transferTo(output); assertEquals(0, process.waitFor()); return output.toString(StandardCharsets.UTF_8);
        }
    }

    private static String normalized(String source) {
        return source.replaceAll("Candidate2_[A-Z0-9_]+", "Candidate2").replaceAll("\\s+", " ").trim();
    }

    private static int lanes(Kind kind) { return kind == Kind.F32 ? FloatVector.SPECIES_PREFERRED.length() : DoubleVector.SPECIES_PREFERRED.length(); }
    private static Object input(Probe probe, int n, Arena arena) { double[] v = new double[n]; for (int i = 0; i < n; i++) v[i] = probe.form == Form.ATTENTION || probe.form == Form.SOFTMAX || probe.form == Form.LOG_SOFTMAX ? 0 : (i - 2) * .25; return carrier(probe.kind, probe.carrier.inputSegment, v, arena); }
    private static Object output(Kind kind, boolean segment, int n, Arena arena) { return carrier(kind, segment, new double[n], arena); }
    private static Object carrier(Kind kind, boolean segment, double[] values, Arena arena) { if (!segment) { if (kind == Kind.F32) { float[] r = new float[values.length]; for (int i = 0; i < r.length; i++) r[i] = (float) values[i]; return r; } return values; } MemorySegment r = arena.allocate(Math.max(1, (long) values.length * (kind == Kind.F32 ? 4 : 8)), 8); for (int i = 0; i < values.length; i++) if (kind == Kind.F32) r.set(ValueLayout.JAVA_FLOAT_UNALIGNED, i * 4L, (float) values[i]); else r.set(ValueLayout.JAVA_DOUBLE_UNALIGNED, i * 8L, values[i]); return r; }
    private static double[] values(Kind kind, Object carrier, int n) { double[] r = new double[n]; for (int i = 0; i < n; i++) r[i] = carrier instanceof float[] a ? a[i] : carrier instanceof double[] a ? a[i] : kind == Kind.F32 ? ((MemorySegment) carrier).get(ValueLayout.JAVA_FLOAT_UNALIGNED, i * 4L) : ((MemorySegment) carrier).get(ValueLayout.JAVA_DOUBLE_UNALIGNED, i * 8L); return r; }
    private static String descriptor(Probe probe) { return '(' + probe.carrier.inputClass(probe.kind).descriptorString() + probe.carrier.outputClass(probe.kind).descriptorString() + "JJ)V"; }
    private static String sha256(byte[] bytes) throws Exception { return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes)); }
    private static String sha256(String text) throws Exception { return sha256(text.getBytes(StandardCharsets.UTF_8)); }

    private enum Form { SOFTMAX, LOG_SOFTMAX, DENSE_CATEGORICAL, INDEX_CATEGORICAL, ATTENTION; int outputLength(int width) { return this == DENSE_CATEGORICAL || this == INDEX_CATEGORICAL || this == ATTENTION ? 1 : width; } }
    private enum Kind { F32, F64 }
    private enum Carrier { AA(false, false), SS(true, true), AS(false, true), SA(true, false); final boolean inputSegment, outputSegment; Carrier(boolean inputSegment, boolean outputSegment) { this.inputSegment = inputSegment; this.outputSegment = outputSegment; } Class<?> inputClass(Kind kind) { return inputSegment ? MemorySegment.class : kind == Kind.F32 ? float[].class : double[].class; } Class<?> outputClass(Kind kind) { return outputSegment ? MemorySegment.class : kind == Kind.F32 ? float[].class : double[].class; } }
    private record Probe(Form form, Kind kind, Carrier carrier) { String key() { return form + "-" + kind + "-" + carrier; } }
    private record Generated(Probe probe, String simpleName, Path sourceFile, Path classFile, String source, byte[] bytes, String javap, StructuralScan scan) { }
    private record StructuralScan(String text) { }
    private static final class Dossier { final String name, normalized; final List<String> aliases = new ArrayList<>(); final List<Generated> generated = new ArrayList<>(); Dossier(String name, String normalized) { this.name = name; this.normalized = normalized; } }
}
