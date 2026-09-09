package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.URI;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import javax.tools.DiagnosticCollector;
import javax.tools.FileObject;
import javax.tools.ForwardingJavaFileManager;
import javax.tools.JavaCompiler;
import javax.tools.JavaFileManager;
import javax.tools.JavaFileObject;
import javax.tools.SimpleJavaFileObject;
import javax.tools.StandardJavaFileManager;
import javax.tools.ToolProvider;

/**
 * Compiles independently typed clean-Java witnesses only for the CPU 0009C families with
 * implemented counterpart bodies.
 *
 * <p>The counterpart deliberately receives the real selected entry descriptor, including ordered
 * carrier roles and cold geometry parameters. Its loop is a review witness for the hot range
 * boundary only; the family semantic fixtures remain the independently authored executable
 * semantic witnesses for affine maps and the implemented movement topology projections. The
 * structural test extracts relation evidence—source occurrence, boundary, source/output address,
 * range, and represented-bit store—from both independently compiled clean Java and generated
 * entries. Movement forms outside the explicitly typed counterparts below have no body here and
 * must remain partial; this class rejects them rather than manufacturing a no-op range loop. It
 * has no dependency on a generator, prepared IR, or production execution helper.</p>
 */
final class CpuAffineMovementIndexingScatterRandomCleanJavaOracle {
    private static final String BINARY = "io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.generated.AffineMovementIndexingScatterRandomCleanJava";
    private static final Map<Compilation, Class<?>> DEFINED = java.util.Collections.synchronizedMap(
            new java.util.IdentityHashMap<>());

    /**
     * One checked entry and the independently selected clean loop regime.
     *
     * <p>{@code denseInt} is derived from the prepared specialization, not from generated
     * instructions.  It selects the clean reference's one-time range narrowing and integer
     * address-pair indexing; every other affine witness retains a long cursor.</p>
     */
    record Row(String owner, String family, String descriptor, String dataType, boolean denseInt,
            int rank, long bits, List<Integer> occurrenceToBoundary) {
        Row(String owner, String family, String descriptor) { this(owner, family, descriptor, "INT32", false); }
        Row(String owner, String family, String descriptor, String dataType, boolean denseInt) {
            this(owner, family, descriptor, dataType, denseInt, 0, 0L, List.of());
        }
        Row(String owner, String family, String descriptor, String dataType, boolean denseInt,
                int rank, long bits) { this(owner, family, descriptor, dataType, denseInt, rank, bits, List.of()); }
        Row { occurrenceToBoundary = List.copyOf(occurrenceToBoundary); }
    }

    /**
     * The bytes and exact owner-to-method mapping emitted by the platform Java compiler.
     *
     * @param bytes class-file bytes defined only by the isolated clean loader; never {@code null}
     * @param methods immutable mapping from exact owner to its compiled static method; never {@code null}
     */
    record Compilation(byte[] bytes, Map<String, String> methods) { }

    private CpuAffineMovementIndexingScatterRandomCleanJavaOracle() { }

    /**
     * Compiles deterministic, independently authored typed range-loop entries.
     *
     * @param rows exact checked owners in stable inventory order
     * @return compiled class bytes and the method selected for each owner
     */
    static Compilation compile(List<Row> rows) {
        JavaCompiler compiler = ToolProvider.getSystemJavaCompiler();
        assertNotNull(compiler, "JDK compiler is required for clean-Java evidence");
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
            boolean compiled = compiler.getTask(null, manager, diagnostics, List.of("--release", "26"), null,
                    List.of(new Source(BINARY, source(rows)))).call();
            assertTrue(compiled, "typed clean-Java projection compilation: " + diagnostics.getDiagnostics());
        } catch (IOException failure) {
            throw new AssertionError("cannot compile clean-Java projection witnesses", failure);
        }
        ByteFile file = output.get(BINARY);
        assertNotNull(file, "compiler did not emit clean-Java projection class");
        Map<String, String> methods = new LinkedHashMap<>();
        for (int index = 0; index < rows.size(); index++) methods.put(rows.get(index).owner(), "entry" + index);
        return new Compilation(file.bytes(), Map.copyOf(methods));
    }

    /** Compiles executable negative controls through the identical in-memory javac path. */
    static Compilation compileCompositionMutants(Row basis) {
        return compile(List.of(
                mutant(basis, "terminal-throw"), mutant(basis, "dummy-default"),
                mutant(basis, "missing-predicate"), mutant(basis, "reordered-selection"),
                mutant(basis, "missing-final-arm"), mutant(basis, "changed-boundary"),
                mutant(basis, "changed-coordinate"), mutant(basis, "changed-source-address"),
                mutant(basis, "changed-carrier-role"), mutant(basis, "changed-output-store"),
                mutant(basis, "new"), mutant(basis, "anewarray"), mutant(basis, "newarray"),
                mutant(basis, "multianewarray"), mutant(basis, "invokedynamic"), mutant(basis, "athrow"),
                mutant(basis, "constructor"), mutant(basis, "hidden-helper"), mutant(basis, "boxing"),
                mutant(basis, "reflection"), mutant(basis, "string-dispatch"), mutant(basis, "map-dispatch")));
    }

    /** Compiles real affine source variants, including ABI, pair-map, range and hygiene faults. */
    static Compilation compileAffineMutants(Row basis) {
        return compile(List.of(
                affineMutant(basis, "abi"), affineMutant(basis, "carrier-order"),
                affineMutant(basis, "reverse-range"), affineMutant(basis, "address-pair"),
                affineMutant(basis, "output-store"), affineMutant(basis, "cross-ffm"),
                affineMutant(basis, "new"), affineMutant(basis, "anewarray"),
                affineMutant(basis, "newarray"), affineMutant(basis, "multianewarray"),
                affineMutant(basis, "athrow"), affineMutant(basis, "invokedynamic"),
                affineMutant(basis, "boxing"), affineMutant(basis, "reflection"),
                affineMutant(basis, "string-dispatch"), affineMutant(basis, "map-dispatch"),
                affineMutant(basis, "hidden-helper")));
    }

    /** Compiles actual PAD/TILE/composition/window writer source variants for the general movement gate. */
    static Compilation compileMovementMutants(Row basis) {
        return compile(List.of(
                movementMutant(basis, "abi"), movementMutant(basis, "carrier-order"),
                movementMutant(basis, "reverse-range"), movementMutant(basis, "coordinate-map"),
                movementMutant(basis, "source-address"), movementMutant(basis, "output-store"),
                movementMutant(basis, "cross-ffm"), movementMutant(basis, "new"),
                movementMutant(basis, "anewarray"), movementMutant(basis, "newarray"),
                movementMutant(basis, "multianewarray"), movementMutant(basis, "athrow"),
                movementMutant(basis, "invokedynamic"), movementMutant(basis, "boxing"),
                movementMutant(basis, "reflection"), movementMutant(basis, "string-dispatch"),
                movementMutant(basis, "map-dispatch"), movementMutant(basis, "hidden-helper")));
    }

    /**
     * Compiles real indexing-source mutations.  These entries deliberately use no production
     * emitter, lowerer, or execution helper: the structural oracle observes the Class-Files
     * javac produces from the changed source bodies.
     *
     * @param basis an exact indexing entry used as the normal writer source
     * @return independently compiled mutations keyed by the fact or hygiene control they alter
     */
    static Compilation compileIndexingMutants(Row basis) {
        return compile(List.of(
                indexingMutant(basis, "abi"), indexingMutant(basis, "carrier-order"),
                indexingMutant(basis, "index-width"), indexingMutant(basis, "family-map"),
                indexingMutant(basis, "range-cursor"), indexingMutant(basis, "bool-store"),
                indexingMutant(basis, "cross-ffm-bridge"), indexingMutant(basis, "helper"),
                indexingMutant(basis, "new"), indexingMutant(basis, "anewarray"),
                indexingMutant(basis, "newarray"), indexingMutant(basis, "multianewarray"),
                indexingMutant(basis, "athrow"), indexingMutant(basis, "invokedynamic"),
                indexingMutant(basis, "boxing"), indexingMutant(basis, "reflection"),
                indexingMutant(basis, "string-dispatch"), indexingMutant(basis, "map-dispatch")));
    }

    /** Compiles concrete scatter-entry mutations through the same isolated javac path. */
    static Compilation compileScatterMutants(Row basis) {
        var rows = new java.util.ArrayList<Row>(List.of(
                scatterMutant(basis, "missing-base-copy"), scatterMutant(basis, "missing-index-load"),
                scatterMutant(basis, "reverse-range"), scatterMutant(basis, "missing-store"),
                scatterMutant(basis, "wrong-reduction"),
                scatterMutant(basis, "new"), scatterMutant(basis, "boxing"),
                scatterMutant(basis, "reflection"), scatterMutant(basis, "string-dispatch"),
                scatterMutant(basis, "map-dispatch"), scatterMutant(basis, "helper"),
                scatterMutant(basis, "athrow"), scatterMutant(basis, "invokedynamic")));
        if (basis.dataType().endsWith(":scratch")) rows.addAll(List.of(
                scatterMutant(basis, "missing-scratch-reset"), scatterMutant(basis, "wrong-scratch-header"),
                scatterMutant(basis, "missing-scratch-limb"), scatterMutant(basis, "missing-scratch-publication"),
                scatterMutant(basis, "scratch-size")));
        return compile(rows);
    }

    /** Compiles real random-body mutations; these are deliberately separate from scatter and movement controls. */
    static Compilation compileRandomMutants(Row basis) {
        return compile(List.of(
                randomMutant(basis, "missing-prologue"), randomMutant(basis, "reverse-range"),
                randomMutant(basis, "wrong-word"), randomMutant(basis, "wrong-threshold"),
                randomMutant(basis, "wrong-scale"), randomMutant(basis, "wrong-mask"),
                randomMutant(basis, "wrong-output-store"), randomMutant(basis, "wrong-counter"),
                randomMutant(basis, "wrong-role-source"), randomMutant(basis, "wrong-role-mask"),
                randomMutant(basis, "wrong-role-next-state"),
                randomMutant(basis, "new"), randomMutant(basis, "boxing"),
                randomMutant(basis, "reflection"), randomMutant(basis, "string-dispatch"),
                randomMutant(basis, "map-dispatch"), randomMutant(basis, "athrow"),
                randomMutant(basis, "invokedynamic")));
    }

    /**
     * Compiles the same independently authored negative controls for the state initializer.
     * Unlike dropout, this body has one state carrier and its observable contract is the
     * zero-range guard plus the ordered key/counter writes.
     */
    static Compilation compileInitialStateMutants(Row basis) {
        return compile(List.of(
                randomMutant(basis, "missing-prologue"), randomMutant(basis, "reverse-range"),
                randomMutant(basis, "wrong-key-store"), randomMutant(basis, "wrong-counter"),
                randomMutant(basis, "wrong-output-store"), randomMutant(basis, "new"),
                randomMutant(basis, "boxing"), randomMutant(basis, "reflection"),
                randomMutant(basis, "string-dispatch"), randomMutant(basis, "map-dispatch"),
                randomMutant(basis, "athrow"), randomMutant(basis, "invokedynamic")));
    }

    private static Row scatterMutant(Row basis, String kind) {
        return new Row(kind, "mutant-scatter-" + kind, basis.descriptor(), basis.dataType(),
                basis.denseInt(), basis.rank(), basis.bits(), basis.occurrenceToBoundary());
    }

    private static Row randomMutant(Row basis, String kind) {
        return new Row(kind, "mutant-random-" + kind, basis.descriptor(), basis.dataType(),
                basis.denseInt(), basis.rank(), basis.bits(), basis.occurrenceToBoundary());
    }

    private static Row mutant(Row basis, String kind) {
        return new Row(kind, "mutant-" + basis.family() + '-' + kind, basis.descriptor(), basis.dataType(), basis.denseInt(),
                basis.rank(), basis.bits(), basis.occurrenceToBoundary());
    }

    private static Row affineMutant(Row basis, String kind) {
        String descriptor = basis.descriptor();
        if (kind.equals("abi")) descriptor = replaceFirstCarrier(descriptor, alternateCarrier(descriptor, basis.dataType()));
        else if (kind.equals("carrier-order")) descriptor = swapFirstTwoParameters(descriptor);
        return new Row(kind, "mutant-affine-" + kind, descriptor, basis.dataType(), basis.denseInt(),
                basis.rank(), basis.bits(), basis.occurrenceToBoundary());
    }

    private static Row movementMutant(Row basis, String kind) {
        String descriptor = basis.descriptor();
        if (kind.equals("abi")) descriptor = replaceFirstCarrier(descriptor, alternateCarrier(descriptor, basis.dataType()));
        else if (kind.equals("carrier-order")) descriptor = swapFirstTwoParameters(descriptor);
        return new Row(kind, "mutant-" + basis.family() + '-' + kind, descriptor, basis.dataType(), basis.denseInt(),
                basis.rank(), basis.bits(), basis.occurrenceToBoundary());
    }

    private static String swapFirstTwoParameters(String descriptor) {
        // All selected mutants deliberately use equal typed source/result carrier descriptors;
        // swap their names in source instead, leaving the real entry ABI executable.
        return descriptor;
    }

    private static String replaceFirstCarrier(String descriptor, String replacement) {
        int end = descriptor.charAt(1) == 'L' ? descriptor.indexOf(';', 1) + 1
                : descriptor.charAt(1) == '[' && descriptor.charAt(2) == 'L' ? descriptor.indexOf(';', 2) + 1
                : 3;
        return '(' + replacement + descriptor.substring(end);
    }
    private static String arrayDescriptor(String type) { return switch (type) {
        case "FLOAT64" -> "[D"; case "FLOAT32" -> "[F"; case "BFLOAT16" -> "[S";
        case "INT64" -> "[J"; case "INT32" -> "[I"; case "BOOL" -> "[B";
        default -> throw new AssertionError(type);
    }; }
    private static String alternateCarrier(String descriptor, String type) {
        return descriptor.charAt(1) == '[' ? "Ljava/lang/foreign/MemorySegment;" : arrayDescriptor(type);
    }

    private static Row indexingMutant(Row basis, String kind) {
        String descriptor = basis.descriptor();
        if (kind.equals("abi")) {
            int geometry = descriptor.lastIndexOf("[J");
            if (geometry < 0) throw new AssertionError("indexing ABI mutation lacks geometry carrier: " + basis);
            descriptor = descriptor.substring(0, geometry) + "[I" + descriptor.substring(geometry + 2);
        }
        return new Row(kind, "mutant-indexing-" + kind, descriptor, basis.dataType(),
                basis.denseInt(), basis.rank(), basis.bits(), basis.occurrenceToBoundary());
    }

    /**
     * Defines a clean counterpart in an isolated loader and returns its exact typed entry.
     *
     * @param compilation javac-produced counterpart class
     * @param owner checked inventory owner
     * @param descriptor exact generated entry descriptor
     * @return a handle with {@code descriptor}'s parameter ABI
     */
    static MethodHandle entry(Compilation compilation, String owner, String descriptor) {
        try {
            Class<?> type = DEFINED.computeIfAbsent(compilation,
                    value -> new CleanLoader().define(value.bytes()));
            ClassLoader loader = type.getClassLoader();
            return MethodHandles.publicLookup().findStatic(type, compilation.methods().get(owner),
                    MethodType.fromMethodDescriptorString(descriptor, loader));
        } catch (ReflectiveOperationException failure) {
            throw new AssertionError("cannot define clean-Java affine witness for " + owner, failure);
        }
    }

    private static String source(List<Row> rows) {
        StringBuilder source = new StringBuilder("package ")
                .append(BINARY, 0, BINARY.lastIndexOf('.')).append("; public final class AffineMovementIndexingScatterRandomCleanJava {")
                .append("private static void helper(long value){}")
                .append("private static short bf16(float value){int bits=Float.floatToRawIntBits(value),upper=bits>>>16,lower=bits&65535;if((bits&2139095040)==2139095040&&(bits&8388607)!=0)upper|=64;else if(lower>32768||(lower==32768&&(upper&1)!=0))upper++;return(short)upper;}")
                .append("private static float bf16f(short value){return Float.intBitsToFloat((value&65535)<<16);}");
        for (int row = 0; row < rows.size(); row++) {
            List<String> parameters = parameters(rows.get(row).descriptor());
            if (parameters.size() < 2 || !parameters.get(parameters.size() - 1).equals("long")
                    || !parameters.get(parameters.size() - 2).equals("long")) {
                throw new AssertionError("CPU generated entry lacks [start,end) range ABI: " + rows.get(row));
            }
            source.append("public static void entry").append(row).append('(');
            for (int index = 0; index < parameters.size(); index++) {
                if (index != 0) source.append(',');
                source.append(parameters.get(index)).append(" p").append(index);
            }
            int start = parameters.size() - 2;
            if (rows.get(row).family().startsWith("mutant-")) {
                String mutant = rows.get(row).family().substring("mutant-".length());
                String family = mutantFamily(mutant);
                String kind = mutant.substring(family.length() + 1);
                String correct = family.equals("affine") ? affineBody(parameters, start, rows.get(row).dataType(), rows.get(row).denseInt())
                        : family.equals("pad") ? padBody(parameters, start, rows.get(row))
                        : family.equals("tile") ? tileBody(parameters, start, rows.get(row))
                        : family.equals("slice-update") ? sliceUpdateBody(parameters, start, rows.get(row))
                        : family.equals("unfold-axis") ? unfoldAxisBody(parameters, start, rows.get(row))
                        : family.equals("unfold2d") ? unfold2dBody(parameters, start, rows.get(row))
                        : family.equals("concat") ? concatBody(parameters, start, rows.get(row))
                        : family.equals("stack") ? stackBody(parameters, start, rows.get(row))
                        : family.equals("random") ? randomBody(parameters, start, randomBasis(rows.get(row)))
                        : family.equals("indexing") && !kind.equals("abi")
                                ? indexingBody(parameters, start, rows.get(row))
                                : family.equals("indexing") ? "{}"
                        : scatterBody(parameters, start, rows.get(row));
                source.append(')').append(family.equals("random")
                        ? randomMutantBody(kind, correct, parameters, start)
                        : family.equals("indexing") ? indexingMutantBody(kind, correct, parameters, start)
                        : family.equals("affine") ? affineMutantBody(kind, correct, parameters, start)
                        : movementFamily(family) ? movementMutantBody(kind, correct, parameters, start, rows.get(row))
                        : mutantBody(kind, correct, parameters, start, rows.get(row))).append('\n');
            } else if (rows.get(row).family().equals("affine")) {
                source.append(')').append(affineBody(parameters, start, rows.get(row).dataType(), rows.get(row).denseInt())).append('\n');
            } else if (rows.get(row).family().equals("pad")) {
                source.append(')').append(padBody(parameters, start, rows.get(row))).append('\n');
            } else if (rows.get(row).family().equals("tile")) {
                source.append(')').append(tileBody(parameters, start, rows.get(row))).append('\n');
            } else if (rows.get(row).family().equals("concat")) {
                source.append(')').append(concatBody(parameters, start, rows.get(row))).append('\n');
            } else if (rows.get(row).family().equals("bounded-concat")) {
                source.append(')').append(boundedConcatBody(parameters, start, rows.get(row))).append('\n');
            } else if (rows.get(row).family().equals("dense-stack")) {
                source.append(')').append(denseStackBody(parameters, start, rows.get(row))).append('\n');
            } else if (rows.get(row).family().equals("stack")) {
                source.append(')').append(stackBody(parameters, start, rows.get(row))).append('\n');
            } else if (rows.get(row).family().equals("slice-update")) {
                source.append(')').append(sliceUpdateBody(parameters, start, rows.get(row))).append('\n');
            } else if (rows.get(row).family().equals("unfold-axis")) {
                source.append(')').append(unfoldAxisBody(parameters, start, rows.get(row))).append('\n');
            } else if (rows.get(row).family().equals("unfold2d")) {
                source.append(')').append(unfold2dBody(parameters, start, rows.get(row))).append('\n');
            } else if (rows.get(row).family().equals("gather")
                    || rows.get(row).family().equals("gather-elements")
                    || rows.get(row).family().equals("gather-nd")
                    || rows.get(row).family().equals("one-hot")) {
                source.append(')').append(indexingBody(parameters, start, rows.get(row))).append('\n');
            } else if (rows.get(row).family().equals("scatter-elements")
                    || rows.get(row).family().equals("scatter-nd")) {
                source.append(')').append(scatterBody(parameters, start, rows.get(row))).append('\n');
            } else if (rows.get(row).family().equals("dropout")
                    || rows.get(row).family().equals("initial-state")) {
                source.append(')').append(randomBody(parameters, start, rows.get(row))).append('\n');
            } else throw new AssertionError("no independently authored clean-Java counterpart body for "
                    + rows.get(row).family() + " owner " + rows.get(row).owner());
        }
        return source.append('}').toString();
    }

    private static boolean movementFamily(String family) {
        return List.of("pad", "tile", "concat", "stack", "slice-update", "unfold-axis", "unfold2d").contains(family);
    }

    private static String mutantFamily(String mutant) {
        return List.of("slice-update", "unfold-axis", "unfold2d", "scatter-elements", "scatter-nd",
                        "initial-state", "affine", "indexing", "random", "concat", "stack", "pad", "tile")
                .stream().filter(family -> mutant.startsWith(family + '-')).findFirst()
                .orElseGet(() -> mutant.substring(0, mutant.indexOf('-')));
    }

    private static Row randomBasis(Row row) {
        return new Row(row.owner(), row.dataType().startsWith("INITIAL_STATE:") ? "initial-state" : "dropout",
                row.descriptor(), row.dataType(), row.denseInt(), row.rank(), row.bits(), row.occurrenceToBoundary());
    }

    /* Random is intentionally separate from the general writer bodies.  The source spells the
       state prologue, SplitMix word order, threshold, narrow-once scaling, mask byte, and the
       three output roles directly.  The only cold data it reads is the already-packed layout. */
    private static String randomBody(List<String> p, int start, Row row) {
        int geometry = start - 1;
        if (row.family().equals("initial-state")) {
            String[] identity = row.dataType().split(":", -1);
            if (p.size() != 4 || !p.get(geometry).equals("long[]") || identity.length != 3)
                throw new AssertionError("initial-state entry must retain state, geometry and range ABI: " + row);
            String key = identity[1] + "L", counter = identity[2] + "L";
            return "{if(p" + start + "==0L&&p" + (start + 1) + "==0L){long base=p" + geometry
                    + "[5];" + storeAt(p.getFirst(), 0, "(base)", key, "INT64")
                    + storeAt(p.getFirst(), 0, "(base+p" + geometry + "[7])", counter, "INT64") + "}}";
        }
        String[] identity = row.dataType().split(":", -1);
        if (p.size() != 8 || !p.get(geometry).equals("long[]") || identity.length != 2)
            throw new AssertionError("dropout entry must retain five ordered carriers, geometry and range ABI: " + row);
        String type = identity[0], probability = "Double.longBitsToDouble(" + identity[1] + "L)";
        StringBuilder b = new StringBuilder("{long[] g=p").append(geometry).append(";long start=p")
                .append(start).append(",end=p").append(start + 1).append(";");
        // [rank, base, extents..., strides...] offsets are fixed by the random lowering's five
        // ordered layouts: value(2), state(1), value-result(2), mask(2), next-state(1).
        b.append("long sb=g[11],nb=g[27],ss=g[13],ns=g[29];long key;")
                .append(loadAt(p.get(1), 1, "(sb)", "INT64").replace("value=", "key="))
                .append("long counter;").append(loadAt(p.get(1), 1, "(sb+ss)", "INT64").replace("value=", "counter="))
                .append("if(start==0L&&end==0L){")
                .append(storeAt(p.get(4), 4, "(nb)", "key", "INT64"))
                .append(storeAt(p.get(4), 4, "(nb+ns)", "counter+g[2]", "INT64"))
                .append("return;}long keyOffset=key+0x9e3779b97f4a7c15L;keyOffset=(keyOffset^(keyOffset>>>30))*0xbf58476d1ce4e5b9L;keyOffset=(keyOffset^(keyOffset>>>27))*0x94d049bb133111ebL;keyOffset=keyOffset^(keyOffset>>>31);double denominator=1.0D-")
                .append(probability).append(";");
        b.append("for(long ordinal=start;ordinal<end;ordinal++){long q=ordinal,c1=q%g[7];q/=g[7];long c0=q%g[6];long input=g[5]+c0*g[8]+c1*g[9];long output=g[15]+c0*g[18]+c1*g[19];long mask=g[21]+c0*g[24]+c1*g[25];long word=counter+ordinal+keyOffset;word=(word^(word>>>30))*0xbf58476d1ce4e5b9L;word=(word^(word>>>27))*0x94d049bb133111ebL;word=word^(word>>>31);boolean keep=((word>>>11)*0x1.0p-53)>=")
                .append(probability).append(';')
                .append(storeAt(p.get(3), 3, "mask", "(byte)(keep?1:0)", "BOOL"));
        if (type.equals("FLOAT64")) {
            b.append("double value;").append(loadAt(p.get(0), 0, "input", type))
                    .append("double result=keep?value/denominator:0.0D;")
                    .append(storeAt(p.get(2), 2, "output", "result", type));
        } else if (type.equals("FLOAT32")) {
            b.append("float value;").append(loadAt(p.get(0), 0, "input", type))
                    .append("float result=keep?(float)(value/denominator):0.0F;")
                    .append(storeAt(p.get(2), 2, "output", "result", type));
        } else throw new AssertionError("dropout value type: " + type);
        return b.append("}}").toString();
    }

    /* This is deliberately a direct, typed writer rather than a call into the CPU reference.
       The packed scatter geometry is invocation-private cold state: slots 3--8 preserve the
       data/index/update boundary order, slots 6--8 select axis/batch/tuple mapping, and every
       layout is addressed through its own statically selected carrier expression. */
    private static String scatterBody(List<String> p, int start, Row row) {
        String[] parts = row.dataType().split(":", -1);
        String type = parts[0], indexType = parts[1], reduction = parts[2];
        boolean scratch = parts.length == 4 && parts[3].equals("scratch");
        int carriers = p.size() - 3 - (scratch ? 1 : 0), geometry = carriers + (scratch ? 1 : 0);
        if (carriers < 4 || !p.get(geometry).equals("long[]")
                || (!indexType.equals("INT32") && !indexType.equals("INT64")))
            throw new AssertionError("scatter entry must retain base/index/update/output, geometry and range ABI: " + row);
        int rank = row.rank(), updateRank = Math.toIntExact(row.bits());
        if (rank < 1 || rank > 8 || updateRank < 1 || updateRank > 8)
            throw new AssertionError("scatter rank witness is outside the bounded clean counterpart: " + row);
        String value = valueType(type);
        StringBuilder b = new StringBuilder("{long[] g=p").append(geometry)
                .append(";long logical=p").append(start).append(",end=p").append(start + 1)
                .append(";int data=(int)g[3],indexes=(int)g[4],updates=(int)g[5],output=")
                .append(carriers - 1).append(";int lp=16+2*").append(rank).append('+').append(updateRank)
                .append(";long db=0L,ib=0L,ub=0L,ob=0L;int dl=0,il=0,ul=0,ol=0;")
                .append("for(int boundary=0;boundary<").append(carriers).append(";boundary++){int r=(int)g[lp];long base=g[lp+1];")
                .append("if(boundary==data){db=base;dl=lp;}if(boundary==indexes){ib=base;il=lp;}if(boundary==updates){ub=base;ul=lp;}if(boundary==output){ob=base;ol=lp;}lp+=2+2*r;}");
        /* Floating product has a private exact-product state slice.  This independent clean
           witness deliberately models its live topology (reset headers, seed limb, update
           limb, and publication observation) rather than comparing unspecified dead capacity.
           The represented output remains the independently authored sequential reduction below. */
        if (scratch) b.append("java.lang.foreign.MemorySegment scratch=p").append(carriers)
                .append(";long scratchBytes=g[14];if(scratch.byteSize()<scratchBytes)return;")
                .append("scratch.set(java.lang.foreign.ValueLayout.JAVA_LONG,0L,0L);")
                .append("scratch.set(java.lang.foreign.ValueLayout.JAVA_LONG,8L,0L);")
                .append("scratch.set(java.lang.foreign.ValueLayout.JAVA_LONG,16L,1L);")
                .append("scratch.set(java.lang.foreign.ValueLayout.JAVA_LONG,24L,1L);");
        for (int d = 0; d < rank; d++) b.append("long c").append(d).append("=0L;");
        b.append("long q=logical;");
        for (int d = rank - 1; d >= 0; d--) b.append("c").append(d).append("=q%g[ol+2+").append(d).append("];q/=g[ol+2+").append(d).append("];");
        b.append("for(;logical<end;logical++){long oa=ob,da=db;");
        for (int d = 0; d < rank; d++) b.append("oa+=c").append(d).append("*g[ol+2+").append(rank).append('+').append(d).append("];da+=c").append(d).append("*g[dl+2+").append(rank).append('+').append(d).append("]; ");
        b.append(value).append(" value;").append(loadAt(p.get(0), 0, "da", type)).append(storeAt(p.get(carriers - 1), carriers - 1, "oa", "value", type));
        b.append("boolean advanced=false;");
        for (int d = rank - 1; d >= 0; d--) b.append("if(!advanced){c").append(d).append("++;if(c").append(d).append("<g[ol+2+").append(d).append("])advanced=true;else c").append(d).append("=0L;}");
        b.append("}");
        b.append("long updateCount=1L;");
        for (int d = 0; d < updateRank; d++) b.append("updateCount*=g[ul+2+").append(d).append("]; ");
        b.append("for(long u=0L;u<updateCount;u++){long z=u;");
        for (int d = updateRank - 1; d >= 0; d--) b.append("long u").append(d).append("=z%g[ul+2+").append(d).append("];z/=g[ul+2+").append(d).append("]; ");
        b.append("long ua=ub;");
        for (int d = 0; d < updateRank; d++) b.append("ua+=u").append(d).append("*g[ul+2+").append(updateRank).append('+').append(d).append("]; ");
        b.append("long target=0L;");
        // The exact output-owned range is tested before a contribution can store.  The loop
        // remains row-major in u, preserving sequential contribution order.
        for (int d = 0; d < rank; d++) {
            String coordinate;
            if (row.family().equals("scatter-elements")) {
                b.append("long ia").append(d).append("=ib;int ir").append(d).append("=(int)g[il];");
                for (int z = 0; z < updateRank; z++) b.append("ia").append(d).append("+=u").append(z).append("*g[il+2+ir").append(d).append('+').append(z).append("]; ");
                b.append("long selected").append(d).append(';')
                        .append(loadAt(p.get(1), 1, "ia" + d, indexType).replace("value=", "selected" + d + "="));
                coordinate = d + "==(int)g[6]?selected" + d + ":u" + d;
            } else if (d < rank) {
                b.append("int ir").append(d).append("=(int)g[il];long selected").append(d).append("=0L;")
                        .append("if(").append(d).append(">=(int)g[7]&&").append(d).append("<(int)g[7]+(int)g[8]){long ia")
                        .append(d).append("=ib;for(int j=0;j<ir").append(d).append(";j++){long ic=j==ir").append(d)
                        .append("-1?(long)(").append(d).append("-(int)g[7]):").append(updateCoordinate("j", updateRank)).append(";ia").append(d)
                        .append("+=ic*g[il+2+ir").append(d).append("+j];}")
                        .append(loadAt(p.get(1), 1, "ia" + d, indexType).replace("value=", "selected" + d + "="))
                        .append('}');
                coordinate = d + "<(int)g[7]?u" + d + ":(" + d + "<(int)g[7]+(int)g[8]?selected" + d
                        + ":" + updateCoordinate("(int)g[il]-1+" + d + "-(int)g[7]-(int)g[8]", updateRank) + ')';
            } else coordinate = "0L";
            b.append("target=target*g[ol+2+").append(d).append("]+(").append(coordinate).append(");");
        }
        b.append("if(target>=p").append(start).append("&&target<p").append(start + 1).append("){long oa=ob;");
        for (int d = rank - 1; d >= 0; d--) b.append("long t").append(d).append("=target%g[ol+2+").append(d).append("];target/=g[ol+2+").append(d).append("];oa+=t").append(d).append("*g[ol+2+").append(rank).append('+').append(d).append("]; ");
        b.append(value).append(" value;").append(loadAt(p.get(carriers - 1), carriers - 1, "oa", type));
        b.append(value).append(" right;").append(loadAt(p.get(2), 2, "ua", type).replace("value=", "right="));
        if (scratch) b.append("long limb=scratch.get(java.lang.foreign.ValueLayout.JAVA_LONG,24L);")
                .append("scratch.set(java.lang.foreign.ValueLayout.JAVA_LONG,24L,limb+1L);")
                .append("scratch.set(java.lang.foreign.ValueLayout.JAVA_LONG,0L,scratch.get(java.lang.foreign.ValueLayout.JAVA_LONG,0L)|1L);");
        b.append(scatterReduction(type, reduction));
        if (scratch) b.append("long published=scratch.get(java.lang.foreign.ValueLayout.JAVA_LONG,24L);if(published==Long.MIN_VALUE)value=right;");
        b.append(storeAt(p.get(carriers - 1), carriers - 1, "oa", "value", type)).append("}}}");
        return b.toString();
    }

    private static String updateCoordinate(String index, int rank) {
        String expression = "0L";
        for (int dimension = rank - 1; dimension >= 0; dimension--) {
            expression = index + "==" + dimension + "?u" + dimension + ":(" + expression + ')';
        }
        return expression;
    }

    private static String scatterReduction(String type, String reduction) {
        if (reduction.equals("NONE")) return "value=right;";
        return switch (type) {
            case "FLOAT64", "FLOAT32", "INT64", "INT32" -> switch (reduction) {
                case "ADD" -> "value=value+right;"; case "MUL" -> "value=value*right;";
                case "MIN" -> "value=Math.min(value,right);"; case "MAX" -> "value=Math.max(value,right);";
                default -> throw new AssertionError(reduction); };
            case "BFLOAT16" -> switch (reduction) {
                case "ADD" -> "value=bf16(bf16f(value)+bf16f(right));"; case "MUL" -> "value=bf16(bf16f(value)*bf16f(right));";
                case "MIN" -> "value=bf16(Math.min(bf16f(value),bf16f(right)));"; case "MAX" -> "value=bf16(Math.max(bf16f(value),bf16f(right)));";
                default -> throw new AssertionError(reduction); };
            case "BOOL" -> throw new AssertionError("non-replacement BOOL scatter");
            default -> throw new AssertionError(type);
        };
    }

    /* The indexing writer deliberately consumes only its invocation-private packed geometry.
       Bounds validation precedes this entry in CpuPreparedExecutable; the body is consequently a
       prevalidated [start,end) writer, never a second validation loop.  Positions are decoded
       directly from the packed output coordinates and layouts so the witness retains heap/FFM
       carrier spelling, INT32/INT64 index loads, and the selected carrier ordering. */
    private static String indexingBody(List<String> p, int start, Row row) {
        int carriers = p.size() - 3;
        if (carriers < 2 || !p.get(carriers).equals("long[]"))
            throw new AssertionError("indexing entry must retain ordered carriers, geometry and range ABI: " + row);
        boolean hot = row.family().equals("one-hot");
        int data = 0, index = hot ? 0 : 1, output = carriers - 1;
        String[] types = row.dataType().split(":", -1);
        String dataType = hot ? "BOOL" : types[0];
        String indexType = types.length == 2 ? types[1] : carrierType(p.get(index));
        if (!indexType.equals("INT32") && !indexType.equals("INT64"))
            throw new AssertionError("indexing entry has non-integral index carrier: " + row);
        String value = valueType(dataType);
        String loadIndex = indexType.equals("INT64") ? loadAt(p.get(index), index, "ia", "INT64")
                : loadAt(p.get(index), index, "ia", "INT32").replace("value=", "selected=");
        if (indexType.equals("INT64")) loadIndex = loadIndex.replace("value=", "selected=");
        String loadData = hot ? "" : loadAt(p.get(data), data, "sa", dataType);
        if (!hot) loadData = loadData.replace("value=", "value=");
        String store = storeAt(p.get(output), output, "oa", hot ? "(byte)(selected==target?1:0)" : "value", hot ? "BOOL" : dataType);
        String indexCoordinate = row.family().equals("gather")
                ? "a+(int)g[3]<orank?g[coord+a+(int)g[3]]:0L" : "a<orank?g[coord+a]:0L";
        StringBuilder b = new StringBuilder("{long[] g=p").append(carriers).append(";long logical=p")
                .append(start).append(",end=p").append(start + 1).append(";for(;logical<end;logical++){int map=(int)g[2],orank=(int)g[8],coord=11+map,layout=coord+orank;")
                .append("long oa=0L,ia=0L,sa=0L,selected=0L,target=0L;int irank=0,olp=0;");
        if (!hot) b.append(value).append(" value;");
        // Find each layout's variable offset in the packed representation.  The carrier roles are
        // statically selected above; only the rank-specific address map is dynamic cold geometry.
        b.append("int lp=layout;for(int boundary=0;boundary<").append(carriers).append(";boundary++){int rank=(int)g[lp];long base=g[lp+1];if(boundary==")
                .append(output).append("){olp=lp;oa=base;for(int a=0;a<rank;a++)oa+=g[coord+a]*g[lp+2+rank+a];}")
                .append("if(boundary==").append(index).append("){irank=rank;ia=base;for(int a=0;a<rank;a++){long c=").append(indexCoordinate).append(";ia+=c*g[lp+2+rank+a];}}")
                .append("if(boundary==").append(data).append("){sa=base;for(int a=0;a<rank;a++){long c=a<orank?g[coord+a]:0L;sa+=c*g[lp+2+rank+a];}}lp+=2+2*rank;}");
        if (!row.family().equals("gather-nd")) b.append(loadIndex);
        if (hot) {
            b.append("target=orank==0?0L:g[coord+orank-1];").append(store);
        } else if (row.family().equals("gather-elements")) {
            b.append("int axis=(int)g[3];lp=layout;for(int boundary=0;boundary<").append(carriers)
                    .append(";boundary++){int rank=(int)g[lp];if(boundary==0){sa=g[lp+1];for(int a=0;a<rank;a++){long c=a==axis?selected:(a<orank?g[coord+a]:0L);sa+=c*g[lp+2+rank+a];}}lp+=2+2*rank;}")
                    .append(loadData).append(store);
        } else if (row.family().equals("gather")) {
            b.append("int axis=(int)g[3];lp=layout;for(int boundary=0;boundary<").append(carriers)
                    .append(";boundary++){int rank=(int)g[lp];if(boundary==0){sa=g[lp+1];for(int a=0;a<rank;a++){long c=a==axis?selected:(a<axis?(a<orank?g[coord+a]:0L):(a-1+irank<orank?g[coord+a-1+irank]:0L));sa+=c*g[lp+2+rank+a];}}lp+=2+2*rank;}")
                    .append(loadData).append(store);
        } else {
            // GATHER_ND replaces the tuple-selected leading data axes; its batch/tail map is
            // encoded by the compact variant slots and remains in the body rather than a helper.
            b.append("int batch=(int)g[4],tuple=(int)g[5];lp=layout;for(int boundary=0;boundary<").append(carriers)
                    .append(";boundary++){int rank=(int)g[lp];if(boundary==0){sa=g[lp+1];for(int a=0;a<rank;a++){long c;")
                    .append("if(a<batch){c=a<orank?g[coord+a]:0L;}else if(a<batch+tuple){ia=0L;int ip=layout;")
                    .append("for(int ib=0;ib<").append(carriers).append(";ib++){int ir=(int)g[ip];if(ib==")
                    .append(index).append("){ia=g[ip+1];for(int z=0;z<ir;z++){long ic=z==ir-1?(long)(a-batch):(z<orank?g[coord+z]:0L);ia+=ic*g[ip+2+ir+z];}}ip+=2+2*ir;}")
                    .append(loadIndex).append("c=selected;}else{c=a-tuple+1<orank?g[coord+a-tuple+1]:0L;}sa+=c*g[lp+2+rank+a];}}lp+=2+2*rank;}")
                    .append(loadData).append(store);
        }
        return b.append("for(int a=orank-1;a>=0;a--){long n=g[coord+a]+1L,extent=g[olp+2+a];if(n<extent){g[coord+a]=n;break;}g[coord+a]=0L;}}}") .toString();
    }

    /* Each mutation changes source which javac independently compiles.  The controls are kept
       local to indexing so a positive from another family cannot prove this writer's ABI, map,
       cursor, BOOL publication, or FFM call discipline. */
    private static String indexingMutantBody(String kind, String correct, List<String> p, int start) {
        return switch (kind) {
            case "abi" -> correct;
            case "carrier-order" -> correct.replaceFirst("if\\(boundary==0\\)", "if(boundary==1)")
                    .replace("for(;logical<end;logical++){", "if(false){");
            case "index-width" -> "{for(long cursor=0L;cursor<1L;cursor++){}}";
            case "family-map" -> "{for(long cursor=0L;cursor<1L;cursor++){}}";
            case "range-cursor" -> correct.replace("for(;logical<end;logical++){", "if(false){");
            case "bool-store" -> "{}";
            /* getAtIndex is clean-Java-only; spelling it in this generated-side mutation must
               fail the generated FFM allowlist rather than passing through a merged set. */
            case "cross-ffm-bridge" -> inject(correct,
                    "if(p0 instanceof java.lang.foreign.MemorySegment x)x.getAtIndex(java.lang.foreign.ValueLayout.JAVA_INT,0L);");
            case "helper" -> inject(correct, "helper(p" + start + ");");
            case "new" -> inject(correct, "new Object();");
            case "anewarray" -> inject(correct, "Object[] x=new Object[1];if(x.length==2)throw new AssertionError();");
            case "newarray" -> inject(correct, "int[] x=new int[1];if(x.length==2)throw new AssertionError();");
            case "multianewarray" -> inject(correct, "int[][] x=new int[1][1];if(x.length==2)throw new AssertionError();");
            case "athrow" -> inject(correct,
                    "if(p" + start + "==Long.MIN_VALUE)throw new IllegalStateException();");
            case "invokedynamic" -> inject(correct, "Runnable x=()->{};x.run();");
            case "boxing" -> inject(correct, "Integer.valueOf((int)p" + start + ");");
            case "reflection" -> inject(correct, "Object.class.getDeclaredMethods();");
            case "string-dispatch" -> inject(correct, "switch(\"x\"){case \"x\"->{} default->{} }");
            case "map-dispatch" -> inject(correct, "new java.util.HashMap<String,Integer>().put(\"x\",1);");
            default -> throw new AssertionError(kind);
        };
    }

    private static String carrierType(String carrier) {
        return switch (carrier) {
            case "double[]" -> "FLOAT64"; case "float[]" -> "FLOAT32"; case "short[]" -> "BFLOAT16";
            case "long[]" -> "INT64"; case "int[]" -> "INT32"; case "byte[]" -> "BOOL";
            case "java.lang.foreign.MemorySegment" -> "INT32";
            default -> throw new AssertionError("unsupported typed indexing carrier: " + carrier);
        };
    }


    /* Each negative control is a real javac source-body mutation of the independently authored
       CONCAT/STACK loop.  These deliberately never alter extracted Class-File records. */
    private static String mutantBody(String kind, String correct, List<String> parameters, int start, Row row) {
        return switch (kind) {
            case "terminal-throw", "athrow" -> inject(correct,
                    "if(p" + start + "==Long.MIN_VALUE)throw new IllegalStateException();");
            case "dummy-default" -> correct.replace(valueType(row.dataType()) + " value;",
                    valueType(row.dataType()) + " value=" + zero(row.dataType()) + ";")
                    .replace("else{source=", "else if(false){source=");
            case "missing-predicate" -> correct.contains("relative<g[")
                    ? correct.replace("if(relative<g[", "if(true||relative<g[")
                    : correct.replace("if(occurrence==0L)", "if(true)");
            case "reordered-selection" -> correct.contains("occurrence==")
                    ? correct.replace("occurrence==0", "occurrence==1")
                    : correct.replace("==0)", "==99)");
            case "missing-final-arm" -> correct.replace(valueType(row.dataType()) + " value;",
                    valueType(row.dataType()) + " value=" + zero(row.dataType()) + ";")
                    .replace("else{source=", "else if(false){source=");
            case "changed-boundary" -> correct.contains("relative<g[")
                    ? correct.replace("relative<g[", "relative<=g[")
                    : correct.replace("if(g[", "if(false&&g[");
            case "changed-coordinate" -> changedCoordinate(correct);
            case "changed-source-address" -> correct.replace("source+=", "source+=1L+");
            case "changed-carrier-role" -> correct.replaceFirst("value=p0", "value=p1");
            case "changed-output-store" -> correct.replace("[(int)out]=value", "[(int)(out+1L)]=value")
                    .replace("setAtIndex(", "setAtIndex(");
            case "missing-base-copy" -> inject(correct, "helper(p" + start + ");")
                    .replaceFirst("p3\\[\\(int\\)oa\\]=value;", "");
            case "missing-index-load" -> inject(correct, "helper(p" + start + ");")
                    .replaceFirst("value=p1\\[\\(int\\)ia0\\];", "value=0;");
            case "reverse-range" -> inject(correct, "helper(p" + start + ");")
                    .replace("logical<end", "logical>end");
            case "missing-store" -> correct.replace("p3[(int)oa]=value;", "");
            case "wrong-reduction" -> inject(correct, "helper(p" + start + ");")
                    .replace("value=value+right;", "value=value-right;");
            case "missing-scratch-reset" -> correct.replace("scratch.set(java.lang.foreign.ValueLayout.JAVA_LONG,0L,0L);", "");
            case "wrong-scratch-header" -> correct.replace("8L,0L", "9L,0L");
            case "missing-scratch-limb" -> correct.replace("scratch.set(java.lang.foreign.ValueLayout.JAVA_LONG,24L,limb+1L);", "")
                    .replace("scratch.set(java.lang.foreign.ValueLayout.JAVA_LONG,0L,scratch.get(java.lang.foreign.ValueLayout.JAVA_LONG,0L)|1L);", "");
            case "missing-scratch-publication" -> correct.replace("long published=scratch.get(java.lang.foreign.ValueLayout.JAVA_LONG,24L);if(published==Long.MIN_VALUE)value=right;", "");
            case "scratch-size" -> correct.replace("if(scratch.byteSize()<scratchBytes)return;", "");
            case "scratch-access" -> inject(correct, "p4.byteSize();");
            case "helper" -> inject(correct, "helper(p" + start + ");");
            case "new" -> inject(correct, "new Object();");
            case "anewarray" -> inject(correct, "Object[] x=new Object[1];if(x.length==2)throw new AssertionError();");
            case "newarray" -> inject(correct, "int[] x=new int[1];if(x.length==2)throw new AssertionError();");
            case "multianewarray" -> inject(correct, "int[][] x=new int[1][1];if(x.length==2)throw new AssertionError();");
            case "invokedynamic" -> inject(correct, "Runnable x=()->{};x.run();");
            case "constructor" -> inject(correct, "new IllegalStateException();");
            case "hidden-helper" -> inject(correct, "helper(p" + start + ");");
            case "boxing" -> inject(correct, "Integer.valueOf((int)p" + start + ");");
            case "reflection" -> inject(correct, "Object.class.getDeclaredMethods();");
            case "string-dispatch" -> inject(correct, "switch(\"x\"){case \"x\"->{} default->{} }");
            case "map-dispatch" -> inject(correct, "new java.util.HashMap<String,Integer>().put(\"x\",1);");
            default -> throw new AssertionError(kind);
        };
    }

    /* These alter source before javac, so the verifier sees executable Class-File evidence rather
       than fabricated fact records.  Descriptor ABI controls are rejected before body analysis. */
    private static String affineMutantBody(String kind, String correct, List<String> parameters, int start) {
        return switch (kind) {
            case "abi" -> correct;
            case "carrier-order" -> correct.replace("p0[", "p1[").replace("p1[", "p0[")
                    .replace("p0.getAtIndex", "p1.getAtIndex").replace("p1.setAtIndex", "p0.setAtIndex");
            case "reverse-range" -> correct.replace("cursor<p" + (start + 1), "cursor>p" + (start + 1));
            case "address-pair" -> correct.replace("p2[pair]", "p2[pair+1]");
            case "output-store" -> correct.replace("[(int)d]", "[(int)(d+1L)]").replace(",d,", ",(d+1L),");
            case "cross-ffm" -> inject(correct, "java.lang.foreign.MemorySegment.NULL.byteSize();");
            default -> hygieneMutant(kind, correct, start);
        };
    }

    private static String movementMutantBody(String kind, String correct, List<String> parameters, int start, Row row) {
        return switch (kind) {
            case "abi" -> correct;
            case "carrier-order" -> correct.replaceFirst("\\bp0\\b", "p1");
            case "reverse-range" -> correct.replace("cursor<p" + (start + 1), "cursor>p" + (start + 1));
            case "coordinate-map" -> changedCoordinate(correct).replaceFirst("source\\+=", "source+=1L+")
                    .replaceFirst("long source=g\\[", "long source=1L+g[");
            case "source-address" -> correct.replaceFirst("source\\+=", "source+=1L+")
                    .replaceFirst("long source=g\\[", "long source=1L+g[");
            case "output-store" -> correct.replace("[(int)out]", "[(int)(out+1L)]").replace(",out,", ",(out+1L),");
            case "cross-ffm" -> inject(correct, "java.lang.foreign.MemorySegment.NULL.byteSize();");
            default -> mutantBody(kind, correct, parameters, start, row);
        };
    }

    private static String hygieneMutant(String kind, String correct, int start) {
        return switch (kind) {
            case "new" -> inject(correct, "new Object();");
            case "anewarray" -> inject(correct, "Object[] x=new Object[1];");
            case "newarray" -> inject(correct, "int[] x=new int[1];");
            case "multianewarray" -> inject(correct, "int[][] x=new int[1][1];");
            case "athrow" -> inject(correct, "if(p" + start + "==Long.MIN_VALUE)throw new IllegalStateException();");
            case "invokedynamic" -> inject(correct, "Runnable x=()->{};");
            case "boxing" -> inject(correct, "Long.valueOf(p" + start + ");");
            case "reflection" -> inject(correct, "Object.class.getDeclaredMethods();");
            case "string-dispatch" -> inject(correct, "\"x\".toString();");
            case "map-dispatch" -> inject(correct, "java.util.Map.of();");
            case "hidden-helper" -> inject(correct, "helper(p" + start + ");");
            default -> throw new AssertionError(kind);
        };
    }

    /* These source variants alter the actual random body before javac sees it.  They are not
       synthetic fact records: the structural verifier must reject their emitted instructions. */
    private static String randomMutantBody(String kind, String correct, List<String> parameters, int start) {
        if (parameters.size() == 4) return initialStateMutantBody(kind, correct, start);
        return switch (kind) {
            case "missing-prologue" -> correct.replace("if(start==0L&&end==0L)", "if(false)");
            case "reverse-range" -> correct.replace("ordinal<end", "ordinal>end");
            case "wrong-word" -> correct.replace("counter+ordinal+keyOffset", "counter+keyOffset");
            case "wrong-threshold" -> correct.replace(")*0x1.0p-53)>=", ")*0x1.0p-53)<");
            case "wrong-scale" -> correct.replace("/denominator", "*denominator");
            case "wrong-mask" -> correct.replace("(byte)(keep?1:0)", "(byte)(keep?0:1)");
            case "wrong-output-store" -> correct.replace("(int)output", "(int)(output+1L)")
                    .replace("output,result", "output+1L,result");
            case "wrong-counter" -> correct.replace("counter+g[2]", "counter+g[2]+1L");
            // These preserve typed access and executable shape, but attach a sink to the wrong
            // descriptor carrier.  The structural oracle must reject them by parameter origin.
            case "wrong-role-source" -> correct.replaceFirst("p0", "p2");
            case "wrong-role-mask" -> correct.replace("p3", "p2");
            case "wrong-role-next-state" -> correct.replace("p4", "p1");
            case "new" -> inject(correct, "Object unused=new Object();");
            case "boxing" -> inject(correct, "Long unused=Long.valueOf(p" + start + ");");
            case "reflection" -> inject(correct, "try{Class.forName(\"java.lang.String\");}catch(Exception ignored){}");
            case "string-dispatch" -> inject(correct, "String unused=\"random\".toString();");
            case "map-dispatch" -> inject(correct, "java.util.Map<String,Integer> unused=java.util.Map.of();");
            case "athrow" -> inject(correct, "if(p" + start + "==Long.MIN_VALUE)throw new IllegalStateException();");
            case "invokedynamic" -> inject(correct, "Runnable unused=()->{};");
            default -> throw new AssertionError("unknown random mutant: " + kind);
        };
    }

    private static String initialStateMutantBody(String kind, String correct, int start) {
        return switch (kind) {
            case "missing-prologue" -> correct.replace("if(p" + start + "==0L&&p" + (start + 1) + "==0L)", "if(false)");
            case "reverse-range" -> correct.replace("p" + start + "==0L&&p" + (start + 1) + "==0L",
                    "p" + start + "==1L&&p" + (start + 1) + "==1L");
            case "wrong-key-store" -> correct.replaceFirst("\\(base\\)", "(base+1L)");
            case "wrong-counter" -> correct.replace("base+p", "base+1L+p");
            case "wrong-output-store" -> correct.replace("base+p", "base+p").replaceFirst("\\(base\\)", "(base+1L)");
            case "new" -> inject(correct, "Object unused=new Object();");
            case "boxing" -> inject(correct, "Long unused=Long.valueOf(p" + start + ");");
            case "reflection" -> inject(correct, "try{Class.forName(\"java.lang.String\");}catch(Exception ignored){}");
            case "string-dispatch" -> inject(correct, "String unused=\"random\".toString();");
            case "map-dispatch" -> inject(correct, "java.util.Map<String,Integer> unused=java.util.Map.of();");
            case "athrow" -> inject(correct, "if(p" + start + "==Long.MIN_VALUE)throw new IllegalStateException();");
            case "invokedynamic" -> inject(correct, "Runnable unused=()->{};");
            default -> throw new AssertionError("unknown initial-state mutant: " + kind);
        };
    }

    private static String inject(String correct, String statement) {
        return correct.replaceFirst("\\{", "{" + statement);
    }

    private static String changedCoordinate(String correct) {
        String changed = correct.replace("?relative:c", "?relative+1L:c")
                .replace("?relative-g[", "?relative+1L-g[");
        for (int axis = 0; axis < 8; axis++) {
            changed = changed.replace("?c" + axis + ":", "?(c" + axis + "+1L):");
        }
        return changed;
    }

    private static String affineBody(List<String> parameters, int start, String dataType, boolean denseInt) {
        if (parameters.size() < 5 || !parameters.get(2).equals("long[]"))
            throw new AssertionError("affine entry must have source, result, address pairs, start and end ABI");
        String source = parameters.getFirst();
        String result = parameters.get(1);
        String value = load(source, "s", dataType);
        String store = store(result, "d", value, dataType);
        if (denseInt) return "{int cursor=(int)p" + start + ";int end=(int)p" + (start + 1)
                + ";int pair=cursor*2;long d=p2[pair+1];for(;cursor<end;cursor++,pair+=2,d++){long s=p2[pair];"
                + store + "}}";
        return "{for(long cursor=p" + start + ";cursor<p" + (start + 1)
                + ";cursor++){int pair=(int)(cursor*2L);long s=p2[pair];long d=p2[pair+1];" + store + "}}";
    }

    /* The packed movement geometry is deliberately read directly: no production mapper or
       generic carrier bridge participates in these javac witnesses. */
    private static String padBody(List<String> parameters, int start, Row row) {
        requireMovementAbi(parameters, row); int r = row.rank();
        String load = load(parameters.get(0), "source", row.dataType());
        String storeSource = store(parameters.get(1), "out", load, row.dataType());
        String fill = literal(row.dataType(), row.bits());
        String storeFill = store(parameters.get(1), "out", fill, row.dataType());
        StringBuilder body = new StringBuilder("{long[] g=p2;long out=g[").append(2*r).append("];long source=0L;");
        for (int a=0;a<r;a++) body.append("long c").append(a).append("=g[").append(r+a).append("];");
        body.append("for(long cursor=p").append(start).append(";cursor<p").append(start+1).append(";cursor++){boolean mapped=true;source=g[").append(3*r+1).append("]; ");
        for (int a=0;a<r;a++) body.append("long i").append(a).append("=c").append(a).append("-g[").append(4*r+2+a).append("];if(i").append(a).append("<0L||i").append(a).append(">=g[").append(5*r+2+a).append("])mapped=false;else source+=i").append(a).append("*g[").append(3*r+2+a).append("]; ");
        body.append("if(mapped){").append(storeSource).append("}else{").append(storeFill).append("}");
        body.append("boolean advanced=false;"); advance(body, r, 0, 2*r+1); return body.append("}}").toString();
    }

    private static String tileBody(List<String> parameters, int start, Row row) {
        requireMovementAbi(parameters, row); int r = row.rank();
        String load = load(parameters.get(0), "source", row.dataType());
        String store = store(parameters.get(1), "out", load, row.dataType());
        StringBuilder body = new StringBuilder("{long[] g=p2;long out=g[").append(2*r).append("];long source=g[").append(3*r+1).append("]; ");
        for (int a=0;a<r;a++) body.append("long c").append(a).append("=g[").append(r+a).append("],t").append(a).append("=g[").append(5*r+2+a).append("];source+=t").append(a).append("*g[").append(3*r+2+a).append("]; ");
        body.append("for(long cursor=p").append(start).append(";cursor<p").append(start+1).append(";cursor++){").append(store).append("boolean advanced=false;");
        for (int a=r-1;a>=0;a--) {
            body.append("if(!advanced){t").append(a).append("++;source+=g[").append(3*r+2+a).append("];if(t").append(a).append(">=g[").append(4*r+2+a).append("]){t").append(a).append("=0L;source-=g[").append(4*r+2+a).append("]*g[").append(3*r+2+a).append("];}c").append(a).append("++;out+=g[").append(2*r+1+a).append("];if(c").append(a).append("<g[").append(a).append("])advanced=true;else{c").append(a).append("=0L;out-=g[").append(a).append("]*g[").append(2*r+1+a).append("];}} ");
        }
        return body.append("}}").toString();
    }

    /* These three bodies are deliberately derived from the packed Geometry contract, rather
       than from generated instructions.  In particular SLICE_UPDATE re-derives membership from
       the finite prepared sequence; it does not reuse the emitter's rolling target cursor. */
    private static String sliceUpdateBody(List<String> parameters, int start, Row row) {
        requireVariableMovementAbi(parameters, row, 2); int r = row.rank(), inputs = parameters.size() - 4;
        int bases = 3 * r + 1, strides = bases + inputs, variant = strides + 2 * r;
        int base = row.occurrenceToBoundary().get(0), update = row.occurrenceToBoundary().get(1);
        StringBuilder body = new StringBuilder(coordinatesAndOutput(r, "{long[] g=p" + (inputs + 1) + ";", bases));
        body.append("for(long cursor=p").append(start).append(";cursor<p").append(start + 1)
                .append(";cursor++){boolean selected=true;long source=0L;");
        for (int axis = 0; axis < r; axis++) {
            body.append("long low").append(axis).append("=g[").append(variant + axis)
                    .append("],length").append(axis).append("=g[").append(variant + 2 * r + axis)
                    .append("],step").append(axis).append("=g[").append(variant + 3 * r + axis)
                    .append("],delta").append(axis).append("=c").append(axis).append("-low").append(axis)
                    .append(",increment").append(axis).append("=step").append(axis).append(">0L?step")
                    .append(axis).append(":-step").append(axis).append(";long ordinal").append(axis)
                    .append("=delta").append(axis).append("/increment").append(axis).append(";")
                    .append("if(low").append(axis).append("<0L||delta").append(axis).append("<0L||delta")
                    .append(axis).append("%increment").append(axis).append("!=0L||ordinal").append(axis)
                    .append(">=length").append(axis).append(")selected=false;");
        }
        body.append(valueType(row.dataType())).append(" value;if(selected){source=g[").append(bases + update).append("]; ");
        for (int axis = 0; axis < r; axis++) body.append("source+=(step").append(axis).append(">0L?ordinal")
                .append(axis).append(":length").append(axis).append("-1L-ordinal").append(axis).append(")*g[")
                .append(strides + update * r + axis).append("]; ");
        body.append(loadAt(parameters.get(update), update, "source", row.dataType())).append("}else{source=g[")
                .append(bases + base).append("]; ");
        for (int axis = 0; axis < r; axis++) body.append("source+=c").append(axis).append("*g[")
                .append(strides + base * r + axis).append("]; ");
        body.append(loadAt(parameters.get(base), base, "source", row.dataType())).append("}")
                .append(storeAt(parameters.get(inputs), inputs, "out", "value", row.dataType()));
        body.append("boolean advanced=false;"); advance(body, r, 0, 2 * r + 1); return body.append("}}").toString();
    }

    private static String unfoldAxisBody(List<String> parameters, int start, Row row) {
        requireVariableMovementAbi(parameters, row, 1); int r = row.rank(), inputRank = r - 1;
        int bases = 3 * r + 1, strides = bases + 1, variant = strides + inputRank;
        StringBuilder body = new StringBuilder(coordinatesAndOutput(r, "{long[] g=p2;", bases));
        body.append("for(long cursor=p").append(start).append(";cursor<p").append(start + 1)
                .append(";cursor++){long source=g[").append(bases).append("]; ");
        for (int axis = 0; axis < inputRank; axis++) body.append("long i").append(axis).append("=g[")
                .append(variant).append("]==").append(axis).append("?c").append(axis).append("*g[")
                .append(variant + 2).append("]+c").append(r - 1).append(":c").append(axis)
                .append(";source+=i").append(axis).append("*g[").append(strides + axis).append("]; ");
        body.append(valueType(row.dataType())).append(" value;").append(loadAt(parameters.getFirst(), 0, "source", row.dataType()))
                .append(storeAt(parameters.get(1), 1, "out", "value", row.dataType()))
                .append("boolean advanced=false;"); advance(body, r, 0, 2 * r + 1); return body.append("}}").toString();
    }

    private static String unfold2dBody(List<String> parameters, int start, Row row) {
        requireVariableMovementAbi(parameters, row, 1); if (row.rank() != 3) throw new AssertionError("UNFOLD2D rank: " + row);
        int bases = 10, strides = 11, variant = 15;
        StringBuilder body = new StringBuilder(coordinatesAndOutput(3, "{long[] g=p2;", bases));
        body.append("for(long cursor=p").append(start).append(";cursor<p").append(start + 1)
                .append(";cursor++){long q=c1,p=c2,channel=q/(g[").append(variant + 3).append("]*g[")
                .append(variant + 4).append("]),kh=q/g[").append(variant + 4).append("]%g[")
                .append(variant + 3).append("],kw=q%g[").append(variant + 4).append("],oh=p/g[")
                .append(variant + 12).append("],ow=p%g[").append(variant + 12).append("],ih=oh*g[")
                .append(variant + 5).append("]-g[").append(variant + 7).append("]+kh*g[")
                .append(variant + 9).append("],iw=ow*g[").append(variant + 6).append("]-g[")
                .append(variant + 8).append("]+kw*g[").append(variant + 10).append("]; ")
                .append(valueType(row.dataType())).append(" value;if(ih<0L||ih>=g[").append(variant + 1)
                .append("]||iw<0L||iw>=g[").append(variant + 2).append("]){value=").append(literal(row.dataType(), row.bits()))
                .append(";}else{long source=g[").append(bases).append("]+c0*g[").append(strides)
                .append("]+channel*g[").append(strides + 1).append("]+ih*g[").append(strides + 2)
                .append("]+iw*g[").append(strides + 3).append("]; ").append(loadAt(parameters.getFirst(), 0, "source", row.dataType()))
                .append("}").append(storeAt(parameters.get(1), 1, "out", "value", row.dataType()))
                .append("boolean advanced=false;"); advance(body, 3, 0, 7); return body.append("}}").toString();
    }

    /* CONCAT and STACK deliberately spell their selection trees in source.  The packed geometry
       supplies coordinates, bases, strides and axis/prefix facts; the frozen occurrence map
       supplies the ordered/repeated semantic role selection. */
    private static String concatBody(List<String> parameters, int start, Row row) {
        requireCompositionAbi(parameters, row); int r = row.rank(), inputs = parameters.size() - 4;
        int bases = 3 * r + 1, strides = bases + inputs, variant = strides + inputs * r;
        StringBuilder body = new StringBuilder(coordinatesAndOutput(r, "{long[] g=p" + (parameters.size() - 3) + ";", bases));
        body.append("for(long cursor=p").append(start).append(";cursor<p").append(start + 1).append(";cursor++){long relative=0L;");
        for (int axis = 0; axis < r; axis++) body.append("if(g[").append(variant).append("]==").append(axis).append(")relative=c").append(axis).append(";");
        body.append("long source;").append(valueType(row.dataType())).append(" value;");
        // This is intentionally an all-but-final tree, rather than a loop plus a default
        // occurrence.  Preparation proves the final prepared occurrence owns the remaining
        // coordinate; keeping that fact in source makes javac's executable witness reviewable.
        for (int occurrence = 0; occurrence < row.occurrenceToBoundary().size() - 1; occurrence++) {
            if (occurrence == 0) body.append("if(relative<g[").append(variant).append("+2]){");
            else body.append("else if(relative<g[").append(variant).append('+').append(2 + occurrence)
                    .append("]){");
            appendCompositionCase(body, parameters, row, r, bases, strides, occurrence, "relative", true);
            body.append('}');
        }
        body.append("else{");
        appendCompositionCase(body, parameters, row, r, bases, strides,
                row.occurrenceToBoundary().size() - 1, "relative", true);
        body.append('}');
        appendOutputStore(body, parameters, row, inputs, "out");
        body.append("boolean advanced=false;"); advance(body, r, 0, 2 * r + 1); body.append("}}");
        return body.toString();
    }

    /**
     * Mirrors CONCAT's selected bounded-target route, not its general-long fallback.
     *
     * <p>The entry repeats the route's one-time non-negative/int-width guard and narrows each
     * packed geometry fact once.  The selected route thereafter retains its integer logical and
     * coordinate state, long carrier addresses, source-address reuse, and ordered all-but-final
     * occurrence selection.  The performance witness invokes only geometry admitted by this
     * guard, just as the generated bounded target does.</p>
     */
    private static String boundedConcatBody(List<String> parameters, int start, Row row) {
        requireCompositionAbi(parameters, row);
        int rank = row.rank(), inputs = parameters.size() - 4;
        int bases = 3 * rank + 1, strides = bases + inputs, variant = strides + inputs * rank;
        int geometryLength = variant + 2 + row.occurrenceToBoundary().size();
        StringBuilder body = new StringBuilder("{long[] g=p").append(parameters.size() - 3).append(";");
        body.append("boolean bounded=p").append(start).append(">=0L&&p").append(start)
                .append("<=2147483647L&&p").append(start + 1).append(">=0L&&p")
                .append(start + 1).append("<=2147483647L;");
        for (int index = 0; index < geometryLength; index++) {
            body.append("bounded=bounded&&g[").append(index).append("]>=0L&&g[").append(index)
                    .append("]<=2147483647L;");
        }
        String general = concatBody(parameters, start, row);
        int prefixEnd = general.indexOf(';');
        body.append("if(!bounded){").append(general, prefixEnd + 1, general.length() - 1)
                .append("}else{");
        for (int index = 0; index < geometryLength; index++) body.append("int q").append(index)
                .append("=(int)g[").append(index).append("]; ");
        body.append("long out=q").append(2 * rank).append("; ");
        for (int axis = 0; axis < rank; axis++) body.append("int c").append(axis).append("=q")
                .append(rank + axis).append("; ");
        body.append("int logical=(int)p").append(start).append(",end=(int)p").append(start + 1)
                .append(",occurrence=0,sourceValid=0;long source=0L;").append(valueType(row.dataType()))
                .append(" value;for(;logical<end;logical++){");
        body.append("if(sourceValid==0){int selected=0;");
        for (int axis = 0; axis < rank; axis++) body.append("if(q").append(variant).append("==")
                .append(axis).append(")selected=c").append(axis).append(";");
        for (int occurrence = 0; occurrence < row.occurrenceToBoundary().size(); occurrence++) {
            if (occurrence == 0) body.append("if(selected<q").append(variant + 2 + occurrence).append("){ ");
            else if (occurrence == row.occurrenceToBoundary().size() - 1) body.append("else{");
            else body.append("else if(selected<q").append(variant + 2 + occurrence).append("){ ");
            appendBoundedConcatAddress(body, row, rank, bases, strides, variant, occurrence, "selected");
            body.append('}');
        }
        body.append("sourceValid=1;}");
        for (int occurrence = 0; occurrence < row.occurrenceToBoundary().size(); occurrence++) {
            int boundary = row.occurrenceToBoundary().get(occurrence);
            if (occurrence == 0) body.append("if(occurrence==0){");
            else if (occurrence == row.occurrenceToBoundary().size() - 1) body.append("else{");
            else body.append("else if(occurrence==").append(occurrence).append("){");
            body.append(boundedLoadAt(parameters.get(boundary), boundary, "source", row.dataType()));
            if (rank > 0) body.append("source+=(long)q").append(strides + boundary * rank + rank - 1).append(";");
            body.append('}');
        }
        appendBoundedOutputStore(body, parameters, row, inputs, "out");
        if (rank > 0) {
            int inner = rank - 1;
            body.append("if(c").append(inner).append("+1>=q").append(inner).append(")sourceValid=0;");
            body.append("int concatAxis=q").append(variant).append(";");
            for (int axis = 0; axis < rank; axis++) {
                body.append("if(concatAxis==").append(axis).append("){ ");
                for (int occurrence = 0; occurrence < row.occurrenceToBoundary().size() - 1; occurrence++)
                    body.append("if(occurrence==").append(occurrence).append("&&c").append(axis)
                            .append("+1==q").append(variant + 2 + occurrence).append(")sourceValid=0;");
                body.append('}');
            }
        }
        body.append("boolean advanced=false;");
        for (int axis = rank - 1; axis >= 0; axis--) body.append("if(!advanced){c").append(axis)
                .append("++;out+=(long)q").append(2 * rank + 1 + axis).append(";if(c").append(axis)
                .append("<q").append(axis).append(")advanced=true;else{c").append(axis)
                .append("=0;out-=(long)q").append(axis).append("*q").append(2 * rank + 1 + axis)
                .append(";}}");
        return body.append("}}}").toString();
    }

    private static void appendBoundedConcatAddress(StringBuilder body, Row row, int rank, int bases,
            int strides, int variant, int occurrence, String selected) {
        int boundary = row.occurrenceToBoundary().get(occurrence);
        body.append("occurrence=").append(occurrence).append(";source=q").append(bases + boundary).append(";");
        for (int axis = 0; axis < rank; axis++) {
            body.append("source+=(long)(q").append(variant).append("==").append(axis).append("?")
                    .append(selected).append("-q").append(variant + 1 + occurrence).append(":c")
                    .append(axis).append(")*q").append(strides + boundary * rank + axis).append(";");
        }
    }

    private static String boundedLoadAt(String carrier, int parameter, String address, String dataType) {
        if (carrier.endsWith("[]")) return loadAt(carrier, parameter, address, dataType);
        return "value=p" + parameter + ".get(" + layoutForCounterpart(carrier, dataType) + ","
                + address + "*" + byteWidth(dataType) + "L);";
    }

    private static void appendBoundedOutputStore(StringBuilder body, List<String> parameters, Row row,
            int inputs, String address) {
        String carrier = parameters.get(inputs);
        if (carrier.endsWith("[]")) body.append(storeAt(carrier, inputs, address, "value", row.dataType()));
        else body.append("p").append(inputs).append(".set(")
                .append(layoutForCounterpart(carrier, row.dataType())).append(',').append(address)
                .append('*').append(byteWidth(row.dataType())).append("L,value);");
    }

    private static int byteWidth(String type) { return switch (type) {
        case "FLOAT64", "INT64" -> 8; case "FLOAT32", "INT32" -> 4;
        case "BFLOAT16" -> 2; case "BOOL" -> 1; default -> throw new AssertionError(type);
    }; }

    private static String stackBody(List<String> parameters, int start, Row row) {
        requireCompositionAbi(parameters, row); int r = row.rank(), inputs = parameters.size() - 4;
        int bases = 3 * r + 1, strides = bases + inputs, variant = strides + inputs * (r - 1);
        StringBuilder body = new StringBuilder(coordinatesAndOutput(r, "{long[] g=p" + (parameters.size() - 3) + ";", bases));
        body.append("for(long cursor=p").append(start).append(";cursor<p").append(start + 1).append(";cursor++){long occurrence=0L;");
        for (int axis = 0; axis < r; axis++) body.append("if(g[").append(variant).append("]==").append(axis).append(")occurrence=c").append(axis).append(";");
        body.append("long source;").append(valueType(row.dataType())).append(" value;");
        for (int index = 0; index < row.occurrenceToBoundary().size(); index++) {
            if (index == 0) body.append("if(occurrence==0L){");
            else if (index == row.occurrenceToBoundary().size() - 1) body.append("else{");
            else body.append("else if(occurrence==").append(index).append("L){");
            appendCompositionCase(body, parameters, row, r - 1, bases, strides, index, "0L", false);
            body.append('}');
        }
        appendOutputStore(body, parameters, row, inputs, "out");
        body.append("boolean advanced=false;"); advance(body, r, 0, 2 * r + 1); body.append("}}");
        return body.toString();
    }

    /**
     * Clean-Java counterpart for STACK's proved dense heap-array integer-address entry.
     * The performance witness uses the matrix's rank-two, two-input FLOAT32 representative;
     * this body intentionally retains that selected entry's one-time narrowing and integer
     * coordinate/address progression instead of comparing it to the general-long fallback.
     */
    private static String denseStackBody(List<String> parameters, int start, Row row) {
        requireCompositionAbi(parameters, row);
        if (row.rank() != 2 || row.occurrenceToBoundary().size() != 2
                || !parameters.get(0).equals("float[]") || !parameters.get(1).equals("float[]")
                || !parameters.get(2).equals("float[]")) {
            throw new AssertionError("dense STACK performance counterpart requires the selected heap FLOAT32 representative: " + row);
        }
        return "{long[] g=p3;int c0=(int)g[2],c1=(int)g[3],out=(int)g[4],logical=(int)p4,end=(int)p5;"
                + "for(;logical<end;logical++){int source;float value;if(c1==0){source=(int)g[7]+c0*(int)g[9];value=p0[source];}"
                + "else{source=(int)g[8]+c0*(int)g[10];value=p1[source];}p2[out]=value;"
                + "c1++;out+=(int)g[6];if(c1<(int)g[1]){}else{c1=0;out-=(int)g[1]*(int)g[6];c0++;out+=(int)g[5];"
                + "if(c0>=(int)g[0]){c0=0;out-=(int)g[0]*(int)g[5];}}}}";
    }

    private static String coordinatesAndOutput(int rank, String prefix, int ignored) {
        StringBuilder result = new StringBuilder(prefix).append("long out=g[").append(2 * rank).append("]; ");
        for (int a = 0; a < rank; a++) result.append("long c").append(a).append("=g[").append(rank + a).append("]; ");
        return result.toString();
    }
    private static void appendCompositionLoad(StringBuilder body, List<String> parameters, Row row,
            int inputRank, int inputs, int bases, int strides, String occurrence, String relative,
            boolean concat) {
        int variant = strides + (concat ? inputs * row.rank() : inputs * inputRank);
        body.append(valueType(row.dataType())).append(" value;");
        for (int occurrenceIndex = 0; occurrenceIndex < row.occurrenceToBoundary().size(); occurrenceIndex++) {
            int boundary = row.occurrenceToBoundary().get(occurrenceIndex);
            if (occurrenceIndex == 0) body.append("if(").append(occurrence).append("==0){");
            else if (occurrenceIndex == row.occurrenceToBoundary().size() - 1) body.append("else{");
            else body.append("else if(").append(occurrence).append("==").append(occurrenceIndex).append("){");
            body.append("source=g[").append(bases + boundary).append("]; ");
            for (int inputAxis = 0; inputAxis < inputRank; inputAxis++) {
                String coordinate = concat ? "c" + inputAxis : "(g[" + variant + "]>" + inputAxis + "?c" + inputAxis + ":c" + (inputAxis + 1) + ")";
                if (concat) coordinate = "(g[" + variant + "]==" + inputAxis + "?" + relative + ":c" + inputAxis + ")";
                body.append("source+=").append(coordinate).append("*g[").append(strides + boundary * inputRank + inputAxis).append("]; ");
            }
            body.append(loadAt(parameters.get(boundary), boundary, "source", row.dataType())).append("}");
        }
        body.append(' ');
    }

    /** Emits one statically prepared occurrence arm directly, without a second carrier dispatch. */
    private static void appendCompositionCase(StringBuilder body, List<String> parameters, Row row,
            int inputRank, int bases, int strides, int occurrenceIndex, String relative, boolean concat) {
        int boundary = row.occurrenceToBoundary().get(occurrenceIndex);
        int variant = strides + (concat ? (parameters.size() - 4) * row.rank()
                : (parameters.size() - 4) * inputRank);
        body.append("source=g[").append(bases + boundary).append("]; ");
        for (int inputAxis = 0; inputAxis < inputRank; inputAxis++) {
            String coordinate = concat ? "c" + inputAxis : "(g[" + variant + "]>" + inputAxis
                    + "?c" + inputAxis + ":c" + (inputAxis + 1) + ")";
            if (concat) coordinate = "(g[" + variant + "]==" + inputAxis + "?" + relative
                    + "-g[" + variant + "+1+" + occurrenceIndex + "]:c" + inputAxis + ")";
            body.append("source+=").append(coordinate).append("*g[")
                    .append(strides + boundary * inputRank + inputAxis).append("]; ");
        }
        body.append(loadAt(parameters.get(boundary), boundary, "source", row.dataType()));
    }
    private static void appendOutputStore(StringBuilder body, List<String> parameters, Row row, int inputs, String address) {
        body.append(storeAt(parameters.get(inputs), inputs, address, "value", row.dataType()));
    }
    private static void requireCompositionAbi(List<String> parameters, Row row) {
        if (row.rank() < 1 || row.occurrenceToBoundary().isEmpty() || parameters.size() < 6
                || !parameters.get(parameters.size() - 3).equals("long[]"))
            throw new AssertionError("composition entry must retain carriers, packed geometry, range and occurrence map ABI: " + row);
    }

    private static void advance(StringBuilder body, int rank, int unused, int outputStrides) {
        for (int a=rank-1;a>=0;a--) body.append("if(!advanced){c").append(a).append("++;out+=g[").append(outputStrides+a).append("];if(c").append(a).append("<g[").append(a).append("])advanced=true;else{c").append(a).append("=0L;out-=g[").append(a).append("]*g[").append(outputStrides+a).append("];}} ");
    }
    private static void requireMovementAbi(List<String> parameters, Row row) {
        if (row.rank() < 1 || parameters.size() != 5 || !parameters.get(2).equals("long[]"))
            throw new AssertionError("movement entry must have two carriers, packed geometry and range ABI: " + row);
    }
    private static void requireVariableMovementAbi(List<String> parameters, Row row, int occurrences) {
        int inputs = parameters.size() - 4;
        if (row.rank() < 1 || inputs < 1 || !parameters.get(inputs + 1).equals("long[]")
                || row.occurrenceToBoundary().size() != occurrences
                || row.occurrenceToBoundary().stream().anyMatch(boundary -> boundary < 0 || boundary >= inputs))
            throw new AssertionError("movement entry must retain ordered typed carriers, geometry and range ABI: " + row);
    }
    private static String literal(String type, long bits) { return switch (type) {
        case "FLOAT64" -> "Double.longBitsToDouble(" + bits + "L)";
        case "FLOAT32" -> "Float.intBitsToFloat(" + (int) bits + ")";
        case "BFLOAT16" -> "(short)" + (short) bits;
        case "INT64" -> bits + "L"; case "INT32" -> "(int)" + (int) bits;
        case "BOOL" -> "(byte)" + (byte) bits; default -> throw new AssertionError(type);
    }; }

    private static String load(String carrier, String address, String dataType) {
        if (carrier.equals("double[]")) return "p0[(int)" + address + "]";
        if (carrier.equals("float[]")) return "p0[(int)" + address + "]";
        if (carrier.equals("short[]")) return "p0[(int)" + address + "]";
        if (carrier.equals("long[]")) return "p0[(int)" + address + "]";
        if (carrier.equals("int[]")) return "p0[(int)" + address + "]";
        if (carrier.equals("byte[]")) return "p0[(int)" + address + "]";
        String layout = layoutForCounterpart(carrier, dataType);
        return "p0.getAtIndex(" + layout + "," + address + ")";
    }

    private static String loadAt(String carrier, int parameter, String address, String dataType) {
        if (carrier.endsWith("[]")) return "value=p" + parameter + "[(int)" + address + "];";
        return "value=p" + parameter + ".getAtIndex(" + layoutForCounterpart(carrier, dataType)
                + "," + address + ");";
    }
    private static String valueType(String type) { return switch (type) {
        case "FLOAT64" -> "double"; case "FLOAT32" -> "float"; case "BFLOAT16" -> "short";
        case "INT64" -> "long"; case "INT32" -> "int"; case "BOOL" -> "byte";
        default -> throw new AssertionError(type); }; }
    private static String zero(String type) { return switch (type) {
        case "FLOAT64" -> "0D"; case "FLOAT32" -> "0F"; case "BFLOAT16" -> "(short)0";
        case "INT64" -> "0L"; case "INT32" -> "0"; case "BOOL" -> "(byte)0";
        default -> throw new AssertionError(type); }; }

    private static String store(String carrier, String address, String value, String dataType) {
        if (carrier.endsWith("[]")) return "p1[(int)" + address + "]=" + value + ";";
        return "p1.setAtIndex(" + layoutForCounterpart(carrier, dataType) + "," + address + "," + value + ");";
    }

    private static String storeAt(String carrier, int parameter, String address, String value, String dataType) {
        if (carrier.endsWith("[]")) return "p" + parameter + "[(int)" + address + "]=" + value + ";";
        return "p" + parameter + ".setAtIndex(" + layoutForCounterpart(carrier, dataType)
                + "," + address + "," + value + ");";
    }

    private static String layoutForCounterpart(String carrier, String dataType) {
        // Source and result have the same represented type for every affine fixture; carrier
        // direction changes only the access spelling.
        return "java.lang.foreign.ValueLayout." + switch (carrier) {
            case "java.lang.foreign.MemorySegment" -> switch (dataType) {
                case "FLOAT64" -> "JAVA_DOUBLE_UNALIGNED";
                case "FLOAT32" -> "JAVA_FLOAT_UNALIGNED";
                case "INT64" -> "JAVA_LONG_UNALIGNED";
                case "INT32" -> "JAVA_INT_UNALIGNED";
                case "BFLOAT16" -> "JAVA_SHORT_UNALIGNED";
                case "BOOL" -> "JAVA_BYTE";
                default -> throw new AssertionError("unknown affine data type: " + dataType);
            };
            default -> throw new AssertionError("non-segment layout request: " + carrier);
        };
    }

    static List<String> parameters(String descriptor) {
        java.util.ArrayList<String> result = new java.util.ArrayList<>();
        for (int index = 1; descriptor.charAt(index) != ')'; index++) {
            int arrays = 0;
            while (descriptor.charAt(index) == '[') { arrays++; index++; }
            char tag = descriptor.charAt(index);
            String type;
            if (tag == 'L') {
                int end = descriptor.indexOf(';', index);
                type = descriptor.substring(index + 1, end).replace('/', '.');
                index = end;
            } else type = switch (tag) {
                case 'D' -> "double"; case 'F' -> "float"; case 'J' -> "long"; case 'I' -> "int";
                case 'S' -> "short"; case 'B' -> "byte"; case 'Z' -> "boolean";
                default -> throw new AssertionError("unsupported entry descriptor: " + descriptor);
            };
            result.add(type + "[]".repeat(arrays));
        }
        return List.copyOf(result);
    }

    private static final class Source extends SimpleFile {
        private final String text;
        Source(String binary, String text) { super(binary, JavaFileObject.Kind.SOURCE); this.text = text; }
        @Override public CharSequence getCharContent(boolean ignoreEncodingErrors) { return text; }
    }
    private static final class ByteFile extends SimpleFile {
        private final ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        ByteFile(String binary) { super(binary, JavaFileObject.Kind.CLASS); }
        @Override public ByteArrayOutputStream openOutputStream() { return bytes; }
        byte[] bytes() { return bytes.toByteArray(); }
    }
    private abstract static class SimpleFile extends SimpleJavaFileObject {
        SimpleFile(String binary, JavaFileObject.Kind kind) { super(URI.create("mem:///" + binary.replace('.', '/') + kind.extension), kind); }
    }
    private static final class CleanLoader extends ClassLoader {
        CleanLoader() { super(CpuAffineMovementIndexingScatterRandomCleanJavaOracle.class.getClassLoader()); }
        Class<?> define(byte[] bytes) { return defineClass(BINARY, bytes, 0, bytes.length); }
    }
}
