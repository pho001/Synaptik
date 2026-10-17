package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import javax.tools.DiagnosticCollector;
import javax.tools.FileObject;
import javax.tools.ForwardingJavaFileManager;
import javax.tools.JavaCompiler;
import javax.tools.JavaFileManager;
import javax.tools.JavaFileObject;
import javax.tools.SimpleJavaFileObject;
import javax.tools.StandardJavaFileManager;
import javax.tools.ToolProvider;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;

/**
 * Compiles the frozen, statically typed performance-oracle methods once for the loss benchmark.
 *
 * <p>The generated methods deliberately use only their concrete array or {@link MemorySegment}
 * parameters.  Selection, compilation, lookup, and argument binding happen before measurement.
 * This test-only owner is separate from the production emitter so it remains a clean-Java review
 * oracle rather than a second production code generator.</p>
 */
final class CpuLossPerformanceOracle {
    private static final String PACKAGE = "io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.generated";
    private static final String SIMPLE_NAME = "LossPerformanceOracleGenerated";
    private static final String BINARY_NAME = PACKAGE + '.' + SIMPLE_NAME;

    private final Class<?> generatedClass;
    private final String source;
    private final byte[] classBytes;

    private CpuLossPerformanceOracle(List<Spec> specs) {
        source = source(specs);
        classBytes = compile(source);
        generatedClass = new OracleClassLoader(CpuLossPerformanceOracle.class.getClassLoader())
                .define(BINARY_NAME, classBytes);
    }

    static CpuLossPerformanceOracle compile(List<Spec> specs) {
        return new CpuLossPerformanceOracle(specs);
    }

    MethodHandle bind(Spec spec, Object left, Object right, Object output, long[] geometry,
            long start, long end) throws NoSuchMethodException,
            IllegalAccessException {
        MethodHandle method = MethodHandles.publicLookup().findStatic(generatedClass, spec.methodName(),
                MethodType.methodType(void.class, carrierClass(spec.leftCarrier()),
                        carrierClass(spec.rightCarrier()), carrierClass(spec.outputCarrier()),
                        long[].class, long.class, long.class));
        return MethodHandles.insertArguments(method, 0, left, right, output, geometry, start, end)
                .asType(MethodType.methodType(void.class));
    }

    String source() { return source; }

    byte[] classBytes() { return classBytes.clone(); }

    record Spec(String methodName, Family family, Floating left, Floating right, Reduction reduction,
                boolean ignore, Carrier leftCarrier, Carrier rightCarrier, Carrier outputCarrier,
                Index index) { }

    enum Family { MSE, DENSE, INDEX }
    enum Floating { BF16, F32, F64 }
    enum Index { I32, I64, UNUSED }
    enum Reduction { NONE, SUM, MEAN }
    enum Carrier { SHORT_ARRAY, FLOAT_ARRAY, DOUBLE_ARRAY, INT_ARRAY, LONG_ARRAY, SEGMENT }

    private static Class<?> carrierClass(Carrier carrier) {
        return switch (carrier) {
            case SHORT_ARRAY -> short[].class;
            case FLOAT_ARRAY -> float[].class;
            case DOUBLE_ARRAY -> double[].class;
            case INT_ARRAY -> int[].class;
            case LONG_ARRAY -> long[].class;
            case SEGMENT -> MemorySegment.class;
        };
    }

    private static String source(List<Spec> specs) {
        StringBuilder result = new StringBuilder("package ").append(PACKAGE).append(";\n")
                .append("import java.lang.foreign.MemorySegment;\n")
                .append("import java.lang.foreign.ValueLayout;\n")
                .append("public final class ").append(SIMPLE_NAME).append(" {\n")
                .append("private ").append(SIMPLE_NAME).append("() {}\n");
        for (Spec spec : specs) method(result, spec);
        return result.append("}\n").toString();
    }

    private static void method(StringBuilder out, Spec s) {
        out.append("public static void ").append(s.methodName()).append('(')
                .append(type(s.leftCarrier())).append(" left,").append(type(s.rightCarrier()))
                .append(" right,").append(type(s.outputCarrier()))
                .append(" output,long[] geometry,long start,long end){");
        switch (s.family()) {
            case MSE -> mse(out, s);
            case DENSE -> categorical(out, s, false);
            case INDEX -> categorical(out, s, true);
        }
        out.append("}\n");
    }

    private static void mse(StringBuilder out, Spec s) {
        String left = load("left", s.leftCarrier(), s.left(), "i");
        if (s.reduction() == Reduction.NONE) {
            out.append("int predictionBase=(int)geometry[4],targetBase=(int)geometry[5],outputBase=(int)geometry[6];")
                    .append("for(int ordinal=(int)start;ordinal<(int)end;ordinal++){int i=predictionBase+ordinal;")
                    .append(accumulator(s)).append(" d=")
                    .append(left).append('-')
                    .append(load("right", s.rightCarrier(), s.right(), "targetBase+ordinal"))
                    .append(';')
                    .append(store(s, "outputBase+ordinal", "d*d")).append("}");
        } else {
            out.append(accumulator(s)).append(" total=0.0").append(s.left() == Floating.F64 || s.right() == Floating.F64 ? "d" : "f").append(';')
                    .append("int predictionBase=(int)geometry[4],targetBase=(int)geometry[5];")
                    .append("for(int ordinal=0;ordinal<(int)geometry[9];ordinal++){int i=predictionBase+ordinal;")
                    .append(accumulator(s)).append(" d=").append(left).append("-")
                    .append(load("right", s.rightCarrier(), s.right(), "targetBase+ordinal"))
                    .append(";total+=d*d;}")
                    .append(store(s, "(int)geometry[6]", s.reduction() == Reduction.MEAN ? "total/(int)geometry[9]" : "total"));
        }
    }

    private static void categorical(StringBuilder out, Spec s, boolean indexed) {
        String acc = accumulator(s);
        String zero = acc.equals("double") ? "0.0d" : "0.0f";
        if (s.reduction() != Reduction.NONE) {
            out.append(acc).append(" total=").append(zero).append(';');
            if (s.reduction() == Reduction.MEAN) out.append("int count=0;");
        }
        out.append("int axis=(int)geometry[1],rank=(int)geometry[0],classes=(int)geometry[10+axis],outer=1,inner=1;")
                .append("for(int coordinate=0;coordinate<rank;coordinate++){if(coordinate<axis)outer*=geometry[10+coordinate];else if(coordinate>axis)inner*=geometry[10+coordinate];}")
                .append("int predictionBase=(int)geometry[4],targetBase=(int)geometry[5]")
                .append(s.reduction() == Reduction.NONE ? ",outputBase=(int)geometry[6];" : ";")
                .append("for(int outerIndex=0;outerIndex<outer;outerIndex++){for(int sample=0;sample<inner;sample++){");
        if (indexed) {
            String target = indexLoad(s, "targetBase");
            out.append("long selected=").append(target).append(';');
            if (s.ignore()) out.append("if(geometry[7]!=0L&&selected==geometry[8]){")
                    .append(s.reduction() == Reduction.NONE ? store(s, "outputBase", zero) : "")
                    .append("predictionBase++;targetBase++;")
                    .append(s.reduction() == Reduction.NONE ? "outputBase++;" : "")
                    .append("continue;}");
        }
        out.append("int base=predictionBase;").append(acc)
                .append(" max=").append(acc.equals("double") ? "Double.NEGATIVE_INFINITY" : "Float.NEGATIVE_INFINITY").append(';')
                .append("for(int clazz=0;clazz<classes;clazz++){ ").append(acc).append(" value=")
                .append(load("left", s.leftCarrier(), s.left(), "base+clazz*inner"))
                .append(";if(value>max)max=value;}").append(acc).append(" sum=").append(zero).append(';')
                .append("for(int clazz=0;clazz<classes;clazz++)sum+=").append(exp(s,
                        load("left", s.leftCarrier(), s.left(), "base+clazz*inner") + "-max")).append(';')
                .append(acc).append(" lse=max+").append(log(s, "sum")).append(';');
        if (indexed) {
            out.append(acc).append(" loss=lse-").append(load("left", s.leftCarrier(), s.left(), "base+(int)selected*inner")).append(';');
        } else {
            out.append(acc).append(" loss=").append(zero).append(';')
                    .append("for(int clazz=0;clazz<classes;clazz++){ ").append(acc).append(" weight=")
                    .append(load("right", s.rightCarrier(), s.right(), "targetBase+clazz*inner"))
                    .append(";if(weight!=").append(zero).append(")loss+=weight*(lse-")
                    .append(load("left", s.leftCarrier(), s.left(), "base+clazz*inner")).append(");}");
        }
        if (s.reduction() == Reduction.NONE) out.append(store(s, "outputBase", "loss"));
        else {
            out.append("total+=loss;");
            if (s.reduction() == Reduction.MEAN) out.append("count++;");
        }
        out.append("predictionBase++;targetBase++;")
                .append(s.reduction() == Reduction.NONE ? "outputBase++;" : "")
                .append("}predictionBase+=(classes-1)*inner;")
                .append(s.family() == Family.DENSE ? "targetBase+=(classes-1)*inner;" : "")
                .append("}");
        if (s.reduction() != Reduction.NONE)
            out.append(store(s, "(int)geometry[6]", s.reduction() == Reduction.MEAN ? "total/count" : "total"));
    }

    private static String accumulator(Spec s) {
        return s.left() == Floating.F64 || (!s.family().equals(Family.INDEX) && s.right() == Floating.F64)
                ? "double" : "float";
    }

    private static String exp(Spec s, String value) {
        return accumulator(s).equals("double") ? "StrictMath.exp(" + value + ')' : "(float)StrictMath.exp((double)(" + value + "))";
    }

    private static String log(Spec s, String value) {
        return accumulator(s).equals("double") ? "StrictMath.log(" + value + ')' : "(float)StrictMath.log((double)" + value + ')';
    }

    private static String load(String variable, Carrier carrier, Floating type, String index) {
        return switch (carrier) {
            case SHORT_ARRAY -> "Float.intBitsToFloat(Short.toUnsignedInt(" + variable + "[" + index + "])<<16)";
            case FLOAT_ARRAY -> variable + '[' + index + ']';
            case DOUBLE_ARRAY -> variable + '[' + index + ']';
            case SEGMENT -> switch (type) {
                case BF16 -> "Float.intBitsToFloat(Short.toUnsignedInt(" + variable + ".get(ValueLayout.JAVA_SHORT_UNALIGNED,(long)(" + index + ")*2))<<16)";
                case F32 -> variable + ".get(ValueLayout.JAVA_FLOAT_UNALIGNED,(long)(" + index + ")*4)";
                case F64 -> variable + ".get(ValueLayout.JAVA_DOUBLE_UNALIGNED,(long)(" + index + ")*8)";
            };
            default -> throw new AssertionError(carrier);
        };
    }

    private static String indexLoad(Spec s, String index) {
        return switch (s.rightCarrier()) {
            case INT_ARRAY -> "right[" + index + ']';
            case LONG_ARRAY -> "right[" + index + ']';
            case SEGMENT -> s.index() == Index.I32
                    ? "right.get(ValueLayout.JAVA_INT_UNALIGNED,(long)(" + index + ")*4)"
                    : "right.get(ValueLayout.JAVA_LONG_UNALIGNED,(long)(" + index + ")*8)";
            default -> throw new AssertionError(s.rightCarrier());
        };
    }

    private static String store(Spec s, String index, String value) {
        return switch (s.outputCarrier()) {
            case SHORT_ARRAY -> "output[" + index + "]=" + bf16(value) + ';';
            case FLOAT_ARRAY -> "output[" + index + "]=" + value + ';';
            case DOUBLE_ARRAY -> "output[" + index + "]=" + value + ';';
            /* Index loss retains the logits type: its integral right operand is represented by
               the synthetic F32 oracle field only because it is never floating-decoded.  Output
               carrier selection must therefore derive index output from the left logits type,
               rather than treating that synthetic field as a promoted floating target. */
            case SEGMENT -> switch (s.left() == Floating.F64
                    || (!s.family().equals(Family.INDEX) && s.right() == Floating.F64)
                    ? Floating.F64
                    : s.left() == Floating.BF16
                            && (s.family().equals(Family.INDEX) || s.right() == Floating.BF16)
                                    ? Floating.BF16 : Floating.F32) {
                case BF16 -> "output.set(ValueLayout.JAVA_SHORT_UNALIGNED,(long)(" + index + ")*2," + bf16(value) + ");";
                case F32 -> "output.set(ValueLayout.JAVA_FLOAT_UNALIGNED,(long)(" + index + ")*4," + value + ");";
                case F64 -> "output.set(ValueLayout.JAVA_DOUBLE_UNALIGNED,(long)(" + index + ")*8," + value + ");";
            };
            default -> throw new AssertionError(s.outputCarrier());
        };
    }

    private static String bf16(String value) {
        String bits = "Float.floatToRawIntBits(" + value + ')';
        return "(short)((((" + bits + "&0x7fffffff)>0x7f800000)?0x7fc0:("
                + bits + "+0x7fff+(('".replace("'", "") + bits + ">>>16)&1)))>>>16)";
    }

    private static String type(Carrier carrier) {
        return switch (carrier) {
            case SHORT_ARRAY -> "short[]"; case FLOAT_ARRAY -> "float[]"; case DOUBLE_ARRAY -> "double[]";
            case INT_ARRAY -> "int[]"; case LONG_ARRAY -> "long[]"; case SEGMENT -> "MemorySegment";
        };
    }

    private static byte[] compile(String source) {
        JavaCompiler compiler = ToolProvider.getSystemJavaCompiler();
        if (compiler == null) throw new IllegalStateException("Java compiler is unavailable in the test runtime");
        DiagnosticCollector<JavaFileObject> diagnostics = new DiagnosticCollector<>();
        try (StandardJavaFileManager standard = compiler.getStandardFileManager(diagnostics, null, null);
                MemoryFileManager files = new MemoryFileManager(standard)) {
            JavaFileObject unit = new SourceFile(BINARY_NAME, source);
            boolean success = compiler.getTask(null, files, diagnostics, List.of("--release", "26", "-classpath",
                    System.getProperty("java.class.path")), null, List.of(unit)).call();
            if (!success) throw new IllegalStateException("could not compile statically typed loss oracle: " + diagnostics.getDiagnostics());
            return files.bytes(BINARY_NAME);
        } catch (IOException failure) { throw new IllegalStateException("could not close loss oracle compiler", failure); }
    }

    private static final class SourceFile extends SimpleJavaFileObject {
        private final String text;
        SourceFile(String name, String text) { super(URI.create("string:///" + name.replace('.', '/') + Kind.SOURCE.extension), Kind.SOURCE); this.text = text; }
        @Override public CharSequence getCharContent(boolean ignoreEncodingErrors) { return text; }
    }
    private static final class BytecodeFile extends SimpleJavaFileObject {
        private final ByteArrayOutputStream output = new ByteArrayOutputStream();
        BytecodeFile(String name) { super(URI.create("bytes:///" + name.replace('.', '/') + Kind.CLASS.extension), Kind.CLASS); }
        @Override public ByteArrayOutputStream openOutputStream() { return output; }
    }
    private static final class MemoryFileManager extends ForwardingJavaFileManager<StandardJavaFileManager> {
        private final Map<String, BytecodeFile> classes = new ConcurrentHashMap<>();
        MemoryFileManager(StandardJavaFileManager delegate) { super(delegate); }
        @Override public JavaFileObject getJavaFileForOutput(JavaFileManager.Location location, String name, JavaFileObject.Kind kind, FileObject sibling) { BytecodeFile file = new BytecodeFile(name); classes.put(name, file); return file; }
        byte[] bytes(String name) { BytecodeFile file = classes.get(name); if (file == null) throw new IllegalStateException("missing compiled oracle class " + name); return file.output.toByteArray(); }
    }
    private static final class OracleClassLoader extends ClassLoader {
        OracleClassLoader(ClassLoader parent) { super(parent); }
        Class<?> define(String name, byte[] bytes) { return defineClass(name, bytes, 0, bytes.length); }
    }
}
