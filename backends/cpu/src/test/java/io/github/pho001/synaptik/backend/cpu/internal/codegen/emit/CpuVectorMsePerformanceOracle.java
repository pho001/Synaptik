package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import java.io.ByteArrayOutputStream;
import java.net.URI;
import java.util.List;
import javax.tools.ForwardingJavaFileManager;
import javax.tools.FileObject;
import javax.tools.JavaCompiler;
import javax.tools.JavaFileManager;
import javax.tools.JavaFileObject;
import javax.tools.SimpleJavaFileObject;
import javax.tools.StandardJavaFileManager;
import javax.tools.ToolProvider;

/** Builds the test-only independently compiled typed Vector API MSE oracle. */
final class CpuVectorMsePerformanceOracle {
    private CpuVectorMsePerformanceOracle() { }

    static byte[] compile(List<Spec> specs) {
        JavaCompiler compiler = ToolProvider.getSystemJavaCompiler();
        if (compiler == null) throw new IllegalStateException("JDK compiler is required");
        var output = new ByteArrayOutputStream();
        JavaFileObject source = new Source(source(specs));
        try (StandardJavaFileManager standard = compiler.getStandardFileManager(null, null, null)) {
            JavaFileManager files = new ForwardingJavaFileManager<>(standard) {
                @Override public JavaFileObject getJavaFileForOutput(Location location, String name,
                        JavaFileObject.Kind kind, FileObject sibling) {
                    return new SimpleJavaFileObject(URI.create("mem:///" + name + kind.extension), kind) {
                        @Override public java.io.OutputStream openOutputStream() { return output; }
                    };
                }
            };
            boolean compiled = compiler.getTask(null, files, null, List.of("--release", "26",
                    "--add-modules", "jdk.incubator.vector"), null, List.of(source)).call();
            if (!compiled) throw new IllegalStateException("vector MSE oracle compilation failed");
        } catch (java.io.IOException failure) {
            throw new IllegalStateException("could not compile vector MSE oracle", failure);
        }
        return output.toByteArray();
    }

    record Spec(String name, Floating type, Carrier prediction, Carrier target, Carrier output,
                boolean sharedInput) { }
    enum Floating { F32, F64 }
    enum Carrier { ARRAY, SEGMENT }

    static String source(List<Spec> specs) {
        StringBuilder out = new StringBuilder("package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;\n"
                + "import java.lang.foreign.*;import java.nio.*;import jdk.incubator.vector.*;\n"
                + "public final class VectorMsePerformanceOracleGenerated{private VectorMsePerformanceOracleGenerated(){}\n");
        for (Spec spec : specs) method(out, spec);
        return out.append("}\n").toString();
    }

    private static void method(StringBuilder out, Spec spec) {
        boolean f32 = spec.type == Floating.F32;
        String scalar = f32 ? "float" : "double";
        String vector = f32 ? "FloatVector" : "DoubleVector";
        String species = vector + ".SPECIES_PREFERRED";
        out.append("public static void ").append(spec.name).append('(')
                .append(type(spec.prediction, scalar)).append(" prediction,");
        if (!spec.sharedInput) out.append(type(spec.target, scalar)).append(" target,");
        out.append(type(spec.output, scalar)).append(" output,long[] geometry,long start,long end){")
                .append("long p=geometry[4]+start,t=geometry[5]+start,o=geometry[6]+start,i=start;")
                .append("int lanes=").append(species).append(".length();while(i+lanes<=end){")
                .append(vector).append(" a=").append(load(spec.prediction, "prediction", species, scalar, "p"))
                .append(",b=").append(load(spec.sharedInput ? spec.prediction : spec.target,
                        spec.sharedInput ? "prediction" : "target", species, scalar, "t"))
                .append(",d=a.sub(b);d.mul(d).").append(store(spec.output, "output", scalar, "o"))
                .append(";p+=lanes;t+=lanes;o+=lanes;i+=lanes;}")
                .append("while(i<end){").append(scalar).append(" d=")
                .append(scalarLoad(spec.prediction, "prediction", scalar, "p")).append('-')
                .append(scalarLoad(spec.sharedInput ? spec.prediction : spec.target,
                        spec.sharedInput ? "prediction" : "target", scalar, "t"))
                .append(';').append(scalarStore(spec.output, "output", scalar, "o", "d*d"))
                .append(";p++;t++;o++;i++;}}\n");
    }

    private static String type(Carrier carrier, String scalar) { return carrier == Carrier.ARRAY ? scalar + "[]" : "MemorySegment"; }
    private static String load(Carrier carrier, String name, String species, String scalar, String index) {
        String vector = scalar.equals("float") ? "FloatVector" : "DoubleVector";
        return carrier == Carrier.ARRAY ? vector + ".fromArray(" + species + ',' + name + ",(int)" + index + ')'
                : vector + ".fromMemorySegment(" + species + ',' + name + ',' + index + "*"
                + (scalar.equals("float") ? 4 : 8) + "L,ByteOrder.nativeOrder())";
    }
    private static String store(Carrier c, String n, String scalar, String i) { return c == Carrier.ARRAY ? "intoArray(" + n + ",(int)" + i + ')' : "intoMemorySegment(" + n + ',' + i + "*" + (scalar.equals("float") ? 4 : 8) + "L,ByteOrder.nativeOrder())"; }
    private static String scalarLoad(Carrier c, String n, String scalar, String i) { return c == Carrier.ARRAY ? n + "[(int)" + i + ']' : n + ".getAtIndex(ValueLayout.JAVA_" + (scalar.equals("float") ? "FLOAT_UNALIGNED" : "DOUBLE_UNALIGNED") + ".withOrder(ByteOrder.nativeOrder())," + i + ')'; }
    private static String scalarStore(Carrier c, String n, String scalar, String i, String v) { return c == Carrier.ARRAY ? n + "[(int)" + i + "]=" + v : n + ".setAtIndex(ValueLayout.JAVA_" + (scalar.equals("float") ? "FLOAT_UNALIGNED" : "DOUBLE_UNALIGNED") + ".withOrder(ByteOrder.nativeOrder())," + i + ',' + v + ')'; }
    private static final class Source extends SimpleJavaFileObject {
        private final String text;
        Source(String text) { super(URI.create("string:///VectorMsePerformanceOracleGenerated.java"), Kind.SOURCE); this.text = text; }
        @Override public CharSequence getCharContent(boolean ignoreEncodingErrors) { return text; }
    }
}
