package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import java.io.ByteArrayOutputStream;
import java.net.URI;
import java.util.List;
import javax.tools.FileObject;
import javax.tools.ForwardingJavaFileManager;
import javax.tools.JavaCompiler;
import javax.tools.JavaFileManager;
import javax.tools.JavaFileObject;
import javax.tools.SimpleJavaFileObject;
import javax.tools.StandardJavaFileManager;
import javax.tools.ToolProvider;

/**
 * Emits the independently {@code javac}-compiled optimal Java Conv oracle for Stage B.
 *
 * <p>Each emitted method performs one cold start-coordinate decode, then traverses explicit
 * {@code n/oc/(od)/oh/ow} loops with incremented width-local input and output cursors. No emitted
 * hot loop calls Synaptik code, reconstructs an ordinal coordinate, or reloads geometry.</p>
 */
final class CpuConvSimdPerformanceOracle {
    static final String SIMPLE_NAME = "ConvSimdPerformanceOracleGenerated";

    private CpuConvSimdPerformanceOracle() { }

    /** Exact rank, primitive type, carrier signature, padding, and grouping for one row. */
    record Spec(String name, boolean conv3d, boolean f32, boolean inputSegment,
            boolean weightSegment, boolean outputSegment, long paddingWidth, long groups) { }

    static byte[] compile(List<Spec> specs) {
        JavaCompiler compiler = ToolProvider.getSystemJavaCompiler();
        if (compiler == null) throw new IllegalStateException("JDK compiler is required");
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        try (StandardJavaFileManager standard = compiler.getStandardFileManager(null, null, null)) {
            JavaFileManager manager = new ForwardingJavaFileManager<>(standard) {
                @Override public JavaFileObject getJavaFileForOutput(Location location, String name,
                        JavaFileObject.Kind kind, FileObject sibling) {
                    return new SimpleJavaFileObject(URI.create("mem:///" + name + kind.extension), kind) {
                        @Override public java.io.OutputStream openOutputStream() { return bytes; }
                    };
                }
            };
            boolean success = compiler.getTask(null, manager, null,
                    List.of("--release", "26", "--add-modules", "jdk.incubator.vector"), null,
                    List.of(new Source(source(specs)))).call();
            if (!success) throw new IllegalStateException("direct Conv oracle compilation failed");
        } catch (java.io.IOException failure) {
            throw new IllegalStateException("direct Conv oracle compilation failed", failure);
        }
        return bytes.toByteArray();
    }

    static String source(List<Spec> specs) {
        StringBuilder out = new StringBuilder();
        out.append("package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;\n")
                .append("import java.lang.foreign.*;\nimport java.nio.*;\n")
                .append("import jdk.incubator.vector.*;\n")
                .append("public final class ").append(SIMPLE_NAME).append(" {\n")
                .append("private ").append(SIMPLE_NAME).append("() {}\n");
        for (Spec spec : specs) appendMethod(out, spec);
        return out.append("}\n").toString();
    }

    private static void appendMethod(StringBuilder out, Spec s) {
        Names n = new Names(s);
        int rank = s.conv3d() ? 5 : 4;
        int boundaries = 3;
        int inputExtents = boundaries;
        int inputStrides = inputExtents + rank;
        int weightExtents = inputStrides + rank;
        int weightStrides = weightExtents + rank;
        int outputExtents = weightStrides + rank;
        int outputStrides = outputExtents + rank;
        out.append("public static void ").append(s.name()).append('(')
                .append(n.inputCarrier).append(" input,").append(n.weightCarrier)
                .append(" weight,").append(n.outputCarrier)
                .append(" output,long[] geometry,long start,long end){\n");
        if (s.inputSegment() || s.weightSegment() || s.outputSegment()) {
            out.append("ByteOrder nativeOrder=ByteOrder.nativeOrder();\n")
                    .append("ValueLayout.").append(s.f32() ? "OfFloat" : "OfDouble")
                    .append(" nativeLayout=ValueLayout.")
                    .append(s.f32() ? "JAVA_FLOAT_UNALIGNED" : "JAVA_DOUBLE_UNALIGNED")
                    .append(".withOrder(nativeOrder);\n");
        }
        appendGeometry(out, s, inputExtents, inputStrides, weightExtents, weightStrides,
                outputExtents, outputStrides);
        appendColdDecode(out, s);
        appendNestedLoops(out, s, n);
        out.append("}\n");
    }

    private static void appendGeometry(StringBuilder out, Spec s, int ie, int is, int we, int ws,
            int oe, int os) {
        out.append("long inputBase=geometry[0],weightBase=geometry[1],outputBase=geometry[2];\n")
                .append("long batches=geometry[").append(ie).append("],inputChannels=geometry[")
                .append(ie + 1).append("];")
                .append("long inputHeight=geometry[").append(ie + (s.conv3d() ? 3 : 2))
                .append("],inputWidth=geometry[").append(ie + (s.conv3d() ? 4 : 3)).append("];\n")
                .append("long inputStrideN=geometry[").append(is).append("],inputStrideC=geometry[")
                .append(is + 1).append("],inputStrideH=geometry[")
                .append(is + (s.conv3d() ? 3 : 2)).append("];\n")
                .append("long channelsPerGroup=geometry[").append(we + 1).append("],kernelHeight=geometry[")
                .append(we + (s.conv3d() ? 3 : 2)).append("],kernelWidth=geometry[")
                .append(we + (s.conv3d() ? 4 : 3)).append("];\n")
                .append("long weightStrideO=geometry[").append(ws).append("],weightStrideC=geometry[")
                .append(ws + 1).append("],weightStrideH=geometry[")
                .append(ws + (s.conv3d() ? 3 : 2)).append("],weightStrideW=geometry[")
                .append(ws + (s.conv3d() ? 4 : 3)).append("];\n")
                .append("long outputChannels=geometry[").append(oe + 1).append("],outputHeight=geometry[")
                .append(oe + (s.conv3d() ? 3 : 2)).append("],outputWidth=geometry[")
                .append(oe + (s.conv3d() ? 4 : 3)).append("];\n")
                .append("long outputStrideN=geometry[").append(os).append("],outputStrideC=geometry[")
                .append(os + 1).append("],outputStrideH=geometry[")
                .append(os + (s.conv3d() ? 3 : 2)).append("];\n");
        if (s.conv3d()) {
            out.append("long inputDepth=geometry[").append(ie + 2).append("],inputStrideD=geometry[")
                    .append(is + 2).append("],kernelDepth=geometry[").append(we + 2)
                    .append("],weightStrideD=geometry[").append(ws + 2)
                    .append("],outputDepth=geometry[").append(oe + 2)
                    .append("],outputStrideD=geometry[").append(os + 2).append("];\n");
        }
        out.append("int lanes=").append(s.f32() ? "FloatVector" : "DoubleVector")
                .append(".SPECIES_PREFERRED.length();\n");
    }

    private static void appendColdDecode(StringBuilder out, Spec s) {
        out.append("long remaining=start,firstOw=remaining%outputWidth;remaining/=outputWidth;\n")
                .append("long firstOh=remaining%outputHeight;remaining/=outputHeight;\n");
        if (s.conv3d()) {
            out.append("long firstOd=remaining%outputDepth;remaining/=outputDepth;\n");
        }
        out.append("long firstOc=remaining%outputChannels,firstN=remaining/outputChannels;\n")
                .append("long ordinal=start;\n");
    }

    private static void appendNestedLoops(StringBuilder out, Spec s, Names n) {
        out.append("for(long batch=firstN;batch<batches&&ordinal<end;batch++){\n")
                .append("long ocStart=batch==firstN?firstOc:0;\n")
                .append("for(long oc=ocStart;oc<outputChannels&&ordinal<end;oc++){\n")
                .append("long group=oc/(outputChannels/").append(s.groups()).append("L);\n")
                .append("long inputGroupBase=inputBase+batch*inputStrideN+group*channelsPerGroup*inputStrideC;\n")
                .append("long weightOcBase=weightBase+oc*weightStrideO;\n")
                .append("long outputOcBase=outputBase+batch*outputStrideN+oc*outputStrideC;\n");
        if (s.conv3d()) {
            out.append("long odStart=batch==firstN&&oc==firstOc?firstOd:0;\n")
                    .append("for(long od=odStart;od<outputDepth&&ordinal<end;od++){\n");
        }
        String firstOuter = s.conv3d()
                ? "batch==firstN&&oc==firstOc&&od==firstOd" : "batch==firstN&&oc==firstOc";
        out.append("long ohStart=").append(firstOuter).append("?firstOh:0;\n")
                .append("for(long oh=ohStart;oh<outputHeight&&ordinal<end;oh++){\n")
                .append("long ow=").append(firstOuter).append("&&oh==firstOh?firstOw:0;\n")
                .append("long inputRowOrigin=inputGroupBase+")
                .append(s.conv3d() ? "od*inputStrideD+" : "").append("oh*inputStrideH;\n")
                .append("long inputWidthCursor=inputRowOrigin+ow-").append(s.paddingWidth()).append("L;\n")
                .append("long outputCursor=outputOcBase+")
                .append(s.conv3d() ? "od*outputStrideD+" : "").append("oh*outputStrideH+ow;\n")
                .append("while(ow<outputWidth&&ordinal<end){\n")
                .append("boolean full=ow-").append(s.paddingWidth()).append("L>=0&&ow-")
                .append(s.paddingWidth()).append("L+lanes+kernelWidth-2<inputWidth")
                .append("&&ow+lanes<=outputWidth&&ordinal+lanes<=end;\n");
        appendCell(out, s, n);
        out.append("}\n}\n");
        if (s.conv3d()) out.append("}\n");
        out.append("}\n}\n");
    }

    private static void appendCell(StringBuilder out, Spec s, Names n) {
        String depthLoop = s.conv3d() ? "for(long kd=0;kd<kernelDepth;kd++)" : "";
        String inputDepth = s.conv3d() ? "+kd*inputStrideD" : "";
        String weightDepth = s.conv3d() ? "+kd*weightStrideD" : "";
        out.append("if(full){\n").append(n.vector).append(" accumulator=").append(n.vector)
                .append(".zero(").append(n.vector).append(".SPECIES_PREFERRED);\n")
                .append("for(long channel=0;channel<channelsPerGroup;channel++)")
                .append(depthLoop).append("for(long kh=0;kh<kernelHeight;kh++){\n")
                .append("long inputKernelBase=inputWidthCursor+channel*inputStrideC")
                .append(inputDepth).append("+kh*inputStrideH;\n")
                .append("long weightKernelBase=weightOcBase+channel*weightStrideC")
                .append(weightDepth).append("+kh*weightStrideH;\n")
                .append("for(long kw=0;kw<kernelWidth;kw++){")
                .append(n.vector).append(" inputVector=").append(vectorLoad(s, n,
                        "inputKernelBase+kw")).append(';')
                .append(n.vector).append(" weightVector=").append(n.vector).append(".broadcast(")
                .append(n.vector).append(".SPECIES_PREFERRED,")
                .append(load(s.weightSegment(), "weight", "weightKernelBase+kw*weightStrideW"))
                .append(");accumulator=accumulator.add(inputVector.mul(weightVector));}\n}\n")
                .append(vectorStore(s, n, "accumulator", "outputCursor")).append(';')
                .append("ow+=lanes;ordinal+=lanes;inputWidthCursor+=lanes;outputCursor+=lanes;\n")
                .append("}else{").append(n.scalar).append(" accumulator=0;\n")
                .append("for(long channel=0;channel<channelsPerGroup;channel++)")
                .append(depthLoop).append("for(long kh=0;kh<kernelHeight;kh++)")
                .append("for(long kw=0;kw<kernelWidth;kw++){long iw=ow-")
                .append(s.paddingWidth()).append("L+kw;")
                .append(n.scalar).append(" inputValue=iw<0||iw>=inputWidth?0:")
                .append(load(s.inputSegment(), "input", "inputWidthCursor+channel*inputStrideC"
                        + inputDepth + "+kh*inputStrideH+kw")).append(';')
                .append(n.scalar).append(" weightValue=")
                .append(load(s.weightSegment(), "weight", "weightOcBase+channel*weightStrideC"
                        + weightDepth + "+kh*weightStrideH+kw*weightStrideW")).append(';')
                .append("accumulator=accumulator+inputValue*weightValue;}\n")
                .append(store(s.outputSegment(), "output", "outputCursor", "accumulator")).append(';')
                .append("ow++;ordinal++;inputWidthCursor++;outputCursor++;}\n");
    }

    private static String vectorLoad(Spec s, Names n, String index) {
        return s.inputSegment()
                ? n.vector + ".fromMemorySegment(" + n.vector + ".SPECIES_PREFERRED,input,("
                        + index + ")*" + n.bytes + "L,nativeOrder)"
                : n.vector + ".fromArray(" + n.vector + ".SPECIES_PREFERRED,input,(int)("
                        + index + "))";
    }

    private static String vectorStore(Spec s, Names n, String value, String index) {
        return s.outputSegment() ? value + ".intoMemorySegment(output,(" + index + ")*"
                + n.bytes + "L,nativeOrder)" : value + ".intoArray(output,(int)(" + index + "))";
    }

    private static String load(boolean segment, String carrier, String index) {
        return segment ? carrier + ".getAtIndex(nativeLayout," + index + ')'
                : carrier + "[(int)(" + index + ")]";
    }

    private static String store(boolean segment, String carrier, String index, String value) {
        return segment ? carrier + ".setAtIndex(nativeLayout," + index + ',' + value + ')'
                : carrier + "[(int)(" + index + ")]=" + value;
    }

    private record Names(String scalar, String vector, String inputCarrier, String weightCarrier,
            String outputCarrier, int bytes) {
        Names(Spec s) {
            this(s.f32() ? "float" : "double", s.f32() ? "FloatVector" : "DoubleVector",
                    s.inputSegment() ? "MemorySegment" : (s.f32() ? "float[]" : "double[]"),
                    s.weightSegment() ? "MemorySegment" : (s.f32() ? "float[]" : "double[]"),
                    s.outputSegment() ? "MemorySegment" : (s.f32() ? "float[]" : "double[]"),
                    s.f32() ? Float.BYTES : Double.BYTES);
        }
    }

    private static final class Source extends SimpleJavaFileObject {
        private final String text;
        Source(String text) {
            super(URI.create("string:///" + SIMPLE_NAME + ".java"), Kind.SOURCE);
            this.text = text;
        }
        @Override public CharSequence getCharContent(boolean ignoreEncodingErrors) { return text; }
    }
}
