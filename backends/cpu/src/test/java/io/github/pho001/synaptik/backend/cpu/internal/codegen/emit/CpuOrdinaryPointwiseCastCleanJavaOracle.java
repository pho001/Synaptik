package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import javax.tools.DiagnosticCollector;
import javax.tools.FileObject;
import javax.tools.ForwardingJavaFileManager;
import javax.tools.JavaCompiler;
import javax.tools.JavaFileManager;
import javax.tools.JavaFileObject;
import javax.tools.StandardJavaFileManager;
import javax.tools.ToolProvider;

/**
 * Compiles typed, test-only clean-Java entry skeletons for the finite ordinary pointwise/CAST
 * inventory. Every row receives an independently authored primitive operation and store body.
 *
 * <p>The semantic bodies are emitted directly into each typed entry, with no Synaptik helper or
 * generic carrier bridge in their hot loops. There are no semantic scaffolds: an unknown row
 * fails witness generation rather than silently receiving an empty range loop.</p>
 */
final class CpuOrdinaryPointwiseCastCleanJavaOracle {
    private static final String BINARY = "io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.generated.OrdinaryPointwiseCastCleanJava";

    record Row(String owner, String operation, String descriptor, String carriers, String access,
               String selectedStrategy, String sourceType, String targetType) {
        Row(String owner, String operation, String descriptor, String carriers, String access,
                String selectedStrategy) { this(owner, operation, descriptor, carriers, access,
                        selectedStrategy, null, null); }
    }
    record Compilation(byte[] bytes, Map<String, String> methods) { }

    private CpuOrdinaryPointwiseCastCleanJavaOracle() { }

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
            assertTrue(compiled, "typed clean-Java ABI compilation: " + diagnostics.getDiagnostics());
        } catch (IOException failure) {
            throw new AssertionError("cannot compile clean-Java ABI witnesses", failure);
        }
        ByteFile file = output.get(BINARY);
        assertNotNull(file, "compiler did not emit clean-Java class");
        Map<String, String> methods = new LinkedHashMap<>();
        for (int index = 0; index < rows.size(); index++) methods.put(rows.get(index).owner(), "entry" + index);
        return new Compilation(file.bytes(), Map.copyOf(methods));
    }

    private static String source(List<Row> rows) {
        StringBuilder java = new StringBuilder("package ")
                .append(BINARY.substring(0, BINARY.lastIndexOf('.'))).append(";\n")
                .append("final class OrdinaryPointwiseCastCleanJava {\n");
        for (int index = 0; index < rows.size(); index++) {
            List<String> parameters = parameters(rows.get(index).descriptor());
            java.append("static void entry").append(index).append('(');
            for (int parameter = 0; parameter < parameters.size(); parameter++) {
                if (parameter != 0) java.append(',');
                java.append(parameters.get(parameter)).append(" p").append(parameter);
            }
            if ("CAST".equals(rows.get(index).operation()) && rows.get(index).sourceType() != null) {
                castBody(java, rows.get(index), parameters.size());
            } else if (semanticBinary(rows.get(index).operation()) && rows.get(index).sourceType() != null) {
                binaryBody(java, rows.get(index), parameters.size());
            } else if (semanticUnaryOrClassification(rows.get(index).operation())
                    && rows.get(index).sourceType() != null) {
                unaryOrClassificationBody(java, rows.get(index), parameters.size());
            } else if (semanticLogicalOrSelection(rows.get(index).operation())
                    && rows.get(index).sourceType() != null) {
                logicalOrSelectionBody(java, rows.get(index), parameters.size());
            } else {
                throw new AssertionError("unimplemented ordinary pointwise/CAST semantic row: "
                        + rows.get(index).owner() + " " + rows.get(index).operation());
            }
        }
        return java.append('}').toString();
    }

    private static boolean semanticBinary(String operation) {
        return switch (operation) {
            case "ADD", "SUB", "MUL", "DIV", "POW", "MIN", "MAX", "GREATER_THAN",
                    "GREATER_OR_EQUAL", "LESS_THAN", "LESS_OR_EQUAL", "EQUAL", "NOT_EQUAL" -> true;
            default -> false;
        };
    }

    private static boolean semanticUnaryOrClassification(String operation) {
        return switch (operation) {
            case "ABS", "NEG", "EXP", "EXPM1", "LOG", "LOG1P", "SQRT", "RECIPROCAL",
                    "RSQRT", "FLOOR", "CEIL", "SIGN", "RELU", "SIGMOID", "TANH",
                    "GELU_EXACT", "GELU_TANH_APPROXIMATION", "SILU", "ERF", "IS_FINITE",
                    "IS_NAN", "IS_INF" -> true;
            default -> false;
        };
    }

    private static boolean semanticLogicalOrSelection(String operation) {
        return switch (operation) {
            case "LOGICAL_AND", "LOGICAL_OR", "LOGICAL_NOT", "WHERE" -> true;
            default -> false;
        };
    }

    /* BOOL inputs are canonical 0/1 representations. WHERE follows the production scalar
       branch order exactly: nonzero selects ordered input one, otherwise input two. BFLOAT16
       branches are copied as raw represented short bits, never decoded and repacked. */
    private static void logicalOrSelectionBody(StringBuilder java, Row row, int parameterCount) {
        List<String> parameters = parameters(row.descriptor());
        switch (row.operation()) {
            case "LOGICAL_AND", "LOGICAL_OR" -> {
                assertTrue(parameterCount == 6, "binary logical ABI: " + row.descriptor());
                boolean leftSegment = parameters.get(0).equals("java.lang.foreign.MemorySegment");
                boolean rightSegment = parameters.get(1).equals("java.lang.foreign.MemorySegment");
                boolean outputSegment = parameters.get(2).equals("java.lang.foreign.MemorySegment");
                java.append("){for(long cursor=p4;cursor<p5;cursor++){long left=p3[2]+(cursor-p4)*p3[5];")
                        .append("long right=p3[3]+(cursor-p4)*p3[6];long out=p3[4]+(cursor-p4)*p3[7];")
                        .append("int leftValue=").append(loadFrom("BOOL", leftSegment, "p0", "left"))
                        .append(";int rightValue=").append(loadFrom("BOOL", rightSegment, "p1", "right"))
                        .append(";int converted=leftValue")
                        .append(row.operation().equals("LOGICAL_AND") ? "&rightValue;" : "|rightValue;")
                        .append(storeTo("BOOL", outputSegment, "p2", "out", "converted"))
                        .append("}}\n");
            }
            case "LOGICAL_NOT" -> {
                assertTrue(parameterCount == 5, "unary logical ABI: " + row.descriptor());
                boolean inputSegment = parameters.get(0).equals("java.lang.foreign.MemorySegment");
                boolean outputSegment = parameters.get(1).equals("java.lang.foreign.MemorySegment");
                java.append("){for(long cursor=p3;cursor<p4;cursor++){long in=p2[2]+(cursor-p3)*p2[4];")
                        .append("long out=p2[3]+(cursor-p3)*p2[5];int converted=")
                        .append(loadFrom("BOOL", inputSegment, "p0", "in"))
                        .append("^1;").append(storeTo("BOOL", outputSegment, "p1", "out", "converted"))
                        .append("}}\n");
            }
            case "WHERE" -> {
                assertTrue(parameterCount == 7, "WHERE ABI: " + row.descriptor());
                String branchType = row.targetType();
                assertTrue(branchType != null && !branchType.equals("BOOL"), "WHERE branch type: " + row.owner());
                boolean conditionSegment = parameters.get(0).equals("java.lang.foreign.MemorySegment");
                boolean trueSegment = parameters.get(1).equals("java.lang.foreign.MemorySegment");
                boolean falseSegment = parameters.get(2).equals("java.lang.foreign.MemorySegment");
                boolean outputSegment = parameters.get(3).equals("java.lang.foreign.MemorySegment");
                java.append("){for(long cursor=p5;cursor<p6;cursor++){long condition=p4[2]+(cursor-p5)*p4[6];")
                        .append("long whenTrue=p4[3]+(cursor-p5)*p4[7];long whenFalse=p4[4]+(cursor-p5)*p4[8];")
                        .append("long out=p4[5]+(cursor-p5)*p4[9];int selected=")
                        .append(loadFrom("BOOL", conditionSegment, "p0", "condition"))
                        .append("; ").append(javaTypeFor(branchType)).append(" converted=selected==0?")
                        .append(loadFrom(branchType, falseSegment, "p2", "whenFalse"))
                        .append(":").append(loadFrom(branchType, trueSegment, "p1", "whenTrue")).append(";")
                        .append(storeTo(branchType, outputSegment, "p3", "out", "converted"))
                        .append("}}\n");
            }
            default -> throw new AssertionError(row.operation());
        }
    }

    /* These bodies deliberately reproduce the scalar generated algorithm.  The numerical
       statements are emitted into every typed entry instead of calling a Synaptik/reference
       kernel or a generic carrier bridge. */
    private static void unaryOrClassificationBody(StringBuilder java, Row row, int parameterCount) {
        List<String> parameters = parameters(row.descriptor());
        assertTrue(parameterCount == 5, "unary pointwise ABI: " + row.descriptor());
        String source = row.sourceType();
        boolean inputSegment = parameters.getFirst().equals("java.lang.foreign.MemorySegment");
        boolean outputSegment = parameters.get(1).equals("java.lang.foreign.MemorySegment");
        java.append("){for(long cursor=p3;cursor<p4;cursor++){long in=p2[2]+(cursor-p3)*p2[4];"
                ).append("long out=p2[3]+(cursor-p3)*p2[5];");
        String loaded = load(source, inputSegment, "in");
        String value = source.equals("FLOAT64") ? loaded
                : source.equals("BFLOAT16") ? "Float.intBitsToFloat(((" + loaded + ")&65535)<<16)" : loaded;
        if (row.operation().startsWith("IS_")) {
            String predicate = switch (row.operation()) {
                case "IS_FINITE" -> "Float.isFinite(" + value + ")";
                case "IS_NAN" -> "Float.isNaN(" + value + ")";
                case "IS_INF" -> "Float.isInfinite(" + value + ")";
                default -> throw new AssertionError(row.operation());
            };
            if (source.equals("FLOAT64")) predicate = predicate.replace("Float.", "Double.");
            java.append("int converted=").append(predicate).append("?1:0;")
                    .append(store("BOOL", outputSegment, "out", "converted"));
        } else {
            boolean directFloat = !source.equals("FLOAT64") && switch (row.operation()) {
                case "ABS", "NEG", "RECIPROCAL", "SIGN", "RELU" -> true;
                default -> false;
            };
            if (directFloat) {
                java.append("float value=(float)").append(value).append(";float computed;");
                switch (row.operation()) {
                    case "ABS" -> java.append("computed=Math.abs(value);");
                    case "NEG" -> java.append("computed=-value;");
                    case "RECIPROCAL" -> java.append("computed=1.0f/value;");
                    case "SIGN" -> java.append("computed=Math.signum(value);");
                    case "RELU" -> java.append("computed=Math.max(+0.0f,value);");
                    default -> throw new AssertionError(row.operation());
                }
                if (source.equals("BFLOAT16")) {
                    java.append("float narrowed=computed;int bits=Float.floatToRawIntBits(narrowed);int converted=((bits&0x7fffffff)>0x7f800000)?0x7fc0:((bits+0x7fff+((bits>>>16)&1))>>>16);");
                } else java.append("float converted=computed;");
                java.append(store(source, outputSegment, "out", "converted"));
                java.append("}}\n");
                return;
            }
            java.append("double value=(double)").append(value).append(";double computed;");
            switch (row.operation()) {
                case "ABS" -> java.append("computed=Math.abs(value);");
                case "NEG" -> java.append("computed=-value;");
                case "EXP" -> java.append("computed=StrictMath.exp(value);");
                case "EXPM1" -> java.append("computed=StrictMath.expm1(value);");
                case "LOG" -> java.append("computed=StrictMath.log(value);");
                case "LOG1P" -> java.append("computed=StrictMath.log1p(value);");
                case "SQRT" -> java.append("computed=StrictMath.sqrt(value);");
                case "RECIPROCAL" -> java.append("computed=1.0d/value;");
                case "RSQRT" -> java.append("computed=1.0d/StrictMath.sqrt(value);");
                case "FLOOR" -> java.append("computed=StrictMath.floor(value);");
                case "CEIL" -> java.append("computed=StrictMath.ceil(value);");
                case "SIGN" -> java.append("computed=Math.signum(value);");
                case "RELU" -> java.append("computed=Math.max(+0.0d,value);");
                case "TANH" -> java.append("computed=StrictMath.tanh(value);");
                case "SIGMOID" -> java.append("if(value<0.0d){double e=StrictMath.exp(value);computed=e/(1.0d+e);}else{computed=1.0d/(1.0d+StrictMath.exp(-value));}");
                case "SILU" -> java.append("if(value==Double.NEGATIVE_INFINITY){computed=-0.0d;}else if(value<0.0d){double e=StrictMath.exp(value);computed=value*e/(1.0d+e);}else{computed=value/(1.0d+StrictMath.exp(-value));}");
                case "ERF" -> appendErf(java, "value", "computed");
                case "GELU_EXACT" -> {
                    java.append("if(value==Double.NEGATIVE_INFINITY){computed=-0.0d;}else{");
                    java.append("double erf;");
                    appendErf(java, "value/Math.sqrt(2.0d)", "erf");
                    java.append("computed=0.5d*value*(1.0d+erf);}");
                }
                case "GELU_TANH_APPROXIMATION" -> java.append("if(value==Double.NEGATIVE_INFINITY){computed=-0.0d;}else{double cube=value*value*value;computed=0.5d*value*(1.0d+StrictMath.tanh(Math.sqrt(2.0d/Math.PI)*value*(1.0d+0.044715d*cube)));}");
                default -> throw new AssertionError(row.operation());
            }
            String converted = source.equals("FLOAT64") ? "computed" : "(float)computed";
            if (source.equals("BFLOAT16")) {
                java.append("float narrowed=(float)computed;int bits=Float.floatToRawIntBits(narrowed);int converted=((bits&0x7fffffff)>0x7f800000)?0x7fc0:((bits+0x7fff+((bits>>>16)&1))>>>16);");
            } else java.append(javaTypeFor(source)).append(" converted=").append(converted).append(';');
            java.append(store(source, outputSegment, "out", "converted"));
        }
        java.append("}}\n");
    }

    private static void appendErf(StringBuilder java, String value, String result) {
        java.append("double erfValue=").append(value).append(";")
                .append("if(Double.isNaN(erfValue)){ ").append(result).append("=Double.NaN;}else if(erfValue==0.0d){")
                .append(result).append("=erfValue;}else if(erfValue==Double.POSITIVE_INFINITY){").append(result)
                .append("=1.0d;}else if(erfValue==Double.NEGATIVE_INFINITY){").append(result).append("=-1.0d;}else{")
                .append("double x=Math.abs(erfValue);double magnitude;")
                .append("if(x<=1.0d){double z=x*x;double p=(((((9.60497373987051638749E0*z+9.00260197203842689217E1)*z+2.23200534594684319226E3)*z+7.00332514112805075473E3)*z+5.55923013010394962768E4));double q=((((((z+3.35617141647503099647E1)*z+5.21357949780152679795E2)*z+4.59432382970980127987E3)*z+2.26290000613890934246E4)*z+4.92673942608635921086E4));magnitude=x*p/q;}")
                .append("else{double e=Math.exp(-x*x);double p;double q;if(x<8.0d){p=((((((((2.46196981473530512524E-10*x+5.64189564831068821977E-1)*x+7.46321056442269912687E0)*x+4.86371970985681366614E1)*x+1.96520832956077098242E2)*x+5.26445194995477358631E2)*x+9.34528527171957607540E2)*x+1.02755188689515710272E3)*x+5.57535335369399327526E2);q=((((((((x+1.32281951154744992508E1)*x+8.67072140885989742329E1)*x+3.54937778887819891062E2)*x+9.75708501743205489753E2)*x+1.82390916687909736289E3)*x+2.24633760818710981792E3)*x+1.65666309194161350182E3)*x+5.57535340817727675546E2);}else{p=(((((5.64189583547755073984E-1*x+1.27536670759978104416E0)*x+5.01905042251180477414E0)*x+6.16021097993053585195E0)*x+7.40974269950448939160E0)*x+2.97886665372100240670E0);q=((((((x+2.26052863220117276590E0)*x+9.39603524938001434673E0)*x+1.20489539808096656605E1)*x+1.70814450747565897222E1)*x+9.60896809063285878198E0)*x+3.36907645100081516050E0);}magnitude=1.0d-e*p/q;}")
                .append(result).append("=Math.copySign(magnitude,erfValue);}");
    }

    /* The ordinary matrix fixtures are rank-one. p3 contains the cold-bound bases for the two
       inputs and output, followed by their effective strides; p4/p5 are global logical bounds. */
    private static void binaryBody(StringBuilder java, Row row, int parameterCount) {
        List<String> parameters = parameters(row.descriptor());
        assertTrue(parameterCount == 6, "binary pointwise ABI: " + row.descriptor());
        String type = row.sourceType();
        assertTrue(type != null, "semantic binary row has source type: " + row.owner());
        boolean firstSegment = parameters.getFirst().equals("java.lang.foreign.MemorySegment");
        boolean secondSegment = parameters.get(1).equals("java.lang.foreign.MemorySegment");
        boolean outputSegment = parameters.get(2).equals("java.lang.foreign.MemorySegment");
        java.append("){for(long cursor=p4;cursor<p5;cursor++){long left=p3[2]+(cursor-p4)*p3[5];")
                .append("long right=p3[3]+(cursor-p4)*p3[6];long out=p3[4]+(cursor-p4)*p3[7];");
        String left = loadFrom(type, firstSegment, "p0", "left");
        String right = loadFrom(type, secondSegment, "p1", "right");
        if (row.operation().startsWith("GREATER") || row.operation().startsWith("LESS")
                || row.operation().equals("EQUAL") || row.operation().equals("NOT_EQUAL")) {
            String comparisonType = type;
            String comparisonLeft = comparisonType.equals("BFLOAT16")
                    ? "Float.intBitsToFloat(((" + left + ")&65535)<<16)" : left;
            String comparisonRight = comparisonType.equals("BFLOAT16")
                    ? "Float.intBitsToFloat(((" + right + ")&65535)<<16)" : right;
            String operator = switch (row.operation()) {
                case "GREATER_THAN" -> ">"; case "GREATER_OR_EQUAL" -> ">=";
                case "LESS_THAN" -> "<"; case "LESS_OR_EQUAL" -> "<=";
                case "EQUAL" -> "=="; case "NOT_EQUAL" -> "!="; default -> throw new AssertionError(row.operation());
            };
            java.append("int converted=(").append(comparisonLeft).append(operator)
                    .append(comparisonRight).append(")?1:0;")
                    .append(storeTo("BOOL", outputSegment, "p2", "out", "converted"));
        } else {
            String valueType = type;
            String arithmeticLeft = valueType.equals("BFLOAT16")
                    ? "Float.intBitsToFloat(((" + left + ")&65535)<<16)" : left;
            String arithmeticRight = valueType.equals("BFLOAT16")
                    ? "Float.intBitsToFloat(((" + right + ")&65535)<<16)" : right;
            String expression = switch (row.operation()) {
                case "ADD" -> arithmeticLeft + "+" + arithmeticRight;
                case "SUB" -> arithmeticLeft + "-" + arithmeticRight;
                case "MUL" -> arithmeticLeft + "*" + arithmeticRight;
                case "DIV" -> arithmeticLeft + "/" + arithmeticRight;
                case "MIN" -> (valueType.equals("INT32") ? "Integer.min(" : valueType.equals("INT64") ? "Long.min(" : "Math.min(") + arithmeticLeft + "," + arithmeticRight + ")";
                case "MAX" -> (valueType.equals("INT32") ? "Integer.max(" : valueType.equals("INT64") ? "Long.max(" : "Math.max(") + arithmeticLeft + "," + arithmeticRight + ")";
                case "POW" -> valueType.equals("FLOAT64") ? "StrictMath.pow(" + arithmeticLeft + "," + arithmeticRight + ")"
                        : "(float)StrictMath.pow(" + arithmeticLeft + "," + arithmeticRight + ")";
                default -> throw new AssertionError(row.operation());
            };
            if (valueType.equals("BFLOAT16")) {
                java.append("float computed=").append(expression).append(";int bits=Float.floatToRawIntBits(computed);")
                        .append("int converted=((bits&0x7fffffff)>0x7f800000)?0x7fc0:((bits+0x7fff+((bits>>>16)&1))>>>16);");
            } else java.append(javaTypeFor(valueType)).append(" converted=").append(expression).append(';');
            java.append(storeTo(valueType, outputSegment, "p2", "out", "converted"));
        }
        java.append("}}\n");
    }


    /* The pointwise CAST fixtures are rank-one.  As with production cold binding, p2 contains
       range-positioned bases and p4/p5 the effective source/output strides. */
    private static void castBody(StringBuilder java, Row row, int parameterCount) {
        String source = row.sourceType(), target = row.targetType();
        boolean inputSegment = parameters(row.descriptor()).getFirst().equals("java.lang.foreign.MemorySegment");
        boolean outputSegment = parameters(row.descriptor()).get(1).equals("java.lang.foreign.MemorySegment");
        java.append("){for(long cursor=p").append(parameterCount - 2).append(";cursor<p")
                .append(parameterCount - 1).append(";cursor++){long in=p2[2]+(cursor-p")
                .append(parameterCount - 2).append(")*p2[4];long out=p2[3]+(cursor-p")
                .append(parameterCount - 2).append(")*p2[5];");
        String load = load(source, inputSegment, "in");
        java.append(javaTypeFor(source)).append(" value=").append(load).append(';');
        appendConversion(java, source, target);
        java.append(store(target, outputSegment, "out", "converted"));
        java.append("}}\n");
    }

    private static String javaTypeFor(String type) { return switch (type) {
        case "FLOAT64" -> "double"; case "FLOAT32" -> "float"; case "BFLOAT16", "INT32", "BOOL" -> "int";
        case "INT64" -> "long"; default -> throw new AssertionError(type); }; }
    private static String load(String type, boolean segment, String address) {
        return loadFrom(type, segment, "p0", address);
    }
    private static String loadFrom(String type, boolean segment, String carrier, String address) {
        String layout = switch (type) { case "FLOAT64" -> "java.lang.foreign.ValueLayout.JAVA_DOUBLE_UNALIGNED";
            case "FLOAT32" -> "java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED";
            case "BFLOAT16" -> "java.lang.foreign.ValueLayout.JAVA_SHORT_UNALIGNED";
            case "INT64" -> "java.lang.foreign.ValueLayout.JAVA_LONG_UNALIGNED";
            case "INT32" -> "java.lang.foreign.ValueLayout.JAVA_INT_UNALIGNED";
            case "BOOL" -> "java.lang.foreign.ValueLayout.JAVA_BYTE"; default -> throw new AssertionError(type); };
        String array = switch (type) { case "FLOAT64" -> "((double[])" + carrier + ")[(int)" + address + "]";
            case "FLOAT32" -> "((float[])" + carrier + ")[(int)" + address + "]"; case "BFLOAT16" -> "((short[])" + carrier + ")[(int)" + address + "]";
            case "INT64" -> "((long[])" + carrier + ")[(int)" + address + "]"; case "INT32" -> "((int[])" + carrier + ")[(int)" + address + "]";
            case "BOOL" -> "((byte[])" + carrier + ")[(int)" + address + "]"; default -> throw new AssertionError(type); };
        return segment ? "((java.lang.foreign.MemorySegment)" + carrier + ").getAtIndex(" + layout + "," + address + ")" : array;
    }
    private static String store(String type, boolean segment, String address, String value) {
        return storeTo(type, segment, "p1", address, value);
    }
    private static String storeTo(String type, boolean segment, String carrier, String address, String value) {
        String layout = switch (type) { case "FLOAT64" -> "java.lang.foreign.ValueLayout.JAVA_DOUBLE_UNALIGNED";
            case "FLOAT32" -> "java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED"; case "BFLOAT16" -> "java.lang.foreign.ValueLayout.JAVA_SHORT_UNALIGNED";
            case "INT64" -> "java.lang.foreign.ValueLayout.JAVA_LONG_UNALIGNED"; case "INT32" -> "java.lang.foreign.ValueLayout.JAVA_INT_UNALIGNED";
            case "BOOL" -> "java.lang.foreign.ValueLayout.JAVA_BYTE"; default -> throw new AssertionError(type); };
        String array = switch (type) { case "FLOAT64" -> "((double[])" + carrier + ")[(int)" + address + "]=" + value + ";";
            case "FLOAT32" -> "((float[])" + carrier + ")[(int)" + address + "]=" + value + ";"; case "BFLOAT16" -> "((short[])" + carrier + ")[(int)" + address + "]=(short)" + value + ";";
            case "INT64" -> "((long[])" + carrier + ")[(int)" + address + "]=" + value + ";"; case "INT32" -> "((int[])" + carrier + ")[(int)" + address + "]=" + value + ";";
            case "BOOL" -> "((byte[])" + carrier + ")[(int)" + address + "]=(byte)" + value + ";"; default -> throw new AssertionError(type); };
        String segmentValue = switch (type) { case "BFLOAT16" -> "(short)" + value;
            case "BOOL" -> "(byte)" + value; default -> value; };
        return segment ? "((java.lang.foreign.MemorySegment)" + carrier + ").setAtIndex(" + layout + "," + address + "," + segmentValue + ");" : array;
    }

    /* These snippets are emitted into each hot method (not dispatched through an oracle). */
    private static void appendConversion(StringBuilder s, String source, String target) {
        if (source.equals(target)) { s.append(javaTypeFor(target)).append(" converted=value;"); return; }
        if (target.equals("BOOL")) { s.append(source.equals("BFLOAT16")
                ? "int converted=(Float.intBitsToFloat((value&65535)<<16)==0f?0:1);"
                : "int converted=(value==0?0:1);"); return; }
        if (target.equals("BFLOAT16")) { appendBfloat(s, source); return; }
        String numeric = source.equals("BFLOAT16") ? "Float.intBitsToFloat((value&65535)<<16)" : "value";
        if (target.equals("FLOAT64")) {
            if (source.equals("FLOAT32")) s.append("int nb=Float.floatToRawIntBits(value);double converted=((nb&0x7f800000)==0x7f800000&&(nb&0x7fffff)!=0)?Double.longBitsToDouble(((long)(nb&0x80000000)<<32)|0x7ff0000000000000L|((long)(nb&0x7fffff)<<29)):(double)value;");
            else if (source.equals("BFLOAT16")) s.append("int nb=value&65535;double converted=((nb&0x7f80)==0x7f80&&(nb&0x7f)!=0)?Double.longBitsToDouble(((long)(nb&0x8000)<<48)|0x7ff0000000000000L|((long)(nb&0x7f)<<45)):(double)Float.intBitsToFloat(nb<<16);");
            else s.append("double converted=(double)").append(numeric).append(";");
        } else if (target.equals("FLOAT32")) {
            if (source.equals("FLOAT64")) s.append("long db=Double.doubleToRawLongBits(value);float converted=((db&0x7ff0000000000000L)==0x7ff0000000000000L&&(db&0xfffffffffffffL)!=0)?Float.NaN:(float)value;");
            else s.append("float converted=(float)").append(numeric).append(";");
        } else if (target.equals("INT64")) s.append("long converted=(long)").append(numeric).append(";");
        else s.append("int converted=(int)").append(numeric).append(";");
    }
    private static void appendBfloat(StringBuilder s, String source) {
        if (source.equals("BOOL")) { s.append("int converted=value==0?0:0x3f80;"); return; }
        if (source.equals("INT64") || source.equals("INT32")) {
            // The Model contract rounds integer magnitude directly to BFLOAT16.  Going through
            // FLOAT32 is observably wrong at double-rounding boundaries.
            String integer = source.equals("INT64") ? "value" : "(long)value";
            s.append("long iv=").append(integer).append(";int converted;if(iv==0){converted=0;}else{int is;long im;if(iv<0){is=0x8000;im=-iv;}else{is=0;im=iv;}int ie=63-Long.numberOfLeadingZeros(im);long ir;if(ie<=7){ir=im<<(7-ie);}else{int ish=ie-7;ir=im>>>ish;long id=im&((1L<<ish)-1L);long ih=1L<<(ish-1);if(id>ih||(id==ih&&(ir&1L)!=0))ir++;}if(ir==256L){ir=128L;ie++;}converted=is|((ie+127)<<7)|((int)ir&127);}");
            return;
        }
        if (source.equals("FLOAT64")) {
            // Direct binary64 RNE packing: this intentionally does not first round to float.
            s.append("long db=Double.doubleToRawLongBits(value);int ds=(int)(db>>>48)&0x8000;int de=(int)(db>>>52)&0x7ff;long dm=db&0xfffffffffffffL;int converted;"
                    + "if(de==0x7ff){converted=dm==0?(ds|0x7f80):0x7fc0;}else if(de>1150){converted=ds|0x7f80;}else if(de<889){converted=ds;}else{int sh=de<897?942-de:45;long sig=(1L<<52)|dm;long q=(sig+(1L<<(sh-1))-1+((sig>>>sh)&1))>>>sh;"
                    + "if(de>=897&&q==256){q=128;de++;}converted=de>=897?(ds|((de-896)<<7)|((int)q&127)):(ds|((int)q));}");
            return;
        }
        String f = source.equals("FLOAT64") ? "(float)value" : source.equals("BFLOAT16") ? "Float.intBitsToFloat((value&65535)<<16)" : "(float)value";
        // F32 input and integral inputs use the contract's RNE packing.  The direct F64 row
        // intentionally evaluates binary64 first; the expression is not routed through any API.
        s.append("float cv=").append(f).append(";int cb=Float.floatToRawIntBits(cv);int converted=((cb&0x7fffffff)>0x7f800000)?0x7fc0:((cb+0x7fff+((cb>>>16)&1))>>>16);");
    }

    private static List<String> parameters(String descriptor) {
        assertTrue(descriptor.startsWith("(") && descriptor.endsWith(")V"), "void entry ABI: " + descriptor);
        List<String> result = new ArrayList<>();
        for (int cursor = 1; descriptor.charAt(cursor) != ')';) {
            int start = cursor;
            while (descriptor.charAt(cursor) == '[') cursor++;
            char tag = descriptor.charAt(cursor++);
            if (tag == 'L') { while (descriptor.charAt(cursor++) != ';') { } }
            String token = descriptor.substring(start, cursor);
            result.add(javaType(token));
        }
        assertTrue(result.size() >= 2 && result.get(result.size() - 1).equals("long")
                && result.get(result.size() - 2).equals("long"), "range ABI: " + descriptor);
        return List.copyOf(result);
    }

    private static String javaType(String descriptor) {
        int arrays = 0;
        while (arrays < descriptor.length() && descriptor.charAt(arrays) == '[') arrays++;
        String base = switch (descriptor.charAt(arrays)) {
            case 'B' -> "byte"; case 'S' -> "short"; case 'I' -> "int"; case 'J' -> "long";
            case 'F' -> "float"; case 'D' -> "double";
            case 'L' -> descriptor.substring(arrays + 1, descriptor.length() - 1).replace('/', '.');
            default -> throw new AssertionError("unsupported selected ABI token " + descriptor);
        };
        return base + "[]".repeat(arrays);
    }


    private static final class Source extends Simple {
        private final String source;
        Source(String binary, String source) { super(binary, Kind.SOURCE); this.source = source; }
        @Override public CharSequence getCharContent(boolean ignoreEncodingErrors) { return source; }
    }
    private static final class ByteFile extends Simple {
        private final ByteArrayOutputStream output = new ByteArrayOutputStream();
        ByteFile(String binary) { super(binary, Kind.CLASS); }
        @Override public ByteArrayOutputStream openOutputStream() { return output; }
        byte[] bytes() { return output.toByteArray(); }
    }
    private abstract static class Simple extends javax.tools.SimpleJavaFileObject {
        Simple(String binary, Kind kind) { super(URI.create("mem:///" + binary.replace('.', '/') + kind.extension), kind); }
    }
}
