package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratorSchema;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuAttentionEvidenceTest.Mapping;
import io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuAttentionEvidenceTest.PerformanceRow;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuAttentionLowering;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.classfile.AccessFlags;
import java.lang.classfile.Attribute;
import java.lang.classfile.ClassFile;
import java.lang.classfile.ClassTransform;
import java.lang.classfile.CodeTransform;
import java.lang.classfile.MethodModel;
import java.lang.classfile.instruction.LineNumber;
import java.lang.constant.ClassDesc;
import java.lang.reflect.Method;
import java.lang.reflect.AccessFlag;
import java.net.URLClassLoader;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import org.junit.jupiter.api.Test;

/**
 * Opt-in exact 992-row, five-fork generated/direct attention performance evidence owner.
 * Synchronous compilation keeps both typed sides out of asynchronous compiler-transition windows.
 */
public final class CpuAttentionPerformanceTest {
  private static final String PACKAGE =
      "io.github.pho001.synaptik.backend.cpu.internal.codegen.emit";
  private static final long MIN_NANOS = 25_000_000L;
  private static final int WARMUPS = 5;
  private static final int ROUNDS = 9;
  private static volatile long sink;

  private CpuAttentionPerformanceTest() {}

  @Test
  void compilationOrderIsCounterbalancedForEveryForkAndRowParity() {
    for (boolean first : List.of(false, true)) {
      boolean[] orders = counterbalancedOrders(first);
      assertEquals(first, orders[0]);
      assertEquals(!first, orders[1]);
      assertEquals(2, orders.length);
    }
  }

  /** Launches exactly five Java 26 fixed-heap forks and retains all required evidence. */
  @Test
  void retainedFiveForkEvidence() throws Throwable {
    assumeTrue("true".equals(System.getenv("SYNAPTIK_CPU_ATTENTION_PERFORMANCE")));
    Path root = root();
    assertFalse(Files.exists(root), "performance evidence root must be fresh");
    Files.createDirectories(root);
    prepareDrivers(root);
    String java = Path.of(System.getProperty("java.home"), "bin", "java").toString();
    String cp = System.getProperty("java.class.path");
    StringBuilder commands = new StringBuilder();
    for (int fork = 0; fork < 5; fork++) {
      List<String> command = List.of(java, "-Xms1g", "-Xmx1g", "-Xbatch", "--add-modules",
          "jdk.incubator.vector", "-cp", cp,
          CpuAttentionPerformanceTest.class.getName(), Integer.toString(fork));
      commands.append(String.join(" ", command)).append('\n');
      Process process = new ProcessBuilder(command).inheritIO().start();
      assertEquals(0, process.waitFor(), "performance fork " + fork);
    }
    Files.writeString(root.resolve("commands.txt"), commands);
    aggregate(root);
    retainIdentity(root);
    writeManifest(root);
  }

  /**
   * Launches retained evidence, prepares drivers, or executes one immutable measured fork.
   *
   * @param args {@code launch}, {@code prepare}, or a fork index with optional smoke row range
   * @throws Throwable if generation, compilation, measurement, or retention fails
   */
  public static void main(String[] args) throws Throwable {
    if (args.length == 1 && args[0].equals("launch")) {
      new CpuAttentionPerformanceTest().retainedFiveForkEvidence();
      return;
    }
    if (args.length == 1 && args[0].equals("prepare")) {
      Path root = root();
      Files.createDirectories(root);
      prepareDrivers(root);
      return;
    }
    int fork = Integer.parseInt(args[0]);
    if (fork < 0 || fork >= 5) throw new IllegalArgumentException("fork must be zero through four");
    Path root = root();
    List<PerformanceRow> rows = CpuAttentionEvidenceTest.performanceRows();
    assertEquals(992, rows.size());
    int firstRow = args.length > 2 ? Integer.parseInt(args[1]) : 0;
    int rowLimit = args.length > 2 ? Integer.parseInt(args[2])
        : args.length > 1 ? Integer.parseInt(args[1]) : rows.size();
    int lastRow = Math.addExact(firstRow, rowLimit);
    if (firstRow < 0 || rowLimit < 1 || lastRow > rows.size())
      throw new IllegalArgumentException("row range");
    StringBuilder report = new StringBuilder();
    environment(report, fork);
    boolean accepted = false;
    try {
      for (int index = firstRow; index < lastRow; index++) {
        PerformanceRow row = rows.get(index);
        boolean[] compilationOrders = counterbalancedOrders(((fork + index) & 1) == 0);
        try (RowState first = RowState.create(root, index, row, compilationOrders[0])) {
          RowState second = first.withSwappedLabels();
          assertSame(first.generated, second.direct,
              "generated first representative must be direct second representative");
          assertSame(first.direct, second.generated,
              "direct first representative must be generated second representative");
          List<RowState> states = List.of(first, second);
          int[][] batches = new int[states.size()][];
          for (int replica = 0; replica < states.size(); replica++) {
            RowState state = states.get(replica);
            state.gate(compilationOrders[replica]);
            batches[replica] = adaptivePair(state.generated, state.generatedArgs, state.direct,
                state.directArgs, compilationOrders[replica]);
          }
          long[] generated = new long[ROUNDS], direct = new long[ROUNDS];
          Random random = new Random(0x0008_0008_2026_0831L
              ^ (long) fork * 0x9e3779b97f4a7c15L ^ index * 0xd1b54a32d192ed03L);
          for (int round = -WARMUPS; round < ROUNDS; round++) {
            boolean measuredGeneratedFirst = random.nextBoolean();
            long gt, dt;
            if (measuredGeneratedFirst) {
              gt = normalizedReplicaMean(states, batches, true, random);
              dt = normalizedReplicaMean(states, batches, false, random);
            } else {
              dt = normalizedReplicaMean(states, batches, false, random);
              gt = normalizedReplicaMean(states, batches, true, random);
            }
            if (round >= 0) {
              generated[round] = gt;
              direct[round] = dt;
              states.forEach(RowState::verifyOutputs);
            }
          }
          long gm = median(generated), dm = median(direct);
          double ratio = (double) gm / dm;
          report.append(String.format(Locale.ROOT,
              "RESULT,%s,%d,%d,%.9f,%d,%d,%s,%s%n", safe(row.name()), gm, dm, ratio,
              java.util.stream.IntStream.range(0, batches.length).map(i -> batches[i][0]).sum(),
              java.util.stream.IntStream.range(0, batches.length).map(i -> batches[i][1]).sum(),
              Arrays.toString(generated), Arrays.toString(direct)));
          if (ratio > 1.15) throw new AssertionError(row.name() + " ratio " + ratio);
        }
        System.gc();
      }
      accepted = true;
    } catch (Throwable failure) {
      report.append("FAILURE,").append(failure.getClass().getName()).append(',')
          .append(safe(failure.getMessage())).append('\n');
      throw failure;
    } finally {
      retainFork(root, accepted, fork, report.toString());
      System.out.print(report);
    }
  }

  private static void prepareDrivers(Path root) throws Exception {
    Path generated = root.resolve("generated-classes");
    Path sources = root.resolve("driver-sources");
    Path classes = root.resolve("driver-classes");
    Path generatedBundle = root.resolve("generated-driver-classes");
    Path directBundle = root.resolve("direct-driver-classes");
    Files.createDirectories(generated.resolve(PACKAGE.replace('.', '/')));
    Files.createDirectories(sources.resolve(PACKAGE.replace('.', '/')));
    Files.createDirectories(classes);
    List<PerformanceRow> rows = CpuAttentionEvidenceTest.performanceRows();
    StringBuilder inventory = new StringBuilder("index,row,key,driver,types,carriers\n");
    var sourceFiles = new ArrayList<String>();
    var generatedNames = new ArrayList<String>();
    for (int index = 0; index < rows.size(); index++) {
      PerformanceRow row = rows.get(index);
      var generatedCase = CpuAttentionEvidenceTest.generated(row.mapping(), row.masked(),
          row.causal(), row.outputs(), row.carriers());
      String binaryName = CpuGeneratorSchema.generatedBinaryName(generatedCase.specialization());
      generatedNames.add(binaryName);
      Path generatedFile = generated.resolve(binaryName.replace('.', '/') + ".class");
      Files.createDirectories(generatedFile.getParent());
      Files.write(generatedFile, generatedCase.bytes());
      String driver = "AttentionDriver_" + index;
      Path source = sources.resolve(PACKAGE.replace('.', '/')).resolve(driver + ".java");
      Files.writeString(source, driverSource(driver, binaryName, row));
      sourceFiles.add(source.toString());
      inventory.append(index).append(',').append(safe(row.name())).append(',')
          .append(generatedCase.key()).append(',').append(driver).append(',')
          .append(quote(generatedCase.types().toString())).append(',')
          .append(quote(row.carriers().toString())).append('\n');
    }
    Files.writeString(root.resolve("rows.csv"), inventory);
    Path arguments = root.resolve("javac-arguments.txt");
    StringBuilder args = new StringBuilder("--release\n26\n-d\n")
        .append(classes).append("\n-classpath\n").append(generated).append('\n');
    sourceFiles.forEach(file -> args.append(file).append('\n'));
    Files.writeString(arguments, args);
    Process compile = new ProcessBuilder(Path.of(System.getProperty("java.home"), "bin", "javac")
        .toString(), "@" + arguments).redirectErrorStream(true).start();
    byte[] output = compile.getInputStream().readAllBytes();
    if (compile.waitFor() != 0) throw new AssertionError(new String(output, StandardCharsets.UTF_8));
    assembleSymmetricDriverBundles(
        classes, generated, generatedBundle, directBundle, generatedNames);
    Files.write(root.resolve("javac-output.txt"), output);
  }

  private static void assembleSymmetricDriverBundles(Path classes, Path generated,
      Path generatedBundle, Path directBundle, List<String> generatedNames) throws Exception {
    ClassFile classFile = ClassFile.of(ClassFile.ConstantPoolSharingOption.NEW_POOL);
    Path packageRoot = classes.resolve(PACKAGE.replace('.', '/'));
    for (int index = 0; index < generatedNames.size(); index++) {
      String generatedName = generatedNames.get(index);
      Path generatedPath = Path.of(generatedName.replace('.', '/') + ".class");
      Path wrapper = packageRoot.resolve("AttentionDriver_" + index + ".class");
      Path oracle = packageRoot.resolve("Oracle_AttentionDriver_" + index + ".class");
      Path generatedWrapper = generatedBundle.resolve(PACKAGE.replace('.', '/'))
          .resolve(wrapper.getFileName());
      Path directWrapper = directBundle.resolve(PACKAGE.replace('.', '/'))
          .resolve(wrapper.getFileName());
      Files.createDirectories(generatedWrapper.getParent());
      Files.createDirectories(directWrapper.getParent());
      Files.copy(wrapper, generatedWrapper);
      Files.copy(wrapper, directWrapper);
      byte[] generatedBytes = canonicalOracleClass(classFile,
          classFile.parse(generated.resolve(generatedPath)), generatedName);
      Path generatedTarget = generatedBundle.resolve(generatedPath);
      Files.createDirectories(generatedTarget.getParent());
      Files.write(generatedTarget, generatedBytes);
      byte[] normalized = canonicalOracleClass(classFile, classFile.parse(oracle), generatedName);
      Path directTarget = directBundle.resolve(generatedPath);
      Files.createDirectories(directTarget.getParent());
      Files.write(directTarget, normalized);
      assertTrue(Arrays.equals(generatedBytes, normalized),
          "generated and javac oracle canonical classfiles differ for row " + index);
      var result = classFile.parse(normalized);
      assertEquals(1, result.methods().size());
      assertEquals("invoke", result.methods().getFirst().methodName().stringValue());
    }
  }

  private static byte[] canonicalOracleClass(ClassFile classFile,
      java.lang.classfile.ClassModel model, String binaryName) {
      var dropTopology = ClassTransform.dropping(element ->
          element instanceof AccessFlags
              || element instanceof Attribute<?>
              || element instanceof MethodModel method
                  && !method.methodName().stringValue().equals("invoke"));
      CodeTransform dropLines = (builder, element) -> {
        if (!(element instanceof LineNumber)) builder.with(element);
      };
      return classFile.transformClass(model, ClassDesc.of(binaryName), dropTopology
          .andThen(ClassTransform.transformingMethodBodies(dropLines))
          .andThen(ClassTransform.endHandler(builder -> builder.withFlags(AccessFlag.FINAL))));
  }

  private static String driverSource(String driver, String generated, PerformanceRow row) {
    List<DataType> types = CpuAttentionEvidenceTest.boundaryTypes(row.mapping(), row.masked(),
        row.outputs());
    String parameters = parameters(types, row.carriers());
    String arguments = arguments(types.size());
    return "package " + PACKAGE + ";\n"
        + "final class " + driver + " {\n"
        + "  static long invoke(" + parameters + ", int batches) {\n"
        + "    long before=System.nanoTime();\n"
        + "    for(int batch=0;batch<batches;batch++) " + generated + ".invoke(" + arguments
        + ",scratch,g,start,end);\n"
        + "    return System.nanoTime()-before;\n  }\n}\n"
        + "final class Oracle_" + driver + " {\n"
        + "  public static void invoke(" + parameters + ") {\n"
        + directBody(row, types)
        + "  }\n}\n";
  }

  private static String parameters(List<DataType> types, List<CarrierAccess> carriers) {
    var values = new ArrayList<String>();
    for (int i = 0; i < types.size(); i++)
      values.add(javaType(types.get(i), carriers.get(i)) + " b" + i);
    values.add("java.lang.foreign.MemorySegment scratch");
    values.add("long[] g");
    values.add("long start");
    values.add("long end");
    return String.join(",", values);
  }

  private static String arguments(int boundaries) {
    var values = new ArrayList<String>();
    for (int i = 0; i < boundaries; i++) values.add("b" + i);
    return String.join(",", values);
  }

  private static String directBody(PerformanceRow row, List<DataType> types) {
    Mapping mapping = row.mapping();
    DataType result = mapping.result();
    String domain = result == DataType.FLOAT64 ? "double" : "float";
    String layout = result == DataType.FLOAT64 ? "JAVA_DOUBLE_UNALIGNED" : "JAVA_FLOAT_UNALIGNED";
    int width = result == DataType.FLOAT64 ? 8 : 4;
    int q = mapping.roles().get(0), k = mapping.roles().get(1), v = mapping.roles().get(2);
    int mask = row.masked() ? mapping.unique() : -1;
    int output = mapping.unique() + (row.masked() ? 1 : 0);
    int weights = output + 1;
    StringBuilder s = new StringBuilder();
    s.append("    int rank=(int)g[6],header=12+rank;\n")
        .append("    for(long row=start;row<end;row++){\n")
        .append("      long qi=row%g[7],batch=row/g[7],qb=g[header],kb=g[header+rank+1],vb=g[header+2*(rank+1)]")
        .append(row.masked() ? ",mb=g[header+3*(rank+1)]" : "")
        .append(",ob=g[header+4*(rank+1)]")
        .append(row.outputs() == 2 ? ",wb=g[header+5*(rank+1)]" : "")
        .append(";\n")
        .append("      for(int axis=rank-3;axis>=0;axis--){long coordinate=batch%g[14+axis];batch/=g[14+axis];qb+=coordinate*g[header+1+axis];kb+=coordinate*g[header+rank+2+axis];vb+=coordinate*g[header+2*(rank+1)+1+axis]")
        .append(row.masked() ? ";mb+=coordinate*g[header+3*(rank+1)+1+axis]" : "")
        .append(";ob+=coordinate*g[header+4*(rank+1)+1+axis]")
        .append(row.outputs() == 2
            ? ";wb+=coordinate*g[header+5*(rank+1)+1+axis]" : "")
        .append(";}\n")
        .append("      qb+=qi*g[header+rank-1]")
        .append(row.masked() ? ";mb+=qi*g[header+3*(rank+1)+rank-1]" : "")
        .append(";ob+=qi*g[header+4*(rank+1)+rank-1]")
        .append(row.outputs() == 2 ? ";wb+=qi*g[header+5*(rank+1)+rank-1]" : "")
        .append(";\n")
        .append("      long eligible=0,positive=0;boolean anyNan=false,allNeg=true;\n")
        .append("      for(long j=0;j<g[8];j++){\n");
    eligibilityOpen(s, row, types, mask);
    s.append("        ").append(domain).append(" score=0;\n")
        .append("        for(long x=0;x<g[9];x++){long qa=qb+x*g[header+rank],ka=kb+j*g[header+2*rank]+x*g[header+2*rank+1];\n")
        .append("          ").append(domain).append(" qv=").append(read(types.get(q), row.carriers().get(q), q, "qa")).append(";\n")
        .append("          ").append(domain).append(" kv=").append(read(types.get(k), row.carriers().get(k), k, "ka")).append(";\n")
        .append("          score+=qv*kv;}\n")
        .append(result == DataType.FLOAT64 ? "        score*=Double.longBitsToDouble(g[12]);\n"
            : "        score*=(float)Double.longBitsToDouble(g[12]);\n")
        .append("        scratch.set(java.lang.foreign.ValueLayout.").append(layout)
        .append(",j*").append(width).append("L,score);eligible++;anyNan|=")
        .append(result == DataType.FLOAT64 ? "Double" : "Float").append(".isNaN(score);if(score==")
        .append(result == DataType.FLOAT64 ? "Double" : "Float")
        .append(".POSITIVE_INFINITY)positive++;if(score!=")
        .append(result == DataType.FLOAT64 ? "Double" : "Float")
        .append(".NEGATIVE_INFINITY)allNeg=false;\n");
    eligibilityClose(s, row);
    s.append("      }\n")
        .append("      int mode=eligible==0||allNeg?0:anyNan?1:positive>0?2:3;\n")
        .append("      if(mode==3){").append(domain).append(" max=")
        .append(result == DataType.FLOAT64 ? "Double" : "Float")
        .append(".NEGATIVE_INFINITY;\n")
        .append("        for(long j=0;j<g[8];j++){\n");
    eligibilityOpen(s, row, types, mask);
    s.append("          ").append(domain).append(" x=scratch.get(java.lang.foreign.ValueLayout.")
        .append(layout).append(",j*").append(width).append("L);if(x>max)max=x;\n");
    eligibilityClose(s, row);
    s.append("        }\n        ").append(domain).append(" sum=0;\n")
        .append("        for(long j=0;j<g[8];j++){\n");
    eligibilityOpen(s, row, types, mask);
    s.append("          ").append(domain).append(" weight=")
        .append(result == DataType.FLOAT64 ? "StrictMath.exp" : "(float)StrictMath.exp")
        .append("(scratch.get(java.lang.foreign.ValueLayout.").append(layout).append(",j*")
        .append(width).append("L)-max);scratch.set(java.lang.foreign.ValueLayout.").append(layout)
        .append(",j*").append(width).append("L,weight);sum+=weight;\n");
    eligibilityClose(s, row);
    s.append("        }\n        for(long j=0;j<g[8];j++){\n");
    eligibilityOpen(s, row, types, mask);
    s.append("          scratch.set(java.lang.foreign.ValueLayout.").append(layout).append(",j*")
        .append(width).append("L,scratch.get(java.lang.foreign.ValueLayout.").append(layout)
        .append(",j*").append(width).append("L)/sum);\n");
    eligibilityClose(s, row);
    s.append("        }}else{for(long j=0;j<g[8];j++){\n");
    eligibilityOpen(s, row, types, mask);
    s.append("        ").append(domain).append(" score=scratch.get(java.lang.foreign.ValueLayout.")
        .append(layout).append(",j*").append(width).append("L),weight=mode==1?")
        .append(result == DataType.FLOAT64 ? "Double" : "Float")
        .append(".NaN:mode==2&&score==").append(result == DataType.FLOAT64 ? "Double" : "Float")
        .append(".POSITIVE_INFINITY?").append(result == DataType.FLOAT64 ? "1d" : "1f")
        .append("/positive:").append(result == DataType.FLOAT64 ? "+0d" : "+0f")
        .append(";scratch.set(java.lang.foreign.ValueLayout.").append(layout).append(",j*")
        .append(width).append("L,weight);\n");
    eligibilityClose(s, row);
    s.append("      }}\n");
    if (row.outputs() == 2) {
      s.append("      for(long j=0;j<g[8];j++){").append(domain).append(" weight;");
      if (row.causal() || row.masked()) {
        s.append("if(").append(eligibility(row, types, mask, "j"))
            .append(")weight=scratch.get(java.lang.foreign.ValueLayout.").append(layout)
            .append(",j*").append(width).append("L);else weight=")
            .append(result == DataType.FLOAT64 ? "+0d" : "+0f").append(';');
      } else s.append("weight=scratch.get(java.lang.foreign.ValueLayout.").append(layout)
          .append(",j*").append(width).append("L);");
      s.append("long wa=wb+j*g[header+5*(rank+1)+rank];\n")
          .append(store(types.get(weights), row.carriers().get(weights), weights, "wa", "weight"))
          .append("      }\n");
    }
    s.append("      for(long d=0;d<g[10];d++){").append(domain)
        .append(" out=mode==1?").append(result == DataType.FLOAT64 ? "Double" : "Float")
        .append(".NaN:").append(result == DataType.FLOAT64 ? "+0d" : "+0f").append(";\n")
        .append("        if(mode!=0&&mode!=1)for(long j=0;j<g[8];j++){\n");
    eligibilityOpen(s, row, types, mask);
    s.append("          ").append(domain).append(" weight=scratch.get(java.lang.foreign.ValueLayout.")
        .append(layout).append(",j*").append(width).append("L);long va=vb+j*g[header+3*rank+1]+d*g[header+3*rank+2];")
        .append(domain).append(" vv=").append(read(types.get(v), row.carriers().get(v), v, "va"))
        .append(";out+=weight*vv;\n");
    eligibilityClose(s, row);
    s.append("        }long oa=ob+d*g[header+5*rank+4];\n")
        .append(store(types.get(output), row.carriers().get(output), output, "oa", "out"))
        .append("      }\n    }\n");
    return s.toString();
  }

  private static void eligibilityOpen(StringBuilder s, PerformanceRow row, List<DataType> types,
      int mask) {
    if (row.causal() || row.masked())
      s.append("        if(").append(eligibility(row, types, mask, "j")).append("){\n");
  }

  private static void eligibilityClose(StringBuilder s, PerformanceRow row) {
    if (row.causal() || row.masked()) s.append("        }\n");
  }

  private static String eligibility(PerformanceRow row, List<DataType> types, int mask,
      String j) {
    var parts = new ArrayList<String>();
    if (row.causal()) parts.add(j + "<=qi");
    if (row.masked()) {
      String address = "(mb+" + j + "*g[header+4*rank+3])";
      parts.add(read(types.get(mask), row.carriers().get(mask), mask, address) + "!=0");
    }
    return parts.isEmpty() ? "true" : String.join("&&", parts);
  }

  private static String read(DataType type, CarrierAccess carrier, int boundary, String address) {
    String raw;
    if (carrier == CarrierAccess.MEMORY_SEGMENT) {
      String layout = switch (type) {
        case BFLOAT16 -> "JAVA_SHORT_UNALIGNED";
        case FLOAT32 -> "JAVA_FLOAT_UNALIGNED";
        case FLOAT64 -> "JAVA_DOUBLE_UNALIGNED";
        case BOOL -> "JAVA_BYTE";
        default -> throw new AssertionError(type);
      };
      raw = "b" + boundary + ".get(java.lang.foreign.ValueLayout." + layout + ",(" + address
          + ")*" + type.byteWidth() + "L)";
    } else raw = "b" + boundary + "[(int)(" + address + ")]";
    return type == DataType.BFLOAT16
        ? "Float.intBitsToFloat(Short.toUnsignedInt(" + raw + ")<<16)" : raw;
  }

  private static String store(DataType type, CarrierAccess carrier, int boundary, String address,
      String value) {
    StringBuilder s = new StringBuilder();
    String stored = value;
    if (type == DataType.BFLOAT16) {
      s.append("int bits=Float.floatToRawIntBits(").append(value)
          .append(");short narrowed=(short)(((bits&0x7fffffff)>0x7f800000)?0x7fc0:((bits+0x7fff+((bits>>>16)&1))>>>16));\n");
      stored = "narrowed";
    }
    if (carrier == CarrierAccess.MEMORY_SEGMENT) {
      String layout = switch (type) {
        case BFLOAT16 -> "JAVA_SHORT_UNALIGNED";
        case FLOAT32 -> "JAVA_FLOAT_UNALIGNED";
        case FLOAT64 -> "JAVA_DOUBLE_UNALIGNED";
        default -> throw new AssertionError(type);
      };
      s.append("b").append(boundary).append(".set(java.lang.foreign.ValueLayout.")
          .append(layout).append(",(").append(address).append(")*").append(type.byteWidth())
          .append("L,").append(stored).append(");\n");
    } else s.append("b").append(boundary).append("[(int)(").append(address).append(")]=")
        .append(stored).append(";\n");
    return s.toString();
  }

  private static String javaType(DataType type, CarrierAccess carrier) {
    if (carrier == CarrierAccess.MEMORY_SEGMENT) return "java.lang.foreign.MemorySegment";
    return switch (type) {
      case BFLOAT16 -> "short[]";
      case FLOAT32 -> "float[]";
      case FLOAT64 -> "double[]";
      case BOOL -> "byte[]";
      default -> throw new AssertionError(type);
    };
  }

  private static final class RowState implements AutoCloseable {
    private final Arena arena;
    private final URLClassLoader generatedLoader;
    private final URLClassLoader directLoader;
    private final Method generated;
    private final Method direct;
    private final Object[] generatedArgs;
    private final Object[] directArgs;
    private final List<Integer> outputBoundaries;
    private final List<DataType> types;

    private RowState(Arena arena, URLClassLoader generatedLoader, URLClassLoader directLoader,
        Method generated, Method direct, Object[] generatedArgs, Object[] directArgs,
        List<Integer> outputBoundaries, List<DataType> types) {
      this.arena = arena;
      this.generatedLoader = generatedLoader;
      this.directLoader = directLoader;
      this.generated = generated;
      this.direct = direct;
      this.generatedArgs = generatedArgs;
      this.directArgs = directArgs;
      this.outputBoundaries = outputBoundaries;
      this.types = types;
    }

    static RowState create(Path root, int index, PerformanceRow row, boolean generatedFirst)
        throws Exception {
      List<DataType> types = CpuAttentionEvidenceTest.boundaryTypes(row.mapping(), row.masked(),
          row.outputs());
      Class<?>[] parameterTypes = new Class<?>[types.size() + 5];
      for (int i = 0; i < types.size(); i++) parameterTypes[i] = carrierClass(row.carriers().get(i));
      parameterTypes[types.size()] = MemorySegment.class;
      parameterTypes[types.size() + 1] = long[].class;
      parameterTypes[types.size() + 2] = long.class;
      parameterTypes[types.size() + 3] = long.class;
      parameterTypes[types.size() + 4] = int.class;
      URLClassLoader generatedLoader, directLoader;
      Class<?> generatedDriver, directDriver;
      if (generatedFirst) {
        generatedLoader = loader(root.resolve("generated-driver-classes"));
        directLoader = loader(root.resolve("generated-driver-classes"));
        generatedDriver = driver(generatedLoader, index);
        directDriver = driver(directLoader, index);
      } else {
        directLoader = loader(root.resolve("generated-driver-classes"));
        generatedLoader = loader(root.resolve("generated-driver-classes"));
        directDriver = driver(directLoader, index);
        generatedDriver = driver(generatedLoader, index);
      }
      Method generated = generatedDriver.getDeclaredMethod("invoke", parameterTypes);
      Method direct = directDriver.getDeclaredMethod("invoke", parameterTypes);
      generated.setAccessible(true);
      direct.setAccessible(true);
      Arena arena = Arena.ofConfined();
      Object[] ga = arguments(row, types, arena);
      Object[] da = ga.clone();
      int firstOutput = row.mapping().unique() + (row.masked() ? 1 : 0);
      var outputs = new ArrayList<Integer>();
      for (int i = 0; i < row.outputs(); i++) outputs.add(firstOutput + i);
      return new RowState(arena, generatedLoader, directLoader, generated, direct, ga, da,
          outputs, types);
    }

    RowState withSwappedLabels() {
      return new RowState(arena, directLoader, generatedLoader, direct, generated,
          directArgs, generatedArgs, outputBoundaries, types);
    }

    private static URLClassLoader loader(Path classes) throws Exception {
      return new URLClassLoader(new java.net.URL[] {classes.toUri().toURL()},
          CpuAttentionPerformanceTest.class.getClassLoader());
    }

    private static Class<?> driver(URLClassLoader loader, int index) throws Exception {
      return Class.forName(PACKAGE + ".AttentionDriver_" + index, true, loader);
    }

    void gate(boolean generatedFirst) throws Exception {
      long[] firstChecksums;
      if (generatedFirst) {
        measure(generated, generatedArgs, 1);
        firstChecksums = outputChecksums(generatedArgs);
        measure(direct, directArgs, 1);
      } else {
        measure(direct, directArgs, 1);
        firstChecksums = outputChecksums(directArgs);
        measure(generated, generatedArgs, 1);
      }
      assertTrue(Arrays.equals(firstChecksums, outputChecksums(generatedFirst
          ? directArgs : generatedArgs)), "generated/direct output checksum mismatch");
      verifyOutputs();
    }

    void verifyOutputs() {
      for (int boundary : outputBoundaries) {
        sink ^= checksum(generatedArgs[boundary], types.get(boundary));
      }
    }

    private long[] outputChecksums(Object[] arguments) {
      long[] result = new long[outputBoundaries.size()];
      for (int index = 0; index < outputBoundaries.size(); index++) {
        int boundary = outputBoundaries.get(index);
        result[index] = checksum(arguments[boundary], types.get(boundary));
      }
      return result;
    }

    @Override public void close() throws Exception {
      arena.close();
      generatedLoader.close();
      directLoader.close();
    }
  }

  private static Object[] arguments(PerformanceRow row, List<DataType> types, Arena arena) {
    int n = types.size();
    Object[] args = new Object[n + 5];
    int inputs = row.mapping().unique() + (row.masked() ? 1 : 0);
    for (int i = 0; i < n; i++) {
      long elements = i < row.mapping().unique() ? 2L * 2 * 32 * 64
          : row.masked() && i == row.mapping().unique() ? 1L * 2 * 32 * 32
          : i == inputs ? 2L * 2 * 32 * 64 : 2L * 2 * 32 * 32;
      args[i] = carrier(types.get(i), row.carriers().get(i), elements, arena);
      if (i < inputs) fill(args[i], types.get(i), 0x51f15eL + i * 97L);
    }
    DataType result = row.mapping().result();
    args[n] = arena.allocate(32L * (result == DataType.FLOAT64 ? 8 : 4), 8);
    args[n + 1] = geometry(row).pack(new long[n]);
    args[n + 2] = 0L;
    args[n + 3] = 128L;
    args[n + 4] = 1;
    return args;
  }

  private static CpuAttentionLowering.Geometry geometry(PerformanceRow row) {
    Mapping m = row.mapping();
    DataType result = m.result();
    long[] dense = {4096, 2048, 64, 1};
    long[] weights = {2048, 1024, 32, 1};
    long[] mask = {0, 1024, 32, 1};
    var roles = new ArrayList<>(m.roles());
    if (row.masked()) roles.add(m.unique());
    return new CpuAttentionLowering.Geometry(new long[] {2, 2}, 32, 32, 64, 64, 128,
        32L * (result == DataType.FLOAT64 ? 8 : 4), m.types().get(0), m.types().get(1),
        m.types().get(2), result,
        result == DataType.FLOAT64 ? 0.125d : (float) 0.125d, roles,
        m.unique() + (row.masked() ? 1 : 0), row.outputs(),
        java.util.Optional.of(new CpuAttentionLowering.NormalizedLayout(0, dense)),
        java.util.Optional.of(new CpuAttentionLowering.NormalizedLayout(0, dense)),
        java.util.Optional.of(new CpuAttentionLowering.NormalizedLayout(0, dense)),
        row.masked() ? java.util.Optional.of(new CpuAttentionLowering.NormalizedLayout(0, mask))
            : java.util.Optional.empty(),
        new CpuAttentionLowering.NormalizedLayout(0, dense),
        row.outputs() == 2
            ? java.util.Optional.of(new CpuAttentionLowering.NormalizedLayout(0, weights))
            : java.util.Optional.empty());
  }

  private static Object carrier(DataType type, CarrierAccess access, long elements, Arena arena) {
    if (access == CarrierAccess.MEMORY_SEGMENT)
      return arena.allocate(elements * type.byteWidth(), Math.max(1, type.byteWidth()));
    int count = Math.toIntExact(elements);
    return switch (type) {
      case BFLOAT16 -> new short[count];
      case FLOAT32 -> new float[count];
      case FLOAT64 -> new double[count];
      case BOOL -> new byte[count];
      default -> throw new AssertionError(type);
    };
  }

  private static void fill(Object carrier, DataType type, long seed) {
    long elements = carrier instanceof MemorySegment segment
        ? segment.byteSize() / type.byteWidth() : java.lang.reflect.Array.getLength(carrier);
    for (long i = 0; i < elements; i++) {
      double value = ((i * 17 + seed) % 31 - 15) * 0.03125;
      if (type == DataType.BOOL) value = i % 7 == 0 ? 0 : 1;
      if (carrier instanceof MemorySegment segment) {
        long p = i * type.byteWidth();
        switch (type) {
          case BFLOAT16 -> segment.set(ValueLayout.JAVA_SHORT_UNALIGNED, p, bf((float) value));
          case FLOAT32 -> segment.set(ValueLayout.JAVA_FLOAT_UNALIGNED, p, (float) value);
          case FLOAT64 -> segment.set(ValueLayout.JAVA_DOUBLE_UNALIGNED, p, value);
          case BOOL -> segment.set(ValueLayout.JAVA_BYTE, p, (byte) value);
          default -> throw new AssertionError(type);
        }
      } else {
        int p = Math.toIntExact(i);
        switch (type) {
          case BFLOAT16 -> ((short[]) carrier)[p] = bf((float) value);
          case FLOAT32 -> ((float[]) carrier)[p] = (float) value;
          case FLOAT64 -> ((double[]) carrier)[p] = value;
          case BOOL -> ((byte[]) carrier)[p] = (byte) value;
          default -> throw new AssertionError(type);
        }
      }
    }
  }

  private static short bf(float value) {
    int bits = Float.floatToRawIntBits(value);
    if ((bits & 0x7fffffff) > 0x7f800000) return (short) 0x7fc0;
    return (short) ((bits + 0x7fff + ((bits >>> 16) & 1)) >>> 16);
  }

  private static Class<?> carrierClass(CarrierAccess carrier) {
    return switch (carrier) {
      case SHORT_ARRAY -> short[].class;
      case FLOAT_ARRAY -> float[].class;
      case DOUBLE_ARRAY -> double[].class;
      case BYTE_ARRAY -> byte[].class;
      case MEMORY_SEGMENT -> MemorySegment.class;
      default -> throw new AssertionError(carrier);
    };
  }

  private static long measure(Method method, Object[] base, int batch) throws Exception {
    Object[] arguments = base.clone();
    arguments[arguments.length - 1] = batch;
    return (long) method.invoke(null, arguments);
  }

  private static boolean[] counterbalancedOrders(boolean generatedFirst) {
    return new boolean[] {generatedFirst, !generatedFirst};
  }

  private static long normalizedReplicaMean(List<RowState> states, int[][] batches,
      boolean generated, Random random) throws Exception {
    int[] order = java.util.stream.IntStream.range(0, states.size()).toArray();
    for (int index = order.length - 1; index > 0; index--) {
      int selected = random.nextInt(index + 1);
      int value = order[index];
      order[index] = order[selected];
      order[selected] = value;
    }
    long[] normalized = new long[states.size()];
    for (int replica : order) {
      RowState state = states.get(replica);
      int side = generated ? 0 : 1;
      Method method = generated ? state.generated : state.direct;
      Object[] arguments = generated ? state.generatedArgs : state.directArgs;
      normalized[replica] = measure(method, arguments, batches[replica][side])
          / batches[replica][side];
    }
    return Math.addExact(normalized[0], normalized[1]) / 2;
  }

  private static int[] adaptivePair(Method generated, Object[] generatedArgs, Method direct,
      Object[] directArgs, boolean generatedFirst) throws Exception {
    int generatedBatch = 32, directBatch = 32;
    boolean generatedReady = false, directReady = false;
    while (!generatedReady || !directReady) {
      long generatedPrevious = 0, generatedLast = 0;
      long directPrevious = 0, directLast = 0;
      for (int probe = 0; probe < 5; probe++) {
        if (generatedFirst) {
          if (!generatedReady) {
            generatedPrevious = generatedLast;
            generatedLast = measure(generated, generatedArgs, generatedBatch);
          }
          if (!directReady) {
            directPrevious = directLast;
            directLast = measure(direct, directArgs, directBatch);
          }
        } else {
          if (!directReady) {
            directPrevious = directLast;
            directLast = measure(direct, directArgs, directBatch);
          }
          if (!generatedReady) {
            generatedPrevious = generatedLast;
            generatedLast = measure(generated, generatedArgs, generatedBatch);
          }
        }
        generatedFirst = !generatedFirst;
      }
      if (!generatedReady)
        generatedReady = generatedPrevious >= MIN_NANOS && generatedLast >= MIN_NANOS;
      if (!directReady) directReady = directPrevious >= MIN_NANOS && directLast >= MIN_NANOS;
      if (!generatedReady) generatedBatch = Math.multiplyExact(generatedBatch, 2);
      if (!directReady) directBatch = Math.multiplyExact(directBatch, 2);
    }
    return new int[] {generatedBatch, directBatch};
  }

  private static long median(long[] values) {
    long[] copy = values.clone();
    Arrays.sort(copy);
    return copy[copy.length / 2];
  }

  private static long checksum(Object value, DataType type) {
    long h = 0;
    long count = value instanceof MemorySegment segment
        ? segment.byteSize() / type.byteWidth() : java.lang.reflect.Array.getLength(value);
    for (long i = 0; i < count; i++) {
      long bits;
      if (value instanceof MemorySegment segment) {
        long p = i * type.byteWidth();
        bits = switch (type) {
          case BFLOAT16 -> Short.toUnsignedLong(segment.get(ValueLayout.JAVA_SHORT_UNALIGNED, p));
          case FLOAT32 -> Integer.toUnsignedLong(Float.floatToRawIntBits(
              segment.get(ValueLayout.JAVA_FLOAT_UNALIGNED, p)));
          case FLOAT64 -> Double.doubleToRawLongBits(
              segment.get(ValueLayout.JAVA_DOUBLE_UNALIGNED, p));
          default -> throw new AssertionError(type);
        };
      } else {
        int p = Math.toIntExact(i);
        bits = switch (type) {
          case BFLOAT16 -> Short.toUnsignedLong(((short[]) value)[p]);
          case FLOAT32 -> Integer.toUnsignedLong(Float.floatToRawIntBits(((float[]) value)[p]));
          case FLOAT64 -> Double.doubleToRawLongBits(((double[]) value)[p]);
          default -> throw new AssertionError(type);
        };
      }
      h = Long.rotateLeft(h, 1) ^ bits;
    }
    return h;
  }

  private static void aggregate(Path root) throws Exception {
    List<Path> forks;
    try (var stream = Files.list(root.resolve("forks"))) { forks = stream.sorted().toList(); }
    assertEquals(5, forks.size());
    List<String> names = null;
    double[][] ratios = new double[5][992];
    double worstFork = 0;
    String worstForkRow = "";
    for (int fork = 0; fork < 5; fork++) {
      var found = new ArrayList<String>();
      int row = 0;
      for (String line : Files.readAllLines(forks.get(fork))) if (line.startsWith("RESULT,")) {
        String[] values = line.split(",", 9);
        found.add(values[1]);
        double ratio = Double.parseDouble(values[4]);
        ratios[fork][row++] = ratio;
        if (ratio > worstFork) { worstFork = ratio; worstForkRow = "fork=" + fork + "," + values[1]; }
      }
      assertEquals(992, row);
      if (names == null) names = List.copyOf(found); else assertEquals(names, found);
    }
    StringBuilder aggregate = new StringBuilder("row,median_ratio,fork_ratios\n");
    double worstAggregate = 0;
    String worstAggregateRow = "";
    for (int row = 0; row < 992; row++) {
      double[] values = new double[5];
      for (int fork = 0; fork < 5; fork++) values[fork] = ratios[fork][row];
      Arrays.sort(values);
      assertTrue(values[2] <= 1.15, names.get(row) + " aggregate " + values[2]);
      if (values[2] > worstAggregate) { worstAggregate = values[2]; worstAggregateRow = names.get(row); }
      aggregate.append(names.get(row)).append(',')
          .append(String.format(Locale.ROOT, "%.9f", values[2])).append(',')
          .append(quote(Arrays.toString(values))).append('\n');
    }
    Files.writeString(root.resolve("aggregates.csv"), aggregate);
    Files.writeString(root.resolve("summary.txt"), "forks=5\nrows=992\nrow_fork_results=4960\n"
        + "aggregates=992\nworst_row_fork=" + worstForkRow + ",ratio="
        + String.format(Locale.ROOT, "%.9f", worstFork) + "\nworst_aggregate="
        + worstAggregateRow + ",ratio=" + String.format(Locale.ROOT, "%.9f", worstAggregate)
        + "\n");
  }

  private static void retainIdentity(Path root) throws Exception {
    Path source = Path.of("src/test/java", PACKAGE.replace('.', '/'),
        "CpuAttentionPerformanceTest.java");
    Files.copy(source, root.resolve("performance-harness-source.java"));
    String javap = Path.of(System.getProperty("java.home"), "bin", "javap").toString();
    Path output = root.resolve("decompilation.txt");
    Files.deleteIfExists(output);
    List<Path> drivers;
    try (var stream = Files.walk(root.resolve("generated-driver-classes"))) {
      drivers = stream.filter(path -> path.getFileName().toString().startsWith("AttentionDriver_")
          && path.toString().endsWith(".class")).sorted().toList();
    }
    for (int first = 0; first < drivers.size(); first += 50) {
      var command = new ArrayList<String>();
      command.add(javap); command.add("-c"); command.add("-v"); command.add("-p");
      drivers.subList(first, Math.min(drivers.size(), first + 50)).forEach(path -> command.add(path.toString()));
      Process process = new ProcessBuilder(command).redirectErrorStream(true).start();
      byte[] bytes = process.getInputStream().readAllBytes();
      assertEquals(0, process.waitFor());
      Files.write(output, bytes, StandardOpenOption.CREATE, StandardOpenOption.APPEND);
    }
    String text = Files.readString(output);
    assertFalse(text.contains("invokeWithArguments"));
    assertFalse(text.contains("CpuAttentionReferenceKernel"));
    assertFalse(text.contains("java/lang/reflect"));
    Files.writeString(root.resolve("environment.txt"), System.getProperties().toString());
  }

  private static void retainFork(Path root, boolean accepted, int fork, String report)
      throws IOException {
    Path directory = root.resolve(accepted ? "forks" : "failed-forks");
    Files.createDirectories(directory);
    Files.writeString(directory.resolve("fork-" + fork + "-" + Instant.now().toEpochMilli()
        + ".csv"), report);
  }

  private static void environment(StringBuilder report, int fork) {
    long total = Runtime.getRuntime().totalMemory(), max = Runtime.getRuntime().maxMemory();
    String java = System.getProperty("java.version");
    report.append("ENV,fork=").append(fork).append(",java=").append(java).append(",vm=")
        .append(System.getProperty("java.vm.name")).append(",os=")
        .append(System.getProperty("os.name")).append(",arch=")
        .append(System.getProperty("os.arch")).append(",processors=")
        .append(Runtime.getRuntime().availableProcessors()).append(",totalMemory=").append(total)
        .append(",maxMemory=").append(max).append(",byteOrder=").append(ByteOrder.nativeOrder())
        .append('\n');
    long low = 900L << 20, high = 1100L << 20;
    if (!java.startsWith("26") || total < low || max < low || max > high)
      throw new AssertionError("requires Java 26 -Xms1g -Xmx1g");
  }

  private static Path root() {
    String value = System.getenv("SYNAPTIK_CPU_ATTENTION_EVIDENCE_ROOT");
    if (value == null || value.isBlank())
      throw new IllegalStateException("SYNAPTIK_CPU_ATTENTION_EVIDENCE_ROOT is required");
    return Path.of(value).toAbsolutePath();
  }

  private static void copyTree(Path source, Path target) throws IOException {
    try (var stream = Files.walk(source)) {
      for (Path path : stream.toList()) if (Files.isRegularFile(path)) {
        Path destination = target.resolve(source.relativize(path));
        Files.createDirectories(destination.getParent());
        Files.copy(path, destination);
      }
    }
  }

  private static void writeManifest(Path root) throws Exception {
    MessageDigest digest = MessageDigest.getInstance("SHA-256");
    StringBuilder manifest = new StringBuilder();
    try (var stream = Files.walk(root)) {
      for (Path file : stream.filter(Files::isRegularFile).sorted().toList()) {
        if (file.getFileName().toString().startsWith("manifest")) continue;
        manifest.append(HexFormat.of().formatHex(digest.digest(Files.readAllBytes(file))))
            .append("  ").append(root.relativize(file)).append('\n');
      }
    }
    Files.writeString(root.resolve("manifest.sha256"), manifest);
    Files.writeString(root.resolve("manifest.digest"), HexFormat.of().formatHex(digest.digest(
        manifest.toString().getBytes(StandardCharsets.UTF_8))));
  }

  private static String quote(String value) { return '"' + value.replace("\"", "\"\"") + '"'; }
  private static String safe(String value) {
    return value == null ? "" : value.replace(',', ';').replace('\n', ' ').replace('\r', ' ');
  }
}
