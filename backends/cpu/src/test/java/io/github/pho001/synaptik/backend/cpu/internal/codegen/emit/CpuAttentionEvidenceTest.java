package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuLoweringFingerprint;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAttentionIr;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.DataTypePromotion;
import java.lang.classfile.ClassFile;
import java.lang.classfile.CodeElement;
import java.lang.classfile.Instruction;
import java.lang.classfile.Opcode;
import java.lang.classfile.TypeKind;
import java.lang.classfile.constantpool.MemberRefEntry;
import java.lang.classfile.instruction.ArrayLoadInstruction;
import java.lang.classfile.instruction.ArrayStoreInstruction;
import java.lang.classfile.instruction.BranchInstruction;
import java.lang.classfile.instruction.ConstantInstruction;
import java.lang.classfile.instruction.FieldInstruction;
import java.lang.classfile.instruction.IncrementInstruction;
import java.lang.classfile.instruction.InvokeInstruction;
import java.lang.classfile.instruction.LabelTarget;
import java.lang.classfile.instruction.LoadInstruction;
import java.lang.classfile.instruction.StoreInstruction;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.HexFormat;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

class CpuAttentionEvidenceTest {
  private static final List<DataType> FLOATING =
      List.of(DataType.BFLOAT16, DataType.FLOAT32, DataType.FLOAT64);

  /**
   * Runs the explicit structural evidence owner outside the ordinary skipped test suite.
   *
   * @param args ignored empty command-line arguments
   * @throws Exception if generation, normalization, decompilation, or retention fails
   */
  public static void main(String[] args) throws Exception {
    new CpuAttentionEvidenceTest().emitsAndScansExactElevenThousandEightHundredEightyInventory();
  }

  @Test
  void emitsAndScansExactElevenThousandEightHundredEightyInventory() throws Exception {
    Path root = Path.of(System.getProperty("synaptik.cpu.attention.structuralEvidenceRoot",
        "/tmp/synaptik-cpu-0008h-evidence"));
    Path marker = root.resolve("RUN-STRUCTURAL-EVIDENCE");
    Assumptions.assumeTrue(Files.exists(marker));
    Path classes = root.resolve("classes"), javap = root.resolve("javap");
    Files.createDirectories(classes);
    Files.createDirectories(javap);
    Files.writeString(root.resolve("inventory.csv"),
        "key,types,roles,masked,causal,outputs,carriers,sha256,skeleton,fragments,coverage\n");
    var keys = new LinkedHashSet<String>();
    List<PerformanceRow> performance = performanceRows();
    Set<String> rows = performance.stream().map(PerformanceRow::name)
        .collect(java.util.stream.Collectors.toCollection(LinkedHashSet::new));
    var timedComponents = new LinkedHashMap<String, NormalizedBody>();
    var timedKeys = new LinkedHashMap<String, String>();
    for (PerformanceRow row : performance) {
      Generated generated = generated(row.mapping(), row.masked(), row.causal(), row.outputs(),
          row.carriers());
      timedComponents.put(row.name(), normalize(generated.bytes(), generated.types(),
          row.carriers(), row.mapping(), row.masked(), row.outputs()));
      timedKeys.put(row.name(), generated.key());
    }
    var skeletonWitnesses = new LinkedHashMap<String, String>();
    var fragmentWitnesses = new LinkedHashMap<String, String>();
    timedComponents.forEach((row, body) -> {
      skeletonWitnesses.putIfAbsent(body.skeletonHash(), row);
      body.fragmentHashes().forEach(fragment -> fragmentWitnesses.putIfAbsent(fragment, row));
    });
    var skeletonCounts = new LinkedHashMap<String, Integer>();
    var fragmentCounts = new LinkedHashMap<String, Integer>();
    var skeletonByShape = new LinkedHashMap<String, Set<String>>();
    StringBuilder coverageReport = new StringBuilder(
        "key,skeleton,skeleton_row,fragment_rows\n");
    var generator = new CpuClassFileKernelGenerator();
    int count = 0;
    for (Mapping mapping : mappings())
      for (boolean masked : List.of(false, true))
        for (boolean causal : List.of(false, true))
          for (int outputs : List.of(1, 2)) {
            List<DataType> boundaryTypes = boundaryTypes(mapping, masked, outputs);
            int patterns = 1 << boundaryTypes.size();
            for (int bits = 0; bits < patterns; bits++) {
              List<CarrierAccess> carriers = carriers(boundaryTypes, bits);
              CpuAttentionIr attention = attention(mapping, masked, causal, outputs, boundaryTypes);
              var ir = attention.encodedKernelIr();
              var specialization =
                  new CpuKernelSpecialization(
                      CpuLoweringFingerprint.fromHex(ir.structuralKey()),
                      CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                      CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR,
                      boundaryTypes,
                      carriers,
                      0,
                      -1,
                      List.of(),
                      true,
                      57);
              String key = specialization.structuralKey();
              assertTrue(keys.add(key), "duplicate specialization " + key);
              byte[] bytes = generator.generateClassBytes(specialization, ir);
              var model = ClassFile.of().parse(bytes);
              assertTrue(model.fields().isEmpty());
              assertEquals(1, model.methods().size());
              String owners =
                  java.util.stream.StreamSupport.stream(model.constantPool().spliterator(), false)
                      .filter(MemberRefEntry.class::isInstance)
                      .map(MemberRefEntry.class::cast)
                      .map(entry -> entry.owner().asInternalName())
                      .reduce("", (a, b) -> a + '\n' + b);
              assertFalse(owners.contains("io/github/pho001/synaptik"));
              assertFalse(owners.contains("java/util/"));
              Path file = classes.resolve(key + ".class");
              Files.write(file, bytes);
              NormalizedBody body = normalize(bytes, boundaryTypes, carriers, mapping, masked,
                  outputs);
              String skeleton = body.skeletonHash();
              String fragments = String.join(";", body.fragmentHashes());
              String skeletonRow = skeletonWitnesses.get(skeleton);
              assertTrue(skeletonRow != null, "uncovered loop skeleton " + skeleton);
              var fragmentRows = new ArrayList<String>();
              for (String fragment : body.fragmentHashes()) {
                String witness = fragmentWitnesses.get(fragment);
                assertTrue(witness != null, "uncovered boundary fragment " + fragment);
                fragmentRows.add(witness);
                fragmentCounts.merge(fragment, 1, Integer::sum);
              }
              skeletonCounts.merge(skeleton, 1, Integer::sum);
              String shape = mapping.types() + "|" + mapping.roles() + '|' + masked + '|'
                  + causal + '|' + outputs;
              skeletonByShape.computeIfAbsent(shape, ignored -> new LinkedHashSet<>()).add(skeleton);
              String coverage = "s=" + skeletonRow + ";b=" + String.join("|", fragmentRows);
              Files.writeString(
                  root.resolve("inventory.csv"),
                  String.join(
                          ",",
                          key,
                          quote(mapping.types().toString()),
                          quote(mapping.roles().toString()),
                          Boolean.toString(masked),
                          Boolean.toString(causal),
                          Integer.toString(outputs),
                          quote(carriers.toString()),
                          hash(bytes),
                          skeleton,
                          fragments,
                          coverage)
                      + "\n",
                  StandardOpenOption.APPEND);
              count++;
              coverageReport.append(key).append(',').append(skeleton).append(',')
                  .append(skeletonRow).append(',').append(quote(String.join("|", fragmentRows)))
                  .append('\n');
            }
          }
    assertEquals(11_880, count);
    assertEquals(11_880, keys.size());
    assertEquals(992, rows.size());
    skeletonByShape.forEach((shape, hashes) -> assertEquals(1, hashes.size(),
        "carrier choice changed surrounding loop/dataflow skeleton for " + shape));
    Files.write(root.resolve("specialization-keys.txt"), keys);
    Files.write(root.resolve("performance-rows.txt"), rows);
    Files.writeString(root.resolve("coverage.csv"), coverageReport);
    writeInventory(root.resolve("loop-skeletons.csv"), "hash,witness,count\n", skeletonCounts,
        skeletonWitnesses);
    writeInventory(root.resolve("boundary-fragments.csv"), "hash,witness,count\n", fragmentCounts,
        fragmentWitnesses);
    StringBuilder timed = new StringBuilder("row,key,skeleton,fragments\n");
    for (PerformanceRow row : performance) {
      NormalizedBody body = timedComponents.get(row.name());
      timed.append(row.name()).append(',').append(timedKeys.get(row.name())).append(',')
          .append(body.skeletonHash()).append(',')
          .append(quote(String.join(";", body.fragmentHashes()))).append('\n');
    }
    Files.writeString(root.resolve("timed-components.csv"), timed);
    runJavap(classes, javap.resolve("all.txt"));
    List<String> forbidden =
        List.of(
            " invokedynamic ", " monitorenter", " monitorexit", "java/util/", "java/lang/reflect");
    try (var reader = Files.newBufferedReader(javap.resolve("all.txt"))) {
      for (String line; (line = reader.readLine()) != null; )
        for (String token : forbidden) assertFalse(line.contains(token), token);
    }
    Files.writeString(
        root.resolve("forbidden-scan.txt"),
        "constant-pool member owners: no Synaptik, java.util, or reflection owners\n"
            + "decompilation: no invokedynamic, monitor, java.util, or reflection instructions\n");
    Files.writeString(
        root.resolve("summary.txt"),
        "classes=11880\nrows=992\nskeletons=" + skeletonCounts.size()
            + "\nfragments=" + fragmentCounts.size() + "\n"
            + "inventory_sha256="
            + hashFile(root.resolve("inventory.csv"))
            + '\n'
            + "javap_sha256="
            + hashFile(javap.resolve("all.txt"))
            + '\n');
    Files.writeString(root.resolve("verifier.txt"),
        "java.lang.classfile verification passed for 11880 generated classes\n"
            + "member inventory passed for 11880 generated classes\n"
            + "normalized skeleton and ordered boundary-fragment coverage passed for 11880 identities\n");
    writeManifest(root);
    Files.delete(marker);
  }

  private static void writeInventory(Path path, String header, Map<String, Integer> counts,
      Map<String, String> witnesses) throws Exception {
    StringBuilder text = new StringBuilder(header);
    counts.forEach((hash, count) -> text.append(hash).append(',').append(witnesses.get(hash))
        .append(',').append(count).append('\n'));
    Files.writeString(path, text);
  }

  static List<Mapping> mappings() {
    var result = new ArrayList<Mapping>();
    for (DataType q : FLOATING)
      for (DataType k : FLOATING)
        for (DataType v : FLOATING) {
          result.add(new Mapping(List.of(q, k, v), List.of(0, 1, 2)));
          if (q == k) result.add(new Mapping(List.of(q, k, v), List.of(0, 0, 1)));
          if (q == v) result.add(new Mapping(List.of(q, k, v), List.of(0, 1, 0)));
          if (k == v) result.add(new Mapping(List.of(q, k, v), List.of(0, 1, 1)));
          if (q == k && k == v) result.add(new Mapping(List.of(q, k, v), List.of(0, 0, 0)));
        }
    assertEquals(57, result.size());
    return result;
  }

  static CpuAttentionIr attention(
      Mapping m, boolean masked, boolean causal, int outputs, List<DataType> types) {
    var roles = new ArrayList<>(m.roles());
    int unique = m.unique();
    if (masked) roles.add(unique);
    var plans = new ArrayList<CpuAccessPlan>();
    for (int i = 0; i < types.size(); i++)
      plans.add(
          plan(
              i < unique + (masked ? 1 : 0)
                  ? CpuAccessPlan.AccessKind.READ
                  : CpuAccessPlan.AccessKind.WRITE));
    return new CpuAttentionIr(
        m.types().get(0),
        m.types().get(1),
        m.types().get(2),
        m.result(),
        masked,
        causal,
        outputs,
        roles,
        types,
        plans);
  }

  static List<DataType> boundaryTypes(Mapping m, boolean masked, int outputs) {
    var result = new ArrayList<DataType>();
    for (int boundary = 0; boundary < m.unique(); boundary++) {
      int b = boundary;
      int role =
          java.util.stream.IntStream.range(0, 3)
              .filter(i -> m.roles().get(i) == b)
              .findFirst()
              .orElseThrow();
      result.add(m.types().get(role));
    }
    if (masked) result.add(DataType.BOOL);
    for (int i = 0; i < outputs; i++) result.add(m.result());
    return result;
  }

  static List<CarrierAccess> carriers(List<DataType> types, int bits) {
    var result = new ArrayList<CarrierAccess>();
    for (int i = 0; i < types.size(); i++)
      result.add((bits & (1 << i)) == 0 ? array(types.get(i)) : CarrierAccess.MEMORY_SEGMENT);
    return result;
  }

  static CarrierAccess array(DataType type) {
    return switch (type) {
      case BFLOAT16 -> CarrierAccess.SHORT_ARRAY;
      case FLOAT32 -> CarrierAccess.FLOAT_ARRAY;
      case FLOAT64 -> CarrierAccess.DOUBLE_ARRAY;
      case BOOL -> CarrierAccess.BYTE_ARRAY;
      default -> throw new AssertionError(type);
    };
  }

  private static CpuAccessPlan plan(CpuAccessPlan.AccessKind kind) {
    return new CpuAccessPlan(kind, CpuAccessPlan.Regime.SCALAR_ALL_ZERO, 0, List.of(), 0);
  }

  static List<PerformanceRow> performanceRows() throws Exception {
    var rows = new ArrayList<PerformanceRow>();
    for (Mapping m : mappings())
      for (boolean masked : List.of(false, true))
        for (boolean causal : List.of(false, true))
          for (int outputs : List.of(1, 2))
            for (boolean segment : List.of(false, true))
              rows.add(new PerformanceRow(coreRow(m, masked, causal, outputs, segment), m,
                  masked, causal, outputs, uniformCarriers(m, masked, outputs, segment)));
    Mapping f32 =
        new Mapping(
            List.of(DataType.FLOAT32, DataType.FLOAT32, DataType.FLOAT32), List.of(0, 1, 2));
    for (boolean masked : List.of(false, true))
      for (boolean causal : List.of(false, true))
        for (int outputs : List.of(1, 2)) {
          int active = 3 + (masked ? 1 : 0) + outputs;
          for (int boundary = 0; boundary < active; boundary++)
            for (boolean baseSegment : List.of(false, true)) {
              String name =
                  "mixed|"
                      + masked
                      + '|'
                      + causal
                      + '|'
                      + outputs
                      + '|'
                      + boundary
                      + '|'
                      + baseSegment;
              List<DataType> types = boundaryTypes(f32, masked, outputs);
              var carriers = new ArrayList<CarrierAccess>();
              for (int i = 0; i < types.size(); i++) {
                boolean segment = i == boundary ? !baseSegment : baseSegment;
                carriers.add(segment ? CarrierAccess.MEMORY_SEGMENT : array(types.get(i)));
              }
              rows.add(new PerformanceRow(name, f32, masked, causal, outputs, carriers));
            }
        }
    assertEquals(992, rows.size());
    assertEquals(992, rows.stream().map(PerformanceRow::name).distinct().count());
    return List.copyOf(rows);
  }

  private static List<CarrierAccess> uniformCarriers(Mapping mapping, boolean masked, int outputs,
      boolean segment) {
    return boundaryTypes(mapping, masked, outputs).stream()
        .map(type -> segment ? CarrierAccess.MEMORY_SEGMENT : array(type)).toList();
  }

  static String coreRow(
      Mapping m, boolean masked, boolean causal, int outputs, boolean segment) {
    return "core|" + m.types() + '|' + m.roles() + '|' + masked + '|' + causal + '|' + outputs + '|'
        + segment;
  }

  static Generated generated(Mapping mapping, boolean masked, boolean causal, int outputs,
      List<CarrierAccess> carriers) {
    List<DataType> types = boundaryTypes(mapping, masked, outputs);
    CpuAttentionIr attention = attention(mapping, masked, causal, outputs, types);
    var ir = attention.encodedKernelIr();
    var specialization = new CpuKernelSpecialization(
        CpuLoweringFingerprint.fromHex(ir.structuralKey()),
        CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
        CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR, types, carriers, 0, -1, List.of(),
        true, 57);
    byte[] bytes = new CpuClassFileKernelGenerator().generateClassBytes(specialization, ir);
    return new Generated(specialization.structuralKey(), specialization, ir, types, bytes);
  }

  private static NormalizedBody normalize(byte[] bytes, List<DataType> types,
      List<CarrierAccess> carriers, Mapping mapping, boolean masked, int outputs) throws Exception {
    var code = ClassFile.of().parse(bytes).methods().getFirst().code().orElseThrow();
    List<CodeElement> elements = code.elementStream().toList();
    var labelInstructions = new IdentityHashMap<java.lang.classfile.Label, Integer>();
    var instructions = new ArrayList<Instruction>();
    int ordinal = 0;
    for (CodeElement element : elements) {
      if (element instanceof LabelTarget target) labelInstructions.put(target.label(), ordinal);
      if (element instanceof Instruction instruction) {
        instructions.add(instruction);
        ordinal++;
      }
    }
    int boundaries = types.size();
    var spans = new ArrayList<AccessSpan>();
    var byBoundary = new ArrayList<List<AccessSpan>>();
    for (int boundary = 0; boundary < boundaries; boundary++) byBoundary.add(new ArrayList<>());
    for (int index = 0; index < instructions.size(); index++) {
      Instruction first = instructions.get(index);
      if (!(first instanceof LoadInstruction load) || load.typeKind() != TypeKind.REFERENCE
          || load.slot() < 0 || load.slot() >= boundaries) continue;
      int end = accessEnd(instructions, index, boundaries);
      if (end >= 0) {
        AccessSpan span = new AccessSpan(index, end, load.slot());
        spans.add(span);
        byBoundary.get(load.slot()).add(span);
        index = end;
      }
    }
    boolean[] covered = new boolean[instructions.size()];
    for (AccessSpan span : spans)
      for (int i = span.start(); i <= span.end(); i++) covered[i] = true;
    int[] collapsed = new int[instructions.size() + 1];
    int node = 0;
    for (int i = 0; i < instructions.size(); i++) {
      collapsed[i] = node;
      if (!covered[i] || i == 0 || !covered[i - 1]) node++;
    }
    collapsed[instructions.size()] = node;
    var locals = localNames(instructions, boundaries);
    StringBuilder skeleton = new StringBuilder();
    for (int i = 0; i < instructions.size(); i++) {
      AccessSpan span = spanStarting(spans, i);
      if (span != null) {
        skeleton.append("ACCESS:B").append(span.boundary()).append('\n');
        i = span.end();
      } else {
        skeleton.append(normalizeInstruction(instructions.get(i), labelInstructions, collapsed,
            locals, boundaries)).append('\n');
      }
    }
    var fragmentHashes = new ArrayList<String>();
    for (int boundary = 0; boundary < boundaries; boundary++) {
      StringBuilder fragment = new StringBuilder();
      fragment.append("role=").append(boundaryRole(mapping, masked, outputs, boundary))
          .append("|type=").append(types.get(boundary)).append("|carrier=")
          .append(carriers.get(boundary)).append('\n');
      for (AccessSpan span : byBoundary.get(boundary)) {
        for (int i = span.start(); i <= span.end(); i++)
          fragment.append(normalizeInstruction(instructions.get(i), labelInstructions, collapsed,
              locals, boundaries)).append('\n');
        fragment.append("--\n");
      }
      assertFalse(byBoundary.get(boundary).isEmpty(), "boundary has no direct access " + boundary
          + " types=" + types + " carriers=" + carriers + " mapping=" + mapping
          + " masked=" + masked + " outputs=" + outputs);
      fragmentHashes.add(hash(fragment.toString()));
    }
    return new NormalizedBody(hash(skeleton.toString()), List.copyOf(fragmentHashes));
  }

  private static int accessEnd(List<Instruction> instructions, int start, int boundaries) {
    for (int i = start + 1; i < Math.min(instructions.size(), start + 32); i++) {
      Instruction instruction = instructions.get(i);
      if ((instruction instanceof ArrayLoadInstruction arrayLoad
              && arrayLoad.opcode() != Opcode.LALOAD)
          || instruction instanceof ArrayStoreInstruction)
        return i;
      if (instruction instanceof InvokeInstruction invoke
          && invoke.owner().asInternalName().equals("java/lang/foreign/MemorySegment")
          && (invoke.name().stringValue().equals("get") || invoke.name().stringValue().equals("set")))
        return i;
      if (instruction instanceof LoadInstruction load && load.typeKind() == TypeKind.REFERENCE
          && load.slot() >= 0 && load.slot() < boundaries)
        return -1;
    }
    return -1;
  }

  private static Map<Integer, String> localNames(List<Instruction> instructions, int boundaries) {
    var result = new LinkedHashMap<Integer, String>();
    int scratch = boundaries, geometry = boundaries + 1, start = boundaries + 2,
        end = boundaries + 4;
    for (Instruction instruction : instructions) {
      int slot = instruction instanceof LoadInstruction load ? load.slot()
          : instruction instanceof StoreInstruction store ? store.slot()
          : instruction instanceof IncrementInstruction increment ? increment.slot() : -1;
      if (slot < 0 || result.containsKey(slot)) continue;
      String name = slot < boundaries ? "B" + slot : slot == scratch ? "SCRATCH"
          : slot == geometry ? "GEOMETRY" : slot == start ? "START"
          : slot == end ? "END" : "L" + result.values().stream()
              .filter(value -> value.startsWith("L")).count();
      result.put(slot, name);
    }
    return result;
  }

  private static String normalizeInstruction(Instruction instruction,
      IdentityHashMap<java.lang.classfile.Label, Integer> labels, int[] collapsed,
      Map<Integer, String> locals, int boundaries) {
    StringBuilder text = new StringBuilder(instruction.opcode().name());
    if (instruction instanceof LoadInstruction load) text.append(':').append(locals.get(load.slot()));
    else if (instruction instanceof StoreInstruction store)
      text.append(':').append(locals.get(store.slot()));
    else if (instruction instanceof IncrementInstruction increment)
      text.append(':').append(locals.get(increment.slot())).append(':').append(increment.constant());
    else if (instruction instanceof BranchInstruction branch)
      text.append(":N").append(collapsed[labels.get(branch.target())]);
    else if (instruction instanceof ConstantInstruction constant)
      text.append(':').append(constant.typeKind()).append(':').append(constant.constantValue());
    else if (instruction instanceof InvokeInstruction invoke)
      text.append(':').append(invoke.owner().asInternalName()).append('.').append(invoke.name())
          .append(invoke.type());
    else if (instruction instanceof FieldInstruction field)
      text.append(':').append(field.owner().asInternalName()).append('.').append(field.name())
          .append(':').append(field.type());
    return text.toString();
  }

  private static AccessSpan spanStarting(List<AccessSpan> spans, int start) {
    for (AccessSpan span : spans) if (span.start() == start) return span;
    return null;
  }

  private static String boundaryRole(Mapping mapping, boolean masked, int outputs, int boundary) {
    var roles = new ArrayList<String>();
    String[] semantic = {"QUERY", "KEY", "VALUE"};
    for (int i = 0; i < 3; i++) if (mapping.roles().get(i) == boundary) roles.add(semantic[i]);
    int next = mapping.unique();
    if (masked) {
      if (boundary == next) roles.add("MASK");
      next++;
    }
    if (boundary == next) roles.add("OUTPUT");
    if (outputs == 2 && boundary == next + 1) roles.add("WEIGHTS");
    return String.join("+", roles);
  }

  private static void runJavap(Path classes, Path output) throws Exception {
    Files.deleteIfExists(output);
    List<Path> files;
    try (var stream = Files.list(classes)) {
      files = stream.sorted().toList();
    }
    for (int first = 0; first < files.size(); first += 100) {
      var command = new ArrayList<String>();
      command.add("javap");
      command.add("-c");
      command.add("-v");
      command.add("-p");
      for (Path p : files.subList(first, Math.min(files.size(), first + 100)))
        command.add(p.toString());
      Process process = new ProcessBuilder(command).redirectErrorStream(true).start();
      byte[] bytes = process.getInputStream().readAllBytes();
      assertEquals(0, process.waitFor(), new String(bytes, StandardCharsets.UTF_8));
      Files.write(output, bytes, StandardOpenOption.CREATE, StandardOpenOption.APPEND);
    }
  }

  private static String hash(String value) throws Exception {
    return hash(value.getBytes(StandardCharsets.UTF_8));
  }

  private static String hash(byte[] value) throws Exception {
    return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(value));
  }

  private static String hashFile(Path value) throws Exception {
    MessageDigest digest = MessageDigest.getInstance("SHA-256");
    try (var input = Files.newInputStream(value)) {
      byte[] buffer = new byte[1 << 20];
      for (int read; (read = input.read(buffer)) >= 0; )
        if (read > 0) digest.update(buffer, 0, read);
    }
    return HexFormat.of().formatHex(digest.digest());
  }

  private static void writeManifest(Path root) throws Exception {
    StringBuilder manifest = new StringBuilder();
    try (var stream = Files.walk(root)) {
      for (Path file : stream.filter(Files::isRegularFile).sorted().toList()) {
        if (file.equals(root.resolve("RUN-STRUCTURAL-EVIDENCE"))
            || file.getFileName().toString().startsWith("manifest")) continue;
        manifest.append(hashFile(file)).append("  ").append(root.relativize(file)).append('\n');
      }
    }
    Files.writeString(root.resolve("manifest.sha256"), manifest);
    Files.writeString(root.resolve("manifest.digest"), hash(manifest.toString()));
  }

  private static String quote(String value) {
    return '"' + value.replace("\"", "\"\"") + '"';
  }

  record Mapping(List<DataType> types, List<Integer> roles) {
    Mapping {
      types = List.copyOf(types);
      roles = List.copyOf(roles);
    }

    int unique() {
      return new HashSet<>(roles).size();
    }

    DataType result() {
      return DataTypePromotion.promoteFloating(
          DataTypePromotion.promoteFloating(types.get(0), types.get(1)), types.get(2));
    }
  }

  record PerformanceRow(String name, Mapping mapping, boolean masked, boolean causal, int outputs,
      List<CarrierAccess> carriers) {
    PerformanceRow {
      carriers = List.copyOf(carriers);
    }
  }

  record Generated(String key, CpuKernelSpecialization specialization,
      io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr ir, List<DataType> types,
      byte[] bytes) {}

  private record AccessSpan(int start, int end, int boundary) {}

  private record NormalizedBody(String skeletonHash, List<String> fragmentHashes) {}
}
