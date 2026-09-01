package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.classfile.CodeBuilder;
import java.lang.classfile.Label;
import java.lang.classfile.Opcode;
import java.lang.classfile.TypeKind;
import java.lang.constant.ClassDesc;
import java.lang.constant.MethodTypeDesc;

/**
 * Emits the CPU-private schema-57 direct scalar scaled-dot-product-attention row algorithm.
 *
 * <p>The emitted body consumes only cold-bound primitive geometry, direct carriers, and assigned
 * per-range scratch. It is not a semantic operation implementation or a fallback route.
 */
public final class CpuAttentionEmitter {
  private static final ClassDesc STRICT = MathDesc(StrictMath.class);
  private static final ClassDesc SHORT = MathDesc(Short.class);
  private static final ClassDesc FLOAT = MathDesc(Float.class);
  private static final ClassDesc DOUBLE = MathDesc(Double.class);
  private static final ClassDesc SEGMENT = ClassDesc.of("java.lang.foreign.MemorySegment");
  private static final ClassDesc VALUE_LAYOUT = ClassDesc.of("java.lang.foreign.ValueLayout");
  private static final ClassDesc F_LAYOUT = ClassDesc.of("java.lang.foreign.ValueLayout$OfFloat");
  private static final ClassDesc D_LAYOUT = ClassDesc.of("java.lang.foreign.ValueLayout$OfDouble");

  /** Creates a stateless CPU-private attention emitter. */
  public CpuAttentionEmitter() {}

  /**
   * Emits complete score, classification, normalization, output, and optional weights loops.
   *
   * @param c non-null Class-File code builder receiving the generated entry body
   * @param s non-null schema-57 scalar specialization with an attention scratch parameter
   * @param ir non-null attention identity containing only code-shaping facts
   * @throws IllegalArgumentException if the specialization and IR do not describe the supported
   *     schema-57 direct scalar attention family
   */
  public void emit(CodeBuilder c, CpuKernelSpecialization s, CpuKernelIr ir) {
    if (!ir.familyIdentity().startsWith("attention:")
        || s.classIdentitySchema() != 57
        || !s.scratchParameter()
        || s.executionStrategy().compute()
            != io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan
                .ExecutionStrategy.Compute.SCALAR)
      throw new IllegalArgumentException("attention generated facts disagree");
    boolean masked = ir.familyIdentity().contains(":masked=true:");
    boolean causal = ir.familyIdentity().contains(":causal=true:");
    boolean two = ir.familyIdentity().contains(":outputs=2:");
    int[] roles = roles(ir.familyIdentity());
    DataType qt = type(ir, "q="),
        kt = type(ir, "k="),
        vt = type(ir, "v="),
        rt = type(ir, "result=");
    var carrier = new CpuCarrierEmitter(c);
    int n = s.carrierPattern().size(), scratch = n, g = n + 1;
    int start = n + 2, end = n + 4;
    int rank = i(c), header = i(c), row = l(c), qi = l(c), batch = l(c);
    int qb = l(c), kb = l(c), vb = l(c);
    int mb = masked ? l(c) : qb, ob = l(c), wb = two ? l(c) : qb, axis = i(c);
    int coord = l(c);
    get(c, g, 6).l2i().istore(rank);
    c.loadConstant(12).iload(rank).iadd().istore(header);
    c.lload(start).lstore(row);
    Label rows = c.newLabel(), done = c.newLabel();
    c.labelBinding(rows).lload(row).lload(end).lcmp().branch(Opcode.IFGE, done);
    c.lload(row);
    get(c, g, 7).lrem().lstore(qi);
    c.lload(row);
    get(c, g, 7).ldiv().lstore(batch);
    base(c, g, header, rank, 0, qb);
    base(c, g, header, rank, 1, kb);
    base(c, g, header, rank, 2, vb);
    if (masked) base(c, g, header, rank, 3, mb);
    base(c, g, header, rank, 4, ob);
    if (two) base(c, g, header, rank, 5, wb);
    c.iload(rank).loadConstant(3).isub().istore(axis);
    Label axes = c.newLabel(), axesDone = c.newLabel();
    c.labelBinding(axes).iload(axis).branch(Opcode.IFLT, axesDone);
    c.lload(batch);
    c.aload(g).loadConstant(14).iload(axis).iadd().laload().lrem().lstore(coord);
    c.lload(batch).aload(g).loadConstant(14).iload(axis).iadd().laload().ldiv().lstore(batch);
    addStride(c, g, header, rank, 0, axis, coord, qb);
    addStride(c, g, header, rank, 1, axis, coord, kb);
    addStride(c, g, header, rank, 2, axis, coord, vb);
    if (masked) addStride(c, g, header, rank, 3, axis, coord, mb);
    addStride(c, g, header, rank, 4, axis, coord, ob);
    if (two) addStride(c, g, header, rank, 5, axis, coord, wb);
    c.iinc(axis, -1).branch(Opcode.GOTO, axes).labelBinding(axesDone);
    addTrailing(c, g, header, rank, 0, rank, 0, qi, qb);
    if (masked) addTrailing(c, g, header, rank, 3, rank, 0, qi, mb);
    addTrailing(c, g, header, rank, 4, rank, 0, qi, ob);
    if (two) addTrailing(c, g, header, rank, 5, rank, 0, qi, wb);

    int valueWidth = rt == DataType.FLOAT64 ? 2 : 1;
    int eligible = axis, positive = axis + 2, anyNan = axis + 4, allNeg = axis + 5;
    int j = axis + 6, score = j + 2, x = score + valueWidth;
    int qa = x + 2, ka = qa + 2, qValue = ka + 2, kValue = qValue + valueWidth;
    // Reserve the same highest slot as javac after its lexical score-loop allocation.
    int reserved;
    do {
      reserved = i(c);
      // Allocation only establishes max_locals; the exact phase-local slots are selected above.
    } while (reserved < kValue);
    c.loadConstant(0L)
        .lstore(eligible)
        .loadConstant(0L)
        .lstore(positive)
        .loadConstant(0)
        .istore(anyNan)
        .loadConstant(1)
        .istore(allNeg)
        .loadConstant(0L)
        .lstore(j);
    Label scores = c.newLabel(), scoresDone = c.newLabel();
    c.labelBinding(scores).lload(j);
    get(c, g, 8).lcmp().branch(Opcode.IFGE, scoresDone);
    Label excluded = c.newLabel();
    eligibility(c, s, carrier, g, header, rank, rank, masked, causal, roles, qi, j, mb,
        excluded);
    zero(c, rt);
    store(c, rt, score);
    c.loadConstant(0L).lstore(x);
    Label dots = c.newLabel(), dotsDone = c.newLabel();
    c.labelBinding(dots).lload(x);
    get(c, g, 9).lcmp().branch(Opcode.IFGE, dotsDone);
    address(c, g, header, rank, 0, rank, qb, 1, x, qa);
    address2(c, g, header, rank, 1, rank, kb, j, x, ka);
    load(c, carrier, s, qt, roles[0], qa, qValue, rt);
    load(c, carrier, s, kt, roles[1], ka, kValue, rt);
    loadValue(c, rt, score);
    loadValue(c, rt, qValue);
    loadValue(c, rt, kValue);
    mul(c, rt);
    add(c, rt);
    store(c, rt, score);
    c.lload(x).loadConstant(1L).ladd().lstore(x).branch(Opcode.GOTO, dots).labelBinding(dotsDone);
    loadValue(c, rt, score);
    scale(c, g, rt);
    store(c, rt, score);
    scratchStore(c, scratch, rt, j, score);
    c.lload(eligible).loadConstant(1L).ladd().lstore(eligible);
    c.iload(anyNan);
    loadValue(c, rt, score);
    c.invokestatic(
            rt == DataType.FLOAT64 ? DOUBLE : FLOAT,
            "isNaN",
            MethodTypeDesc.of(
                TypeKind.BOOLEAN.upperBound(),
                rt == DataType.FLOAT64 ? TypeKind.DOUBLE.upperBound()
                    : TypeKind.FLOAT.upperBound()))
        .ior()
        .istore(anyNan);
    Label notPos = c.newLabel();
    loadValue(c, rt, score);
    posInf(c, rt);
    cmp(c, rt);
    c.branch(Opcode.IFNE, notPos);
    c.lload(positive).loadConstant(1L).ladd().lstore(positive).labelBinding(notPos);
    Label stillNeg = c.newLabel();
    loadValue(c, rt, score);
    negInf(c, rt);
    cmp(c, rt);
    c.branch(Opcode.IFEQ, stillNeg);
    c.loadConstant(0).istore(allNeg).labelBinding(stillNeg).labelBinding(excluded);
    c.lload(j)
        .loadConstant(1L)
        .ladd()
        .lstore(j)
        .branch(Opcode.GOTO, scores)
        .labelBinding(scoresDone);

    int mode = j; // 0 zero, 1 nan, 2 positive infinity, 3 ordinary
    Label zeroMode = c.newLabel(), eligibleMode = c.newLabel(), nanMode = c.newLabel();
    Label ordinaryMode = c.newLabel(), modeReady = c.newLabel();
    c.lload(eligible).loadConstant(0L).lcmp().branch(Opcode.IFEQ, zeroMode);
    c.iload(allNeg).branch(Opcode.IFEQ, eligibleMode);
    c.labelBinding(zeroMode).loadConstant(0).branch(Opcode.GOTO, modeReady);
    c.labelBinding(eligibleMode).iload(anyNan).branch(Opcode.IFEQ, nanMode);
    c.loadConstant(1).branch(Opcode.GOTO, modeReady);
    c.labelBinding(nanMode).lload(positive).loadConstant(0L).lcmp()
        .branch(Opcode.IFLE, ordinaryMode);
    c.loadConstant(2).branch(Opcode.GOTO, modeReady).labelBinding(ordinaryMode).loadConstant(3)
        .labelBinding(modeReady).istore(mode);

    // Convert raw scores to final weights in place.
    int max = mode + 1;
    int maxJ = max + valueWidth, maxScore = maxJ + 2;
    int sum = max + valueWidth, expJ = sum + valueWidth, weight = expJ + 2;
    int resultBits = rt == DataType.BFLOAT16 ? mode + 6 : -1;
    int resultRepresented = rt == DataType.BFLOAT16 ? mode + 7 : -1;
    Label nonOrd = c.newLabel(), classified = c.newLabel();
    c.iload(mode).loadConstant(3).branch(Opcode.IF_ICMPNE, nonOrd);
    negInf(c, rt);
    store(c, rt, max);
    c.loadConstant(0L).lstore(maxJ);
    Label maxLoop = c.newLabel(), maxDone = c.newLabel();
    c.labelBinding(maxLoop).lload(maxJ);
    get(c, g, 8).lcmp().branch(Opcode.IFGE, maxDone);
    Label maxSkip = c.newLabel();
    eligibility(c, s, carrier, g, header, rank, rank, masked, causal, roles, qi, maxJ, mb,
        maxSkip);
    scratchLoad(c, scratch, rt, maxJ, maxScore);
    Label keep = c.newLabel();
    loadValue(c, rt, maxScore);
    loadValue(c, rt, max);
    cmp(c, rt);
    c.branch(Opcode.IFLE, keep);
    loadValue(c, rt, maxScore);
    store(c, rt, max);
    c.labelBinding(keep).labelBinding(maxSkip);
    c.lload(maxJ).loadConstant(1L).ladd().lstore(maxJ).branch(Opcode.GOTO, maxLoop)
        .labelBinding(maxDone);
    zero(c, rt);
    store(c, rt, sum);
    c.loadConstant(0L).lstore(expJ);
    Label expLoop = c.newLabel(), expDone = c.newLabel();
    c.labelBinding(expLoop).lload(expJ);
    get(c, g, 8).lcmp().branch(Opcode.IFGE, expDone);
    Label expSkip = c.newLabel();
    eligibility(c, s, carrier, g, header, rank, rank, masked, causal, roles, qi, expJ, mb,
        expSkip);
    scratchLoadValue(c, scratch, rt, expJ);
    loadValue(c, rt, max);
    sub(c, rt);
    toDouble(c, rt)
        .invokestatic(
            STRICT,
            "exp",
            MethodTypeDesc.of(TypeKind.DOUBLE.upperBound(), TypeKind.DOUBLE.upperBound()));
    if (rt == DataType.FLOAT64) c.dstore(weight);
    else c.d2f().fstore(weight);
    scratchStore(c, scratch, rt, expJ, weight);
    loadValue(c, rt, sum);
    loadValue(c, rt, weight);
    add(c, rt);
    store(c, rt, sum);
    c.labelBinding(expSkip);
    c.lload(expJ).loadConstant(1L).ladd().lstore(expJ).branch(Opcode.GOTO, expLoop)
        .labelBinding(expDone);
    c.loadConstant(0L).lstore(expJ);
    Label divLoop = c.newLabel(), divDone = c.newLabel();
    c.labelBinding(divLoop).lload(expJ);
    get(c, g, 8).lcmp().branch(Opcode.IFGE, divDone);
    Label divSkip = c.newLabel();
    eligibility(c, s, carrier, g, header, rank, rank, masked, causal, roles, qi, expJ, mb,
        divSkip);
    scratchDivide(c, scratch, rt, expJ, sum);
    c.labelBinding(divSkip);
    c.lload(expJ)
        .loadConstant(1L)
        .ladd()
        .lstore(expJ)
        .branch(Opcode.GOTO, divLoop)
        .labelBinding(divDone)
        .branch(Opcode.GOTO, classified);
    int classJ = mode + 1, classScore = classJ + 2, classWeight = classScore + valueWidth;
    c.labelBinding(nonOrd).loadConstant(0L).lstore(classJ);
    Label classLoop = c.newLabel(), classDone = c.newLabel();
    c.labelBinding(classLoop).lload(classJ);
    get(c, g, 8).lcmp().branch(Opcode.IFGE, classDone);
    Label classSkip = c.newLabel();
    eligibility(c, s, carrier, g, header, rank, rank, masked, causal, roles, qi, classJ, mb,
        classSkip);
    scratchLoad(c, scratch, rt, classJ, classScore);
    Label positiveWeight = c.newLabel(), zeroWeight = c.newLabel(), modeStore = c.newLabel();
    c.iload(mode).loadConstant(1).branch(Opcode.IF_ICMPNE, positiveWeight);
    nan(c, rt);
    c.branch(Opcode.GOTO, modeStore).labelBinding(positiveWeight);
    c.iload(mode).loadConstant(2).branch(Opcode.IF_ICMPNE, zeroWeight);
    loadValue(c, rt, classScore);
    posInf(c, rt);
    cmp(c, rt);
    c.branch(Opcode.IFNE, zeroWeight);
    one(c, rt);
    c.lload(positive);
    if (rt == DataType.FLOAT64) c.l2d().ddiv();
    else c.l2f().fdiv();
    c.branch(Opcode.GOTO, modeStore).labelBinding(zeroWeight);
    zero(c, rt);
    c.labelBinding(modeStore);
    store(c, rt, classWeight);
    scratchStore(c, scratch, rt, classJ, classWeight);
    c.labelBinding(classSkip);
    c.lload(classJ)
        .loadConstant(1L)
        .ladd()
        .lstore(classJ)
        .branch(Opcode.GOTO, classLoop)
        .labelBinding(classDone)
        .labelBinding(classified);

    // Optional weights store includes explicit positive zero for excluded positions.
    if (two) {
      int weightsJ = mode + 1, weightsValue = weightsJ + 2;
      int weightsAddress = weightsValue + valueWidth;
      c.loadConstant(0L).lstore(weightsJ);
      Label wl = c.newLabel(), wd = c.newLabel();
      c.labelBinding(wl).lload(weightsJ);
      get(c, g, 8).lcmp().branch(Opcode.IFGE, wd);
      if (masked || causal) {
        Label wz = c.newLabel(), selected = c.newLabel();
        eligibility(c, s, carrier, g, header, rank, rank, masked, causal, roles, qi, weightsJ,
            mb,
            wz);
        scratchLoad(c, scratch, rt, weightsJ, weightsValue);
        c.branch(Opcode.GOTO, selected).labelBinding(wz);
        zero(c, rt);
        store(c, rt, weightsValue);
        c.labelBinding(selected);
      } else {
        scratchLoad(c, scratch, rt, weightsJ, weightsValue);
      }
      address(c, g, header, rank, 5, rank, wb, 1, weightsJ, weightsAddress);
      storeResult(c, carrier, s, rt, (int) getConstant(ir, 5), weightsAddress, weightsValue,
          resultBits, resultRepresented);
      c.lload(weightsJ).loadConstant(1L).ladd().lstore(weightsJ).branch(Opcode.GOTO, wl)
          .labelBinding(wd);
    }

    int d = mode + 1, outSum = d + 2, outJ = outSum + valueWidth;
    int outWeight = outJ + 2, vAddress = outWeight + valueWidth;
    int vValue = vAddress + 2, outAddress = outJ;
    c.loadConstant(0L).lstore(d);
    Label outs = c.newLabel(), outsDone = c.newLabel();
    c.labelBinding(outs).lload(d);
    get(c, g, 10).lcmp().branch(Opcode.IFGE, outsDone);
    Label zeroOut = c.newLabel(), outInitialized = c.newLabel(), storeOut = c.newLabel();
    c.iload(mode).loadConstant(1).branch(Opcode.IF_ICMPNE, zeroOut);
    nan(c, rt);
    c.branch(Opcode.GOTO, outInitialized).labelBinding(zeroOut);
    zero(c, rt);
    c.labelBinding(outInitialized);
    store(c, rt, outSum);
    c.iload(mode).branch(Opcode.IFEQ, storeOut);
    c.iload(mode).loadConstant(1).branch(Opcode.IF_ICMPEQ, storeOut);
    c.loadConstant(0L).lstore(outJ);
    Label vl = c.newLabel(), vd = c.newLabel();
    c.labelBinding(vl).lload(outJ);
    get(c, g, 8).lcmp().branch(Opcode.IFGE, vd);
    Label vskip = c.newLabel();
    eligibility(c, s, carrier, g, header, rank, rank, masked, causal, roles, qi, outJ, mb,
        vskip);
    scratchLoad(c, scratch, rt, outJ, outWeight);
    address2(c, g, header, rank, 2, rank, vb, outJ, d, vAddress);
    load(c, carrier, s, vt, roles[2], vAddress, vValue, rt);
    loadValue(c, rt, outSum);
    loadValue(c, rt, outWeight);
    loadValue(c, rt, vValue);
    mul(c, rt);
    add(c, rt);
    store(c, rt, outSum);
    c.labelBinding(vskip);
    c.lload(outJ)
        .loadConstant(1L)
        .ladd()
        .lstore(outJ)
        .branch(Opcode.GOTO, vl)
        .labelBinding(vd)
        .labelBinding(storeOut);
    address(c, g, header, rank, 4, rank, ob, 1, d, outAddress);
    storeResult(c, carrier, s, rt, (int) getConstant(ir, 4), outAddress, outSum, resultBits,
        resultRepresented);
    c.lload(d).loadConstant(1L).ladd().lstore(d).branch(Opcode.GOTO, outs).labelBinding(outsDone);
    c.lload(row).loadConstant(1L).ladd().lstore(row).branch(Opcode.GOTO, rows).labelBinding(done);
  }

  private static int getConstant(CpuKernelIr ir, int layout) {
    int outputs = ir.stores().size();
    return ir.values().size() - outputs + (layout == 5 ? 1 : 0);
  }

  private static DataType type(CpuKernelIr ir, String key) {
    String f = ir.familyIdentity();
    int p = f.indexOf(key) + key.length();
    return DataType.valueOf(f.substring(p, f.indexOf(':', p)));
  }

  private static int[] roles(String identity) {
    int p = identity.indexOf(":roles=[") + 8;
    int end = identity.indexOf(']', p);
    String[] parts = identity.substring(p, end).split(", ");
    int[] result = new int[parts.length];
    for (int i = 0; i < parts.length; i++) result[i] = Integer.parseInt(parts[i]);
    return result;
  }

  private static ClassDesc MathDesc(Class<?> c) {
    return ClassDesc.of(c.getName());
  }

  private static int i(CodeBuilder c) {
    return c.allocateLocal(TypeKind.INT);
  }

  private static int l(CodeBuilder c) {
    return c.allocateLocal(TypeKind.LONG);
  }

  private static int value(CodeBuilder c, DataType t) {
    return c.allocateLocal(t == DataType.FLOAT64 ? TypeKind.DOUBLE : TypeKind.FLOAT);
  }

  private static CodeBuilder get(CodeBuilder c, int g, int x) {
    return c.aload(g).loadConstant(x).laload();
  }

  private static CodeBuilder layoutIndex(CodeBuilder c, int header, int rank, int layout) {
    c.iload(header);
    if (layout == 1) c.iload(rank).iadd().loadConstant(1).iadd();
    else if (layout > 1)
      c.loadConstant(layout).iload(rank).loadConstant(1).iadd().imul().iadd();
    return c;
  }

  private static void base(
      CodeBuilder c, int g, int header, int width, int layout, int target) {
    c.aload(g);
    layoutIndex(c, header, width, layout).laload().lstore(target);
  }

  private static void addStride(
      CodeBuilder c,
      int g,
      int header,
      int width,
      int layout,
      int axis,
      int coord,
      int target) {
    c.lload(target).lload(coord).aload(g).iload(header);
    if (layout == 0) c.loadConstant(1).iadd();
    else if (layout == 1) c.iload(width).iadd().loadConstant(2).iadd();
    else
      c.loadConstant(layout)
          .iload(width)
          .loadConstant(1)
          .iadd()
          .imul()
          .iadd()
          .loadConstant(1)
          .iadd();
    c.iload(axis).iadd().laload()
        .lmul()
        .ladd()
        .lstore(target);
  }

  private static void addTrailing(
      CodeBuilder c,
      int g,
      int header,
      int width,
      int layout,
      int r,
      int trailing,
      int coord,
      int target) {
    c.lload(target).lload(coord).aload(g).iload(header);
    if (layout == 0) c.iload(r).iadd();
    else
      c.loadConstant(layout)
          .iload(width)
          .loadConstant(1)
          .iadd()
          .imul()
          .iadd()
          .iload(r)
          .iadd();
    c.loadConstant(1 - trailing).isub().laload()
        .lmul()
        .ladd()
        .lstore(target);
  }

  private static void address(
      CodeBuilder c,
      int g,
      int header,
      int width,
      int layout,
      int r,
      int base,
      int trailing,
      int coord,
      int target) {
    addressValue(c, g, header, width, layout, r, base, trailing, coord).lstore(target);
  }

  private static CodeBuilder addressValue(
      CodeBuilder c,
      int g,
      int header,
      int width,
      int layout,
      int r,
      int base,
      int trailing,
      int coord) {
    c.lload(base).lload(coord).aload(g);
    if (width == r && layout == 5 && trailing == 1) {
      c.iload(header)
          .loadConstant(5)
          .iload(r)
          .loadConstant(1)
          .iadd()
          .imul()
          .iadd()
          .iload(r)
          .iadd();
    } else trailingIndex(c, header, width, layout, r, trailing);
    return c.laload()
        .lmul()
        .ladd();
  }

  private static CodeBuilder trailingIndex(
      CodeBuilder c, int header, int width, int layout, int r, int trailing) {
    if (width != r) {
      return layoutIndex(c, header, width, layout)
          .loadConstant(1)
          .iadd()
          .iload(r)
          .loadConstant(2 - trailing)
          .isub()
          .iadd();
    }
    c.iload(header);
    int coefficient = layout + 1;
    if (coefficient == 1) c.iload(r).iadd();
    else c.loadConstant(coefficient).iload(r).imul().iadd();
    int offset = layout - 1 + trailing;
    if (offset != 0) c.loadConstant(offset).iadd();
    return c;
  }

  private static void address2(
      CodeBuilder c,
      int g,
      int header,
      int width,
      int layout,
      int r,
      int base,
      int outer,
      int inner,
      int target) {
    c.lload(base).lload(outer).aload(g);
    trailingIndex(c, header, width, layout, r, 0).laload().lmul().ladd()
        .lload(inner).aload(g);
    trailingIndex(c, header, width, layout, r, 1).laload().lmul().ladd().lstore(target);
  }

  private static void eligibility(
      CodeBuilder c,
      CpuKernelSpecialization s,
      CpuCarrierEmitter ce,
      int g,
      int header,
      int width,
      int r,
      boolean masked,
      boolean causal,
      int[] roles,
      int qi,
      int j,
      int mb,
      Label excluded) {
    if (causal) c.lload(j).lload(qi).lcmp().branch(Opcode.IFGT, excluded);
    if (masked) {
      int boundary = roles[3];
      var access = s.carrierPattern().get(boundary);
      ce.beginFrozenLoadAtStackAddress(DataType.BOOL, access, boundary);
      addressValue(c, g, header, width, 3, r, mb, 1, j);
      ce.endFrozenLoadAtStackAddress(DataType.BOOL, access);
      c.branch(Opcode.IFEQ, excluded);
    }
  }

  private static void load(
      CodeBuilder c,
      CpuCarrierEmitter ce,
      CpuKernelSpecialization s,
      DataType t,
      int role,
      int address,
      int local,
      DataType domain) {
    int boundary = role;
    ce.loadFrozen(t, s.carrierPattern().get(boundary), boundary, address, false);
    if (t == DataType.BFLOAT16) {
      c.invokestatic(
              SHORT,
              "toUnsignedInt",
              MethodTypeDesc.of(TypeKind.INT.upperBound(), TypeKind.SHORT.upperBound()))
          .loadConstant(16)
          .ishl()
          .invokestatic(
              FLOAT,
              "intBitsToFloat",
              MethodTypeDesc.of(TypeKind.FLOAT.upperBound(), TypeKind.INT.upperBound()));
    }
    if (domain == DataType.FLOAT64 && t != DataType.FLOAT64) c.f2d();
    else if (domain != DataType.FLOAT64 && t == DataType.FLOAT64) c.d2f();
    store(c, domain, local);
  }

  private static void storeResult(
      CodeBuilder c,
      CpuCarrierEmitter ce,
      CpuKernelSpecialization s,
      DataType t,
      int boundary,
      int address,
      int local,
      int bits,
      int represented) {
    if (t == DataType.BFLOAT16) {
      c.fload(local)
          .invokestatic(
              FLOAT,
              "floatToRawIntBits",
              MethodTypeDesc.of(TypeKind.INT.upperBound(), TypeKind.FLOAT.upperBound()))
          .istore(bits);
      Label finite = c.newLabel(), rounded = c.newLabel();
      c.iload(bits)
          .loadConstant(0x7fffffff)
          .iand()
          .loadConstant(0x7f800000)
          .branch(Opcode.IF_ICMPLE, finite)
          .loadConstant(0x7fc0)
          .branch(Opcode.GOTO, rounded)
          .labelBinding(finite)
          .iload(bits)
          .loadConstant(0x7fff)
          .iadd()
          .iload(bits)
          .loadConstant(16)
          .iushr()
          .loadConstant(1)
          .iand()
          .iadd()
          .loadConstant(16)
          .iushr()
          .labelBinding(rounded)
          .i2s()
          .istore(represented);
      ce.storeFrozen(t, s.carrierPattern().get(boundary), boundary, address, represented, false);
    } else ce.storeFrozen(t, s.carrierPattern().get(boundary), boundary, address, local, false);
  }

  private static void storePositiveZero(
      CodeBuilder c,
      CpuCarrierEmitter ce,
      CpuKernelSpecialization s,
      DataType t,
      int boundary,
      int address,
      int local,
      int represented) {
    if (t == DataType.BFLOAT16) {
      c.loadConstant(0).istore(represented);
      ce.storeFrozen(t, s.carrierPattern().get(boundary), boundary, address, represented, false);
    } else {
      zero(c, t);
      store(c, t, local);
      ce.storeFrozen(t, s.carrierPattern().get(boundary), boundary, address, local, false);
    }
  }

  private static void scratchStore(CodeBuilder c, int scratch, DataType t, int j, int v) {
    c.aload(scratch)
        .getstatic(
            VALUE_LAYOUT,
            t == DataType.FLOAT64 ? "JAVA_DOUBLE_UNALIGNED" : "JAVA_FLOAT_UNALIGNED",
            t == DataType.FLOAT64 ? D_LAYOUT : F_LAYOUT)
        .lload(j)
        .loadConstant(t == DataType.FLOAT64 ? 8L : 4L)
        .lmul();
    loadValue(c, t, v);
    c.invokeinterface(
        SEGMENT,
        "set",
        MethodTypeDesc.of(
            TypeKind.VOID.upperBound(),
            t == DataType.FLOAT64 ? D_LAYOUT : F_LAYOUT,
            TypeKind.LONG.upperBound(),
            t == DataType.FLOAT64 ? TypeKind.DOUBLE.upperBound() : TypeKind.FLOAT.upperBound()));
  }

  private static void scratchLoad(CodeBuilder c, int scratch, DataType t, int j, int v) {
    scratchLoadValue(c, scratch, t, j);
    store(c, t, v);
  }

  private static void scratchLoadValue(CodeBuilder c, int scratch, DataType t, int j) {
    c.aload(scratch)
        .getstatic(
            VALUE_LAYOUT,
            t == DataType.FLOAT64 ? "JAVA_DOUBLE_UNALIGNED" : "JAVA_FLOAT_UNALIGNED",
            t == DataType.FLOAT64 ? D_LAYOUT : F_LAYOUT)
        .lload(j)
        .loadConstant(t == DataType.FLOAT64 ? 8L : 4L)
        .lmul()
        .invokeinterface(
            SEGMENT,
            "get",
            MethodTypeDesc.of(
                t == DataType.FLOAT64 ? TypeKind.DOUBLE.upperBound() : TypeKind.FLOAT.upperBound(),
                t == DataType.FLOAT64 ? D_LAYOUT : F_LAYOUT,
                TypeKind.LONG.upperBound()));
  }

  private static void scratchDivide(
      CodeBuilder c, int scratch, DataType t, int j, int denominator) {
    c.aload(scratch)
        .getstatic(
            VALUE_LAYOUT,
            t == DataType.FLOAT64 ? "JAVA_DOUBLE_UNALIGNED" : "JAVA_FLOAT_UNALIGNED",
            t == DataType.FLOAT64 ? D_LAYOUT : F_LAYOUT)
        .lload(j)
        .loadConstant(t == DataType.FLOAT64 ? 8L : 4L)
        .lmul();
    scratchLoadValue(c, scratch, t, j);
    loadValue(c, t, denominator);
    div(c, t);
    c.invokeinterface(
        SEGMENT,
        "set",
        MethodTypeDesc.of(
            TypeKind.VOID.upperBound(),
            t == DataType.FLOAT64 ? D_LAYOUT : F_LAYOUT,
            TypeKind.LONG.upperBound(),
            t == DataType.FLOAT64 ? TypeKind.DOUBLE.upperBound() : TypeKind.FLOAT.upperBound()));
  }

  private static CodeBuilder loadValue(CodeBuilder c, DataType t, int v) {
    return t == DataType.FLOAT64 ? c.dload(v) : c.fload(v);
  }

  private static void store(CodeBuilder c, DataType t, int v) {
    if (t == DataType.FLOAT64) c.dstore(v);
    else c.fstore(v);
  }

  private static void zero(CodeBuilder c, DataType t) {
    if (t == DataType.FLOAT64) c.loadConstant(+0.0d);
    else c.loadConstant(+0.0f);
  }

  private static void one(CodeBuilder c, DataType t) {
    if (t == DataType.FLOAT64) c.loadConstant(1.0d);
    else c.loadConstant(1.0f);
  }

  private static void nan(CodeBuilder c, DataType t) {
    if (t == DataType.FLOAT64) c.loadConstant(Double.NaN);
    else c.loadConstant(Float.NaN);
  }

  private static void posInf(CodeBuilder c, DataType t) {
    if (t == DataType.FLOAT64) c.loadConstant(Double.POSITIVE_INFINITY);
    else c.loadConstant(Float.POSITIVE_INFINITY);
  }

  private static void negInf(CodeBuilder c, DataType t) {
    if (t == DataType.FLOAT64) c.loadConstant(Double.NEGATIVE_INFINITY);
    else c.loadConstant(Float.NEGATIVE_INFINITY);
  }

  private static void cmp(CodeBuilder c, DataType t) {
    if (t == DataType.FLOAT64) c.dcmpl();
    else c.fcmpl();
  }

  private static void add(CodeBuilder c, DataType t) {
    if (t == DataType.FLOAT64) c.dadd();
    else c.fadd();
  }

  private static void sub(CodeBuilder c, DataType t) {
    if (t == DataType.FLOAT64) c.dsub();
    else c.fsub();
  }

  private static void mul(CodeBuilder c, DataType t) {
    if (t == DataType.FLOAT64) c.dmul();
    else c.fmul();
  }

  private static void div(CodeBuilder c, DataType t) {
    if (t == DataType.FLOAT64) c.ddiv();
    else c.fdiv();
  }

  private static CodeBuilder toDouble(CodeBuilder c, DataType t) {
    return t == DataType.FLOAT64 ? c : c.f2d();
  }

  private static void scale(CodeBuilder c, int g, DataType t) {
    get(c, g, 12)
        .invokestatic(
            DOUBLE,
            "longBitsToDouble",
            MethodTypeDesc.of(TypeKind.DOUBLE.upperBound(), TypeKind.LONG.upperBound()));
    if (t == DataType.FLOAT64) c.dmul();
    else c.d2f().fmul();
  }

}
