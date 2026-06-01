#!/usr/bin/env node

const fs = require("fs");
const path = require("path");

const CHECK = process.argv.includes("--check");
const ROOT = path.resolve(__dirname, "..");
const KERNEL_ROOT = path.join(ROOT, "src/main/java/backend/cpu1/kernels");

const DTYPES = [
  {
    id: "BF16",
    method: "bf16",
    dataType: "BFLOAT16",
    arrayType: "short",
    arrayAccessor: "bfloat16Array",
    layout: "JAVA_SHORT",
    bytes: "Short.BYTES",
    vectorSpecies: "F32",
    zero: "0.0f",
  },
  {
    id: "F32",
    method: "f32",
    dataType: "FLOAT32",
    arrayType: "float",
    arrayAccessor: "float32Array",
    layout: "JAVA_FLOAT",
    bytes: "Float.BYTES",
    vectorSpecies: "F32",
    zero: "0.0f",
  },
  {
    id: "F64",
    method: "f64",
    dataType: "FLOAT64",
    arrayType: "double",
    arrayAccessor: "float64Array",
    layout: "JAVA_DOUBLE",
    bytes: "Double.BYTES",
    vectorSpecies: "F64",
    zero: "0.0d",
  },
];

const BOOL_DTYPE = {
  id: "BOOL",
  method: "bool",
  dataType: "BOOL",
  arrayType: "byte",
  arrayAccessor: "boolArray",
  layout: "JAVA_BYTE",
  bytes: "Byte.BYTES",
  vectorSpecies: null,
  zero: "false",
};

const LAYOUTS = [
  { id: "CONTIGUOUS", method: "Contiguous", vector: true },
  { id: "BROADCAST_INNER", method: "BroadcastInner", scalar: false, vector: true },
  { id: "STRIDED_GENERIC", method: "StridedGeneric", vector: false },
  { id: "STRIDED_RANK2", method: "StridedRank2", vector: false, rank: 2 },
  { id: "STRIDED_RANK3", method: "StridedRank3", vector: false, rank: 3 },
  { id: "STRIDED_RANK4", method: "StridedRank4", vector: false, rank: 4 },
];

const OPS = [
  {
    id: "ADD",
    packageName: "backend.cpu1.kernels.elementwise.binary.add",
    dir: "elementwise/binary/add",
    className: "Cpu1AddLoops",
    arity: "binary",
    operator: "+",
    vectorMethod: "add",
    segmentVectorPrefix: "add",
  },
  {
    id: "SUB",
    packageName: "backend.cpu1.kernels.elementwise.binary.sub",
    dir: "elementwise/binary/sub",
    className: "Cpu1SubLoops",
    arity: "binary",
    operator: "-",
    vectorMethod: "sub",
  },
  {
    id: "MUL",
    packageName: "backend.cpu1.kernels.elementwise.binary.mul",
    dir: "elementwise/binary/mul",
    className: "Cpu1MulLoops",
    arity: "binary",
    operator: "*",
    vectorMethod: "mul",
    segmentVectorPrefix: "mul",
  },
  {
    id: "DIV",
    packageName: "backend.cpu1.kernels.elementwise.binary.div",
    dir: "elementwise/binary/div",
    className: "Cpu1DivLoops",
    arity: "binary",
    operator: "/",
    vectorMethod: "div",
  },
  {
    id: "MIN",
    packageName: "backend.cpu1.kernels.elementwise.binary.min",
    dir: "elementwise/binary/min",
    className: "Cpu1MinLoops",
    arity: "binary",
    binaryKind: "min",
    vectorMethod: "min",
  },
  {
    id: "MAX",
    packageName: "backend.cpu1.kernels.elementwise.binary.max",
    dir: "elementwise/binary/max",
    className: "Cpu1MaxLoops",
    arity: "binary",
    binaryKind: "max",
    vectorMethod: "max",
  },
  {
    id: "POW_TENSOR",
    packageName: "backend.cpu1.kernels.elementwise.binary.powtensor",
    dir: "elementwise/binary/powtensor",
    className: "Cpu1PowTensorLoops",
    arity: "binary",
    binaryKind: "powTensor",
    supportsVector: false,
  },
  {
    id: "WHERE",
    packageName: "backend.cpu1.kernels.elementwise.where",
    dir: "elementwise/where",
    className: "Cpu1WhereLoops",
    arity: "ternary",
    supportsVector: false,
  },
  {
    id: "GT",
    packageName: "backend.cpu1.kernels.elementwise.compare",
    dir: "elementwise/compare",
    className: "Cpu1CompareLoops",
    arity: "binary",
    binaryKind: "gt",
    methodPrefix: "gt",
    outputDataType: BOOL_DTYPE,
    storageKinds: ["ARRAY"],
    supportsVector: false,
  },
  {
    id: "GE",
    packageName: "backend.cpu1.kernels.elementwise.compare",
    dir: "elementwise/compare",
    className: "Cpu1CompareLoops",
    arity: "binary",
    binaryKind: "ge",
    methodPrefix: "ge",
    outputDataType: BOOL_DTYPE,
    storageKinds: ["ARRAY"],
    supportsVector: false,
  },
  {
    id: "LT",
    packageName: "backend.cpu1.kernels.elementwise.compare",
    dir: "elementwise/compare",
    className: "Cpu1CompareLoops",
    arity: "binary",
    binaryKind: "lt",
    methodPrefix: "lt",
    outputDataType: BOOL_DTYPE,
    storageKinds: ["ARRAY"],
    supportsVector: false,
  },
  {
    id: "LE",
    packageName: "backend.cpu1.kernels.elementwise.compare",
    dir: "elementwise/compare",
    className: "Cpu1CompareLoops",
    arity: "binary",
    binaryKind: "le",
    methodPrefix: "le",
    outputDataType: BOOL_DTYPE,
    storageKinds: ["ARRAY"],
    supportsVector: false,
  },
  {
    id: "EQ",
    packageName: "backend.cpu1.kernels.elementwise.compare",
    dir: "elementwise/compare",
    className: "Cpu1CompareLoops",
    arity: "binary",
    binaryKind: "eq",
    methodPrefix: "eq",
    outputDataType: BOOL_DTYPE,
    storageKinds: ["ARRAY"],
    supportsVector: false,
  },
  {
    id: "NE",
    packageName: "backend.cpu1.kernels.elementwise.compare",
    dir: "elementwise/compare",
    className: "Cpu1CompareLoops",
    arity: "binary",
    binaryKind: "ne",
    methodPrefix: "ne",
    outputDataType: BOOL_DTYPE,
    storageKinds: ["ARRAY"],
    supportsVector: false,
  },
  {
    id: "LOGICAL_AND",
    packageName: "backend.cpu1.kernels.elementwise.logical",
    dir: "elementwise/logical",
    className: "Cpu1LogicalLoops",
    arity: "binary",
    binaryKind: "logicalAnd",
    methodPrefix: "logicalAnd",
    inputDataTypes: [BOOL_DTYPE],
    outputDataType: BOOL_DTYPE,
    storageKinds: ["ARRAY"],
    supportsVector: false,
  },
  {
    id: "LOGICAL_OR",
    packageName: "backend.cpu1.kernels.elementwise.logical",
    dir: "elementwise/logical",
    className: "Cpu1LogicalLoops",
    arity: "binary",
    binaryKind: "logicalOr",
    methodPrefix: "logicalOr",
    inputDataTypes: [BOOL_DTYPE],
    outputDataType: BOOL_DTYPE,
    storageKinds: ["ARRAY"],
    supportsVector: false,
  },
  {
    id: "LOGICAL_NOT",
    packageName: "backend.cpu1.kernels.elementwise.logical",
    dir: "elementwise/logical",
    className: "Cpu1LogicalLoops",
    arity: "unary",
    unaryKind: "logicalNot",
    methodPrefix: "logicalNot",
    inputDataTypes: [BOOL_DTYPE],
    outputDataType: BOOL_DTYPE,
    storageKinds: ["ARRAY"],
    supportsVector: false,
  },
  {
    id: "MUL_SCALAR",
    packageName: "backend.cpu1.kernels.elementwise.unary.mulscalar",
    dir: "elementwise/unary/mulscalar",
    className: "Cpu1MulScalarLoops",
    arity: "unary",
    unaryKind: "mulScalar",
  },
  {
    id: "NEG",
    packageName: "backend.cpu1.kernels.elementwise.unary.neg",
    dir: "elementwise/unary/neg",
    className: "Cpu1NegLoops",
    arity: "unary",
    unaryKind: "neg",
  },
  {
    id: "ABS",
    packageName: "backend.cpu1.kernels.elementwise.unary.abs",
    dir: "elementwise/unary/abs",
    className: "Cpu1AbsLoops",
    arity: "unary",
    unaryKind: "abs",
    vectorOperator: "ABS",
  },
  {
    id: "INV",
    packageName: "backend.cpu1.kernels.elementwise.unary.inv",
    dir: "elementwise/unary/inv",
    className: "Cpu1InvLoops",
    arity: "unary",
    unaryKind: "inv",
  },
  {
    id: "EXP",
    packageName: "backend.cpu1.kernels.elementwise.unary.exp",
    dir: "elementwise/unary/exp",
    className: "Cpu1ExpLoops",
    arity: "unary",
    unaryKind: "exp",
    vectorOperator: "EXP",
  },
  {
    id: "ERF",
    packageName: "backend.cpu1.kernels.elementwise.unary.erf",
    dir: "elementwise/unary/erf",
    className: "Cpu1ErfLoops",
    arity: "unary",
    unaryKind: "erf",
    supportsVector: false,
  },
  {
    id: "FAST_EXP",
    packageName: "backend.cpu1.kernels.elementwise.unary.exp",
    dir: "elementwise/unary/exp",
    className: "Cpu1ExpLoops",
    arity: "unary",
    unaryKind: "fastExp",
    methodPrefix: "fastExp",
    supportsVector: false,
  },
  {
    id: "LOG",
    packageName: "backend.cpu1.kernels.elementwise.unary.log",
    dir: "elementwise/unary/log",
    className: "Cpu1LogLoops",
    arity: "unary",
    unaryKind: "log",
    vectorOperator: "LOG",
  },
  {
    id: "TANH",
    packageName: "backend.cpu1.kernels.elementwise.unary.tanh",
    dir: "elementwise/unary/tanh",
    className: "Cpu1TanhLoops",
    arity: "unary",
    unaryKind: "tanh",
    vectorOperator: "TANH",
  },
  {
    id: "FAST_TANH",
    packageName: "backend.cpu1.kernels.elementwise.unary.tanh",
    dir: "elementwise/unary/tanh",
    className: "Cpu1TanhLoops",
    arity: "unary",
    unaryKind: "fastTanh",
    methodPrefix: "fastTanh",
    supportsVector: false,
  },
  {
    id: "SIGMOID",
    packageName: "backend.cpu1.kernels.elementwise.unary.sigmoid",
    dir: "elementwise/unary/sigmoid",
    className: "Cpu1SigmoidLoops",
    arity: "unary",
    unaryKind: "sigmoid",
  },
  {
    id: "SQRT",
    packageName: "backend.cpu1.kernels.elementwise.unary.sqrt",
    dir: "elementwise/unary/sqrt",
    className: "Cpu1SqrtLoops",
    arity: "unary",
    unaryKind: "sqrt",
    vectorOperator: "SQRT",
  },
  {
    id: "POW",
    packageName: "backend.cpu1.kernels.elementwise.unary.pow",
    dir: "elementwise/unary/pow",
    className: "Cpu1PowLoops",
    arity: "unary",
    unaryKind: "powScalar",
    supportsVector: false,
  },
  {
    id: "CLAMP_MIN",
    packageName: "backend.cpu1.kernels.elementwise.unary.clampmin",
    dir: "elementwise/unary/clampmin",
    className: "Cpu1ClampMinLoops",
    arity: "unary",
    unaryKind: "clampMin",
  },
  {
    id: "CLAMP_MAX",
    packageName: "backend.cpu1.kernels.elementwise.unary.clampmax",
    dir: "elementwise/unary/clampmax",
    className: "Cpu1ClampMaxLoops",
    arity: "unary",
    unaryKind: "clampMax",
  },
  {
    id: "FLOOR",
    packageName: "backend.cpu1.kernels.elementwise.unary.floor",
    dir: "elementwise/unary/floor",
    className: "Cpu1FloorLoops",
    arity: "unary",
    unaryKind: "floor",
    supportsVector: false,
  },
  {
    id: "CEIL",
    packageName: "backend.cpu1.kernels.elementwise.unary.ceil",
    dir: "elementwise/unary/ceil",
    className: "Cpu1CeilLoops",
    arity: "unary",
    unaryKind: "ceil",
    supportsVector: false,
  },
  {
    id: "SIGN",
    packageName: "backend.cpu1.kernels.elementwise.unary.sign",
    dir: "elementwise/unary/sign",
    className: "Cpu1SignLoops",
    arity: "unary",
    unaryKind: "sign",
    supportsVector: false,
  },
  {
    id: "RELU",
    packageName: "backend.cpu1.kernels.elementwise.unary.relu",
    dir: "elementwise/unary/relu",
    className: "Cpu1ReluLoops",
    arity: "unary",
    unaryKind: "relu",
  },
];

function generatedHeader() {
  return [
    "// GENERATED by scripts/generate-cpu1-elementwise.js. Do not edit by hand.",
    "",
  ].join("\n");
}

function indent(text, spaces) {
  const pad = " ".repeat(spaces);
  return text.split("\n").map(line => (line ? pad + line : line)).join("\n");
}

function methodName(dtype, storage, layout, vectorKind) {
  return methodNameFor(null, dtype, storage, layout, vectorKind);
}

function methodNameFor(op, dtype, storage, layout, vectorKind) {
  const prefix = op?.methodPrefix ?? "";
  if (op?.arity === "ternary" && op.trueDtype && op.falseDtype) {
    const storageWord = storage === "ARRAY" ? "Array" : "Segment";
    const vectorWord = vectorKind === "VECTOR" ? "Vector" : "Scalar";
    return `${dtype.method}From${capitalize(op.trueDtype.method)}${capitalize(op.falseDtype.method)}${storageWord}${layout.method}${vectorWord}`;
  }
  const storageWord = storage === "ARRAY" ? "Array" : "Segment";
  const vectorWord = vectorKind === "VECTOR" ? "Vector" : "Scalar";
  const base = `${dtype.method}${storageWord}${layout.method}${vectorWord}`;
  return prefix ? `${prefix}${base[0].toUpperCase()}${base.slice(1)}` : base;
}

function capitalize(value) {
  return `${value[0].toUpperCase()}${value.slice(1)}`;
}

function storageKind(storage) {
  return storage === "ARRAY" ? "JAVA_ARRAY" : "MEMORY_SEGMENT";
}

function entries() {
  const out = [];
  for (const op of OPS) {
    if (op.arity === "ternary") {
      for (const outputDtype of DTYPES) {
        for (const trueDtype of DTYPES) {
          for (const falseDtype of DTYPES) {
            if (promoteFloating(trueDtype, falseDtype).id !== outputDtype.id) {
              continue;
            }
            const typedOp = { ...op, trueDtype, falseDtype };
            for (const storage of op.storageKinds ?? ["ARRAY", "SEGMENT"]) {
              for (const layout of LAYOUTS) {
                if (layout.scalar !== false) {
                  out.push({
                    op: typedOp,
                    dtype: outputDtype,
                    storage,
                    layout,
                    vectorKind: "SCALAR",
                    inputDTypes: [BOOL_DTYPE, trueDtype, falseDtype],
                  });
                }
              }
            }
          }
        }
      }
      continue;
    }
    for (const dtype of op.inputDataTypes ?? DTYPES) {
      for (const storage of op.storageKinds ?? ["ARRAY", "SEGMENT"]) {
        for (const layout of LAYOUTS) {
          if (layout.scalar !== false) {
            out.push({ op, dtype, storage, layout, vectorKind: "SCALAR", inputDTypes: inputDTypesFor(op, dtype) });
          }
          if (layout.vector && op.supportsVector !== false) {
            out.push({ op, dtype, storage, layout, vectorKind: "VECTOR", inputDTypes: inputDTypesFor(op, dtype) });
          }
        }
      }
    }
  }
  return out.sort((a, b) => kernelId(a).localeCompare(kernelId(b)));
}

function promoteFloating(left, right) {
  if (left.id === "F64" || right.id === "F64") return DTYPES[2];
  if (left.id === "F32" || right.id === "F32") return DTYPES[1];
  return DTYPES[0];
}

function inputDTypesFor(op, dtype) {
  if (op.inputDataTypes) {
    return op.arity === "binary" ? [dtype, dtype] : [dtype];
  }
  if (op.arity === "binary") {
    return [dtype, dtype];
  }
  return [dtype];
}

function kernelId(entry) {
  if (entry.op.id === "WHERE") {
    return `${entry.op.id}_${entry.dtype.id}_FROM_${entry.op.trueDtype.id}_${entry.op.falseDtype.id}_${entry.storage}_${entry.layout.id}_${entry.vectorKind}`;
  }
  return `${entry.op.id}_${entry.dtype.id}_${entry.storage}_${entry.layout.id}_${entry.vectorKind}`;
}

function readArray(dtype, name, offset) {
  if (dtype.id === "BOOL") {
    return `${name}[${offset}] != 0`;
  }
  if (dtype.id === "BF16") {
    return `TensorDTypeOps.fromBFloat16Bits(${name}[${offset}])`;
  }
  return `${name}[${offset}]`;
}

function writeArray(dtype, name, offset, value) {
  if (dtype.id === "BOOL") {
    return `${name}[${offset}] = (byte) (${value} ? 1 : 0);`;
  }
  if (dtype.id === "BF16") {
    return `${name}[${offset}] = TensorDTypeOps.toBFloat16Bits(${value});`;
  }
  return `${name}[${offset}] = ${value};`;
}

function readSegment(dtype, name, offset) {
  const raw = `${name}.get(${dtype.layout}, (long) (${offset}) * ${dtype.bytes})`;
  if (dtype.id === "BOOL") {
    return `${raw} != 0`;
  }
  if (dtype.id === "BF16") {
    return `TensorDTypeOps.fromBFloat16Bits(${raw})`;
  }
  return raw;
}

function writeSegment(dtype, name, offset, value) {
  const stored = dtype.id === "BF16" ? `TensorDTypeOps.toBFloat16Bits(${value})`
    : dtype.id === "BOOL" ? `(byte) (${value} ? 1 : 0)`
      : value;
  return `${name}.set(${dtype.layout}, (long) (${offset}) * ${dtype.bytes}, ${stored});`;
}

function binaryExpr(op, left, right) {
  switch (op.binaryKind) {
    case "min":
      return `Math.min(${left}, ${right})`;
    case "max":
      return `Math.max(${left}, ${right})`;
    case "powTensor":
      return `(float) Math.pow(${left}, ${right})`;
    case "gt":
      return `${left} > ${right}`;
    case "ge":
      return `${left} >= ${right}`;
    case "lt":
      return `${left} < ${right}`;
    case "le":
      return `${left} <= ${right}`;
    case "eq":
      return `${left} == ${right}`;
    case "ne":
      return `${left} != ${right}`;
    case "logicalAnd":
      return `${left} && ${right}`;
    case "logicalOr":
      return `${left} || ${right}`;
    case undefined:
      break;
    default:
      throw new Error(`Unsupported binary kind ${op.binaryKind}`);
  }
  return `${left} ${op.operator} ${right}`;
}

function binaryExprForDType(op, dtype, left, right) {
  if (op.binaryKind === "powTensor" && dtype.id === "F64") {
    return `Math.pow(${left}, ${right})`;
  }
  return binaryExpr(op, left, right);
}

function unaryExpr(op, dtype, value) {
  const scalar = scalarArg(dtype);
  switch (op.unaryKind) {
    case "relu":
      return `${value} > ${dtype.zero} ? ${value} : ${dtype.zero}`;
    case "neg":
      return `-${value}`;
    case "logicalNot":
      return `!${value}`;
    case "mulScalar":
      return `${value} * ${scalar}`;
    case "abs":
      return dtype.id === "F64" ? `Math.abs(${value})` : `Math.abs(${value})`;
    case "inv":
      return dtype.id === "F64" ? `1.0d / ${value}` : `1.0f / ${value}`;
    case "exp":
      return dtype.id === "F64" ? `Math.exp(${value})` : `(float) Math.exp(${value})`;
    case "erf":
      return dtype.id === "F64" ? `SpecialFunctions.erf(${value})` : `SpecialFunctions.erf(${value})`;
    case "fastExp":
      return dtype.id === "F64"
        ? `FastTranscendentals.fastExpF64(${value})`
        : `FastTranscendentals.fastExpF32(${value})`;
    case "log":
      return dtype.id === "F64" ? `Math.log(${value})` : `(float) Math.log(${value})`;
    case "tanh":
      return dtype.id === "F64" ? `Math.tanh(${value})` : `(float) Math.tanh(${value})`;
    case "fastTanh":
      return dtype.id === "F64"
        ? `FastTranscendentals.fastTanhF64(${value})`
        : `FastTranscendentals.fastTanhF32(${value})`;
    case "sigmoid":
      return dtype.id === "F64"
        ? `1.0d / (1.0d + Math.exp(-${value}))`
        : `1.0f / (1.0f + (float) Math.exp(-${value}))`;
    case "sqrt":
      return dtype.id === "F64" ? `Math.sqrt(${value})` : `(float) Math.sqrt(${value})`;
    case "powScalar":
      return dtype.id === "F64" ? `Math.pow(${value}, ${scalar})` : `(float) Math.pow(${value}, ${scalar})`;
    case "clampMin":
      return `Math.max(${value}, ${scalar})`;
    case "clampMax":
      return `Math.min(${value}, ${scalar})`;
    case "floor":
      return dtype.id === "F64" ? `Math.floor(${value})` : `(float) Math.floor(${value})`;
    case "ceil":
      return dtype.id === "F64" ? `Math.ceil(${value})` : `(float) Math.ceil(${value})`;
    case "sign":
      return `${value} > ${dtype.zero} ? ${dtype.id === "F64" ? "1.0d" : "1.0f"} : (${value} < ${dtype.zero} ? ${dtype.id === "F64" ? "-1.0d" : "-1.0f"} : ${dtype.zero})`;
    default:
      throw new Error(`Unsupported unary kind ${op.unaryKind}`);
  }
}

function scalarValueType(dtype) {
  if (dtype.id === "BOOL") {
    return "boolean";
  }
  return dtype.id === "F64" ? "double" : "float";
}

function outputDType(op, dtype) {
  return op.outputDataType ?? dtype;
}

function scalarArg(dtype) {
  return dtype.id === "F64" ? "args.scalarF64()" : "args.scalarF32()";
}

function vectorLiteral(dtype, value) {
  if (dtype.id === "F64") {
    return `${value}d`;
  }
  return `${value}f`;
}

function unaryVectorSetup(op, dtype, vectorClass) {
  switch (op.unaryKind) {
    case "relu":
      return [`${vectorClass} zero = ${vectorClass}.zero(${dtype.vectorSpecies});`];
    case "sigmoid":
      return [
        `${vectorClass} half = ${vectorClass}.broadcast(${dtype.vectorSpecies}, ${vectorLiteral(dtype, "0.5")});`,
        `${vectorClass} one = ${vectorClass}.broadcast(${dtype.vectorSpecies}, ${vectorLiteral(dtype, "1.0")});`,
      ];
    case "mulScalar":
      return [`${vectorClass} scalar = ${vectorClass}.broadcast(${dtype.vectorSpecies}, ${scalarArg(dtype)});`];
    case "clampMin":
      return [`${vectorClass} min = ${vectorClass}.broadcast(${dtype.vectorSpecies}, ${scalarArg(dtype)});`];
    case "clampMax":
      return [`${vectorClass} max = ${vectorClass}.broadcast(${dtype.vectorSpecies}, ${scalarArg(dtype)});`];
    case "inv":
      return [`${vectorClass} one = ${vectorClass}.broadcast(${dtype.vectorSpecies}, ${vectorLiteral(dtype, "1.0")});`];
    case "neg":
    case "abs":
    case "exp":
    case "log":
    case "tanh":
    case "sqrt":
      return [];
    default:
      throw new Error(`Unsupported vector unary kind ${op.unaryKind}`);
  }
}

function unaryVectorExpr(op, dtype, inputExpression) {
  switch (op.unaryKind) {
    case "relu":
      return `${inputExpression}.max(zero)`;
    case "neg":
      return `${inputExpression}.neg()`;
    case "mulScalar":
      return `${inputExpression}.mul(scalar)`;
    case "abs":
      return `${inputExpression}.lanewise(VectorOperators.ABS)`;
    case "inv":
      return `one.div(${inputExpression})`;
    case "exp":
    case "log":
    case "tanh":
    case "sqrt":
      return `${inputExpression}.lanewise(VectorOperators.${op.vectorOperator})`;
    case "sigmoid":
      return `${inputExpression}.mul(half).lanewise(VectorOperators.TANH).add(one).mul(half)`;
    case "clampMin":
      return `${inputExpression}.max(min)`;
    case "clampMax":
      return `${inputExpression}.min(max)`;
    default:
      throw new Error(`Unsupported vector unary kind ${op.unaryKind}`);
  }
}

function vectorClass(dtype) {
  return dtype.id === "F64" ? "DoubleVector" : "FloatVector";
}

function arrayBroadcastVectorLoad(dtype, arrayName, baseName, modeName, linearIndex, innerIndex) {
  const klass = vectorClass(dtype);
  return `(${modeName} == BROADCAST_SCALAR ? ${klass}.broadcast(${dtype.vectorSpecies}, ${arrayName}[${baseName}]) : (${modeName} == BROADCAST_INNER ? ${klass}.fromArray(${dtype.vectorSpecies}, ${arrayName}, ${baseName} + ${innerIndex}) : ${klass}.fromArray(${dtype.vectorSpecies}, ${arrayName}, ${baseName} + ${linearIndex})))`;
}

function arrayBroadcastScalarRead(dtype, arrayName, baseName, modeName, linearIndex, innerIndex) {
  const offset = `${modeName} == BROADCAST_SCALAR ? ${baseName} : (${modeName} == BROADCAST_INNER ? ${baseName} + ${innerIndex} : ${baseName} + ${linearIndex})`;
  return readArray(dtype, arrayName, offset);
}

function segmentBroadcastVectorLoad(dtype, segmentName, baseName, modeName, linearIndex, innerIndex) {
  const klass = vectorClass(dtype);
  return `(${modeName} == BROADCAST_SCALAR ? ${klass}.broadcast(${dtype.vectorSpecies}, ${segmentName}.get(${dtype.layout}, ${baseName})) : (${modeName} == BROADCAST_INNER ? ${klass}.fromMemorySegment(${dtype.vectorSpecies}, ${segmentName}, ${baseName} + (long) ${innerIndex} * ${dtype.bytes}, ORDER) : ${klass}.fromMemorySegment(${dtype.vectorSpecies}, ${segmentName}, ${baseName} + (long) ${linearIndex} * ${dtype.bytes}, ORDER)))`;
}

function segmentBroadcastScalarRead(dtype, segmentName, baseName, modeName, linearIndex, innerIndex) {
  return `(${modeName} == BROADCAST_SCALAR ? ${segmentName}.get(${dtype.layout}, ${baseName}) : (${modeName} == BROADCAST_INNER ? ${segmentName}.get(${dtype.layout}, ${baseName} + (long) ${innerIndex} * ${dtype.bytes}) : ${segmentName}.get(${dtype.layout}, ${baseName} + (long) ${linearIndex} * ${dtype.bytes})))`;
}

function arrayDeclarations(op, dtype) {
  const outDType = outputDType(op, dtype);
  if (op.arity === "ternary") {
    return [
      "Cpu1TensorView conditionView = args.input(0);",
      "Cpu1TensorView trueView = args.input(1);",
      "Cpu1TensorView falseView = args.input(2);",
      "Cpu1TensorView outputView = args.output();",
      "byte[] condition = conditionView.boolArray();",
      `${op.trueDtype.arrayType}[] trueValues = trueView.${op.trueDtype.arrayAccessor}();`,
      `${op.falseDtype.arrayType}[] falseValues = falseView.${op.falseDtype.arrayAccessor}();`,
      `${outDType.arrayType}[] output = outputView.${outDType.arrayAccessor}();`,
    ];
  }
  if (op.arity === "binary") {
    return [
      "Cpu1TensorView leftView = args.input(0);",
      "Cpu1TensorView rightView = args.input(1);",
      "Cpu1TensorView outputView = args.output();",
      `${dtype.arrayType}[] left = leftView.${dtype.arrayAccessor}();`,
      `${dtype.arrayType}[] right = rightView.${dtype.arrayAccessor}();`,
      `${outDType.arrayType}[] output = outputView.${outDType.arrayAccessor}();`,
    ];
  }
  return [
    "Cpu1TensorView inputView = args.input(0);",
    "Cpu1TensorView outputView = args.output();",
    `${dtype.arrayType}[] input = inputView.${dtype.arrayAccessor}();`,
    `${outDType.arrayType}[] output = outputView.${outDType.arrayAccessor}();`,
  ];
}

function segmentDeclarations(op) {
  if (op.arity === "ternary") {
    return [
      "Cpu1TensorView conditionView = args.input(0);",
      "Cpu1TensorView trueView = args.input(1);",
      "Cpu1TensorView falseView = args.input(2);",
      "Cpu1TensorView outputView = args.output();",
      "MemorySegment condition = conditionView.segment();",
      "MemorySegment trueValues = trueView.segment();",
      "MemorySegment falseValues = falseView.segment();",
      "MemorySegment output = outputView.segment();",
    ];
  }
  if (op.arity === "binary") {
    return [
      "Cpu1TensorView leftView = args.input(0);",
      "Cpu1TensorView rightView = args.input(1);",
      "Cpu1TensorView outputView = args.output();",
      "MemorySegment left = leftView.segment();",
      "MemorySegment right = rightView.segment();",
      "MemorySegment output = outputView.segment();",
    ];
  }
  return [
    "Cpu1TensorView inputView = args.input(0);",
    "Cpu1TensorView outputView = args.output();",
    "MemorySegment input = inputView.segment();",
    "MemorySegment output = outputView.segment();",
  ];
}

function contiguousScalar(op, dtype, storage) {
  const outDType = outputDType(op, dtype);
  const lines = storage === "ARRAY" ? arrayDeclarations(op, dtype) : segmentDeclarations(op);
  if (op.arity === "ternary") {
    const condition = storage === "ARRAY"
      ? readArray(BOOL_DTYPE, "condition", "conditionOffset")
      : readSegment(BOOL_DTYPE, "condition", "conditionOffset");
    const trueValue = storage === "ARRAY"
      ? readArray(op.trueDtype, "trueValues", "trueOffset")
      : readSegment(op.trueDtype, "trueValues", "trueOffset");
    const falseValue = storage === "ARRAY"
      ? readArray(op.falseDtype, "falseValues", "falseOffset")
      : readSegment(op.falseDtype, "falseValues", "falseOffset");
    lines.push(
      "int conditionBase = conditionView.storageOffset();",
      "int trueBase = trueView.storageOffset();",
      "int falseBase = falseView.storageOffset();",
      "int outputBase = outputView.storageOffset();",
      "for (int i = startInclusive; i < endExclusive; i++) {",
      "    int conditionOffset = conditionBase + i;",
      "    int trueOffset = trueBase + i;",
      "    int falseOffset = falseBase + i;",
      "    int outputOffset = outputBase + i;",
      `    ${scalarValueType(outDType)} selected = ${condition} ? ${trueValue} : ${falseValue};`,
      storage === "ARRAY"
        ? `    ${writeArray(outDType, "output", "outputOffset", "selected")}`
        : `    ${writeSegment(outDType, "output", "outputOffset", "selected")}`,
      "}",
    );
    return lines.join("\n");
  }
  if (op.arity === "binary") {
    lines.push(
      "int leftBase = leftView.storageOffset();",
      "int rightBase = rightView.storageOffset();",
      "int outputBase = outputView.storageOffset();",
      "for (int i = startInclusive; i < endExclusive; i++) {",
      "    int leftOffset = leftBase + i;",
      "    int rightOffset = rightBase + i;",
      "    int outputOffset = outputBase + i;",
    );
    const left = storage === "ARRAY" ? readArray(dtype, "left", "leftOffset") : readSegment(dtype, "left", "leftOffset");
    const right = storage === "ARRAY" ? readArray(dtype, "right", "rightOffset") : readSegment(dtype, "right", "rightOffset");
    const expr = binaryExprForDType(op, dtype, left, right);
    lines.push(storage === "ARRAY"
      ? `    ${writeArray(outDType, "output", "outputOffset", expr)}`
      : `    ${writeSegment(dtype, "output", "outputOffset", expr)}`);
    lines.push("}");
    return lines.join("\n");
  }
  lines.push(
    "int inputBase = inputView.storageOffset();",
    "int outputBase = outputView.storageOffset();",
    "for (int i = startInclusive; i < endExclusive; i++) {",
    "    int inputOffset = inputBase + i;",
    "    int outputOffset = outputBase + i;",
  );
  const value = storage === "ARRAY" ? readArray(dtype, "input", "inputOffset") : readSegment(dtype, "input", "inputOffset");
  const expr = unaryExpr(op, dtype, "value");
  lines.push(`    ${scalarValueType(dtype)} value = ${value};`);
  lines.push(storage === "ARRAY"
    ? `    ${writeArray(outDType, "output", "outputOffset", expr)}`
    : `    ${writeSegment(dtype, "output", "outputOffset", expr)}`);
  lines.push("}");
  return lines.join("\n");
}

function arrayBroadcastInnerVector(op, dtype) {
  if (dtype.id === "BF16") {
    return bf16BroadcastInnerVector(op, "ARRAY");
  }
  const lines = arrayDeclarations(op, dtype);
  const klass = vectorClass(dtype);
  const outDType = outputDType(op, dtype);
  if (op.arity === "binary") {
    const leftVector = arrayBroadcastVectorLoad(dtype, "left", "leftBase", "leftMode", "i", "vectorInner");
    const rightVector = arrayBroadcastVectorLoad(dtype, "right", "rightBase", "rightMode", "i", "vectorInner");
    const leftScalar = arrayBroadcastScalarRead(dtype, "left", "leftBase", "leftMode", "i", "scalarInner");
    const rightScalar = arrayBroadcastScalarRead(dtype, "right", "rightBase", "rightMode", "i", "scalarInner");
    lines.push(
      "int leftBase = leftView.storageOffset();",
      "int rightBase = rightView.storageOffset();",
      "int outputBase = outputView.storageOffset();",
      "int leftMode = broadcastVectorMode(leftView);",
      "int rightMode = broadcastVectorMode(rightView);",
      "int innerSize = broadcastVectorInnerSize(outputView);",
      "int i = startInclusive;",
      "while (i < endExclusive) {",
      "    int rowInner = i % innerSize;",
      "    int rowEnd = Math.min(endExclusive, i + innerSize - rowInner);",
      `    int upper = i + ${dtype.vectorSpecies}.loopBound(rowEnd - i);`,
      `    for (; i < upper; i += ${dtype.vectorSpecies}.length()) {`,
      "        int vectorInner = i % innerSize;",
      `        ${leftVector}`,
      `                .${op.vectorMethod}(${rightVector})`,
      "                .intoArray(output, outputBase + i);",
      "    }",
      "    for (; i < rowEnd; i++) {",
      "        int scalarInner = i % innerSize;",
      `        output[outputBase + i] = ${binaryExprForDType(op, dtype, leftScalar, rightScalar)};`,
      "    }",
      "}",
    );
    return lines.join("\n");
  }
  const inputVector = arrayBroadcastVectorLoad(dtype, "input", "inputBase", "inputMode", "i", "vectorInner");
  const inputScalar = arrayBroadcastScalarRead(dtype, "input", "inputBase", "inputMode", "i", "scalarInner");
  lines.push(
    "int inputBase = inputView.storageOffset();",
    "int outputBase = outputView.storageOffset();",
    "int inputMode = broadcastVectorMode(inputView);",
    "int innerSize = broadcastVectorInnerSize(outputView);",
    "int i = startInclusive;",
  );
  lines.push(...unaryVectorSetup(op, dtype, klass));
  lines.push(
    "while (i < endExclusive) {",
    "    int rowInner = i % innerSize;",
    "    int rowEnd = Math.min(endExclusive, i + innerSize - rowInner);",
    `    int upper = i + ${dtype.vectorSpecies}.loopBound(rowEnd - i);`,
    `    for (; i < upper; i += ${dtype.vectorSpecies}.length()) {`,
    "        int vectorInner = i % innerSize;",
    `        ${unaryVectorExpr(op, dtype, inputVector)}`,
    "                .intoArray(output, outputBase + i);",
    "    }",
    "    for (; i < rowEnd; i++) {",
    "        int scalarInner = i % innerSize;",
    `        ${scalarValueType(dtype)} value = ${inputScalar};`,
    `        ${writeArray(outDType, "output", "outputBase + i", unaryExpr(op, dtype, "value"))}`,
    "    }",
    "}",
  );
  return lines.join("\n");
}

function arrayVector(op, dtype) {
  if (dtype.id === "BF16") {
    return bf16Vector(op, "ARRAY");
  }
  const lines = arrayDeclarations(op, dtype);
  if (op.arity === "binary") {
    lines.push(
      "int leftBase = leftView.storageOffset();",
      "int rightBase = rightView.storageOffset();",
      "int outputBase = outputView.storageOffset();",
      `int upper = startInclusive + ${dtype.vectorSpecies}.loopBound(endExclusive - startInclusive);`,
      "int i = startInclusive;",
      `for (; i < upper; i += ${dtype.vectorSpecies}.length()) {`,
      `    ${dtype.id === "F32" ? "FloatVector" : "DoubleVector"}.fromArray(${dtype.vectorSpecies}, left, leftBase + i)`,
      `            .${op.vectorMethod}(${dtype.id === "F32" ? "FloatVector" : "DoubleVector"}.fromArray(${dtype.vectorSpecies}, right, rightBase + i))`,
      "            .intoArray(output, outputBase + i);",
      "}",
      "for (; i < endExclusive; i++) {",
      `    output[outputBase + i] = ${binaryExprForDType(op, dtype, "left[leftBase + i]", "right[rightBase + i]")};`,
      "}",
    );
    return lines.join("\n");
  }
  const vectorClass = dtype.id === "F32" ? "FloatVector" : "DoubleVector";
  lines.push(
    "int inputBase = inputView.storageOffset();",
    "int outputBase = outputView.storageOffset();",
    `int upper = startInclusive + ${dtype.vectorSpecies}.loopBound(endExclusive - startInclusive);`,
    "int i = startInclusive;",
  );
  lines.push(...unaryVectorSetup(op, dtype, vectorClass));
  lines.push(
    `for (; i < upper; i += ${dtype.vectorSpecies}.length()) {`,
    `    ${unaryVectorExpr(op, dtype, `${vectorClass}.fromArray(${dtype.vectorSpecies}, input, inputBase + i)`)}`,
    "            .intoArray(output, outputBase + i);",
    "}",
    "for (; i < endExclusive; i++) {",
    `    ${scalarValueType(dtype)} value = input[inputBase + i];`,
    `    output[outputBase + i] = ${unaryExpr(op, dtype, "value")};`,
    "}",
  );
  return lines.join("\n");
}

function bf16BroadcastInnerVector(op, storage) {
  const dtype = DTYPES[0];
  const lines = storage === "ARRAY" ? arrayDeclarations(op, dtype) : segmentDeclarations(op);
  const read = (name, offset) => storage === "ARRAY"
    ? `TensorDTypeOps.fromBFloat16Bits(${name}[${offset}])`
    : `TensorDTypeOps.fromBFloat16Bits(${name}.get(JAVA_SHORT, (long) (${offset}) * Short.BYTES))`;
  const write = (name, offset, value) => storage === "ARRAY"
    ? `${name}[${offset}] = TensorDTypeOps.toBFloat16Bits(${value});`
    : `${name}.set(JAVA_SHORT, (long) (${offset}) * Short.BYTES, TensorDTypeOps.toBFloat16Bits(${value}));`;
  const offset = (mode, base, linear, inner) => `${mode} == BROADCAST_SCALAR ? ${base} : (${mode} == BROADCAST_INNER ? ${base} + ${inner} : ${base} + ${linear})`;
  if (op.arity === "binary") {
    lines.push(
      "int leftBase = leftView.storageOffset();",
      "int rightBase = rightView.storageOffset();",
      "int outputBase = outputView.storageOffset();",
      "int leftMode = broadcastVectorMode(leftView);",
      "int rightMode = broadcastVectorMode(rightView);",
      "int innerSize = broadcastVectorInnerSize(outputView);",
      "float[] leftChunk = new float[F32.length()];",
      "float[] rightChunk = new float[F32.length()];",
      "float[] outputChunk = new float[F32.length()];",
      "int i = startInclusive;",
      "while (i < endExclusive) {",
      "    int rowInner = i % innerSize;",
      "    int rowEnd = Math.min(endExclusive, i + innerSize - rowInner);",
      "    int upper = i + F32.loopBound(rowEnd - i);",
      "    for (; i < upper; i += F32.length()) {",
      "        int vectorInner = i % innerSize;",
      "        for (int lane = 0; lane < F32.length(); lane++) {",
      `            leftChunk[lane] = ${read("left", offset("leftMode", "leftBase", "i + lane", "vectorInner + lane"))};`,
      `            rightChunk[lane] = ${read("right", offset("rightMode", "rightBase", "i + lane", "vectorInner + lane"))};`,
      "        }",
      "        FloatVector.fromArray(F32, leftChunk, 0)",
      `                .${op.vectorMethod}(FloatVector.fromArray(F32, rightChunk, 0))`,
      "                .intoArray(outputChunk, 0);",
      "        for (int lane = 0; lane < F32.length(); lane++) {",
      `            ${write("output", "outputBase + i + lane", "outputChunk[lane]")}`,
      "        }",
      "    }",
      "    for (; i < rowEnd; i++) {",
      "        int scalarInner = i % innerSize;",
      `        float result = ${binaryExprForDType(op, dtype, read("left", offset("leftMode", "leftBase", "i", "scalarInner")), read("right", offset("rightMode", "rightBase", "i", "scalarInner")))};`,
      `        ${write("output", "outputBase + i", "result")}`,
      "    }",
      "}",
    );
    return lines.join("\n");
  }
  lines.push(
    "int inputBase = inputView.storageOffset();",
    "int outputBase = outputView.storageOffset();",
    "int inputMode = broadcastVectorMode(inputView);",
    "int innerSize = broadcastVectorInnerSize(outputView);",
    "float[] inputChunk = new float[F32.length()];",
    "float[] outputChunk = new float[F32.length()];",
    "int i = startInclusive;",
  );
  lines.push(...unaryVectorSetup(op, dtype, "FloatVector"));
  lines.push(
    "while (i < endExclusive) {",
    "    int rowInner = i % innerSize;",
    "    int rowEnd = Math.min(endExclusive, i + innerSize - rowInner);",
    "    int upper = i + F32.loopBound(rowEnd - i);",
    "    for (; i < upper; i += F32.length()) {",
    "        int vectorInner = i % innerSize;",
    "        for (int lane = 0; lane < F32.length(); lane++) {",
    `            inputChunk[lane] = ${read("input", offset("inputMode", "inputBase", "i + lane", "vectorInner + lane"))};`,
    "        }",
    `        ${unaryVectorExpr(op, dtype, "FloatVector.fromArray(F32, inputChunk, 0)")}`,
    "                .intoArray(outputChunk, 0);",
    "        for (int lane = 0; lane < F32.length(); lane++) {",
    `            ${write("output", "outputBase + i + lane", "outputChunk[lane]")}`,
    "        }",
    "    }",
    "    for (; i < rowEnd; i++) {",
    "        int scalarInner = i % innerSize;",
    `        float value = ${read("input", offset("inputMode", "inputBase", "i", "scalarInner"))};`,
    `        ${write("output", "outputBase + i", unaryExpr(op, dtype, "value"))}`,
    "    }",
    "}",
  );
  return lines.join("\n");
}

function bf16Vector(op, storage) {
  const dtype = DTYPES[0];
  const lines = storage === "ARRAY" ? arrayDeclarations(op, dtype) : segmentDeclarations(op);
  const read = (name, offset) => storage === "ARRAY"
    ? `TensorDTypeOps.fromBFloat16Bits(${name}[${offset}])`
    : `TensorDTypeOps.fromBFloat16Bits(${name}.get(JAVA_SHORT, (long) (${offset}) * Short.BYTES))`;
  const write = (name, offset, value) => storage === "ARRAY"
    ? `${name}[${offset}] = TensorDTypeOps.toBFloat16Bits(${value});`
    : `${name}.set(JAVA_SHORT, (long) (${offset}) * Short.BYTES, TensorDTypeOps.toBFloat16Bits(${value}));`;
  if (op.arity === "binary") {
    lines.push(
      "int leftBase = leftView.storageOffset();",
      "int rightBase = rightView.storageOffset();",
      "int outputBase = outputView.storageOffset();",
      "float[] leftChunk = new float[F32.length()];",
      "float[] rightChunk = new float[F32.length()];",
      "float[] outputChunk = new float[F32.length()];",
      "int upper = startInclusive + F32.loopBound(endExclusive - startInclusive);",
      "int i = startInclusive;",
      "for (; i < upper; i += F32.length()) {",
      "    for (int lane = 0; lane < F32.length(); lane++) {",
      `        leftChunk[lane] = ${read("left", "leftBase + i + lane")};`,
      `        rightChunk[lane] = ${read("right", "rightBase + i + lane")};`,
      "    }",
      "    FloatVector.fromArray(F32, leftChunk, 0)",
      `            .${op.vectorMethod}(FloatVector.fromArray(F32, rightChunk, 0))`,
      "            .intoArray(outputChunk, 0);",
      "    for (int lane = 0; lane < F32.length(); lane++) {",
      `        ${write("output", "outputBase + i + lane", "outputChunk[lane]")}`,
      "    }",
      "}",
      "for (; i < endExclusive; i++) {",
      `    float result = ${binaryExprForDType(op, dtype, read("left", "leftBase + i"), read("right", "rightBase + i"))};`,
      `    ${write("output", "outputBase + i", "result")}`,
      "}",
    );
    return lines.join("\n");
  }
  lines.push(
    "int inputBase = inputView.storageOffset();",
    "int outputBase = outputView.storageOffset();",
    "float[] inputChunk = new float[F32.length()];",
    "float[] outputChunk = new float[F32.length()];",
    "int upper = startInclusive + F32.loopBound(endExclusive - startInclusive);",
    "int i = startInclusive;",
  );
  lines.push(...unaryVectorSetup(op, dtype, "FloatVector"));
  lines.push(
    "for (; i < upper; i += F32.length()) {",
    "    for (int lane = 0; lane < F32.length(); lane++) {",
    `        inputChunk[lane] = ${read("input", "inputBase + i + lane")};`,
    "    }",
    `    ${unaryVectorExpr(op, dtype, "FloatVector.fromArray(F32, inputChunk, 0)")}`,
    "            .intoArray(outputChunk, 0);",
    "    for (int lane = 0; lane < F32.length(); lane++) {",
    `        ${write("output", "outputBase + i + lane", "outputChunk[lane]")}`,
    "    }",
    "}",
    "for (; i < endExclusive; i++) {",
    `    float value = ${read("input", "inputBase + i")};`,
    `    ${write("output", "outputBase + i", unaryExpr(op, dtype, "value"))}`,
    "}",
  );
  return lines.join("\n");
}

function segmentBroadcastInnerVector(op, dtype) {
  if (dtype.id === "BF16") {
    return bf16BroadcastInnerVector(op, "SEGMENT");
  }
  const lines = segmentDeclarations(op);
  const klass = vectorClass(dtype);
  if (op.arity === "binary") {
    const leftVector = segmentBroadcastVectorLoad(dtype, "left", "leftBase", "leftMode", "i", "vectorInner");
    const rightVector = segmentBroadcastVectorLoad(dtype, "right", "rightBase", "rightMode", "i", "vectorInner");
    const leftScalar = segmentBroadcastScalarRead(dtype, "left", "leftBase", "leftMode", "i", "scalarInner");
    const rightScalar = segmentBroadcastScalarRead(dtype, "right", "rightBase", "rightMode", "i", "scalarInner");
    lines.push(
      `long leftBase = (long) leftView.storageOffset() * ${dtype.bytes};`,
      `long rightBase = (long) rightView.storageOffset() * ${dtype.bytes};`,
      `long outputBase = (long) outputView.storageOffset() * ${dtype.bytes};`,
      "int leftMode = broadcastVectorMode(leftView);",
      "int rightMode = broadcastVectorMode(rightView);",
      "int innerSize = broadcastVectorInnerSize(outputView);",
      "int i = startInclusive;",
      "while (i < endExclusive) {",
      "    int rowInner = i % innerSize;",
      "    int rowEnd = Math.min(endExclusive, i + innerSize - rowInner);",
      `    int upper = i + ${dtype.vectorSpecies}.loopBound(rowEnd - i);`,
      `    for (; i < upper; i += ${dtype.vectorSpecies}.length()) {`,
      "        int vectorInner = i % innerSize;",
      `        long outputOffset = outputBase + (long) i * ${dtype.bytes};`,
      `        ${leftVector}`,
      `                .${op.vectorMethod}(${rightVector})`,
      "                .intoMemorySegment(output, outputOffset, ORDER);",
      "    }",
      "    for (; i < rowEnd; i++) {",
      "        int scalarInner = i % innerSize;",
      `        long outputOffset = outputBase + (long) i * ${dtype.bytes};`,
      `        output.set(${dtype.layout}, outputOffset, ${binaryExprForDType(op, dtype, leftScalar, rightScalar)});`,
      "    }",
      "}",
    );
    return lines.join("\n");
  }
  const inputVector = segmentBroadcastVectorLoad(dtype, "input", "inputBase", "inputMode", "i", "vectorInner");
  const inputScalar = segmentBroadcastScalarRead(dtype, "input", "inputBase", "inputMode", "i", "scalarInner");
  lines.push(
    `long inputBase = (long) inputView.storageOffset() * ${dtype.bytes};`,
    `long outputBase = (long) outputView.storageOffset() * ${dtype.bytes};`,
    "int inputMode = broadcastVectorMode(inputView);",
    "int innerSize = broadcastVectorInnerSize(outputView);",
    "int i = startInclusive;",
  );
  lines.push(...unaryVectorSetup(op, dtype, klass));
  lines.push(
    "while (i < endExclusive) {",
    "    int rowInner = i % innerSize;",
    "    int rowEnd = Math.min(endExclusive, i + innerSize - rowInner);",
    `    int upper = i + ${dtype.vectorSpecies}.loopBound(rowEnd - i);`,
    `    for (; i < upper; i += ${dtype.vectorSpecies}.length()) {`,
    "        int vectorInner = i % innerSize;",
    `        long outputOffset = outputBase + (long) i * ${dtype.bytes};`,
    `        ${unaryVectorExpr(op, dtype, inputVector)}`,
    "                .intoMemorySegment(output, outputOffset, ORDER);",
    "    }",
    "    for (; i < rowEnd; i++) {",
    "        int scalarInner = i % innerSize;",
    `        long outputOffset = outputBase + (long) i * ${dtype.bytes};`,
    `        ${scalarValueType(dtype)} value = ${inputScalar};`,
    `        output.set(${dtype.layout}, outputOffset, ${unaryExpr(op, dtype, "value")});`,
    "    }",
    "}",
  );
  return lines.join("\n");
}

function segmentVector(op, dtype) {
  if (dtype.id === "BF16") {
    return bf16Vector(op, "SEGMENT");
  }
  const lines = segmentDeclarations(op);
  const vectorClass = dtype.id === "F32" ? "FloatVector" : "DoubleVector";
  if (op.arity === "binary") {
    lines.push(
      `long leftBase = (long) leftView.storageOffset() * ${dtype.bytes};`,
      `long rightBase = (long) rightView.storageOffset() * ${dtype.bytes};`,
      `long outputBase = (long) outputView.storageOffset() * ${dtype.bytes};`,
      `int upper = startInclusive + ${dtype.vectorSpecies}.loopBound(endExclusive - startInclusive);`,
      "int i = startInclusive;",
      `for (; i < upper; i += ${dtype.vectorSpecies}.length()) {`,
      `    long byteIndex = (long) i * ${dtype.bytes};`,
      `    ${vectorClass}.fromMemorySegment(${dtype.vectorSpecies}, left, leftBase + byteIndex, ORDER)`,
      `            .${op.vectorMethod}(${vectorClass}.fromMemorySegment(${dtype.vectorSpecies}, right, rightBase + byteIndex, ORDER))`,
      "            .intoMemorySegment(output, outputBase + byteIndex, ORDER);",
      "}",
      "for (; i < endExclusive; i++) {",
      `    long leftOffset = leftBase + (long) i * ${dtype.bytes};`,
      `    long rightOffset = rightBase + (long) i * ${dtype.bytes};`,
      `    long outputOffset = outputBase + (long) i * ${dtype.bytes};`,
      `    output.set(${dtype.layout}, outputOffset, ${binaryExprForDType(op, dtype, `left.get(${dtype.layout}, leftOffset)`, `right.get(${dtype.layout}, rightOffset)`)});`,
      "}",
    );
    return lines.join("\n");
  }
  lines.push(
    `long inputBase = (long) inputView.storageOffset() * ${dtype.bytes};`,
    `long outputBase = (long) outputView.storageOffset() * ${dtype.bytes};`,
    `int upper = startInclusive + ${dtype.vectorSpecies}.loopBound(endExclusive - startInclusive);`,
    "int i = startInclusive;",
  );
  lines.push(...unaryVectorSetup(op, dtype, vectorClass));
  lines.push(
    `for (; i < upper; i += ${dtype.vectorSpecies}.length()) {`,
    `    long byteIndex = (long) i * ${dtype.bytes};`,
    `    ${unaryVectorExpr(op, dtype, `${vectorClass}.fromMemorySegment(${dtype.vectorSpecies}, input, inputBase + byteIndex, ORDER)`)}`,
    "            .intoMemorySegment(output, outputBase + byteIndex, ORDER);",
    "}",
    "for (; i < endExclusive; i++) {",
    `    long inputOffset = inputBase + (long) i * ${dtype.bytes};`,
    `    long outputOffset = outputBase + (long) i * ${dtype.bytes};`,
    `    ${scalarValueType(dtype)} value = input.get(${dtype.layout}, inputOffset);`,
    `    output.set(${dtype.layout}, outputOffset, ${unaryExpr(op, dtype, "value")});`,
    "}",
  );
  return lines.join("\n");
}

function genericScalar(op, dtype, storage) {
  const outDType = outputDType(op, dtype);
  const lines = [];
  if (op.arity === "ternary") {
    if (storage === "ARRAY") {
      lines.push(
        "byte[] condition = args.input(0).boolArray();",
        `${op.trueDtype.arrayType}[] trueValues = args.input(1).${op.trueDtype.arrayAccessor}();`,
        `${op.falseDtype.arrayType}[] falseValues = args.input(2).${op.falseDtype.arrayAccessor}();`,
        `${outDType.arrayType}[] output = args.output().${outDType.arrayAccessor}();`,
      );
    } else {
      lines.push(
        "MemorySegment condition = args.input(0).segment();",
        "MemorySegment trueValues = args.input(1).segment();",
        "MemorySegment falseValues = args.input(2).segment();",
        "MemorySegment output = args.output().segment();",
      );
    }
    const condition = storage === "ARRAY"
      ? readArray(BOOL_DTYPE, "condition", "conditionOffset.offset(i)")
      : readSegment(BOOL_DTYPE, "condition", "conditionOffset.offset(i)");
    const trueValue = storage === "ARRAY"
      ? readArray(op.trueDtype, "trueValues", "trueOffset.offset(i)")
      : readSegment(op.trueDtype, "trueValues", "trueOffset.offset(i)");
    const falseValue = storage === "ARRAY"
      ? readArray(op.falseDtype, "falseValues", "falseOffset.offset(i)")
      : readSegment(op.falseDtype, "falseValues", "falseOffset.offset(i)");
    lines.push(
      "Cpu1GenericOffsetPlan conditionOffset = args.inputGenericOffsetPlan(0);",
      "Cpu1GenericOffsetPlan trueOffset = args.inputGenericOffsetPlan(1);",
      "Cpu1GenericOffsetPlan falseOffset = args.inputGenericOffsetPlan(2);",
      "Cpu1GenericOffsetPlan outputOffset = args.outputGenericOffsetPlan();",
      "for (int i = startInclusive; i < endExclusive; i++) {",
      `    ${scalarValueType(outDType)} selected = ${condition} ? ${trueValue} : ${falseValue};`,
      storage === "ARRAY"
        ? `    ${writeArray(outDType, "output", "outputOffset.offset(i)", "selected")}`
        : `    ${writeSegment(outDType, "output", "outputOffset.offset(i)", "selected")}`,
      "}",
    );
    return lines.join("\n");
  }
  if (op.arity === "binary") {
    if (storage === "ARRAY") {
      lines.push(
        `${dtype.arrayType}[] left = args.input(0).${dtype.arrayAccessor}();`,
        `${dtype.arrayType}[] right = args.input(1).${dtype.arrayAccessor}();`,
        `${outDType.arrayType}[] output = args.output().${outDType.arrayAccessor}();`,
      );
    } else {
      lines.push(
        "MemorySegment left = args.input(0).segment();",
        "MemorySegment right = args.input(1).segment();",
        "MemorySegment output = args.output().segment();",
      );
    }
    lines.push(
      "Cpu1GenericOffsetPlan leftOffset = args.inputGenericOffsetPlan(0);",
      "Cpu1GenericOffsetPlan rightOffset = args.inputGenericOffsetPlan(1);",
      "Cpu1GenericOffsetPlan outputOffset = args.outputGenericOffsetPlan();",
      "for (int i = startInclusive; i < endExclusive; i++) {",
    );
    const left = storage === "ARRAY" ? readArray(dtype, "left", "leftOffset.offset(i)") : readSegment(dtype, "left", "leftOffset.offset(i)");
    const right = storage === "ARRAY" ? readArray(dtype, "right", "rightOffset.offset(i)") : readSegment(dtype, "right", "rightOffset.offset(i)");
    const out = "outputOffset.offset(i)";
    const expr = binaryExprForDType(op, dtype, left, right);
    lines.push(storage === "ARRAY"
      ? `    ${writeArray(outDType, "output", out, expr)}`
      : `    ${writeSegment(dtype, "output", out, expr)}`);
    lines.push("}");
    return lines.join("\n");
  }
  if (storage === "ARRAY") {
    lines.push(
      `${dtype.arrayType}[] input = args.input(0).${dtype.arrayAccessor}();`,
      `${outDType.arrayType}[] output = args.output().${outDType.arrayAccessor}();`,
    );
  } else {
    lines.push(
      "MemorySegment input = args.input(0).segment();",
      "MemorySegment output = args.output().segment();",
    );
  }
  lines.push(
    "Cpu1GenericOffsetPlan inputOffset = args.inputGenericOffsetPlan(0);",
    "Cpu1GenericOffsetPlan outputOffset = args.outputGenericOffsetPlan();",
    "for (int i = startInclusive; i < endExclusive; i++) {",
  );
  const value = storage === "ARRAY" ? readArray(dtype, "input", "inputOffset.offset(i)") : readSegment(dtype, "input", "inputOffset.offset(i)");
  const expr = unaryExpr(op, dtype, "value");
  lines.push(`    ${scalarValueType(dtype)} value = ${value};`);
  lines.push(storage === "ARRAY"
    ? `    ${writeArray(outDType, "output", "outputOffset.offset(i)", expr)}`
    : `    ${writeSegment(dtype, "output", "outputOffset.offset(i)", expr)}`);
  lines.push("}");
  return lines.join("\n");
}

function rankScalar(op, dtype, storage, rank) {
  const outDType = outputDType(op, dtype);
  const lines = storage === "ARRAY" ? arrayDeclarations(op, dtype) : segmentDeclarations(op);
  if (rank >= 2) lines.push("int d1 = outputView.shape(1);");
  if (rank >= 3) lines.push("int d2 = outputView.shape(2);");
  if (rank >= 4) lines.push("int d3 = outputView.shape(3);");
  if (op.arity === "ternary") {
    const condition = storage === "ARRAY"
      ? readArray(BOOL_DTYPE, "condition", "conditionOffset")
      : readSegment(BOOL_DTYPE, "condition", "conditionOffset");
    const trueValue = storage === "ARRAY"
      ? readArray(op.trueDtype, "trueValues", "trueOffset")
      : readSegment(op.trueDtype, "trueValues", "trueOffset");
    const falseValue = storage === "ARRAY"
      ? readArray(op.falseDtype, "falseValues", "falseOffset")
      : readSegment(op.falseDtype, "falseValues", "falseOffset");
    lines.push(
      "int conditionBase = conditionView.storageOffset();",
      "int trueBase = trueView.storageOffset();",
      "int falseBase = falseView.storageOffset();",
      "int outputBase = outputView.storageOffset();",
      "for (int i = startInclusive; i < endExclusive; i++) {",
    );
    lines.push(...rankIndexLines(rank));
    lines.push(
      `    int conditionOffset = conditionBase + ${rankOffset("conditionView", rank)};`,
      `    int trueOffset = trueBase + ${rankOffset("trueView", rank)};`,
      `    int falseOffset = falseBase + ${rankOffset("falseView", rank)};`,
      `    int outputOffset = outputBase + ${rankOffset("outputView", rank)};`,
      `    ${scalarValueType(outDType)} selected = ${condition} ? ${trueValue} : ${falseValue};`,
      storage === "ARRAY"
        ? `    ${writeArray(outDType, "output", "outputOffset", "selected")}`
        : `    ${writeSegment(outDType, "output", "outputOffset", "selected")}`,
      "}",
    );
    return lines.join("\n");
  }
  if (op.arity === "binary") {
    lines.push(
      "int leftBase = leftView.storageOffset();",
      "int rightBase = rightView.storageOffset();",
      "int outputBase = outputView.storageOffset();",
      "for (int i = startInclusive; i < endExclusive; i++) {",
    );
    lines.push(...rankIndexLines(rank));
    lines.push(
      `    int leftOffset = leftBase + ${rankOffset("leftView", rank)};`,
      `    int rightOffset = rightBase + ${rankOffset("rightView", rank)};`,
      `    int outputOffset = outputBase + ${rankOffset("outputView", rank)};`,
    );
    const left = storage === "ARRAY" ? readArray(dtype, "left", "leftOffset") : readSegment(dtype, "left", "leftOffset");
    const right = storage === "ARRAY" ? readArray(dtype, "right", "rightOffset") : readSegment(dtype, "right", "rightOffset");
    const expr = binaryExprForDType(op, dtype, left, right);
    lines.push(storage === "ARRAY"
      ? `    ${writeArray(outDType, "output", "outputOffset", expr)}`
      : `    ${writeSegment(dtype, "output", "outputOffset", expr)}`);
    lines.push("}");
    return lines.join("\n");
  }
  lines.push(
    "int inputBase = inputView.storageOffset();",
    "int outputBase = outputView.storageOffset();",
    "for (int i = startInclusive; i < endExclusive; i++) {",
  );
  lines.push(...rankIndexLines(rank));
  lines.push(
    `    int inputOffset = inputBase + ${rankOffset("inputView", rank)};`,
    `    int outputOffset = outputBase + ${rankOffset("outputView", rank)};`,
  );
  const value = storage === "ARRAY" ? readArray(dtype, "input", "inputOffset") : readSegment(dtype, "input", "inputOffset");
  const expr = unaryExpr(op, dtype, "value");
  lines.push(`    ${scalarValueType(dtype)} value = ${value};`);
  lines.push(storage === "ARRAY"
    ? `    ${writeArray(outDType, "output", "outputOffset", expr)}`
    : `    ${writeSegment(dtype, "output", "outputOffset", expr)}`);
  lines.push("}");
  return lines.join("\n");
}

function rankIndexLines(rank) {
  if (rank === 2) {
    return [
      "    int c0 = i / d1;",
      "    int c1 = i - c0 * d1;",
    ];
  }
  if (rank === 3) {
    return [
      "    int c2 = i % d2;",
      "    int rem = i / d2;",
      "    int c1 = rem % d1;",
      "    int c0 = rem / d1;",
    ];
  }
  return [
    "    int c3 = i % d3;",
    "    int rem = i / d3;",
    "    int c2 = rem % d2;",
    "    rem /= d2;",
    "    int c1 = rem % d1;",
    "    int c0 = rem / d1;",
  ];
}

function rankOffset(view, rank) {
  return Array.from({ length: rank }, (_, i) => `c${i} * ${view}.stride(${i})`).join(" + ");
}

function methodBody(entry) {
  const { op, dtype, storage, layout, vectorKind } = entry;
  if (vectorKind === "VECTOR") {
    if (layout.id === "BROADCAST_INNER") {
      return storage === "ARRAY" ? arrayBroadcastInnerVector(op, dtype) : segmentBroadcastInnerVector(op, dtype);
    }
    return storage === "ARRAY" ? arrayVector(op, dtype) : segmentVector(op, dtype);
  }
  if (layout.id === "CONTIGUOUS") return contiguousScalar(op, dtype, storage);
  if (layout.id === "STRIDED_GENERIC") return genericScalar(op, dtype, storage);
  return rankScalar(op, dtype, storage, layout.rank);
}

function loopClasses() {
  const unique = new Map();
  for (const op of OPS) {
    const key = `${op.packageName}.${op.className}`;
    if (!unique.has(key)) {
      unique.set(key, {
        packageName: op.packageName,
        dir: op.dir,
        className: op.className,
        ids: [],
      });
    }
    unique.get(key).ids.push(op.id);
  }
  return [...unique.values()];
}

function loopFile(loopClass) {
  const opEntries = entries().filter(entry =>
    entry.op.packageName === loopClass.packageName && entry.op.className === loopClass.className
  );
  const methods = opEntries.map(entry => {
    const method = methodNameFor(entry.op, entry.dtype, entry.storage, entry.layout, entry.vectorKind);
    return `    public static void ${method}(Cpu1KernelArgs args, int startInclusive, int endExclusive) {\n${indent(methodBody(entry), 8)}\n    }`;
  }).join("\n\n");
  return generatedHeader() + `package ${loopClass.packageName};

import backend.cpu1.exec.Cpu1KernelArgs;
import backend.cpu1.exec.Cpu1TensorView;
import backend.cpu1.offset.Cpu1GenericOffsetPlan;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import tensor.dtype.TensorDTypeOps;
import utils.FastTranscendentals;
import utils.SpecialFunctions;

import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

/**
 * Static ${loopClass.ids.join("/").toLowerCase()} loops selected by Cpu1KernelId at prepare time.
 */
public final class ${loopClass.className} {
    private static final VectorSpecies<Float> F32 = FloatVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Double> F64 = DoubleVector.SPECIES_PREFERRED;
    private static final ByteOrder ORDER = ByteOrder.nativeOrder();
    private static final int BROADCAST_CONTIGUOUS = 0;
    private static final int BROADCAST_SCALAR = 1;
    private static final int BROADCAST_INNER = 2;

    private ${loopClass.className}() {
    }

    private static int broadcastVectorMode(Cpu1TensorView view) {
        if (isBroadcastScalar(view)) {
            return BROADCAST_SCALAR;
        }
        if (isBroadcastInner(view)) {
            return BROADCAST_INNER;
        }
        return BROADCAST_CONTIGUOUS;
    }

    private static int broadcastVectorInnerSize(Cpu1TensorView outputView) {
        int rank = outputView.rank();
        return rank == 0 ? 1 : outputView.shape(rank - 1);
    }

    private static boolean isBroadcastScalar(Cpu1TensorView view) {
        for (int dim = 0; dim < view.rank(); dim++) {
            if (view.stride(dim) != 0) {
                return false;
            }
        }
        return true;
    }

    private static boolean isBroadcastInner(Cpu1TensorView view) {
        int rank = view.rank();
        if (rank == 0 || view.contiguous()) {
            return false;
        }
        for (int dim = 0; dim < rank - 1; dim++) {
            if (view.stride(dim) != 0) {
                return false;
            }
        }
        return view.stride(rank - 1) == 1;
    }

${methods}
}
`;
}

function kernelIdFile() {
  const constants = entries().map((entry, index, all) => {
    const comma = index === all.length - 1 ? ";" : ",";
    const inputDataTypes = entry.inputDTypes
      .map(dtype => `DataType.${dtype.dataType}`)
      .join(", ");
    return `    ${kernelId(entry)}(
            Operation.OpType.${entry.op.id},
            DataType.${entry.dtype.dataType},
            List.of(${inputDataTypes}),
            Cpu1LayoutKind.${entry.layout.id},
            Cpu1StorageKind.${storageKind(entry.storage)},
            Cpu1VectorizationKind.${entry.vectorKind}
    )${comma}`;
  }).join("\n");
  return generatedHeader() + `package backend.cpu1.kernels;

import backend.cpu1.storage.Cpu1StorageKind;
import operations.Operation;
import tensor.DataType;

import java.util.List;

/**
 * Prepare-time identifier for a concrete cpu1 kernel loop.
 */
public enum Cpu1KernelId {
${constants}

    private final Operation.OpType opType;
    private final DataType dataType;
    private final List<DataType> inputDataTypes;
    private final Cpu1LayoutKind layoutKind;
    private final Cpu1StorageKind storageKind;
    private final Cpu1VectorizationKind vectorizationKind;

    Cpu1KernelId(
            Operation.OpType opType,
            DataType dataType,
            List<DataType> inputDataTypes,
            Cpu1LayoutKind layoutKind,
            Cpu1StorageKind storageKind,
            Cpu1VectorizationKind vectorizationKind
    ) {
        this.opType = opType;
        this.dataType = dataType;
        this.inputDataTypes = inputDataTypes;
        this.layoutKind = layoutKind;
        this.storageKind = storageKind;
        this.vectorizationKind = vectorizationKind;
    }

    public Cpu1KernelKey key() {
        return Cpu1KernelKey.of(opType, dataType, inputDataTypes, layoutKind, storageKind, vectorizationKind);
    }
}
`;
}

function dispatchFile() {
  const cases = entries().map(entry => {
    return `            case ${kernelId(entry)} -> ${entry.op.className}::${methodNameFor(entry.op, entry.dtype, entry.storage, entry.layout, entry.vectorKind)};`;
  }).join("\n");
  const imports = [...new Set(OPS.map(op => `import ${op.packageName}.${op.className};`))]
    .sort()
    .join("\n");
  return generatedHeader() + `package backend.cpu1.kernels;

${imports}

import java.util.Objects;

/**
 * Resolves prepared kernel ids to concrete range runners outside the hot launch path.
 */
public final class Cpu1KernelDispatch {
    private Cpu1KernelDispatch() {
    }

    public static Cpu1KernelRangeRunner runnerFor(Cpu1KernelId kernelId) {
        Objects.requireNonNull(kernelId, "kernelId cannot be null");
        return switch (kernelId) {
${cases}
        };
    }
}
`;
}

function targets() {
  const result = [
    [path.join(KERNEL_ROOT, "Cpu1KernelId.java"), kernelIdFile()],
    [path.join(KERNEL_ROOT, "Cpu1KernelDispatch.java"), dispatchFile()],
  ];
  for (const loopClass of loopClasses()) {
    result.push([
      path.join(KERNEL_ROOT, loopClass.dir, `${loopClass.className}.java`),
      loopFile(loopClass),
    ]);
  }
  return result;
}

function main() {
  const mismatches = [];
  for (const [file, content] of targets()) {
    if (CHECK) {
      const current = fs.existsSync(file) ? fs.readFileSync(file, "utf8") : null;
      if (current !== content) mismatches.push(path.relative(ROOT, file));
    } else {
      fs.mkdirSync(path.dirname(file), { recursive: true });
      fs.writeFileSync(file, content);
    }
  }
  if (mismatches.length > 0) {
    console.error("cpu1 generated sources are out of date:");
    for (const file of mismatches) console.error(`  ${file}`);
    console.error("Run: node scripts/generate-cpu1-elementwise.js");
    process.exit(1);
  }
  if (CHECK) {
    console.log("cpu1 generated sources are up to date.");
  } else {
    console.log("Generated cpu1 elementwise sources.");
  }
}

main();
