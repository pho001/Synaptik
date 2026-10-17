#!/usr/bin/env python3
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

TOOLS = ("clang", "metal", "metallib", "metal-objdump", "metal-nm")
FLOATING_OPCODES = frozenset({
    "fneg", "fadd", "fsub", "fmul", "fdiv", "frem", "fcmp",
    "sitofp", "uitofp", "fptosi", "fptoui",
})
LLVM_OPCODES = frozenset({
    "ret", "br", "switch", "indirectbr", "invoke", "callbr", "resume",
    "catchswitch", "catchret", "cleanupret", "unreachable",
    "fneg", "add", "fadd", "sub", "fsub", "mul", "fmul", "udiv", "sdiv",
    "fdiv", "urem", "srem", "frem", "shl", "lshr", "ashr", "and", "or",
    "xor", "extractelement", "insertelement", "shufflevector", "extractvalue",
    "insertvalue", "alloca", "load", "store", "fence", "cmpxchg", "atomicrmw",
    "getelementptr", "trunc", "zext", "sext", "fptrunc", "fpext", "fptoui",
    "fptosi", "uitofp", "sitofp", "ptrtoint", "inttoptr", "bitcast",
    "addrspacecast", "icmp", "fcmp", "phi", "select", "freeze", "call",
    "va_arg", "landingpad", "catchpad", "cleanuppad",
})
FAST_MATH_FLAGS = frozenset({
    "fast", "nnan", "ninf", "nsz", "arcp", "contract", "afn", "reassoc",
})
NEGATIVE_EXPECTATIONS = {
    "audit_forbidden_fadd": "fadd",
    "audit_forbidden_fmul": "fmul",
    "audit_forbidden_fdiv": "fdiv",
    "audit_forbidden_fma": "fma",
}


@dataclass(frozen=True)
class LlvmInstruction:
    function: str
    line: int
    opcode: str
    callee: str | None
    fast_math_flags: tuple[str, ...]

    @property
    def forbidden_class(self) -> str | None:
        if self.opcode in FLOATING_OPCODES:
            return self.opcode
        if self.opcode == "call" and self.callee is not None:
            if self.callee.startswith("air.fma.") or self.callee.startswith("llvm.fma."):
                return "fma"
        return None


class ForbiddenAirInstruction(ValueError):
    def __init__(self, function: str, instructions: list[LlvmInstruction]):
        self.function = function
        self.instructions = instructions
        classes = ", ".join(
            instruction.forbidden_class or "unknown" for instruction in instructions
        )
        super().__init__(f"{function} contains forbidden AIR instructions: {classes}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str], environment: dict[str, str]) -> dict[str, object]:
    completed = subprocess.run(
        command,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "command": command,
        "exitCode": completed.returncode,
        "stdout": completed.stdout.replace("\r\n", "\n"),
        "stderr": completed.stderr.replace("\r\n", "\n"),
    }


def require_success(result: dict[str, object]) -> str:
    if result["exitCode"] != 0:
        raise RuntimeError(
            f"provider command failed: {result['command']}\n"
            f"{result['stdout']}{result['stderr']}"
        )
    return str(result["stdout"]).strip()


def provider_inventory(output: Path, developer_dir: Path) -> None:
    environment = dict(os.environ)
    environment["DEVELOPER_DIR"] = str(developer_dir)
    xcrun = "/usr/bin/xcrun"
    xcodebuild = str(developer_dir / "usr/bin/xcodebuild")
    bootstrap = run([xcodebuild, "-downloadComponent", "MetalToolchain"], environment)
    require_success(bootstrap)

    xcrun_version = run([xcrun, "--version"], environment)
    xcode_version = run([xcodebuild, "-version"], environment)
    sdk_path = run([xcrun, "--sdk", "macosx", "--show-sdk-path"], environment)
    sdk_version = run([xcrun, "--sdk", "macosx", "--show-sdk-version"], environment)
    sdk_build = run([xcrun, "--sdk", "macosx", "--show-sdk-build-version"], environment)
    toolchain_path = run(
        [xcrun, "--sdk", "macosx", "--show-toolchain-path"], environment
    )
    require_success(xcrun_version)
    require_success(xcode_version)
    inventory = []
    for name in TOOLS:
        found = run(
            [xcrun, "--no-cache", "--sdk", "macosx", "--find", name], environment
        )
        path = Path(require_success(found))
        if not path.is_file():
            raise RuntimeError(f"provider returned a non-file tool: {name}: {path}")
        version = run([str(path), "--version"], environment)
        require_success(version)
        inventory.append({
            "name": name,
            "path": str(path),
            "resolvedPath": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "version": {
                "stdout": version["stdout"],
                "stderr": version["stderr"],
            },
        })
    document = {
        "schema": "synaptik.metal.audit-provider.v1",
        "developerDir": str(developer_dir),
        "bootstrap": bootstrap,
        "xcrun": {
            "path": xcrun,
            "bytes": Path(xcrun).stat().st_size,
            "sha256": sha256(Path(xcrun)),
            "version": {
                "stdout": xcrun_version["stdout"],
                "stderr": xcrun_version["stderr"],
            },
        },
        "xcode": {
            "path": xcodebuild,
            "version": {
                "stdout": xcode_version["stdout"],
                "stderr": xcode_version["stderr"],
            },
        },
        "sdk": {
            "path": require_success(sdk_path),
            "version": require_success(sdk_version),
            "build": require_success(sdk_build),
            "toolchainPath": require_success(toolchain_path),
        },
        "tools": inventory,
    }
    output.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def callee_name(instruction: str) -> str | None:
    marker = instruction.find("@")
    if marker < 0:
        return None
    cursor = marker + 1
    if cursor < len(instruction) and instruction[cursor] == '"':
        end = cursor + 1
        while end < len(instruction):
            if instruction[end] == '"' and instruction[end - 1] != "\\":
                return instruction[cursor + 1:end]
            end += 1
        raise ValueError(f"unterminated quoted LLVM callee: {instruction}")
    end = cursor
    while end < len(instruction) and (
            instruction[end].isalnum() or instruction[end] in "-._$"):
        end += 1
    return instruction[cursor:end] or None


def parse_instruction(function: str, line_number: int, source: str) -> LlvmInstruction | None:
    text = source.split(";", 1)[0].strip()
    if not text or text.endswith(":") or text == "}":
        return None
    if " = " in text:
        assignment, remainder = text.split(" = ", 1)
        if not assignment.startswith("%"):
            raise ValueError(f"malformed LLVM assignment at line {line_number}: {text}")
        text = remainder
    tokens = text.split()
    if not tokens:
        return None
    while tokens and tokens[0] in {"tail", "musttail", "notail"}:
        tokens.pop(0)
    if not tokens:
        return None
    opcode = tokens[0]
    if opcode not in LLVM_OPCODES:
        if (opcode == "]" or opcode.startswith("%") or opcode.startswith("[")
                or (opcode.startswith("i") and opcode[1:].isdigit())):
            return None
        raise ValueError(f"unknown LLVM opcode at line {line_number}: {opcode}")
    flags = tuple(sorted(FAST_MATH_FLAGS.intersection(tokens[1:])))
    return LlvmInstruction(
        function=function,
        line=line_number,
        opcode=opcode,
        callee=callee_name(text) if opcode == "call" else None,
        fast_math_flags=flags,
    )


def parse_functions(text: str) -> dict[str, list[LlvmInstruction]]:
    functions: dict[str, list[LlvmInstruction]] = {}
    current: str | None = None
    for line_number, line in enumerate(text.replace("\r\n", "\n").splitlines(), 1):
        stripped = line.strip()
        if current is None:
            if not stripped.startswith("define "):
                continue
            marker = stripped.find("@")
            opening = stripped.find("(", marker + 1)
            if marker < 0 or opening < 0 or not stripped.endswith("{"):
                raise ValueError(f"malformed LLVM function definition at line {line_number}")
            symbol = stripped[marker + 1:opening]
            if symbol.startswith('"') and symbol.endswith('"'):
                symbol = symbol[1:-1]
            if not symbol or symbol in functions:
                raise ValueError(f"invalid or duplicate LLVM function: {symbol}")
            current = symbol
            functions[current] = []
            continue
        if stripped == "}":
            current = None
            continue
        instruction = parse_instruction(current, line_number, line)
        if instruction is not None:
            functions[current].append(instruction)
    if current is not None:
        raise ValueError(f"unterminated LLVM function: {current}")
    return functions


def instruction_record(instruction: LlvmInstruction) -> dict[str, object]:
    result: dict[str, object] = {
        "line": instruction.line,
        "opcode": instruction.opcode,
    }
    if instruction.callee is not None:
        result["callee"] = instruction.callee
    if instruction.fast_math_flags:
        result["fastMathFlags"] = list(instruction.fast_math_flags)
    forbidden = instruction.forbidden_class
    if forbidden is not None:
        result["forbiddenClass"] = forbidden
    return result


def function_ledger(
        symbol: str, instructions: list[LlvmInstruction]) -> dict[str, object]:
    opcode_counts: dict[str, int] = {}
    for instruction in instructions:
        opcode_counts[instruction.opcode] = opcode_counts.get(instruction.opcode, 0) + 1
    forbidden = [
        instruction_record(instruction)
        for instruction in instructions
        if instruction.forbidden_class is not None or instruction.fast_math_flags
    ]
    return {
        "symbol": symbol,
        "instructionCount": len(instructions),
        "opcodeCounts": dict(sorted(opcode_counts.items())),
        "forbidden": forbidden,
        "accepted": not forbidden,
    }


def assert_no_forbidden(symbol: str, instructions: list[LlvmInstruction]) -> None:
    forbidden = [
        instruction
        for instruction in instructions
        if instruction.forbidden_class is not None or instruction.fast_math_flags
    ]
    if forbidden:
        raise ForbiddenAirInstruction(symbol, forbidden)


def generated_ledger(ir: Path, output: Path, fixture_metadata: Path) -> None:
    metadata = json.loads(fixture_metadata.read_text(encoding="utf-8"))
    steps = [fixture["step"] for fixture in metadata["fixtures"]]
    if not steps or any(not isinstance(step, int) or step < 0 for step in steps):
        raise ValueError("generated fixture metadata has invalid step ordinals")
    symbols = [f"synaptik_pw_g1_s{step}" for step in steps]
    if len(symbols) != len(set(symbols)):
        raise ValueError("generated fixture metadata has duplicate symbols")
    functions = parse_functions(ir.read_text(encoding="utf-8"))
    ledgers = []
    for symbol in symbols:
        if symbol not in functions:
            raise ValueError(f"generated AIR function absent: {symbol}")
        assert_no_forbidden(symbol, functions[symbol])
        ledgers.append(function_ledger(symbol, functions[symbol]))
    output.write_text(
        json.dumps({
            "schema": "synaptik.metal.air-instruction-ledger.v1",
            "authority": "structured-llvm-instruction-parser",
            "forbiddenOpcodes": sorted(FLOATING_OPCODES),
            "forbiddenFmaCallees": ["air.fma.*", "llvm.fma.*"],
            "functions": ledgers,
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def negative_ledger(ir: Path, output: Path) -> None:
    functions = parse_functions(ir.read_text(encoding="utf-8"))
    results = []
    for symbol, expected_class in NEGATIVE_EXPECTATIONS.items():
        if symbol not in functions:
            raise ValueError(f"negative AIR function absent: {symbol}")
        rejected = False
        detected: list[LlvmInstruction] = []
        try:
            assert_no_forbidden(symbol, functions[symbol])
        except ForbiddenAirInstruction as error:
            rejected = True
            detected = error.instructions
        classes = [instruction.forbidden_class for instruction in detected]
        if not rejected or classes != [expected_class]:
            raise AssertionError(
                f"negative provider result drift for {symbol}: {classes}"
            )
        results.append({
            "symbol": symbol,
            "expectedClass": expected_class,
            "detected": [instruction_record(instruction) for instruction in detected],
            "rejected": True,
        })
    output.write_text(
        json.dumps({
            "schema": "synaptik.metal.air-negative-ledger.v1",
            "authority": "structured-llvm-instruction-parser",
            "results": results,
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def artifact_observation(output: Path, paths: list[Path]) -> None:
    if not paths:
        raise ValueError("artifact observation is empty")
    names = [path.name for path in paths]
    if len(names) != len(set(names)):
        raise ValueError("artifact observation has duplicate basenames")
    for path in paths:
        if not path.is_file():
            raise ValueError(f"artifact is absent: {path}")
    output.write_text(
        json.dumps({
            "schema": "synaptik.metal.audit-artifacts.v1",
            "artifacts": [
                {
                    "name": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for path in paths
            ],
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(
            "usage: audit-provider.py inventory|generated-ledger|negative-ledger ..."
        )
    command = sys.argv[1]
    if command == "inventory" and len(sys.argv) == 4:
        provider_inventory(Path(sys.argv[2]), Path(sys.argv[3]))
    elif command == "generated-ledger" and len(sys.argv) == 5:
        generated_ledger(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
    elif command == "negative-ledger" and len(sys.argv) == 4:
        negative_ledger(Path(sys.argv[2]), Path(sys.argv[3]))
    elif command == "artifacts" and len(sys.argv) >= 4:
        artifact_observation(Path(sys.argv[2]), list(map(Path, sys.argv[3:])))
    else:
        raise SystemExit(f"invalid audit-provider invocation: {sys.argv[1:]}")


if __name__ == "__main__":
    main()
