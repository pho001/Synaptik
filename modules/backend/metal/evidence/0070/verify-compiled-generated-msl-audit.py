#!/usr/bin/env python3
import hashlib
import json
import re
import sys
from pathlib import Path

EVIDENCE = Path(__file__).resolve().parent
MANIFEST = json.loads((EVIDENCE / "compiled-generated-msl-audit.json").read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()



def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    require(len(sys.argv) == 2, "usage: verify-compiled-generated-msl-audit.py WORK")
    work = Path(sys.argv[1]).resolve()
    require(MANIFEST["schema"] == "synaptik.metal.generated-msl-audit.v2",
            "compiled-audit schema drift")
    require(MANIFEST["developerDir"] == "/Applications/Xcode.app/Contents/Developer",
            "unapproved developer directory")
    require(
        MANIFEST["compilerFlags"]
        == ["-std=metal3.2", "-fno-fast-math", "-Wall", "-Wextra", "-Werror",
            "-isysroot", "<absolute SDKROOT>"],
        "compiler flag policy drift",
    )
    provider_reference_policy = MANIFEST["providerReference"]
    provider_reference = EVIDENCE / provider_reference_policy["path"]
    require(provider_reference.stat().st_size == provider_reference_policy["bytes"],
            "provider reference byte count drift")
    require(digest(provider_reference) == provider_reference_policy["sha256"],
            "provider reference SHA-256 drift")
    provider = json.loads((work / "provider-inventory.json").read_text(encoding="utf-8"))
    reference_provider = json.loads(provider_reference.read_text(encoding="utf-8"))
    require(provider == reference_provider, "Xcode audit-provider inventory drift")
    require(provider["schema"] == "synaptik.metal.audit-provider.v1",
            "audit-provider inventory schema drift")
    require(provider["developerDir"] == MANIFEST["developerDir"],
            "provider developer directory drift")
    require(
        provider["bootstrap"]["command"]
        == [MANIFEST["developerDir"] + "/usr/bin/xcodebuild",
            "-downloadComponent", "MetalToolchain"]
        and provider["bootstrap"]["exitCode"] == 0
        and provider["bootstrap"]["stderr"] == ""
        and provider["bootstrap"]["stdout"] != "",
        "Metal toolchain bootstrap result drift",
    )
    require(
        [tool["name"] for tool in provider["tools"]]
        == ["clang", "metal", "metallib", "metal-objdump", "metal-nm"],
        "available Xcode tool inventory drift",
    )
    for tool in provider["tools"]:
        require(
            set(tool) == {"name", "path", "resolvedPath", "bytes", "sha256", "version"}
            and tool["bytes"] > 0
            and len(tool["sha256"]) == 64
            and tool["version"]["stdout"] != "",
            f"incomplete provider identity: {tool['name']}",
        )
    expected_runtime_reflection = {
        "pipelineOptions": ["MTLPipelineOptionBindingInfo", "MTLPipelineOptionBufferTypeInfo"],
        "input": {
            "name": "input",
            "index": 0,
            "bindingAccess": "readOnly",
            "pointerAccess": "readOnly",
            "elementType": "uint",
            "alignment": 4,
            "bytes": 4,
        },
        "output": {
            "name": "output",
            "index": 1,
            "bindingAccess": "readWrite",
            "pointerAccess": "readWrite",
            "elementType": "uint",
            "alignment": 4,
            "bytes": 4,
            "airAccess": "writeOnly",
        },
        "meta": {
            "name": "meta",
            "index": 2,
            "bindingAccess": "readOnly",
            "pointerAccess": "readOnly",
            "dataType": "struct",
            "alignment": 8,
            "bytes": 32,
            "members": [
                {"name": "elementCount", "type": "ulong", "offset": 0},
                {"name": "gridWidth", "type": "ulong", "offset": 8},
                {"name": "gridHeight", "type": "ulong", "offset": 16},
                {"name": "scalar", "type": "uint", "offset": 24},
                {"name": "reserved", "type": "uint", "offset": 28},
            ],
        },
    }
    require(
        MANIFEST["runtimeReflection"] == expected_runtime_reflection,
        "runtime reflection evidence drift",
    )

    observation = json.loads(
        (work / "artifact-observation.json").read_text(encoding="utf-8")
    )
    require(observation["schema"] == "synaptik.metal.audit-artifacts.v1",
            "artifact observation schema drift")
    observed = {artifact["name"]: artifact for artifact in observation["artifacts"]}
    require(len(observed) == len(observation["artifacts"]),
            "duplicate artifact observations")
    require(set(observed) == set(MANIFEST["artifacts"]),
            "artifact observation set drift")
    for name, expected in MANIFEST["artifacts"].items():
        path = EVIDENCE / name if name == "forbidden-floating-fixtures.metal" else work / name
        require(path.is_file(), f"missing audit artifact: {name}")
        actual = {
            "name": name,
            "bytes": path.stat().st_size,
            "sha256": digest(path),
        }
        require(observed[name] == actual, f"actual artifact hash ledger drift: {name}")
        if expected.get("nondeterministicContainer"):
            require(
                set(expected)
                == {"referenceBytes", "referenceSha256", "nondeterministicContainer",
                    "actualHashLedger", "semanticAuthority"}
                and expected["referenceBytes"] > 0
                and len(expected["referenceSha256"]) == 64
                and expected["actualHashLedger"] == "artifact-observation.json"
                and expected["semanticAuthority"]
                    in {"generated-fixtures.air.instructions.json",
                        "forbidden-floating-fixtures.rejections.json"},
                f"unapproved compiled-container policy: {name}",
            )
            continue
        require(actual["bytes"] == expected["bytes"], f"byte count drift: {name}")
        require(actual["sha256"] == expected["sha256"], f"SHA-256 drift: {name}")

    metadata = json.loads((work / "fixtures.json").read_text(encoding="utf-8"))
    fixed_facts = MANIFEST["fixedCorpus"]
    require(metadata["fixedBytes"] == fixed_facts["bytes"], "fixed corpus byte count drift")
    require(metadata["fixedSha256"] == fixed_facts["sha256"], "fixed corpus metadata hash drift")
    require(metadata["components"] == fixed_facts["components"], "fixed component ledger drift")
    require(
        metadata["generatedBytes"] == 49 + sum(f["bytes"] for f in MANIFEST["fixtures"]),
        "generated byte formula drift",
    )
    require(
        metadata["totalBytes"] == metadata["fixedBytes"] + metadata["generatedBytes"],
        "total source byte formula drift",
    )
    expected_metadata = []
    for fixture in MANIFEST["fixtures"]:
        expected_metadata.append({
            "functionBytes": fixture["bytes"],
            "opcodes": [opcode.lower() for opcode in fixture["instructions"]],
            "siteKind": fixture["siteKind"],
            "step": fixture["step"],
        })
    require(metadata["fixtures"] == expected_metadata, "fixture instruction manifest drift")
    require(
        [fixture["instructions"] for fixture in MANIFEST["fixtures"][:4]]
        == [["FLOOR"], ["CEIL"], ["SIGN"], ["RELU"]],
        "singleton operation ledger drift",
    )
    require(
        [len(fixture["instructions"]) for fixture in MANIFEST["fixtures"][4:]]
        == list(range(2, 9)),
        "2..8 chain ledger drift",
    )

    source = (work / "generated-fixtures.metal").read_text(encoding="utf-8")
    fixed = source[:MANIFEST["fixedCorpus"]["bytes"]].encode("utf-8")
    require(hashlib.sha256(fixed).hexdigest() == MANIFEST["fixedCorpus"]["sha256"], "fixed corpus hash drift")
    require(source.count("kernel void synaptik_pw_g1_s") == len(MANIFEST["fixtures"]), "generated function count drift")
    require("#include <metal_stdlib>\nusing namespace metal;\n" in source, "generated source prefix absent")
    require(
        "// synaptik pointwise fusion generator schema 2\n" in source
        and "generator schema 1" not in source,
        "generated source schema marker drift",
    )
    generated = source[MANIFEST["fixedCorpus"]["bytes"]:]
    require(
        hashlib.sha256(generated.encode("utf-8")).hexdigest()
        == metadata["generatedSha256"],
        "generated source hash drift",
    )
    require(
        hashlib.sha256(source.encode("utf-8")).hexdigest()
        == metadata["assembledSha256"],
        "assembled source hash drift",
    )
    for fixture in MANIFEST["fixtures"]:
        symbol = fixture["name"]
        function = re.search(
            rf"kernel void {re.escape(symbol)}\(.*?\n\}}",
            generated,
            flags=re.DOTALL,
        )
        require(function is not None, f"generated source site absent: {symbol}")
        body = function.group(0)
        helpers = re.findall(r"= (floor_bits|ceil_bits|sign_bits|relu_bits)\(", body)
        require(
            helpers == [opcode.lower() + "_bits" for opcode in fixture["instructions"]],
            f"generated source opcode ledger drift: {symbol}",
        )

    instruction_ledger = json.loads(
        (work / "generated-fixtures.air.instructions.json").read_text(encoding="utf-8")
    )
    require(
        MANIFEST["semanticReproducibilityAuthority"]
        == "generated-fixtures.air.instructions.json"
        and instruction_ledger["schema"]
        == "synaptik.metal.air-instruction-ledger.v1"
        and instruction_ledger["authority"] == "structured-llvm-instruction-parser",
        "semantic AIR authority drift",
    )
    require(
        instruction_ledger["forbiddenFmaCallees"] == ["air.fma.*", "llvm.fma.*"],
        "FMA callee policy drift",
    )
    instruction_functions = {
        function["symbol"]: function for function in instruction_ledger["functions"]
    }
    require(
        len(instruction_functions) == len(instruction_ledger["functions"]),
        "duplicate structured AIR function ledger",
    )

    ir = (work / "generated-fixtures.air.ll.canonical").read_text(encoding="utf-8")
    nm = (work / "generated-fixtures.air.nm.canonical").read_text(encoding="utf-8")
    facts = MANIFEST["requiredAirFacts"]
    require(facts["pointMetaType"] in ir, "PointMeta AIR layout drift")
    require(set(instruction_functions) == set(facts["generatedSymbols"]),
            "structured AIR symbol ledger drift")
    for symbol in facts["generatedSymbols"]:
        nm_lines = [line for line in nm.splitlines() if line.endswith(" T " + symbol)]
        require(len(nm_lines) == 1, f"generated AIR symbol missing or duplicated: {symbol}")
        match = re.search(rf"^define void @{re.escape(symbol)}\((.*?)\n\}}", ir, flags=re.MULTILINE | re.DOTALL)
        require(match is not None, f"generated AIR body absent: {symbol}")
        body = match.group(0)
        signature = body.splitlines()[0]
        require(signature.count('"air-buffer-no-alias"') == facts["bufferCount"], f"buffer reflection count drift: {symbol}")
        require('i32 addrspace(1)* nocapture noundef readonly' in signature, f"input access drift: {symbol}")
        require('i32 addrspace(1)* nocapture noundef writeonly' in signature, f"output access drift: {symbol}")
        require(f'dereferenceable({facts["pointMetaBytes"]})' in signature and 'align 8' in signature, f"PointMeta binding drift: {symbol}")
        require(body.count("load i32, i32 addrspace(1)*") == 1, f"generated unit must load one boundary input: {symbol}")
        require(body.count("store i32") == 1, f"generated unit must store one boundary output: {symbol}")
        structured = instruction_functions[symbol]
        require(
            structured["accepted"] is True and structured["forbidden"] == []
            and structured["instructionCount"] == sum(structured["opcodeCounts"].values()),
            f"structured AIR instruction rejection: {symbol}",
        )
        require(structured["opcodeCounts"].get("store") == 1,
                f"structured AIR output-store ledger drift: {symbol}")
        require("call void @exact_" not in body,
                f"generated unit retained an unfused helper call: {symbol}")
    negative = json.loads(
        (work / "forbidden-floating-fixtures.rejections.json").read_text(
            encoding="utf-8"
        )
    )
    require(
        negative["schema"] == "synaptik.metal.air-negative-ledger.v1"
        and negative["authority"] == "structured-llvm-instruction-parser",
        "negative AIR ledger schema drift",
    )
    require(
        [
            {
                "symbol": result["symbol"],
                "forbiddenClass": result["expectedClass"],
            }
            for result in negative["results"]
        ]
        == MANIFEST["negativeFixtures"],
        "negative fixture expectation drift",
    )
    for result in negative["results"]:
        require(result["rejected"] is True and len(result["detected"]) == 1,
                f"negative fixture was not rejected: {result['symbol']}")
        detected = result["detected"][0]
        require(
            detected["forbiddenClass"] == result["expectedClass"],
            f"negative fixture class drift: {result['symbol']}",
        )
        if result["expectedClass"] == "fma":
            require(
                detected["opcode"] == "call"
                and (detected["callee"].startswith("air.fma.")
                     or detected["callee"].startswith("llvm.fma.")),
                "FMA negative fixture callee drift",
            )

    print("Task 0070 compiled generated-MSL audit passed")


if __name__ == "__main__":
    main()
