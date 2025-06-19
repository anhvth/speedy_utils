"""
generate_client.py

A script to generate a synchronous GeneratedClient from an OpenAPI JSON spec.
Usage:
    python generate_client.py openapi.json > generated_client.py
"""

# Renamed from gen.py to openapi_client_codegen.py

import argparse
import json
import sys
from typing import Any, Dict, List


def pascal_case(s: str) -> str:
    """Convert snake-case or camelCase to PascalCase"""
    parts = [p for p in s.replace("-", "_").split("_") if p]
    return "".join(p.title() for p in parts)


def snake_case(s: str) -> str:
    """Convert camelCase or PascalCase to snake_case"""
    out = []
    for c in s:
        if c.isupper():
            if out:
                out.append("_")
            out.append(c.lower())
        else:
            out.append(c)
    return "".join(out)


def map_openapi_type(prop: Dict[str, Any]) -> str:
    t = prop.get("type")
    if t == "string":
        fmt = prop.get("format")
        return "datetime" if fmt == "date-time" else "str"
    if t == "integer":
        return "int"
    if t == "boolean":
        return "bool"
    if t == "array":
        items = prop.get("items", {})
        return f"List[{map_openapi_type(items)}]"
    if t == "object":
        return "Dict[str, Any]"
    return "Any"


def generate_models(components: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    schemas = components.get("schemas", {})
    for name, schema in schemas.items():
        if "enum" in schema:
            lines.append(f"class {name}(str, Enum):")
            for val in schema["enum"]:
                key = snake_case(val).upper()
                lines.append(f"    {key} = '{val}'")
            lines.append("")
            continue
        lines.append(f"class {name}(BaseModel):")
        props = schema.get("properties", {})
        required = set(schema.get("required", []))
        if not props:
            lines.append("    pass")
        else:
            for prop_name, prop in props.items():
                t = map_openapi_type(prop)
                optional = prop_name not in required
                type_hint = f"Optional[{t}]" if optional else t
                default = " = None" if optional else ""
                lines.append(f"    {prop_name}: {type_hint}{default}")
        lines.append("")
    return lines


def generate_client(spec: Dict[str, Any]) -> List[str]:
    paths = spec.get("paths", {})
    models = spec.get("components", {}).get("schemas", {})
    lines: List[str] = []
    lines.append("class GeneratedClient:")
    lines.append('    """Client generated from OpenAPI spec."""')
    lines.append("")
    # __init__
    lines.append(
        "    def __init__(self, base_url: Optional[str]=None, api_key: Optional[str]=None, timeout: float=30.0, app: Any=None) -> None:"
    )
    lines.append('        """Initialize the generated client."""')
    lines.append("        from fastapi.testclient import TestClient")
    lines.append("        import httpx")
    lines.append("        from pydantic import BaseModel")
    lines.append('        headers = {"Content-Type": "application/json"}')
    lines.append("        if api_key:")
    lines.append('            headers["Authorization"] = f"Bearer {api_key}"')
    lines.append("")
    lines.append("        if app is not None:")
    lines.append('            self.base_url = "http://testserver"')
    lines.append("            self.client = TestClient(app)")
    lines.append("            self.client.headers.update(headers)")
    lines.append("        else:")
    lines.append(
        '            self.base_url = base_url.rstrip("/") if base_url else "http://localhost"'
    )
    lines.append(
        "            self.client = httpx.Client(base_url=self.base_url, headers=headers, timeout=timeout)"
    )
    lines.append("")

    for path, ops in paths.items():
        for method, op in ops.items():
            op_id = op.get("operationId", snake_case(method + path))
            func_name = snake_case(op_id)
            summary = op.get("summary", "").strip()
            # collect parameters
            req_params: List[str] = ["self"]
            opt_params: List[str] = []
            # path params (required)
            path_params = [p for p in op.get("parameters", []) if p.get("in") == "path"]
            for p in path_params:
                req_params.append(f"{p['name']}: str")
            # requestBody props
            body_props = None
            if "requestBody" in op:
                content = op["requestBody"].get("content", {})
                media = content.get("application/json") or next(iter(content.values()))
                schema_ref = media.get("schema", {})
                if "$ref" in schema_ref:
                    ref = schema_ref["$ref"].split("/")[-1]
                    schema = models.get(ref, {})
                    body_props = schema.get("properties", {})
                    required = set(schema.get("required", []))
                    for name_, prop in body_props.items():
                        t = map_openapi_type(prop)
                        if name_ in required:
                            req_params.append(f"{name_}: {t}")
                        else:
                            opt_params.append(f"{name_}: Optional[{t}] = None")
            # query params (always optional)
            query_params = [
                p for p in op.get("parameters", []) if p.get("in") == "query"
            ]
            for p in query_params:
                t = "str"
                opt_params.append(f"{p['name']}: Optional[{t}] = None")
            # combine signature
            sig = req_params + opt_params
            param_str = ", ".join(sig)
            # determine return type
            ret_type = "Any"
            resp200 = op.get("responses", {}).get("200", {})
            content200 = resp200.get("content", {})
            if content200:
                schema200 = next(iter(content200.values())).get("schema", {})
                if "$ref" in schema200:
                    ret_type = schema200["$ref"].split("/")[-1]
                elif "items" in schema200 and "$ref" in schema200["items"]:
                    inner = schema200["items"]["$ref"].split("/")[-1]
                    ret_type = f"List[{inner}]"
            # method definition
            lines.append(f"    def {func_name}({param_str}) -> {ret_type}:")
            if summary:
                lines.append(f'        """{summary}."""')
            # build URL
            url = f"f'{path}'" if "{" in path else f"'{path}'"
            m = method.lower()
            # build call
            if body_props and m in ("post", "put", "patch"):
                payload_items = []
                for name_ in body_props:
                    payload_items.append(f"'{name_}': {name_}")
                payload = ", ".join(payload_items)
                lines.append(
                    f"        resp = self.client.{m}({url}, json={{ {payload} }})"
                )
            elif query_params and m == "get":
                qpairs = ", ".join(f"'{p['name']}': {p['name']}" for p in query_params)
                lines.append(
                    f"        resp = self.client.get({url}, params={{ {qpairs} }})"
                )
            else:
                lines.append(f"        resp = self.client.{m}({url})")
            lines.append("        resp.raise_for_status()")
            # parse response
            if ret_type.startswith("List["):
                inner = ret_type[5:-1]
                lines.append(
                    f"        return [{inner}.model_validate(item) for item in resp.json()]"
                )
            elif ret_type in models:
                lines.append(f"        return {ret_type}.model_validate(resp.json())")
            else:
                lines.append("        return resp.json()")
            lines.append("")

    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a synchronous client from OpenAPI spec."
    )
    parser.add_argument(
        "spec",
        type=str,
        help="OpenAPI JSON file path or URL (e.g. openapi.json or http://localhost:8084/openapi.json)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output file path (default: stdout)",
    )

    args = parser.parse_args()

    try:
        spec_src = args.spec
        if spec_src.startswith("http://") or spec_src.startswith("https://"):
            import httpx

            response = httpx.get(spec_src)
            spec = response.json()
        else:
            with open(spec_src, "r", encoding="utf-8") as f:
                spec = json.load(f)
        out: List[str] = []
        # imports
        out.append("from typing import Any, Dict, List, Optional")
        out.append("from datetime import datetime")
        out.append("import httpx")
        out.append("from fastapi.testclient import TestClient")
        out.append("from pydantic import BaseModel")
        out.append("from enum import Enum")
        out.append("")
        # models
        out.extend(generate_models(spec.get("components", {})))
        # client
        out.extend(generate_client(spec))
        output_str = "\n".join(out)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output_str)
            print(f"\033[0;32mClient wrapper generated at {args.output}\033[0m")
        else:
            print(output_str)
    except Exception as e:
        print(
            f"\033[0;31mFailed to generate client wrapper: {e}\033[0m", file=sys.stderr
        )
        sys.exit(1)
