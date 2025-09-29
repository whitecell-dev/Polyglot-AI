#!/usr/bin/env python3
"""
AXIS-PY: Zero-boilerplate Python reasoning library
Replaces Pydantic, FastAPI, LangChain, and Mypy with YAML + rules + optional AI
Now includes SPC (Service Pipeline Configuration) runtime compatible with EDT microkernel
"""

import json
import os
import sys
import time
import hashlib
import re
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from collections import deque

# Optional imports
try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    yaml = None

try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    requests = None


def validate(data: dict, schema: Union[dict, str]) -> Tuple[bool, List[str]]:
    """
    Validate data against schema (dict or YAML path).

    Schema format:
        {"name": "str", "age": "int", "email": "str?"}

    Returns:
        (is_valid, list_of_errors)
    """
    # Load schema from YAML if path provided
    if isinstance(schema, str):
        if not HAS_YAML:
            return False, ["YAML support requires: pip install pyyaml"]
        try:
            with open(schema, "r") as f:
                schema = yaml.safe_load(f)
        except Exception as e:
            return False, [f"Failed to load schema: {e}"]

    errors = []

    # Validate each field
    for field, type_spec in schema.items():
        # Handle optional fields (ending with ?)
        is_optional = type_spec.endswith("?")
        if is_optional:
            type_spec = type_spec[:-1]

        # Check if field exists
        if field not in data:
            if not is_optional:
                errors.append(f"Missing required field: {field}")
            continue

        value = data[field]

        # Type checking and coercion
        if type_spec == "str":
            if not isinstance(value, str):
                try:
                    data[field] = str(value)
                except Exception:
                    errors.append(f"{field}: expected str, got {type(value).__name__}")

        elif type_spec == "int":
            if not isinstance(value, int):
                try:
                    data[field] = int(value)
                except Exception:
                    errors.append(f"{field}: expected int, got {type(value).__name__}")

        elif type_spec == "float":
            if not isinstance(value, (int, float)):
                try:
                    data[field] = float(value)
                except Exception:
                    errors.append(
                        f"{field}: expected float, got {type(value).__name__}"
                    )

        elif type_spec == "bool":
            if not isinstance(value, bool):
                # Coerce common bool representations
                if isinstance(value, str):
                    if value.lower() in ("true", "1", "yes", "on"):
                        data[field] = True
                    elif value.lower() in ("false", "0", "no", "off"):
                        data[field] = False
                    else:
                        errors.append(f"{field}: expected bool, got {value}")
                else:
                    errors.append(f"{field}: expected bool, got {type(value).__name__}")

        elif type_spec == "list":
            if not isinstance(value, list):
                errors.append(f"{field}: expected list, got {type(value).__name__}")

        elif type_spec == "dict":
            if not isinstance(value, dict):
                errors.append(f"{field}: expected dict, got {type(value).__name__}")

        else:
            errors.append(f"{field}: unknown type {type_spec}")

    return len(errors) == 0, errors


def typecheck(schema: dict):
    """
    Decorator that validates function arguments against schema.
    Raises TypeError on validation failure.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(data: dict, *args, **kwargs):
            is_valid, errors = validate(data, schema)
            if not is_valid:
                raise TypeError(f"Validation failed: {'; '.join(errors)}")
            return func(data, *args, **kwargs)

        return wrapper

    return decorator


def fallback(
    fn: Callable, llm: str = "gpt-4", prompt: str = "Fix this: {input}"
) -> Callable:
    """
    Wrap function with LLM fallback. If fn fails or returns None, call LLM.

    Note: Requires OPENAI_API_KEY environment variable.
    """

    @wraps(fn)
    def wrapper(input_data: Any) -> Any:
        # Try the original function
        try:
            result = fn(input_data)
            if result is not None:
                return result
        except Exception as e:
            # Log the error if debug mode
            if os.getenv("AXIS_DEBUG"):
                print(f"Function {fn.__name__} failed: {e}", file=sys.stderr)

        # Fallback to LLM
        if not HAS_REQUESTS:
            raise RuntimeError("LLM fallback requires: pip install requests")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "LLM fallback requires OPENAI_API_KEY environment variable"
            )

        # Format the prompt
        formatted_prompt = prompt.format(input=json.dumps(input_data))

        # Call OpenAI API
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": llm,
                    "messages": [{"role": "user", "content": formatted_prompt}],
                    "temperature": 0.7,
                },
            )
            response.raise_for_status()

            # Extract the response
            content = response.json()["choices"][0]["message"]["content"]

            # Try to parse as JSON if it looks like JSON
            if content.strip().startswith("{") or content.strip().startswith("["):
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    pass

            return content

        except Exception as e:
            raise RuntimeError(f"LLM fallback failed: {e}")

    return wrapper


class RuleEngine:
    """
    YAML-based rule engine with if/then/else logic.
    """

    def __init__(self, rules: Union[dict, str]):
        """
        Initialize with rules dict or YAML file path.
        """
        if isinstance(rules, str):
            if not HAS_YAML:
                raise RuntimeError("YAML support requires: pip install pyyaml")
            with open(rules, "r") as f:
                self.rules = yaml.safe_load(f)
        else:
            self.rules = rules

        # Normalize rules structure
        if "rules" in self.rules:
            self.rule_list = self.rules["rules"]
        elif isinstance(self.rules, list):
            self.rule_list = self.rules
        else:
            raise ValueError("Rules must be a list or dict with 'rules' key")

    def _evaluate_condition(self, condition: str, context: dict) -> bool:
        """
        Safely evaluate a condition string against context.
        """
        # Basic safety check
        dangerous = ["import", "exec", "eval", "__", "open", "file"]
        for d in dangerous:
            if d in condition:
                raise ValueError(f"Unsafe condition: {condition}")

        try:
            # Create safe evaluation context
            safe_context = dict(context)
            safe_context.update(
                {
                    "len": len,
                    "str": str,
                    "int": int,
                    "float": float,
                    "bool": bool,
                    "True": True,
                    "False": False,
                    "None": None,
                }
            )

            # Evaluate the condition
            return eval(condition, {"__builtins__": {}}, safe_context)
        except Exception as e:
            if os.getenv("AXIS_DEBUG"):
                print(
                    f"Condition evaluation failed: {condition} - {e}", file=sys.stderr
                )
            return False

    def run(self, input_data: dict) -> dict:
        """
        Execute rules against input data, return merged result.
        """
        result = {}

        for rule in self.rule_list:
            # Handle if/then rules
            if "if" in rule:
                condition = rule["if"]
                if self._evaluate_condition(condition, input_data):
                    if "then" in rule:
                        result.update(rule["then"])
                    break

            # Handle else rules (no condition)
            elif "else" in rule:
                result.update(rule.get("then", {}))
                break

        return result

    def serve(self, port: int = 8000, host: str = "0.0.0.0"):
        """
        Start HTTP server for rules (uses stdlib http.server).
        """
        from http.server import HTTPServer, BaseHTTPRequestHandler
        import json

        engine = self

        class RuleHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                # Read request body
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length)

                try:
                    # Parse JSON input
                    input_data = json.loads(body.decode("utf-8"))

                    # Run rules
                    result = engine.run(input_data)

                    # Send response
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(result).encode("utf-8"))

                except json.JSONDecodeError:
                    self.send_error(400, "Invalid JSON")
                except Exception as e:
                    self.send_error(500, str(e))

            def do_GET(self):
                # Simple health check
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"AXIS Rule Engine Running\n")

            def log_message(self, format, *args):
                # Suppress default logging unless debug mode
                if os.getenv("AXIS_DEBUG"):
                    super().log_message(format, *args)

        # Start server
        server = HTTPServer((host, port), RuleHandler)
        print(f"AXIS Rule Engine serving at http://{host}:{port}")
        print("POST JSON to / to execute rules")
        print("Press Ctrl+C to stop")

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down...")
            server.shutdown()


# ============================================================================
# SPC RUNTIME - Compatible with EDT Microkernel
# ============================================================================


def eval_expression(expr: str, context: dict) -> Any:
    """
    Safely evaluate expressions compatible with JS microkernel semantics.
    Translates JS-style expressions to Python equivalents.

    JS: row.region === 'US'
    PY: row['region'] == 'US'

    JS: data.userId
    PY: data['userId']
    """
    # Translate JS-style expressions to Python
    expr_py = expr

    # Replace === with ==
    expr_py = expr_py.replace("===", "==")
    expr_py = expr_py.replace("!==", "!=")

    # Replace JS logical operators
    expr_py = expr_py.replace("&&", " and ")
    expr_py = expr_py.replace("||", " or ")
    expr_py = expr_py.replace("!", " not ")

    # Handle dot notation: convert obj.prop to obj['prop']
    # This regex finds patterns like word.word but not numbers
    expr_py = re.sub(r"(\w+)\.(\w+)", r"\1['\2']", expr_py)

    # Handle template literals in expressions (basic support)
    # Convert ${expr} to str(expr)
    expr_py = re.sub(r"\$\{([^}]+)\}", r"str(\1)", expr_py)

    # Safety check
    dangerous = ["import", "exec", "eval", "__", "open", "file", "compile"]
    for d in dangerous:
        if d in expr_py:
            raise ValueError(f"Unsafe expression: {expr}")

    try:
        # Build safe evaluation context
        safe_context = {
            "data": context.get("data"),
            "state": context.get("state", {}),
            "row": context.get("row"),
            # Add common functions
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "abs": abs,
            "min": min,
            "max": max,
            "sum": sum,
            "round": round,
            "True": True,
            "False": False,
            "None": None,
            "null": None,  # JS compatibility
            "undefined": None,  # JS compatibility
        }

        # Evaluate the expression
        result = eval(expr_py, {"__builtins__": {}}, safe_context)
        return result

    except Exception as e:
        if os.getenv("AXIS_DEBUG"):
            print(
                f"Expression evaluation failed: {expr} -> {expr_py}: {e}",
                file=sys.stderr,
            )
        return None


def template_string(template: str, context: dict) -> str:
    """
    Process template strings with {{ expressions }}.
    Compatible with JS microkernel templating.
    """
    if not isinstance(template, str):
        return template

    def replacer(match):
        expr = match.group(1).strip()
        try:
            result = eval_expression(expr, context)
            return str(result) if result is not None else ""
        except Exception:
            return match.group(0)  # Return original on error

    # Replace {{ expr }} with evaluated result
    return re.sub(r"\{\{(.+?)\}\}", replacer, template)


def template_object(obj: Any, context: dict) -> Any:
    """
    Recursively process template strings in an object.
    Compatible with JS microkernel templateObject.
    """
    if isinstance(obj, str):
        # Check if entire string is a template expression
        if obj.startswith("{{") and obj.endswith("}}"):
            expr = obj[2:-2].strip()
            return eval_expression(expr, context)
        # Otherwise process as template string
        return template_string(obj, context)

    elif isinstance(obj, list):
        return [template_object(item, context) for item in obj]

    elif isinstance(obj, dict):
        return {key: template_object(value, context) for key, value in obj.items()}

    else:
        return obj


def compute_hash(*parts) -> str:
    """
    Compute deterministic hash from parts.
    Mirrors JS microkernel hash function.
    """
    s = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(s.encode()).hexdigest()[:8]


def apply_rules(data: dict, rules: List[dict], context: dict) -> dict:
    """
    Apply transformation rules to data.
    Compatible with JS microkernel applyRules.
    """
    result = json.loads(json.dumps(data))  # Deep copy

    for rule in rules:
        # Check if condition
        if "if" in rule:
            condition = rule["if"]
            if not eval_expression(
                condition, {"data": result, "state": context.get("state", {})}
            ):
                continue

        # Apply then transformations
        if "then" in rule:
            for key, template in rule["then"].items():
                result[key] = template_object(
                    template, {"data": result, "state": context.get("state", {})}
                )

    return result


def pick(obj: dict, fields: List[str]) -> dict:
    """
    Pick specified fields from object.
    Compatible with JS microkernel pick function.
    """
    return {field: obj[field] for field in fields if field in obj}


# ============================================================================
# PRIMITIVE HANDLERS - Compatible with EDT Microkernel
# ============================================================================


async def connector_handler(id: str, spec: dict, ctx: dict) -> Tuple[dict, List[dict]]:
    """
    Connector primitive handler - fetches data from URLs or files.
    Returns (patch, events) tuple compatible with JS microkernel.
    """
    url = spec.get("url", "")
    output_key = spec.get("outputKey", f"{id}_data")
    rules = spec.get("rules", {}).get("rules", [])

    try:
        # Process URL templates
        final_url = template_string(url, {"state": ctx["state"]})

        # Fetch data
        if final_url.startswith("http://") or final_url.startswith("https://"):
            if not HAS_REQUESTS:
                raise RuntimeError("HTTP connector requires: pip install requests")

            response = requests.get(final_url, timeout=30)
            response.raise_for_status()

            # Try to parse as JSON
            try:
                data = response.json()
            except json.JSONDecodeError:
                data = {"text": response.text}

        elif final_url.startswith("file://"):
            # Local file support
            file_path = final_url.replace("file://", "")
            with open(file_path, "r") as f:
                content = f.read()
                try:
                    data = json.loads(content)
                except json.JSONDecodeError:
                    data = {"text": content}

        else:
            # Mock data for testing (matches JS microkernel behavior)
            data = {"userId": 1, "id": 1, "title": "Test data", "completed": False}

        # Apply transformation rules if present
        if rules:
            data = apply_rules(data, rules, ctx)

        # Return patch and events
        patch = {output_key: data}
        events = [{"name": "connector.fetched", "for": id, "data": {"url": final_url}}]

        return patch, events

    except Exception as e:
        # Error handling
        ctx["log"]("error", f"Connector {id} failed: {e}")
        patch = {f"{id}_error": str(e)}
        events = []
        return patch, events


def processor_handler(id: str, spec: dict, ctx: dict) -> Tuple[dict, List[dict]]:
    """
    Processor primitive handler - transforms data through pipes.
    Returns (patch, events) tuple compatible with JS microkernel.
    """
    input_key = spec.get("inputKey", "")
    output_key = spec.get("outputKey", f"{id}_output")
    transform = spec.get("transform", [])
    pipes = spec.get("pipes", [])
    persist = spec.get("persist", [])

    # Get input data from state
    input_data = ctx["state"].get(input_key)

    if input_data is None:
        return {output_key: None}, []

    result = input_data

    # Apply transform rules
    if transform:
        result = apply_rules(result, transform, ctx)

    # Apply pipes
    for pipe in pipes:
        # Select (filter)
        if "select" in pipe:
            if isinstance(result, list):
                filtered = []
                for row in result:
                    if eval_expression(
                        pipe["select"], {"row": row, "state": ctx["state"]}
                    ):
                        filtered.append(row)
                result = filtered
            else:
                # For single objects, keep or discard
                if not eval_expression(
                    pipe["select"], {"row": result, "state": ctx["state"]}
                ):
                    result = None

        # Project (pick fields)
        if "project" in pipe and result:
            if isinstance(result, list):
                result = [pick(row, pipe["project"]) for row in result]
            else:
                result = pick(result, pipe["project"])

        # Derive (compute new fields)
        if "derive" in pipe and result:
            if isinstance(result, list):
                for row in result:
                    for key, expr in pipe["derive"].items():
                        row[key] = eval_expression(
                            expr, {"data": row, "row": row, "state": ctx["state"]}
                        )
            elif isinstance(result, dict):
                for key, expr in pipe["derive"].items():
                    result[key] = eval_expression(
                        expr, {"data": result, "state": ctx["state"]}
                    )
            else:
                # Wrap primitive in object
                result = {"value": result}
                for key, expr in pipe["derive"].items():
                    result[key] = eval_expression(
                        expr, {"data": result, "state": ctx["state"]}
                    )

    # Build patch
    patch = {output_key: result}

    # Handle persistence
    persisted = []
    if result and isinstance(result, dict):
        for key in result.keys():
            # Auto-persist keys starting with underscore or in persist list
            should_persist = key.startswith("_") or key in persist
            if should_persist:
                ctx["state"][key] = result[key]
                persisted.append(key)

    # Build events
    events = [{"name": "processor.computed", "for": id}]
    if persisted:
        events.append(
            {"name": "processor.persisted", "for": id, "data": {"persisted": persisted}}
        )

    return patch, events


def monitor_handler(id: str, spec: dict, ctx: dict) -> Tuple[dict, List[dict]]:
    """
    Monitor primitive handler - evaluates checks and thresholds.
    Returns (patch, events) tuple compatible with JS microkernel.
    """
    checks = spec.get("checks", [])
    thresholds = spec.get("thresholds", {})
    emit = spec.get("emit", "onChange")

    results = {}

    for check in checks:
        check_name = check.get("name", "unnamed")
        data_key = check.get("dataKey", "")
        expression = check.get("expression", "True")

        # Get data from state
        data = ctx["state"].get(data_key)

        # Evaluate expression
        value = eval_expression(expression, {"data": data, "state": ctx["state"]})

        if value is None:
            results[check_name] = {"value": None, "status": "unknown"}
            continue

        # Check thresholds
        threshold = thresholds.get(check_name, {})
        status = "normal"

        if "above" in threshold and value > threshold["above"]:
            status = "critical"
        if "below" in threshold and value < threshold["below"]:
            status = "critical"

        results[check_name] = {"value": value, "status": status, "threshold": threshold}

    # Determine if we should emit events
    output_key = f"{id}_monitoring"
    prev_results = ctx["state"].get(output_key)
    changed = json.dumps(prev_results) != json.dumps(results)

    should_emit = (
        emit == "always"
        or (emit == "onChange" and changed)
        or (
            emit == "onTrue"
            and any(r["status"] == "critical" for r in results.values())
        )
    )

    # Build response
    patch = {output_key: results}
    events = []

    if should_emit:
        events.append({"name": "monitor.alert", "for": id, "data": results})

    return patch, events


def adapter_handler(id: str, spec: dict, ctx: dict) -> Tuple[dict, List[dict]]:
    """
    Adapter primitive handler - sends data to external systems.
    Returns (patch, events) tuple compatible with JS microkernel.
    """
    kind = spec.get("kind", "webhook")

    if kind != "webhook":
        return {}, []

    url = spec.get("url", "")
    input_key = spec.get("inputKey")
    body_template = spec.get("body", {})
    idempotency_template = spec.get("idempotency_key")

    # Get input data
    input_data = ctx["state"].get(input_key, {}) if input_key else {}

    # Compute idempotency key
    if idempotency_template:
        idemp_key = template_object(
            idempotency_template, {"state": ctx["state"], "data": input_data}
        )
    else:
        idemp_key = compute_hash(id, ctx["clock"]())

    # Check if already sent (dedupe)
    dedupe_key = f"__sent_{compute_hash(id, idemp_key)}"
    if dedupe_key in ctx["state"]:
        ctx["log"]("info", f"Adapter {id}: Skipped (idempotent)")
        return {}, []

    # Process body template
    body = template_object(body_template, {"state": ctx["state"], "data": input_data})

    try:
        # In production, would actually send webhook
        # For now, just log it
        ctx["log"]("info", f"Adapter {id}: Would send webhook to {url}")

        # Mark as sent
        patch = {
            dedupe_key: ctx["clock"](),
            f"{id}_last_sent": {
                "timestamp": ctx["clock"](),
                "body": body,
                "idempotency_key": idemp_key,
            },
        }

        events = [{"name": "adapter.sent", "for": id, "data": {"url": url}}]

        return patch, events

    except Exception as e:
        ctx["log"]("error", f"Adapter {id} failed: {e}")
        return {f"{id}_error": str(e)}, []


def aggregator_handler(id: str, spec: dict, ctx: dict) -> Tuple[dict, List[dict]]:
    """
    Aggregator primitive handler - windowed aggregation of data.
    Returns (patch, events) tuple compatible with JS microkernel.
    """
    input_key = spec.get("inputKey", "")
    output_key = spec.get("outputKey", f"{id}_aggregated")
    window = spec.get("window", {})
    reduce = spec.get("reduce", {})

    # Get input data
    input_data = ctx["state"].get(input_key)

    if input_data is None:
        return {}, []

    # Get or create window state
    window_key = f"__window_{id}"
    window_data = ctx["state"].get(window_key, {"items": [], "start": ctx["clock"]()})

    # Add new item to window
    window_data["items"].append({"data": input_data, "timestamp": ctx["clock"]()})

    # Apply window size
    window_size_sec = window.get("size_sec", 30)
    now = datetime.now()

    # Filter items within window
    filtered_items = []
    for item in window_data["items"]:
        item_time = datetime.fromisoformat(item["timestamp"].replace("Z", "+00:00"))
        age_seconds = (now - item_time).total_seconds()
        if age_seconds < window_size_sec:
            filtered_items.append(item)

    window_data["items"] = filtered_items

    # Apply reduce operation
    result = filtered_items

    if reduce.get("emit"):
        emit_type = reduce["emit"]

        if emit_type == "latest":
            result = filtered_items[-1]["data"] if filtered_items else None
        elif emit_type == "count":
            result = len(filtered_items)
        elif emit_type == "all":
            result = [item["data"] for item in filtered_items]

    # Build response
    patch = {window_key: window_data, output_key: result}

    events = [
        {"name": "aggregator.window", "for": id, "data": {"count": len(filtered_items)}}
    ]

    return patch, events


def vault_handler(id: str, spec: dict, ctx: dict) -> Tuple[dict, List[dict]]:
    """
    Vault primitive handler - manages secrets securely.
    Returns (patch, events) tuple compatible with JS microkernel.
    """
    provider = spec.get("provider", "env")
    secrets = spec.get("secrets", [])
    rotation_policy = spec.get("rotation_policy", {})

    # Vault state key
    vault_key = f"__vault_{id}"
    vault_data = ctx["state"].get(vault_key, {"secrets": {}, "lastRotation": None})

    try:
        # Check rotation policy
        needs_rotation = False
        if rotation_policy:
            interval_hours = rotation_policy.get("interval_hours", 24)
            if vault_data["lastRotation"]:
                last_rotation = datetime.fromisoformat(
                    vault_data["lastRotation"].replace("Z", "+00:00")
                )
                age_hours = (datetime.now() - last_rotation).total_seconds() / 3600
                needs_rotation = age_hours > interval_hours
            else:
                needs_rotation = True

        if needs_rotation:
            ctx["log"]("info", f"Vault {id}: Rotating secrets per policy")
            vault_data["lastRotation"] = ctx["clock"]()

        # Store secret references (not actual values for security)
        for secret_ref in secrets:
            # In production, would fetch from actual vault provider
            # For now, try to get from environment variables
            vault_data["secrets"][secret_ref] = {
                "ref": secret_ref,
                "provider": provider,
                "lastAccessed": ctx["clock"](),
                # Don't store actual value in state for security
                "exists": os.getenv(secret_ref) is not None,
            }

        # Build response
        patch = {
            vault_key: vault_data,
            f"{id}_status": {
                "provider": provider,
                "secretCount": len(secrets),
                "lastRotation": vault_data["lastRotation"],
                "healthy": True,
            },
        }

        events = [
            {
                "name": "vault.rotated" if needs_rotation else "vault.accessed",
                "for": id,
                "data": {"provider": provider, "secretCount": len(secrets)},
            }
        ]

        return patch, events

    except Exception as e:
        ctx["log"]("error", f"Vault {id} failed: {e}")


return {
    f"{id}_error": str(e),
    f"{id}_status": {"provider": provider, "healthy": False},
}, []


# ============================================================================
# SPC RUNTIME ENGINE
# ============================================================================


class SPCEngine:
    """
    SPC (Service Pipeline Configuration) execution engine.
    Compatible with EDT microkernel from Pandas-as-a-Service.
    """

    def __init__(self):
        self.spc = None
        self.state = {}
        self.handlers = {}
        self.running = False
        self.metrics = {"ticks": 0, "events": 0}
        self.logs = []

        # Register default handlers
        self.register_handler("connector", connector_handler)
        self.register_handler("processor", processor_handler)
        self.register_handler("monitor", monitor_handler)
        self.register_handler("adapter", adapter_handler)
        self.register_handler("aggregator", aggregator_handler)
        self.register_handler("vault", vault_handler)

    def register_handler(self, type: str, handler: Callable):
        """Register a primitive handler."""
        self.handlers[type] = handler
        if os.getenv("AXIS_DEBUG"):
            print(f"Registered handler: {type}", file=sys.stderr)

    def load_spc(self, spc: Union[dict, str]):
        """
        Load SPC from dict or file path (JSON/YAML).
        """
        if isinstance(spc, str):
            # Load from file
            with open(spc, "r") as f:
                if spc.endswith(".yaml") or spc.endswith(".yml"):
                    if not HAS_YAML:
                        raise RuntimeError("YAML support requires: pip install pyyaml")
                    self.spc = yaml.safe_load(f)
                else:
                    self.spc = json.load(f)
        else:
            self.spc = spc

        # Initialize state if not present
        if "state" not in self.spc:
            self.spc["state"] = {}

        self.state = self.spc["state"]

        # Validate SPC structure
        if "spc_version" not in self.spc:
            raise ValueError("Invalid SPC: missing spc_version")
        if "services" not in self.spc:
            raise ValueError("Invalid SPC: missing services")

        return self

    def create_context(self) -> dict:
        """
        Create execution context for handlers.
        Mirrors JS microkernel context.
        """
        return {
            "state": self.state,
            "services": self.spc.get("services", {}),
            "clock": lambda: datetime.now().isoformat() + "Z",
            "hash": compute_hash,
            "log": self.log,
            "fetch": self.fetch,
        }

    def log(self, level: str, message: str, meta: dict = None):
        """Log a message with level."""
        entry = {
            "level": level,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "meta": meta or {},
        }
        self.logs.append(entry)

        # Print to stderr in debug mode or for errors
        if os.getenv("AXIS_DEBUG") or level == "error":
            print(f"[{entry['timestamp']}] {level.upper()}: {message}", file=sys.stderr)

    def fetch(self, url: str, options: dict = None) -> dict:
        """
        Fetch helper for handlers.
        Mirrors JS microkernel fetch behavior.
        """
        if not HAS_REQUESTS:
            raise RuntimeError("Fetch requires: pip install requests")

        try:
            response = requests.get(url, **(options or {}))
            response.raise_for_status()

            # Try to return JSON, fallback to text
            try:
                return response.json()
            except json.JSONDecodeError:
                return {"text": response.text}

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Fetch failed: {e}")

    async def tick(self) -> dict:
        """
        Execute one tick of the pipeline.
        Returns dict with state and events.
        Async to support async handlers (like connector).
        """
        if not self.spc:
            raise RuntimeError("No SPC loaded")

        ctx = self.create_context()
        all_events = []

        # Process each service
        for service_id, service in self.spc.get("services", {}).items():
            # Skip if not running
            if service.get("status") != "running":
                continue

            service_type = service.get("type")
            if not service_type:
                self.log("warn", f"Service {service_id} has no type")
                continue

            # Get handler
            handler = self.handlers.get(service_type)
            if not handler:
                self.log("warn", f"No handler for type: {service_type}")
                continue

            try:
                # Execute handler
                spec = service.get("spec", {})

                # Support both sync and async handlers
                import asyncio
                import inspect

                if inspect.iscoroutinefunction(handler):
                    # Async handler
                    patch, events = await handler(service_id, spec, ctx)
                else:
                    # Sync handler
                    patch, events = handler(service_id, spec, ctx)

                # Apply patch to state
                if patch:
                    self.state.update(patch)

                # Collect events
                if events:
                    all_events.extend(events)

                # Update last run time
                service["lastRun"] = ctx["clock"]()

            except Exception as e:
                self.log("error", f"Service {service_id} failed: {e}")
                self.state[f"{service_id}_error"] = str(e)

        # Update metrics
        self.metrics["ticks"] += 1
        self.metrics["events"] += len(all_events)

        return {"state": self.state, "events": all_events, "metrics": self.metrics}

    def run_once(self) -> dict:
        """
        Synchronous wrapper for tick().
        """
        import asyncio

        # Get or create event loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Run tick
        if loop.is_running():
            # Already in async context
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self.tick())
                return future.result()
        else:
            # Not in async context
            return loop.run_until_complete(self.tick())

    def run_continuous(self, interval: float = 5.0):
        """
        Run pipeline continuously with interval (seconds).
        """
        self.running = True
        self.log("info", f"Starting continuous execution (interval: {interval}s)")

        try:
            while self.running:
                result = self.run_once()

                # Print events if any
                if result["events"]:
                    for event in result["events"]:
                        print(
                            f"Event: {event['name']} for {event['for']}",
                            file=sys.stderr,
                        )

                # Wait for next tick
                time.sleep(interval)

        except KeyboardInterrupt:
            self.log("info", "Stopping continuous execution")
            self.running = False

    def stop(self):
        """Stop continuous execution."""
        self.running = False

    def get_state(self) -> dict:
        """Get current state."""
        return self.state

    def get_logs(self) -> List[dict]:
        """Get execution logs."""
        return self.logs

    def export_state(self, path: str):
        """Export current state to JSON file."""
        with open(path, "w") as f:
            json.dump(
                {
                    "state": self.state,
                    "metrics": self.metrics,
                    "exported_at": datetime.now().isoformat(),
                },
                f,
                indent=2,
            )
        self.log("info", f"State exported to {path}")


def run_spc(spc: Union[dict, str], handlers: dict = None, state: dict = None) -> dict:
    """
    Execute an SPC with optional custom handlers and initial state.
    Returns dict with final state and events.

    This is the main entry point for programmatic use.
    """
    engine = SPCEngine()

    # Register custom handlers if provided
    if handlers:
        for type_name, handler in handlers.items():
            engine.register_handler(type_name, handler)

    # Load SPC
    engine.load_spc(spc)

    # Set initial state if provided
    if state:
        engine.state.update(state)

    # Run one tick
    return engine.run_once()


# ============================================================================
# CLI INTERFACE
# ============================================================================


def main():
    """
    CLI entry point for running SPC files.

    Usage:
        python core.py run pipeline.spc.json [--watch] [--interval 5]
        python core.py validate pipeline.spc.json
        python core.py serve --port 8080
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="AXIS-PY: Declarative Pipeline Runtime"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run an SPC file")
    run_parser.add_argument("file", help="Path to SPC file (JSON or YAML)")
    run_parser.add_argument("--watch", action="store_true", help="Run continuously")
    run_parser.add_argument(
        "--interval", type=float, default=5.0, help="Tick interval in seconds"
    )
    run_parser.add_argument("--output", help="Export final state to JSON file")
    run_parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate an SPC file")
    validate_parser.add_argument("file", help="Path to SPC file")

    # Serve command (for rule engine)
    serve_parser = subparsers.add_parser("serve", help="Start HTTP server for rules")
    serve_parser.add_argument("--port", type=int, default=8000, help="Server port")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Server host")
    serve_parser.add_argument("--rules", help="Path to rules YAML file")

    args = parser.parse_args()

    # Handle commands
    if args.command == "run":
        # Set debug mode
        if args.debug:
            os.environ["AXIS_DEBUG"] = "1"

        try:
            # Create engine
            engine = SPCEngine()
            engine.load_spc(args.file)

            print(f"Loaded SPC: {engine.spc.get('meta', {}).get('name', 'Unnamed')}")
            print(f"Services: {len(engine.spc.get('services', {}))}")
            print("-" * 50)

            if args.watch:
                # Continuous execution
                engine.run_continuous(interval=args.interval)
            else:
                # Single tick
                result = engine.run_once()

                # Print results
                print("\n📊 Final State:")
                print(json.dumps(result["state"], indent=2))

                if result["events"]:
                    print("\n📡 Events:")
                    for event in result["events"]:
                        print(f"  - {event['name']} for {event['for']}")

                print(f"\n📈 Metrics: {result['metrics']}")

                # Export if requested
                if args.output:
                    engine.export_state(args.output)

        except Exception as e:
            print(f"❌ Error: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "validate":
        try:
            engine = SPCEngine()
            engine.load_spc(args.file)

            # Validate each service
            errors = []
            for service_id, service in engine.spc.get("services", {}).items():
                if "type" not in service:
                    errors.append(f"Service {service_id}: missing type")
                if "spec" not in service:
                    errors.append(f"Service {service_id}: missing spec")

                # Check handler exists
                if service.get("type") not in engine.handlers:
                    errors.append(
                        f"Service {service_id}: unknown type '{service.get('type')}'"
                    )

            if errors:
                print("❌ Validation failed:")
                for error in errors:
                    print(f"  - {error}")
                sys.exit(1)
            else:
                print("✅ SPC is valid")
                print(f"  Version: {engine.spc.get('spc_version')}")
                print(f"  Services: {len(engine.spc.get('services', {}))}")

        except Exception as e:
            print(f"❌ Invalid SPC: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "serve":
        # Start rule engine HTTP server
        if args.rules:
            engine = RuleEngine(args.rules)
        else:
            # Default rules
            engine = RuleEngine(
                {
                    "rules": [
                        {
                            "if": "True",
                            "then": {"status": "ok", "message": "AXIS Rule Engine"},
                        }
                    ]
                }
            )

        engine.serve(port=args.port, host=args.host)

    else:
        parser.print_help()


# Convenience functions for common patterns
def load_rules(path: str) -> RuleEngine:
    """Load rules from YAML file."""
    return RuleEngine(path)


def quick_validate(data: dict, **types) -> dict:
    """
    Quick validation with keyword arguments.
    Example: quick_validate(data, name="str", age="int", email="str?")
    """
    is_valid, errors = validate(data, types)
    if not is_valid:
        raise ValueError(f"Validation failed: {'; '.join(errors)}")
    return data


# Example usage patterns
if __name__ == "__main__":
    # Check if running as script with args
    if len(sys.argv) > 1:
        main()
    else:
        # Example usage
        print("AXIS-PY Examples:\n")

        # Example 1: Validation
        data = {"name": "Alice", "age": "25"}
        is_valid, errors = validate(data, {"name": "str", "age": "int"})
        print(f"1. Validation - Valid: {is_valid}, Data: {data}")

        # Example 2: Rules
        rules = {
            "rules": [
                {"if": "age >= 18", "then": {"status": "adult"}},
                {"else": None, "then": {"status": "minor"}},
            ]
        }
        engine = RuleEngine(rules)
        print(f"2. Rules - Result: {engine.run({'age': 25})}")

        # Example 3: SPC execution
        sample_spc = {
            "spc_version": "1.0",
            "meta": {"name": "Example Pipeline"},
            "services": {
                "demo-processor": {
                    "type": "processor",
                    "status": "running",
                    "spec": {
                        "inputKey": "input",
                        "outputKey": "output",
                        "pipes": [{"derive": {"doubled": "data * 2"}}],
                    },
                }
            },
            "state": {"input": 21},
        }

        result = run_spc(sample_spc)
        print(f"3. SPC - Output: {result['state'].get('output')}")

        print("\nRun with arguments for more options:")
        print("  python core.py run pipeline.spc.json")
        print("  python core.py serve --port 8080")

