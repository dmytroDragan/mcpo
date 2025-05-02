from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union
from typing_extensions import ForwardRef
from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo

def _handle_all_of(schemas, model_cache, model_name, field_name, is_required, schema_defs, _processing_refs, field_kwargs) -> Tuple[Any, FieldInfo]:
    # Merge all sub-schemas in 'allOf' into a single schema
    merged = {"type": "object", "properties": {}, "required": []}
    for sub in schemas:
        if "properties" in sub:
            merged["properties"].update(sub["properties"])
        if "required" in sub:
            merged["required"].extend(sub["required"])
    # Recursively process the merged schema
    return process_schema_property(
        model_cache, merged, model_name, field_name, is_required, schema_defs, _processing_refs
    )

def _handle_union(schemas, union_name, model_cache, field_name, schema_defs, _processing_refs, field_kwargs) -> Tuple[Any, FieldInfo]:
    # Build a Union type from all sub-schemas in 'anyOf' or 'oneOf'
    types = []
    for i, sub_schema in enumerate(schemas):
        type_info, _ = process_schema_property(
            model_cache, sub_schema, union_name, f"{field_name}_u{i}", True, schema_defs, _processing_refs
        )
        if type_info not in types:
            types.append(type_info)
    if len(types) == 1:
        return types[0], Field(**field_kwargs)
    return Union[tuple(types)], Field(**field_kwargs)

def process_schema_property(
        model_cache: Dict[str, Type],
        schema: Dict[str, Any],
        model_name: str,
        field_name: str,
        is_required: bool,
        schema_defs: Optional[Dict[str, Any]] = None,
        _processing_refs: Optional[Set[str]] = None
) -> Tuple[Any, FieldInfo]:
    """
    Recursively process a JSON schema property and return the corresponding
    Python type and Pydantic FieldInfo for use in dynamic model creation.

    Args:
        model_cache: ModelCache instance for caching models.
        schema: The JSON schema property definition.
        model_name: Name of the parent model.
        field_name: Name of the field being processed.
        is_required: Whether the field is required.
        schema_defs: Optional dictionary of schema definitions for $ref resolution.
        _processing_refs: Internal set to track references and avoid circular refs.

    Returns:
        Tuple of (type, FieldInfo) for use in Pydantic model creation.
    """
    if _processing_refs is None:
        _processing_refs = set()

    # Prepare field metadata and description
    field_kwargs = {
        "json_schema_extra": {"metadata": {}},
        "description": schema.get("description", ""),
    }

    # Set default value for the field
    if "default" in schema:
        field_kwargs["default"] = schema["default"]
    elif not is_required:
        field_kwargs["default"] = None
    else:
        field_kwargs["default"] = ...

    def handle_ref(ref_path: str):
        # Handle $ref resolution, including circular references
        ref_name = ref_path.split("/")[-1]
        if ref_name in _processing_refs:
            return ForwardRef(ref_name), Field(**field_kwargs)
        if ref_name.startswith("_"):
            return Dict[str, Any], Field(**field_kwargs)
        if schema_defs and ref_name in schema_defs:
            _processing_refs.add(ref_name)
            ref_schema = schema_defs[ref_name]
            if ref_name in model_cache and model_cache.get(ref_name) is not None:
                _processing_refs.remove(ref_name)
                return model_cache[ref_name], Field(**field_kwargs)
            model_cache[ref_name] = None  # placeholder for circular refs
            type_info, _ = process_schema_property(
                model_cache, ref_schema, ref_name, ref_name, True, schema_defs, _processing_refs
            )
            _processing_refs.remove(ref_name)
            if isinstance(type_info, ForwardRef):
                type_info = create_model(ref_name, __base__=BaseModel)
            model_cache[ref_name] = type_info
            return type_info, Field(**field_kwargs)
        return Dict[str, Any], Field(**field_kwargs)

    # Handle $ref property
    if "$ref" in schema:
        return handle_ref(schema["$ref"])

    # Handle allOf composition
    if "allOf" in schema:
        return _handle_all_of(
            schema["allOf"], model_cache, model_name, field_name, is_required, schema_defs, _processing_refs, field_kwargs
        )

    # Handle anyOf/oneOf composition (union types)
    if "anyOf" in schema or "oneOf" in schema:
        sub_schemas = schema.get("anyOf") or schema.get("oneOf")
        return _handle_union(
            sub_schemas, f"{model_name}_{field_name}_union", model_cache, field_name, schema_defs, _processing_refs, field_kwargs
        )

    # Handle object type
    if schema.get("type") == "object":
        if not schema.get("properties"):
            return Dict[str, Any], Field(**field_kwargs)
        full_model_name = f"{model_name}_{field_name}_model"
        if full_model_name in model_cache and model_cache.get(full_model_name) is not None:
            return model_cache[full_model_name], Field(**field_kwargs)
        fields = {}
        required_fields = schema.get("required", [])
        # Recursively process each property
        for prop_name, prop_schema in schema["properties"].items():
            prop_type, prop_field = process_schema_property(
                model_cache, prop_schema, full_model_name, prop_name, prop_name in required_fields, schema_defs, _processing_refs
            )
            fields[prop_name] = (prop_type, prop_field)
        model = create_model(full_model_name, **fields)
        model_cache[full_model_name] = model
        # Add object constraints to metadata
        if "minProperties" in schema:
            field_kwargs["json_schema_extra"]["metadata"]["min_properties"] = schema["minProperties"]
        if "maxProperties" in schema:
            field_kwargs["json_schema_extra"]["metadata"]["max_properties"] = schema["maxProperties"]
        if "additionalProperties" in schema:
            field_kwargs["json_schema_extra"]["metadata"]["additional_properties"] = schema["additionalProperties"]
        return model, Field(**field_kwargs)

    # Handle array type
    if schema.get("type") == "array":
        if "items" in schema:
            item_type, _ = process_schema_property(
                model_cache, schema["items"], model_name, f"{field_name}_item", True, schema_defs, _processing_refs
            )
            type_info = List[item_type]
        else:
            type_info = List[Any]
        # Add array constraints to metadata
        if "minItems" in schema:
            field_kwargs["json_schema_extra"]["metadata"]["min_items"] = schema["minItems"]
        if "maxItems" in schema:
            field_kwargs["json_schema_extra"]["metadata"]["max_items"] = schema["maxItems"]
        if "uniqueItems" in schema:
            field_kwargs["json_schema_extra"]["metadata"]["unique_items"] = schema["uniqueItems"]
        return type_info, Field(**field_kwargs)

    # Handle primitive types and type arrays
    t = schema.get("type")
    type_info = Any
    if t == "string":
        type_info = str
        # Add string constraints to metadata
        if "minLength" in schema:
            field_kwargs["json_schema_extra"]["metadata"]["min_length"] = schema["minLength"]
        if "maxLength" in schema:
            field_kwargs["json_schema_extra"]["metadata"]["max_length"] = schema["maxLength"]
        if "pattern" in schema:
            field_kwargs["json_schema_extra"]["metadata"]["pattern"] = schema["pattern"]
        if "format" in schema:
            field_kwargs["json_schema_extra"]["metadata"]["format"] = schema["format"]
    elif t == "integer":
        type_info = int
    elif t == "number":
        type_info = float
    elif t == "boolean":
        type_info = bool
    elif t == "null":
        type_info = None
    elif isinstance(t, list):
        # Handle multiple types (e.g., ["string", "null"])
        types = []
        for typ in t:
            if typ == "string":
                types.append(str)
            elif typ == "integer":
                types.append(int)
            elif typ == "number":
                types.append(float)
            elif typ == "boolean":
                types.append(bool)
            elif typ == "null":
                types.append(type(None))
            elif typ == "array":
                types.append(List[Any])
            elif typ == "object":
                types.append(Dict[str, Any])
        if len(types) == 1:
            type_info = types[0]
        else:
            type_info = Union[tuple(types)]
    elif t is not None:
        type_info = Any

    # Add numeric constraints to metadata
    if t in ("number", "integer"):
        if "minimum" in schema:
            field_kwargs["json_schema_extra"]["metadata"]["minimum"] = schema["minimum"]
        if "maximum" in schema:
            field_kwargs["json_schema_extra"]["metadata"]["maximum"] = schema["maximum"]
        if "exclusiveMinimum" in schema:
            field_kwargs["json_schema_extra"]["metadata"]["exclusive_minimum"] = schema["exclusiveMinimum"]
        if "exclusiveMaximum" in schema:
            field_kwargs["json_schema_extra"]["metadata"]["exclusive_maximum"] = schema["exclusiveMaximum"]
        if "multipleOf" in schema:
            field_kwargs["json_schema_extra"]["metadata"]["multiple_of"] = schema["multipleOf"]

    # Add enum and const constraints to metadata
    if "enum" in schema:
        field_kwargs["json_schema_extra"]["metadata"]["enum"] = schema["enum"]
    if "const" in schema:
        field_kwargs["json_schema_extra"]["metadata"]["const"] = schema["const"]

    return type_info, Field(**field_kwargs)