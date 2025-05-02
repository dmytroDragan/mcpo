from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Tuple, Set, Union, ForwardRef, Callable, Awaitable

from fastapi import HTTPException
from mcp import ClientSession
from mcp.types import (
    CallToolResult,
    PARSE_ERROR,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    INVALID_PARAMS,
    INTERNAL_ERROR,
)
from mcpo.utils.common_logging import logger
from pydantic import Field, create_model, BaseModel
from pydantic.fields import FieldInfo
from mcpo.utils.schema_utils import process_schema_property

MCP_ERROR_TO_HTTP_STATUS = {
    PARSE_ERROR: 400,
    INVALID_REQUEST: 400,
    METHOD_NOT_FOUND: 404,
    INVALID_PARAMS: 422,
    INTERNAL_ERROR: 500,
}

def _process_schema_property(
        _model_cache: Dict[str, Type],
        prop_schema: Dict[str, Any],
        model_name_prefix: str,
        prop_name: str,
        is_required: bool,
        schema_defs: Optional[Dict] = None,
        _processing_refs: Optional[Set[str]] = None
) -> Tuple[Any, FieldInfo]:
    return process_schema_property(
        _model_cache, prop_schema,
        model_name_prefix, prop_name, 
        is_required, schema_defs, _processing_refs)

def get_model_fields(
        name: str,
        properties: Dict[str, Any],
        required: List[str],
        schema_defs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Tuple[Type, Field]]:
    """
    Generate a dictionary of Pydantic model fields from a JSON schema.

    Args:
        name: Name of the model.
        properties: Dictionary of property schemas.
        required: List of required property names.
        schema_defs: Optional dictionary of schema definitions for $ref resolution.

    Returns:
        Dictionary mapping property names to (type, Field) tuples.
    """
    fields = {}
    model_cache = {}
    for prop_name, prop_schema in properties.items():
        try:
            type_hint, field_info = _process_schema_property(
                model_cache,
                prop_schema,
                name,
                prop_name,
                prop_name in required,
                schema_defs,
                )
            fields[prop_name] = (type_hint, field_info)
        except Exception as e:
            logger.error(f"Error processing parameter {prop_name}: {str(e)}", exc_info=True)
            continue

    return fields


def get_tool_handler(
        session: ClientSession,
        endpoint_name: str,
        form_model_fields: Dict[str, Tuple[Type, Field]],
        response_model_fields: Optional[Dict[str, Tuple[Type, Field]]] = None,
) -> Tuple[Callable[[BaseModel], Awaitable[Any]], Optional[Type[BaseModel]]]:
    """
    Dynamically creates a FastAPI endpoint handler and response model for a given tool.

    Args:
        session: The MCP client session used to execute the tool.
        endpoint_name: The name of the endpoint/tool.
        form_model_fields: Dictionary mapping request field names to (type, Field) tuples for the request model.
        response_model_fields: Optional dictionary mapping response field names to (type, Field) tuples for the response model.

    Returns:
        A tuple containing:
            - An async handler function for FastAPI, accepting the generated request model.
            - The generated Pydantic response model class, or None if not provided.
    """
    # Create form model with proper field definitions
    form_model = create_model(
        f"{endpoint_name}_form_model",
        __base__=BaseModel,
        **{k: (v[0], v[1]) for k, v in form_model_fields.items()}
    )

    # Create response model if fields are provided
    response_model = None
    if response_model_fields:
        response_model = create_model(
            f"{endpoint_name}_response_model",
            __base__=BaseModel,
            **{k: (v[0], v[1]) for k, v in response_model_fields.items()}
        )

    async def handler(form_data: form_model):
        try:
            # Transform filters for search endpoint
            params = form_data.model_dump()

            logger.debug(f"Executing tool {endpoint_name} with params: {params}")
            result = await session.execute_tool(endpoint_name, params)
            logger.debug(f"Tool {endpoint_name} result type: {type(result)}")
            logger.debug(f"Tool {endpoint_name} result: {result}")

            if isinstance(result, CallToolResult):
                logger.debug(f"CallToolResult fields: {result.model_fields}")
                logger.debug(f"CallToolResult isError: {getattr(result, 'isError', None)}")
                logger.debug(f"CallToolResult content: {getattr(result, 'content', None)}")
                logger.debug(f"CallToolResult meta: {getattr(result, 'meta', None)}")

                if result.isError:
                    raise HTTPException(status_code=400, detail="Tool execution failed")
                # Return the raw result for now, let the tool handle its own response format
                return result
            return result
        except HTTPException:
            raise  # Let FastAPI handle HTTPException properly       
        except Exception as e:
            logger.error(f"Error executing tool {endpoint_name}: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Tool execution failed: {str(e)}")

    handler.__name__ = endpoint_name
    return handler, response_model