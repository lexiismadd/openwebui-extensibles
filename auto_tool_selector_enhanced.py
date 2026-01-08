"""
title: Auto Tool Selector Enhanced
author: lexiismadd
author_url: https://github.com/lexiismadd
version: 2.2.0
description: >
    Automatically selects relevant tools for a user query based on conversation
    history and available tool metadata. Works with Open WebUI 0.6.43.
    Supports both built-in features and custom tools, avoiding duplicates.
    Uses expert model (expert model or task model) for intelligent tool selection.
    Install via Functions in Open WebUI.
    Forked and updated from original Auto Tool Selector by siwadon-jay (https://github.com/siwadon-jay)
required_open_webui_version: 0.6.43
requirements: aiohttp, loguru, orjson

================================================================================
CONFIGURATION OPTIONS
================================================================================

**Valve Settings:**
- expert_model_url: OpenAI-compatible API URL for the expert model (optional)
- expert_model_api_key: API key for the expert model (optional)
- expert_model_id: Model ID for the expert model (optional)
- Other settings control tool filtering, history length, etc.

**Expert Model Priority (for tool selection):**
1. Expert Model (Custom API) - if configured
2. Task Model (Open WebUI Config: Local → External → Workspace → Current)
3. Current/Base Model - final fallback

**Benefits of Expert Model:**
- Use a fast, cheap model specifically for tool selection
- Separate tool selection concerns from the main response
- Full control over which model makes tool decisions
================================================================================
"""

import traceback
from pydantic import BaseModel, Field
from typing import Callable, Awaitable, Any, Literal, Optional, Dict, List
import json
import orjson
import re
import ast
import logging
from difflib import SequenceMatcher
from string import Template

from open_webui.models.users import Users
from open_webui.models.tools import Tools
from open_webui.utils.misc import get_last_user_message

logger = logging.getLogger(__name__)

STATUS_MESSAGES = {
    "analyse_query": {
        "simple": "Analyzing query...",
        "friendly": "Analyzing your request to determine which tools would be most helpful...",
        "professional": "Analyzing query to identify relevant tools and features..."
    },
    "query_expert_model": {
        "simple": "Selecting tools...",
        "friendly": "Analyzing available tools to find the best match for your request...",
        "professional": "Querying expert model for tool selection..."
    },
    "no_tool_available": {
        "simple": "No tools available",
        "friendly": "There are no tools available for me to use... Should there be?",
        "professional": "Unable to identify any tools for selection..."
    },
    "prepare_tools": {
        "simple": Template("Preparing $count tools..."),
        "friendly": Template("I found $count tools that might help..."),
        "professional": Template("Analyzing $count available tools...")
    },
    "features_tools_msg": {
        "simple": Template("Selected $count tool(s)"),
        "friendly": Template("Good news! I found $parts that I can use..."),
        "professional": Template("Selected $parts: $tool_ids")
    },
    "no_tool_found": {
        "simple": "No relevant tools found",
        "friendly": "I'm sorry, but I can't seem to find a suitable tool to help answer your query",
        "professional": "No relevant tools found for this query"
    }
}


class Filter:
    """
    Automatically selects relevant tools for a user query based on:
    - Conversation history
    - Available tools and built-in features
    - Expert model analysis (expert model, task model, or current model)
    """

    class Valves(BaseModel):
        priority: int = Field(
            default=0,
            description="Priority order for filter execution. Lower values run first."
        )
        template: str = Field(
            default="""Available Tools and Features:
{{TOOLS}}

Analyze the user's query and conversation history to select ALL relevant tools and features that would be helpful.

Guidelines:
- Select tools that directly address the user's request
- Consider the context from conversation history
- You can select multiple tools if they work together
- Built-in features (web_search, image_generation, code_interpreter) should be selected when appropriate
- Custom tools should be selected based on their descriptions
- Avoid selecting tools with nearly identical functionality
- If unsure, lean towards selecting tools that might be helpful

Return ONLY a valid JSON array of tool/feature IDs. No explanation, no markdown, just the array.
Examples: ["web_search_and_crawl", "image_generation", "chat_history_search"] or [] if no tools match.

Important: Return an empty array [] if no tools are relevant, not null or undefined.""",
            description="System prompt template for expert model tool selection. Use {{TOOLS}} placeholder."
        )
        status: bool = Field(
            title="More status updates",
            default=False,
            description="Show more status updates during tool selection"
        )
        status_messages: Literal["simple","friendly","professional"] = Field(
            title="Status Message Tone",
            default="simple",
            description="What type of tone should the status messages from the function use?"
        )
        debug: bool = Field(
            title="Enable debug logging",
            default=False,
            description="Enable detailed debug logging"
        )
        max_tools: int = Field(
            title="Maximum Tools",
            default=50,
            ge=1,
            le=100,
            description="Maximum number of tools to consider"
        )
        similarity_threshold: float = Field(
            title="Similarity threshold",
            default=0.8,
            ge=0.5,
            le=1.0,
            description="Threshold for filtering similar tools (0.5-1.0)"
        )
        max_history_chars: int = Field(
            title="Maximum history chars",
            default=1000,
            ge=100,
            le=5000,
            description="Maximum characters from conversation history to include"
        )
        enable_builtin_websearch: bool = Field(
            title="Enable Built-in Web Search",
            default=True,
            description="Enable automatic selection of built-in web search"
        )
        enable_builtin_image_generation: bool = Field(
            title="Enable Built-in Image Generator",
            default=True,
            description="Enable automatic selection of built-in image generation"
        )
        enable_builtin_code_interpreter: bool = Field(
            title="Enable Built-in Code Interpreter",
            default=True,
            description="Enable automatic selection of built-in code interpreter"
        )
        enable_other_builtin_features: bool = Field(
            title="Enable All Other Built-in Features",
            default=True,
            description="Enable automatic selection of other built-in features not listed above"
        )
        enable_custom_tools: bool = Field(
            title="Enable Custom Tool Selection",
            default=True,
            description="Enable automatic selection of custom tools"
        )
        exclude_custom_tools: str = Field(
            title="Exclude Specific Custom Tools",
            default="",
            description="Comma-separated list of tool IDs to exclude from automatic selection. Leave blank to automatically select from all available custom tools."
        )
        force_custom_tools: str = Field(
            title="Force Specific Custom Tools",
            default="",
            description="Comma-separated list of tool IDs to force the use of."
        )
        # Expert Model for tool selection (EXPERT MODEL - PRIORITY 1)
        expert_model_url: str = Field(
            title="Expert Model API URL",
            default="",
            description="Expert Model API URL.Example: 'https://api.openai.com/v1' or 'http://ollama:11434'. If blank, falls back to task model."
        )
        expert_model_api_key: str = Field(
            title="Expert Model API Key",
            default="",
            description="API key for the expert model. Leave blank if no authentication is required."
        )
        expert_model_id: str = Field(
            title="Expert Model ID",
            default="",
            description="Model ID for the expert model (e.g., 'gpt-4o-mini', 'llama3.2:latest').  A fast, cheap model like 'gpt-4o-mini' or lightweight thinking model like 'deepseek-r1:7b' can work well. If blank, falls back to task model."
        )

    def __init__(self) -> None:
        self.valves = self.Valves()
        self.s_key = self.valves.status_messages if self.valves.status_messages else "simple"
        logger.info(f"Auto Tool Selector Enhanced v2.2.0 Instantiated")

    # -------------------- Helper: Event emitter --------------------
    async def emit_status(
        self,
        __event_emitter__: Callable[[Dict[str, Any]], Awaitable[None]],
        step: str,
        description: str,
        done: bool = False,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit status updates to the UI"""
        if self.valves.status or self.valves.debug:
            data = {"step": step, "description": description, "done": done}
            if extra:
                data.update(extra)
            await __event_emitter__({"type": "status", "data": data})

    # -------------------- Helper: Prepare tools --------------------
    def prepare_tools(
        self, __model__: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, str]]:
        """
        Gather all available tools and built-in features.
        Returns a list of dicts with 'id', 'description', and 'type' keys.
        """
        all_tools: List[Dict[str, str]] = []

        # Add built-in features if enabled
        if self.valves.enable_builtin_websearch or self.valves.enable_builtin_image_generation or self.valves.enable_builtin_code_interpreter:
            built_in_features = dict()
            if self.valves.enable_builtin_websearch:
                built_in_features["web_search"] = "Search the internet for current information, news, facts, or research"
            if self.valves.enable_builtin_image_generation:
                built_in_features["image_generation"] = "Generate, create, or produce images based on text descriptions"
            if self.valves.enable_builtin_code_interpreter:
                built_in_features["code_interpreter"] = "Execute Python code, analyze data, create visualizations, or perform calculations"

            all_tools.extend([
                {"id": k, "description": v, "type": "builtin"}
                for k, v in built_in_features.items()
            ])

        # Add custom tools if enabled
        if self.valves.enable_custom_tools:
            try:
                custom_tools = Tools.get_tools()
                custom_tool_exclusions = [str(tool_id).strip().lower() for tool_id in self.valves.exclude_custom_tools.split(",")]
                for tool in custom_tools:
                    tool_id = tool.id
                    if str(tool_id).lower() in custom_tool_exclusions:
                        continue
                    
                    description = ""
                    if hasattr(tool, "meta") and tool.meta:
                        if hasattr(tool.meta, "description"):
                            description = tool.meta.description
                        elif isinstance(tool.meta, dict):
                            description = tool.meta.get("description", "")
                    
                    if not description and hasattr(tool, "name"):
                        description = f"Custom tool: {tool.name}"
                    elif not description:
                        description = f"Custom tool: {tool_id}"
                    
                    all_tools.append({
                        "id": tool_id,
                        "description": description,
                        "type": "custom"
                    })
            except Exception as e:
                logger.error(f"Error loading custom tools: {e}")
                if self.valves.debug:
                    logger.exception(f"Full traceback:{traceback.format_exc()}")

        # Filter by model's allowed tools if specified
        if __model__ and __model__.get("info", {}).get("meta", {}).get("toolIds"):
            available_tool_ids: List[str] = __model__["info"]["meta"]["toolIds"]
            all_tools = [t for t in all_tools if t["id"] in available_tool_ids]

        return all_tools[: self.valves.max_tools]

    # -------------------- Helper: Summarize history --------------------
    @staticmethod
    def summarize_history(messages: List[Dict[str, Any]], max_chars: int = 1000) -> str:
        """
        Create a concise summary of recent conversation history.
        """
        last_messages: List[Dict[str, Any]] = messages[-15:]
        summary_lines: List[str] = []

        for msg in last_messages:
            role: str = msg.get("role", "").upper()
            content = msg.get("content", "")
            
            if isinstance(content, str):
                content_text = content
            elif isinstance(content, list):
                content_text = " ".join([
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in content
                ])
            else:
                content_text = str(content)
            
            content_text = content_text.replace("\n", " ").strip()
            
            if content_text and role in ["USER", "ASSISTANT"]:
                if len(content_text) > 200:
                    content_text = content_text[:200] + "..."
                summary_lines.append(f"{role}: {content_text}")

        summary_text: str = " | ".join(summary_lines)
        
        if len(summary_text) > max_chars:
            summary_text = "..." + summary_text[-(max_chars - 3):]
        
        return summary_text

    # -------------------- Helper: Parse model response --------------------
    @staticmethod
    def parse_model_response(content: str) -> List[str]:
        """
        Extract tool IDs from the model's response.
        Handles various formats including JSON arrays and lists.
        """
        selected_tool_ids: List[str] = []
        
        if not content or not isinstance(content, str):
            return selected_tool_ids

        content = content.strip()
        
        # Try to find JSON array patterns
        matches: List[str] = re.findall(r'\[[\s\S]*?\]', content)
        
        for raw_list in matches:
            try:
                parsed = json.loads(raw_list)
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if item]
            except json.JSONDecodeError:
                pass
            
            try:
                parsed = ast.literal_eval(raw_list)
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if item]
            except (ValueError, SyntaxError):
                pass
        
        # If no valid array found, try to extract quoted strings
        quoted_matches = re.findall(r'["\']([^"\']+)["\']', content)
        if quoted_matches:
            return [match.strip() for match in quoted_matches]
        
        return selected_tool_ids

    # -------------------- Helper: Filter similar tools --------------------
    @staticmethod
    def filter_similar_tools(
        tool_ids: List[str],
        available_tools: List[Dict[str, str]],
        threshold: float = 0.8,
    ) -> List[str]:
        """
        Remove tools with very similar names/descriptions to avoid redundancy.
        """
        if not tool_ids or threshold >= 1.0:
            return tool_ids

        filtered: List[str] = []
        
        tool_lookup: Dict[str, str] = {
            t["id"]: (t.get("description", "") or t["id"]).lower()
            for t in available_tools
        }
        
        seen_descriptions: List[str] = []

        for tid in tool_ids:
            description = tool_lookup.get(tid, tid.lower())
            
            is_similar = any(
                SequenceMatcher(None, description, seen).ratio() >= threshold
                for seen in seen_descriptions
            )
            
            if not is_similar:
                filtered.append(tid)
                seen_descriptions.append(description)
            elif threshold > 0.9:
                filtered.append(tid)

        return filtered

    # -------------------- Main inlet function --------------------
    async def inlet(
        self,
        body: Dict[str, Any],
        __event_emitter__: Callable[[Dict[str, Any]], Awaitable[None]],
        __request__: Any,
        __user__: Optional[Dict[str, Any]] = None,
        __model__: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Main filter function that processes incoming requests and selects appropriate tools.
        Uses expert model (Expert Model → Task Model → Current Model) for tool selection.
        """
        try:
            self.s_key = self.valves.status_messages if self.valves.status_messages else "simple"
            messages: List[Dict[str, Any]] = body.get("messages", [])
            user_message: Optional[str] = get_last_user_message(messages)
            
            logger.info("=" * 60)
            logger.info("Starting Auto Tool Selection (Enhanced v2.2.0)")
            logger.info(f"    └ Tone: {self.s_key}")
            
            if self.valves.debug:
                logger.info(f"    └ Body: {json.dumps(body)}")
            
            if not user_message:
                if self.valves.debug:
                    logger.info("No user message found, skipping tool selection")
                return body

            await self.emit_status(
                __event_emitter__, 
                "start", 
                STATUS_MESSAGES.get("analyse_query",{}).get(self.s_key)
            )

            # Step 1: Prepare available tools
            available_tools: List[Dict[str, str]] = self.prepare_tools(__model__)
            
            if not available_tools:
                status_message = STATUS_MESSAGES.get("no_tool_available",{}).get(self.s_key)
                if self.valves.debug:
                    logger.warning("No tools available for selection")
                await self.emit_status(
                    __event_emitter__,
                    "done",
                    status_message,
                    done=True
                )
                return body

            tool_descriptions: str = "\n".join([
                f"- {t['id']} ({t.get('type', 'unknown')}): {t['description']}"
                for t in available_tools
            ])

            # Step 2: Summarize conversation history
            summary_history: str = self.summarize_history(
                messages, self.valves.max_history_chars
            )

            if self.valves.debug:
                logger.info(f"History summary ({len(summary_history)} chars): {summary_history[:200]}...")

            # Get model information
            model_id: Optional[str] = body.get("model")
            models: Dict[str, Any] = getattr(__request__.app.state, "MODELS", {})
            
            current_model_info = models.get(model_id, {})
            
            base_model_id = model_id
            is_workspace_model = False
            
            if current_model_info:
                info = current_model_info.get("info", {})
                if info and "base_model_id" in info:
                    base_model_id = info["base_model_id"]
                    is_workspace_model = True
                    logger.info(f"Detected workspace model '{model_id}' wrapping base model '{base_model_id}'")
                elif info and "id" in info and info["id"] != model_id:
                    base_model_id = info["id"]
                    is_workspace_model = True
                    logger.info(f"Detected workspace model '{model_id}' with base model '{base_model_id}'")
            
            logger.info(f"Current model: {model_id}, Base model: {base_model_id}, Is workspace: {is_workspace_model}")

            # Step 3: Query Expert Model for tool selection
            await self.emit_status(
                __event_emitter__, 
                "query_expert_model", 
                STATUS_MESSAGES.get("query_expert_model",{}).get(self.s_key)
            )
            
            logger.info("Querying expert model for tool selection")

            # Apply template
            system_prompt = self.valves.template.replace("{{TOOLS}}", tool_descriptions)

            # Get expert model configuration
            api_url, expert_model_id, provider = self.get_expert_model_config(
                __request__, models, model_id, base_model_id, is_workspace_model
            )

            # Get prepare_tools status message
            msg_template = STATUS_MESSAGES.get("prepare_tools",{}).get(self.s_key)
            if msg_template:
                try:
                    status_message = msg_template.substitute(count=len(available_tools))
                except KeyError:
                    status_message = msg_template.substitute()
            else:
                status_message = f"Found {len(available_tools)} available tools"
                
            await self.emit_status(
                __event_emitter__,
                "prepare_tools",
                status_message
            )

            # If no expert model is available, return without tools
            selected_tool_ids: List[str] = []
            if api_url:
                expert_prompt = f"""Conversation Context:
{summary_history}

Current User Query:
{user_message}

Task: Select all relevant tools and features that would help address this query.

Guidelines:
- Select tools that directly address the user's request
- Consider the context from conversation history
- You can select multiple tools if they work together
- Built-in features (web_search, image_generation, code_interpreter) should be selected when appropriate
- Custom tools should be selected based on their descriptions
- Avoid selecting tools with nearly identical functionality
- If unsure, lean towards selecting tools that might be helpful

Return ONLY a valid JSON array of tool/feature IDs. No explanation, no markdown, just the array.
Examples: ["web_search_and_crawl", "image_generation", "chat_history_search"] or [] if no tools match.
Important: Return an empty array [] if no tools are relevant."""

                try:
                    selected_tool_ids, response_data = await self.query_model_for_tools(
                        api_url=api_url,
                        api_model_id=expert_model_id,
                        system_prompt=system_prompt,
                        user_prompt=expert_prompt,
                        api_key=self.valves.expert_model_api_key.strip() if self.valves.expert_model_api_key else None,
                        provider=provider,
                    )
                    logger.info(f"Expert model suggested tools: {selected_tool_ids}")
                except Exception as e:
                    logger.error(f"Error querying expert model for tool suggestions: {e}")
                    logger.exception(f"Traceback:\n{traceback.format_exc()}")
                    if response_data and "error" in response_data:
                        error_msg = response_data["error"].get("message", "Unknown error")
                        logger.error(f"Expert model API error: {error_msg}")
                    selected_tool_ids = []
            else:
                logger.warning("No expert model available, proceeding without tool selection")

            # Step 4: Validate selected tools exist in available tools
            valid_ids_set = {t["id"].lower() for t in available_tools}
            valid_id_lookup = {t["id"].lower(): t["id"] for t in available_tools}

            selected_tool_ids = [
                valid_id_lookup.get(tid.lower(), tid)
                for tid in selected_tool_ids
                if tid.lower() in valid_ids_set
            ]

            # Add forced tools if configured
            if self.valves.force_custom_tools:
                forced_tools = [tool_id.strip() for tool_id in self.valves.force_custom_tools.split(",")]
                if self.valves.debug:
                    logger.info(f"Adding forced tool IDs: {forced_tools}")
                for tool_id in forced_tools:
                    if tool_id.lower() in valid_ids_set:
                        original_id = valid_id_lookup[tool_id.lower()]
                        if original_id not in selected_tool_ids:
                            selected_tool_ids.append(original_id)
                    else:
                        logger.warning(f"Force tool '{tool_id}' not found in available tools, ignoring")

            # Filter out similar tools
            if self.valves.similarity_threshold < 1.0:
                original_count = len(selected_tool_ids)
                selected_tool_ids = self.filter_similar_tools(
                    selected_tool_ids, available_tools, self.valves.similarity_threshold
                )
                if self.valves.debug and len(selected_tool_ids) < original_count:
                    logger.info(f"Filtered {original_count - len(selected_tool_ids)} similar tools")

            # Update body with selected tools
            if selected_tool_ids:
                builtin_features = [tid for tid in selected_tool_ids 
                                if tid in ["web_search", "image_generation", "code_interpreter"]]
                custom_tools = [tid for tid in selected_tool_ids 
                                if tid not in builtin_features]

                if builtin_features:
                    features: Dict[str, bool] = body.get("features", {})
                    for feature_id in builtin_features:
                        features[feature_id] = True
                    body["features"] = features

                logger.info(f"Instructing use of tools: {custom_tools}")
                if custom_tools:
                    body["tool_ids"] = custom_tools

                feature_msg = f"{len(builtin_features)} feature{'s' if len(builtin_features) > 1 else ''}" if builtin_features else ""
                tool_msg = f"{len(custom_tools)} tool{'s' if len(custom_tools) > 1 else ''}" if custom_tools else ""
                parts = " and ".join([p for p in [feature_msg, tool_msg] if p])

                msg_template = STATUS_MESSAGES.get("features_tools_msg",{}).get(self.s_key)
                if msg_template:
                    try:
                        status_desc = msg_template.safe_substitute(parts=parts, tool_ids=", ".join(selected_tool_ids), count=len(selected_tool_ids))
                    except Exception as e:
                        if self.valves.debug:
                            logger.error(f"Error formatting status message: {e}")
                        status_desc = f"Selected {len(selected_tool_ids)} tool(s)"
                else:
                    status_desc = f"Selected {len(selected_tool_ids)} tool(s)"
                
                await self.emit_status(
                    __event_emitter__,
                    "done",
                    status_desc,
                    done=True,
                )
            else:
                await self.emit_status(
                    __event_emitter__,
                    "done",
                    STATUS_MESSAGES.get("no_tool_found",{}).get(self.s_key),
                    done=True,
                )

            logger.info("=" * 60)
            logger.info("Auto Tool Selection Complete")
            logger.info(f"    └ Final selected tools: {selected_tool_ids}")
            logger.info("=" * 60)

        except Exception as e:
            logger.exception(f"Error in AutoToolSelector.inlet: {e}")
            await self.emit_status(
                __event_emitter__,
                "error",
                f"Error selecting tools: {str(e)}",
                done=True,
            )

        return body

    # -------------------- Helper: Query model for tool selection --------------------
    async def query_model_for_tools(
        self,
        api_url: str,
        api_model_id: str,
        system_prompt: str,
        user_prompt: str,
        api_key: Optional[str] = None,
        provider: str = "openai",
    ) -> tuple[List[str], Optional[Dict[str, Any]]]:
        """
        Query a model for tool selection and return parsed tool IDs.
        """
        import aiohttp

        if provider == "ollama":
            chat_payload = {
                "model": api_model_id,
                "prompt": f"System: {system_prompt}\n\nUser: {user_prompt}",
                "stream": False,
                "options": {"temperature": 0.1}
            }
        else:
            chat_payload = {
                "model": api_model_id,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "temperature": 0.1,
            }

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        async with aiohttp.ClientSession() as session:
            async with session.post(
                api_url, 
                json=chat_payload, 
                headers=headers, 
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Model API call failed with status {response.status}: {error_text}")
                    raise Exception(f"Model API call failed with status {response.status}")

                response_data = await response.json()

                if provider == "ollama":
                    content = response_data.get("response", "")
                else:
                    choices = response_data.get("choices", [])
                    if choices:
                        content = choices[0].get("message", {}).get("content", "")
                    else:
                        content = ""

                logger.info(f"Model API call to {api_url} succeeded")
                if self.valves.debug:
                    logger.debug(f"Model response content: {content}")

                return self.parse_model_response(content), response_data

    # -------------------- Helper: Determine expert model configuration --------------------
    def get_expert_model_config(
        self,
        __request__: Any,
        models: Dict[str, Any],
        model_id: str,
        base_model_id: str,
        is_workspace_model: bool,
    ) -> tuple[Optional[str], Optional[str], str]:
        """
        Determine the expert model configuration using priority fallback.
        """
        # Check if expert model is configured (Priority 1)
        if self.valves.expert_model_url and self.valves.expert_model_url.strip():
            expert_url = self.valves.expert_model_url.strip().rstrip('/')
            if not expert_url.endswith('/v1'):
                expert_url = expert_url.rstrip('/') + '/v1'
            expert_url = f"{expert_url}/chat/completions"
            
            expert_model = self.valves.expert_model_id.strip() if self.valves.expert_model_id.strip() else "expert-model"
            
            logger.info(f"Using Expert Model (Priority 1): {expert_url} with model {expert_model}")
            return expert_url, expert_model, "openai"

        # Fall back to Open WebUI task model config (Priority 2)
        logger.info("Expert Model not configured, falling back to task model")

        try:
            task_model_config_local = getattr(__request__.app.state.config, "TASK_MODEL", None)
            task_model_config_external = getattr(__request__.app.state.config, "TASK_MODEL_EXTERNAL", None)
        except AttributeError as e:
            logger.warning(f"Could not access task model config: {e}")
            task_model_config_local = None
            task_model_config_external = None

        logger.info(f"Task model configs - Local: '{task_model_config_local}', External: '{task_model_config_external}'")

        task_model_id = None

        # PRIORITY 1: Local task model
        if not task_model_id and task_model_config_local:
            if task_model_config_local in models:
                task_model_id = task_model_config_local
                logger.info(f"Priority 1: Using configured LOCAL task model: {task_model_id}")
            else:
                logger.warning(f"Priority 1: Local task model '{task_model_config_local}' not found, moving to next priority")

        # PRIORITY 2: External task model
        if not task_model_id and task_model_config_external:
            if task_model_config_external in models:
                task_model_id = task_model_config_external
                logger.info(f"Priority 2: Using configured EXTERNAL task model: {task_model_id}")
            else:
                logger.warning(f"Priority 2: External task model '{task_model_config_external}' not found, moving to next priority")

        # PRIORITY 3: Workspace base model
        if not task_model_id and is_workspace_model:
            if base_model_id and base_model_id in models and base_model_id != model_id:
                task_model_id = base_model_id
                logger.info(f"Priority 3: Using workspace base model: {task_model_id}")
            else:
                logger.warning(f"Priority 3: Workspace base model '{base_model_id}' not available, moving to next priority")

        # PRIORITY 4: Current base model (final fallback)
        if not task_model_id:
            if base_model_id and base_model_id in models:
                task_model_id = base_model_id
                logger.info(f"Priority 4: Using current base model as fallback: {task_model_id}")
            else:
                logger.error(f"Priority 4: Current base model '{base_model_id}' not available")

        if not task_model_id or task_model_id not in models:
            logger.error("No valid expert model found after checking all priorities")
            return None, None, "openai"

        # Determine provider and API URL
        model_info = models.get(task_model_id, {})
        provider = "openai"
        base_url = "http://localhost:8080/v1"

        if isinstance(model_info, dict):
            if model_info.get('ollama'):
                provider = "ollama"
                base_url = "http://localhost:11434"
            elif model_info.get('openai'):
                openai_config = model_info.get('openai')
                if isinstance(openai_config, dict):
                    base_url = openai_config.get('base_url', base_url)
                else:
                    base_url = openai_config if isinstance(openai_config, str) else base_url
        else:
            if hasattr(model_info, 'ollama') and getattr(model_info, 'ollama', None):
                provider = "ollama"
                base_url = "http://localhost:11434"
            elif hasattr(model_info, 'openai'):
                openai_config = getattr(model_info, 'openai', None)
                if openai_config and hasattr(openai_config, 'get'):
                    base_url = openai_config.get('base_url', base_url)

        model_id_lower = task_model_id.lower()
        if provider == "openai":
            if ':' in task_model_id and not 'gpt' in model_id_lower and not 'claude' in model_id_lower:
                provider = "ollama"
                base_url = "http://localhost:11434"
            elif any(x in model_id_lower for x in ['ollama', 'llama', 'mistral', 'codellama', 'dolphin']):
                provider = "ollama"
                base_url = "http://localhost:11434"

        base_url = base_url.rstrip('/')
        if provider == "ollama":
            api_url = f"{base_url}/api/generate"
        else:
            if not base_url.endswith('/v1'):
                api_url = f"{base_url}/v1/chat/completions"
            else:
                api_url = f"{base_url}/chat/completions"

        logger.info(f"Expert model: {task_model_id}")
        logger.info(f"Expert model provider: {provider}, API URL: {api_url}")

        return api_url, task_model_id, provider
