"""
Chat History Search - Search through your past conversations
Title: Chat History Search
Author: lexiismadd
Version: 2.0.1
Required Open WebUI Version: 0.6.3
"""

from pydantic import BaseModel, Field
from typing import Optional, Callable, Awaitable
import requests
import json
import os
import sys
from loguru import logger

# Remove default handler - will be reconfigured in __init__ based on valves
logger.remove()


class Tools:
    class Valves(BaseModel):
        QDRANT_HOST: str = Field(
            default="http://qdrant:6333",
            description="Qdrant host URL (use container name if in same network)",
        )
        QDRANT_API_KEY: str = Field(
            default="",
            description="Qdrant API key for authentication (leave empty if no auth required)",
        )
        COLLECTION_NAME: str = Field(
            default="documents",
            description="Qdrant collection name (usually 'documents' in Open WebUI)",
        )
        EMBEDDING_ENGINE: str = Field(
            default="",
            description="Embedding engine: 'ollama', 'openai', or leave empty to auto-detect from Open WebUI",
        )
        EMBEDDING_MODEL: str = Field(
            default="",
            description="Embedding model name (leave empty to auto-detect from Open WebUI)",
        )
        EMBEDDING_API_BASE_URL: str = Field(
            default="",
            description="API base URL for embeddings (leave empty to auto-detect from Open WebUI or use Ollama)",
        )
        EMBEDDING_API_KEY: str = Field(
            default="",
            description="API key for embedding service (leave empty if not required or to use Open WebUI config)",
        )
        OLLAMA_HOST: str = Field(
            default="http://ollama:11434",
            description="Ollama host URL (used as fallback if other settings not configured)",
        )
        LOG_LEVEL: str = Field(
            default="INFO",
            description="Logging level: DEBUG, INFO, WARNING, ERROR",
        )
        CONSOLE_LOG_LEVEL: str = Field(
            default="INFO",
            description="Console logging level (separate from file): DEBUG, INFO, WARNING, ERROR",
        )

    def __init__(self):
        self.valves = self.Valves()
        self._embedding_config = None
        self._logger_configured = False
        self._configure_logging()
        
        logger.info("Chat History Search tool initialized")
        logger.debug(
            f"Valves configuration: QDRANT_HOST={self.valves.QDRANT_HOST}, "
            f"COLLECTION_NAME={self.valves.COLLECTION_NAME}, "
            f"EMBEDDING_ENGINE={self.valves.EMBEDDING_ENGINE or 'auto-detect'}, "
            f"EMBEDDING_MODEL={self.valves.EMBEDDING_MODEL or 'auto-detect'}, "
            f"EMBEDDING_API_BASE_URL={self.valves.EMBEDDING_API_BASE_URL or 'auto-detect'}, "
            f"OLLAMA_HOST={self.valves.OLLAMA_HOST}, "
            f"QDRANT_API_KEY_SET={bool(self.valves.QDRANT_API_KEY)}, "
            f"EMBEDDING_API_KEY_SET={bool(self.valves.EMBEDDING_API_KEY)}, "
            f"LOG_LEVEL={self.valves.LOG_LEVEL}, "
            f"CONSOLE_LOG_LEVEL={self.valves.CONSOLE_LOG_LEVEL}"
        )
    
    def _configure_logging(self):
        """Configure logging based on valve settings"""
        if self._logger_configured:
            return
        
        # Validate log levels
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
        file_level = self.valves.LOG_LEVEL.upper() if self.valves.LOG_LEVEL.upper() in valid_levels else "INFO"
        console_level = self.valves.CONSOLE_LOG_LEVEL.upper() if self.valves.CONSOLE_LOG_LEVEL.upper() in valid_levels else "INFO"
        
        # Configure console logging
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=console_level,
            filter=lambda record: record["extra"].get("tool") == "chat_history_search",
        )
        
        # Configure file logging
        try:
            logger.add(
                "/app/backend/data/logs/chat_history_search.log",
                rotation="10 MB",
                retention="7 days",
                level=file_level,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
                filter=lambda record: record["extra"].get("tool") == "chat_history_search",
            )
        except Exception as e:
            # Fallback if file logging fails (e.g., permission issues)
            print(f"Warning: Could not set up file logging: {e}")
        
        self._logger_configured = True
        
        # Bind tool identifier to all logs from this instance
        logger.configure(extra={"tool": "chat_history_search"})

    async def search_past_conversations(
        self,
        query: str,
        limit: int = 5,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> str:
        """
        Search through your past chat conversations using semantic search.

        :param query: What to search for in your chat history
        :param limit: Maximum number of results to return (default: 5)
        :return: Relevant messages from your past conversations
        """
        user_id = __user__.get("id") if __user__ else "anonymous"
        logger.info(f"Search requested by user_id={user_id}, query='{query}', limit={limit}")

        try:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Searching conversations for: {query}",
                            "done": False,
                        },
                    }
                )

            # Get user ID for filtering
            user_id = __user__.get("id") if __user__ else None
            logger.debug(f"User ID for filtering: {user_id}")

            # Generate embedding for the query
            logger.info("Generating embedding for search query")
            embedding = await self._generate_embedding(query, __user__)

            if not embedding:
                error_msg = "Unable to generate search embedding. Check embedding configuration."
                logger.error(error_msg)
                return f"❌ {error_msg}"

            logger.info(f"Successfully generated embedding vector of length {len(embedding)}")

            # Search Qdrant
            search_url = f"{self.valves.QDRANT_HOST}/collections/{self.valves.COLLECTION_NAME}/points/search"
            logger.debug(f"Querying Qdrant at {search_url}")

            search_payload = {
                "vector": embedding,
                "limit": limit * 2,  # Get more results to filter
                "with_payload": True,
                "with_vector": False,
            }

            # Add user filter if available
            if user_id:
                search_payload["filter"] = {
                    "must": [{"key": "user_id", "match": {"value": user_id}}]
                }
                logger.debug(f"Applied user filter for user_id={user_id}")

            logger.debug(f"Search payload: limit={search_payload['limit']}, with_filter={bool(user_id)}")

            # Prepare headers with API key if provided
            headers = {"Content-Type": "application/json"}
            if self.valves.QDRANT_API_KEY:
                headers["api-key"] = self.valves.QDRANT_API_KEY
                logger.debug("Using API key for Qdrant authentication")
            else:
                logger.debug("No API key provided for Qdrant (connecting without auth)")

            response = requests.post(
                search_url,
                json=search_payload,
                headers=headers,
                timeout=10,
            )

            if response.status_code != 200:
                error_msg = f"Qdrant search failed with status {response.status_code}: {response.text}"
                logger.error(error_msg)
                return f"❌ Search error: {response.status_code}"

            results = response.json().get("result", [])
            logger.info(f"Qdrant returned {len(results)} results")

            if not results:
                logger.warning("No relevant conversations found in Qdrant")
                return "📭 No relevant conversations found."

            # Format results
            formatted_results = []
            for idx, result in enumerate(results[:limit], 1):
                payload = result.get("payload", {})
                score = result.get("score", 0)
                logger.debug(f"Processing result {idx}: score={score:.4f}")

                # Extract content (Open WebUI may store in different fields)
                content = self._extract_content(payload)

                if content:
                    formatted_results.append(
                        f"**Result {idx}** (Relevance: {score:.2f})\n{content}"
                    )
                    logger.debug(f"Result {idx}: Added content snippet (length={len(content)})")
                else:
                    logger.debug(f"Result {idx}: No content found in payload")

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Found {len(formatted_results)} relevant messages",
                            "done": True,
                        },
                    }
                )

            if not formatted_results:
                logger.warning("No text content found in search results after processing")
                return "📭 No relevant text content found in search results."

            logger.info(f"Successfully formatted {len(formatted_results)} results for user")
            return "🔍 **Search Results:**\n\n" + "\n\n---\n\n".join(formatted_results)

        except requests.exceptions.Timeout as e:
            error_msg = f"Request timeout: {str(e)}"
            logger.error(error_msg)
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": f"❌ {error_msg}", "done": True}}
                )
            return f"❌ {error_msg}"
        except requests.exceptions.ConnectionError as e:
            error_msg = f"Connection error: {str(e)}"
            logger.error(error_msg)
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": f"❌ {error_msg}", "done": True}}
                )
            return f"❌ {error_msg}"
        except Exception as e:
            error_msg = f"Error searching conversations: {str(e)}"
            logger.exception(error_msg)  # This will include the full traceback
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": f"❌ {error_msg}", "done": True}}
                )
            return f"❌ {error_msg}"

    async def _get_embedding_config(self, user: Optional[dict] = None) -> Optional[dict]:
        """
        Retrieve embedding configuration from Open WebUI.
        Priority: Valve overrides > Environment variables > Fallback defaults
        """
        if self._embedding_config:
            logger.debug("Using cached embedding configuration")
            return self._embedding_config

        try:
            logger.info("Retrieving embedding configuration")
            
            # Method 1: Check if user provided complete override in valves
            if self.valves.EMBEDDING_ENGINE and self.valves.EMBEDDING_MODEL:
                config = {
                    "engine": self.valves.EMBEDDING_ENGINE.lower(),
                    "model": self.valves.EMBEDDING_MODEL,
                    "api_base_url": self.valves.EMBEDDING_API_BASE_URL or self.valves.OLLAMA_HOST,
                    "api_key": self.valves.EMBEDDING_API_KEY or "",
                }
                logger.info(
                    f"Using complete valve override: engine={config['engine']}, "
                    f"model={config['model']}, url={config['api_base_url']}, "
                    f"api_key_set={bool(config['api_key'])}"
                )
                self._embedding_config = config
                return config
            
            # Method 1b: Partial valve override (only model specified)
            if self.valves.EMBEDDING_MODEL:
                # Determine engine based on API URL or default to Ollama
                engine = "ollama"
                api_base_url = self.valves.EMBEDDING_API_BASE_URL or self.valves.OLLAMA_HOST
                
                if self.valves.EMBEDDING_API_BASE_URL:
                    # If custom API URL is provided, assume OpenAI-compatible
                    if "ollama" not in api_base_url.lower():
                        engine = "openai"
                
                config = {
                    "engine": engine,
                    "model": self.valves.EMBEDDING_MODEL,
                    "api_base_url": api_base_url,
                    "api_key": self.valves.EMBEDDING_API_KEY or "",
                }
                logger.info(
                    f"Using partial valve override: engine={config['engine']}, "
                    f"model={config['model']}, url={config['api_base_url']}, "
                    f"api_key_set={bool(config['api_key'])}"
                )
                self._embedding_config = config
                return config

            # Method 2: Try to get from Open WebUI environment variables
            # These are available when tools run in the same process as Open WebUI
            rag_engine = os.getenv("RAG_EMBEDDING_ENGINE", "")
            rag_model = os.getenv("RAG_EMBEDDING_MODEL", "")
            rag_api_base = os.getenv("RAG_EMBEDDING_OPENAI_API_BASE_URL", "")
            rag_api_key = os.getenv("RAG_EMBEDDING_OPENAI_API_KEY", "")
            
            logger.debug(
                f"Environment variables: RAG_EMBEDDING_ENGINE={rag_engine}, "
                f"RAG_EMBEDDING_MODEL={rag_model}, "
                f"RAG_EMBEDDING_OPENAI_API_BASE_URL={rag_api_base}"
            )
            
            # Also check for Ollama base URL
            ollama_base_url = os.getenv("OLLAMA_BASE_URL", "")
            if ollama_base_url and ";" in ollama_base_url:
                # Handle multiple Ollama URLs, take the first one
                original_url = ollama_base_url
                ollama_base_url = ollama_base_url.split(";")[0]
                logger.debug(f"Multiple Ollama URLs detected, using first: {ollama_base_url} (from {original_url})")

            if rag_model:
                config = {
                    "engine": rag_engine.lower() if rag_engine else ("openai" if rag_api_base else "ollama"),
                    "model": rag_model,
                    "api_base_url": rag_api_base or ollama_base_url or self.valves.OLLAMA_HOST,
                    "api_key": rag_api_key or "",
                }
                logger.info(
                    f"Using environment configuration: engine={config['engine']}, "
                    f"model={config['model']}, url={config['api_base_url']}, "
                    f"api_key_set={bool(config['api_key'])}"
                )
                self._embedding_config = config
                return config

            # Method 3: Final fallback - use Ollama with default model
            default_config = {
                "engine": "ollama",
                "model": "nomic-embed-text",
                "api_base_url": ollama_base_url or self.valves.OLLAMA_HOST,
                "api_key": "",
            }
            
            logger.warning(
                f"No embedding configuration found, using fallback: "
                f"model={default_config['model']}, url={default_config['api_base_url']}"
            )
            return default_config

        except Exception as e:
            logger.error(f"Error getting embedding config: {e}")
            logger.exception("Full traceback:")
            # Return safe default
            default_config = {
                "engine": "ollama",
                "model": "nomic-embed-text",
                "api_base_url": self.valves.OLLAMA_HOST,
                "api_key": "",
            }
            logger.warning(f"Returning safe default configuration: {default_config}")
            return default_config

    async def _generate_embedding(self, text: str, user: Optional[dict] = None) -> Optional[list]:
        """Generate embedding vector for text using Open WebUI's configured embedding service"""
        try:
            logger.debug(f"Generating embedding for text of length {len(text)}")
            config = await self._get_embedding_config(user)
            
            if not config:
                logger.error("Failed to retrieve embedding configuration")
                return None

            engine = config.get("engine", "").lower()
            model = config.get("model", "")
            api_base_url = config.get("api_base_url", "")
            api_key = config.get("api_key", "")

            logger.info(f"Using embedding engine: {engine}, model: {model}, url: {api_base_url}")

            # Handle different embedding engines
            if engine == "ollama" or (not engine and "ollama" in api_base_url.lower()):
                logger.debug("Routing to Ollama embedding generation")
                return await self._generate_ollama_embedding(text, model, api_base_url)
            
            elif engine in ["openai", ""] and api_base_url:
                logger.debug("Routing to OpenAI-compatible embedding generation")
                return await self._generate_openai_compatible_embedding(
                    text, model, api_base_url, api_key
                )
            
            else:
                # Fallback to sentence-transformers
                logger.warning(f"Unknown engine '{engine}', attempting sentence-transformers fallback")
                return await self._generate_sentence_transformer_embedding(text, model)

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            logger.exception("Full traceback:")
            return None

    async def _generate_ollama_embedding(
        self, text: str, model: str, api_base_url: str
    ) -> Optional[list]:
        """Generate embedding using Ollama"""
        try:
            if not api_base_url:
                api_base_url = "http://ollama:11434"
                logger.debug(f"No API base URL provided, using default: {api_base_url}")
            
            if not model:
                model = "nomic-embed-text"
                logger.debug(f"No model provided, using default: {model}")

            url = f"{api_base_url.rstrip('/')}/api/embeddings"
            
            logger.info(f"Requesting Ollama embedding from {url} with model {model}")
            logger.debug(f"Text length: {len(text)} characters")
            
            response = requests.post(
                url,
                json={"model": model, "prompt": text},
                headers={"Content-Type": "application/json"},
                timeout=30,
            )

            if response.status_code == 200:
                data = response.json()
                embedding = data.get("embedding")
                if embedding:
                    logger.info(f"Successfully generated Ollama embedding (vector length: {len(embedding)})")
                    return embedding
                else:
                    logger.error("Ollama response missing 'embedding' field")
                    logger.debug(f"Response data: {data}")
                    return None
            else:
                logger.error(f"Ollama embedding request failed: status={response.status_code}")
                logger.error(f"Response: {response.text}")
                return None

        except requests.exceptions.Timeout as e:
            logger.error(f"Ollama embedding request timed out: {e}")
            return None
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Failed to connect to Ollama at {api_base_url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Ollama embedding generation error: {e}")
            logger.exception("Full traceback:")
            return None

    async def _generate_openai_compatible_embedding(
        self, text: str, model: str, api_base_url: str, api_key: str
    ) -> Optional[list]:
        """Generate embedding using OpenAI-compatible API (OpenAI, Azure, etc.)"""
        try:
            if not api_base_url:
                logger.error("No API base URL provided for OpenAI-compatible embedding")
                return None
            
            if not model:
                model = "text-embedding-3-small"
                logger.debug(f"No model provided, using default: {model}")

            url = f"{api_base_url.rstrip('/')}/embeddings"
            
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
                logger.debug("Using API key for authentication")
            else:
                logger.warning("No API key provided for OpenAI-compatible API")

            logger.info(f"Requesting OpenAI-compatible embedding from {url} with model {model}")
            logger.debug(f"Text length: {len(text)} characters")

            response = requests.post(
                url,
                json={"model": model, "input": text},
                headers=headers,
                timeout=30,
            )

            if response.status_code == 200:
                data = response.json()
                if "data" in data and len(data["data"]) > 0:
                    embedding = data["data"][0].get("embedding")
                    if embedding:
                        logger.info(f"Successfully generated OpenAI-compatible embedding (vector length: {len(embedding)})")
                        return embedding
                    else:
                        logger.error("OpenAI-compatible response missing 'embedding' field")
                        logger.debug(f"Response data: {data}")
                        return None
                else:
                    logger.error("OpenAI-compatible response has unexpected format")
                    logger.debug(f"Response data: {data}")
                    return None
            else:
                logger.error(f"OpenAI-compatible API request failed: status={response.status_code}")
                logger.error(f"Response: {response.text}")
                return None

        except requests.exceptions.Timeout as e:
            logger.error(f"OpenAI-compatible API request timed out: {e}")
            return None
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Failed to connect to OpenAI-compatible API at {api_base_url}: {e}")
            return None
        except Exception as e:
            logger.error(f"OpenAI-compatible API embedding error: {e}")
            logger.exception("Full traceback:")
            return None

    async def _generate_sentence_transformer_embedding(
        self, text: str, model: str = None
    ) -> Optional[list]:
        """Generate embedding using local sentence-transformers (fallback)"""
        try:
            logger.info("Attempting to use sentence-transformers for embedding generation")
            from sentence_transformers import SentenceTransformer

            if not model or not model.startswith("sentence-transformers/"):
                model = "sentence-transformers/all-MiniLM-L6-v2"
                logger.debug(f"Using default sentence-transformer model: {model}")
            
            logger.info(f"Loading sentence-transformer model: {model}")
            st_model = SentenceTransformer(model)
            
            logger.debug(f"Encoding text of length {len(text)}")
            embedding = st_model.encode(text).tolist()
            
            logger.info(f"Successfully generated sentence-transformer embedding (vector length: {len(embedding)})")
            return embedding

        except ImportError:
            logger.error("sentence-transformers library is not installed")
            logger.info("Install with: pip install sentence-transformers")
            return None
        except Exception as e:
            logger.error(f"Sentence transformer embedding error: {e}")
            logger.exception("Full traceback:")
            return None

    def _extract_content(self, payload: dict) -> Optional[str]:
        """Extract content from payload (handles various Open WebUI formats)"""
        logger.debug(f"Extracting content from payload with keys: {list(payload.keys())}")
        
        # Try common content fields
        content = (
            payload.get("content")
            or payload.get("text")
            or payload.get("data", {}).get("content")
            or payload.get("message", {}).get("content")
        )

        if content and len(content.strip()) > 0:
            original_length = len(content)
            # Truncate if too long
            truncated = content[:500] + ("..." if len(content) > 500 else "")
            logger.debug(f"Extracted content: original_length={original_length}, truncated_length={len(truncated)}")
            return truncated
        
        logger.debug("No content found in payload")
        return None