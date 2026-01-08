"""
title: Home Assistant
author: lexiismadd
author_url: https://github.com/lexiismadd
funding_url: https://github.com/open-webui
version: 2.0.0
license: MIT
requirements: aiohttp, loguru
description: Home Assistant tool with smart area-prioritized entity detection and intelligent matching.
"""

import asyncio
import difflib
import json
import re
from typing import Any, Callable, Optional

import aiohttp
from loguru import logger
from pydantic import BaseModel, Field


class Tools:
    class Valves(BaseModel):
        HA_URL: str = Field(
            default="",
            description="Home Assistant URL (e.g., http://homeassistant.local:8123)",
        )
        HA_TOKEN: str = Field(
            default="",
            description="Home Assistant Long-Lived Access Token",
        )
        DISCOVER_DOMAINS: str = Field(
            default="light,switch,climate,cover,fan,lock,media_player,sensor,binary_sensor,weather,camera,vacuum,scene",
            description="Comma-separated list of Home Assistant domains to discover",
        )
        INCLUDED_ENTITIES: str = Field(
            default="",
            description="Comma-separated list of entity_id patterns to include (supports wildcards, e.g., 'light.living_room_*,switch.bedroom_*'). Leave empty to include all.",
        )
        EXCLUDED_ENTITIES: str = Field(
            default="",
            description="Comma-separated list of entity_id patterns to exclude (supports wildcards, e.g., 'sensor.*_battery,*_update')",
        )
        USE_WEBSOCKET: bool = Field(
            default=True,
            description="Use WebSocket API (required for entity/area detection)",
        )
        AREA_MATCH_THRESHOLD: float = Field(
            default=0.6,
            description="Minimum score (0-1) to accept an area-specific match. Lower = more permissive, higher = stricter.",
        )
        GLOBAL_FALLBACK_THRESHOLD: float = Field(
            default=0.5,
            description="Minimum score (0-1) for global search fallback to be considered a valid match.",
        )

    def __init__(self):
        self.valves = self.Valves()
        self._ws = None
        self._ws_message_id = 0
        self._entity_cache = None
        self._area_cache = None
        self._known_areas = None  # Cache for area names
        logger.info("Home Assistant tool initialized")

    # =========================================================================
    # WebSocket Connection Management
    # =========================================================================

    async def _get_ws_connection(self):
        """Get or create a WebSocket connection to Home Assistant."""
        if self.valves.USE_WEBSOCKET:
            if self._ws is None or self._ws.closed:
                ws_url = f"{self.valves.HA_URL.rstrip('/')}/api/websocket"
                async with aiohttp.ClientSession() as session:
                    self._ws = await session.ws_connect(ws_url)
                    # Authenticate
                    auth_msg = await self._ws.receive_json()
                    if auth_msg["type"] == "auth_required":
                        await self._ws.send_json({
                            "type": "auth",
                            "access_token": self.valves.HA_TOKEN
                        })
                        auth_result = await self._ws.receive_json()
                        if auth_result["type"] != "auth_ok":
                            raise Exception("WebSocket authentication failed")
        return self._ws

    async def _ws_send_command(self, command: dict, timeout: float = 10.0) -> dict:
        """Send a command via WebSocket and wait for response."""
        ws = await self._get_ws_connection()
        self._ws_message_id += 1
        command["id"] = self._ws_message_id
        await ws.send_json(command)
        
        # Wait for response with matching id
        while True:
            response = await asyncio.wait_for(ws.receive_json(), timeout=timeout)
            if response.get("id") == self._ws_message_id:
                return response
            # Handle event messages differently
            if response.get("type") == "event":
                continue

    async def close_ws(self):
        """Close WebSocket connection."""
        if self._ws and not self._ws.closed:
            await self._ws.close()
            self._ws = None

    # =========================================================================
    # REST API Helper
    # =========================================================================

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[dict] = None,
    ) -> Any:
        """Make an authenticated request to Home Assistant API."""
        if not self.valves.HA_URL or not self.valves.HA_TOKEN:
            raise ValueError("Home Assistant URL and Token must be configured in valves")

        url = f"{self.valves.HA_URL.rstrip('/')}/api/{endpoint.lstrip('/')}"
        headers = {
            "Authorization": f"Bearer {self.valves.HA_TOKEN}",
            "Content-Type": "application/json",
        }

        async with aiohttp.ClientSession() as session:
            try:
                if method.upper() == "GET":
                    async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as response:
                        response.raise_for_status()
                        return await response.json()
                elif method.upper() == "POST":
                    async with session.post(url, headers=headers, json=data, timeout=aiohttp.ClientTimeout(total=10)) as response:
                        response.raise_for_status()
                        return await response.json()
            except aiohttp.ClientError as e:
                raise Exception(f"Home Assistant API request failed: {str(e)}")

    # =========================================================================
    # Entity and Area Discovery
    # =========================================================================

    async def _fetch_all_entities(self) -> list[dict]:
        """Fetch all entities from Home Assistant via REST API."""
        states = await self._make_request("GET", "states")
        entities = self._filter_entities(states)
        return entities

    async def _fetch_areas_via_websocket(self) -> list[dict]:
        """Fetch areas via WebSocket for room/area detection."""
        try:
            result = await self._ws_send_command({"type": "config/area_registry/list"})
            if result.get("success"):
                return result.get("result", [])
        except Exception as e:
            logger.warning(f"Failed to fetch areas via WebSocket: {e}")
        return []

    async def _fetch_entity_registry_via_websocket(self) -> list[dict]:
        """Fetch entity registry via WebSocket to get area associations."""
        try:
            result = await self._ws_send_command({"type": "config/entity_registry/list"})
            if result.get("success"):
                return result.get("result", [])
        except Exception as e:
            logger.warning(f"Failed to fetch entity registry via WebSocket: {e}")
        return []

    async def _build_entity_map(self) -> dict:
        """Build a comprehensive map of entities with area info from all sources."""
        if self._entity_cache is not None:
            return self._entity_cache

        # Fetch entities via REST
        entities = await self._fetch_all_entities()

        # If using WebSocket, enrich with area info
        area_info = {}
        entity_registry = []
        
        if self.valves.USE_WEBSOCKET:
            try:
                # Fetch areas
                areas = await self._fetch_areas_via_websocket()
                area_dict = {a["area_id"]: a["name"] for a in areas}
                
                # Fetch entity registry for area mappings
                entity_registry = await self._fetch_entity_registry_via_websocket()
                
                # Build area lookup from entity registry
                for entry in entity_registry:
                    if "area_id" in entry and entry["area_id"]:
                        area_info[entry["entity_id"]] = {
                            "area_id": entry["area_id"],
                            "area_name": area_dict.get(entry["area_id"], entry["area_id"]),
                            "device_id": entry.get("device_id"),
                            "original_name": entry.get("name", ""),
                        }
            except Exception as e:
                logger.warning(f"WebSocket enrichment failed: {e}")

        # Build entity map
        entity_map = {}
        for entity in entities:
            entity_id = entity.get("entity_id", "")
            attributes = entity.get("attributes", {})
            
            # Get area info from multiple sources
            area_from_attr = attributes.get("area_name", "")
            area_from_registry = area_info.get(entity_id, {})
            
            # Prefer WebSocket registry data, fall back to attributes
            area_name = area_from_registry.get("area_name") or area_from_attr
            area_id = area_from_registry.get("area_id")
            
            entity_map[entity_id] = {
                "entity": entity,
                "area_id": area_id,
                "area_name": area_name,
                "friendly_name": attributes.get("friendly_name", entity_id),
                "device_class": attributes.get("device_class"),
                "domain": entity_id.split(".")[0] if "." in entity_id else "",
                "entity_name": entity_id.split(".")[1] if "." in entity_id else entity_id,
            }

        self._entity_cache = entity_map
        return entity_map

    async def _get_known_areas(self) -> dict:
        """Get dictionary of known areas (name -> area_id)."""
        if self._known_areas is not None:
            return self._known_areas

        entity_map = await self._build_entity_map()
        
        # Build area list from entity map
        areas = {}
        for entity_id, entity_info in entity_map.items():
            area_name = entity_info["area_name"]
            area_id = entity_info["area_id"]
            if area_name and area_name not in areas:
                areas[area_name.lower()] = {
                    "area_id": area_id,
                    "area_name": area_name,
                }

        # Also try to fetch areas via WebSocket for comprehensive list
        if self.valves.USE_WEBSOCKET:
            try:
                ws_areas = await self._fetch_areas_via_websocket()
                for area in ws_areas:
                    area_name = area.get("name", "")
                    area_id = area.get("area_id", "")
                    if area_name and area_name.lower() not in areas:
                        areas[area_name.lower()] = {
                            "area_id": area_id,
                            "area_name": area_name,
                        }
            except Exception as e:
                logger.warning(f"Could not fetch additional areas: {e}")

        self._known_areas = areas
        return areas

    def _invalidate_cache(self):
        """Invalidate entity cache to force refresh."""
        self._entity_cache = None
        self._area_cache = None
        self._known_areas = None
        self._entity_words_cache = None

    # =========================================================================
    # Entity Filtering
    # =========================================================================

    def _matches_pattern(self, entity_id: str, patterns: list[str]) -> bool:
        """Check if entity_id matches any of the patterns (supports wildcards)."""
        if not patterns:
            return False
        return any(fnmatch.fnmatch(entity_id, pattern.strip()) for pattern in patterns)

    def _filter_entities(self, entities: list[dict]) -> list[dict]:
        """Filter entities based on domain, included, and excluded patterns."""
        domains = [d.strip() for d in self.valves.DISCOVER_DOMAINS.split(",") if d.strip()]
        included = [p.strip() for p in self.valves.INCLUDED_ENTITIES.split(",") if p.strip()]
        excluded = [p.strip() for p in self.valves.EXCLUDED_ENTITIES.split(",") if p.strip()]

        filtered = []
        for entity in entities:
            entity_id = entity.get("entity_id", "")
            domain = entity_id.split(".")[0] if "." in entity_id else ""

            if domain not in domains:
                continue
            if excluded and self._matches_pattern(entity_id, excluded):
                continue
            if included and not self._matches_pattern(entity_id, included):
                continue

            filtered.append(entity)

        return filtered

    # =========================================================================
    # Area Detection from Query
    # =========================================================================

    def _extract_potential_area_mentions(self, query: str) -> list[str]:
        """
        Extract potential area mentions from the query.
        
        Looks for:
        - Preposition patterns: "in the X", "at the X", "to the X"
        - Standalone area names that might be room references
        
        Returns list of potential area names (lowercase).
        """
        query_lower = query.lower()
        
        # Patterns that indicate an area specification
        area_patterns = [
            r"\b(?:in|at|to|for|on|under|over|near)\s+(?:the\s+)?([a-z\s]+?)(?:\s+(?:light|lights|fan|temperature|thermostat|switch|sensor|climate|device|them|it|that|this))?(?:\s|$)",
            r"\b(?:the\s+)?([a-z]+?)\s+(?:light|lights|fan|temperature|thermostat|switch|sensor|climate|device)\b",
            r"\b(?:upstairs|downstairs|ground\s*floor|first\s*floor|second\s*floor|master\s*bedroom|guest\s*room|living\s*room|dining\s*room|bathroom|toilet|restroom|garage|basement|attic|corridor|hallway|entrance|lobby|office|study|bedroom|kitchen|lounge|garden|patio|deck|balcony|porch|staircase|landing)\b",
        ]
        
        potential_areas = []
        
        for pattern in area_patterns:
            matches = re.findall(pattern, query_lower)
            for match in matches:
                if isinstance(match, tuple):
                    match = " ".join(m for m in match if m)
                if match and len(match.strip()) > 2:
                    potential_areas.append(match.strip())
        
        # Also check for common area names as standalone words
        common_areas = [
            "kitchen", "bathroom", "bedroom", "living room", "dining room",
            "hallway", "corridor", "office", "study", "garage", "basement",
            "attic", "garden", "patio", "deck", "balcony", "porch", "lobby",
            "entrance", "toilet", "restroom", "upstairs", "downstairs",
            "ground floor", "first floor", "second floor", "master bedroom",
            "guest room", "nursery", "laundry", "utility", "closet", "wardrobe",
        ]
        
        for area in common_areas:
            if f" {area} " in f" {query_lower} " or query_lower.endswith(f" {area}") or query_lower.startswith(f"{area} "):
                if area not in potential_areas:
                    potential_areas.append(area)
        
        return potential_areas

    async def _detect_area_from_query(self, query: str) -> tuple[Optional[dict], float]:
        """
        Detect which area (if any) the user is referring to in their query.
        
        Uses fuzzy matching against known areas in Home Assistant.
        
        Returns:
            tuple: (area_info dict or None, confidence_score 0-1)
        """
        query_lower = query.lower()
        
        # Get known areas
        known_areas = await self._get_known_areas()
        
        if not known_areas:
            logger.info("No areas configured in Home Assistant")
            return None, 0.0
        
        # Extract potential area mentions from query
        potential_mentions = self._extract_potential_area_mentions(query)
        
        logger.info(f"Potential area mentions in query: {potential_mentions}")
        
        best_match = None
        best_score = 0.0
        
        for mention in potential_mentions:
            mention_clean = mention.strip()
            
            for area_name_lower, area_info in known_areas.items():
                # Exact match
                if area_name_lower == mention_clean:
                    return area_info, 1.0
                
                # Fuzzy match
                ratio = difflib.SequenceMatcher(None, mention_clean, area_name_lower).ratio()
                
                # Check if mention is substring of area name
                if mention_clean in area_name_lower:
                    ratio = max(ratio, 0.85)
                
                # Check if area name is substring of mention
                if area_name_lower in mention_clean:
                    ratio = max(ratio, 0.8)
                
                # Check for partial word matches (longer matches are better)
                mention_words = set(mention_clean.split())
                area_words = set(area_name_lower.split())
                word_overlap = len(mention_words & area_words) / max(len(mention_words), len(area_words), 1)
                
                if word_overlap > 0.7 and ratio < 0.9:
                    ratio = max(ratio, word_overlap)
                
                if ratio > best_score:
                    best_score = ratio
                    best_match = area_info
        
        # Require minimum confidence for area detection
        if best_score >= 0.7:
            logger.info(f"Detected area: {best_match['area_name']} (score: {best_score:.2f})")
            return best_match, best_score
        
        logger.info(f"No area detected (best score: {best_score:.2f})")
        return None, 0.0

    # =========================================================================
    # Entity Matching
    # =========================================================================

    def _extract_device_name(self, user_request: str) -> str:
        """
        Extract the device/location name from user request.
        
        Examples:
        - "turn on the toilet light" -> "toilet light"
        - "what's the temperature in the kitchen" -> "kitchen temperature"
        - "dim the living room lights" -> "living room lights"
        """
        request_lower = user_request.lower()
        
        # Action words and patterns to remove
        action_patterns = [
            r"^(turn|switch|set|get|check|what's|whats|what is|is the|are the)\s+",
            r"\s+(on|off|up|down|to|in|at|for)$",
            r"\s+(on|off|up|down)$",
            r"\b(in|at|to|for|on|under|over)\s+(?:the\s+)?(?:[a-z]+\s+)?(light|lights|fan|temperature|thermostat|switch|sensor|climate|device|tv|lamp|outlet)\b",
        ]
        
        cleaned = request_lower
        for pattern in action_patterns:
            cleaned = re.sub(pattern, " ", cleaned).strip()
        
        # Remove common words
        stop_words = {
            "the", "a", "an", "please", "thanks", "thank", "you", "my",
            "all", "every", "any", "some", "that", "this", "these", "those",
        }
        
        words = [w for w in cleaned.split() if w not in stop_words and len(w) > 1]
        device_name = " ".join(words)
        
        return device_name

    def _generate_entity_candidates(self, device_name: str) -> list[str]:
        """
        Generate possible entity_id patterns from a device name.
        
        Examples:
        - "toilet light" -> ["toilet_light", "toilet", "light_toilet", "toilet_light_1", "toilet_light_2"]
        - "kitchen leds" -> ["kitchen_leds", "kitchen", "kitchen_led", "leds_kitchen"]
        - "hallway" -> ["hallway", "hallway_light"]
        """
        device_name_clean = device_name.lower().strip()
        words = device_name_clean.split()
        
        candidates = set()
        
        # Original device name as-is (with spaces converted to underscores)
        candidates.add(device_name_clean.replace(" ", "_"))
        
        # Single word device names
        if len(words) == 1:
            word = words[0]
            # Just the word
            candidates.add(word)
            # With common suffixes
            candidates.add(f"{word}_light")
            candidates.add(f"{word}_switch")
            candidates.add(f"{word}_sensor")
            candidates.add(f"{word}_1")
            candidates.add(f"{word}_2")
            # Without trailing s
            if word.endswith("s"):
                candidates.add(word[:-1])
                candidates.add(f"{word[:-1]}_light")
        
        # Multi-word device names
        else:
            # joined_with_underscores
            joined = "_".join(words)
            candidates.add(joined)
            
            # reversed words
            reversed_words = "_".join(reversed(words))
            candidates.add(reversed_words)
            
            # room_location + device_type pattern (e.g., "kitchen_leds" from "kitchen leds")
            # Assume last word might be device type
            if len(words) >= 2:
                device_type = words[-1]  # e.g., "leds" from "kitchen leds"
                location = "_".join(words[:-1])  # e.g., "kitchen"
                
                # Clean up plural forms for device type
                device_type_clean = device_type.rstrip("s")
                candidates.add(f"{location}_{device_type}")
                candidates.add(f"{location}_{device_type_clean}")
                candidates.add(f"{location}_{device_type_clean}s")
                
                # Also try with device type last (e.g., "leds_kitchen")
                candidates.add(f"{device_type}_{location}")
                candidates.add(f"{device_type_clean}_{location}")
            
            # Try adding common suffixes
            candidates.add(f"{joined}_1")
            candidates.add(f"{joined}_2")
            candidates.add(f"{joined}_3")
            
            # Try with "light" suffix if not already present
            if "light" not in device_name_clean:
                candidates.add(f"{joined}_light")
                candidates.add(f"{joined}_switch")
            
            # Try without trailing numbers (e.g., "kitchen_leds" -> "kitchen_led")
            if any(c.isdigit() for c in joined):
                base = re.sub(r"_\d+$", "", joined)
                candidates.add(base)
        
        return list(candidates)

    def _calculate_match_score(self, request_words: list[str], entity_info: dict) -> float:
        """
        Calculate match score between user request and entity.
        
        Uses multiple strategies:
        1. Direct string matching
        2. Word-by-word overlap
        3. Fuzzy matching on entity_id parts
        4. Friendly name matching
        5. Area name matching
        """
        if not request_words:
            return 0.0

        entity_id = entity_info.get("entity_id", "").lower()
        friendly_name = entity_info.get("friendly_name", "").lower()
        area_name = entity_info.get("area_name", "").lower()
        entity_name = entity_info.get("entity_name", "").lower()
        domain = entity_info.get("domain", "").lower()

        # Preprocess entity parts
        entity_id_words = entity_id.replace(".", " ").replace("_", " ").split()
        friendly_words = friendly_name.replace(".", " ").replace("_", " ").split()
        area_words = area_name.replace(".", " ").replace("_", " ").split() if area_name else []

        # Calculate word overlap ratios
        request_set = set(request_words)
        friendly_set = set(friendly_words)
        entity_set = set(entity_id_words)
        area_set = set(area_words)

        overlap_friendly = len(request_set & friendly_set) / max(len(request_set), 1)
        overlap_entity = len(request_set & entity_set) / max(len(request_set), 1)
        overlap_area = len(request_set & area_set) / max(len(request_set), 1) if area_set else 0

        # Build candidate entity_ids and check for matches
        device_name = " ".join(request_words)
        candidates = self._generate_entity_candidates(device_name)
        
        candidate_match = 0.0
        for candidate in candidates:
            if candidate == entity_name:
                candidate_match = 1.0
                break
            if candidate in entity_name:
                candidate_match = max(candidate_match, 0.8)
            if entity_name in candidate:
                candidate_match = max(candidate_match, 0.7)
            ratio = difflib.SequenceMatcher(None, candidate, entity_name).ratio()
            candidate_match = max(candidate_match, ratio * 0.6)

        # Check if request words match area + device pattern
        area_device_match = 0.0
        for req_word in request_words:
            if req_word in area_words:
                area_device_match = 0.6
                remaining_words = [w for w in request_words if w != req_word]
                if remaining_words:
                    for rw in remaining_words:
                        if rw in entity_id_words or rw in friendly_words:
                            area_device_match = 0.8

        # Combine scores with weights
        combined_score = (
            overlap_friendly * 0.25 +
            overlap_entity * 0.20 +
            overlap_area * 0.15 +
            candidate_match * 0.30 +
            area_device_match * 0.10
        )

        return min(combined_score, 1.0)

    def _score_entities(self, entities: dict, request_words: list[str]) -> list[dict]:
        """
        Score all entities and return sorted list.
        
        Returns:
            list: Sorted list of dicts with entity_id, score, and info
        """
        scored_entities = []
        for entity_id, entity_info in entities.items():
            score = self._calculate_match_score(request_words, entity_info)
            if score > 0.1:
                scored_entities.append({
                    "entity_id": entity_id,
                    "entity": entity_info["entity"],
                    "friendly_name": entity_info["friendly_name"],
                    "area_name": entity_info["area_name"],
                    "domain": entity_info["domain"],
                    "score": score,
                })

        scored_entities.sort(key=lambda x: x["score"], reverse=True)
        return scored_entities

    async def _find_matching_entities(
        self, 
        user_request: str, 
        entity_map: dict,
        area_constraint: Optional[dict] = None
    ) -> tuple[list[dict], str]:
        """
        Find entities matching the user request with area-prioritized search.
        
        Two-stage search:
        1. If area detected, search only entities in that area
        2. If no good match in area, fall back to global search
        
        Returns:
            tuple: (list of scored entities, search_method used)
        """
        device_name = self._extract_device_name(user_request)
        request_words = device_name.lower().split()
        
        logger.info(f"Extracted device name: '{device_name}' from request: '{user_request}'")
        logger.info(f"Request words: {request_words}")

        # Stage 1: Area-constrained search (if area detected)
        if area_constraint:
            area_name = area_constraint["area_name"]
            logger.info(f"Stage 1: Searching in area '{area_name}'")
            
            # Filter to entities in the specified area
            area_entities = {
                eid: einfo 
                for eid, einfo in entity_map.items()
                if einfo["area_name"] and einfo["area_name"].lower() == area_name.lower()
            }
            
            logger.info(f"Found {len(area_entities)} entities in area '{area_name}'")
            
            if area_entities:
                # Score only area-constrained entities
                scored_area_entities = self._score_entities(area_entities, request_words)
                
                # Log top area matches
                if scored_area_entities:
                    top_area = scored_area_entities[0]
                    logger.info(f"Best area match: {top_area['entity_id']} (score: {top_area['score']:.3f})")
                
                # Check if we have a good enough match
                if scored_area_entities and scored_area_entities[0]["score"] >= self.valves.AREA_MATCH_THRESHOLD:
                    logger.info(f"Stage 1 success: Found good match in area '{area_name}'")
                    return scored_area_entities, "area_priority"
                
                logger.info(f"Stage 1: No good match in area (best score: {scored_area_entities[0]['score'] if scored_area_entities else 0:.3f})")
                logger.info(f"Stage 2: Falling back to global search")
        
        # Stage 2: Global search
        logger.info("Stage 2: Searching all entities")
        scored_entities = self._score_entities(entity_map, request_words)
        
        if scored_entities:
            top_match = scored_entities[0]
            logger.info(f"Best global match: {top_match['entity_id']} (score: {top_match['score']:.3f})")
            
            # If we had an area constraint but no match, note it
            if area_constraint:
                logger.info(f"Global search fallback: No suitable match found in area '{area_constraint['area_name']}'")
                return scored_entities, "global_fallback"
        
        return scored_entities, "global"

    # =========================================================================
    # Response Formatting
    # =========================================================================

    def _format_entity_for_llm(self, entity: dict) -> dict:
        """Format entity information for LLM understanding."""
        entity_id = entity.get("entity_id", "")
        attributes = entity.get("attributes", {})
        state = entity.get("state", "unknown")

        domain = entity_id.split(".")[0] if "." in entity_id else ""

        info = {
            "entity_id": entity_id,
            "domain": domain,
            "friendly_name": attributes.get("friendly_name", entity_id),
            "state": state,
        }

        # Add area/room if available
        if "area_name" in attributes:
            info["area"] = attributes["area_name"]

        # Add device class for sensors
        if "device_class" in attributes:
            info["device_class"] = attributes["device_class"]

        # Add unit of measurement for sensors
        if "unit_of_measurement" in attributes:
            info["unit"] = attributes["unit_of_measurement"]

        # Add relevant domain-specific attributes
        if domain == "light" and state == "on":
            if "brightness" in attributes:
                info["brightness_pct"] = round((attributes["brightness"] / 255) * 100)
        elif domain == "climate":
            if "current_temperature" in attributes:
                info["current_temp"] = attributes["current_temperature"]
            if "temperature" in attributes:
                info["target_temp"] = attributes["temperature"]
        elif domain == "cover":
            if "current_position" in attributes:
                info["position"] = attributes["current_position"]

        return info

    def _explain_match(self, match: dict, user_request: str, search_method: str, detected_area: Optional[str] = None) -> str:
        """Explain why this entity was matched."""
        score = match["score"]
        friendly = match["friendly_name"]
        area = match["area_name"]
        entity_id = match["entity_id"]

        explanations = []
        
        if search_method == "area_priority" and detected_area:
            explanations.append(f"Found in the '{detected_area}' area (area-prioritized search)")
        elif search_method == "global_fallback" and detected_area:
            explanations.append(f"Not found in '{detected_area}', expanded search globally")
        
        if area and area != entity_id.split(".")[1].replace("_", " ") and area.lower() != detected_area.lower() if detected_area else True:
            explanations.append(f"Located in the '{area}' area")
        
        if score >= 0.8:
            explanations.append("High confidence match")
        elif score >= 0.6:
            explanations.append("Good match based on name similarity")
        elif score >= 0.4:
            explanations.append("Best available match found")
        else:
            explanations.append("Match found via fuzzy search")

        return "; ".join(explanations) if explanations else "Entity found in Home Assistant"

    # =========================================================================
    # Main Tool Functions
    # =========================================================================

    async def control_home_assistant(
        self,
        user_request: str,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> str:
        """
        [PRIMARY FUNCTION] Control or query Home Assistant based on natural language request.

        This function uses smart entity detection with area-prioritized search. When you mention
        a room or area in your request, it first looks for matching devices in that area. Only
        if no suitable match is found there does it expand the search to your entire home.

        Examples:
        - "turn on the toilet light" -> Detects "light.toilet_light" via global search
        - "turn on kitchen lights" -> First searches Kitchen area, finds "light.kitchen_leds"
        - "dim the bedroom lights" -> First searches Bedroom area, finds "light.bedroom_ceiling"
        - "what's the temperature in the hallway" -> First searches Hallway area, finds sensor

        :param user_request: Natural language request (e.g., "turn on toilet light", "what's the temperature")
        :return: JSON object with matching entities and search metadata
        """
        if not self.valves.HA_URL or not self.valves.HA_TOKEN:
            return {
                "error": "NOT_CONFIGURED",
                "message": "Home Assistant is not configured. Please set HA_URL and HA_TOKEN in the tool valves."
            }

        try:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Fetching Home Assistant entities...", "done": False},
                    }
                )

            # Build entity map with area info
            entity_map = await self._build_entity_map()
            logger.info(f"Loaded {len(entity_map)} entities from Home Assistant")

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Detecting area from request...", "done": False},
                    }
                )

            # Detect area from query
            detected_area, area_confidence = await self._detect_area_from_query(user_request)
            
            if detected_area:
                logger.info(f"Area detected: {detected_area['area_name']} (confidence: {area_confidence:.2f})")
            else:
                logger.info("No area detected in query")

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Finding matching entities...", "done": False},
                    }
                )

            # Find matching entities with area-prioritized search
            scored_entities, search_method = await self._find_matching_entities(
                user_request, 
                entity_map,
                area_constraint=detected_area
            )

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Analyzing results...", "done": False},
                    }
                )

            # Format top matches for LLM
            top_matches = scored_entities[:20]

            # Group by domain
            by_domain = {}
            for item in top_matches:
                entity = item["entity"]
                domain = item["domain"]
                if domain not in by_domain:
                    by_domain[domain] = []
                by_domain[domain].append({
                    "entity_id": item["entity_id"],
                    "friendly_name": item["friendly_name"],
                    "area": item["area_name"],
                    "state": entity.get("state", "unknown"),
                    "match_score": round(item["score"], 3),
                })

            # Build response
            if len(top_matches) == 0:
                result = {
                    "user_request": user_request,
                    "message": "No matching entities found in Home Assistant.",
                    "suggestions": [
                        "Try being more specific about the room or device type",
                        "Check the entity name in Home Assistant Developer Tools",
                        "Make sure the entity is in an enabled domain (light, switch, climate, etc.)"
                    ],
                    "entities_found": 0,
                }
            else:
                best_match = top_matches[0]
                detected_area_name = detected_area["area_name"] if detected_area else None
                
                result = {
                    "user_request": user_request,
                    "search_metadata": {
                        "detected_area": detected_area_name,
                        "area_confidence": round(area_confidence, 2),
                        "search_method": search_method,
                        "search_explanation": self._get_search_method_explanation(
                            search_method, detected_area_name, best_match["score"]
                        ),
                    },
                    "best_match": {
                        "entity_id": best_match["entity_id"],
                        "friendly_name": best_match["friendly_name"],
                        "area": best_match["area_name"],
                        "domain": best_match["domain"],
                        "match_score": round(best_match["score"], 3),
                        "state": best_match["entity"].get("state", "unknown"),
                    },
                    "entities_found": len(top_matches),
                    "entities_by_domain": by_domain,
                    "instructions": {
                        "step_1": "Use the best_match.entity_id for your action (or select from entities_by_domain)",
                        "step_2": "Call execute_action() with the EXACT entity_id string",
                        "step_3": "For state queries, use action_type='get_state'",
                        "step_4": "For control actions, use action_type='call_service' with appropriate service",
                        "critical": "DO NOT modify or guess the entity_id - use the EXACT value from best_match or entities_by_domain",
                    },
                    "match_explanation": self._explain_match(
                        best_match, user_request, search_method, detected_area_name
                    ),
                }

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"Found {len(top_matches)} matching entities", "done": True},
                    }
                )

            return result

        except Exception as e:
            error_msg = f"Error processing Home Assistant request: {str(e)}"
            logger.error(error_msg)
            return {
                "error": "EXCEPTION",
                "message": error_msg
            }

    def _get_search_method_explanation(
        self, 
        search_method: str, 
        detected_area: Optional[str], 
        best_score: float
    ) -> str:
        """Get human-readable explanation of which search method was used."""
        if search_method == "area_priority":
            return (
                f"Area-prioritized search: Found matching device in '{detected_area}' area. "
                f"This is the device most likely to match your request since you mentioned this area."
            )
        elif search_method == "global_fallback":
            return (
                f"Global fallback search: You mentioned '{detected_area}' but no matching device "
                f"was found there, so I searched your entire home. The best match has a score of {best_score:.2f}."
            )
        else:
            return (
                "Global search: No specific area was detected in your request, so I searched "
                "all devices in your home."
            )

    async def execute_action(
        self,
        action_type: str,
        entity_id: str,
        service: Optional[str] = None,
        friendly_name: Optional[str] = None,
        additional_data: Optional[str] = None,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> str:
        """
        Execute an action on Home Assistant after getting context from control_home_assistant().

        IMPORTANT: You must use the EXACT entity_id from control_home_assistant() response.
        Do NOT guess, truncate, or modify the entity_id in any way.

        Examples of CORRECT usage:
        - If context shows "light.hallway_light" -> Use "light.hallway_light" ✓
        - If context shows "light.toilet_light" -> Use "light.toilet_light" ✓

        Examples of INCORRECT usage:
        - Context shows "light.hallway_light" but you use "light.hallway" ✗
        - Context shows "sensor.living_room_temperature" but you use "sensor.living_room" ✗

        :param action_type: Either "get_state" or "call_service"
        :param entity_id: EXACT entity_id from control_home_assistant() - no modifications!
        :param service: Service name if action_type is "call_service" (e.g., "turn_on", "turn_off", "toggle")
        :param friendly_name: Friendly name of the entity (optional, for display purposes)
        :param additional_data: Optional JSON string with service parameters
        :return: JSON object with result of the action
        """
        if not self.valves.HA_URL or not self.valves.HA_TOKEN:
            return {
                "error": "NOT_CONFIGURED",
                "message": "Home Assistant is not configured."
            }

        # Helper function to format responses
        def format_ha_response(raw_response: dict) -> str:
            """Clean and summarize Home Assistant tool output for display."""
            success = raw_response.get("success", True)
            domain = raw_response.get("domain")
            service_name = raw_response.get("service")
            entity_id_response = raw_response.get("entity_id")
            entity_name = raw_response.get("friendly_name", entity_id_response)
            message = raw_response.get("result", "")

            if service_name in ["turn_on", "turn_off"]:
                return f"{entity_name} turned {service_name.replace('_', ' ')}."
            elif service_name in ["toggle"]:
                return f"{entity_name} toggled."
            elif service_name in ["set_temperature", "set_brightness"]:
                return f"{service_name.replace('_', ' ').capitalize()} executed for {entity_name}."
            elif message:
                cleaned = re.sub(r"Successfully called \S+ on ", "", message)
                return f"{entity_name} updated." if not cleaned else cleaned

            return f"Action {service_name} executed on {entity_name}."

        try:
            result = {
                "success": False,
                "action": action_type,
                "domain": "",
                "service": "",
                "friendly_name": "",
                "entity_id": entity_id,
                "result": "",
                "message": ""
            }

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"{str(action_type).replace('_', ' ').capitalize()} {f'{service} ' if service else ''}on {entity_id}...", "done": False},
                    }
                )

            # Validate entity_id format
            if "." not in entity_id:
                return {
                    "error": "INVALID_ENTITY_ID",
                    "message": f"Invalid entity_id format: {entity_id}. Must be 'domain.entity_name'"
                }

            domain = entity_id.split(".")[0]

            if action_type == "call_service":
                if not service:
                    return {
                        "error": "MISSING_PARAMETER",
                        "message": "service parameter is required when action_type is 'call_service'"
                    }

                # Verify entity exists
                try:
                    entity = await self._make_request("GET", f"states/{entity_id}")
                    friendly_name = entity.get("attributes", {}).get("friendly_name", entity_id) if not friendly_name else friendly_name
                    state = entity.get("state", "unknown")
                except Exception as api_error:
                    if "404" in str(api_error) or "Not Found" in str(api_error):
                        return await self._handle_entity_not_found(entity_id)
                    raise

                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": f"Calling {service} on {friendly_name}...", "done": False},
                        }
                    )

                # Prepare service data
                service_data = {"entity_id": entity_id}

                # Parse additional data if provided
                if additional_data:
                    try:
                        extra = json.loads(additional_data)
                        service_data.update(extra)
                    except json.JSONDecodeError:
                        return {
                            "error": "INVALID_JSON",
                            "message": f"Invalid JSON in additional_data: {additional_data}"
                        }

                # Call the service
                try:
                    await self._make_request(
                        "POST",
                        f"services/{domain}/{service}",
                        data=service_data
                    )
                except Exception as api_error:
                    if "404" in str(api_error) or "Not Found" in str(api_error):
                        return await self._handle_entity_not_found(entity_id)
                    raise

                result["success"] = True
                result["domain"] = domain
                result["service"] = service
                result["friendly_name"] = friendly_name
                result["entity_id"] = entity_id
                result["result"] = f"Successfully called {service} on {domain} {friendly_name}"
                result["message"] = format_ha_response(result)

                if additional_data:
                    result["parameters"] = json.loads(additional_data)

                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": f"Successfully executed {service} on {friendly_name}", "done": True},
                        }
                    )

                logger.info(f"Called {domain}.{service} on {entity_id}")

                # If this was a state-changing action, get the updated state
                if service in ["turn_on", "turn_off", "toggle", "set_temperature", "set_brightness"]:
                    await asyncio.sleep(1.5)  # Wait for state to update
                    action_type = "get_state"

            if action_type == "get_state":
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": f"Getting state for {friendly_name if friendly_name else entity_id}...", "done": False},
                        }
                    )

                try:
                    entity = await self._make_request("GET", f"states/{entity_id}")
                    friendly_name = entity.get("attributes", {}).get("friendly_name", entity_id) if not friendly_name else friendly_name
                    state = entity.get("state", "unknown")
                    attributes = entity.get("attributes", {})
                except Exception as api_error:
                    if "404" in str(api_error) or "Not Found" in str(api_error):
                        return await self._handle_entity_not_found(entity_id)
                    raise

                result["success"] = True
                result["domain"] = domain
                result["friendly_name"] = friendly_name
                result["state"] = state
                result["last_changed"] = entity.get("last_changed", "unknown")
                result["last_updated"] = entity.get("last_updated", "unknown")
                result["attributes"] = attributes

                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": f"Retrieved state for {friendly_name}: {state}", "done": True},
                        }
                    )

                logger.info(f"Retrieved state for {entity_id}")

            else:
                return {
                    "error": "INVALID_ACTION_TYPE",
                    "message": f"Invalid action_type: {action_type}. Must be 'get_state' or 'call_service'"
                }

            return result

        except Exception as e:
            error_msg = f"Error executing action: {str(e)}"
            logger.error(error_msg)
            return {
                "error": "EXCEPTION",
                "message": error_msg
            }

    async def _handle_entity_not_found(self, attempted_entity_id: str) -> dict:
        """Handle entity not found errors with smart suggestions."""
        logger.warning(f"Entity not found: {attempted_entity_id}")

        # Extract domain and attempted name
        domain = attempted_entity_id.split(".")[0] if "." in attempted_entity_id else ""
        attempted_name = attempted_entity_id.split(".")[1] if "." in attempted_entity_id else attempted_entity_id

        # Get fresh entity map
        try:
            entity_map = await self._build_entity_map()
            
            # Find similar entities in the same domain
            candidates = []
            
            for entity_id, entity_info in entity_map.items():
                entity_domain = entity_id.split(".")[0] if "." in entity_id else ""
                
                if entity_domain != domain:
                    continue
                
                friendly_name = entity_info["friendly_name"]
                entity_name = entity_info["entity_name"]
                area_name = entity_info["area_name"]

                # Calculate multiple similarity scores
                name_similarity = difflib.SequenceMatcher(None, attempted_name.lower(), entity_name.lower()).ratio()
                friendly_similarity = difflib.SequenceMatcher(None, attempted_name.lower(), friendly_name.lower()).ratio()
                area_similarity = difflib.SequenceMatcher(None, attempted_name.lower(), area_name.lower()).ratio() if area_name else 0

                # Check for substring matches
                substring_bonus = 0.2 if attempted_name.lower() in entity_name.lower() else 0
                reverse_substring = 0.15 if entity_name.lower() in attempted_name.lower() else 0

                # Combined score with weights
                combined_score = max(
                    name_similarity * 0.5 + friendly_similarity * 0.3 + area_similarity * 0.1 + substring_bonus + reverse_substring,
                    name_similarity,
                    friendly_similarity,
                    area_similarity
                )

                if combined_score > 0.3:
                    candidates.append({
                        "entity_id": entity_id,
                        "friendly_name": friendly_name,
                        "area": area_name,
                        "similarity_score": round(combined_score, 2),
                    })

            # Sort by similarity
            candidates.sort(key=lambda x: x["similarity_score"], reverse=True)

            if candidates:
                return {
                    "error": "ENTITY_NOT_FOUND",
                    "attempted_entity_id": attempted_entity_id,
                    "message": f"The entity_id '{attempted_entity_id}' does NOT exist in Home Assistant.",
                    "possible_matches": candidates[:10],
                    "required_action": {
                        "step_1": "Call control_home_assistant() again with the same user request",
                        "step_2": "Find the correct entity_id from the response (likely one from possible_matches)",
                        "step_3": "Call execute_action() again with the EXACT correct entity_id",
                    },
                    "warning": "DO NOT use the attempted_entity_id - it is WRONG. You MUST call control_home_assistant() again.",
                }
            else:
                return {
                    "error": "ENTITY_NOT_FOUND",
                    "attempted_entity_id": attempted_entity_id,
                    "message": f"The entity_id '{attempted_entity_id}' does NOT exist in Home Assistant.",
                    "possible_matches": [],
                    "required_action": {
                        "step_1": "Call control_home_assistant() again with a more specific request",
                        "step_2": "Check available entities in the Home Assistant UI",
                    },
                    "warning": "DO NOT guess entity_ids. You MUST call control_home_assistant() again.",
                }

        except Exception as e:
            return {
                "error": "ENTITY_NOT_FOUND",
                "attempted_entity_id": attempted_entity_id,
                "message": f"The entity_id '{attempted_entity_id}' does not exist.",
                "search_error": str(e),
                "required_action": {
                    "step_1": "Call control_home_assistant() again to get the correct entity_id",
                },
                "warning": "You MUST call control_home_assistant() again.",
            }

    async def validate_connection(self) -> dict:
        """
        Validate the Home Assistant connection when valves are saved.
        This is called automatically by Open WebUI when valve values are updated.
        """
        if not self.valves.HA_URL or not self.valves.HA_TOKEN:
            return {
                "status": "not_configured",
                "message": "Home Assistant URL and Token are not configured. Please configure them to enable the tool."
            }

        try:
            # Test REST API
            config = await self._make_request("GET", "config")
            states = await self._make_request("GET", "states")
            filtered = self._filter_entities(states)

            # Get known areas
            known_areas = await self._get_known_areas()
            area_count = len(known_areas)

            # Test WebSocket if enabled
            ws_status = "disabled"
            if self.valves.USE_WEBSOCKET:
                try:
                    await self._get_ws_connection()
                    ws_status = "connected"
                    await self.close_ws()
                except Exception as ws_error:
                    ws_status = f"failed: {str(ws_error)}"

            logger.info(f"✓ Connected to Home Assistant - {len(filtered)} entities, {area_count} areas (WebSocket: {ws_status})")
            return {
                "status": "success",
                "message": f"✓ Connected to Home Assistant\n"
                          f"Location: {config.get('location_name', 'Unknown')}\n"
                          f"Version: {config.get('version', 'Unknown')}\n"
                          f"Discovered Entities: {len(filtered)}\n"
                          f"Known Areas: {area_count}\n"
                          f"WebSocket: {ws_status}",
            }
        except Exception as e:
            logger.error(f"✗ Failed to connect: {str(e)}")
            return {
                "status": "error",
                "message": f"✗ Failed to connect to Home Assistant: {str(e)}\n"
                          f"Please check your HA_URL and HA_TOKEN configuration.",
            }

    async def list_areas(self, __event_emitter__: Callable[[dict], Any] = None) -> dict:
        """
        List all areas/rooms configured in Home Assistant.
        Useful for discovering what rooms are available.

        :return: JSON object with list of areas and their entities
        """
        if not self.valves.HA_URL or not self.valves.HA_TOKEN:
            return {
                "error": "NOT_CONFIGURED",
                "message": "Home Assistant is not configured."
            }

        try:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Fetching areas and entities...", "done": False},
                    }
                )

            # Build entity map with area info
            entity_map = await self._build_entity_map()

            # Group entities by area
            areas = {}
            for entity_id, entity_info in entity_map.items():
                area_name = entity_info["area_name"] or "Unknown Area"
                if area_name not in areas:
                    areas[area_name] = {
                        "area_name": area_name,
                        "entities": [],
                    }
                areas[area_name]["entities"].append({
                    "entity_id": entity_id,
                    "friendly_name": entity_info["friendly_name"],
                    "domain": entity_info["domain"],
                    "state": entity_info["entity"].get("state", "unknown"),
                })

            # Convert to list
            area_list = [
                {"area_name": name, **data}
                for name, data in areas.items()
            ]

            # Sort by area name
            area_list.sort(key=lambda x: x["area_name"])

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"Found {len(area_list)} areas", "done": True},
                    }
                )

            return {
                "total_areas": len(area_list),
                "areas": area_list,
                "instructions": {
                    "step_1": "Find the area you want to control",
                    "step_2": "Call control_home_assistant() with a request like 'turn on lights in [area_name]'",
                    "step_3": "Use the entity_id from the response to execute actions",
                }
            }

        except Exception as e:
            error_msg = f"Error listing areas: {str(e)}"
            logger.error(error_msg)
            return {
                "error": "EXCEPTION",
                "message": error_msg
            }
