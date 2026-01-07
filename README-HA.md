# 🏠 Home Assistant for Open WebUI

**Version:** 1.3.0

**Author:** lexiismadd

This tool connects to your **Home Assistant** instance to allow control of your devices directly from Open WebUI.
This tool was inspired by a few other Home Assistant tools on the Open WebUI community that have not had an update in some time, and the newer versions of Home Assistant no longer work with those tools.

---

## ✨ Key Features

* **Natural Language Processing:** Turn vague requests like "make it cozy in here" or "is the garage open?" into actionable commands.
* **Intelligent Entity Discovery:** Automatically fetches and filters your Home Assistant entities based on domains, inclusions, and exclusions.
* **Fuzzy Matching Logic:** Uses advanced string matching (via `difflib`) to find the correct device even if you don't know its exact technical name.
* **Multi-Step Workflow Safety:** Employs a robust 2-step process (Identify → Execute) to ensure the AI never "guesses" an entity ID or performs an unintended action.
* **Real-time State Monitoring:** Fetches the most recent state (brightness, temperature, position, etc.) immediately after an action to confirm success.
* **Thumbnail/Media Support:** Supports camera entities and media player discovery within your smart home.

---

## 🚀 How It Works

The tool operates in a structured sequence to ensure precision:

1. **Request Parsing:** When you give a command, the tool refreshes its knowledge of your home by fetching all current entity states.
2. **Contextual Filtering:** It filters thousands of entities down to a few dozen that most closely match your keywords (e.g., "light", "bedroom", "70 degrees").
3. **Action Execution:**
* **Query:** If you asked a question, it returns the current state and attributes.
* **Command:** If you issued a command, it identifies the exact service (like `light.turn_on`) and applies it to the specific `entity_id`.
4. **Verification:** The tool waits for the device to respond and returns a human-readable confirmation.

---

## ⚙️ Configuration (Valves)

| Valve | Description |
| --- | --- |
| **HA_URL** | The URL of your Home Assistant instance (e.g., `http://192.168.1.50:8123`). |
| **HA_TOKEN** | Your Long-Lived Access Token generated in Home Assistant profile settings. |
| **DISCOVER_DOMAINS** | Comma-separated list of domains to track (Default: `light, switch, climate, vacuum`, etc.). |
| **INCLUDED_ENTITIES** | (Optional) Wildcard patterns to specifically include (e.g., `light.living_room_*`). |
| **EXCLUDED_ENTITIES** | (Optional) Wildcard patterns to hide from the AI (e.g., `sensor.*_battery`). |

---

## 🛠️ Requirements

* **Home Assistant:** Minimum v2025.12.4 with the REST API exposed to Open WebUI (a Long Lived Access Token is required).
* **Network:** Open WebUI must have network line-of-sight to the Home Assistant URL.

---

## 📖 Usage Examples

**Climate Control:**

> "Set the thermostat in the lounge room to 23 degrees."

**Lighting & Ambience:**

> "Turn off all the lights in the dining room."
> "Dim the hallway light to 20%."

**Status Checks:**

> "Is the front door locked?
> "What is the current humidity outside?"

**Complex Automation:**

> "Run the 'Movie Night' scene and close the bedroom blinds."

---

## Special Notes on Include vs Exclude

If you specify an *INCLUSION* pattern for `INCLUDED_ENTITIES`, then **ONLY** those entities that match that pattern will be available to the LLM.
However, if you specify an *EXCLUSION* pattern for `EXCLUDED_ENTITIES`, then **ALL** entities **EXCEPT** those that match the pattern will be available to the LLM.

---

## Issues

Please log any issues [on the Github repo](https://github.com/lexiismadd/openwebui-extensibles/issues)

---
