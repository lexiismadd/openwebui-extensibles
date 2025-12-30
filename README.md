# Home Assistant Control Tool for Open WebUI

A comprehensive Python tool for Open WebUI v0.6.4+ that enables LLMs to control and monitor your local Home Assistant instance.

## Features

- 🔍 **Entity Discovery**: Automatically discover and filter Home Assistant entities
- 🎛️ **Service Execution**: Call any Home Assistant service (lights, switches, climate, etc.)
- 📊 **State Monitoring**: Query current states and attributes of entities
- 🔧 **Flexible Filtering**: Include/exclude entities using wildcard patterns
- 💾 **Smart Caching**: Efficient caching to reduce API calls
- ✅ **Connection Validation**: Automatic validation when configuration is saved

## Compatible Home Assistant Domains

- **light**: Control lights (on/off, brightness, color)
- **switch**: Toggle switches
- **climate**: Control HVAC systems (temperature, mode)
- **cover**: Control covers/blinds (open/close, position)
- **fan**: Control fans (on/off, speed)
- **lock**: Lock/unlock doors
- **media_player**: Control media players (play/pause, volume)
- **sensor**: Read sensor values (temperature, humidity, etc.)
- **binary_sensor**: Read binary sensor states
- **weather**: Get weather information
- **camera**: Access camera entities
- **vacuum**: Control vacuum cleaners
- **scene**: Activate scenes
- **script**: Run scripts
- **automation**: Control automations

## Installation

### Step 1: Create Long-Lived Access Token in Home Assistant

1. Open your Home Assistant instance
2. Click on your profile (bottom left)
3. Scroll down to "Long-Lived Access Tokens"
4. Click "Create Token"
5. Give it a name (e.g., "Open WebUI Integration")
6. Copy the token (you won't be able to see it again!)

### Step 2: Install the Tool in Open WebUI

1. Log into Open WebUI as an admin
2. Navigate to **Settings** → **Tools**
3. Click **"+"** to add a new tool
4. Copy and paste the entire contents of `home_assistant_tool.py`
5. Click **Save**

### Step 3: Configure the Valves

1. Find the "Home Assistant Control" tool in your tools list
2. Click the **settings icon** (⚙️) to configure valves
3. Set the following values:

   - **HA_URL**: Your Home Assistant URL (e.g., `http://homeassistant.local:8123` or `http://192.168.1.100:8123`)
   - **HA_TOKEN**: Your long-lived access token from Step 1
   - **DISCOVER_DOMAINS**: Comma-separated domains to discover (default includes most common domains)
   - **INCLUDED_ENTITIES**: (Optional) Comma-separated patterns to include (e.g., `light.living_room_*,switch.bedroom_*`)
   - **EXCLUDED_ENTITIES**: (Optional) Comma-separated patterns to exclude (e.g., `sensor.*_battery,*_update`)

4. Click **Save** - the tool will automatically validate the connection

## Configuration Examples

### Example 1: Basic Configuration
```
HA_URL: http://homeassistant.local:8123
HA_TOKEN: eyJhbGc...your_token_here
DISCOVER_DOMAINS: light,switch,climate,sensor
INCLUDED_ENTITIES: (leave blank to include all)
EXCLUDED_ENTITIES: (leave blank to exclude none)
```

### Example 2: Only Living Room and Bedroom Devices
```
HA_URL: http://192.168.1.50:8123
HA_TOKEN: eyJhbGc...your_token_here
DISCOVER_DOMAINS: light,switch,climate,media_player
INCLUDED_ENTITIES: light.living_room_*,light.bedroom_*,switch.living_room_*,switch.bedroom_*,climate.living_room,media_player.living_room_tv
EXCLUDED_ENTITIES: 
```

### Example 3: Exclude Battery and Update Sensors
```
HA_URL: https://ha.mydomain.com
HA_TOKEN: eyJhbGc...your_token_here
DISCOVER_DOMAINS: light,switch,climate,sensor,binary_sensor
INCLUDED_ENTITIES: 
EXCLUDED_ENTITIES: sensor.*_battery,sensor.*_update,binary_sensor.*_update
```

## Usage Examples

Once configured, you can interact with Home Assistant through natural language conversations with your LLM:

### Discovery Commands

**"Show me all my Home Assistant devices"**
- LLM will call `discover_entities()` to list all configured entities

**"Refresh my Home Assistant entity list"**
- LLM will call `discover_entities(refresh=True)` to force a cache refresh

### State Queries

**"What's the temperature in the living room?"**
- LLM will search for temperature sensors in the living room and call `get_entity_state()`

**"Are the bedroom lights on?"**
- LLM will check the state of bedroom lights

**"Show me the current state of the thermostat"**
- LLM will query the climate entity and display all attributes

### Control Commands

**"Turn on the living room lights"**
- LLM will call `call_service(domain="light", service="turn_on", entity_id="light.living_room")`

**"Turn off all bedroom lights"**
- LLM will identify bedroom lights and call the turn_off service

**"Set the living room brightness to 50%"**
- LLM will call `call_service(domain="light", service="turn_on", entity_id="light.living_room", additional_data='{"brightness_pct": 50}')`

**"Set the thermostat to 72 degrees"**
- LLM will call `call_service(domain="climate", service="set_temperature", entity_id="climate.thermostat", additional_data='{"temperature": 72}')`

**"Open the garage door"**
- LLM will call `call_service(domain="cover", service="open_cover", entity_id="cover.garage_door")`

**"Toggle the kitchen switch"**
- LLM will call `call_service(domain="switch", service="toggle", entity_id="switch.kitchen")`

### Advanced Commands

**"What services are available for lights?"**
- LLM will call `list_services(domain="light")` to show all light services

**"Show me all available Home Assistant services"**
- LLM will call `list_services()` without a domain filter

## Tool Functions

### 1. `discover_entities(refresh=False)`
Discover and list all configured Home Assistant entities.

**Parameters:**
- `refresh` (bool): Force refresh the entity cache (default: False)

**Returns:** Formatted list of entities organized by domain

### 2. `get_entity_state(entity_id)`
Get the current state and attributes of a specific entity.

**Parameters:**
- `entity_id` (str): The entity_id to query (e.g., 'light.living_room')

**Returns:** Detailed state information including all attributes

### 3. `call_service(domain, service, entity_id, additional_data=None)`
Call a Home Assistant service on one or more entities.

**Parameters:**
- `domain` (str): Service domain (e.g., 'light', 'switch', 'climate')
- `service` (str): Service name (e.g., 'turn_on', 'turn_off', 'set_temperature')
- `entity_id` (str): Target entity_id or comma-separated list
- `additional_data` (str, optional): JSON string with extra service data

**Returns:** Confirmation message

### 4. `list_services(domain=None)`
List available Home Assistant services.

**Parameters:**
- `domain` (str, optional): Filter by domain (e.g., 'light')

**Returns:** Formatted list of services with descriptions

## Common Service Examples

### Lights
```python
# Turn on
call_service("light", "turn_on", "light.living_room")

# Turn on with brightness (0-255 or use brightness_pct for 0-100)
call_service("light", "turn_on", "light.living_room", '{"brightness_pct": 75}')

# Turn on with color
call_service("light", "turn_on", "light.living_room", '{"rgb_color": [255, 0, 0]}')

# Turn off
call_service("light", "turn_off", "light.living_room")
```

### Climate
```python
# Set temperature
call_service("climate", "set_temperature", "climate.thermostat", '{"temperature": 72}')

# Set HVAC mode
call_service("climate", "set_hvac_mode", "climate.thermostat", '{"hvac_mode": "heat"}')

# Turn off
call_service("climate", "turn_off", "climate.thermostat")
```

### Covers
```python
# Open cover
call_service("cover", "open_cover", "cover.garage_door")

# Close cover
call_service("cover", "close_cover", "cover.garage_door")

# Set position (0-100)
call_service("cover", "set_cover_position", "cover.living_room_blinds", '{"position": 50}')
```

### Media Players
```python
# Play
call_service("media_player", "media_play", "media_player.living_room_tv")

# Pause
call_service("media_player", "media_pause", "media_player.living_room_tv")

# Set volume (0.0-1.0)
call_service("media_player", "volume_set", "media_player.living_room_tv", '{"volume_level": 0.5}')
```

## Troubleshooting

### Connection Issues

**Error: "Failed to connect to Home Assistant"**
- Verify your HA_URL is correct and accessible from Open WebUI
- Ensure Home Assistant is running
- Check if the URL includes `http://` or `https://`
- Try accessing the URL in a browser from the same network

**Error: "401 Unauthorized"**
- Your access token is invalid or expired
- Create a new long-lived access token in Home Assistant
- Make sure you copied the entire token

### Entity Discovery Issues

**"No entities found matching the configured filters"**
- Check your INCLUDED_ENTITIES and EXCLUDED_ENTITIES patterns
- Verify DISCOVER_DOMAINS includes the domains you want
- Use `refresh=True` when discovering entities to bypass cache

### Network Considerations

- If Open WebUI is running in Docker, ensure it can reach your Home Assistant instance
- You may need to use the Docker host IP instead of `localhost`
- For Docker Desktop: use `host.docker.internal` instead of `localhost`
- For Docker on Linux: use the bridge network IP or host network mode

## Security Notes

- **Never share your long-lived access token**
- Consider creating a dedicated Home Assistant user with limited permissions for Open WebUI
- Use HTTPS for your Home Assistant URL when possible
- The token is stored in Open WebUI's database - ensure your Open WebUI instance is secure

## Version Compatibility

- **Open WebUI**: v0.6.4+
- **Home Assistant**: 2025.12+
- **Python**: 3.11+ (bundled with Open WebUI)

## Requirements

- `aiohttp`: Async HTTP library (automatically installed by Open WebUI)

## License

MIT License - Feel free to modify and distribute

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify your Home Assistant logs for API errors
3. Check Open WebUI logs for tool execution errors
4. Ensure your Home Assistant version is 2025.12 or later

## Changelog

### Version 1.0.0
- Initial release
- Entity discovery with domain filtering
- Include/exclude patterns with wildcard support
- State querying
- Service execution
- Connection validation
- Smart caching (5-minute cache duration)
- Support for all major Home Assistant domains
