## [2.0.0] - 2026-01-08
### Added
- Uses Home Assistant Websockets instead of just the REST API
- Add thresholds for entity detection by area and globally
- Added better logging

### Fixed
- Entity list synchronisation fixed to follow filters and domains.
- Removed `automation` and `script` from domain defaults to reduce the entity list. (Can still be manually added by the admins in Valves)

## [1.2.0] - 2026-01-07
### Added
- Synchronising entity list with Home Assistant
- Include and Exclude filters
- Allows LLM to inspect the entity list
- Restrict entity visibility to LLM by domain

### Fixed
- Improved entity detection using fuzzy word matching
- Added retries with broader fuzzy word matching if entity not detected

## [1.0.0] - 2026-01-06
### Added
- Home Assistant REST API connection.
- Device detection
- Entity detection
