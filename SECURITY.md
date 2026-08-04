# Security Guidelines

## Important: Do not leak secrets!

Never zip, share, or commit the root of this project without first ensuring the following files and directories are excluded:
- `.env`
- `.env.local`
- `venv/` (and any other virtual environment directories)
- `frontend/node_modules/`
- Any data files containing sensitive information or large datasets

The `.env` file contains sensitive database credentials and API keys. Treat it with the utmost care.
