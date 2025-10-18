# Quick Start: Using AIrsenal with Browser Token Authentication

## What Happened?

The FPL API changed in October 2025 and now blocks automated logins. I've created a workaround that lets you extract your authentication from your browser and use it with AIrsenal.

## Quick Solution (5 minutes)

### Option 1: Extract Browser Token (Recommended for full functionality)

**Step 1** - Run the extraction helper:

```bash
uv run python extract_browser_auth.py
```

Tip: If validation seems slow or you don't see output right away, you can try:

- Increase timeout: `uv run python extract_browser_auth.py --timeout 20`
- Skip validation: `uv run python extract_browser_auth.py --no-validate`
- Show errors: `uv run python extract_browser_auth.py --debug`

**Step 2** - Follow the prompts:

1. Login to FPL in your browser (https://fantasy.premierleague.com/)
2. Use browser DevTools to extract your token (detailed instructions provided)
3. Paste the token when prompted
4. Script validates (may take up to 10s by default) and saves it

**Step 3** - Use AIrsenal normally:

```bash
uv run airsenal_run_optimization --weeks_ahead 3
```

### Option 2: Database-Only Mode (Quick, but limited)

Just update your database and run optimization:

```bash
# Update database with public data (no login needed)
uv run airsenal_update_db

# Run optimization with database data
uv run airsenal_run_optimization --weeks_ahead 3
```

**Limitations**: Squad data may be slightly out of date, can't make transfers via API

## Files Created

I've created the following helper files:

1. **`extract_browser_auth.py`** - Interactive script to extract and save your browser token
2. **`test_token_loading.py`** - Test if your token is loaded correctly
3. **`FPL_AUTH_WORKAROUND.md`** - Detailed documentation
4. **Modified `data_fetcher.py`** - Now automatically loads cached tokens

## How to Extract Your Token (Manual Method)

If you want to do it manually without the script:

### Method 1: Using Network Tab (Most Reliable)

1. Login to https://fantasy.premierleague.com/
2. Press **F12** (or Cmd+Option+I on Mac)
3. Click **Network** tab
4. Navigate to your team: https://fantasy.premierleague.com/my-team
5. In the Network tab, look for a request to **`me`** or **`my-team`**
6. Click on it, then click **Headers** section
7. Scroll down to find **Request Headers**
8. Find `X-API-Authorization: Bearer <very-long-token>`
9. Copy everything AFTER `Bearer ` (the long token string)

### Method 2: Using Local Storage (If Available)

1. Login to https://fantasy.premierleague.com/
2. Press **F12** (or Cmd+Option+I on Mac)
3. Click **Application** tab (Chrome) or **Storage** tab (Firefox)
4. In the sidebar, expand **Local Storage**
5. Click on `https://fantasy.premierleague.com`
6. Look for a key starting with `oidc.user:`
7. Click on it and look at the JSON value
8. Find and copy the `access_token` value

### Method 3: Using Console (Alternative)

1. Login to https://fantasy.premierleague.com/
2. Press **F12** (or Cmd+Option+I on Mac)
3. Click **Console** tab
4. Try this first:

```javascript
JSON.parse(
  localStorage.getItem(
    "oidc.user:https://account.premierleague.com/:bfcbaf69-aade-4c1b-8f00-c1cb8a193030"
  )
).access_token;
```

5. If that gives an error, try listing all localStorage keys:

```javascript
Object.keys(localStorage).filter((k) => k.includes("oidc"));
```

6. Then use the correct key name in the first command

### Save it:

Create/edit `~/.airsenal/.fpl_auth_token` (on Mac: `~/Library/Application Support/airsenal/.fpl_auth_token`):

```json
{
  "access_token": "paste-your-very-long-token-here",
  "team_id": 7770441,
  "player_name": "Your Name"
}
```

## Testing Your Setup

```bash
# Test if token is loaded
uv run python test_token_loading.py

# Try running optimization
uv run airsenal_run_optimization --weeks_ahead 3
```

## When Tokens Expire

Tokens typically last a few hours to days. When expired:

1. You'll see "token invalid" warnings
2. Simply re-run `uv run python extract_browser_auth.py`
3. Takes ~1 minute to refresh

## What I Changed in the Code

1. **`data_fetcher.py`**:

   - Added `_try_load_cached_token()` method
   - Called during `__init__` to check for cached tokens
   - Updated error messages to suggest token extraction
   - Updated browser impersonation to `chrome131`

2. **Created helper scripts**:
   - `extract_browser_auth.py` - User-friendly token extraction
   - `test_token_loading.py` - Verify token loading works
   - `FPL_AUTH_WORKAROUND.md` - Full documentation

## Next Steps

**Choose your path:**

### Path A: Full API Access (with browser token)

```bash
# 1. Extract token
uv run python extract_browser_auth.py

# 2. Verify it works
uv run python test_token_loading.py

# 3. Run optimization
uv run airsenal_run_optimization --weeks_ahead 3
```

### Path B: Database-Only (simpler, but limited)

```bash
# Just run with database data
uv run airsenal_update_db
uv run airsenal_run_optimization --weeks_ahead 3
```

## Troubleshooting

**"ModuleNotFoundError: No module named 'airsenal'"**
→ Always use: `uv run python extract_browser_auth.py` (not just `python`)

**Pasted token and nothing happens**
→ Validation can take up to the timeout (default 10s). You can:

- Increase timeout, e.g. `uv run python extract_browser_auth.py --timeout 20`
- Skip validation and just save with `--no-validate`
- Run with `--debug` to see detailed errors

**Token doesn't work after saving**
→ Restart any running AIrsenal processes
→ Check token file: `cat ~/Library/Application\ Support/airsenal/.fpl_auth_token`

**Token expires quickly**
→ Normal behavior, FPL sessions have limited lifetime
→ Re-extract when needed (takes ~1 min)

## Need Help?

See `FPL_AUTH_WORKAROUND.md` for detailed documentation, or ask me any questions!
