# Quick Start: Using AIrsenal with Browser Token Authentication

## What Happened?

The FPL API changed in October 2025 and now blocks automated logins. I've created a workaround that lets you extract your authentication from your browser and use it with AIrsenal.

## Quick Solution (5 minutes)

### Option 1: Extract Browser Token (Recommended for full functionality)

**Step 1** - Run the extraction helper:

```bash
uv run python extract_browser_auth.py
```

**OR** if pasting doesn't work, pass the token directly:

```bash
uv run python extract_browser_auth.py --token YOUR_TOKEN_HERE
```

**Step 2** - Follow the prompts:

1. Login to FPL in your browser (https://fantasy.premierleague.com/)
2. Extract your token using browser DevTools (instructions provided by the script)
3. If prompted, paste the token and press ENTER
4. Script validates and saves it

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

### Using Network Tab

1. Login to https://fantasy.premierleague.com/
2. Press **F12** (or Cmd+Option+I on Mac) to open Developer Tools
3. Click **Network** tab
4. Refresh the page (F5 or Cmd+R)
5. In the Network list, look for a request that shows **your team ID number**
   - It will be a 7-digit number like `1234567`
   - This is usually one of the first requests
6. Click on that request
7. In the right panel, click the **Headers** tab
8. Scroll down to **Request Headers** section
9. Find `X-API-Authorization: Bearer <very-long-token>`
10. Copy ONLY the token part (everything after `Bearer `)
    - Don't include the word "Bearer"
    - The token should be a very long string of random characters

### Save it manually:

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

**Pasting token doesn't work**
→ Use the --token parameter instead:
```bash
uv run python extract_browser_auth.py --token YOUR_TOKEN_HERE
```

**Can't find the network request**
→ The request name is your team ID (a 7-digit number), not "me" or "my-team"
→ Try refreshing the page to see new requests appear

**Token doesn't work after saving**
→ Restart any running AIrsenal processes
→ Check token file: `cat ~/Library/Application\ Support/airsenal/.fpl_auth_token`

**Token expires quickly**
→ Normal behavior, FPL sessions have limited lifetime
→ Re-extract when needed (takes ~1 min)

## Need Help?

See `FPL_AUTH_WORKAROUND.md` for detailed documentation, or ask me any questions!
