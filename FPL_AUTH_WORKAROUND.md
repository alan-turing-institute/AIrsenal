# FPL Authentication Workaround

## Problem

As of October 2025, the Fantasy Premier League (FPL) authentication API has changed to implement stronger bot protection. The new login flow requires JavaScript execution and browser-based authentication that cannot be automated with simple HTTP requests.

## Symptoms

- `KeyError: 'interactionToken'` when trying to login
- HTTP 403 errors when accessing authenticated endpoints
- Warnings about login failures in AIrsenal

## Workaround Solution

You can manually extract your authentication token from your browser and use it with AIrsenal.

### Step 1: Extract Your Browser Token

Run the extraction script:

```bash
uv run python extract_browser_auth.py
```

Follow the on-screen instructions:

1. **Login via Browser**: Open https://fantasy.premierleague.com/ and login normally
2. **Extract Token**: Use browser DevTools to extract your access token (the script provides detailed instructions)
3. **Validate & Save**: The script will validate your token and save it for AIrsenal to use

### Step 2: Use AIrsenal Normally

Once you've saved your token, AIrsenal will automatically use it:

```bash
# Update database (doesn't require login)
uv run airsenal_update_db

# Run optimization (will use cached token)
uv run airsenal_run_optimization --weeks_ahead 3

# Make transfers (if token is valid)
uv run airsenal_make_transfers
```

## How It Works

1. The `extract_browser_auth.py` script saves your token to `~/.airsenal/.fpl_auth_token`
2. When `FPLDataFetcher` initializes, it checks for this cached token
3. If found and valid, it uses the token instead of trying to login
4. If the token is invalid/expired, it falls back to database-only mode

## Token Lifespan

Browser tokens typically expire after:

- **A few hours** if you're actively using the site
- **Several days** if idle
- **When you logout** from the FPL website

When your token expires, you'll see warnings and need to re-run the extraction script.

## Security Notes

⚠️ **Your token file contains your FPL session credentials**

- Keep `~/.airsenal/.fpl_auth_token` private
- Don't commit it to git
- Don't share it with others
- Delete it when you're done: `rm ~/.airsenal/.fpl_auth_token`

## Alternative: Database-Only Mode

If you don't want to extract tokens, you can use AIrsenal without login:

```bash
# Update all public data (works without login)
uv run airsenal_update_db

# Run optimization with database data
uv run airsenal_run_optimization --weeks_ahead 3
```

**Limitations**:

- Uses last saved squad data (may be out of date)
- Can't make transfers via API
- Can't access league data
- Free transfer count may be inaccurate

## Future Solution

The AIrsenal maintainers will need to implement one of:

1. **Browser automation** (Selenium/Playwright) for full login flow
2. **Alternative API discovery** if FPL provides a different endpoint
3. **Manual token workflow** (what this workaround provides)

## Extracting Tokens Manually (Advanced)

If you prefer not to use the script, you can manually extract tokens:

### Method 1: Network Tab (Most Reliable)

1. Login to https://fantasy.premierleague.com/
2. Press F12 → Network tab
3. Refresh the page or navigate to https://fantasy.premierleague.com/my-team
4. Look for a request to `me` or `my-team`
5. Click it → Headers → Request Headers
6. Find `X-API-Authorization: Bearer <token>`
7. Copy everything after `Bearer `

### Method 2: Console (May not work)

1. Login to https://fantasy.premierleague.com/
2. Press F12 → Console tab
3. Try pasting and running:
   ```javascript
   JSON.parse(
     localStorage.getItem(
       "oidc.user:https://account.premierleague.com/:bfcbaf69-aade-4c1b-8f00-c1cb8a193030"
     )
   ).access_token;
   ```
4. If you get an error about `null`, use Method 1 instead

### Method 3: Application/Storage Tab

1. Login to https://fantasy.premierleague.com/
2. Press F12 → Application tab (Chrome) or Storage tab (Firefox)
3. Expand Local Storage → https://fantasy.premierleague.com
4. Find key starting with `oidc.user:`
5. Copy the `access_token` value

Then create `~/.airsenal/.fpl_auth_token`:

```json
{
  "access_token": "your-long-token-here",
  "team_id": 123456,
  "player_name": "Your Name"
}
```

## Troubleshooting

**Token doesn't work immediately:**

- Restart any running AIrsenal processes
- Check the token file exists: `ls -la ~/.airsenal/.fpl_auth_token`
- Verify token format is valid JSON

**Token keeps expiring:**

- This is normal FPL behavior
- Re-extract when needed (takes ~1 minute)
- Consider using database-only mode for non-critical operations

**Script can't find airsenal module:**

- Run from the AIrsenal directory
- Always use: `uv run python extract_browser_auth.py`

## Questions?

- Check the [AIrsenal documentation](https://github.com/alan-turing-institute/AIrsenal)
- Open an issue on GitHub
- This is a temporary workaround while the FPL API situation is resolved
