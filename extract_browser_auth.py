#!/usr/bin/env python3
"""
Extract FPL authentication tokens from your browser for use with AIrsenal.

This script helps you manually login via browser and then use those credentials
with AIrsenal, working around the new bot-protection API changes.

Quick usage:
    uv run python extract_browser_auth.py

Advanced options:
    uv run python extract_browser_auth.py --timeout 10        # set validation timeout (seconds)
    uv run python extract_browser_auth.py --no-validate       # skip validation, just save token
    uv run python extract_browser_auth.py --token YOURTOKEN   # pass token via flag
    uv run python extract_browser_auth.py --debug             # verbose errors

Note: Validation may take a few seconds depending on network conditions.
"""

import json
import sys
import argparse
from curl_cffi import requests

API_HOME = "https://fantasy.premierleague.com/api"

def print_header(text):
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def print_step(step_num, text):
    print(f"\n[Step {step_num}] {text}")

def test_token(token: str, timeout: int = 10, debug: bool = False):
    """Test if an access token is valid."""
    try:
        session = requests.Session(impersonate="chrome131")
        headers = {"X-API-Authorization": f"Bearer {token}"}
        response = session.get(f"{API_HOME}/me/", headers=headers, timeout=timeout)
        if response.status_code == 200:
            data = response.json()
            if "player" in data:
                return True, data, None
            return False, None, "Token validated but no player info returned"
        return False, None, f"HTTP {response.status_code}: {response.text[:200]}"
    except Exception as e:
        return False, None, str(e)

def save_token(token, user_data):
    """Save the token to a file that AIrsenal can use"""
    import os
    from airsenal.framework.env import AIRSENAL_HOME
    
    # Create a token cache file
    token_file = os.path.join(AIRSENAL_HOME, ".fpl_auth_token")
    
    auth_data = {
        "access_token": token,
        "team_id": user_data.get("player", {}).get("entry"),
        "player_name": f"{user_data.get('player', {}).get('first_name', '')} {user_data.get('player', {}).get('last_name', '')}".strip(),
    }
    
    with open(token_file, "w") as f:
        json.dump(auth_data, f, indent=2)
    
    print(f"\n✓ Token saved to: {token_file}")
    return auth_data

def main():
    parser = argparse.ArgumentParser(description="Extract and cache FPL auth token for AIrsenal")
    parser.add_argument("--timeout", type=int, default=10, help="Validation timeout in seconds (default: 10)")
    parser.add_argument("--no-validate", action="store_true", help="Skip validation and save the token directly")
    parser.add_argument("--token", type=str, default=None, help="Provide token via CLI instead of interactive paste")
    parser.add_argument("--debug", action="store_true", help="Print detailed error messages on failures")
    args = parser.parse_args()

    print_header("FPL Browser Authentication Extractor")
    
    print("""
This tool helps you extract authentication tokens from your browser
to use with AIrsenal, working around the new FPL API restrictions.
    """)
    
    print_step(1, "Open your browser and login to FPL")
    print("""
    1. Open: https://fantasy.premierleague.com/
    2. Click 'Sign In' and login with your credentials
    3. Once logged in, navigate to: https://fantasy.premierleague.com/my-team
    """)
    
    input("\nPress ENTER when you are logged in and viewing your team...")
    
    print_step(2, "Extract the authentication token")
    print("""
Now we need to extract your access token from the browser using the Network tab.

    Using Network Tab:
    1. Press F12 (or Cmd+Option+I on Mac) to open Developer Tools
    2. Click the 'Network' tab
    3. Refresh the page (F5 or Cmd+R)
    4. In the Network list, look for a request that shows your team ID number
       (it will be a 7-digit number like '1234567')
    5. Click on that request
    6. In the right panel, click the 'Headers' tab
    7. Scroll down to the 'Request Headers' section
    8. Find the line 'X-API-Authorization: Bearer <very-long-token>'
    9. Copy ONLY the token part (everything after 'Bearer ')
       - Don't include the word 'Bearer' itself
       - The token should be a very long string of random characters
    """)
    
    print("\n" + "-" * 70)
    print("\nIMPORTANT: If pasting the token interactively doesn't work,")
    print("you can pass it as a parameter instead:")
    print("  uv run python extract_browser_auth.py --token YOUR_TOKEN_HERE")
    print("-" * 70)
    
    if args.token:
        token = args.token.strip()
        print("\nToken provided via --token flag.")
    else:
        token = input("\nPaste your access token here and press ENTER: ").strip()
    
    # Clean up the token (remove quotes if present)
    token = token.strip('"').strip("'").strip()
    
    if not token:
        print("\n✗ No token provided. Exiting.")
        sys.exit(1)
    print(f"\nToken received. Length: {len(token)} characters")

    if args.no_validate:
        print_step(3, "Skipping validation (per --no-validate)")
        dummy_user = {"player": {"entry": None, "first_name": "", "last_name": ""}}
        auth_data = save_token(token, dummy_user)
        print_step(4, "Next steps")
        print(f"""
Your authentication token has been saved!

To use it with AIrsenal, you need to modify the data_fetcher.py file
to load this token instead of trying to login.

Token details:
  - Team ID: {auth_data['team_id']}
  - Player: {auth_data['player_name']}
  - Token (first 20 chars): {token[:20]}...

IMPORTANT: This token will eventually expire (usually after a few hours or days).
When it does, you'll need to run this script again to get a fresh token.

You can now try running:
  uv run airsenal_run_optimization --weeks_ahead 3

Note: You may need to restart any running AIrsenal processes for the
token to be picked up.
            """)
        return

    print_step(3, "Validating token")
    print(f"Testing the token against FPL API (timeout {args.timeout}s)...", flush=True)

    is_valid, user_data, err = test_token(token, timeout=args.timeout, debug=args.debug)

    if is_valid:
        print("\n✓ Token is VALID!")
        print(f"✓ Team ID: {user_data.get('player', {}).get('entry')}")
        print(f"✓ Player: {user_data.get('player', {}).get('first_name', '')} {user_data.get('player', {}).get('last_name', '')}")

        save_choice = input("\nSave this token for AIrsenal to use? (y/n): ").lower()
        if save_choice == 'y':
            auth_data = save_token(token, user_data)
            print_step(4, "Next steps")
            print(f"""
Your authentication token has been saved!

You can now try running:
  uv run airsenal_run_optimization --weeks_ahead 3

If you encounter failures later, re-run this script to refresh the token.
            """)
        else:
            print("\n✓ Token validated but not saved.")
            print("You can run this script again when ready to save it.")
    else:
        print("\n✗ Token validation failed.")
        if args.debug and err:
            print(f"Reason: {err}")
        print("You can:")
        print("  - Re-try with a fresh token (ensure it's copied fully)")
        print("  - Use --timeout to increase the wait, e.g. --timeout 20")
        print("  - Skip validation and save anyway with --no-validate")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted. Exiting.")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
