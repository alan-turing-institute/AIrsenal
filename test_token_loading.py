#!/usr/bin/env python3
"""
Test that the cached token loading works correctly
"""

import os
import json
import tempfile
from airsenal.framework.env import AIRSENAL_HOME

# Create a fake token file for testing
token_file = os.path.join(AIRSENAL_HOME, ".fpl_auth_token")

print(f"Token file location: {token_file}")
print(f"Token file exists: {os.path.exists(token_file)}")

if os.path.exists(token_file):
    print("\nCurrent token file contents:")
    with open(token_file, 'r') as f:
        print(json.dumps(json.load(f), indent=2))
else:
    print("\nNo token file found - this is normal if you haven't run extract_browser_auth.py yet")

print("\n" + "=" * 70)
print("Testing FPLDataFetcher initialization...")
print("=" * 70)

from airsenal.framework.data_fetcher import FPLDataFetcher

fetcher = FPLDataFetcher()

print(f"\nLogin status: {fetcher.logged_in}")
print(f"Login failed: {fetcher.login_failed}")
print(f"Has auth headers: {bool(fetcher.headers)}")

if fetcher.headers:
    auth_header = fetcher.headers.get('X-API-Authorization', '')
    if auth_header:
        print(f"Auth header preview: {auth_header[:50]}...")

print("\n" + "=" * 70)
print("Summary:")
print("=" * 70)

if fetcher.logged_in:
    print("✓ Successfully loaded cached authentication token!")
    print("✓ You can now use AIrsenal with full API access")
elif os.path.exists(token_file):
    print("⚠ Token file exists but is invalid/expired")
    print("→ Run: python extract_browser_auth.py")
else:
    print("ℹ No cached token found")
    print("→ To enable API access, run: python extract_browser_auth.py")
    print("→ Or continue without login (database-only mode)")
