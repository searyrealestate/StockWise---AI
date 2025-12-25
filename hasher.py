# hasher.py
import sys
# 1. This is the explicit, correct import
from streamlit_authenticator.utilities.hasher import Hasher

# Get passwords from command line arguments
if len(sys.argv) < 3:
    print("Usage: python hasher.py <password_for_user1> <password_for_user2>")
    sys.exit(1)

passwords_to_hash = [sys.argv[1], sys.argv[2]]

# 2. This is the correct method call:
# We call the 'hash_list' method *directly on the Hasher class*.
# We do not create an instance with Hasher()
# This avoids all previous errors.
hashed_passwords = Hasher.hash_list(passwords_to_hash)

print("Copy these hashed passwords into your config.yaml file:")
print(hashed_passwords)