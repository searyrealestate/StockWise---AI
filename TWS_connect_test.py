# TWS_connect_test.py
"""
TWS API Minimal Connection Test
===============================

This script serves as a simple, standalone diagnostic tool to test the fundamental
API connection to a running instance of Interactive Brokers Trader Workstation (TWS)
or IB Gateway.

It is designed to be the simplest possible client that uses the official `ibapi`
library to confirm that a connection handshake can be successfully established.

How it Works:
-------------
1.  It defines a minimal `EClient` and `EWrapper` class.
2.  It attempts to connect to TWS on the specified host and port.
3.  It starts the required API background thread.
4.  It then waits for the `nextValidId` callback from the TWS API. The successful
    receipt of this callback is the definitive confirmation that the API
    connection and handshake are complete and successful.
5.  If `nextValidId` is not received within a 20-second timeout, the test fails.

Purpose:
--------
This script is useful for troubleshooting connection issues in a clean environment,
separate from any more complex application logic. If this script fails, it indicates
a problem with the TWS settings, the network, or the environment, rather than a
bug in the main application code.

Pre-conditions for a successful test:
-------------------------------------
-   TWS or IB Gateway must be running and logged in.
-   In TWS, the API settings must be configured correctly:
    -   "Enable ActiveX and Socket Clients" must be checked.
    -   The IP "127.0.0.1" must be listed in the "Trusted IP Addresses".
-   The port number in this script (default: 7497) must match the "Socket port"
    in the TWS API settings.

"""

import threading
import time
from ibapi.client import EClient
from ibapi.wrapper import EWrapper


class TWSConnectionTest(EWrapper, EClient):
    """A minimal client to test the TWS API connection."""

    def __init__(self):
        EClient.__init__(self, self)
        self.connection_acknowledged = threading.Event()

    def nextValidId(self, orderId: int):
        """This callback is the definitive proof that the connection handshake is complete."""
        super().nextValidId(orderId)
        print("✅ SUCCESS: Received nextValidId. Connection handshake is complete.")
        self.connection_acknowledged.set()

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        """Handles any errors from the TWS API."""
        super().error(reqId, errorCode, errorString, advancedOrderRejectJson)
        print(f"❌ TWS API ERROR: Code={errorCode}, Message='{errorString}'")


def run_test():
    print("--- Starting Minimal TWS Connection Test ---")
    client = TWSConnectionTest()

    # --- Configuration ---
    host = "127.0.0.1"
    port = 7497  # Default for TWS Paper Trading. Change if your port is different.
    client_id = 101  # Use a unique ID that is not in use.

    print(f"[*] Attempting to connect to TWS at {host}:{port} with Client ID {client_id}...")
    client.connect(host, port, client_id)

    # The TWS API runs in a separate thread.
    api_thread = threading.Thread(target=client.run, daemon=True)
    api_thread.start()
    print("[*] API thread started. Waiting for connection acknowledgement...")

    # Wait for the nextValidId callback to be triggered.
    # This is the most reliable way to confirm a successful connection.
    acknowledged = client.connection_acknowledged.wait(timeout=20)

    print("\n--- Test Result ---")
    if acknowledged:
        print("🎉 CONNECTION SUCCESSFUL")
    else:
        print("🔥 CONNECTION FAILED: Timed out waiting for handshake acknowledgement from TWS.")
        print("\n   Troubleshooting:")
        print("   1. Is TWS running and logged in?")
        print("   2. Are API settings correct (Enable ActiveX, Trusted IP 127.0.0.1)?")
        print(f"  3. Is Client ID {client_id} already in use in TWS?")
        print("   4. Is a firewall or antivirus blocking the connection?")

    client.disconnect()
    print("[*] Disconnected.")


if __name__ == "__main__":
    run_test()