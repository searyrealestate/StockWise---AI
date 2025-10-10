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
        self.has_critical_error = False
        self.error_messages = []

    def nextValidId(self, orderId: int):
        """This callback is the definitive proof that the connection handshake is complete."""
        super().nextValidId(orderId)
        print("SUCCESS: Received nextValidId. Connection handshake is complete.")
        self.connection_acknowledged.set()

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        """Handles any errors from the TWS API."""
        super().error(reqId, errorCode, errorString, advancedOrderRejectJson)
        if errorCode not in [2104, 2106, 2158, 2108, 1100, 1102]:
            self.has_critical_error = True
            self.error_messages.append(f"TWS API ERROR: Code={errorCode}, Message='{errorString}'")
            print(f"❌ {self.error_messages[-1]}")
        else:
            print(f"   (Info: Code={errorCode}, Message='{errorString}')")


def run_test():
    print("--- Starting Minimal TWS Connection Test ---")
    client = TWSConnectionTest()
    host = "127.0.0.1"
    port = 7497
    client_id = 101

    print(f"[*] Attempting to connect to TWS at {host}:{port} with Client ID {client_id}...")
    client.connect(host, port, client_id)

    # --- Set daemon=False to allow for graceful shutdown ---
    api_thread = threading.Thread(target=client.run, daemon=False)
    api_thread.start()
    print("[*] API thread started. Waiting for connection acknowledgement...")

    acknowledged = client.connection_acknowledged.wait(timeout=20)

    print("\n--- Test Result ---")
    if acknowledged and not client.has_critical_error:
        print("CONNECTION SUCCESSFUL")
    else:
        print("CONNECTION FAILED.")
        if not acknowledged:
            print("   - Reason: Timed out waiting for handshake from TWS.")
        critical_errors = [msg for msg in client.error_messages if "Market data farm" not in msg]
        for msg in critical_errors:
            print(f"   - Reason: {msg}")

    # --- Give the API thread a moment to process disconnect ---
    time.sleep(1)
    client.disconnect()
    print("[*] Disconnected.")


if __name__ == "__main__":
    run_test()