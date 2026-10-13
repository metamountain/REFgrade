"""Relay-Poller für die 4070: holt Nachrichten ab, zeigt sie an, antwortet.

Aufruf: python relay_poller.py
Läuft endlos, pollt alle 3 Sekunden.
"""
import json
import sys
import time
import urllib.request
import urllib.error

RELAY = "http://127.0.0.1:8764"
SESSION = "refgrade"
POLL_INTERVAL = 3


def poll():
    try:
        req = urllib.request.Request(
            f"{RELAY}/poll?session={SESSION}",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            data = json.loads(r.read())
            return data.get("messages", [])
    except (urllib.error.URLError, OSError):
        return []


def send(msg: str):
    try:
        req = urllib.request.Request(
            f"{RELAY}/send",
            data=json.dumps({"from": SESSION, "to": "laptop", "message": msg}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read())
    except (urllib.error.URLError, OSError) as e:
        print(f"  [send failed: {e}]")
        return None


def main():
    print(f"[relay-poller] session={SESSION} relay={RELAY}")
    print("[relay-poller] waiting for messages... (Ctrl+C to stop)")
    while True:
        msgs = poll()
        for m in msgs:
            print(f"\n=== MESSAGE from {m.get('from', '?')} ===")
            print(m.get("message", ""))
            print("=====================================\n")
            # Auto-Antwort: Status melden
            send("ACK: Nachricht erhalten. Führe aus...")
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[relay-poller] stopped.")
