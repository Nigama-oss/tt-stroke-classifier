import asyncio
from bleak import BleakScanner

async def simple_scan():
    print("Scanning for devices...")
    devices = await BleakScanner.discover(timeout=10)
    for d in devices:
        print(d)

asyncio.run(simple_scan())
