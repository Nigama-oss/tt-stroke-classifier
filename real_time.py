import asyncio
from bleak import BleakScanner, BleakClient
import joblib
import numpy as np

# Load your trained model
model, label_encoder = joblib.load('stroke_classifier_model.pkl')

# BLE UUIDs
SERVICE_UUID = '84582cd0-3df0-4e73-9496-29010d7445dd'
AX_UUID = '84582cd1-3df0-4e73-9496-29010d7445dd'
AY_UUID = '84582cd2-3df0-4e73-9496-29010d7445dd'
AZ_UUID = '84582cd3-3df0-4e73-9496-29010d7445dd'
GX_UUID = '84582cd4-3df0-4e73-9496-29010d7445dd'
GY_UUID = '84582cd5-3df0-4e73-9496-29010d7445dd'
GZ_UUID = '84582cd6-3df0-4e73-9496-29010d7445dd'

# Buffer for incoming data
imu_data = {
    'ax': None,
    'ay': None,
    'az': None,
    'gx': None,
    'gy': None,
    'gz': None
}

# Handle incoming data
def notification_handler(uuid, data):
    global imu_data
    value = np.frombuffer(data, dtype=np.float32)[0]
    if uuid == AX_UUID:
        imu_data['ax'] = value
    elif uuid == AY_UUID:
        imu_data['ay'] = value
    elif uuid == AZ_UUID:
        imu_data['az'] = value
    elif uuid == GX_UUID:
        imu_data['gx'] = value
    elif uuid == GY_UUID:
        imu_data['gy'] = value
    elif uuid == GZ_UUID:
        imu_data['gz'] = value

    # When we have a full set of data
    if all(v is not None for v in imu_data.values()):
        feature_vector = np.array([[imu_data['ax'], imu_data['ay'], imu_data['az'],
                                    imu_data['gx'], imu_data['gy'], imu_data['gz']]])
        prediction = model.predict(feature_vector)
        label = label_encoder.inverse_transform(prediction)[0]
        print(f"Predicted Stroke: {label}")

        # Clear for next prediction
        for key in imu_data:
            imu_data[key] = None

async def run_ble_predictor():
    print("Scanning for devices...")
    devices = await BleakScanner.discover()

    target_device = None
    for d in devices:
        print(f"Found: {d.address} {d.name}")
        if d.name == "Settorezero_IMU":
            target_device = d
            break

    if not target_device:
        print("Target device not found.")
        return

    async with BleakClient(target_device.address) as client:
        print(f"Connected to {target_device.address}")

        # Start notifications
        await client.start_notify(AX_UUID, lambda s, d: notification_handler(AX_UUID, d))
        await client.start_notify(AY_UUID, lambda s, d: notification_handler(AY_UUID, d))
        await client.start_notify(AZ_UUID, lambda s, d: notification_handler(AZ_UUID, d))
        await client.start_notify(GX_UUID, lambda s, d: notification_handler(GX_UUID, d))
        await client.start_notify(GY_UUID, lambda s, d: notification_handler(GY_UUID, d))
        await client.start_notify(GZ_UUID, lambda s, d: notification_handler(GZ_UUID, d))

        print("Listening for IMU data...")
        while True:
            await asyncio.sleep(1)  # keep running

if __name__ == "__main__":
    asyncio.run(run_ble_predictor())
