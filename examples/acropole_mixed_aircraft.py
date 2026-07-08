# --8<-- [start:input0]
import aerocore.acropole as ac
import numpy as np

model = ac.load_model()
database = ac.load_aircraft_database()
aircraft_codes = ("A320", "AT76", "B738", "A320")
aircraft = [database[code] for code in aircraft_codes]

groundspeed = np.asarray([180.0, 180.0, 450.0, 250.0])
altitude = np.asarray([0.0, 12_000.0, 30_000.0, 40_000.0])
vertical_rate = np.asarray([3_000.0, 1_500.0, 0.0, -2_000.0])
airspeed = np.asarray([190.0, 190.0, 460.0, 260.0])
groundspeed_rate = np.asarray([2.0, 1.0, 0.0, -1.0])
airspeed_rate = np.asarray([1.5, 0.8, 0.0, -0.8])
mass = np.asarray([65_000.0, 18_000.0, 65_000.0, 60_000.0])

oew = np.asarray([item.oew for item in aircraft])
max_tow = np.asarray([item.max_tow for item in aircraft])
features = np.column_stack(
    (
        [item.engine_type for item in aircraft],
        vertical_rate / 60,
        groundspeed_rate,
        airspeed_rate,
        [item.wing_area for item in aircraft],
        [item.max_alt for item in aircraft],
        [item.max_tas for item in aircraft],
        altitude,
        groundspeed,
        airspeed,
        vertical_rate,
        (mass - oew) / (max_tow - oew),
    )
)
standardised = ac.standardise(model.standardisation, features)
normalised_flow = ac.predict_standardised(model.weights, standardised, xp=np)
fuel_scale = np.asarray(
    [item.fuel_flow_per_engine_to * item.engine_count for item in aircraft]
)

print(normalised_flow * fuel_scale)
# --8<-- [end:input0]

"""
--8<-- [start:output0]
[2.04228825 0.23620985 0.80334929 0.11475204]
--8<-- [end:output0]
"""
