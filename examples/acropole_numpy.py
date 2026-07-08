# --8<-- [start:input0]
import numpy as np
from aerocore.acropole import fuel_flow, load_aircraft_database, load_model

model = load_model()
aircraft = load_aircraft_database()["A320"]
print(
    fuel_flow(
        model=model,
        aircraft=aircraft,
        groundspeed=np.asarray([180.0, 450.0, 250.0]),
        altitude=np.asarray([0.0, 30_000.0, 40_000.0]),
        vertical_rate=np.asarray([3_000.0, 0.0, -2_000.0]),
        xp=np,
    )
)
# --8<-- [end:input0]

"""
--8<-- [start:output0]
[1.91229012 0.75688555 0.09444639]
--8<-- [end:output0]
"""
