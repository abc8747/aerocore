# --8<-- [start:input0]
import torch

from aerocore.acropole import fuel_flow, load_aircraft_database, load_model
from aerocore.utils import tree_map

device = torch.device("cuda")
model = tree_map(lambda array: torch.from_numpy(array).to(device), load_model())
aircraft = load_aircraft_database()["A320"]
print(
    fuel_flow(
        model=model,
        aircraft=aircraft,
        groundspeed=torch.tensor([180.0, 450.0, 250.0], device=device),
        altitude=torch.tensor([0.0, 30_000.0, 40_000.0], device=device),
        vertical_rate=torch.tensor([3_000.0, 0.0, -2_000.0], device=device),
        xp=torch,
    )
    .cpu()
    .numpy()
)
# --8<-- [end:input0]

"""
--8<-- [start:output0]
[1.9122902  0.7568858  0.09444637]
--8<-- [end:output0]
"""
