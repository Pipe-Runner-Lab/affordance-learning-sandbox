Follow this guide:
https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs/blob/main/docs/instanceable_assets.md#converting-existing-assets

In the script editor, enter the following code and run:

```bash
from omniisaacgymenvs.utils.usd_utils.create_instanceable_assets import convert_asset_instanceable
convert_asset_instanceable(
    asset_usd_path="/home/piperunner/Projects/research/project-mirage/src/mirage/assets/usd/franka_lidar.usd",
    source_prim_path="/Root/franka",
    save_as_path="/home/piperunner/Projects/research/project-mirage/src/mirage/assets/usd/franka_lidar_instanceable.usd"
)
```
