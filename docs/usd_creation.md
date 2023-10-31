# Working with USD

To save USD, first set the part of the USD we are interested in to "default prim" by right clicking on the part of the USD we are interested in and selecting "Set as Default Prim". Then go to "save as" or "collect as". The later can be used to also export materials and textures.

# Converting existing assets to instanceable assets

Follow this guide:
https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs/blob/main/docs/instanceable_assets.md#converting-existing-assets

**Note:** omniisaacgymenvs has to be installed in the isaac provided python env.

In the script editor, enter the following code and run:

```python
from omniisaacgymenvs.utils.usd_utils.create_instanceable_assets import convert_asset_instanceable
convert_asset_instanceable(
    asset_usd_path="/home/piperunner/Projects/research/project-mirage/assets/usd/franka_lidar.usd",
    source_prim_path="/franka",
    save_as_path="/home/piperunner/Projects/research/project-mirage/assets/usd/franka_lidar_instanceable.usd"
)
```
