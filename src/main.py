from mujoco import viewer
import mujoco
import xml.etree.ElementTree as ET

from constants import paths

tree = ET.parse(paths.MODEL_PATH / 'sample.xml')
root = tree.getroot()
xml_string = ET.tostring(root, encoding='unicode')

model = mujoco.MjModel.from_xml_string(xml_string)

viewer.launch(model)