from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology
import torch




base_model = GroundingDINO(ontology=CaptionOntology({"cardboard box": "box"}), )
# box_detector = GroundingDINO(ontology=ontology)
base_model.label("./test", extension=".jpg")
