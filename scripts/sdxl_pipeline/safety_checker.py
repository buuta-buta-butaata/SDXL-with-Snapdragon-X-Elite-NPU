import logging
import numpy as np

from utils import tagger

logger = logging.getLogger(__name__)


class RatingChecker:
    nsfw_set = {"sensitive", "questionable", "explicit"}
    
    def __init__(self, model_path="wd14_tagger.onnx", csv_path="selected_tags.csv"):
        tags_dict_path = csv_path
        self.tagger = tagger.WD14Tagger(model_path, csv_path, [tagger.Category.RATING])
        
    def check_rating(self, image, threshold=0.5):
        """ Returns:
        - has_nsfw_concept: 
        - ratings: "sensitive" or "questionable" or "explicit"
        """
    
        if not isinstance(image, type(np.array)):
            image = self.tagger.preprocess_image(image)

        prompt = self.tagger.get_prompt(image, threshold=threshold)
        prompt = [x.strip() for x in prompt.split(",")]
        prompt_set = set(prompt)

        if len(self.nsfw_set & prompt_set) > 0:
            return True, list(self.nsfw_set & prompt_set)[0]

        return False, None
