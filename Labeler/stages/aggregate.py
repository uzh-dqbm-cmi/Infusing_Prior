"""Define mention aggregator class."""
import numpy as np
from tqdm import tqdm

from constants import *


class Aggregator(object):
    """Aggregate mentions of observations from radiology reports."""
    def __init__(self, categories, verbose=False):
        self.categories = categories

        self.verbose = verbose

    def dict_to_vec(self, d):
        vec = []
        for category in self.categories:
            # There was a mention of the category.
            if category in d:
                label_list = d[category]
                # Only one label, no conflicts.
                if len(label_list) == 1:
                    vec.append(label_list[0])
                # Multiple labels.
                else:
                    # Case 1. There is negated and uncertain.
                    if NEGATIVE in label_list and UNCERTAIN in label_list:
                        vec.append(UNCERTAIN)
                    # Case 2. There is negated and positive.
                    elif NEGATIVE in label_list and POSITIVE in label_list:
                        vec.append(POSITIVE)
                    # Case 3. There is uncertain and positive.
                    elif UNCERTAIN in label_list and POSITIVE in label_list:
                        vec.append(POSITIVE)
                    # Case 4. All labels are the same.
                    else:
                        vec.append(label_list[0])

            # No mention of the category
            else:
                vec.append(NEGATIVE)

        return vec


    def aggregate(self, collection):
        documents = collection.documents
        if self.verbose:
            print("Aggregating mentions...")
            documents = tqdm(documents)

        label_dict = {"uid": [], "Keyword": [], "Label": []}
        for document in documents:
            label_dict["uid"].append(document.id)
            passage = document.passages[0]

            temp_label = []   

            phrase = ""
            uncertain_phrase = ""   
            for annotation in passage.annotations:
                category = annotation.infons[OBSERVATION]

                if POSITIVE in annotation.infons:
                    phrase += annotation.text + " "
                    temp_label.append(POSITIVE)
                else:
                    temp_label.append(NEGATIVE)

            #conclusion
            if POSITIVE in temp_label:
                label_dict["Keyword"].append(phrase)
                label_dict["Label"].append(POSITIVE)                    
            else: 
                label_dict["Keyword"].append("")
                label_dict["Label"].append(NEGATIVE)                  

        return label_dict
