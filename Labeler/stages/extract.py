"""Define observation extractor class."""
import re
import itertools
from collections import defaultdict
from tqdm import tqdm
from constants import *

import bioc


class Extractor(object):
    """Extract observations from impression sections of reports."""
    def __init__(self, mention_phrases_dir,  verbose=False):
        self.verbose = verbose
        self.mention_phrases= self.load_phrases(mention_phrases_dir, KEYWORD)    
        
        self.unmention_phrases = self.get_unmention_phrases(part = KEYWORD)

    def load_phrases(self, phrases_dir, observation):
        """Read in map from observations to phrases for matching."""
        observation2phrases = defaultdict(list)
        for phrases_path in phrases_dir.glob("*.txt"):
            with phrases_path.open() as f:
                for line in f:
                    phrase = line.strip().replace("_", " ")
                    if line:
                        observation2phrases[observation].append(phrase)

        if self.verbose:
            print(f"Loading {observation} phrases for "
                  f"{len(observation2phrases)} observations.")

        return observation2phrases

    def get_unmention_phrases(self, part):
        observation2phrases = defaultdict(list)

        for observation, phrases in self.mention_phrases.items():
            for phrase in phrases:
                if phrase in PHRASES:
                    observation2phrases[part].append("no " + phrase)
                    observation2phrases[part].append("absence of " + phrase)
                    observation2phrases[part].append("without " + phrase)
                    observation2phrases[part].append("without a " + phrase)

        return observation2phrases

    def overlaps_with_unmention(self, unmention_phrases, sentence, observation, start, end):
        """Return True if a given match overlaps with an unmention phrase."""
        unmention_overlap = False
        unmention_list = unmention_phrases.get(observation, [])
        for unmention in unmention_list:
            unmention_matches = re.finditer(unmention, sentence.text)
            for unmention_match in unmention_matches:
                unmention_start, unmention_end = unmention_match.span(0)
                #print(sentence.text, start, end, unmention_start, unmention_end)                  
                if start < unmention_end and end > unmention_start:
                    unmention_overlap = True
                    break  # break early if overlap is found
            if unmention_overlap:
                break  # break early if overlap is found

        return unmention_overlap

    def add_match(self, impression, sentence, ann_index, phrase,
                  observation, start, end):
        """Add the match data and metadata to the impression object
        in place."""
        annotation = bioc.BioCAnnotation()
        annotation.id = ann_index
        annotation.infons['CUI'] = None
        annotation.infons['semtype'] = None
        annotation.infons['term'] = phrase
        annotation.infons[OBSERVATION] = observation
        annotation.infons['annotator'] = 'Phrase'
        length = end - start
        annotation.add_location(bioc.BioCLocation(sentence.offset + start,
                                                  length))
        annotation.text = sentence.text[start:start+length]

        impression.annotations.append(annotation)

    def unmention_match_test(self, passage, mention_phrases, unmention_phrases):
        for sentence in passage.sentences:
            obs_phrases = mention_phrases.items()
            for observation, phrases in obs_phrases:
                for phrase in phrases:
                    matches = re.finditer(phrase, sentence.text)
                    for match in matches:
                        start, end = match.span(0)
                        if self.overlaps_with_unmention(unmention_phrases, sentence, 
                                                            observation, start, end): 
                            return True # return early if overlap is found
        return False 

    def extract(self, collection):
        """Extract the observations in each report.

        Args:
            collection (BioCCollection): Impression passages of each report.

        Return:
            extracted_mentions
        """

        # The BioCCollection consists of a series of documents.
        # Each document is a report (just the Impression section
        # of the report.)
        documents = collection.documents
        if self.verbose:
            print("Extracting mentions...")
            documents = tqdm(documents)
        for document in documents:
            # Get the report section.
            report = document.passages[0]

            annotation_index = itertools.count(len(report.annotations))

            report_test = self.unmention_match_test(report, self.mention_phrases, self.unmention_phrases)

            #Extract passage only when there is no unmention phrases
            if report_test: # this is first image for sure.
                continue
            else:   
                #Extract Report   
                for sentence in report.sentences:
                    obs_phrases = self.mention_phrases.items()
                    for observation, phrases in obs_phrases:
                        for phrase in phrases:
                            matches = re.finditer(phrase, sentence.text)
                            for match in matches:
                                start, end = match.span(0)                
                                self.add_match(report,
                                               sentence,
                                               str(next(annotation_index)),
                                               phrase,
                                               observation,
                                               start,
                                               end)
