"""Define report loader class."""
import re
import bioc
import pandas as pd
from negbio.pipeline import text2bioc, ssplit, section_split

from constants import *

import bioc
from negbio.pipeline.text2bioc import *

class Loader(object):
    """Report impression loader."""
    def __init__(self, reports_path, extract_impression=False):
        self.reports_path = reports_path
        self.extract_impression = extract_impression
        self.punctuation_spacer = str.maketrans({key: f"{key} "
                                                 for key in ".,"})
        self.splitter = ssplit.NegBioSSplitter(newline=False)

    def load(self):
        """Load and clean the reports."""
        collection = bioc.BioCCollection()
        reports = pd.read_csv(self.reports_path, 
                                header=0,
                              names=["uid","Report"])
        reports = reports.fillna("")
        uid = reports["uid"].tolist()
        report = reports["Report"].tolist()

        for i, id_report in enumerate(zip(uid, report)):
            uid, report= id_report
            clean_report = self.clean(report)

            new_document = self.combine_document(str(uid), clean_report)

            split_document = self.splitter.split_doc(new_document)
            #assert len(split_document.passages) == 1,\
            #    ('Each document must have a single passage, ' +
            #     'the Impression section.')

            collection.add_document(split_document)

        self.reports = reports
        self.collection = collection

    def clean(self, report):
        """Clean the report text."""
        lower_report = report.lower()
        # Change `and/or` to `or`.
        corrected_report = re.sub('and/or',
                                  'or',
                                  lower_report)
        # Change any `XXX/YYY` to `XXX or YYY`.
        corrected_report = re.sub('(?<=[a-zA-Z])/(?=[a-zA-Z])',
                                  ' or ',
                                  corrected_report)
        # Clean double periods
        clean_report = corrected_report.replace("..", ".")
        # Insert space after commas and periods.
        clean_report = clean_report.translate(self.punctuation_spacer)
        # Convert any multi white spaces to single white spaces.
        clean_report = ' '.join(clean_report.split())
        # Remove empty sentences
        clean_report = re.sub(r'\.\s+\.', '.', clean_report)

        return clean_report


    def combine_document(self, id, report):
        document = bioc.BioCDocument()
        document.id = id
        
        report = printable(report).replace('\r\n', '\n')
        passage = bioc.BioCPassage()
        passage.offset = 0
        passage.text = report
        passage.infons['title'] = "Report"
        document.add_passage(passage)

        return document