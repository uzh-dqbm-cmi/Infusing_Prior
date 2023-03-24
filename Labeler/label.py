"""Entry-point script to label radiology reports."""
import pandas as pd

from args import ArgParser
from loader import Loader
from stages import Extractor, Classifier, Aggregator
from constants import *


def write(reports, label_dict, output_path, verbose=False):
    """Write labeled reports to specified path."""
    uid = reports["uid"].tolist()
    report = reports["Report"].tolist()

    labeled_reports = pd.DataFrame({"uid": uid, "Report": report})
    for index, category in enumerate(CATEGORIES):
        labeled_reports[category] = label_dict[category]

    if verbose:
        print(f"Writing reports and labels to {output_path}.")
    labeled_reports[["uid", "Report"] + CATEGORIES].to_csv(output_path,
                                                   index=False)


def label(args):
    """Label the provided report(s)."""

    loader = Loader(args.reports_path, args.extract_impression)

    extractor = Extractor(args.mention_phrases_dir, verbose=args.verbose)
    classifier = Classifier(args.key_phrase_path, verbose=args.verbose)
    aggregator = Aggregator(CATEGORIES, verbose=args.verbose)

    # Load reports in place.
    loader.load()
    # Extract observation mentions in place.
    extractor.extract(loader.collection)
    # Classify mentions in place.
    classifier.classify(loader.collection)


    # Aggregate mentions to obtain one set of labels for each report.
    label_dict = aggregator.aggregate(loader.collection)

    write(loader.reports, label_dict, args.output_path, args.verbose)


if __name__ == "__main__":
    parser = ArgParser()
    label(parser.parse_args())

