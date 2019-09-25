from case_based_similarity.CaseBasedSimilarity import CBS
from configuration.Configuration import Configuration


def main():
    config = Configuration()
    cbs = CBS(config, True)
    cbs.print_info_all_encoders()


if __name__ == '__main__':
    main()
