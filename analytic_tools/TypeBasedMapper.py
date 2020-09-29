import os
import sys
import json

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from configuration.ConfigChecker import ConfigChecker
from configuration.Configuration import Configuration

groups = [

    [
        ['a_15_1_x', 'a_15_1_y', 'a_15_1_z'],
        ['a_16_3_x', 'a_16_3_y', 'a_16_3_z'],

    ],

    [
        'txt18_vsg_x',
        'txt18_vsg_y',
        'txt18_vsg_z',
    ],

    [
        'hPa_15',
        'hPa_17',
        'hPa_18',
    ],

    # TODO nach Schalter und Lichtschranke unterscheiden
    [
        ['txt15_i1', 'txt15_i2', 'txt15_i3', 'txt15_i6', 'txt15_i7', 'txt15_i8', ],
        ['txt16_i1', 'txt16_i2', 'txt16_i3', 'txt16_i4', 'txt16_i5', ],
        ['txt17_i1', 'txt17_i2', 'txt17_i3', 'txt17_i5', ],
        ['txt18_i1', 'txt18_i2', 'txt18_i3', ],
        ['txt19_i1', 'txt19_i4', 'txt19_i5', 'txt19_i6', 'txt19_i7', 'txt19_i8', ]
    ],

    # TODO Nach typen unterscheiden, also Band und andere
    [
        ['txt15_m1.finished', ],
        ['txt16_m1.finished', 'txt16_m2.finished', 'txt16_m3.finished', ],
        ['txt17_m1.finished', 'txt17_m2.finished', ],
        ['txt18_m1.finished', 'txt18_m2.finished', 'txt18_m3.finished', ],
        ['txt19_m1.finished', 'txt19_m2.finished', 'txt19_m3.finished', 'txt19_m4.finished', ]
    ],

    # TODO Andere Unterscheidung nötig?
    [
        ['txt15_o5', 'txt15_o6', 'txt15_o7', 'txt15_o8', ],
        ['txt16_o7', 'txt16_o8', ],
        ['txt17_o5', 'txt17_o6', 'txt17_o7', 'txt17_o8', ],
        ['txt18_o7', 'txt18_o8', ]
    ]

]


# TODO:
#  Wie bei anderen Matching wie bei Accs herstellen?
#  Wie 2. Ebene berücksichtigen?


def main():
    config = Configuration()
    # dataset = FullDataset(config.training_data_folder, config, training=True)
    # dataset.load()

    checker = ConfigChecker(config, None, 'preprocessing', training=None)
    checker.pre_init_checks()

    features = np.load(config.training_data_folder + 'feature_names.npy')

    groups_2d = []

    # Reduce groups defined above so that the second grouping dimension is ignored
    for group in groups:
        group_1d = []

        if type(group[0]) == list:
            for elem in group:
                group_1d.extend(elem)
        else:
            group_1d = group

        groups_2d.append(group_1d)

    # Print index -> feature for comparison
    # for index, feature in enumerate(features):
    #     print(index, feature)

    # Print newly reduced groups for comparison
    # for group in groups_2d:
    #     print(*group, sep=', ')


    groups_as_indices = []

    # Map the feature names to their indices in the dataset
    for group in groups_2d:

        indices_of_group = []

        for elem in group:
            # find the index of the feature
            index_of_elem = np.nonzero(features == elem)[0][0]
            indices_of_group.append(int(index_of_elem))

        groups_as_indices.append(indices_of_group)

    # for group in groups_as_indices:
    #     print(*group, sep=', ')

    with open('../data/feature_selection/typeMapping.json', 'w', encoding='utf-8') as f:
        json.dump(groups_as_indices, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()
