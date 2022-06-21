import itertools
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import owlready2 as owl
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from neural_network.Dataset import FullDataset
#from analytic_tools.TSNE_Vis import discrete_cmap
from configuration.Configuration import Configuration
from matplotlib.colors import ListedColormap
import matplotlib as mlp
import matplotlib.pyplot as plt
import networkx as nx

def discrete_cmap(N, base_cmap=None):
    # base = plt.cm.get_cmap(base_cmap)
    # color_list = base(np.linspace(0, 1, N))
    # # Set first color (will be no_failure) to grey
    # color_list[0] = np.array([0.5, 0.5, 0.5, 1])
    # cmap_name = base.name + str(N)
    # base.from_list(cmap_name, color_list, N)
    '''    color_list = [
        '#808080',
        '#FF0000',
        '#FFFF00',
        '#00FF00',
        '#008000',
        '#00FFFF',
        '#000080',
        '#FF00FF',
        '#800000',
        '#008080',
        '#0000FF',
        '#800080',
        '#DFFF00',
        '#FFBF00',
        '#FF7F50',
        '#DE3163',
        '#40E0D0',
        '#CCCCFF',
        '#8e44ad'
    ]'''
    color_list = [
        '#d5dadc', #0
        '#fcc3a7', #1
        '#8fd1bb', #2
        '#b2d4f5', #3
        '#848f94', #4
        '#f8cc8c',#74abe2
        '#f99494',
        '#5899DA',
        '#ef8d5d',
        '#13A4B4',
        '#f5b04d',
        '#3fb68e',
        '#DFFF00',
        '#FFBF00',
        '#FF7F50',
        '#DE3163',
        '#40E0D0',
        '#CCCCFF',
        '#8e44ad'
    ]

    if N < len(color_list):
        color_list = color_list[0:N]

    color_list = [mlp.colors.hex2color(c) for c in color_list]

    assert len(color_list) == N

    return ListedColormap(color_list, name='OrangeBlue')

def plot(feature_names, linked_features, responsible_relations, force_self_loops, display_labels):
    # Recreate the ADJ but with different values = colors based on the relationship of the connection
    n = feature_names.size
    a_plot = pd.DataFrame(index=feature_names, columns=feature_names, data=np.zeros(shape=(n, n)))
    '''
    color_values = {
        'no_relation': 0,
        'self_loops': 1,
        'component': 2,
        'same_iri': 3,
        'connection': 4,
        'actuation': 5,
        'calibration': 6,
        'precondition': 7,
        'postcondition': 8,
    }
    '''
    color_values = {
        'no_relation': 0, 'hosts-directly-addressable-component': 1, 'has-pipe-connection': 2, 'has-same-higher-level-component': 3, 'has-same-controller': 4,
            'has-same-property_1': 5, 'has-same-property_2': 6, 'has-same-property_3': 7, 'property-used-to-control': 8,
            'property-of-higher-level-comp-used-to-control': 9, 'not-in-same-state': 10, 'has-same-potential-failure-mode': 11}

    if force_self_loops:

        for f_j in feature_names:
            a_plot.loc[f_j, f_j] = color_values['self_loops']

    for (f_j, f_i), r in zip(linked_features, responsible_relations):
        c_val = color_values[r.split("#")[0]]

        if c_val > a_plot.loc[f_i, f_j]:
            a_plot.loc[f_i, f_j] = c_val

    size = 22 if display_labels else 15
    dpi = 200 if display_labels else 300

    font = {'family': 'serif','size': 14}
    plt.rc('font', **font)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(size, size), dpi=dpi)
    im = ax.imshow(a_plot.values, interpolation='none', cmap=discrete_cmap(len(list(color_values.keys())), 'jet'), )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.5)
    plt.colorbar(im, cax=cax)
    im.set_clim(vmin=-0.5, vmax=len(list(color_values.keys())) - 0.5)
    ax.set_title(color_values)

    ax.set_ylabel('i (target)')
    ax.set_xlabel('j (source)')

    ax.tick_params(which='minor', width=0)
    ax.set_xticks(np.arange(-.5, n, 10), minor=True)
    ax.set_yticks(np.arange(-.5, n, 10), minor=True)

    # Gridlines based on minor ticks
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.75)

    if display_labels:
        # Minor ticks with width = 0 so they are not really visible
        ax.set_xticks(np.arange(0, n, 1), minor=False)
        ax.set_yticks(np.arange(0, n, 1), minor=False)

        features = [f[0:20] if len(f) > 20 else f for f in a_plot.columns]

        ax.set_xticklabels(features, minor=False)
        ax.set_yticklabels(features, minor=False)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=75, ha="right", rotation_mode="anchor")

    fig.tight_layout()
    #fig.savefig(f"../logs/{config.a_pre_file.split('.')[0]}.pdf", dpi=dpi, bbox_inches='tight')
    fig.savefig(config.data_folder_prefix + 'training_data/knowledge/Plot_adj.pdf', dpi=dpi, bbox_inches='tight')

    plt.show()


# Split string at pos th occurrence of sep
def split(strng, sep, pos):
    strng = strng.split(sep)
    return sep.join(strng[:pos]), sep.join(strng[pos:])


def name_2_iri(feature_names: np.ndarray):
    """
    Creates a mapping from feature names to the iris in the ontology.
    :param feature_names: A numpy array with features names that matches the order of features in the dataset.
    :return: A dictionary that matches the feature names to their iri.
    """

    base_iri = 'FTOnto:'
    feature_2_iri = {}
    manual_corrections = {
        base_iri + "OV_1_Compressor_8": base_iri + "OV_1_WT_1_Compressor_8",
        base_iri + "WT_1_Compressor_8": base_iri + "OV_1_WT_1_Compressor_8",
        base_iri + "OV_2_Compressor_8": base_iri + "OV_2_WT_2_Compressor_8",
        base_iri + "WT_2_Compressor_8": base_iri + "OV_2_WT_2_Compressor_8",
        base_iri + "MM_1_Pneumatic_System_Pressure": base_iri + "MM_1_Compressor_8",
        base_iri + "OV_1_WT_1_Pneumatic_System_Pressure": base_iri + "OV_1_WT_1_Compressor_8",
        base_iri + "SM_1_Pneumatic_System_Pressure": base_iri + "SM_1_Compressor_8",
        base_iri + "VGR_1_Pneumatic_System_Pressure": base_iri + "VGR_1_Compressor_7",
    }

    for feature in feature_names:

        # Remember whether a matching iri could be found
        matched_defined_type = True
        iri_comps = []
        print("feature beginning: ", feature)
        # Derive iri from feature name
        main_component, specific_part = split(feature, '_', 2)
        iri_comps.append(main_component.upper())
        identifier, type = split(specific_part, '_', 1)

        if main_component.startswith('shop'):
            substring = split(feature, '_', 4)[1].title()
            a, b = split(substring, '_', -3)
            iri_comps = [a.upper(), b]
        elif identifier.startswith('i'):
            # Light barrier and position switches
            nbr = identifier[1]
            type = split(type, '_', 2)[0].title()
            iri_comps.append(type)
            iri_comps.append(nbr)
        elif identifier.startswith('m'):
            # Motor speeds
            nbr = identifier[1]
            iri_comps.append('Motor')
            iri_comps.append(nbr)
        elif identifier.startswith('o'):
            # Valves and compressors
            nbr = identifier[1]
            iri_comps.append(split(type, '_', 1)[0].title())
            iri_comps.append(nbr)
        elif identifier in ['current', 'target']:
            iri_comps.append('Crane_Jib')
        elif identifier == 'temper':
            # Untested because not present in dataset
            iri_comps.append('Temperature')
        elif main_component.startswith('bmx'):
            main_component = split(main_component, '_', 1)[1].upper()
            iri_comps = [main_component, identifier, 'Crane_Jib']
        elif main_component.startswith('acc'):
            # Acceleration sensors are associated with the component they observe e.g. a motor
            main_component = split(main_component, '_', 1)[1].upper()

            if type.startswith('m'):
                iri_comps = [main_component, identifier, 'Motor', type.split('_')[1]]
            elif type.startswith('comp'):
                iri_comps = [main_component, identifier, 'Compressor_8']
        elif main_component == 'sm_1' and identifier == 'detected':
            iri_comps = [main_component.upper(), 'Color', 'Sensor', '2']
        else:
            # No matching iri was found
            matched_defined_type = False

        if matched_defined_type:
            iri = base_iri + '_'.join(iri_comps)

            if iri in manual_corrections.keys():
                iri = manual_corrections.get(iri)

            feature_2_iri[feature] = iri
        else:
            # Mark the no valid iri was found for this feature
            feature_2_iri[feature] = None

        feature_2_iri[feature] = iri
        print("feature: ", feature)
    return feature_2_iri


def invert_dict(d: dict):
    """
    Creates an inverted dictionary of d: All values of a key k in d will become a key with value k in the inverted dict.
    :param d: The dictionary that should be inverted.
    :return: The resulting inverted dict.
    """

    inverted_dict = {}

    for key, value in d.items():
        if value in inverted_dict.keys():
            inverted_dict[value].append(key)
        else:
            inverted_dict[value] = [key]

    return inverted_dict


def tuple_corrections(feature_tuples, iri=None):
    # Remove self loops
    feature_tuples = [(a, b) for (a, b) in feature_tuples if a != b]

    # Add inverse, necessary because not all relations are present in the ontology
    feature_tuples.extend([(b, a) for (a, b) in feature_tuples])

    # Remove duplicates
    feature_tuples = list(set(feature_tuples))

    return feature_tuples


def store_mapping(config: Configuration, feature_2_iri: dict):
    """
    Stores the dictionary mapping of features to their iri such that it can be used by other programs,
        mainly GenerateFeatureEmbeddings.py
    :param config: The configuration object.
    :param feature_2_iri: dictionary mapping features to their iri.
    """
    print("feature_2_iri: ", feature_2_iri)
    with open(config.attribute_to_iri_mapping_file, 'w') as outfile:
        print("outfile", outfile)
        json.dump(feature_2_iri, outfile, sort_keys=True, indent=2)


def check_mapping(feature_2_iri: dict):
    """
    Ensures a iri could be determined for all features.
    :param feature_2_iri: The dictionary mapping features to their iri.
    """
    print("feature_2_iri: ", feature_2_iri)
    for feature, iri in feature_2_iri.items():
        if iri is None:
            raise ValueError(f'No IRI could be generated for feature {feature}.'
                             ' This would cause problems when trying to assign an embedding or finding relations.')

def extract_rel_name(r1):
    if "#" in r1:
        rel_str = "#" + str(r1.split("#")[-1])
    else:
        rel_str = "#" + str(r1.split("/")[-1])

    return rel_str

# config = configuration of file paths etc.
# feature_names: list with features (i.e. data streams)
def onto_2_matrix(config, feature_names, daemon=True, temp_id=None):
    ##############################

    # Settings which relations to include in the generated adjacency matrix.
    component_of_relation = True
    iri_relation = True
    connected_to_relation = True
    calibration_relation = True
    actuates_relation = True
    monitors_relation = True
    same_controller_relation = True
    isInputFor_relation = True
    sosaHosts_relation = True
    observableProperty_relation = True
    observableProperty_relation_with_Comp = True
    same_Property = True
    observableProperty_relation_used_for_control = True
    actuatesHostsProperty_relation = True
    not_in_same_state = True
    has_same_Failure_mode = False
    both_precondition_same_service = False
    both_postcondition_same_service = False

    # Should not be used. Used the corresponding gsl mod instead.
    force_self_loops = False

    # Settings regarding the adj plot.
    plot_labels = True
    print_linked_features = True

    # Generate RCA features instead of ADJMat
    useFoRCA= False
    ##############################

    # Consistency check to ensure the intended configuration is used.
    if not all([component_of_relation, iri_relation, connected_to_relation, calibration_relation, actuates_relation,
                both_postcondition_same_service, both_precondition_same_service, not force_self_loops]):
        '''
        if not daemon:
            reply = input('Configuration deviates from the set standard. Continue?\n')
            if reply.lower() != 'y':
                sys.exit(-1)
        else:
            raise ValueError('Configuration deviates from the set standard.')
        '''

    if daemon and temp_id is None:
        raise ValueError('If running this as a daemon service a temporary id must be passed.')

    # importing the module
    import json

    # Opening JSON file containing a mapping from a feature name
    # (i.e. data stream or in case of RCA a component) to its IRI of the corresponding ontology
    with open(config.attribute_to_iri_mapping_file) as json_file:
        feature_2_iri = json.load(json_file)
    with open(config.component_to_iri_mapping_file) as json_file:
        component_2_iri = json.load(json_file)

    '''
    # Create dictionary that matches feature names to the matched iri
    feature_2_iri = name_2_iri(feature_names)
    print("feature_2_iri *** ", feature_2_iri)

    '''

    check_mapping(feature_2_iri)
    #store_mapping(config, feature_2_iri)

    # Invert such that iri is matched to list of associated features
    iri_2_features = invert_dict(feature_2_iri)
    iri_2_components = invert_dict(component_2_iri)

    #print("iri_2_features: ", iri_2_features)
    #print("iri_2_components: ", iri_2_components)

    # Load ontology into owlready2 for further processing (querrying, reasoning etc.)
    onto_file = config.ftonto_file
    ontology = owl.get_ontology("file://" + onto_file).load()

    # If desired, doing the reasoning first:
    # owl.JAVA_EXE = "/usr/bin/java/"
    # owl.sync_reasoner_hermit()

    #
    # The following part extracts feature - feature relations
    # or if useRCA active, then component - feature relations
    # Which relationships or pattern are considered are defined
    # at the beginning.
    #

    # Stores pairs of features in a list that are extracted based on a relationship in the ontology.
    linked_features = []
    # e.g. [('txt15_i3', 'txt15_m1.finished'), ('txt18_i3', 'txt18_m1.finished'), ... ]
    # Stores the name as string of the relationship for each pair of linked features
    responsible_relations = []
    # e.g. ['component', 'component', ... ]

    # Link features (data streams) which are matched to the same iri (entity of the ontology)
    if iri_relation:
        matched_iri_dict = {}

        for name, iri in feature_2_iri.items():
            if iri is None:
                continue

            if iri in matched_iri_dict:
                matched_iri_dict[iri].append(name)
            else:
                matched_iri_dict[iri] = [name]

        for iri, feature_lists in matched_iri_dict.items():
            feature_tuples = list(itertools.product(feature_lists, repeat=2))
            feature_tuples = tuple_corrections(feature_tuples, iri)
            linked_features.extend(feature_tuples)

            # Assign same relation for plotting
            responsible_relations.extend(['same_iri' for _ in range(len(feature_tuples))])

        print(" ---- ---- ----")
        print("Same IRI: ")
        print("responsible_relations: ", responsible_relations)
        print("linked_features: ", linked_features)
        print(" ---- ---- ----")

    # Compositional / Part-Of Relationship
    if component_of_relation:
        #r1 = 'http://iot.uni-trier.de/FTOnto#isComponentOf' #'FTOnto:isComponentOf'
        #r2 = 'http://iot.uni-trier.de/FTOnto#hasComponent'  #'FTOnto:hasComponent'

        is_of_relations = [
            ('http://iot.uni-trier.de/FTOnto#isMountedOn',
             'http://iot.uni-trier.de/FTOnto#hasMountedOn'),
            ('http://www.w3.org/ns/sosa/isHostedBy',
             'http://www.w3.org/ns/sosa/hosts'),
            ('http://iot.uni-trier.de/FTOnto#isComponentOfPneuSys',
             'http://iot.uni-trier.de/FTOnto#PneuSysHasComponent'),
            ('http://iot.uni-trier.de/FTOnto#isComponentOf',
             'http://iot.uni-trier.de/FTOnto#hasComponent')
        ]
        cnt = 0
        for r1, r2 in is_of_relations:
            if useFoRCA:
                feature_tuples = infer_connections(component_2_iri, iri_2_features, r1,direct_relation=False, symmetric_relation=False, r2=r2, iri_2_components=iri_2_components)
            else:
                feature_tuples = infer_connections(feature_2_iri, iri_2_features, r1, direct_relation=False,  symmetric_relation=False, r2=r2)
            feature_tuples = tuple_corrections(feature_tuples)
            cnt = cnt + len(feature_tuples)
            linked_features.extend(feature_tuples)
            rel_str = extract_rel_name(r1)
            responsible_relations.extend(['has-same-higher-level-component'+rel_str for _ in range(len(feature_tuples))])

        print(" ---- ---- ----")
        print("Found ", cnt, " for has-same-higher-level-component pattern.")
        #print("hasComponent/ComponentOf: ")
        #print("feature_tuples: ", feature_tuples)
        #print("responsible_relations: ", responsible_relations)
        #print("linked_features: ", linked_features)
        print(" ---- ---- ----")
    #print(sddssd)

    # Pneumatic System Pipe connection
    if connected_to_relation:

        r = 'http://iot.uni-trier.de/FTOnto#isConnectedViaPipeTo'#'FTOnto:isConnectedTo'
        if useFoRCA:
            feature_tuples = infer_connections(component_2_iri, iri_2_features, r,
                                           direct_relation=True, symmetric_relation=True, iri_2_components=iri_2_components)
        else:
            feature_tuples = infer_connections(feature_2_iri, iri_2_features, r,
                                               direct_relation=True, symmetric_relation=True)
        feature_tuples = tuple_corrections(feature_tuples)
        linked_features.extend(feature_tuples)
        responsible_relations.extend(['has-pipe-connection' for _ in range(len(feature_tuples))])

        print(" ---- ---- ----")
        print("Found ", len(feature_tuples), " for has-pipe-connection pattern.")
        #print("responsible_relations: ", feature_tuples)
        #print("responsible_relations: ", responsible_relations)
        #print("linked_features: ", linked_features)
        print(" ---- ---- ----")

    # Resolved in SNN/SOSA Property construct
    '''
    if calibration_relation:
        r1 = 'http://iot.uni-trier.de/FTOnto#calibrates'#'FTOnto:calibrates'
        r2 = 'http://iot.uni-trier.de/FTOnto#isCalibratedBy'#'FTOnto:isCalibratedBy'

        if useFoRCA:
            feature_tuples = infer_connections(component_2_iri, iri_2_features, r1,
                                               direct_relation=True, symmetric_relation=False, r2=r2, iri_2_components=iri_2_components)
        else:
            feature_tuples = infer_connections(feature_2_iri, iri_2_features, r1,
                                           direct_relation=True, symmetric_relation=False, r2=r2)

        feature_tuples = tuple_corrections(feature_tuples)
        linked_features.extend(feature_tuples)
        responsible_relations.extend(['calibration' for _ in range(len(feature_tuples))])

    print(" ---- ---- ----")
    print("calibrates: ")
    #print("responsible_relations: ", feature_tuples)
    #print("responsible_relations: ", responsible_relations)
    #print("linked_features: ", linked_features)
    print(" ---- ---- ----")
    '''

    # Resolved in SNN/SOSA Property construct
    '''
    if actuates_relation:
        # Superclasses FTOnto:actuates and FTOnto:isActuatedBy not present

        actuation_relations = [
            ('http://iot.uni-trier.de/FTOnto#actuatesHorizontallyForwardBackward',
             'http://iot.uni-trier.de/FTOnto#isActuatedHorizontallyForwardBackwardBy'),
            ('http://iot.uni-trier.de/FTOnto#actuatesHorizontallyLeftRight',
             'http://iot.uni-trier.de/FTOnto#isActuatedHorizontallyLeftRightBy'),
            ('http://iot.uni-trier.de/FTOnto#actuatesRotationallyAroundVerticalAxis',
             'http://iot.uni-trier.de/FTOnto#isActuatedRotationallyAroundVerticalAxisBy'),
            ('http://iot.uni-trier.de/FTOnto#actuatesVertically',
             'http://iot.uni-trier.de/FTOnto#isActuatedVerticallyBy')
        ]


        for r1, r2 in actuation_relations:
            if useFoRCA:
                feature_tuples = infer_connections(component_2_iri, iri_2_features, r1,
                                               direct_relation=True, symmetric_relation=False, r2=r2, iri_2_components=iri_2_components)
            else:
                feature_tuples = infer_connections(feature_2_iri, iri_2_features, r1,
                                                   direct_relation=True, symmetric_relation=False, r2=r2)
            feature_tuples = tuple_corrections(feature_tuples)
            linked_features.extend(feature_tuples)

            # Assign same relation for plotting
            responsible_relations.extend(['actuation' for _ in range(len(feature_tuples))])


        print(" ---- ---- ----")
        print("actuates: ")
        #print("responsible_relations: ", feature_tuples)
        #print("responsible_relations: ", responsible_relations)
        #print("linked_features: ", linked_features)
        print(" ---- ---- ----")
    '''

    # Resolved in mounted on relation (part-of)
    '''
    if monitors_relation:
        r1 = 'http://iot.uni-trier.de/FTOnto#monitores'#'FTOnto:calibrates'
        r2 = 'http://iot.uni-trier.de/FTOnto#isMonitoredBy'#'FTOnto:isCalibratedBy'

        if useFoRCA:
            feature_tuples = infer_connections(component_2_iri, iri_2_features, r1,
                                           direct_relation=True, symmetric_relation=False, r2=r2, iri_2_components=iri_2_components)
        else:
            feature_tuples = infer_connections(feature_2_iri, iri_2_features, r1,
                                               direct_relation=True, symmetric_relation=False, r2=r2)
        feature_tuples = tuple_corrections(feature_tuples)
        linked_features.extend(feature_tuples)
        responsible_relations.extend(['monitores' for _ in range(len(feature_tuples))])

        print(" ---- ---- ----")
        print("monitors: ")
        print("responsible_relations: ", feature_tuples)
        #print("responsible_relations: ", responsible_relations)
        #print("linked_features: ", linked_features)
        print(" ---- ---- ----")
    '''

    if same_controller_relation:
        #r1 = 'http://iot.uni-trier.de/FTOnto#isControlledBy'#'FTOnto:calibrates'
        #r2 = 'http://iot.uni-trier.de/FTOnto#controls'#'FTOnto:isCalibratedBy'

        is_of_relations = [
            ('http://iot.uni-trier.de/FTOnto#isControlledBy',
             'http://iot.uni-trier.de/FTOnto#controls'),
            ('http://iot.uni-trier.de/FTOnto#isInputFor',
             'http://iot.uni-trier.de/FTOnto#getsInputFrom')
        ]
        cnt = 0
        for r1, r2 in is_of_relations:
            if useFoRCA:
                feature_tuples = infer_connections(component_2_iri, iri_2_features, r1,
                                               direct_relation=False, symmetric_relation=False, r2=r2, iri_2_components=iri_2_components)
            else:
                feature_tuples = infer_connections(feature_2_iri, iri_2_features, r1,
                                                   direct_relation=False, symmetric_relation=False, r2=r2)
            feature_tuples = tuple_corrections(feature_tuples)
            cnt = cnt + len(feature_tuples)
            linked_features.extend(feature_tuples)
            rel_str = extract_rel_name(r2)
            responsible_relations.extend(['has-same-controller'+rel_str for _ in range(len(feature_tuples))])

        print(" ---- ---- ----")
        print("Found ", cnt, " for has-same-controller pattern.")
        #print("responsible_relations: ", feature_tuples)
        #print("responsible_relations: ", responsible_relations)
        #print("linked_features: ", linked_features)
        print(" ---- ---- ----")

    if sosaHosts_relation:
        r1 = 'http://www.w3.org/ns/sosa/hosts'#'FTOnto:calibrates'
        r2 = 'http://www.w3.org/ns/sosa/isHostedBy'#'FTOnto:isCalibratedBy'

        if useFoRCA:
            feature_tuples = infer_connections(component_2_iri, iri_2_features, r1,
                                           direct_relation=True, symmetric_relation=False, r2=r2, iri_2_components=iri_2_components)
        else:
            feature_tuples = infer_connections(feature_2_iri, iri_2_features, r1,
                                               direct_relation=True, symmetric_relation=False, r2=r2)
        feature_tuples = tuple_corrections(feature_tuples)
        linked_features.extend(feature_tuples)
        responsible_relations.extend(['hosts-directly-addressable-component' for _ in range(len(feature_tuples))])

        print(" ---- ---- ----")
        print("Found ", len(feature_tuples), " for hosts-directly-addressable-component pattern.")
        #print("responsible_relations: ", feature_tuples)
        #print("responsible_relations: ", responsible_relations)
        #print("linked_features: ", linked_features)
        print(" ---- ---- ----")

    if observableProperty_relation:
        #r1 = 'http://www.w3.org/ns/sosa/observes'   #'FTOnto:calibrates'
        r2 = 'http://www.w3.org/ns/ssn/hasProperty' #'FTOnto:isCalibratedBy'
        r3 = 'http://www.w3.org/ns/ssn/isPropertyOf'
        #r4 = 'http://www.w3.org/ns/sosa/isObservedBy'

        is_of_relations_r1 = ['http://www.w3.org/ns/sosa/observes', 'http://www.w3.org/ns/ssn/forProperty',
                              'http://iot.uni-trier.de/FTOnto#affects', 'http://iot.uni-trier.de/FTOnto#actuates',
                              'http://iot.uni-trier.de/FTOnto#actuatesHorizontallyForwardBackward',
                              'http://iot.uni-trier.de/FTOnto#actuatesHorizontallyLeftRight',
                              'http://iot.uni-trier.de/FTOnto#actuatesRotationallyAroundVerticalAxis',
                              'http://iot.uni-trier.de/FTOnto#actuatesVertically',
                              'http://iot.uni-trier.de/FTOnto#regulates', 'http://iot.uni-trier.de/FTOnto#generates_vaccuum',
                              'http://iot.uni-trier.de/FTOnto#pushes_vertically'
        ]

        is_of_relations_r4 = ['http://www.w3.org/ns/sosa/isObservedBy', 'http://iot.uni-trier.de/FTOnto#is_affected_by',
                              'http://iot.uni-trier.de/FTOnto#isActuatedBy', 'http://iot.uni-trier.de/FTOnto#isActuatedHorizontallyForwardBackwardBy',
                              'http://iot.uni-trier.de/FTOnto#isActuatedHorizontallyLeftRightBy',
                              'http://iot.uni-trier.de/FTOnto#isActuatedRotationallyAroundVerticalAxisBy',
                              'http://iot.uni-trier.de/FTOnto#isActuatedVerticallyBy', 'http://iot.uni-trier.de/FTOnto#isregulatedBy',
                              'http://iot.uni-trier.de/FTOnto#vacuum_is_generated_by', 'http://iot.uni-trier.de/FTOnto#is_pushed_vertically'
                              ]

        ''' USED FOR Queries with a middle entity    
        
         SELECT ?y WHERE {
        { ?sensor <http://www.w3.org/ns/sosa/observes> ?Property . ?Property <http://www.w3.org/ns/ssn/isPropertyOf> ?y. } 
        UNION
         { ?y <http://www.w3.org/ns/ssn/hasProperty> ?Property . ?Property <http://www.w3.org/ns/sosa/isObservedBy> ?sensor. }
        }
        UNION
         { ?Property <http://www.w3.org/ns/ssn/isPropertyOf> ?y . ?Property <http://www.w3.org/ns/sosa/isObservedBy> ?sensor. }
        }
        UNION
         { ?y <http://www.w3.org/ns/ssn/hasProperty> ?Property . ?sensor <http://www.w3.org/ns/sosa/observes> ?Property. }
        }
        
        '''
        cnt = 0
        for r1 in is_of_relations_r1:
            for r4 in is_of_relations_r4:
                if useFoRCA:
                    feature_tuples = infer_connections(component_2_iri, iri_2_features, r1,
                                                   direct_relation=False, symmetric_relation=False, r2=r2, r3=r3,r4=r4, iri_2_components=iri_2_components)
                else:
                    feature_tuples = infer_connections(feature_2_iri, iri_2_features, r1,
                                                       direct_relation=False, symmetric_relation=False, r2=r2, r3=r3, r4=r4,is_chain=True)
                feature_tuples = tuple_corrections(feature_tuples)
                cnt = cnt + len(feature_tuples)
                #print("len(feature_tuples): ", len(feature_tuples))
                #print("cnt: ", cnt)
                linked_features.extend(feature_tuples)
                rel_r1 = extract_rel_name(r1)
                rel_r4 = extract_rel_name(r4)
                responsible_relations.extend(['has-same-property_1'+rel_r1+rel_r4 for _ in range(len(feature_tuples))])
        #print(sddssd)
        print(" ---- ---- ----")
        print("Found ", cnt, " for has-same-property_1 pattern.")
        #print("responsible_relations: ", feature_tuples)
        #print("responsible_relations: ", responsible_relations)
        #print("linked_features: ", linked_features)
        print(" ---- ---- ----")

    if observableProperty_relation_with_Comp:
        r1 = 'http://www.w3.org/ns/sosa/observes'
        r3 = 'http://www.w3.org/ns/ssn/isPropertyOf'
        #r5 = 'http://iot.uni-trier.de/FTOnto#isMountedOn'
        #r5 = 'http://www.w3.org/ns/sosa/isHostedBy'
        #r5 = 'http://iot.uni-trier.de/FTOnto#hasComponent'
        r2 = 'http://www.w3.org/ns/sosa/isObservedBy'
        r4 = 'http://www.w3.org/ns/ssn/hasProperty'
        #r6 = 'http://iot.uni-trier.de/FTOnto#hasMountedOn'
        #r6 = 'http://www.w3.org/ns/sosa/hosts'
        #r6 = 'http://iot.uni-trier.de/FTOnto#isComponentOf'

        ''' USED FOR Quieries with a middle entity 
        a observes x . x isPropertyOf z. y is MountedOn z. 
            SELECT ?y ?z WHERE {{ <http://iot.uni-trier.de/FTOnto#SM_Light_Barrier_3> <http://www.w3.org/ns/sosa/observes> ?x . 
            ?x <http://www.w3.org/ns/ssn/isPropertyOf> ?z. ?y <http://iot.uni-trier.de/FTOnto#isMountedOn> ?z. } 
            UNION {?z <http://iot.uni-trier.de/FTOnto#hasMountedOn> ?y . ?z <http://www.w3.org/ns/ssn/hasProperty> ?x .  
            ?x <http://www.w3.org/ns/sosa/isObservedBy> <http://iot.uni-trier.de/FTOnto#SM_Light_Barrier_3> .}}

        '''

        is_of_relations = [
            ('http://iot.uni-trier.de/FTOnto#isMountedOn',
             'http://iot.uni-trier.de/FTOnto#hasMountedOn'),
            ('http://www.w3.org/ns/sosa/isHostedBy',
             'http://www.w3.org/ns/sosa/hosts'),
            ('http://iot.uni-trier.de/FTOnto#isComponentOfPneuSys',
             'http://iot.uni-trier.de/FTOnto#PneuSysHasComponent'),
            ('http://iot.uni-trier.de/FTOnto#isComponentOf',
             'http://iot.uni-trier.de/FTOnto#hasComponent')
        ]
        is_of_relations_r1_r2 = [
            ('http://www.w3.org/ns/sosa/isObservedBy',
             'http://www.w3.org/ns/sosa/observes'),
            ('http://iot.uni-trier.de/FTOnto#is_affected_by',
             'http://iot.uni-trier.de/FTOnto#affects'),
            ('http://iot.uni-trier.de/FTOnto#isActuatedBy',
             'http://iot.uni-trier.de/FTOnto#actuates'),
            ('http://iot.uni-trier.de/FTOnto#isActuatedHorizontallyForwardBackwardBy',
             'http://iot.uni-trier.de/FTOnto#actuatesHorizontallyForwardBackward'),
            ('http://iot.uni-trier.de/FTOnto#isActuatedHorizontallyLeftRightBy',
             'http://iot.uni-trier.de/FTOnto#actuatesHorizontallyLeftRight'),
            ('http://iot.uni-trier.de/FTOnto#isActuatedRotationallyAroundVerticalAxisBy',
             'http://iot.uni-trier.de/FTOnto#actuatesRotationallyAroundVerticalAxis'),
            ('http://iot.uni-trier.de/FTOnto#isActuatedVerticallyBy',
             'http://iot.uni-trier.de/FTOnto#actuatesVertically'),
            ('http://iot.uni-trier.de/FTOnto#isregulatedBy',
             'http://iot.uni-trier.de/FTOnto#regulates'),
            ('http://iot.uni-trier.de/FTOnto#vacuum_is_generated_by',
             'http://iot.uni-trier.de/FTOnto#generates_vaccuum'),
            ('http://iot.uni-trier.de/FTOnto#is_pushed_vertically',
             'http://iot.uni-trier.de/FTOnto#pushes_vertically')
        ]


        # e.g. SELECT ?y ?z ?r WHERE {
        # { <http://iot.uni-trier.de/FTOnto#SM_Light_Barrier_3> <http://www.w3.org/ns/sosa/observes> ?x .
        # ?x <http://www.w3.org/ns/ssn/isPropertyOf> ?z. ?y ?r ?z. }
        # UNION {?z <http://iot.uni-trier.de/FTOnto#hasMountedOn> ?y .
        # ?z <http://www.w3.org/ns/ssn/hasProperty> ?x .
        # ?x <http://www.w3.org/ns/sosa/isObservedBy> <http://iot.uni-trier.de/FTOnto#SM_Light_Barrier_3> .}}

        cnt = 0
        for r2, r1 in is_of_relations_r1_r2:
            for r5, r6 in is_of_relations:
                if useFoRCA:
                    feature_tuples = infer_connections(component_2_iri, iri_2_features, r1,
                                                       direct_relation=False, symmetric_relation=False, r2=r2, r3=r3, r4=r4, r5=r5, r6=r6,
                                                       iri_2_components=iri_2_components)
                else:
                    feature_tuples = infer_connections(feature_2_iri, iri_2_features, r1,
                                                       direct_relation=False, symmetric_relation=False, r2=r2, r3=r3, r4=r4, r5=r5, r6=r6,
                                                       is_chain=False)
                    feature_tuples = tuple_corrections(feature_tuples)
                    cnt = cnt + len(feature_tuples)
                    linked_features.extend(feature_tuples)
                    rel_r6 = extract_rel_name(r6)
                    rel_r2 = extract_rel_name(r2)
                    responsible_relations.extend(['has-same-property_2'+rel_r6+rel_r2 for _ in range(len(feature_tuples))])
        print(" ---- ---- ----")
        print("Found ", cnt, " for has-same-property_2 pattern.")
        # print("responsible_relations: ", feature_tuples)
        # print("responsible_relations: ", responsible_relations)
        # print("linked_features: ", linked_features)
        print(" ---- ---- ----")

    # Two components act on / observe the same Property
    if same_Property:
        #r1 = 'http://iot.uni-trier.de/FTOnto#isComponentOf' #'FTOnto:isComponentOf'
        #r2 = 'http://iot.uni-trier.de/FTOnto#hasComponent'  #'FTOnto:hasComponent'

        is_of_relations_r1 = ['http://www.w3.org/ns/sosa/observes', 'http://www.w3.org/ns/ssn/forProperty',
                              'http://iot.uni-trier.de/FTOnto#affects', 'http://iot.uni-trier.de/FTOnto#actuates',
                              'http://iot.uni-trier.de/FTOnto#actuatesHorizontallyForwardBackward',
                              'http://iot.uni-trier.de/FTOnto#actuatesHorizontallyLeftRight',
                              'http://iot.uni-trier.de/FTOnto#actuatesRotationallyAroundVerticalAxis',
                              'http://iot.uni-trier.de/FTOnto#actuatesVertically',
                              'http://iot.uni-trier.de/FTOnto#regulates', 'http://iot.uni-trier.de/FTOnto#generates_vaccuum',
                              'http://iot.uni-trier.de/FTOnto#pushes_vertically'
        ]

        is_of_relations_r2 = ['http://www.w3.org/ns/sosa/isObservedBy', 'http://iot.uni-trier.de/FTOnto#is_affected_by',
                              'http://iot.uni-trier.de/FTOnto#isActuatedBy', 'http://iot.uni-trier.de/FTOnto#isActuatedHorizontallyForwardBackwardBy',
                              'http://iot.uni-trier.de/FTOnto#isActuatedHorizontallyLeftRightBy',
                              'http://iot.uni-trier.de/FTOnto#isActuatedRotationallyAroundVerticalAxisBy',
                              'http://iot.uni-trier.de/FTOnto#isActuatedVerticallyBy', 'http://iot.uni-trier.de/FTOnto#isregulatedBy',
                              'http://iot.uni-trier.de/FTOnto#vacuum_is_generated_by', 'http://iot.uni-trier.de/FTOnto#is_pushed_vertically'
                              ]

        #  SELECT ?x WHERE {
        #  { <http://iot.uni-trier.de/FTOnto#SM_Valve_5> <http://iot.uni-trier.de/FTOnto#regulates> ?y . ?x <http://iot.uni-trier.de/FTOnto#regulates> ?y . }
        #  UNION
        #  { ?y <http://iot.uni-trier.de/FTOnto#isActuatedVerticallyBy> <http://iot.uni-trier.de/FTOnto#SM_Valve_5> .  ?y <http://iot.uni-trier.de/FTOnto#isActuatedVerticallyBy> ?x .}
        #  }
        cnt=0
        for r1 in is_of_relations_r1:
            for r2 in is_of_relations_r2:
                if useFoRCA:
                    feature_tuples = infer_connections(component_2_iri, iri_2_features, r1,direct_relation=False, symmetric_relation=False, r2=r2, iri_2_components=iri_2_components)
                else:
                    feature_tuples = infer_connections(feature_2_iri, iri_2_features, r1, direct_relation=False,  symmetric_relation=False, r3=r2,r2=r1,r4=r2, is_chain=True)
                feature_tuples = tuple_corrections(feature_tuples)
                cnt = cnt + len(feature_tuples)
                linked_features.extend(feature_tuples)
                rel_str_1 = extract_rel_name(r1)
                rel_str_2 = extract_rel_name(r2)
                responsible_relations.extend(['has-same-property_3'+rel_str_1+rel_str_2 for _ in range(len(feature_tuples))])

        print(" ---- ---- ----")
        print("Found ", cnt, " for has-same-property_3 pattern.")
        #print("hasComponent/ComponentOf: ")
        #print("feature_tuples: ", feature_tuples)
        #print("responsible_relations: ", responsible_relations)
        #print("linked_features: ", linked_features)
        # print(sddsds)
        print(" ---- ---- ----")

    if same_Property:
        #r1 = 'http://iot.uni-trier.de/FTOnto#isComponentOf' #'FTOnto:isComponentOf'
        #r2 = 'http://iot.uni-trier.de/FTOnto#hasComponent'  #'FTOnto:hasComponent'

        is_of_relations_r1 = ['http://www.w3.org/ns/sosa/observes', 'http://www.w3.org/ns/ssn/forProperty',
                              'http://iot.uni-trier.de/FTOnto#affects', 'http://iot.uni-trier.de/FTOnto#actuates',
                              'http://iot.uni-trier.de/FTOnto#actuatesHorizontallyForwardBackward',
                              'http://iot.uni-trier.de/FTOnto#actuatesHorizontallyLeftRight',
                              'http://iot.uni-trier.de/FTOnto#actuatesRotationallyAroundVerticalAxis',
                              'http://iot.uni-trier.de/FTOnto#actuatesVertically',
                              'http://iot.uni-trier.de/FTOnto#regulates', 'http://iot.uni-trier.de/FTOnto#generates_vaccuum',
                              'http://iot.uni-trier.de/FTOnto#pushes_vertically'
                            ]
        is_of_relations_r1_2 = [
          'http://www.w3.org/ns/sosa/isObservedBy', 'http://iot.uni-trier.de/FTOnto#is_affected_by',
          'http://iot.uni-trier.de/FTOnto#isActuatedBy', 'http://iot.uni-trier.de/FTOnto#isActuatedHorizontallyForwardBackwardBy',
          'http://iot.uni-trier.de/FTOnto#isActuatedHorizontallyLeftRightBy',
          'http://iot.uni-trier.de/FTOnto#isActuatedRotationallyAroundVerticalAxisBy',
          'http://iot.uni-trier.de/FTOnto#isActuatedVerticallyBy', 'http://iot.uni-trier.de/FTOnto#isregulatedBy',
          'http://iot.uni-trier.de/FTOnto#vacuum_is_generated_by', 'http://iot.uni-trier.de/FTOnto#is_pushed_vertically'
          ]

        #  SELECT ?x WHERE {
        #  { <http://iot.uni-trier.de/FTOnto#SM_Valve_5> <http://iot.uni-trier.de/FTOnto#regulates> ?y . ?x <http://iot.uni-trier.de/FTOnto#regulates> ?y . }
        #  UNION
        #  { ?y <http://iot.uni-trier.de/FTOnto#isActuatedVerticallyBy> <http://iot.uni-trier.de/FTOnto#SM_Valve_5> .  ?y <http://iot.uni-trier.de/FTOnto#isActuatedVerticallyBy> ?x .}
        #  }
        cnt = 0
        for rels in [is_of_relations_r1, is_of_relations_r1_2]:
            for r1 in rels:
                for r2 in rels:
                    if useFoRCA:
                        feature_tuples = infer_connections(component_2_iri, iri_2_features, r1,direct_relation=False, symmetric_relation=False, r2=r2, iri_2_components=iri_2_components)
                    else:
                        feature_tuples = infer_connections(feature_2_iri, iri_2_features, r1, direct_relation=False,  symmetric_relation=False, r2=r2, is_chain=True)
                    feature_tuples = tuple_corrections(feature_tuples)
                    cnt = cnt + len(feature_tuples)
                    linked_features.extend(feature_tuples)
                    rel_str_1 = extract_rel_name(r1)
                    rel_str_2 = extract_rel_name(r2)
                    responsible_relations.extend(['has-same-property_3'+rel_str_1+rel_str_2 for _ in range(len(feature_tuples))])

        print(" ---- ---- ----")
        print("Found ", cnt, " for has-same-property_3 pattern.")
        #print("hasComponent/ComponentOf: ")
        #print("feature_tuples: ", feature_tuples)
        #print("responsible_relations: ", responsible_relations)
        #print("linked_features: ", linked_features)
        #print(sddsds)
        print(" ---- ---- ----")

    if observableProperty_relation_used_for_control:
        r1 = 'http://www.w3.org/ns/sosa/observes'  # 'FTOnto:calibrates'
        r2 = 'http://iot.uni-trier.de/FTOnto#dependsOn'  # 'FTOnto:isCalibratedBy'
        r3 = 'http://iot.uni-trier.de/FTOnto#used_to_control'
        r4 = 'http://www.w3.org/ns/sosa/isObservedBy'

        ''' USED FOR Quieries with a middle entity    
        SELECT ?r ?x ?r1 ?y WHERE {
        { <http://iot.uni-trier.de/FTOnto#AccSensor_ADXL345_1> <http://www.w3.org/ns/sosa/observes> ?x . ?x <http://www.w3.org/ns/ssn/isPropertyOf> ?y. } 
        UNION
         { ?y <http://www.w3.org/ns/ssn/hasProperty> ?x . ?x <http://www.w3.org/ns/sosa/isObservedBy> <http://iot.uni-trier.de/FTOnto#AccSensor_ADXL345_1> . }
        }

         SELECT ?r ?x ?r1 ?y ?z WHERE {
        { ?z <http://www.w3.org/ns/sosa/observes> ?x . ?x <http://iot.uni-trier.de/FTOnto#used_to_control> ?y. } 
        UNION
         { ?y <http://www.w3.org/ns/ssn/hasProperty> ?x . ?x <http://www.w3.org/ns/sosa/isObservedBy> ?z. }
        }

        '''

        is_of_relations_r1_r2 = [
            ('http://www.w3.org/ns/sosa/isObservedBy',
             'http://www.w3.org/ns/sosa/observes'),
            ('http://iot.uni-trier.de/FTOnto#is_affected_by',
             'http://iot.uni-trier.de/FTOnto#affects'),
            ('http://iot.uni-trier.de/FTOnto#isActuatedBy',
             'http://iot.uni-trier.de/FTOnto#actuates'),
            ('http://iot.uni-trier.de/FTOnto#isActuatedHorizontallyForwardBackwardBy',
             'http://iot.uni-trier.de/FTOnto#actuatesHorizontallyForwardBackward'),
            ('http://iot.uni-trier.de/FTOnto#isActuatedHorizontallyLeftRightBy',
             'http://iot.uni-trier.de/FTOnto#actuatesHorizontallyLeftRight'),
            ('http://iot.uni-trier.de/FTOnto#isActuatedRotationallyAroundVerticalAxisBy',
             'http://iot.uni-trier.de/FTOnto#actuatesRotationallyAroundVerticalAxis'),
            ('http://iot.uni-trier.de/FTOnto#isActuatedVerticallyBy',
             'http://iot.uni-trier.de/FTOnto#actuatesVertically'),
            ('http://iot.uni-trier.de/FTOnto#isregulatedBy',
             'http://iot.uni-trier.de/FTOnto#regulates'),
            ('http://iot.uni-trier.de/FTOnto#vacuum_is_generated_by',
             'http://iot.uni-trier.de/FTOnto#generates_vaccuum'),
            ('http://iot.uni-trier.de/FTOnto#is_pushed_vertically',
             'http://iot.uni-trier.de/FTOnto#pushes_vertically')
        ]
        cnt = 0
        for r4, r1 in is_of_relations_r1_r2:
            if useFoRCA:
                feature_tuples = infer_connections(component_2_iri, iri_2_features, r1,
                                                   direct_relation=False, symmetric_relation=False, r2=r2, r3=r3, r4=r4,
                                                   iri_2_components=iri_2_components)
            else:
                feature_tuples = infer_connections(feature_2_iri, iri_2_features, r1,
                                                   direct_relation=False, symmetric_relation=False, r2=r2, r3=r3, r4=r4,
                                                   is_chain=True)
            feature_tuples = tuple_corrections(feature_tuples)
            cnt = cnt + len(feature_tuples)
            linked_features.extend(feature_tuples)
            responsible_relations.extend(['property-used-to-control' for _ in range(len(feature_tuples))])
        # print(sddssd)
        print(" ---- ---- ----")
        print("Found ", cnt, " for property-used-to-control pattern.")
        # print("responsible_relations: ", feature_tuples)
        # print("responsible_relations: ", responsible_relations)
        # print("linked_features: ", linked_features)
        print(" ---- ---- ----")

    if observableProperty_relation_used_for_control:
        r1 = 'http://www.w3.org/ns/sosa/observes'  # 'FTOnto:calibrates'
        r2 = 'http://iot.uni-trier.de/FTOnto#dependsOn'  # 'FTOnto:isCalibratedBy'
        r3 = 'http://iot.uni-trier.de/FTOnto#used_to_control'
        r4 = 'http://www.w3.org/ns/sosa/isObservedBy'
        r5 = "http://iot.uni-trier.de/FTOnto#hasComponent"
        r6 = 'http://iot.uni-trier.de/FTOnto#isComponentOf'

        # SELECT ?x ?z ?c WHERE { ?x <http://www.w3.org/ns/sosa/observes> ?y. ?y <http://iot.uni-trier.de/FTOnto#used_to_control> ?z. ?z <http://iot.uni-trier.de/FTOnto#hasComponent> ?c }
        #q = "SELECT ?y WHERE {{ <" + a + "> <" + r1 + "> ?x . ?x <" + r3 + "> ?z . ?z <" + r5 + "> ?y . } "

        ''' USED FOR Quieries with a middle entity    
        SELECT ?r ?x ?r1 ?y WHERE {
        { <http://iot.uni-trier.de/FTOnto#AccSensor_ADXL345_1> <http://www.w3.org/ns/sosa/observes> ?x . ?x <http://www.w3.org/ns/ssn/isPropertyOf> ?y. } 
        UNION
         { ?y <http://www.w3.org/ns/ssn/hasProperty> ?x . ?x <http://www.w3.org/ns/sosa/isObservedBy> <http://iot.uni-trier.de/FTOnto#AccSensor_ADXL345_1> . }
        }

         SELECT ?r ?x ?r1 ?y ?z WHERE {
        { ?z <http://www.w3.org/ns/sosa/observes> ?x . ?x <http://iot.uni-trier.de/FTOnto#used_to_control> ?y. } 
        UNION
         { ?y <http://www.w3.org/ns/ssn/hasProperty> ?x . ?x <http://www.w3.org/ns/sosa/isObservedBy> ?z. }
        }

        '''

        is_of_relations_r1_r2 = [
            ('http://www.w3.org/ns/sosa/isObservedBy',
             'http://www.w3.org/ns/sosa/observes'),
            ('http://iot.uni-trier.de/FTOnto#is_affected_by',
             'http://iot.uni-trier.de/FTOnto#affects'),
            ('http://iot.uni-trier.de/FTOnto#isActuatedBy',
             'http://iot.uni-trier.de/FTOnto#actuates'),
            ('http://iot.uni-trier.de/FTOnto#isActuatedHorizontallyForwardBackwardBy',
             'http://iot.uni-trier.de/FTOnto#actuatesHorizontallyForwardBackward'),
            ('http://iot.uni-trier.de/FTOnto#isActuatedHorizontallyLeftRightBy',
             'http://iot.uni-trier.de/FTOnto#actuatesHorizontallyLeftRight'),
            ('http://iot.uni-trier.de/FTOnto#isActuatedRotationallyAroundVerticalAxisBy',
             'http://iot.uni-trier.de/FTOnto#actuatesRotationallyAroundVerticalAxis'),
            ('http://iot.uni-trier.de/FTOnto#isActuatedVerticallyBy',
             'http://iot.uni-trier.de/FTOnto#actuatesVertically'),
            ('http://iot.uni-trier.de/FTOnto#isregulatedBy',
             'http://iot.uni-trier.de/FTOnto#regulates'),
            ('http://iot.uni-trier.de/FTOnto#vacuum_is_generated_by',
             'http://iot.uni-trier.de/FTOnto#generates_vaccuum'),
            ('http://iot.uni-trier.de/FTOnto#is_pushed_vertically',
             'http://iot.uni-trier.de/FTOnto#pushes_vertically')
        ]
        cnt = 0
        for r4, r1 in is_of_relations_r1_r2:
            if useFoRCA:
                feature_tuples = infer_connections(component_2_iri, iri_2_features, r1,
                                                   direct_relation=False, symmetric_relation=False, r2=r2, r3=r3, r4=r4, r5=r5, r6=r6,
                                                   is_chain=True,
                                                   iri_2_components=iri_2_components)
            else:
                feature_tuples = infer_connections(feature_2_iri, iri_2_features, r1,
                                                   direct_relation=False, symmetric_relation=False, r2=r2, r3=r3, r4=r4, r5=r5, r6=r6,
                                                   is_chain=True)
            feature_tuples = tuple_corrections(feature_tuples)
            cnt = cnt + len(feature_tuples)
            linked_features.extend(feature_tuples)
            responsible_relations.extend(['property-of-higher-level-comp-used-to-control' for _ in range(len(feature_tuples))])
        # print(sddssd)
        print(" ---- ---- ----")
        print("Found ", cnt, " for property-of-higher-level-comp-used-to-control.")
        # print("responsible_relations: ", feature_tuples)
        # print("responsible_relations: ", responsible_relations)
        # print("linked_features: ", linked_features)
        print(" ---- ---- ----")

    '''
    if actuatesHostsProperty_relation:
        actuation_relations = [
            ('http://www.w3.org/ns/sosa/isHostedBy',
                'http://iot.uni-trier.de/FTOnto#actuatesHorizontallyForwardBackward',
             'http://iot.uni-trier.de/FTOnto#isActuatedHorizontallyForwardBackwardBy',
             'http://www.w3.org/ns/sosa/hosts'),
            ('http://www.w3.org/ns/sosa/isHostedBy',
                'http://iot.uni-trier.de/FTOnto#actuatesHorizontallyLeftRight',
             'http://iot.uni-trier.de/FTOnto#isActuatedHorizontallyLeftRightBy',
             'http://www.w3.org/ns/sosa/hosts'),
            ('http://www.w3.org/ns/sosa/isHostedBy',
                'http://iot.uni-trier.de/FTOnto#actuatesRotationallyAroundVerticalAxis',
             'http://iot.uni-trier.de/FTOnto#isActuatedRotationallyAroundVerticalAxisBy',
             'http://www.w3.org/ns/sosa/hosts'),
            ('http://www.w3.org/ns/sosa/isHostedBy',
                'http://iot.uni-trier.de/FTOnto#actuatesVertically',
             'http://iot.uni-trier.de/FTOnto#isActuatedVerticallyBy',
             'http://www.w3.org/ns/sosa/hosts'),
            ('http://www.w3.org/ns/sosa/isHostedBy',
             'http://iot.uni-trier.de/FTOnto#actuates',
             'http://iot.uni-trier.de/FTOnto#isActuatedBy',
             'http://www.w3.org/ns/sosa/hosts'),
            ('http://www.w3.org/ns/sosa/isHostedBy',
             'http://iot.uni-trier.de/FTOnto#regulates',
             'http://iot.uni-trier.de/FTOnto#isregulatedBy',
             'http://www.w3.org/ns/sosa/hosts'),
            ('http://www.w3.org/ns/sosa/isHostedBy',
             'http://iot.uni-trier.de/FTOnto#affects',
             'http://iot.uni-trier.de/FTOnto#is_affected_by',
             'http://www.w3.org/ns/sosa/hosts'),
            ('http://www.w3.org/ns/sosa/isHostedBy',
             'http://iot.uni-trier.de/FTOnto#generates_vaccuum',
             'http://iot.uni-trier.de/FTOnto#vacuum_is_generated_by',
             'http://www.w3.org/ns/sosa/hosts'),
            ('http://www.w3.org/ns/sosa/isHostedBy',
             'http://iot.uni-trier.de/FTOnto#generates_vaccuum',
             'http://iot.uni-trier.de/FTOnto#vacuum_is_generated_by',
             'http://www.w3.org/ns/sosa/hosts'),
            ('http://www.w3.org/ns/sosa/isHostedBy',
             'http://iot.uni-trier.de/FTOnto#pushes_vertically',
             'http://iot.uni-trier.de/FTOnto#is_pushed_vertically',
             'http://www.w3.org/ns/sosa/hosts')
        ]

        #r1 = 'http://www.w3.org/ns/sosa/isHostedBy'   #'FTOnto:calibrates'
        #r2 = 'http://iot.uni-trier.de/FTOnto#actuates' #'FTOnto:isCalibratedBy'
        #r3 = 'http://iot.uni-trier.de/FTOnto#isActuated'
        #r4 = 'http://www.w3.org/ns/sosa/hosts'
        



        for r1, r2, r3, r4 in actuation_relations:
            if useFoRCA:
                feature_tuples = infer_connections(component_2_iri, iri_2_features, r1,
                                                   direct_relation=False, symmetric_relation=False, r2=r3, r3=r2, r4=r4,
                                                   iri_2_components=iri_2_components)
                print("......")
            else:
                feature_tuples = infer_connections(feature_2_iri, iri_2_features, r1,
                                                   direct_relation=False, symmetric_relation=False, r2=r3, r3=r2, r4=r4)
            feature_tuples = tuple_corrections(feature_tuples)
            linked_features.extend(feature_tuples)
            responsible_relations.extend(['actuate-hosts' for _ in range(len(feature_tuples))])

            print(" ---- ---- ----")
            print("actuate-hosts: ")
            # print("responsible_relations: ", feature_tuples)
            # print("responsible_relations: ", responsible_relations)
            # print("linked_features: ", linked_features)
            print(" ---- ---- ----")
    '''

    if not_in_same_state:
        # http://iot.uni-trier.de/PredM#not_in_same_state_at_the_same_time
        r = 'http://iot.uni-trier.de/PredM#not_in_same_state_at_the_same_time'  # 'FTOnto:isConnectedTo'
        if useFoRCA:
            feature_tuples = infer_connections(component_2_iri, iri_2_features, r,
                                               direct_relation=False, symmetric_relation=False,
                                               iri_2_components=iri_2_components)
        else:
            feature_tuples = infer_connections(feature_2_iri, iri_2_features, r,
                                               direct_relation=False, symmetric_relation=False)
        feature_tuples = tuple_corrections(feature_tuples)
        linked_features.extend(feature_tuples)
        responsible_relations.extend(['not-in-same-state' for _ in range(len(feature_tuples))])

        print(" ---- ---- ----")
        print("Found ", len(feature_tuples), " for not-in-same-state pattern.")
        # print("responsible_relations: ", feature_tuples)
        # print("responsible_relations: ", responsible_relations)
        # print("linked_features: ", linked_features)
        print(" ---- ---- ----")

    if has_same_Failure_mode:
        r2 = 'http://iot.uni-trier.de/PredM#isDetectableInDataStreamOf_Context' #'FTOnto:isComponentOf'
        r1 = 'http://iot.uni-trier.de/PredM#isRelevantForFM_Context'  #'FTOnto:hasComponent'

        if useFoRCA:
            feature_tuples = infer_connections(component_2_iri, iri_2_features, r1, direct_relation=False,
                                               symmetric_relation=False, r2=r2, iri_2_components=iri_2_components)
        else:
            feature_tuples = infer_connections(feature_2_iri, iri_2_features, r1, direct_relation=False,
                                               symmetric_relation=False, r2=r2)
            feature_tuples = tuple_corrections(feature_tuples)
            linked_features.extend(feature_tuples)
            rel_str = extract_rel_name(r1)
            responsible_relations.extend(
                ['has-same-potential-failure-mode' + rel_str for _ in range(len(feature_tuples))])

        print(" ---- ---- ----")
        print("Found ", len(feature_tuples), " for has-same-potential-failure-mode pattern.")
        # print("hasComponent/ComponentOf: ")
        # print("feature_tuples: ", feature_tuples)
        # print("responsible_relations: ", responsible_relations)
        # print("linked_features: ", linked_features)
        print(" ---- ---- ----")
    # print(sddssd)

    # Load the service pre- and postcondtions from a file.
    if both_precondition_same_service or both_postcondition_same_service:
        with open(config.get_additional_data_path('service_condition_pairs.json'), 'r') as f:
            service_condition_pairs = json.load(f)
            precondition_pairs = service_condition_pairs['precondition_pairs']
            postcondition_pairs = service_condition_pairs['postcondition_pairs']

    if both_precondition_same_service:
        iri_tuples = []

        for key_iri, values in precondition_pairs.items():
            iri_tuples.extend([(key_iri, value_iri) for value_iri in values])

        feature_tuples = feature_tuples_from_iri_tuples(iri_tuples, iri_2_features)
        linked_features.extend(feature_tuples)

        # Assign same relation for plotting
        responsible_relations.extend(['precondition' for _ in range(len(feature_tuples))])

    if both_postcondition_same_service:
        iri_tuples = []

        for key_iri, values in postcondition_pairs.items():
            iri_tuples.extend([(key_iri, value_iri) for value_iri in values])

        feature_tuples = feature_tuples_from_iri_tuples(iri_tuples, iri_2_features)
        linked_features.extend(feature_tuples)

        # Assign same relation for plotting
        responsible_relations.extend(['postcondition' for _ in range(len(feature_tuples))])

    if not daemon and print_linked_features:

        # Iterate of extracted feature pairs and build a data frame
        rows = []
        for a, (b, c) in zip(responsible_relations, linked_features):
            if useFoRCA:
                # Merge component and data stream mappings
                feature_2_iri_extened_by_component = feature_2_iri.copy()  # start with keys and values of x
                feature_2_iri_extened_by_component.update(component_2_iri)  # modifies z with keys and values of y
                rows.append([a, b, feature_2_iri_extened_by_component.get(b), c, feature_2_iri_extened_by_component.get(c)])
            else:
                rows.append([a, b, feature_2_iri.get(b), c, feature_2_iri.get(c)])

        df = pd.DataFrame(data=rows, columns=['Relation', 'Feature 1', 'IRI Feature 1', 'Feature 2', 'IRI Feature 2'])
        print(df.to_string())

    #
    import json
    high = ["monitores", "hosts", "actuation"]
    medium = ["component"]
    low = []
    irrelevant = []
    component_relevance = {}
    for component in component_2_iri:
        component_relevance[component] = {} #[('high',[]), ('medium',[]), ('low',[]), ('irrelevant',[])]
    for component in component_2_iri:
        #print("component: ", component)
        #compoent_relevance[component] = []
        for a, (b, c) in zip(responsible_relations, linked_features):
            #print(a,"-",b,"-",c)
            if a in high and (component == b or component ==c ):
                if (component == b):
                    component_relevance[component]['high'] = c
                else:
                    component_relevance[component]['high'] = b

            elif a in medium and (component == b or component == c ):
                if (component == b):
                    component_relevance[component]['medium'] = c
                else:
                    component_relevance[component]['medium'] = b


    print("COMPONENT:",component_relevance)

    # Generate Adj Matrix
    n = feature_names.size
    a_df = pd.DataFrame(index=feature_names, columns=feature_names, data=np.zeros(shape=(n, n)))

    for f_j, f_i in linked_features:
        if f_i != f_j:
            a_df.loc[f_i, f_j] = 1

    if force_self_loops:
        for f_i in feature_names:
            a_df.loc[f_i, f_i] = 1

    a_df.index.name = 'Features'

    # Compare:

    org_adjmat = dataset.graph_adjacency_matrix_attributes
    new_adjmat = a_df.values
    #print("new_adjmat shape: ", new_adjmat.shape)
    #np.random.shuffle(new_adjmat)
    #print("new_adjmat shape: ", new_adjmat.shape)
    #save to csv:
    np.savetxt("adjmat_new.csv", new_adjmat, delimiter=",")

    print("org_adjmat.shape:",org_adjmat.shape,"| new_adjmat.shape", new_adjmat.shape)

    diff_mat = org_adjmat - new_adjmat
    print("Entries old:",np.sum(org_adjmat),"| Entries new:", np.sum(new_adjmat), "| Differences:", np.sum(np.abs(diff_mat)))
    print("diff_mat: ", diff_mat)
    print("Positions with differences:", np.argwhere(diff_mat != 0,))
    for i in np.argwhere(diff_mat != 0,):
            print("Difference found:",diff_mat[i[0],i[1]],"at features:", dataset.feature_names_all[i[0]], dataset.feature_names_all[i[1]])

    if daemon:
        config.a_pre_file = f'temp/predefined_a_{temp_id}.xlsx'

        if not os.path.exists(config.get_additional_data_path('temp/')):
            os.makedirs(config.get_additional_data_path('temp/'))

        a_df.to_excel(config.get_additional_data_path(config.a_pre_file))
    else:
        a_df.to_excel(config.data_folder_prefix + 'training_data/knowledge/'+config.adj_mat_file)
        a_analysis(a_df)
        plot(feature_names, linked_features, responsible_relations, force_self_loops, display_labels=plot_labels)
        #plot_for_thesis(feature_names, linked_features, responsible_relations)
        thesis_output(feature_2_iri, responsible_relations, linked_features)

        labels = {}
        for i in range(dataset.feature_names_all.shape[0]):
            labels[i] = str(dataset.feature_names_all[i]).replace(".finished","").replace("txt","")


        plt.clf()
        plt.cla()
        plt.close()

        # print graph
        rows, cols = np.where(new_adjmat == 1)
        edges = zip(rows.tolist(), cols.tolist())
        gr = nx.Graph()
        all_rows = range(0, new_adjmat.shape[0])
        for n in all_rows:
            gr.add_node(n)
        gr.add_edges_from(edges)
        pos = nx.nx_pydot.graphviz_layout(gr)
        #pos = nx.nx_pydot.graphviz_layout(gr, prog="neato")
        #pos = nx.spring_layout(gr,scale=5)
        #pos = nx.spring_layout(gr, scale=10)

        nx.draw(gr, labels=labels, pos= pos, with_labels=True, font_size=7, font_family='serif', node_size=600, edge_color='dimgray', node_color='lightsteelblue', font_weight='bold')
        '''
        nx.draw(gr, with_labels=True,
        node_color='skyblue', node_size=2200, width=3, edge_cmap=plt.cm.OrRd,
        arrowstyle='->',arrowsize=20,
        font_size=10, font_weight="bold",
        pos=nx.random_layout(gr, seed=13))
        '''
        #plt.figure(figsize=(64,64))
        # fig.savefig(f"../logs/{config.a_pre_file.split('.')[0]}.pdf", dpi=dpi, bbox_inches='tight')
        plt.savefig(config.data_folder_prefix + 'training_data/knowledge/Plot_adj_graph.png', dpi=1000)
        #plt.show()

def thesis_output(feature_2_iri, responsible_relations, linked_features):
    def shorten_feature(feature):
        limit = 35
        return feature[0:limit - 2] + '...' if len(feature) > limit else feature

    def combine_relations(relations):
        '''
        rel_2_int = {
            'no_relation': 0, 'self_loops': 1, 'component': 2, 'same_iri': 3, 'connection': 4,
            'actuation': 5, 'calibration': 6, 'precondition': 7, 'postcondition': 8,
        }
        '''
        rel_2_int = {
            'no_relation': 0, 'hosts-directly-addressable-component': 1, 'has-pipe-connection': 2, 'has-same-higher-level-component': 3, 'has-same-controller': 4,
            'has-same-property_1': 5, 'has-same-property_2': 6, 'has-same-property_3': 7, 'property-used-to-control': 8,
            'property-of-higher-level-comp-used-to-control': 9, 'not-in-same-state': 10, 'has-same-potential-failure-mode': 11
        }
        print("-------------")
        print(rel_2_int)
        print("-------------")
        #for rel in sorted(relations):
        #    print("rel: ", rel, rel.split("#")[0], str(rel_2_int.get(rel.split("#")[0])))

        relations = [str(rel_2_int.get(rel.split("#")[0])) for rel in sorted(relations)]
        relations = [rel for rel in sorted(relations)]
        return ', '.join(relations)

    features, iris = [], []

    for feature, iri in feature_2_iri.items():
        features.append(feature)
        iris.append(iri)

    features = [shorten_feature(f) for f in features]
    data = np.array([features, iris]).T
    features_2_iri_df = pd.DataFrame(columns=['Datenstrom', 'IRI'], data=data)
    features_2_iri_df.index.name = 'Index'
    # features_2_iri_df = features_2_iri_df.sort_values(by='IRI', ascending=True)
    print(features_2_iri_df.to_latex(longtable=True, label='tab:streams2iri'))

    rows = []
    for a, (b, c) in zip(responsible_relations, linked_features):
        rows.append([a, b, c])

    df = pd.DataFrame(data=rows, columns=[
                      'Relation', 'Feature 1', 'Feature 2'])

    df['F1'] = df.apply(lambda x: x['Feature 1'] if x['Feature 1']
                        > x['Feature 2'] else x['Feature 2'], axis=1)
    df['F2'] = df.apply(lambda x: x['Feature 1'] if x['Feature 1']
                        < x['Feature 2'] else x['Feature 2'], axis=1)
    df = df.sort_values(by=['F1', 'F2'], ascending=False)
    df = df.drop_duplicates(subset=['F1', 'F2', 'Relation'])
    df = df.groupby(['F1', 'F2'])['Relation'].apply(
        combine_relations).reset_index()
    df = df.drop(df.loc[df['Relation'] == 'component'].index).reset_index()

    df['Datenstrom 1'] = df['F1'].apply(shorten_feature)
    df['Datenstrom 2'] = df['F2'].apply(shorten_feature)
    df = df[['Datenstrom 1', 'Relation', 'Datenstrom 2']]

    print(df.to_string())
    # print(df.to_latex(longtable=True, label='tab:relations', index=False))

def feature_tuples_from_iri_tuples(iri_tuples, iri_2_features: dict):
    feature_tuples = []

    # Create all feature pairs for each iri pair (some features are mapped to the same iri)
    for iri_1, iri_2 in iri_tuples:

        if not iri_1 in iri_2_features.keys() or not iri_2 in iri_2_features.keys():
            continue

        features_iri_1 = iri_2_features.get(iri_1)
        features_iri_2 = iri_2_features.get(iri_2)
        pairs = list(itertools.product(features_iri_1, features_iri_2))
        feature_tuples.extend(pairs)

    feature_tuples = tuple_corrections(feature_tuples)

    return feature_tuples


def a_analysis(a_df: pd.DataFrame):
    print('\nFeatures without links:')
    temp = a_df.loc[(a_df == 0).all(axis=1)]
    print(*temp.index.values, sep='\n')
    print()

    # noinspection PyArgumentList
    temp = a_df.sum(axis=0, skipna=True).sort_values(ascending=False)
    print(temp.to_string())


def prepare_query(a, r1, direct_relation, symmetric_relation, r2=None, r3=None, r4=None, is_chain=False, r5=None, r6=None):
    if not direct_relation:
        #assert r2 is not None, 'if not direct, a second relation must be passed'
        if r2 == None:
            return "SELECT ?z2 WHERE { ?x <" + r1 + "> ?y. <" + a + "> ?r ?x . ?z2 ?r2 ?y . }"
        if r3 ==None and r4 == None:
            if is_chain == False:
                return "SELECT ?x WHERE {{ <" + a + "> <" + r1 + "> ?y . ?x <" + r1 + "> ?y . } " + \
                       "UNION { ?y <" + r2 + "> <" + a + "> .  ?y <" + r2 + "> ?x .}}"
            else:
                q = "SELECT ?x WHERE { <" + a + "> <" + r1 + "> ?y . ?x <" + r2 + "> ?y . } "
                #print("q: ", q)
                return  q
        else:
            if r5 == None and r6 == None:
                if is_chain == False:
                    # Indirect connection such as observable property
                    q = "SELECT ?y WHERE {{ <" + a + "> <" + r1 + "> ?y . ?y <" + r3 + "> ?x . } " + \
                           "UNION { ?x <" + r2 + "> ?y .  ?y <" + r4 + "> <" + a + "> .}}"

                    #print("q: ", q)
                    return q
                else:
                    # Indirect connection such as observable property
                    q = "SELECT ?y WHERE {{ <" + a + "> <" + r1 + "> ?x . ?x <" + r3 + "> ?y . } " + \
                        "UNION { ?y <" + r2 + "> ?x . ?x <" + r4 + "> <" + a + "> .}"+ \
                        "UNION { ?x <" + r3 + "> ?y . ?x <" + r4 + "> <" + a + "> .}" + \
                        "UNION { ?y <" + r2 + "> ?x . <" + a + "> <" + r1 + "> ?x .}}"

                    #print("q: ", q)
                    return q
            else:
                if is_chain == False:
                    # Observable property component
                    q = "SELECT ?y WHERE {{ <" + a + "> <" + r1 + "> ?x . ?x <" + r3 + "> ?z . ?y <" + r5 + "> ?z . } " + \
                        "UNION { ?z <" + r6 + "> ?y .  ?z <" + r4 + "> ?x. ?x <" + r2 + "> <" + a + "> .}}"
                    #print("q: ", q)
                    return q
                else:
                    # SELECT ?x ?z ?c WHERE { ?x <http://www.w3.org/ns/sosa/observes> ?y. ?y <http://iot.uni-trier.de/FTOnto#used_to_control> ?z. ?z <http://iot.uni-trier.de/FTOnto#hasComponent> ?c }
                    q = "SELECT ?y WHERE { <" + a + "> <" + r1 + "> ?x . ?x <" + r3 + "> ?z . ?z <" + r5 + "> ?y . } "
                    #print("q: ", q)
                    return q
            ''' USED FOR Quieries with a middle entity    
            SELECT ?r ?x ?r1 ?y WHERE {
            { <http://iot.uni-trier.de/FTOnto#AccSensor_ADXL345_1> <http://www.w3.org/ns/sosa/observes> ?x . ?x <http://www.w3.org/ns/ssn/isPropertyOf> ?y. } 
            UNION
             { ?y <http://www.w3.org/ns/ssn/hasProperty> ?x . ?x <http://www.w3.org/ns/sosa/isObservedBy> <http://iot.uni-trier.de/FTOnto#AccSensor_ADXL345_1> . }
            }
            '''

    if symmetric_relation:
        return "SELECT ?x WHERE {{ <" + a + "> <" + r1 + "> ?x . } " + \
               " UNION { ?x <" + r1 + "> <" + a + "> . }}"
    else:
        assert r2 is not None, 'if not symmetric, a second relation must be passed'
        return "SELECT ?x WHERE {{ <" + a + "> <" + r1 + "> ?x . } " + \
               "UNION { <" + a + "> <" + r2 + "> ?x . }" + \
               "}"


def infer_connections(feature_2_iri, iri_2_features, r1, direct_relation, symmetric_relation, r2=None, r3=None, r4=None, iri_2_components=None,is_chain=False, r5=None,r6=None):
    tuples = []

    iris = list(set(feature_2_iri.values()))

    for name, iri in feature_2_iri.items():
        if iri is None:
            continue

        q = prepare_query(iri, r1, direct_relation=direct_relation, symmetric_relation=symmetric_relation, r2=r2, r3=r3, r4=r4, is_chain=is_chain, r5=r5, r6=r6)
        #print("q: ", q)
        try:
            results = list(owl.default_world.sparql(q))
            #print("results: ", results)
        except ValueError:
            results = []
            print(f'Query error for feature {name} with assigned IRI {iri} with query: {q}')
        if not results == []:
            print("q: ", q)
            print("results: ", results)
        # Since results are in form of owl-file-name.Label, we replace them with the ontology namespace / prefix: http://iot.uni-trier.de/FTOnto#
        relevant_results = [str(res[0]).replace(config.ftonto_file_name + ".", 'http://iot.uni-trier.de/FTOnto#') for res in results]

        #relevant_results = [str(res[0]).replace('.', ':') for res in results]
        #print("relevant_results: ", relevant_results)
        relevant_results = [iri for iri in relevant_results if iri in iris]
        #print("relevant_results: ", relevant_results)
        f = []
        for res_iri in relevant_results:
            #print("res_iri: ", res_iri)
            if res_iri in iri_2_features:
                f.extend(iri_2_features.get(res_iri))
            else:
                if not iri_2_components == None:
                    if res_iri in iri_2_components:
                        f.extend(iri_2_components.get(res_iri))
                    else:
                        print("IRI not found as data stream feature: ", res_iri, " for ", name, "with relation: ", r1)

        tuples.extend([(name, res_name) for res_name in f])

    return tuples

# Generates a heat map of the reconstruction error and shows the real input and its reconstruction
def plot_heatmap_of_reconstruction_error2(input, output, rec_error_matrix, id, y_label=""):
    #print("example id: ", id)
    fig, axs = plt.subplots(4,3, gridspec_kw = {'wspace':0.05, 'hspace':0.05}) # rows/dim
    fig.suptitle('Reconstruction Error of '+str(id))
    fig.subplots_adjust(hspace=0, wspace=0)
    #print("input shape: ", input.shape)
    #print("output shape: ", output.shape)
    #print("rec_error_matrix shape: ", rec_error_matrix.shape)
    #X_valid_y[i_example, :, :, :],
    #output = pred[i_example, :, :, :], rec_error_matrix = reconstruction_error_matrixes[i_example, :, :, :]

    # the left end of the figure differs
    bottom = 0.05
    height = 0.9
    width = 0.15  # * 4 = 0.6 - minus the 0.1 padding 0.3 left for space
    left1, left2, left3, left4 = 0.05, 0.25, 1 - 0.25 - width, 1 - 0.05 - width

    for dim in range(3):
        #print("dim: ", dim)
        axs[0,dim].imshow(input[:,:,dim], cmap='hot', interpolation='nearest')
        #cax = plt.axes([0.95, 0.05, 0.05, 0.9])
        axs[1,dim].imshow(output[:,:,dim], cmap='hot', interpolation='nearest')
        axs[2,dim].imshow(rec_error_matrix[:,:,dim], cmap='hot', interpolation='nearest')
        pos0 = axs[3, dim].imshow(rec_error_matrix[:, :, dim], cmap='hot', interpolation='nearest')
        cbar = fig.colorbar(pos0, ax=axs[3, dim],orientation='horizontal',shrink=0.50)
        cbar.ax.tick_params(labelsize=3, labelrotation=45)
        #plt.colorbar(axs[0,dim])
        #axs[1, dim].colorbar()
        #axs[2, dim].colorbar()
        axs[0,dim].set_xticks([], [])  # note you need two lists one for the positions and one for the labels
        axs[0,dim].set_yticks([], [])  # same for y ticks
        axs[1,dim].set_xticks([], [])  # note you need two lists one for the positions and one for the labels
        axs[1,dim].set_yticks([], [])  # same for y ticks
        axs[2,dim].set_xticks([], [])  # note you need two lists one for the positions and one for the labels
        axs[2,dim].set_yticks([], [])  # same for y ticks
        axs[3,dim].set_xticks([], [])  # note you need two lists one for the positions and one for the labels
        axs[3,dim].set_yticks([], [])  # same for y ticks
        #axs[0,dim].set_aspect('equal')
        #axs[1, dim].set_aspect('equal')
        #axs[2, dim].set_aspect('equal')
        plt.subplots_adjust(hspace=.005)
    #plt.imshow(rec_error_matrix, cmap='hot', interpolation='nearest')
    filename="heatmaps/" +str(id)+"-"+"rec_error_matrix_"+y_label+".png"
    #print("filename: ", filename)
    fig.tight_layout()
    #fig.subplots_adjust(wspace=1.1,hspace=-0.1)
    #plt.subplots_adjust(bottom=0.1,top=0.2, hspace=0.1)
    plt.savefig(filename, dpi=700)
    plt.clf()
    #print("plot_heatmap_of_reconstruction_error")



if __name__ == '__main__':
    config = Configuration()
    dataset = FullDataset(config.training_data_folder, config, training=False, model_selection=False)
    dataset.load(selected_class=0)
    #dataset = Dataset(config.training_data_folder, config)
    #dataset.load()

    print()
    adj_mat_expert = [
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1,
         1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1,
         1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0,
         1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
         0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
         1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
         1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
         1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
         1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
         0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1,
         0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1,
         0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1,
         1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1,
         0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1,
         0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1,
         0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0,
         0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
         0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
         1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
         1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
         0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]]

    adj_mat_expert_gcn_processed = [
        [0.25,0.25,0.25,0,0,0,0,0,0,0,0,0,0,0,0,0.25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0.25,0.25,0.25,0,0,0,0,0,0,0,0,0,0,0,0,0.25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0.25,0.25,0.25,0,0,0,0,0,0,0,0,0,0,0,0,0.25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0.25,0.25,0.25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0.25,0.25,0.25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0.25,0.25,0.25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0.2,0,0,0,0,0,0,0,0,0,0.2,0.2,0.2,0.2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0.166666672,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.166666672,0.166666672,0,0,0,0,0,0,0.166666672,0.166666672,0.166666672,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0.333333343,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.333333343,0.333333343,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0.0909090936,0.0909090936,0.0909090936,0.0909090936,0.0909090936,0.0909090936,0.0909090936,0.0909090936,0.0909090936,0.0909090936,0.0909090936,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0.0909090936,0.0909090936,0.0909090936,0.0909090936,0.0909090936,0.0909090936,0.0909090936,0.0909090936,0.0909090936,0.0909090936,0.0909090936,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0,0,0,0.0833333358,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0.0526315793,0.0526315793,0.0526315793,0.0526315793,0.0526315793,0.0526315793,0.0526315793,0.0526315793,0.0526315793,0.0526315793,0.0526315793,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0526315793,0.0526315793,0.0526315793,0.0526315793,0.0526315793,0.0526315793,0.0526315793,0.0526315793,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0.0526315793,0.0526315793,0.0526315793,0.0526315793,0.0526315793,0.0526315793,0.0526315793,0.0526315793,0.0526315793,0.0526315793,0.0526315793,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0526315793,0.0526315793,0.0526315793,0.0526315793,0.0526315793,0.0526315793,0.0526315793,0.0526315793,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0.0526315793,0.0526315793,0.0526315793,0.0526315793,0.0526315793,0.0526315793,0.0526315793,0.0526315793,0.0526315793,0.0526315793,0.0526315793,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0526315793,0.0526315793,0.0526315793,0.0526315793,0.0526315793,0.0526315793,0.0526315793,0.0526315793,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0.0714285746,0.0714285746,0.0714285746,0,0,0,0,0,0,0.0714285746,0.0714285746,0.0714285746,0.0714285746,0.0714285746,0.0714285746,0.0714285746,0.0714285746,0.0714285746,0.0714285746,0.0714285746,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0.0833333358,0,0,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0.0833333358,0,0,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0.0833333358,0,0,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0.0833333358,0,0,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0.0909090936,0,0,0,0,0,0,0,0,0.0909090936,0.0909090936,0.0909090936,0.0909090936,0.0909090936,0.0909090936,0.0909090936,0.0909090936,0.0909090936,0.0909090936,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0,0,0.0833333358,0,0,0.0833333358,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0.0769230798,0.0769230798,0.0769230798,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0769230798,0.0769230798,0.0769230798,0.0769230798,0.0769230798,0.0769230798,0.0769230798,0.0769230798,0.0769230798,0.0769230798,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0.0714285746,0,0,0,0,0,0,0,0,0,0,0,0,0.0714285746,0.0714285746,0.0714285746,0.0714285746,0.0714285746,0.0714285746,0.0714285746,0.0714285746,0.0714285746,0.0714285746,0,0,0,0,0,0,0.0714285746,0.0714285746,0.0714285746,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0.0625,0,0,0,0,0,0,0,0,0,0,0,0,0.0625,0.0625,0.0625,0.0625,0.0625,0.0625,0.0625,0.0625,0.0625,0.0625,0,0,0,0.0625,0,0.0625,0.0625,0.0625,0.0625,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.111111112,0.111111112,0.111111112,0.111111112,0.111111112,0.111111112,0.111111112,0.111111112,0.111111112,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.111111112,0.111111112,0.111111112,0.111111112,0.111111112,0.111111112,0.111111112,0.111111112,0.111111112,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.1,0,0,0,0,0,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.055555556,0.055555556,0.055555556,0.055555556,0.055555556,0.055555556,0.055555556,0.055555556,0.055555556,0.055555556,0,0.055555556,0.055555556,0.055555556,0.055555556,0.055555556,0.055555556,0.055555556,0.055555556,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.111111112,0.111111112,0.111111112,0.111111112,0.111111112,0.111111112,0.111111112,0.111111112,0.111111112,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0909090936,0,0,0,0,0.0909090936,0.0909090936,0.0909090936,0.0909090936,0.0909090936,0.0909090936,0.0909090936,0.0909090936,0.0909090936,0.0909090936,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0.0833333358,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0.0833333358,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0.0833333358,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0.0833333358,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0.0769230798,0.0769230798,0.0769230798,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0769230798,0,0,0,0,0,0,0.0769230798,0.0769230798,0.0769230798,0.0769230798,0.0769230798,0.0769230798,0.0769230798,0.0769230798,0,0,0,0.0769230798,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0.0769230798,0.0769230798,0.0769230798,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0769230798,0,0,0,0,0,0,0.0769230798,0.0769230798,0.0769230798,0.0769230798,0.0769230798,0.0769230798,0.0769230798,0.0769230798,0,0,0,0.0769230798,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0.0769230798,0.0769230798,0.0769230798,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0769230798,0,0,0,0,0,0,0.0769230798,0.0769230798,0.0769230798,0.0769230798,0.0769230798,0.0769230798,0.0769230798,0.0769230798,0,0,0,0.0769230798,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0.0625,0.0625,0.0625,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0625,0,0,0,0,0,0,0.0625,0.0625,0.0625,0.0625,0.0625,0.0625,0.0625,0.0625,0.0625,0.0625,0.0625,0.0625,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0.0769230798,0.0769230798,0.0769230798,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0769230798,0,0,0,0,0,0,0.0769230798,0.0769230798,0.0769230798,0.0769230798,0.0769230798,0.0769230798,0.0769230798,0.0769230798,0,0,0,0.0769230798,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0.0769230798,0.0769230798,0.0769230798,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0769230798,0,0,0,0,0,0,0.0769230798,0.0769230798,0.0769230798,0.0769230798,0.0769230798,0.0769230798,0.0769230798,0.0769230798,0,0,0,0.0769230798,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0.0714285746,0,0,0,0.0714285746,0.0714285746,0.0714285746,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0714285746,0,0,0,0,0,0,0.0714285746,0.0714285746,0.0714285746,0.0714285746,0.0714285746,0.0714285746,0.0714285746,0.0714285746,0,0,0,0.0714285746,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0.0714285746,0,0,0,0.0714285746,0.0714285746,0.0714285746,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0714285746,0,0,0,0,0,0,0.0714285746,0.0714285746,0.0714285746,0.0714285746,0.0714285746,0.0714285746,0.0714285746,0.0714285746,0,0,0,0.0714285746,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.25,0,0,0,0,0.25,0.25,0.25,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.25,0,0,0,0,0.25,0.25,0.25,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.25,0,0,0,0,0.25,0.25,0.25,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.055555556,0.055555556,0.055555556,0.055555556,0.055555556,0.055555556,0.055555556,0.055555556,0,0,0,0.055555556,0.055555556,0.055555556,0.055555556,0.055555556,0.055555556,0.055555556,0.055555556,0.055555556,0.055555556],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]]

    adj_mat_1_learned = [
        [0.164607659, 0, 0, 0, 0, 0, 0, 0, 0, 0.198957, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.156983793, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.188337803, 0, 0, 0, 0, 0, 0, 0, 0, 0.145305127, 0, 0, 0.145808652, 0],
[0, 0.130688593, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.125436202, 0, 0, 0, 0, 0.128864303, 0, 0, 0, 0.135053009, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.119493872, 0.123548932, 0, 0, 0, 0.115819722, 0, 0, 0.121095382, 0, 0, 0],
[0, 0, 0.17198427, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.156462595, 0, 0, 0.169046506, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.187420785, 0.139711514, 0, 0, 0, 0, 0, 0, 0, 0.175374299, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0.0991936252, 0, 0, 0.117085077, 0, 0, 0, 0, 0, 0, 0.106238268, 0, 0, 0, 0, 0, 0, 0, 0.124949053, 0, 0, 0, 0, 0, 0.117252119, 0, 0, 0, 0.11212872, 0, 0, 0, 0, 0, 0, 0, 0, 0.0974875242, 0.118904538, 0, 0, 0, 0, 0, 0, 0.106761083, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0.105748147, 0, 0, 0, 0.0856871083, 0.115978666, 0, 0, 0.106523134, 0, 0.0893434137, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0847702175, 0.131388053, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.108703949, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0877343193, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0841229931],
[0, 0, 0, 0.0960600674, 0.114229821, 0.105249204, 0, 0, 0, 0, 0, 0.0913721099, 0, 0.105066478, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0949329883, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0961937308, 0, 0, 0, 0.1050689, 0, 0, 0, 0, 0, 0, 0, 0.100314662, 0, 0.0915120691, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0.489067048, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.510933, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0.262037069, 0, 0.259077519, 0, 0, 0, 0.249639466, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.229245946, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0.126272991, 0, 0, 0, 0, 0, 0, 0, 0.115087323, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.118893713, 0, 0, 0, 0.105585, 0, 0, 0, 0, 0, 0, 0, 0.107089765, 0.147766426, 0, 0.152965635, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.126339138, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0.268774629, 0, 0, 0, 0, 0, 0, 0.262877256, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.242670238, 0, 0, 0, 0.225677893, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0.0910477117, 0, 0, 0.120518953, 0, 0, 0.115943357, 0, 0, 0, 0, 0, 0, 0, 0, 0.104343556, 0, 0, 0, 0, 0, 0, 0, 0, 0.108207658, 0, 0, 0, 0, 0, 0, 0.129403055, 0, 0, 0.106143802, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.113325916, 0, 0, 0, 0, 0.111066, 0, 0],
[0.0964543819, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.117676765, 0, 0, 0, 0, 0, 0, 0, 0.130829528, 0.0998270884, 0, 0, 0.12156795, 0.108605333, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.120460033, 0, 0, 0, 0, 0.101113364, 0, 0, 0, 0, 0, 0, 0, 0.103465565, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.161659047, 0, 0, 0, 0, 0, 0, 0.170593545, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.127765581, 0, 0, 0, 0, 0.161067367, 0, 0, 0.188661128, 0, 0, 0, 0, 0, 0, 0, 0, 0.190253422, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.18089585, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.148528352, 0, 0, 0, 0, 0, 0, 0, 0, 0.173008338, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.168885186, 0, 0.154706508, 0, 0, 0, 0.173975721, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.22122325, 0.189884931, 0, 0, 0.21392335, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.200177506, 0.174790964, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0.0896567, 0, 0, 0, 0.0940810144, 0, 0, 0.100576371, 0, 0.10050074, 0, 0, 0.0997545719, 0, 0, 0, 0, 0, 0, 0.0967653617, 0.0865605697, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.101767294, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.115095437, 0, 0, 0, 0.115241893],
[0, 0, 0, 0, 0, 0.476965, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.52303493, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0.151873112, 0, 0, 0, 0, 0, 0.162954524, 0, 0, 0, 0, 0, 0, 0, 0, 0.170888424, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.161447868, 0, 0, 0, 0, 0, 0, 0, 0.201647714, 0.151188314, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0.155791864, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.151318863, 0, 0, 0, 0, 0, 0.164299905, 0.124944299, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.128097028, 0, 0, 0, 0, 0.147594899, 0, 0, 0.127953172, 0, 0, 0, 0, 0],
[0, 0, 0.175073564, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.266775668, 0, 0, 0, 0.206585824, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.163721517, 0, 0, 0, 0, 0, 0, 0, 0.187843472, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0.24757345, 0.225134119, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.287104189, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.240188271, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0.122889414, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.132354721, 0, 0, 0, 0.132912, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.117664441, 0, 0, 0, 0, 0, 0.115888618, 0, 0.127371386, 0, 0, 0, 0, 0, 0, 0, 0, 0.1201628, 0, 0, 0, 0, 0.130756587, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0.250111192, 0, 0, 0, 0, 0, 0.140082866, 0, 0, 0, 0, 0, 0, 0, 0.112162091, 0, 0.271418512, 0, 0.108864099, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.117361225, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0.101277091, 0, 0, 0, 0, 0, 0, 0, 0.116858497, 0, 0, 0, 0, 0, 0.104179524, 0.113766767, 0, 0, 0, 0, 0.110068478, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.111081205, 0, 0, 0, 0, 0, 0, 0.132464215, 0, 0, 0.0976538658, 0, 0, 0, 0, 0, 0, 0, 0, 0.112650372],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.172655061, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.140233979, 0, 0.163453907, 0, 0, 0.165263176, 0, 0, 0.152752146, 0.205641687, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0.214271665, 0, 0, 0, 0, 0, 0.304030389, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.267741054, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.213956907, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.250001967, 0, 0.233784482, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.272237629, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.243975967, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0.142152697, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.153737307, 0, 0.169016212, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.181304023, 0.185777739, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.168012008, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0.267908275, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.382786453, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.349305332, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0.172753215, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.15572907, 0, 0.184288919, 0, 0, 0, 0, 0, 0, 0, 0.164225549, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.160305962, 0, 0, 0.1626973, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.114869282, 0, 0.103199512, 0.0985219255, 0, 0, 0, 0, 0.118221395, 0, 0.132079259, 0, 0, 0, 0, 0, 0, 0.114040159, 0, 0, 0, 0, 0, 0.109698415, 0.111041673, 0, 0, 0, 0.0983283147, 0, 0],
[0.114439674, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.178895786, 0, 0, 0.144406348, 0, 0.120673709, 0, 0, 0, 0, 0.123516642, 0, 0, 0, 0, 0.138673604, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1793942, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.242045447, 0, 0.202554852, 0, 0, 0.157182395, 0, 0, 0.187034711, 0, 0, 0, 0, 0, 0, 0.211182699, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.141039938, 0, 0, 0, 0.145207062, 0, 0, 0, 0.132101536, 0, 0, 0, 0, 0, 0, 0.126079932, 0, 0, 0, 0.145899594, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.149323732, 0, 0, 0, 0.160348192, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0.125234857, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.143713161, 0, 0, 0, 0, 0, 0.132206514, 0, 0.139447093, 0, 0, 0, 0, 0, 0.147066414, 0, 0, 0, 0, 0, 0.151812226, 0, 0, 0, 0.160519674, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.134376571, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.139190659, 0, 0, 0, 0, 0, 0.142871201, 0, 0, 0, 0, 0.163878813, 0.160717696, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.125705093, 0, 0, 0, 0, 0, 0, 0.133259907, 0],
[0, 0, 0, 0, 0.323268026, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.340474963, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.336257, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.138978392, 0, 0, 0, 0, 0, 0, 0.126188189, 0, 0, 0, 0, 0, 0, 0.16005972, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.155280635, 0.142790198, 0.132605046, 0, 0, 0, 0, 0, 0, 0, 0.144097775, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.301851898, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.34935534, 0, 0, 0, 0, 0.348792791, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0.149626464, 0, 0, 0, 0, 0, 0, 0.193591207, 0, 0, 0, 0, 0.155536726, 0, 0, 0, 0, 0.16062817, 0, 0, 0, 0, 0, 0, 0, 0.153657392, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.18696, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0.113157727, 0, 0, 0, 0.121418275, 0, 0, 0, 0, 0, 0.143758178, 0, 0.11114046, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.131439716, 0, 0, 0.11879427, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.129994482, 0.130296826, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.369208574, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.322196901, 0, 0, 0, 0, 0, 0, 0.308594435, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.277062833, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.243623689, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.205265373, 0, 0.27404809],
[0, 0, 0, 0.105106048, 0, 0, 0, 0, 0, 0.128913537, 0.0992694721, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.119975977, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.100258179, 0, 0.1028485, 0, 0, 0, 0, 0.10784705, 0, 0, 0.117695883, 0, 0, 0, 0, 0.118085399, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0.130997241, 0.162466362, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.146634027, 0, 0, 0, 0, 0, 0, 0, 0, 0.11455588, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.146203607, 0, 0, 0, 0, 0, 0.157316774, 0.141826093, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.196956903, 0, 0, 0.232820764, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.25484702, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.315375268, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.125970662, 0, 0, 0.100702345, 0, 0, 0, 0, 0.104800895, 0.12082129, 0.099163, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.109315902, 0, 0, 0, 0, 0, 0, 0.117259651, 0, 0, 0, 0, 0, 0, 0, 0, 0.0933924168, 0.12857388, 0, 0],
[0, 0, 0, 0, 0, 0.218461856, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.22178264, 0, 0, 0.29311204, 0.266643465, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0.163534135, 0, 0, 0, 0.156882063, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.204194397, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.324011296, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.151378095],
[0, 0, 0.181135327, 0, 0, 0, 0, 0, 0, 0, 0, 0.240654528, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.174460188, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.211780787, 0, 0.191969275, 0, 0, 0, 0, 0, 0, 0],
[0.0793411583, 0, 0, 0, 0, 0, 0, 0.0749947429, 0, 0.0984904766, 0, 0.0827988759, 0, 0, 0, 0, 0, 0, 0, 0.0865937844, 0, 0, 0, 0, 0, 0.0785781, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0904073194, 0, 0.0687717, 0, 0, 0, 0, 0.0849983469, 0, 0, 0, 0.174996, 0, 0, 0, 0, 0, 0, 0.0800294727, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.25676012, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.230318561, 0, 0, 0, 0, 0.244070515, 0, 0, 0, 0, 0, 0.268850714, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.227834851, 0, 0, 0, 0.238474, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.270605505, 0, 0, 0, 0, 0, 0, 0, 0.263085693, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.151385874, 0, 0.15466933, 0.127473593, 0.133151427, 0, 0.142078042, 0, 0, 0, 0.130220264, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.161021501, 0, 0, 0, 0, 0],
[0, 0.206864536, 0, 0, 0, 0.175676629, 0, 0, 0, 0, 0.200811684, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.196099713, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.220547378, 0, 0, 0, 0],
[0, 0.100014307, 0, 0, 0, 0, 0, 0.0971573517, 0.128475666, 0, 0, 0, 0, 0, 0, 0, 0, 0.103453495, 0, 0, 0, 0, 0, 0, 0, 0, 0.130162984, 0, 0, 0, 0, 0, 0, 0, 0.0984804109, 0, 0.0895915627, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.132240221, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.120423988, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.321644425, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.344278336, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.33407715, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0.199674219, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.183716133, 0, 0, 0, 0.209955975, 0, 0, 0, 0, 0.210678548, 0, 0, 0, 0, 0.195975184, 0],
[0, 0.148173854, 0, 0, 0, 0, 0.156275883, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.161944821, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.151406944, 0.204329252, 0, 0, 0, 0, 0.177869245]]

    adj_mat_2_learned = [
        [0.105437264, 0, 0.0940903425, 0, 0, 0.103175886, 0, 0, 0, 0.0962696, 0, 0, 0, 0, 0.094419539, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.106223211, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.102324389, 0, 0, 0, 0.107520126, 0.0954246894, 0, 0, 0, 0, 0.0951149538, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0.146415278, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.144423336, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.119431265, 0, 0, 0, 0, 0.145498767, 0.12718004, 0.164623499, 0, 0, 0, 0, 0, 0.152427807, 0, 0, 0, 0, 0],
[0, 0, 0.362189025, 0, 0, 0, 0, 0, 0, 0, 0.355013311, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.282797605, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0.35440293, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.178548753, 0.158206239, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.156896725, 0, 0, 0, 0, 0, 0, 0.151945293, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0.206865504, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.192271799, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.192404643, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.206274822, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.202183262, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0.139103651, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.132087499, 0.134540945, 0, 0.14149484, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.144435361, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.141549379, 0, 0.16678834, 0],
[0, 0, 0, 0, 0.162171781, 0, 0.146950081, 0, 0, 0.146106288, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.140637517, 0, 0.127560124, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.14168179, 0, 0, 0, 0, 0.134892434, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0.204273537, 0, 0, 0.140074983, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.204944029, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.219906628, 0, 0, 0.230800793],
[0, 0, 0, 0, 0, 0.108011395, 0, 0, 0.115250751, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.111071, 0, 0, 0, 0, 0, 0, 0, 0, 0.0965944827, 0, 0.099516362, 0, 0, 0.11061731, 0, 0, 0, 0, 0, 0.133002594, 0, 0, 0, 0, 0, 0, 0, 0, 0.118008628, 0, 0, 0, 0, 0, 0, 0.107927389, 0, 0, 0, 0, 0, 0],
[0, 0, 0.0994662344, 0, 0, 0, 0, 0.095482409, 0.091916889, 0.091602169, 0, 0, 0, 0, 0, 0, 0, 0, 0.0937188044, 0, 0, 0.0798793137, 0, 0, 0, 0, 0, 0, 0, 0, 0.0783832446, 0, 0.0963947102, 0, 0.105084807, 0, 0, 0, 0, 0, 0, 0, 0.0798119381, 0, 0, 0.0882595, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.166510418, 0, 0, 0.141697228, 0, 0, 0.205475017, 0, 0, 0, 0, 0.148077965, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.168022022, 0, 0, 0.170217246, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0.232602909, 0, 0, 0.198369652, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.216058105, 0, 0, 0, 0, 0.181851894, 0, 0, 0, 0, 0.17111747, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0.23519139, 0, 0, 0, 0, 0, 0, 0, 0.20833078, 0, 0.275300026, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.281177789, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0.242500946, 0, 0, 0.231757358, 0.257868767, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2678729, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.206743687, 0, 0, 0, 0, 0, 0, 0.173171014, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.207340479, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.183773264, 0, 0, 0, 0, 0, 0, 0.228971511, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.99999994, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.279179543, 0, 0, 0, 0.226745307, 0, 0.238727763, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.255347371, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0.182460308, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.203559697, 0, 0.204212874, 0, 0, 0, 0, 0, 0.188175097, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.221592039, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.252806038, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.224791288, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.249310851, 0, 0, 0, 0, 0, 0, 0.273091853, 0],
[0, 0, 0, 0.24395296, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.263165206, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.263552845, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.229329035, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.217316628, 0, 0, 0, 0.443064302, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.188596874, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.151022226, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.152350172, 0, 0.137314349, 0, 0.136732191, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.148577288, 0, 0, 0.151886031, 0.127853572, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.145286411, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0.2632218, 0, 0, 0, 0.228180319, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.259938359, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.248659432, 0, 0, 0, 0],
[0.174466893, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.184073314, 0, 0, 0, 0, 0, 0, 0.282640666, 0, 0, 0, 0, 0, 0, 0, 0, 0.141167015, 0, 0, 0, 0, 0, 0, 0, 0.217652127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0.194902539, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.164686516, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.147573292, 0, 0, 0, 0, 0.19377844, 0, 0, 0, 0, 0, 0, 0.141737819, 0.157321438, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0.218090475, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.200106069, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.168453097, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.227527887, 0, 0, 0.185822502, 0, 0, 0, 0, 0],
[0, 0, 0.280816406, 0.224597141, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.183029786, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.161201477, 0, 0, 0.15035519, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0.192514524, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.195476264, 0, 0, 0, 0, 0, 0, 0.216887042, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.200054735, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.195067465, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0.13554734, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.153818771, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.178465202, 0, 0.1818088, 0.218864739, 0, 0, 0.131495059, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.178325713, 0, 0, 0, 0.161797211, 0, 0, 0, 0, 0, 0, 0, 0.195003211, 0.173245534, 0, 0, 0.152563021, 0, 0.139065266, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0.363964915, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.329980671, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.306054443, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.329702079, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.36110732, 0, 0.309190601, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0.314143687, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.358374834, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.327481449, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0.244651034, 0, 0, 0, 0.177169845, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.173472956, 0, 0, 0, 0, 0, 0, 0, 0, 0.191622987, 0, 0.213083103, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.236864984, 0, 0, 0, 0, 0, 0, 0.30174768, 0, 0.225261703, 0, 0, 0, 0, 0, 0.236125633, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0.136466831, 0, 0, 0, 0, 0, 0, 0.137653649, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.167972952, 0, 0, 0, 0, 0, 0, 0, 0.142669842, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.138510585, 0, 0.118626736, 0, 0, 0, 0, 0, 0.158099413, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.202010393, 0, 0, 0, 0, 0, 0, 0.148945272, 0, 0, 0, 0, 0, 0, 0, 0.149918094, 0, 0, 0.164406657, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.159148157, 0, 0, 0, 0, 0, 0, 0, 0, 0.175571382],
[0, 0, 0, 0, 0, 0, 0, 0.158551946, 0, 0, 0, 0.189807534, 0, 0, 0, 0, 0, 0.197226539, 0, 0, 0, 0.145000249, 0, 0, 0.14064458, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.168769136, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.255844325, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.244871706, 0.203536451, 0, 0, 0, 0, 0, 0.295747519, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.325617939, 0, 0, 0, 0.342119962, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.332262069, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0.230579913, 0, 0, 0, 0, 0, 0, 0, 0.203322262, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.154618636, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.222876057, 0, 0, 0.188603044, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.511831939, 0, 0, 0, 0, 0, 0, 0, 0, 0.48816812, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.480457634, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.519542277, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.162987366, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.174939126, 0, 0, 0.160524368, 0, 0, 0, 0, 0.165360913, 0, 0, 0.183016315, 0, 0, 0, 0, 0, 0, 0, 0.153171927, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.282078803, 0, 0, 0, 0, 0, 0, 0.241757914, 0.250840276, 0, 0, 0, 0, 0.225322917, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.154533267, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1494551, 0, 0, 0.154592469, 0.165601164, 0, 0.211641401, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.164176643, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.377574921, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.309810281, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.312614799, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.297942817, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.377867848, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.324189305, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0.178147122, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.215705737, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.211327627, 0, 0, 0.186409742, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.208409756, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0.195902646, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.198247865, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.186208397, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.198613912, 0, 0, 0, 0, 0, 0, 0, 0.221027106, 0, 0],
[0, 0, 0, 0, 0.121046118, 0, 0, 0, 0, 0, 0.118760571, 0, 0, 0.109320849, 0.142641827, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.127038315, 0, 0, 0, 0, 0.133097798, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.126673922, 0, 0.121420622, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.238077238, 0, 0.268996745, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.223743498, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.269182473, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.130340412, 0, 0, 0.126008049, 0, 0, 0, 0, 0.136130705, 0, 0, 0, 0, 0, 0.201014519, 0, 0, 0.126241297, 0, 0, 0, 0, 0, 0, 0, 0, 0.14106971, 0, 0.139195278, 0, 0, 0, 0, 0],
[0.224507838, 0.181352347, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.193325579, 0, 0, 0, 0.182030916, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.218783289, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.227773383, 0, 0, 0, 0, 0, 0.183417976, 0, 0, 0, 0, 0, 0, 0, 0, 0.189131066, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.198463261, 0, 0, 0, 0, 0.201214239],
[0, 0, 0, 0, 0, 0, 0.171098754, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.174704805, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.225996763, 0, 0, 0, 0, 0, 0.202729121, 0, 0, 0, 0.225470543],
[0, 0, 0, 0, 0, 0, 0, 0, 0.473913491, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.526086569, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.185592085, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.21054931, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.211366802, 0, 0, 0, 0, 0.392491788, 0, 0],
[0, 0.141573638, 0, 0, 0, 0, 0.185848, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.16047658, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.176331654, 0, 0, 0, 0.152832344, 0, 0, 0.18293786, 0],
[0, 0.1646678, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.182112709, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.190961257, 0, 0, 0, 0.250814468, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.211443841]]

    adj_mat_3_learned = [
        [0.480545729, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.519454241, 0, 0, 0, 0, 0, 0],
[0, 0.324679762, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.312918484, 0, 0, 0, 0, 0, 0, 0.362401783, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0.14908202, 0, 0, 0, 0, 0, 0, 0.149858221, 0, 0, 0.134904265, 0, 0, 0.136756271, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.132351816, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.156686321, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.140361086, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0.35749349, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.322078973, 0, 0, 0.320427597, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0.350781858, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.315793037, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.333425075, 0],
[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0.130086213, 0, 0, 0.120722905, 0, 0, 0, 0.120164536, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.144243717, 0, 0, 0, 0, 0, 0, 0, 0, 0.103172481, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.127587378, 0.142120644, 0, 0, 0.111902043, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0.302078426, 0, 0, 0.279149055, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.418772548, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0.240121886, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.268777281, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.245938554, 0, 0, 0.245162308, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0.239447266, 0.194746584, 0, 0.197839707, 0, 0, 0, 0, 0.200231358, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.167734906, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.164309382, 0, 0, 0, 0, 0, 0.329911143, 0, 0, 0, 0, 0, 0, 0, 0.166831136, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.146758273, 0, 0, 0, 0.192190081, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0.193598494, 0, 0, 0, 0, 0.200386703, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.227027491, 0, 0, 0, 0, 0, 0, 0, 0, 0.193848625, 0, 0, 0.185138717, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.340200365, 0, 0, 0, 0, 0, 0, 0, 0.307803124, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.351996571, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0.238401785, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.243152007, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.276144773, 0.242301494, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.178902119, 0, 0, 0, 0.160858959, 0, 0, 0, 0, 0, 0.152183577, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.179323614, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.174409077, 0, 0, 0.154322624, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.23389782, 0, 0, 0, 0.212953746, 0, 0.280726612, 0, 0, 0, 0.272421807, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.477547675, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.209542766, 0, 0, 0, 0, 0, 0, 0.312909633],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.197481766, 0, 0, 0, 0, 0.386393815, 0, 0, 0, 0.186908022, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.229216427, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0.194467589, 0, 0, 0, 0.166049436, 0, 0, 0, 0.15236032, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.169765279, 0, 0, 0, 0, 0, 0, 0, 0, 0.144944161, 0, 0.17241323, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.253540128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.250736296, 0, 0, 0, 0.262746245, 0.232977405, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.499668479, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.500331581, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0.241513684, 0, 0, 0, 0, 0, 0.256138206, 0, 0, 0, 0, 0, 0, 0, 0.271853954, 0, 0, 0, 0, 0, 0.230494052, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.463982195, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.536017776],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.244455203, 0.252637357, 0, 0, 0, 0.256508589, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.246398896, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.250052065, 0, 0, 0, 0, 0, 0, 0, 0.251547694, 0, 0.242943928, 0.255456358, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.193995938, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.202333272, 0.235824674, 0, 0, 0, 0, 0, 0.17027007, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.197576046, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0.40800494, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.318953276, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.273041755, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.280670643, 0, 0, 0, 0, 0.233416304, 0, 0, 0.264382154, 0, 0, 0, 0.221530855, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.349681407, 0, 0.334185213, 0.31613338, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.51199621, 0, 0, 0, 0, 0, 0, 0, 0, 0.488003761, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0.362795919, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.341829717, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.295374423, 0, 0, 0, 0],
[0.162851155, 0, 0.178422928, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.140417144, 0, 0, 0, 0, 0, 0, 0.165957659, 0, 0, 0, 0.155410439, 0, 0.196940705, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.314596564, 0, 0.34304285, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.342360556, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.245040193, 0, 0.243280873, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.228897527, 0, 0, 0, 0, 0.282781333, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.146011636, 0.1731361, 0, 0, 0, 0.129393414, 0, 0.158711314, 0, 0.161886364, 0, 0, 0, 0, 0.230861127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0.161237508, 0, 0, 0, 0, 0, 0, 0, 0.149229661, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.17535001, 0, 0, 0, 0, 0, 0, 0.153490722, 0, 0, 0.190858349, 0, 0, 0, 0, 0, 0, 0.169833705, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.240235105, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.250274628, 0, 0, 0, 0, 0, 0, 0, 0.252024919, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.257465392, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0.266967863, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.230897948, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.257957548, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.244176671, 0, 0, 0, 0],
[0, 0, 0.166780144, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.15364933, 0, 0, 0, 0, 0, 0.135349452, 0, 0.13477391, 0, 0.121214546, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.143505782, 0.144726858, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.335084587, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.337858468, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.327057, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.486394405, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.513605595, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0.118086375, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.134783253, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.109881178, 0, 0, 0, 0, 0, 0.120364137, 0.272280633, 0.130538851, 0, 0, 0, 0, 0, 0, 0.114065535, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.184767142, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.168120846, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.162524462, 0, 0, 0, 0.180747509, 0, 0, 0, 0, 0, 0, 0.160838857, 0, 0, 0, 0, 0, 0, 0.143001214, 0, 0],
[0, 0, 0, 0, 0, 0.208765283, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.15689899, 0, 0.184113398, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.157131121, 0, 0, 0.136046931, 0, 0, 0, 0, 0, 0, 0.157044306, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0.237719446, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.256344795, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.257452697, 0, 0, 0, 0, 0, 0, 0, 0.248483092, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0.158640787, 0, 0, 0, 0, 0.134978086, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.135272682, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.16229184, 0, 0, 0, 0, 0, 0.153949648, 0, 0.132571369, 0.122295566, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0.212422177, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.282116771, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.254011273, 0, 0.251449794, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0.521803916, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.478196, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0.323738217, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.321473032, 0, 0, 0, 0, 0.35478875, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.304810613, 0.353782445, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.341407, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.223778799, 0.251287162, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.267084271, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.257849783, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0.334589779, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.364543766, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.300866485, 0, 0, 0, 0, 0],
[0, 0.15964523, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.169188082, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.145280197, 0, 0, 0, 0, 0, 0, 0.15527457, 0, 0, 0, 0, 0, 0, 0, 0, 0.136342898, 0, 0.108318835, 0.125950173, 0],
[0, 0, 0, 0, 0.361957848, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.313570768, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.324471325, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.375188231, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.318741918, 0, 0.306069881],
[0, 0, 0, 0.146172509, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.153133422, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.220120221, 0, 0, 0, 0.14223963, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.167985484, 0, 0, 0, 0.170348734, 0],
[0, 0.208686113, 0, 0, 0, 0, 0, 0, 0.176481128, 0, 0, 0, 0, 0.141748786, 0, 0, 0, 0, 0.155836597, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.152641788, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.164605588]]

    # Variant 7 of Log_Test_ANO_cnn2d_GCN_GSL_Var7_RandInit_Owl2Vec_GCNPreProc_knn5out_Adjmasked_printed_al.txt:
    adj_mat_4_learned = [[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0.581662834,0.0234358814,0.171483144,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.223418087,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0.0590736978,0.828663409,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0104540167,0,0,0,0,0,0,0.101808876,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.134057835,0.674266577,0,0.0190364718,0,0,0.00609322358,0.166545898,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0.0115250163,0,0,0,0,0,0,0,0,0.119738,0,0,0.85118103,0,0,0.0175558962,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.138663262,0.073812753,0,0.0850385055,0,0.452120513,0.071175836,0.131536,0.0401795246,0.00747359078,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.158611506,0.0320047326,0,0.0694300383,0,0,0.600507081,0.139446616,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.136659548,0.0570405833,0,0.0627030209,0,0,0.0610929243,0.170337886,0.512165964,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.840052724,0,0.00929425564,0,0,0,0,0.0356147289,0,0,0,0,0,0.0161607042,0.0988775045,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.948975861,0,0,0,0,0,0.0510241054,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0291029885,0,0.925931,0,0,0,0,0.0449659899,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0.000250538404,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.113418594,0,0,0,0,0,0.793906271,0.0777758434,0,0,0,0.0146487411,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0785061568,0,0,0,0.0233574286,0.0131485518,0.817474425,0.00310455519,0,0.0644089058],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0645113885,0,0,0,0,0,0,0.935488641,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.116520859,0,0.0589754581,0.0613137372,0.053402856,0.0454678051,0.0386530235,0.0727723241,0.0194691867,0.533424795]]

    # Variant 8 of Log_Test_ANO_cnn2d_GCN_GSL_Var8_RandInit_Owl2Vec_GCNPreProc_knn5out_Adjmasked_printed_al.txt:
    adj_mat_5_learned = [[0.680510521,0.319489509,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0.20467703,0.43596071,0.0710078105,0,0,0,0,0,0,0,0,0,0,0,0,0.288354367,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0.140063554,0.859936476,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0.789151669,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.210848331,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0.322104692,0.128524661,0,0,0.0489562154,0,0.126581162,0.137177274,0,0.236656,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0.184360817,0.46203959,0,0,0,0,0.136091143,0.0591416955,0.0314544179,0.12691237,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0.908498883,0,0,0,0,0,0.0915011317,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0.0892326385,0,0,0,0.587101161,0.132703453,0,0,0,0,0.000314232602,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0529491194,0,0,0,0,0.137699291,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0.138090178,0.610932887,0,0.101451203,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0872692242,0,0,0,0.0622564889,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0.192491934,0,0,0,0,0,0,0,0.114368215,0.0857203677,0.0270719286,0,0,0,0.291027039,0.115214951,0.0562326126,0.117872871,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0.161575168,0.0485628396,0,0,0,0.0630018339,0.150198296,0.37939316,0,0.197268695,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0.0522732772,0.0773355812,0,0,0,0,0,0.767851591,0.102539591,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0.000414132141,0,0,0.239384815,0.0894954428,0,0,0,0,0.131964371,0.169412255,0.0435101353,0.325818807,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0.000534941035,0,0,0,0,0,0.999465048,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.368845373,0.23767978,0.130151495,0,0,0.191615865,0,0,0.0717075616,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.18782866,0.291483492,0.162932113,0.0202730391,0,0.203466699,0.130944148,0.00307181827,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.116627879,0.184752494,0.330519855,0.154471889,0,0.213627875,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.270282507,0.578317106,0.151400402,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.207478091,0.792521894,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.143777132,0.193188861,0.178880513,0,0,0.276759595,0,0.102742895,0,0.104651026,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.309980035,0,0,0,0,0.690019906,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.270730495,0,0.729269505,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0.138308451,0,0,0,0,0,0,0,0,0,0,0,0,0.100637458,0,0,0,0,0,0,0,0.517653346,0,0,0,0,0,0,0,0,0,0.243400753,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0750702471,0.0655639842,0.040823739,0,0,0.0810424685,0,0,0.057631664,0.214324504,0,0,0,0.0469397195,0,0.100844011,0.114233144,0.104643352,0.0988831595,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.959161639,0,0.0408383645,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.38311702,0,0.0842066184,0.175786644,0,0,0.219461411,0.137428254,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0408383645,0,0.959161639,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0914489254,0,0.416067541,0,0.165238112,0.0708133802,0,0,0,0,0.0279704109,0.017411191,0.0531884432,0,0,0,0.157861903,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.178151563,0,0.0683432892,0.388271213,0,0.166791707,0.19844225,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.127410784,0,0.0854918957,0,0.107541054,0.0570352264,0.27078706,0.113923512,0.121778786,0.116031624,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0.0305870231,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.216797456,0,0,0,0,0.174732283,0.171127319,0.406755924,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0950754061,0.116062254,0,0.136168897,0,0,0.121492684,0.106904231,0.0342719108,0.237712041,0.152312592,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.132810161,0.130316347,0,0.101319484,0,0,0.0510874577,0.121031068,0,0.18098101,0.282454431,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0732114613,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.51252079,0,0.0506167188,0.0447288752,0.146566018,0,0.125801608,0.0465545282,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.556104302,0.0209075641,0,0,0,0.19040437,0.232583746,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0.061263103,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0670865849,0.0255387928,0.679286838,0,0,0,0.166824728,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.668289959,0,0.0140176015,0.170034826,0,0,0.0284692626,0.119188361,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0475066826,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.133316949,0,0,0,0.466190666,0.144824103,0,0,0,0,0,0.208161652,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.237022266,0.762977779,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0.0218002759,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0993766785,0.138621435,0.0994299,0.103010781,0.0237294566,0,0.40486455,0.109166943,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0.0954654366,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.15443328,0,0,0,0,0,0,0.0369724147,0.170235649,0,0,0,0.0261112899,0.109751076,0.40703091,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.715527534,0.112437092,0.172035411,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0241573881,0,0,0,0,0.089109078,0.56707263,0.319660962,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0899620429,0,0,0,0,0.121277958,0.28434217,0.504417777,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.109711304,9.43211671e-06,0,0,0,0,0,0.245705128,0.0780823305,0.105583578,0.0446941741,0.0949446484,0.100036919,0.114253961,0.0114724338,0.0955060497,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.490800619,0,0.12814863,0,0,0,0,0.266749114,0.114301629],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.133692801,0,0.311118543,0.0883763134,0.121900082,0.0899473354,0.130141124,0.124823853,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.172603548,0,0.66106,0,0,0,0.166336417,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.144300967,0,0.36829102,0.230643928,0,0,0.129301891,0.127462193],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.137125358,0,0,0,0.21092236,0.336799711,0.109246187,0,0.157412648,0.0484937057],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.132822067,0.0885682628,0.119481824,0,0,0.0926504955,0.285636157,0.00977804,0.150076404,0.120986812],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.216003612,0.135467723,0,0,0,0.538380504,0,0.110148229],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.107885987,0.150850326,0.0907105207,0,0.0974455774,0.129722804,0.145830244,0,0.277554572,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0992423445,0,0.0594548397,0.147482321,0,0.180498883,0.0871842131,0,0.426137418]]

    adj_mat_expert = np.asarray(adj_mat_expert)
    adj_mat_1_learned = np.asarray(adj_mat_1_learned)
    adj_mat_2_learned = np.asarray(adj_mat_2_learned)
    adj_mat_3_learned = np.asarray(adj_mat_3_learned)

    learned = np.zeros((61,61,3))
    expert = np.zeros((61,61,3))
    learned[:,:,0] = adj_mat_1_learned
    learned[:,:,1] = adj_mat_2_learned
    learned[:,:,2] = adj_mat_3_learned
    expert[:, :, 0] = adj_mat_expert_gcn_processed
    expert[:, :, 1] = adj_mat_expert_gcn_processed
    expert[:, :, 2] = adj_mat_expert_gcn_processed

    diff = learned - expert

    for i in range(adj_mat_1_learned.shape[0]):
        incomming = np.argwhere(adj_mat_3_learned[i,:] >0)
        outgoinig = np.argwhere(adj_mat_3_learned[:,i] >0)
        print()
        print("Features:", dataset.feature_names_all[i])
        print("incomming", dataset.feature_names_all[incomming])
        print("outgoinig", dataset.feature_names_all[outgoinig])


    plot_heatmap_of_reconstruction_error2(learned, expert, diff, 2)

