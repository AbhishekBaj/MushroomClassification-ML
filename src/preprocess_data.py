import pandas as pd

def preprocess_data(df):
    # df = pd.read_csv("../data/agaricus_lepiota.data", delimiter=',', header=None)
    # df.columns =['class', 'cap_shape', 'cap_surface', 'cap_color', 'bruises', 'odor', 'gill_attachment', 'gill_spacing',
    #              'gill_size', 'gill_color', 'stalk_shape', 'stalk_root', 'stalk_surface_above_ring', 'stalk_surface_below_ring',
    #                'stalk_color_above_ring', 'stalk_color_below_ring', 'veil_type', 'veil_color', 'ring_number', 'ring_type', 
    #                'spore_print_color', 'population', 'habitat']
    
    # # move class column to the end
    # df = df[['cap_shape', 'cap_surface', 'cap_color', 'bruises', 'odor', 'gill_attachment', 'gill_spacing',
    #              'gill_size', 'gill_color', 'stalk_shape', 'stalk_root', 'stalk_surface_above_ring', 'stalk_surface_below_ring',
    #                'stalk_color_above_ring', 'stalk_color_below_ring', 'veil_type', 'veil_color', 'ring_number', 'ring_type', 
    #                'spore_print_color', 'population', 'habitat', 'class']] 

    df['class'] = df['class'].replace(['e'], 0) # class
    df['class'] = df['class'].replace(['p'], 1) # poisonous

    df['cap_shape'] = df['cap_shape'].replace(['b'], 0) # bell
    df['cap_shape'] = df['cap_shape'].replace(['c'], 1) # conical
    df['cap_shape'] = df['cap_shape'].replace(['x'], 2) # convex
    df['cap_shape'] = df['cap_shape'].replace(['f'], 3) # flat
    df['cap_shape'] = df['cap_shape'].replace(['k'], 4) # knobbed
    df['cap_shape'] = df['cap_shape'].replace(['s'], 5) # sunken

    df['cap_surface'] = df['cap_surface'].replace(['f'], 0) # fibrous
    df['cap_surface'] = df['cap_surface'].replace(['g'], 1) # grooves
    df['cap_surface'] = df['cap_surface'].replace(['y'], 2) # scaly
    df['cap_surface'] = df['cap_surface'].replace(['s'], 3) # smooth

    df['cap_color'] = df['cap_color'].replace(['n'], 0) # brown
    df['cap_color'] = df['cap_color'].replace(['b'], 1) # buff
    df['cap_color'] = df['cap_color'].replace(['c'], 2) # cinnamon
    df['cap_color'] = df['cap_color'].replace(['g'], 3) # gray
    df['cap_color'] = df['cap_color'].replace(['r'], 4) # green
    df['cap_color'] = df['cap_color'].replace(['p'], 5) # pink
    df['cap_color'] = df['cap_color'].replace(['u'], 6) # purple
    df['cap_color'] = df['cap_color'].replace(['e'], 7) # red
    df['cap_color'] = df['cap_color'].replace(['w'], 8) # white
    df['cap_color'] = df['cap_color'].replace(['y'], 9) # yellow

    df['bruises'] = df['bruises'].replace(['t'], 0) # bruises
    df['bruises'] = df['bruises'].replace(['f'], 1) # no

    df['odor'] = df['odor'].replace(['a'], 0) # almond
    df['odor'] = df['odor'].replace(['l'], 1) # anise
    df['odor'] = df['odor'].replace(['c'], 2) # creosote
    df['odor'] = df['odor'].replace(['y'], 3) # fishy
    df['odor'] = df['odor'].replace(['f'], 4) # foul
    df['odor'] = df['odor'].replace(['m'], 5) # musty
    df['odor'] = df['odor'].replace(['n'], 6) # none
    df['odor'] = df['odor'].replace(['p'], 7) # pungent
    df['odor'] = df['odor'].replace(['s'], 8) # spicy

    df['gill_attachment'] = df['gill_attachment'].replace(['a'], 0) # attached
    df['gill_attachment'] = df['gill_attachment'].replace(['d'], 1) # descending
    df['gill_attachment'] = df['gill_attachment'].replace(['f'], 2) # free
    df['gill_attachment'] = df['gill_attachment'].replace(['n'], 3) # notched

    df['gill_spacing'] = df['gill_spacing'].replace(['c'], 0) # close
    df['gill_spacing'] = df['gill_spacing'].replace(['w'], 1) # crowded
    df['gill_spacing'] = df['gill_spacing'].replace(['d'], 2) # distant

    df['gill_size'] = df['gill_size'].replace(['b'], 0) # broad
    df['gill_size'] = df['gill_size'].replace(['n'], 1) # narrow

    df['gill_color'] = df['gill_color'].replace(['k'], 0) # black
    df['gill_color'] = df['gill_color'].replace(['n'], 1) # brown
    df['gill_color'] = df['gill_color'].replace(['b'], 2) # buff
    df['gill_color'] = df['gill_color'].replace(['h'], 3) # chocolate
    df['gill_color'] = df['gill_color'].replace(['g'], 4) # gray
    df['gill_color'] = df['gill_color'].replace(['r'], 5) # green
    df['gill_color'] = df['gill_color'].replace(['o'], 6) # orange
    df['gill_color'] = df['gill_color'].replace(['p'], 7) # pink
    df['gill_color'] = df['gill_color'].replace(['u'], 8) # purple
    df['gill_color'] = df['gill_color'].replace(['e'], 9) # red
    df['gill_color'] = df['gill_color'].replace(['w'], 10) # white
    df['gill_color'] = df['gill_color'].replace(['y'], 11) # yellow

    df['stalk_shape'] = df['stalk_shape'].replace(['e'], 0) # enlarging
    df['stalk_shape'] = df['stalk_shape'].replace(['t'], 1) # tapering

    df['stalk_root'] = df['stalk_root'].replace(['b'], 0) # bulbous
    df['stalk_root'] = df['stalk_root'].replace(['c'], 1) # club
    df['stalk_root'] = df['stalk_root'].replace(['u'], 2) # cup
    df['stalk_root'] = df['stalk_root'].replace(['e'], 3) # equal
    df['stalk_root'] = df['stalk_root'].replace(['z'], 4) # rhizomorphs
    df['stalk_root'] = df['stalk_root'].replace(['r'], 5) # rooted
    df['stalk_root'] = df['stalk_root'].replace(['?'], 6) # missing

    df['stalk_surface_above_ring'] = df['stalk_surface_above_ring'].replace(['f'], 0) # fibrous
    df['stalk_surface_above_ring'] = df['stalk_surface_above_ring'].replace(['y'], 1) # scaly
    df['stalk_surface_above_ring'] = df['stalk_surface_above_ring'].replace(['k'], 2) # silky
    df['stalk_surface_above_ring'] = df['stalk_surface_above_ring'].replace(['s'], 3) # smooth

    df['stalk_surface_below_ring'] = df['stalk_surface_below_ring'].replace(['f'], 0) # fibrous
    df['stalk_surface_below_ring'] = df['stalk_surface_below_ring'].replace(['y'], 1) # scaly
    df['stalk_surface_below_ring'] = df['stalk_surface_below_ring'].replace(['k'], 2) # silky
    df['stalk_surface_below_ring'] = df['stalk_surface_below_ring'].replace(['s'], 3) # smooth

    df['stalk_color_above_ring'] = df['stalk_color_above_ring'].replace(['n'], 0) # brown
    df['stalk_color_above_ring'] = df['stalk_color_above_ring'].replace(['b'], 1) # buff
    df['stalk_color_above_ring'] = df['stalk_color_above_ring'].replace(['c'], 2) # cinnamon
    df['stalk_color_above_ring'] = df['stalk_color_above_ring'].replace(['g'], 3) # gray
    df['stalk_color_above_ring'] = df['stalk_color_above_ring'].replace(['o'], 4) # orange
    df['stalk_color_above_ring'] = df['stalk_color_above_ring'].replace(['p'], 5) # pink
    df['stalk_color_above_ring'] = df['stalk_color_above_ring'].replace(['e'], 6) # red
    df['stalk_color_above_ring'] = df['stalk_color_above_ring'].replace(['w'], 7) # white
    df['stalk_color_above_ring'] = df['stalk_color_above_ring'].replace(['y'], 8) # yellow

    df['stalk_color_below_ring'] = df['stalk_color_below_ring'].replace(['n'], 0) # brown
    df['stalk_color_below_ring'] = df['stalk_color_below_ring'].replace(['b'], 1) # buff
    df['stalk_color_below_ring'] = df['stalk_color_below_ring'].replace(['c'], 2) # cinnamon
    df['stalk_color_below_ring'] = df['stalk_color_below_ring'].replace(['g'], 3) # gray
    df['stalk_color_below_ring'] = df['stalk_color_below_ring'].replace(['o'], 4) # orange
    df['stalk_color_below_ring'] = df['stalk_color_below_ring'].replace(['p'], 5) # pink
    df['stalk_color_below_ring'] = df['stalk_color_below_ring'].replace(['e'], 6) # red
    df['stalk_color_below_ring'] = df['stalk_color_below_ring'].replace(['w'], 7) # white
    df['stalk_color_below_ring'] = df['stalk_color_below_ring'].replace(['y'], 8) # yellow

    df['veil_type'] = df['veil_type'].replace(['p'], 0) # partial
    df['veil_type'] = df['veil_type'].replace(['u'], 1) # universal

    df['veil_color'] = df['veil_color'].replace(['n'], 0) # brown
    df['veil_color'] = df['veil_color'].replace(['o'], 1) # orange
    df['veil_color'] = df['veil_color'].replace(['w'], 2) # white
    df['veil_color'] = df['veil_color'].replace(['y'], 3) # yellow

    df['ring_number'] = df['ring_number'].replace(['n'], 0) # none
    df['ring_number'] = df['ring_number'].replace(['o'], 1) # one
    df['ring_number'] = df['ring_number'].replace(['t'], 2) # two

    df['ring_type'] = df['ring_type'].replace(['c'], 0) # cobwebby
    df['ring_type'] = df['ring_type'].replace(['e'], 1) # evanescent
    df['ring_type'] = df['ring_type'].replace(['f'], 2) # flaring
    df['ring_type'] = df['ring_type'].replace(['l'], 3) # large
    df['ring_type'] = df['ring_type'].replace(['n'], 4) # none
    df['ring_type'] = df['ring_type'].replace(['p'], 5) # pendant
    df['ring_type'] = df['ring_type'].replace(['s'], 6) # sheathing
    df['ring_type'] = df['ring_type'].replace(['z'], 7) # zone

    df['spore_print_color'] = df['spore_print_color'].replace(['k'], 0) # black
    df['spore_print_color'] = df['spore_print_color'].replace(['n'], 1) # brown
    df['spore_print_color'] = df['spore_print_color'].replace(['b'], 2) # buff
    df['spore_print_color'] = df['spore_print_color'].replace(['h'], 3) # chocolate
    df['spore_print_color'] = df['spore_print_color'].replace(['r'], 4) # green
    df['spore_print_color'] = df['spore_print_color'].replace(['o'], 5) # orange
    df['spore_print_color'] = df['spore_print_color'].replace(['u'], 6) # purple
    df['spore_print_color'] = df['spore_print_color'].replace(['w'], 7) # white
    df['spore_print_color'] = df['spore_print_color'].replace(['y'], 8) # yellow

    df['population'] = df['population'].replace(['a'], 0) # abundant
    df['population'] = df['population'].replace(['c'], 1) # clustered
    df['population'] = df['population'].replace(['n'], 2) # numerous
    df['population'] = df['population'].replace(['s'], 3) # scattered
    df['population'] = df['population'].replace(['v'], 4) # several
    df['population'] = df['population'].replace(['y'], 5) # solitary

    df['habitat'] = df['habitat'].replace(['g'], 0) # grasses
    df['habitat'] = df['habitat'].replace(['l'], 0) # leaves
    df['habitat'] = df['habitat'].replace(['m'], 0) # meadows
    df['habitat'] = df['habitat'].replace(['p'], 0) # paths
    df['habitat'] = df['habitat'].replace(['u'], 0) # urban
    df['habitat'] = df['habitat'].replace(['w'], 0) # waste
    df['habitat'] = df['habitat'].replace(['d'], 0) # woods

    return df