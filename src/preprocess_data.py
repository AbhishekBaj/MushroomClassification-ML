import pandas as pd

def preprocess_data():
    df = pd.read_csv("../data/agaricus-lepiota.data", delimiter=',', header=None)
    df.columns =['edible', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing',
                 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
                   'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type', 
                   'spore-print-color', 'population', 'habitat']
    
    # move edible column to the end
    df = df[['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing',
                 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
                   'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type', 
                   'spore-print-color', 'population', 'habitat', 'edible']] 

    df['edible'] = df['edible'].replace(['e'], 0) # edible
    df['edible'] = df['edible'].replace(['p'], 1) # poisonous

    df['cap-shape'] = df['cap-shape'].replace(['b'], 0) # bell
    df['cap-shape'] = df['cap-shape'].replace(['c'], 1) # conical
    df['cap-shape'] = df['cap-shape'].replace(['x'], 2) # convex
    df['cap-shape'] = df['cap-shape'].replace(['f'], 3) # flat
    df['cap-shape'] = df['cap-shape'].replace(['k'], 4) # knobbed
    df['cap-shape'] = df['cap-shape'].replace(['s'], 5) # sunken

    df['cap-surface'] = df['cap-surface'].replace(['f'], 0) # fibrous
    df['cap-surface'] = df['cap-surface'].replace(['g'], 1) # grooves
    df['cap-surface'] = df['cap-surface'].replace(['y'], 2) # scaly
    df['cap-surface'] = df['cap-surface'].replace(['s'], 3) # smooth

    df['cap-color'] = df['cap-color'].replace(['n'], 0) # brown
    df['cap-color'] = df['cap-color'].replace(['b'], 1) # buff
    df['cap-color'] = df['cap-color'].replace(['c'], 2) # cinnamon
    df['cap-color'] = df['cap-color'].replace(['g'], 3) # gray
    df['cap-color'] = df['cap-color'].replace(['r'], 4) # green
    df['cap-color'] = df['cap-color'].replace(['p'], 5) # pink
    df['cap-color'] = df['cap-color'].replace(['u'], 6) # purple
    df['cap-color'] = df['cap-color'].replace(['e'], 7) # red
    df['cap-color'] = df['cap-color'].replace(['w'], 8) # white
    df['cap-color'] = df['cap-color'].replace(['y'], 9) # yellow

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

    df['gill-attachment'] = df['gill-attachment'].replace(['a'], 0) # attached
    df['gill-attachment'] = df['gill-attachment'].replace(['d'], 1) # descending
    df['gill-attachment'] = df['gill-attachment'].replace(['f'], 2) # free
    df['gill-attachment'] = df['gill-attachment'].replace(['n'], 3) # notched

    df['gill-spacing'] = df['gill-spacing'].replace(['c'], 0) # close
    df['gill-spacing'] = df['gill-spacing'].replace(['w'], 1) # crowded
    df['gill-spacing'] = df['gill-spacing'].replace(['d'], 2) # distant

    df['gill-size'] = df['gill-size'].replace(['b'], 0) # broad
    df['gill-size'] = df['gill-size'].replace(['n'], 1) # narrow

    df['gill-color'] = df['gill-color'].replace(['k'], 0) # black
    df['gill-color'] = df['gill-color'].replace(['n'], 1) # brown
    df['gill-color'] = df['gill-color'].replace(['b'], 2) # buff
    df['gill-color'] = df['gill-color'].replace(['h'], 3) # chocolate
    df['gill-color'] = df['gill-color'].replace(['g'], 4) # gray
    df['gill-color'] = df['gill-color'].replace(['r'], 5) # green
    df['gill-color'] = df['gill-color'].replace(['o'], 6) # orange
    df['gill-color'] = df['gill-color'].replace(['p'], 7) # pink
    df['gill-color'] = df['gill-color'].replace(['u'], 8) # purple
    df['gill-color'] = df['gill-color'].replace(['e'], 9) # red
    df['gill-color'] = df['gill-color'].replace(['w'], 10) # white
    df['gill-color'] = df['gill-color'].replace(['y'], 11) # yellow

    df['stalk-shape'] = df['stalk-shape'].replace(['e'], 0) # enlarging
    df['stalk-shape'] = df['stalk-shape'].replace(['t'], 1) # tapering

    df['stalk-root'] = df['stalk-root'].replace(['b'], 0) # bulbous
    df['stalk-root'] = df['stalk-root'].replace(['c'], 1) # club
    df['stalk-root'] = df['stalk-root'].replace(['u'], 2) # cup
    df['stalk-root'] = df['stalk-root'].replace(['e'], 3) # equal
    df['stalk-root'] = df['stalk-root'].replace(['z'], 4) # rhizomorphs
    df['stalk-root'] = df['stalk-root'].replace(['r'], 5) # rooted
    df['stalk-root'] = df['stalk-root'].replace(['?'], 6) # missing

    df['stalk-surface-above-ring'] = df['stalk-surface-above-ring'].replace(['f'], 0) # fibrous
    df['stalk-surface-above-ring'] = df['stalk-surface-above-ring'].replace(['y'], 1) # scaly
    df['stalk-surface-above-ring'] = df['stalk-surface-above-ring'].replace(['k'], 2) # silky
    df['stalk-surface-above-ring'] = df['stalk-surface-above-ring'].replace(['s'], 3) # smooth

    df['stalk-surface-below-ring'] = df['stalk-surface-below-ring'].replace(['f'], 0) # fibrous
    df['stalk-surface-below-ring'] = df['stalk-surface-below-ring'].replace(['y'], 1) # scaly
    df['stalk-surface-below-ring'] = df['stalk-surface-below-ring'].replace(['k'], 2) # silky
    df['stalk-surface-below-ring'] = df['stalk-surface-below-ring'].replace(['s'], 3) # smooth

    df['stalk-color-above-ring'] = df['stalk-color-above-ring'].replace(['n'], 0) # brown
    df['stalk-color-above-ring'] = df['stalk-color-above-ring'].replace(['b'], 1) # buff
    df['stalk-color-above-ring'] = df['stalk-color-above-ring'].replace(['c'], 2) # cinnamon
    df['stalk-color-above-ring'] = df['stalk-color-above-ring'].replace(['g'], 3) # gray
    df['stalk-color-above-ring'] = df['stalk-color-above-ring'].replace(['o'], 4) # orange
    df['stalk-color-above-ring'] = df['stalk-color-above-ring'].replace(['p'], 5) # pink
    df['stalk-color-above-ring'] = df['stalk-color-above-ring'].replace(['e'], 6) # red
    df['stalk-color-above-ring'] = df['stalk-color-above-ring'].replace(['w'], 7) # white
    df['stalk-color-above-ring'] = df['stalk-color-above-ring'].replace(['y'], 8) # yellow

    df['stalk-color-below-ring'] = df['stalk-color-below-ring'].replace(['n'], 0) # brown
    df['stalk-color-below-ring'] = df['stalk-color-below-ring'].replace(['b'], 1) # buff
    df['stalk-color-below-ring'] = df['stalk-color-below-ring'].replace(['c'], 2) # cinnamon
    df['stalk-color-below-ring'] = df['stalk-color-below-ring'].replace(['g'], 3) # gray
    df['stalk-color-below-ring'] = df['stalk-color-below-ring'].replace(['o'], 4) # orange
    df['stalk-color-below-ring'] = df['stalk-color-below-ring'].replace(['p'], 5) # pink
    df['stalk-color-below-ring'] = df['stalk-color-below-ring'].replace(['e'], 6) # red
    df['stalk-color-below-ring'] = df['stalk-color-below-ring'].replace(['w'], 7) # white
    df['stalk-color-below-ring'] = df['stalk-color-below-ring'].replace(['y'], 8) # yellow

    df['veil-type'] = df['veil-type'].replace(['p'], 0) # partial
    df['veil-type'] = df['veil-type'].replace(['u'], 1) # universal

    df['veil-color'] = df['veil-color'].replace(['n'], 0) # brown
    df['veil-color'] = df['veil-color'].replace(['o'], 1) # orange
    df['veil-color'] = df['veil-color'].replace(['w'], 2) # white
    df['veil-color'] = df['veil-color'].replace(['y'], 3) # yellow

    df['ring-number'] = df['ring-number'].replace(['n'], 0) # none
    df['ring-number'] = df['ring-number'].replace(['o'], 1) # one
    df['ring-number'] = df['ring-number'].replace(['t'], 2) # two

    df['ring-type'] = df['ring-type'].replace(['c'], 0) # cobwebby
    df['ring-type'] = df['ring-type'].replace(['e'], 1) # evanescent
    df['ring-type'] = df['ring-type'].replace(['f'], 2) # flaring
    df['ring-type'] = df['ring-type'].replace(['l'], 3) # large
    df['ring-type'] = df['ring-type'].replace(['n'], 4) # none
    df['ring-type'] = df['ring-type'].replace(['p'], 5) # pendant
    df['ring-type'] = df['ring-type'].replace(['s'], 6) # sheathing
    df['ring-type'] = df['ring-type'].replace(['z'], 7) # zone

    df['spore-print-color'] = df['spore-print-color'].replace(['k'], 0) # black
    df['spore-print-color'] = df['spore-print-color'].replace(['n'], 1) # brown
    df['spore-print-color'] = df['spore-print-color'].replace(['b'], 2) # buff
    df['spore-print-color'] = df['spore-print-color'].replace(['h'], 3) # chocolate
    df['spore-print-color'] = df['spore-print-color'].replace(['r'], 4) # green
    df['spore-print-color'] = df['spore-print-color'].replace(['o'], 5) # orange
    df['spore-print-color'] = df['spore-print-color'].replace(['u'], 6) # purple
    df['spore-print-color'] = df['spore-print-color'].replace(['w'], 7) # white
    df['spore-print-color'] = df['spore-print-color'].replace(['y'], 8) # yellow

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

def preprocess_tree():
  column_names = ['edible', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing',
      'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
      'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type', 
      'spore-print-color', 'population', 'habitat']

  X = pd.read_csv('../data/agaricus-lepiota.data', delimiter=',', header=None, names=column_names)
  return X


