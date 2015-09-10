#Flattens an arbitrarily nested list
def flatten(l):
    return [item for sublist in l for item in sublist]
