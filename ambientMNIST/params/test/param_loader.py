import pickle

# load a.param
with open('a.param', 'rb') as f:
    a = pickle.load(f)
    print(type(a))
    print(a.shape)
    print(a)