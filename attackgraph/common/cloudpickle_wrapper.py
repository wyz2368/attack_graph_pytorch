""" Wrapper that does object serialization through cloudpickle. """

class CloudpickleWrapper(object):
    def __init__(self, fn):
        self.fn = fn

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.fn)

    def __setstate__(self, ob):
        import pickle
        self.fn = pickle.loads(ob)

    def __call__(self):
        return self.fn
