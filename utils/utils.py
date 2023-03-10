class DictLikeObject(dict):
    """
    A dict that allows for object-like property access syntax.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, item, value):
        self[item] = value

    def __str__(self):
        return super(DictLikeObject, self).__str__()

    def __repr__(self):
        return super(DictLikeObject, self).__repr__()