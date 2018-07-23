from multiprocessing.reduction import ForkingPickler, AbstractReducer


class ForkingPicklerProtocol2(ForkingPickler):
    """
    Substitute class to get correct protocol version on Python 3 multiprocessing lib
    https://stackoverflow.com/questions/45119053/how-to-change-the-serialization-method-used-by-pythons-multiprocessing
    """
    def __init__(self, *args):
        if len(args) > 1:
            args[1] = 2
        else:
            args.append(2)
        super(ForkingPicklerProtocol2, self).__init__(*args)

    @classmethod
    def dumps(cls, obj, protocol=2):
        return ForkingPickler.dumps(obj, protocol)


def dump(obj, file, protocol=2):
    ForkingPicklerProtocol2(file, protocol).dump(obj)


class Pickle2Reducer(AbstractReducer):
    ForkingPicker = ForkingPicklerProtocol2
    register = ForkingPicklerProtocol2.register
    dump = dump
