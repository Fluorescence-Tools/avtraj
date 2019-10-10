from __future__ import annotations

import json
import yaml


class PythonBase(object):
    """The class PythonBase provides base functionality to load and save keyword
    arguments used to initiate the class as a dict. Moreover, the __setstate__
    and the __getstate__ magic methods are implemented for basic functionality
    to recover and copy states of objects. Parameters passed as keyword arguments
    can be directly accesses as an attribute of the object.

    Examples
    --------
    >>> pb = PythonBase(name='name_a', lo="uu")
    >>> pb
    {'lo': 'uu', 'name': 'name_a'}
    >>> bc = PythonBase(parameter="ala", lol=1)
    >>> bc.lol
    1
    >>> bc.parameter
    ala
    >>> bc.to_dict()
    {'lol': 1, 'parameter': 'ala', 'verbose': False}
    >>> bc.from_dict({'jj': 22, 'zu': "auf"})
    >>> bc.jj
    22
    >>> bc.zu
    auf

    Methods
    -------
    save(self, filename, file_type='json')
        Saves the class either to a JSON file (file_type='json') or a YAML (file_type='yaml')
        depending on the the optional parameter 'file_type'.

    load(self, filename, file_type='json')
        Loads a JSON or YAML file to the class

    to_dict(self)
        Returns a python dictionary representing the state of the object

    to_json(self, indent=4, sort_keys=True)
        Returns a JSON file representing the state of the object.

    to_yaml(self)
        Returns a YAML file representing the state of the object.

    from_yaml
        Loads a YAML file and updates the state of the object

    from_json
        Loads a JSON file and updates the state of the object
    """

    def save(
            self,
            filename: str,
            file_type: str = 'json'
    ):
        if file_type == "yaml":
            txt = self.to_yaml()
        else:
            txt = self.to_json()
        with open(filename, 'w') as fp:
            fp.write(txt)

    def load(
            self,
            filename: str,
            file_type: str = 'json'):
        if file_type == "json":
            self.from_json(filename=filename)
        else:
            self.from_yaml(filename=filename)

    def to_dict(self):
        return self.__getstate__()

    def to_json(self, indent=4, sort_keys=True):
        return json.dumps(self.__getstate__(), indent=indent, sort_keys=sort_keys)

    def to_yaml(self):
        return yaml.dump(self.__getstate__())

    def from_yaml(self, string="", filename=None):
        if filename is not None:
            with open(filename, 'r') as fp:
                j = yaml.load(fp)
        elif string is not None:
            j = json.loads(string)
        else:
            j = dict()
        self.__setstate__(j)

    def from_json(
            self,
            string: str = "",
            filename: str = None
    ):
        """Reads the content of a JSON file into the object via the
        __setstate__ method. If a filename is not specified (None) the
        string is parsed.

        Parameters
        ----------

        string : str
            A string containing the JSON file

        filename: str
            The filename to be opened

        """
        state = dict()
        if filename is not None:
            with open(filename, 'r') as fp:
                state = json.load(fp)
        elif string is not None:
            state = json.loads(string)
        else:
            pass
        self.__setstate__(state)

    def __setstate__(self, state):
        self.kw = state

    def __getstate__(self):
        try:
            state = self.kw.copy()
        except AttributeError:
            state = dict()
        return state

    def __setattr__(self, k, v):
        try:
            kw = self.__dict__['kw']
        except KeyError:
            kw = dict()
            self.__dict__['kw'] = kw
        if k in kw:
            self.__dict__['kw'][k] = v

        # Set the attributes normally
        propobj = getattr(self.__class__, k, None)
        if isinstance(propobj, property):
            if propobj.fset is None:
                raise AttributeError("can't set attribute")
            propobj.fset(self, v)
        else:
            super(PythonBase, self).__setattr__(k, v)

    def __getattr__(self, key):
        return self.kw[key]

    def __repr__(self):
        return str(self.__getstate__())

    def __str__(self):
        s = 'class: %s\n' % self.__class__.__name__
        for key in self.kw.keys():
            s += "%s: \t %s\n" % (key, self.kw[key])
        return s

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__()
        name = kwargs.pop('name', self.__class__.__name__)
        kwargs['name'] = name() if callable(name) else name
        kwargs['verbose'] = kwargs.get('verbose', False)
        self.kw = kwargs