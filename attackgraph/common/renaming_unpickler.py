""" Class that unpickles objects that have had their package-path modified.

Resources:
 - https://stackoverflow.com/questions/40914066/unpickling-objects-after-renaming-a-module
 - https://stackoverflow.com/questions/2121874/python-pickling-after-changing-a-modules-directory
"""
import pickle


class RenamingUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        """ Find the class in a module.

        :param module:
        :param name:
        :return:
        :rtype: str.
        """
        print(module)
        print(module[:23])
        # `algorithm` module was renamed to `rl`.
        if module[:23] == "attackgraph.algorithms.":
            print("Renaming")
            module = f"attackgraph.rl.{module[23:]}"

        return super().find_class(module, name)
