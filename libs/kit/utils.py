import os
import importlib
import tempfile
from contextlib import contextmanager


def get_class(classname):
    """ returns the class object for the given class name

    :param classname: name of the class
    :return: class object
    """

    parts = classname.split(".")
    if len(parts) > 1:
        importlib.import_module(parts[0])
        module = ".".join([part for part in parts[:-1]])
        module = importlib.import_module(module)
        classname = f"module.{parts[-1]}"

    return eval(classname)


@contextmanager
def temp_working_directory(dir_path=None):
    try:
        original_cwd_path = os.getcwd()

        if dir_path is None:
            with tempfile.TemporaryDirectory() as tmp_dir_path:
                os.chdir(tmp_dir_path)
                yield tmp_dir_path
        else:
            os.chdir(dir_path)
            yield dir_path

    finally:
        os.chdir(original_cwd_path)