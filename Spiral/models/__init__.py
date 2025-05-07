# initilize all models, support dynamic import
import os
import importlib

modules = os.listdir(os.path.join(os.path.abspath('.'),'models'))
modules = [os.path.splitext(module)[0] for module in modules if '__' not in module]

for module in modules:
    importlib.import_module('.' + module, package=__name__)