import os

classes = []
for _file in os.listdir(os.path.dirname(__file__)):
    if _file != '__init__.py' and _file[-3:] == '.py':
        file_name = module[:-3]
        classes.append(file_name)

__all__ = classes