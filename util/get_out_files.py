import os
import shutil


def paths(file_path):
    path_collection = []
    path_target_collection = []
    path_target_dir = []
    for dirpath, dirnames, filenames in os.walk(file_path):
        for file_name in filenames:
            if file_name == 'out.json' or file_name == 'args.json' or file_name == 'command.txt':
                fullpath = os.path.join(dirpath, file_name)
                path_collection.append(fullpath)
                path_target_collection.append(fullpath.replace('results', 'results_out'))
                path_target_dir.append(dirpath.replace('results', 'results_out'))
    return path_collection, path_target_collection, path_target_dir


source_path = ''


if __name__ == "__main__":
    path_collection, path_target_collection, path_target_dir = paths(source_path)

    for (source, target, dir_path) in zip(path_collection, path_target_collection, path_target_dir):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        print(source, target)
        shutil.copyfile(source, target)
