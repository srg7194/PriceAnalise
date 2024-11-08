import json
import yaml
from yaml import SafeLoader
import os
import glob
import shutil
import pandas as pd
from functools import wraps
from tqdm import tqdm
import time
from colorama import Fore, Style, init


def delay(seconds=0.5):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    time.sleep(seconds)
    return decorator


def get_progress(obj, debug=False, info=False, warning=False, error=False, critical=False):
    color = "\033[0m"
    reset_color = "\033[0m"
    if debug:
        color = Fore.WHITE
    elif info:
        color = Fore.GREEN
    elif warning:
        color = Fore.YELLOW
    elif error:
        color = Fore.RED
    elif critical:
        color = Fore.LIGHTRED_EX

    desc = "Processing connections"
    bar_format=f"{color}{{l_bar}}{{bar}}{{r_bar}}{reset_color}"

    if type(obj) == pd.DataFrame:
        obj = tqdm(obj.iterrows(), total=obj.shape[0], desc=desc, bar_format=bar_format)
    else:
        obj = tqdm(obj, desc=desc, bar_format=bar_format)
    return obj


def df_print(df, count_rows=False):
    if count_rows:
        if count_rows < 0:
            count_rows = abs(count_rows)
            print(df.tail(count_rows).to_markdown(tablefmt="fancy_grid"))
        else:
            print(df.head(count_rows).to_markdown(tablefmt="fancy_grid"))
    else:
        print(df.to_markdown(tablefmt="fancy_grid"))


def file_exists(file_path):
    """
    Проверяет, существует ли файл.

    Parameters:
    - file_path (str): Путь к файлу.

    Returns:
    - bool: True, если файл существует, False в противном случае.
    """
    return os.path.exists(file_path)


def get_files_list(dir_path, file_pattern=False):
    if os.path.exists(dir_path):
        if file_pattern:
            file_pattern = f'{dir_path}/{file_pattern}'
            return glob.glob(file_pattern)
        return os.listdir(dir_path)
    return list()


def create_file(file_path):
    """
    Создает файл.

    Parameters:
    - file_path (str): Путь к файлу.

    Returns:
    - bool: True, если файл успешно создан, False в противном случае.
    """
    try:
        with open(file_path, 'w'):
            pass
        return True
    except Exception as e:
        print(f"Ошибка при создании файла: {e}")
        return False


def create_directory(directory_path):
    """
    Создает директорию, если она не существует.

    Parameters:
    - directory_path (str): Путь к директории.

    Returns:
    - bool: True, если директория успешно создана или уже существует, False в противном случае.
    """
    try:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        return True
    except Exception as e:
        print(f"Ошибка при создании директории: {e}")
        return False


def delete_file(path, is_folder=False):
    """
    Удаляет файл.

    Parameters:
    - file_path (str): Путь к файлу.

    Returns:
    - bool: True, если файл успешно удален, False в противном случае.
    """
    if not is_folder:
        if os.path.exists(path):
            os.remove(path)
    else:
        if os.path.exists(path):
            shutil.rmtree(path)


def get_file_extension(file_path):
    """
    Получает расширение файла.

    Parameters:
    - file_path (str): Путь к файлу.

    Returns:
    - str: Расширение файла.
    """
    return os.path.splitext(file_path)[1]


def get_file_size(file_path):
    """
    Получает размер файла в байтах.

    Parameters:
    - file_path (str): Путь к файлу.

    Returns:
    - int: Размер файла в байтах.
    """
    return os.path.getsize(file_path)


def dict_to_yaml(data, path):
    """
    Метод для преобразования словаря в YAML и записи данных в файл.

    Parameters:
    - data (dict): Словарь данных.
    - path (str): Путь к файлу YAML.
    """
    with open(path, 'w', encoding='utf-8') as o_file:
        yaml.dump(data, o_file, default_flow_style=False, allow_unicode=True)


def yaml_to_dict(path):
    """
    Метод для чтения данных из файла YAML и преобразования их в словарь.

    Parameters:
    - path (str): Путь к файлу YAML.

    Returns:
    - dict: Словарь данных, прочитанных из файла.
    """
    with open(path, encoding='utf-8') as o_file:
        res = dict(yaml.load(o_file, Loader=SafeLoader))
    return res


def read_excel_file(file_path, sheet_names=None, data_to_dict=False):
    """
    Метод для чтения данных из файла Excel.

    Parameters:
    - file_path (str): Путь к файлу Excel.
    - sheet_names (list): Список имен листов для чтения.

    Returns:
    - dict: Словарь данных, прочитанных из файла Excel.
    """
    with pd.ExcelFile(file_path) as xls:
        if sheet_names is None:
            data = {sheet_name: xls.parse(sheet_name) for sheet_name in xls.sheet_names}
        else:
            data = {sheet_name: xls.parse(sheet_name) for sheet_name in sheet_names}

    if data_to_dict:
        reformat_data = {}
        for k, v in data.items():
            reformat_data.update({k: v.to_dict('records')})
        data = reformat_data
    return data


def write_excel_file(data_dict, file_path):
    """
    Метод для записи данных в файл Excel.

    Parameters:
    - data_dict (dict): Словарь данных для записи.
    - file_path (str): Путь к файлу Excel.
    """
    with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
        for sheet_name, df in data_dict.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)


def json_to_dict(file_path):
    """
    Метод для чтения данных из файла JSON и преобразования их в словарь.

    Parameters:
    - file_path (str): Путь к файлу JSON.

    Returns:
    - dict: Словарь данных, прочитанных из файла JSON.
    """
    with open(file_path, 'r', encoding='utf-8') as json_file:
        data_dict = json.load(json_file)
    return data_dict


def dict_to_json(data_dict, file_path):
    """
    Метод для преобразования словаря в JSON и записи данных в файл.

    Parameters:
    - data_dict (dict): Словарь данных.
    - file_path (str): Путь к файлу JSON.
    """
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data_dict, json_file, indent=2, ensure_ascii=False)


def read_binary_file(file_path):
    """
    Метод для чтения бинарных данных из файла.

    Parameters:
    - file_path (str): Путь к бинарному файлу.

    Returns:
    - bytes: Бинарные данные из файла.
    """
    with open(file_path, 'rb') as binary_file:
        content = binary_file.read()
        return content


def write_binary_file(data, file_path):
    """
    Метод для записи бинарных данных в файл.

    Parameters:
    - data (bytes): Бинарные данные.
    - file_path (str): Путь к бинарному файлу.
    """
    with open(file_path, 'wb') as binary_file:
        binary_file.write(data)
