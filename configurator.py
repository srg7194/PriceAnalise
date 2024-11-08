import time

from addition.api import *
from addition.other import *
from tqdm import tqdm

from addition.log import logging, CustomLogger, log_decorator
logger = CustomLogger(level=logging.DEBUG, name='CONFIGURATOR')
logger_loader = CustomLogger(level=logging.DEBUG, name='CONFIGURATOR LOADER')

class Config:
    def __init__(self, creds_path, local_start):
        logger.log_debug('Создание конфигурации...')
        self.creds_path = creds_path
        self.save_path_xslx = 'files/config/settings.xlsx'
        self.save_path_json = 'files/config/config.json'
        self.creds = json_to_dict(self.creds_path)

        if not local_start:
            logger.log_debug('Скачивание файла настроек')
            self.save_config()
        self.default_config = read_excel_file(self.save_path_xslx)

    def save_config(self):
        service_account_path = self.creds['config']['path']
        file_link = self.creds['config']['link']

        init = GoogleDriveAPI(service_account_path)
        fileid = init.get_file_id(file_link)
        init.download_file(real_file_id=fileid, file_path=self.save_path_xslx)
        return self.save_path_xslx

    def get_config(self):
        config = {
            'credential': self.creds,
            'loader': Loader(self.default_config).get_config()
        }
        logger.log_debug(f'Сохранение файла конфигурации - {self.save_path_json}')
        dict_to_json(config, self.save_path_json)
        logger.log_debug('Создание конфигурации завершено')
        return config


class Loader:
    def __init__(self, default_config):
        self.default_config = default_config

    def filter_enabled(self):
        connection = self.default_config['connection']
        enable_connection = connection.loc[connection['enable'] == True]
        return enable_connection

    def update_data(self):
        timeframe = self.default_config['timeframe']
        timeframe['junior'] = timeframe['junior'].where(~timeframe['junior'].isna(), False)
        return timeframe

    @staticmethod
    @delay()
    def get_data_loader(enable_connection, timeframe):
        config = []
        for i, s in get_progress(obj=enable_connection, debug=True, comment='Процесс выполнения'):
            temp = dict(s)
            for el in ['enable', 'id']:
                temp.pop(el)
            enable_timeframe = (timeframe.loc[timeframe['Id_connection'] == s['id']])
            enable_timeframe.pop('Id_connection')
            enable_timeframe = enable_timeframe.to_dict('records')
            temp.update({'timeframe': enable_timeframe})
            config.append(temp)
        return config

    def get_config(self):
        logger_loader.log_debug('Фильтрация активных настроек')
        enable_connection = self.filter_enabled()
        logger_loader.log_debug('Модификация значений')
        timeframe = self.update_data()
        logger_loader.log_debug('Создание файла конфигурации')
        config = self.get_data_loader(enable_connection, timeframe)
        return config

