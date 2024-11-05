import pandas as pd
import sqlite3
import pymysql
import sqlalchemy
from sqlalchemy import Table, Column, Integer, String
from sqlalchemy import create_engine, exc, text, orm, dialects
from sqlalchemy.orm import sessionmaker
# import clickhouse_sqlalchemy
# from clickhouse_driver import Client
import clickhouse_connect

def convert_data_types(source, target, type='', src_type=''):
    # Пример сопоставления типов данных между различными базами данных и Pandas DataFrame
    type_mapping = pd.DataFrame({
        'pandas': {
            'TEXT': 'object',
            'INTEGER': 'int64',
            'REAL': 'float64',
            'BOOL': 'bool',
            'DATETIME': 'datetime64[ns]',
            'DATE': 'datetime64[ns]',  # Добавлен тип данных date
            'TIME': 'datetime64[ns]',  # Добавлен тип данных time
            'BLOB': 'object',  # Предположим, что blob представлен как объект
            'FLOAT': 'float64',  # Добавлен тип данных float
        },
        'sqlite': {
            'TEXT': 'TEXT',
            'INTEGER': 'INTEGER',
            'REAL': 'REAL',
            'BOOL': 'INTEGER',  # SQLite использует целые числа для представления булевых значений
            'DATETIME': 'DATETIME',
            'DATE': 'DATE',  # Добавлен тип данных date
            'TIME': 'TIME',  # Добавлен тип данных time
            'BLOB': 'BLOB',
            'FLOAT': 'REAL',  # Добавлен тип данных float
        },
        'mysql': {
            'TEXT': 'TEXT',
            'INTEGER': 'INT',
            'REAL': 'DOUBLE',
            'BOOL': 'TINYINT',  # MySQL использует TINYINT для представления булевых значений
            'DATETIME': 'DATETIME',
            'DATE': 'DATE',  # Добавлен тип данных date
            'TIME': 'TIME',  # Добавлен тип данных time
            'BLOB': 'BLOB',
            'FLOAT': 'DOUBLE',  # Добавлен тип данных float
        },
        'mssql': {
            'TEXT': 'NVARCHAR(MAX)',
            'INTEGER': 'INT',
            'REAL': 'FLOAT',
            'BOOL': 'BIT',  # MSSQL использует BIT для представления булевых значений
            'DATETIME': 'DATETIME',
            'DATE': 'DATE',  # Добавлен тип данных date
            'TIME': 'TIME',  # Добавлен тип данных time
            'BLOB': 'VARBINARY(MAX)',
            'FLOAT': 'FLOAT',  # Добавлен тип данных float
        },
        'postgresql': {
            'TEXT': 'TEXT',
            'INTEGER': 'INTEGER',
            'REAL': 'REAL',
            'BOOL': 'BOOLEAN',  # PostgreSQL имеет отдельный тип BOOLEAN для булевых значений
            'DATETIME': 'TIMESTAMP',
            'DATE': 'DATE',  # Добавлен тип данных date
            'TIME': 'TIME',  # Добавлен тип данных time
            'BLOB': 'BYTEA',
            'FLOAT': 'REAL',  # Добавлен тип данных float
        },
        'clickhouse': {
            'TEXT': 'String',
            'INTEGER': 'Int32',
            'REAL': 'Float32',
            'BOOL': 'Int8',  # ClickHouse использует Int8 для представления булевых значений
            'DATETIME': 'DateTime',
            'DATE': 'Date',  # Добавлен тип данных date
            'TIME': 'DateTime',  # Добавлен тип данных time
            'BLOB': 'String',  # Пример, замените на соответствующий тип в ClickHouse
            'FLOAT': 'Float32',  # Добавлен тип данных float
        },
    })

    # Создаем словарь сопоставления для указанных исходного и целевого источников данных
    result = type_mapping[[source, target]]

    if type != '':
        return result.loc[result.index == type].to_dict('records')[0][target]
    if src_type != '':
        return result.loc[result[source] == src_type].to_dict('records')[0][target]
    return result


class SQLite():
    def __init__(self):
        pass


class MySQL():
    def __init__(self):
        pass


class ClickHouse():
    def __init__(self):
        pass


class DataBase:
    def __init__(self, db_type, db_name, host=False, port=False, username=False, password=False):
        """
        Инициализация объекта базы данных.

        Parameters:
        - db_type (str): Тип базы данных.
        - database (str): Название базы данных.
        - host (str): Хост базы данных.
        - port (int): Порт базы данных.
        - username (str): Имя пользователя базы данных.
        - password (str): Пароль для доступа к базе данных.
        """
        self.db_type = db_type
        self.db_name = db_name
        self.host = host
        self.port = port
        self.username = username
        self.password = password

    ''' Вспомогательные методы '''

    def get_table_columns(self, keys: dict):
        """
        Получение строкового представления колонок таблицы.

        Parameters:
        - keys (dict): Словарь с ключами.

        Returns:
        - str: Строковое представление колонок таблицы.
        """
        return str(keys). \
            replace('"', ""). \
            replace("'", ""). \
            replace(":", ""). \
            replace("{", ""). \
            replace("}", ""). \
            replace('Europe/Moscow', "'Europe/Moscow'"). \
            replace('UTC', "'UTC'")

    def parsing_df_col_types(self, df):
        res = {column_name: convert_data_types('pandas', 'clickhouse', src_type=column_type) for
               column_name, column_type in df.dtypes.items()}
        df = False
        return res

    def __create_engine(self, without_bd=False):
        """
        Внутренний метод для создания движка SQLAlchemy.

        Parameters:
        - without_bd (bool): Флаг для определения, создавать движок с базой данных или без.

        Returns:
        - engine: Объект SQLAlchemy Engine.
        """
        if not without_bd:
            if self.db_type == 'clickhouse':
                return clickhouse_connect.get_client(database=self.db_name, host=self.host, port=self.port,
                                                     user=self.username, password=self.password)
            else:
                if self.db_type == 'sqlite':
                    db_path = os.getcwd()
                    if self.host:
                        db_path = self.host
                    conn_str = f"sqlite:///{db_path}/{self.db_name}.db"
                    return create_engine(conn_str)
                elif self.db_type == 'mysql':
                    conn_str = f"mysql+pymysql://{self.username}:{self.password}@{self.host}:{self.port}/{self.db_name}"
                    return create_engine(conn_str)
                elif self.db_type == 'mssql':
                    conn_str = f"mssql+pyodbc://{self.username}:{self.password}@{self.host}:{self.port}/{self.db_name}?driver=ODBC+Driver+17+for+SQL+Server"
                    return create_engine(conn_str)
                elif self.db_type == 'postgres':
                    conn_str = f"postgresql+psycopg2://{self.username}:{self.password}@{self.host}:{self.port}/{self.db_name}"
                    return create_engine(conn_str)
        else:
            if self.db_type == 'clickhouse':
                return clickhouse_connect.get_client(host=self.host, port=self.port,
                                                     user=self.username, password=self.password)
            else:
                if self.db_type == 'sqlite+pysqlite':
                    conn_str = f"sqlite:///{self.db_name}"
                    return create_engine(conn_str)
                elif self.db_type == 'mysql':
                    conn_str = f"mysql+pymysql://{self.username}:{self.password}@{self.host}:{self.port}"
                    return create_engine(conn_str)
                elif self.db_type == 'mssql':
                    conn_str = f"mssql+pyodbc://{self.username}:{self.password}@{self.host}:{self.port}?driver=ODBC+Driver+17+for+SQL+Server"
                    return create_engine(conn_str)
                elif self.db_type == 'postgres':
                    conn_str = f"postgresql+psycopg2://{self.username}:{self.password}@{self.host}:{self.port}"
                    return create_engine(conn_str)
                else:
                    raise ValueError(f"Unsupported database type: {self.db_type}")

    ''' Методы для управления БД '''

    def check_database_exists(self):
        """
        Проверка существования базы данных.

        Returns:
        - bool: True, если база данных существует, False в противном случае.
        """
        query = f"SHOW DATABASES LIKE '{self.db_name}'"
        return self.execute_query(query, 'exist', without_bd=True)

    def create_database(self):
        """
        Создание базы данных.
        """
        if not self.check_database_exists():
            self.execute_query(f"CREATE DATABASE {self.db_name}", without_bd=True)

    def drop_database(self):
        """
        Удаление базы данных.
        """
        if self.check_database_exists():
            self.execute_query(f"DROP DATABASE {self.db_name}", without_bd=True)

    ''' Методы изменения БД'''

    def rename_database(self, new_name):
        """
        Переименование базы данных.

        Parameters:
        - new_name (str): Новое имя базы данных.
        """
        if self.check_database_exists():
            self.execute_query(f"RENAME DATABASE {self.db_name} TO {new_name}", without_bd=True)

    ''' Методы для управления таблицами '''

    def check_table_exists(self, table_name):
        """
        Проверка существования таблицы.

        Parameters:
        - table_name (str): Имя таблицы.

        Returns:
        - bool: True, если таблица существует, False в противном случае.
        """
        query = f"SHOW TABLES LIKE '{table_name}'"
        if self.db_type == 'sqlite':
            query = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
        return self.execute_query(query, 'exist')

    def create_table(self, table_name, table_columns, engine=False, table_partition=False, order=False):
        """
        Создание таблицы.

        Parameters:
        - table_name (str): Имя таблицы.
        - table_columns (dict): Словарь с колонками таблицы.
        - engine (str): Тип движка таблицы (по умолчанию 'MergeTree()').
        - table_partition (str): Ключ разделения таблицы (по умолчанию 'qwe').
        - order (str): Порядок сортировки таблицы (по умолчанию 'qwe').
        """
        if not self.check_table_exists(table_name):
            first_col = list(table_columns.keys())[0]
            table_columns = self.get_table_columns(table_columns)
            if self.db_type == 'clickhouse':
                if not engine:
                    engine = 'MergeTree()'
                if not order:
                    order = first_col
                # order = get_tech_name(order)

            query = f"""CREATE TABLE 
                            {table_name} (
                                {table_columns}
                             )"""
            if engine:
                query += f' engine {engine}'
            if order:
                query += f' order by {order}'
            if table_partition:
                query += f' partition by {table_partition}'

            self.execute_query(query)

    def drop_table(self, table_name):
        """
        Удаление таблицы.

        Parameters:
        - table_name (str): Имя таблицы.
        """
        if self.check_table_exists(table_name):
            self.execute_query(f"DROP TABLE {table_name}")

    ''' Методы изменения таблиц '''

    def rename_table(self, old_name, new_name):
        """
        Переименование таблицы.

        Parameters:
        - old_name (str): Старое имя таблицы.
        - new_name (str): Новое имя таблицы.
        """
        if self.check_table_exists(old_name):
            self.execute_query(f"""RENAME TABLE {old_name} TO {new_name}""")

    def alter_column_data_type(self, table_name, column_name, new_data_type):
        """
        Изменение типа данных колонки таблицы.

        Parameters:
        - table_name (str): Имя таблицы.
        - column_name (str): Имя колонки.
        - new_data_type (str): Новый тип данных.
        """
        if self.check_table_exists(table_name):
            query = f"ALTER TABLE {table_name} ALTER COLUMN {column_name} TYPE {new_data_type}"
            self.execute_query(query)

    def add_column(self, table_name, column_name, data_type):
        """
        Добавление новой колонки в таблицу.

        Parameters:
        - table_name (str): Имя таблицы.
        - column_name (str): Имя новой колонки.
        - data_type (str): Тип данных новой колонки.
        """
        if self.check_table_exists(table_name):
            query = f"ALTER TABLE {table_name} ADD COLUMN {column_name} {data_type}"
            self.execute_query(query)

    def get_info_table_columns(self, table_name):
        """
        Получение списка колонок и их типов данных из таблицы.

        Parameters:
        - table_name (str): Имя таблицы.

        Returns:
        - result_dict: Словарь, где ключи - имена колонок, значения - их типы данных.
        """
        if self.check_table_exists(table_name):
            query = f"SELECT name, type FROM system.columns WHERE table = '{table_name}'"
            result = self.execute_query(query)
            result_dict = dict(zip(result['name'], result['type']))
        return result_dict

    def get_query_created_table(self, table_name, new_table_name=False):
        """
        Получение запроса создавшего таблицу.

        Parameters:
        - table_name (str): Имя таблицы.

        Returns:
        - query (str): Запрос. Если new_table_name задан, то table_name будет заменен на new_table_name.
        """
        if self.check_table_exists(table_name):
            query = f"SHOW CREATE TABLE {self.db_name}.{table_name}"
            result = self.execute_query(query, 'list')[0]['statement']
            if new_table_name:
                result = result.replace(table_name, new_table_name)
            return result

    ''' Методы изменения данных '''

    def update_table_from_dict(self, table_name, data: dict, filt=False, return_query=False):
        update = []
        for k, v in data.items():
            if isinstance(v, str):
                update.append(f"{k} = '{v}'")
            elif isinstance(v, (int, float)):
                update.append(f"{k} = {v}")
            elif isinstance(v, (pd.Timestamp, datetime.datetime)):
                update.append(f"{k} = '{v}'")
        update = ', '.join(update)

        query = f'''
            ALTER TABLE
                {table_name}
            UPDATE
                {update}'''

        # print(query)

        if filt:
            where = False
            where_parts = []
            if isinstance(filt, dict):
                for k, v in filt.items():
                    if isinstance(v, str):
                        where_parts.append(f"{k} = '{v}'")
                    elif isinstance(v, (int, float, np.int64)):
                        where_parts.append(f"{k} = {v}")
                    elif isinstance(v, (pd.Timestamp, datetime.datetime)):
                        where_parts.append(f"{k} = '{v}'")
            elif isinstance(filt, list):
                if k in filt:
                    if isinstance(v, str):
                        where_parts.append(f"{k} = '{v}'")
                    elif isinstance(v, (int, float)):
                        where_parts.append(f"{k} = {v}")
                    elif isinstance(v, (pd.Timestamp, datetime.datetime)):
                        where_parts.append(f"{k} = '{v}'")
            elif isinstance(filt, str):
                where = filt
            if where_parts:
                where = ' and '.join(where_parts)
            if where:
                query += f''' 
                where {where}'''
        if return_query:
            return query
        self.execute_query(query)

    def update_table_from_df(self, table_name, data: pd.DataFrame, filt=False, return_query=False):
        querys = []
        for row_dict in data.to_dict('records'):
            data = row_dict
            if filt:
                if isinstance(filt, dict):
                    filt = filt
                elif isinstance(filt, str):
                    if filt in row_dict:
                        filt = {filt: row_dict[filt]}
                    else:
                        filt = filt
                elif isinstance(filt, list):
                    filt = {key: row_dict[key] for key in row_dict if key in filt}
            querys.append(self.update_table_from_dict(table_name, data, filt, return_query))
        if return_query:
            return querys

    def update_table_from_table(self, src_table, update_table, update_columns, filt):
        # Генерация SET части запроса
        if isinstance(update_columns, dict):
            set_part = ', '.join([f't1.{k} = t2.{v}' for k, v in update_columns.items()])
        elif isinstance(update_columns, list):
            set_part = ', '.join([f't1.{k} = t2.{k}' for k in update_columns])
        elif isinstance(update_columns, str):
            set_part = f', t1.{update_columns} = t2.{update_columns}'

        # Генерация WHERE части запроса
        if isinstance(update_columns, dict):
            where_part = ' AND '.join([f't1.{k} = t2.{v}' for k, v in filt.items()])
        elif isinstance(update_columns, list):
            where_part = ' AND '.join([f't1.{k} = t2.{k}' for k in filt])
        elif isinstance(update_columns, str):
            where_part = f' AND t1.{update_columns} = t2.{update_columns}'

        # Формирование SQL-запроса
        query = f'''
            UPDATE {src_table} AS t1
            SET
                {set_part}
            FROM {update_table} AS t2
            WHERE
                {where_part}
        '''
        self.execute_query(query)

    def delete_from_table(self, table_name, filt=False, return_query=False):
        query = f'''
            DELETE FROM
                {table_name}'''

        if filt:
            where = False
            if isinstance(filt, dict):
                where = ' AND '.join([f"{k} = '{v}'" if isinstance(v, (str, pd.Timestamp))
                                      else f"{k} = {v}" for k, v in filt.items()])
            elif isinstance(filt, str):
                where = filt
            if where:
                query += f''' 
                    WHERE 
                        {where}'''
        if return_query:
            return query
        self.execute_query(query)

    def delete_from_table_by_df(self, table_name, data: pd.DataFrame, filt=False, return_query=False):
        querys = []
        for row_dict in data.to_dict('records'):
            data = row_dict
            if filt:
                if isinstance(filt, dict):
                    filt = filt
                elif isinstance(filt, str):
                    if filt in row_dict:
                        filt = {filt: row_dict[filt]}
                    else:
                        filt = filt
                elif isinstance(filt, list):
                    filt = {key: row_dict[key] for key in row_dict if key in filt}
            querys.append(self.delete_from_table(table_name, filt, return_query))
        if return_query:
            return querys

    ''' Метод для выполнения SQL-запросов с использованием SQLAlchemy '''

    def execute_query(self, query, format_return=False, without_bd=False):
        # print(query)
        """
        Выполняет SQL-запрос с использованием SQLAlchemy Engine, обрабатывает результаты
        и возвращает данные в указанном формате.

        Параметры:
        - query (str): SQL-запрос для выполнения.
        - format_return (str or bool): Формат, в который нужно преобразовать результат.
          Поддерживаемые значения: 'df' (DataFrame), 'list' (список словарей),
          'exist' (проверка наличия данных), False (без дополнительного преобразования).
        - without_bd (bool): Флаг, указывающий, нужно ли использовать базу данных.
          Если True, выполняется без подключения к базе данных.

        Возвращает:
        - В зависимости от format_return:
          - Если format_return='df', возвращает pandas DataFrame.
          - Если format_return='list', возвращает список словарей.
          - Если format_return='exist', возвращает True, если данные существуют, иначе False.
          - Если format_return=False, возвращает оригинальные данные без преобразования.

        В случае ошибки выполнения SQL-запроса возвращает None.
        """
        # Создание SQLAlchemy Engine
        engine = self.__create_engine(without_bd=without_bd)

        # Проверка типа базы данных (ClickHouse или другая)
        if self.db_type == 'clickhouse':
            # Для ClickHouse, установка лимита запросов на 0 для SELECT и SHOW запросов
            if 'select' in query.lower() or 'show' in query.lower():
                engine.query_limit = 0
                data = engine.query_df(query)

                # Если тип данных - список, преобразовать в пустой DataFrame
                if type(data) == list:
                    data = pd.DataFrame()

                # Получение названий столбцов
                columns = data.columns
                engine.close()
            else:
                # Выполнение запросов
                data = engine.command(query)
                engine.close()
        else:
            Session = sessionmaker(bind=engine)
            with Session() as session:
                # Для других баз данных (не ClickHouse), использование SQLAlchemy Session
                result = session.execute(text(query))
                try:
                    data = result.fetchall()
                    columns = result.keys()
                except:
                    data = pd.DataFrame()
                    columns = []
                # Закрытие сессии
                session.close()

        try:
            engine.close()
        except:
            try:
                session.close()
            except:
                pass

        # Обработка формата возвращаемых данных
        if format_return == 'df':
            # Преобразование в DataFrame, если указан соответствующий формат
            data = pd.DataFrame(data, columns=columns)
        elif format_return == 'list':
            # Преобразование в список словарей, если указан соответствующий формат
            data = pd.DataFrame(data, columns=columns)
            data = data.to_dict('records')
        elif format_return == 'exist':
            # Преобразование в список словарей и проверка на наличие данных
            data = pd.DataFrame(data, columns=columns)
            data = data.to_dict('records')
            if data:
                if data[0]:
                    return True
            return False
        return data

    ''' Сложные составные методы'''

    def drop_view(self, view_name: str):
        db_request = f"""DROP VIEW IF EXISTS {self.db_name}.{view_name}"""
        self.execute_query(db_request)

    def create_simple_view(self, view_name, query):
        self.drop_view(view_name)

        db_request = f"""
                    CREATE VIEW 
                        {self.db_name}.{view_name}
                    AS ({query})"""
        self.execute_query(db_request)

    def create_materialized_view(self, view_name, query, table_name, table_columns, engine=False, table_partition=False,
                                 order=False):
        self.drop_view(view_name)
        self.create_table(table_name, table_columns, engine, table_partition, order)

        db_request = f"""
            create materialized view if not exists {self.db_name}.{view_name} to {table_name} as 
                ({query})
        """
        self.execute_query(db_request)

    def create_live_view(self, view_name, query):
        self.drop_view(view_name)

        db_request = f'''
            create live view if not exists {self.db_name}.{view_name} WITH REFRESH 1 AS 
                ({query})
        '''
        self.execute_query(db_request)

    def create_table_from_df(self, dataframe, table_name, table_columns=False, engine=False,
                             table_partition=False, order=False, drop_temp_table=True):
        if not table_columns:
            table_columns = self.parsing_df_col_types(dataframe)
        self.create_table(table_name, table_columns, engine=engine, table_partition=table_partition, order=order)
        self.write_data_to_table(dataframe, table_name, operation='append_all', drop_temp_table=drop_temp_table)
        dataframe = False

    def substitution_table(self, table_name, drop_old=True):
        temp_table_name = f'{table_name}_temp'
        old_table_name = f'{table_name}_old'

        if self.check_table_exists(old_table_name):
            self.drop_table(old_table_name)
        query = f'RENAME TABLE {table_name} TO {old_table_name}, {temp_table_name} TO {table_name}'
        self.execute_query(query)

        if drop_old:
            if self.check_table_exists(old_table_name):
                self.drop_table(old_table_name)

    def write_data_to_table(self, dataframe, table_name, operation='append', drop_temp_table=True,
                            compare_cols=[], exclude_columns=[], compare_filt_col=False, remove_old=True):
        """
        Загружает данные из датафрейма в базу данных.

        Параметры:
        - dataframe: pandas DataFrame, данные для загрузки
        - table_name: str, имя таблицы в базе данных
        - operation: str, операция загрузки ('append', 'replace', 'update')

        Варианты операции:
        - 'append': Добавить все содержимое датафрейма в БД.
        - 'replace': Заменить все содержимое БД на содержимое датафрейма.
        - 'update': Заменить только то, что есть в БД, а остальное добавить.

        Возвращает True при успешной загрузке, False в противном случае.
        """
        if self.db_type == 'clickhouse':
            engine = create_engine(f'clickhouse://{self.username}:{self.password}@{self.host}:{self.port}/{self.db_name}')
        else:
            engine = self.__create_engine()

        old_data = False
        new_data = False
        new_data = dataframe

        if not new_data.empty:
            # Если таблица не была создана ранее
            if not self.check_table_exists(table_name):
                self.create_table_from_df(new_data, table_name)
            else:
                if operation == 'append_all':
                    new_data.to_sql(table_name, engine, if_exists='append', index=False)

                if 'replace_all' in operation:
                    temp_table_name = f'{table_name}_temp'
                    if operation == 'replace_all':
                        # Проверяем наличие временной таблицы и если есть, удаляем
                        if self.check_table_exists(temp_table_name):
                            self.drop_table(temp_table_name)
                        # Создаем временную таблицу и загружаем данные
                        self.execute_query(self.get_query_created_table(table_name, new_table_name=temp_table_name))
                        new_data.to_sql(temp_table_name, engine, if_exists='append', index=False)
                        # Подменяем таблицу
                        self.substitution_table(table_name, drop_old=drop_temp_table)
                    elif operation == 'replace_all_first':
                        # Проверяем наличие временной таблицы и если есть, удаляем
                        if self.check_table_exists(temp_table_name):
                            self.drop_table(temp_table_name)
                        # Создаем временную таблицу и загружаем данные
                        self.execute_query(self.get_query_created_table(table_name, new_table_name=temp_table_name))
                        new_data.to_sql(temp_table_name, engine, if_exists='append', index=False)
                    elif operation in ['replace_all_middle', 'replace_all_end']:
                        new_data.to_sql(temp_table_name, engine, if_exists='append', index=False)
                        if operation == 'replace_all_end':
                            # Подменяем таблицу
                            self.substitution_table(table_name, drop_old=drop_temp_table)

                elif 'append_diff' in operation:
                    # Загружаем данные из основной и временной таблицы
                    query = f"SELECT * FROM {table_name}"
                    old_data = self.execute_query(query)
                    if old_data.empty:
                        new_data.to_sql(table_name, engine, if_exists='append', index=False)
                    else:
                        # Загружаем данные во временную таблицу и достаем их снова
                        temp_table_name = f'{table_name}_temp'
                        self.execute_query(self.get_query_created_table(table_name, new_table_name=temp_table_name))
                        new_data.to_sql(temp_table_name, engine, if_exists='append', index=False)
                        query = f"SELECT * FROM {temp_table_name}"
                        new_data = self.execute_query(query)
                        if self.check_table_exists(temp_table_name):
                            self.drop_table(temp_table_name)

                        diff_data = compare_df(old_data, new_data, compare_cols, exclude_columns, return_data='all')
                        # Добавляем новые
                        diff_data['only_new'].to_sql(table_name, engine, if_exists='append', index=False,
                                                     method='multi')
                        if operation == 'append_diff_with_drop':
                            # Удаляем отсутствующие в новом из старого
                            if remove_old and operation == 'append_diff_with_drop':
                                for i, s in diff_data['only_old'].iterrows():
                                    # print(self.delete_from_table(table_name, filt=dict(s), return_query=True))
                                    self.execute_query(
                                        self.delete_from_table(table_name, filt=dict(s), return_query=True))
                        diff_data = False

        try:
            engine.dispose()
            logger.log_debug(f'1 - Успешная попытка закрытия соединения {self.db_type}, table_name: {table_name}')
        except Exception as err:
            logger.log_error(
                f'1 - Неуспешная попытка закрытия соединения {self.db_type}, table_name: {table_name}, err: {err}')
            try:
                engine.close()
                logger.log_debug(f'2 - Успешная попытка закрытия соединения {self.db_type}, table_name: {table_name}')
            except Exception as err:
                logger.log_error(
                    f'2 - Неуспешная попытка закрытия соединения {self.db_type}, table_name: {table_name}, err: {err}')
        old_data = False
        new_data = False