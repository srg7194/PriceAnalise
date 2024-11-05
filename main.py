from addition.api import *
from addition.other import *

if __name__ == '__main__':
    creds = json_to_dict('files/secret/credentials.json')
    print(creds['GoogleConfig'])

    init = GoogleDriveAPI(service_account_file=creds['GoogleConfig']['path'])
    fileid = init.get_file_id(creds['GoogleConfig']['link'])
    init.download_file(real_file_id=fileid, file_path='files/config/config.xlsx')

    config = pd.read_excel('files/config/config.xlsx', 'Подключения')
    df_print(config)

    for i, s in config.iterrows():
        print(s['Активно'] == True)

