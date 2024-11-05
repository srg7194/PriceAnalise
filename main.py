from addition.api import *
from addition.other import *
import pprint

creds_path = 'files/secret/credentials.json'

if __name__ == '__main__':
    creds = json_to_dict(creds_path)
    print(creds['config'])

    init = GoogleDriveAPI(service_account_file=creds['GoogleConfig']['path'])
    fileid = init.get_file_id(creds['GoogleConfig']['link'])

    filepath = 'files/config/config.xlsx'
    init.download_file(real_file_id=fileid, file_path=filepath)

    data = read_excel_file(filepath, sheet_names=None, data_to_dict=False)

    pprint.pprint(data['Таймфреймы'])
    # config = pd.read_excel('files/config/config.xlsx', 'Таймфреймы')
    # df_print(config)

    # for i, s in config.iterrows():
    #     print(s['Активно'] == True)