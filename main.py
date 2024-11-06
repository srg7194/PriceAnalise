import pprint
from configurator import Configurator

creds_path = 'files/secret/creds.json'
local_start = True


if __name__ == '__main__':
    config = Configurator(creds_path, local_start).get_config()
    pprint.pprint(config)
