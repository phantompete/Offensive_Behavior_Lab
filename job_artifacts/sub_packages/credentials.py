import ads
import os
import configparser
import shutil
from zipfile import ZipFile
from tempfile import NamedTemporaryFile
import urllib
import re
import sqlalchemy
from sqlalchemy import create_engine
import cx_Oracle

def create_uri(user_name, password, wallet_name, wallet_storage_directory):
    
    database_name = wallet_name
    database_user = user_name
    database_password = password
    
    wallet_storage_directory = wallet_storage_directory

    # Create the wallet directory if missing: 
    ads.set_documentation_mode(False)

    os.makedirs(wallet_storage_directory, mode=0o700, exist_ok=True)

    wallet_path = os.path.join(wallet_storage_directory, database_name)

    # Prepare to store ADB connection information
    adb_config = os.path.join(wallet_storage_directory, '.credentials')

    # Write a configuration file for login creds.
    config = configparser.ConfigParser()
    config.read(adb_config)
    config[database_name] = {'tns_admin': wallet_path,
                             'sid': '{}_medium'.format(database_name.lower()),
                             'user': database_user,
                             'password': database_password}
    with open(adb_config, 'w') as configfile:
        config.write(configfile)


    # Read in the credentials configuration files
    my_config = configparser.ConfigParser()
    my_config.read(adb_config)

    # Access a setting
    print(my_config[database_name].get('user'))

    # Limit the information to a specific database
    my_creds = my_config[database_name]
    print(my_creds.get('user'))


    # extract the wallet
    wallet_file = 'Wallet_{}.zip'.format(database_name)
    wallet_filename = os.path.join(wallet_storage_directory, wallet_file)
    if not os.path.exists(wallet_filename):
        print("The file {} does not exist.".format(wallet_filename))
        print("Please copy the Wallet file, {}, into the directory {} then rerun this cell.".format(wallet_file, wallet_filename))
    else:
        os.makedirs(wallet_path, mode=0o700, exist_ok=True)
        with ZipFile(wallet_filename, 'r') as zipObj:
            zipObj.extractall(wallet_path)


    # Update the sqlnet.ora

    sqlnet_path = os.path.join(wallet_path, 'sqlnet.ora')
    sqlnet_original_path = os.path.join(wallet_path, 'sqlnet.ora.original')
    sqlnet_backup_path = os.path.join(wallet_path, 'sqlnet.ora.backup')
    if not os.path.exists(sqlnet_original_path):
        shutil.copy(sqlnet_path, sqlnet_original_path)
    if os.path.exists(sqlnet_path):
        shutil.copy(sqlnet_path, sqlnet_backup_path)
    sqlnet_re = re.compile('(WALLET_LOCATION\s*=.*METHOD_DATA\s*=.*DIRECTORY\s*=\s*\")(.*)(\".*)', 
                           re.IGNORECASE)
    tmp = NamedTemporaryFile()
    with open(sqlnet_path, 'rt') as sqlnet:
        for line in sqlnet:
            tmp.write(bytearray(sqlnet_re.subn(r'\1{}\3'.format(wallet_path), line)[0], 
                                encoding='utf-8'))
    tmp.flush()
    shutil.copy(tmp.name, sqlnet_path)
    tmp.close()

    # Add TNS_ADMIN to the environment
    os.environ['TNS_ADMIN'] = config[database_name].get('tns_admin')

    # Test the database connection
    creds = config[database_name]
    connect = 'sqlplus ' + creds.get('user') + '/' + creds.get('password') + '@' + creds.get('sid')
    print(os.popen(connect).read())

    # Get the URI to connect to the database
    uri='oracle+cx_oracle://' + creds.get('user') + ':' + creds.get('password') + '@' + creds.get('sid')
    
    engine = create_engine(uri)

    return engine, wallet_filename
