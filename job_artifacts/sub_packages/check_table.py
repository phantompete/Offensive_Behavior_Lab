
def check_table(SCHEMA_NAME, password, wallet_name, wallet_filename):
    
    import ads
    service_name = wallet_name.lower() + "_high"
    #wallet_location = wallet_storage_directory + "/" + "Wallet_" + wallet_name + ".zip"
    
    print("service name is in check_table " + service_name)
    print("wallet location is in check_table " + wallet_filename)

    creds = {"user_name": SCHEMA_NAME,
        "password":  password,
        "service_name": service_name,
        "wallet_location": wallet_filename}
    
    print(creds)
    
    try:
        check_table_exists = pd.DataFrame.ads.read_sql("SELECT COUNT(*) AS CHECKX FROM ocw_run_results", connection_parameters=creds)
        checkx = check_table_exists['CHECKX'][0]  #checkx will be '1' in table exits
        print("Table already exist, so append table")
        table_status = 'append'            

    except:
        table_status = 'replace'
        print("Table status is replace")

    return table_status
