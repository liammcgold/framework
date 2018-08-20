import pandas as pd

import os

log_file="log.csv"

columns=['operation','size','time','memory usage']









def log(operation,size,time,mem):




    if os.path.isfile(log_file):

        data = pd.read_csv(log_file)

    else:

        data = pd.DataFrame(columns=columns)


    data=pd.concat([data,pd.DataFrame(data=[[operation,size,time,abs(mem)]],columns=columns)],axis=0)

    data=data[columns]

    data.to_csv(path_or_buf=log_file)


