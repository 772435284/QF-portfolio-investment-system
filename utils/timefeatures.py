import numpy as np
import pandas as pd
import datetime

def date_to_index(date_string,start_datetime,date_format):
    # Transfer the date to index 0, 1, 2 ,3...
    return (datetime.datetime.strptime(date_string, date_format) - start_datetime).days

def index_to_date(index,start_datetime,date_format):
    # Transfer index back to date
    return (start_datetime + datetime.timedelta(index)).strftime(date_format)

