#libraries
import pandas as pd 
import matplotlib as plot 


#global variables
DJI_data: str = "./"
SAP_data: str = "./"
NAS_data: str = "./"


#main fucntion
def main() -> None:
    #read in csv files
    DJI: pd = pd.read_csv(DJI_data)
    SAP: pd = pd.read_csv(SAP_data)
    NAS: pd = pd.read_csv(NAS_data)
    
    #clean up pd files
    DJI = custom_clean_up(DJI)
    SAP = custom_clean_up(SAP)
    NAS = custom_clean_up(NAS)
    
    
    
    return


#custom functions
def custom_clean_up(dataframe: pd) -> pd:
    dataframe.drop_na(inplace = True)
    return dataframe

#custom classes


#execute if main

#execute main
if __name__ == "__main__":
  main()