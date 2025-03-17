#libraries
import pandas as pd 
import matplotlib.pyplot as plot 
import numpy as np


#global variables
DJI_data: str = "./DJI.csv"
SAP_data: str = "./SAP.csv"
NAS_data: str = "./NAS.csv"


#main function
def main() -> None:
    #read in csv files
    DJI: pd = pd.read_csv(DJI_data)
    SAP: pd = pd.read_csv(SAP_data)
    NAS: pd = pd.read_csv(NAS_data)
    
    #clean up files
    DJI = custom_clean_up(DJI)
    SAP = custom_clean_up(SAP)
    NAS = custom_clean_up(NAS)
    
    #calculate the open-close change for a given day
    DJI["Difference"] = DJI["Close"] - DJI["Open"]
    SAP["Difference"] = SAP["Close"] - SAP["Open"]
    NAS["Difference"] = NAS["Close"] - NAS["Open"]
    
    #plot correlations
    pre_screen_information(DJI, "DOw Jones Industrial Average")
    
    
    #clean up memeory
    del DJI
    del SAP
    del NAS
    return


#custom functions
    #standardizes file cleanups
def custom_clean_up(dataframe: pd) -> pd:
    #test for empty dataframe and return error
    if len(dataframe) == 0:
        print(f"{dataframe} has no data")
        return dataframe
    
    # remove NAs
    dataframe.drop_na(inplace = True)
    
    #convert columns to numeric except the first date column
    for i in range(2, len(dataframe.columns)):
        dataframe.iloc[:,i] = dataframe.iloc[:,i].astype(float)
        
    return dataframe

    #develops and returns correlations with correplation plots
def pre_screen_information(dataframe: pd, output_prefix: str) -> None:
    #verifies that the dataframe has data
    if len(dataframe) == 0:
        print(f"{dataframe} is empty on correlation analyis")
        return
    
    #isolates all but the date information
    df_subset: pd = dataframe.iloc[:,1:]
    
    #creates a correlation matrix and saves it as a text file
    correlations: object = open(f"{output_prefix}_correlations.txt", "w")
    correlations.write(str(df_subset.corr()))
    correlations.close()
    
    #creates correlation plots of all the data
    columns_names: list = df_subset.columns.tolist()
    length = len(columns_names)
    for i in range(1, length - 1):
        for j in range(i + 1, length):
            plotter_support(df_subset.iloc[:, i], df_subset.iloc[:, j], output_prefix)
    return

    #create plots
def plotter_support(df1: pd, df2: pd, name: str) -> None:
    #create labels
    x_label: list[str] = df1.column.tolist()
    y_label: list[str] = df2.column.tolist()
    file_label = f"{name}_{x_label[0]}_vs_{y_label[0]}.tiff"
    
    #extracts data
    x_data: np = df1.DataFrame.to_numpy
    y_data: np = df2.DataFrame.to_numpy
    
    #generates plos
    plot.scatter(x_data, y_data)
    plot.xlabel(x_label[0])
    plot.ylabel(y_label[0])
    plot.title(f"{x_label}_vs_{y_label}")
    plot.savefig(file_label)
    
    #cleans up memory
    plot.clf
    del x_data
    del y_data 
    del x_label
    del y_label
    del file_label
    return

    #develop the array for a phase plot
def phase_plot_data(dataframe: pd, phases: int) -> list:
    #makes sure the number of phases is an integer
    if type(phases) != int:
        return 
    
    length: int = len(dataframe)
    return 

#custom classes


#execute if main

if __name__ == "__main__":
  main()