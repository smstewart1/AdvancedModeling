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
    
    #plot correlations
    pre_screen_information(DJI, "Dow Jones Industrial Average")
    
    #create phase plots
    create_phase_lots(DJI, "Dow Jones Industrial Average")
    
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
    dataframe.dropna(inplace = True)
    dataframe.replace(',', '', regex = True, inplace = True)
        
    #convert columns to numeric except the first date column
    for i in ["Open", "Close", "High", "Low"]:
        dataframe[i] = dataframe[i].astype(float)
        
    #create a difference columns
    dataframe["Difference"] = dataframe["Close"] - dataframe["Open"]
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
    for i in range(0, length - 1):
        for j in range(i + 1, length):
            plotter_support(df_subset[columns_names[i]], df_subset[columns_names[j]], columns_names[i], columns_names[j], output_prefix)
    return

    #create plots
def plotter_support(df1: pd, df2: pd, df1_name: str, df2_name: str, name: str) -> None:
    #create labels
    file_label = f"{name}_{df1_name}_vs_{df2_name}.tiff"
       
    #generates plos
    plot.scatter(df1, df2)
    plot.xlabel(df1_name)
    plot.ylabel(df2_name)
    plot.title(f"{name} - {df1_name} vs {df2_name}")
    plot.savefig(file_label)
    
    #cleans up memory
    plot.clf()
    plot.close()
    del file_label
    return

    #develop the array for a phase plot
def phase_plot_data(dataframe: pd, phases: int) -> list:
    #makes sure the number of phases is an integer
    if type(phases) != int:
        return 
    
    initial_list: list = dataframe.tolist()
    return_list: list = []
    length: int = len(dataframe)    
    for i in range(0, phases):
        return_list.append(initial_list[i:length - phases + i])
    
    return return_list

    #create phase plots
def create_phase_lots(data: pd, title: str) -> None:
    #generate phase plots
    for i in ["Open", "Close", "High", "Low"]:
        TDPhaseData = phase_plot_data(data[i], 2)
        two_D_phase_plot(TDPhaseData, f"{title} - {i}")
        del TDPhaseData
        TDPhaseData = phase_plot_data(data[i], 3)
        three_D_phase_plot(TDPhaseData, f"{title} - {i}")
        del TDPhaseData
    
    return

    #2D Phase plot
def two_D_phase_plot(data: list, plot_name: str) -> None:
    #create labels
    file_label = f"2Dphase_plot_{plot_name}.tiff"
    
    #generates plots
    plot.scatter(x = data[0], y = data[1], s = 1)
    plot.xlabel("N")
    plot.ylabel("N + 1" )
    plot.title(f"Phase Plot {plot_name}")
    plot.savefig(file_label)
    
    #cleans up memory
    plot.clf()
    plot.close()
    del file_label
    return
    
    #3D Plot
def three_D_phase_plot(data: list, plot_name: str) -> None:
    #create labels
    file_label = f"3Dphase_plot_{plot_name}.tiff"
    
    #generates plots
    fig = plot.figure()
    ax = fig.add_subplot(projection = '3d')
    ax.scatter(data[0], data[1], data[2])
    ax.set_xlabel("N")
    ax.set_ylabel("N + 1" )
    ax.set_zlabel("N + 2")
    plot.title(f"3D Phase Plot {plot_name}")
    plot.savefig(file_label)
    
    #cleans up memory
    plot.clf()
    plot.close()
    del fig
    del ax
    del file_label
    return
    
#custom classes

#execute if main

if __name__ == "__main__":
  main()