#libraries
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn import linear_model, metrics, datasets
from sklearn.preprocessing import StandardScaler
import matplotlib.cm as cm
from sklearn import metrics


#global variables
DJI_data: str = "./DJI.csv"
SAP_data: str = "./SAP.csv"
NAS_data: str = "./NAS.csv"
features: list = ["Open", "Close", "High", "Low"]

#main function
def main() -> None:
    #read in csv files
    DJI: pd = pd.read_csv(DJI_data)
    #SAP: pd = pd.read_csv(SAP_data)
    #NAS: pd = pd.read_csv(NAS_data)
    
    #clean up files
    DJI = custom_clean_up(DJI)
    #SAP = custom_clean_up(SAP)
    #NAS = custom_clean_up(NAS)
    
    #plot correlations
    #pre_screen_information(DJI, "Dow Jones Industrial Average")
    
    #create phase plots
    #create_phase_lots(DJI, "Dow Jones Industrial Average")
    
    #3d plots 
    # for i, j in enumerate(features):
    #     for k in range(i + 1, len(features) - 1):
    #         three_D_plot(DJI, j, features[k], "Dow Jones Industrial Average")
    
    #create a normalized array for fitting

    
    #K means clustering
    # Sensitivity analysis for number of clusters
    #k_means(DJI, features, "Dow Jones Industrial Average")
    
    #plotting clusters
    Cluster_plots("Open", "Close", "Difference", 4, DJI, "Down Jones Industrial Average")
    
    #clean up memeory
    del DJI
    #del SAP
    #del NAS
    return


#custom functions-------------------------------------------------------------------------------------------

    #Logit (machine learning)
def logistic_modeling(yvar: str, dataframe: pd, title: str, *args) -> None:
    #verify a complete dataframe
    if len(dataframe) == 0:
        print("Empty Dataframe - Logistic Regression")
        return
    
    #verify features have been lister
    if len(args) == 0:
        print("No features gives - Logistic Regression")
        return
    
    f_list: list = []
    sf_list: str = ""
    for i in args:
        f_list.append(i)
        sf_list = sf_list + i + " "

    #assemble data
    X: pd = dataframe[f_list].copy()
    Y: pd = dataframe[yvar].copy()
    
    #train the model
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)
    
    logr = linear_model.LogisticRegression()
    logr.fit(x_train, y_train)
    
    #test predictions
    predicted = logr.predict(x_test)
    
    #make confusion matrix
    confusion_matrix = metrics.confusion_matrix(y_train, predicted)
    
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ["Losses", "Gains"])
    cm_display.plot()
    
    plt.savefig(f"{sf_list} vs {yvar} ML Confusion.tiff")
    
    plt.clf()
    plt.close()
    
    #memory cleanup
    del cm_display
    del confusion_matrix
    del predicted
    del X
    del Y
    del x_test
    del x_train
    del y_test
    del y_train
    del f_list
    del sf_list
    
    return
        ##create linear regressions
def Multi_linear_regressions(dataframe: pd, yvar: str, title: str, *args) -> None:
    #check for an empty dataframe
    if len(dataframe) == 0:
        print("Empty Dataframe - Linear Regression")
        return 
    
    #checks that features have been included
    if len(args) == 0:
        print("No features are specificed - Linear Regression")
        return
    
    #generates feature list
    f_list = []
    for i in args:
        f_list.append(i)
            
    #perform linear regression
    x: pd = dataframe[f_list].copy()
    y: pd = dataframe[yvar].copy()
    
    #build up the training data and fit the linear model
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)
    reg = linear_model.LinearRegression()
    reg.fit(x_train, y_train)
    
    #plot the linear regression
    
        #plot the training data and residuals
    plt.scatter(reg.predict(x_train), reg.predict(x_train) - y_train, color = "green", s = 10, label = "Training Data") 
    
        #plot the validation set
    plt.scatter(reg.predict(x_test), reg.predict(x_test) - y_test, color = "blue", s = 10, label = "Validation Data") 
    
    #create list of args for title
    
    text = ""
    for i in f_list:
        text = text + i +" "
    
    plt.xlabel("Data Point")
    plt.ylabel("Residuals")
    plt.title(f"Residual Plots from Linear Regressions\n{text} vs {yvar} - {title}")
    
    plt.savefig(f"LR Residuals {args} vs {yvar} - {title}.tiff")
    
    plt.clf()
    plt.close()
    
    del f_list
    del text
    del x
    del y
    del x_train
    del y_train
    del x_test
    del y_test
    
    return 
    
    #plot 3D clusters - thanks for SKLearn website
def Cluster_plots(xvar: str, yvar: str, zvar: str, clusters: int, df: pd, title: str) -> None:
    #check for empty dataframe
    if len(df) == 0:
        print("Empty Dataframe - Cluster Plot")
        return
    
    #extracts out the relevant dataframe
    dataframe: pd = df[[xvar, yvar, zvar]].copy()
    
    #createa a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    
    #subplot for the silhouete plot 
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(dataframe) + (clusters + 1) * 10])
    ax1.set_title(f"Silhouette plot for {xvar}-{yvar}-{zvar}: {title}")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    
    #createa a cluster model
    clusterer = KMeans(n_clusters = clusters, random_state = 42)
    cluster_labels = clusterer.fit_predict(dataframe)
        
    # Aggregate the silhouette scores for samples belonging to cluster, and sort them
    sample_silhouette_values = silhouette_samples(dataframe, cluster_labels)
    y_lower = 10
    for i in range(clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor = color, edgecolor = color, alpha = 0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples
    
    #build up cluster plot
    colors = cm.nipy_spectral(cluster_labels.astype(float) / clusters)
    ax2.scatter(dataframe.iloc[:, 0], dataframe.iloc[:, 1], marker = ".", s = 30, lw = 0, alpha = 0.7, c = colors, edgecolor = "k")
    
    #label clusters
    centers = clusterer.cluster_centers_
    
        # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker = "o", c = "white", alpha = 1, s = 200, edgecolor = "k")
    
    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker = "$%d$" % i, alpha = 1, s = 50, edgecolor = "k")
        
    ax2.set_title(f"Clustering of {title}")
    ax2.set_xlabel(xvar)
    ax2.set_ylabel(yvar)
        
    plt.suptitle("Silhouette analysis for KMeans clustering on sample data with n_clusters = %d" %clusters, fontsize = 14, fontweight = "bold")
    
    plt.savefig(f"Cluster Plot {xvar} {yvar} {zvar} {title}.tiff")
    plt.clf()
    plt.close()
    
    del dataframe
    del centers
    del colors
    del y_lower
    del sample_silhouette_values
    
    return

    #run the k-means analysis of a dataset
def k_means(dataframe: pd, features: list, title_info: str) -> None:
    #test for empty dataframe
    if len(dataframe) == 0:
        print("Empty Data Frame - k_mean function")
        return
    
    #check to make sure there are at least 4 data points per cluster
    if len(dataframe) < 10 * 4:
        print("Not enough data points for cluster analysis")
        return
    
    #remove only the relevant columns
    df2 = dataframe[features].copy()
    df2["Difference"] = dataframe["Difference"].copy()
    df2['Day'] = range(len(df2))
    
    #normals and find best k-means
    cluster_range = range(2, 10)
    
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df2) #change date to a number and drop the date column
    df2 = pd.DataFrame(data = scaled, columns = df2.columns)
    
    for i, j in enumerate(features):
        for k in range(i + 1, len(features)):
            silhouette_scores = []
            for n_clusters in cluster_range:
                kmeans = KMeans(n_clusters = n_clusters, random_state = 42)
                kmeans_labels = kmeans.fit_predict(df2[[j, features[k], "Difference"]])
                silhouette_avg = silhouette_score(df2[[j, features[k], "Difference"]], kmeans_labels)
                silhouette_scores.append(silhouette_avg)
            k_means_plots(j, features[k], cluster_range, silhouette_scores, title_info)
    
    del cluster_range
    del silhouette_scores
    del df2
    del scaled
    
    return

    #plot and save k-means
def k_means_plots(xvar: str, yvar: str, cluster_range: list, silhouette_scores: list, title_info: str) -> None:
    #test for empty data sets
    if len(silhouette_scores) == 0 or len(cluster_range) == 0:
        print("Empty Silhouette Scores")
        return
    
    #create plot    
    plt.figure(figsize=(10, 6))
    plt.plot(cluster_range, silhouette_scores, marker='o', linestyle='--')
    plt.title(f'Sensitivity Analysis {xvar} - {yvar} - Difference: Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    plt.savefig('sensitivity_analysis.png')
    plt.savefig(f"{title_info} {xvar} {yvar}.tiff")
    plt.clf()
    plt.close()
    return


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
    for i in features:
        dataframe[i] = dataframe[i].astype(float)
        
    dataframe["Gain"] = dataframe["Diffrence"].map(gain_day)
        
    #create a difference columns
    dataframe["Difference"] = dataframe["Close"] - dataframe["Open"]
    return dataframe

    #create conditions for binary outcomes on changes
def gain_day(value: float) -> int:
    if value > 0:
        return 1
    return 0

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
    plt.scatter(df1, df2)
    plt.xlabel(df1_name)
    plt.ylabel(df2_name)
    plt.title(f"{name} - {df1_name} vs {df2_name}")
    plt.savefig(file_label)
    
    #cleans up memory
    plt.clf()
    plt.close()
    del file_label
    return

    #develop the array for a phase plot
def phase_plot_data(dataframe: pd, phases: int) -> list:
    #makes sure the number of phases is an integer and that the dataframe has data
    if type(phases) != int or len(dataframe) == 0:
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
    plt.scatter(x = data[0], y = data[1], s = 1)
    plt.xlabel("N")
    plt.ylabel("N + 1" )
    plt.title(f"Phase Plot {plot_name}")
    plt.savefig(file_label)
    
    #cleans up memory
    plt.clf()
    plt.close()
    del file_label
    return
    
    #3D Phase Plot
def three_D_phase_plot(data: list, plot_name: str) -> None:
    #create labels
    file_label = f"3Dphase_plot_{plot_name}.tiff"
    
    #generates plots
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    ax.scatter(data[0], data[1], data[2])
    ax.set_xlabel("N")
    ax.set_ylabel("N + 1" )
    ax.set_zlabel("N + 2")
    plt.title(f"3D Phase Plot {plot_name}")
    plt.savefig(file_label)
    
    #cleans up memory
    plt.clf()
    plt.close()
    del fig
    del ax
    del file_label
    return

    #3D Plot vs difference
def three_D_plot(data: list, xvar: str, yvar: str, plot_name: str) -> None:
    #create labels
    file_label = f"3D_plot_{plot_name} {xvar} {yvar} difference.tiff"
    
    #generates plots
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    ax.scatter(data[xvar], data[yvar], data["Difference"])
    ax.set_xlabel(xvar)
    ax.set_ylabel(yvar)
    ax.set_zlabel("Difference")
    plt.title(f"3D Phase Plot {plot_name}")
    plt.savefig(file_label)
    
    #cleans up memory
    plt.clf()
    plt.close()
    del fig
    del ax
    del file_label
    return    
#custom classes

#execute if main

if __name__ == "__main__":
  main()