import tslearn
import pandas as pd
import cupy
import numpy
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import re
from astropy.table import Table
from cuml.multiclass import OneVsRestClassifier
from cuml.svm import LinearSVC
from cuml.linear_model import LogisticRegression
from cuml.metrics import accuracy_score
from cuml.cluster import KMeans as KMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split

# slurm 
import os
slurm_array_task_id=int(os.environ["SLURM_ARRAY_TASK_ID"])

# mlflow
import mlflow
from mlflow import log_metric, log_param, log_artifact
mlflow.set_tracking_uri("file:/gpfsscratch/rech/owt/commun/galaxy_classification/mlruns")
mlflow.set_experiment("cluster_classifier")

def set_pipeline_parameters():
    match slurm_array_task_id:
        case 0:
            num_clusters=3
            model_type='rdf'
            rdf_num_estimators=100
            rdf_max_depth=10
            svm_kernel=None
        case 1:
            num_clusters=3
            model_type='svm'
            rdf_num_estimators=None
            rdf_max_depth=None
            svm_kernel='linear'
        default:
            num_clusters=None
            model_type=None
            rdf_num_estimators=None
            rdf_max_depth=None
            svm_kernel=None
    return {'num_clusters':num_clusters,'model_type':model_type,'rdf_num_estimators':rdf_num_estimators,'rdf_max_depth':rdf_max_depth,'svm_kernel':svm_kernel}

def read_sfh_binned_data(dirname,filename):
    result = re.search('binned_SFHs-(.*)levels-JWST_(.*).txt', filename)
    if result:
        levels = int(result.group(1))
        z_bins = result.group(2)
        IDs, SFH_lev = read_sfh_data(dirname,filename,levels)
        return IDs, SFH_lev, levels, z_bins
    else:
        print("Filename doesn't contain levels or z_bins")

def read_sfh_data(dirname,filename,levels):
    data = join(dirname, filename)
    if isfile(data):
        df = pd.read_csv(data,sep='\t')    
        levels=df.columns[2:levels+2]
        SFH_lev=df[levels].values
        IDs = df['id_L19'].astype(int)
        return IDs, SFH_lev
    
def run_kmeans(x,num_clusters):
    x = TimeSeriesScalerMeanVariance().fit_transform(x)        
    X_train=cupy.asarray(x)
    kmeans = KMeans(n_clusters=num_clusters, max_iter=300, init='scalable-k-means++')
    kmeans.fit(X_train)
    cluster = cupy.asnumpy(kmeans.fit_predict(X_train))
    cluster_centers = cupy.asnumpy(kmeans.cluster_centers_)
    return x, cluster, cluster_centers

def get_sfh_cluster(dirname,filename,num_clusters):
    df_cluster = pd.DataFrame()
    IDs, SFH_lev, _, _ = read_sfh_binned_data(dirname,filename)
    _, cluster, _ = run_kmeans(SFH_lev,num_clusters)
    df_cluster['ID'] = IDs
    df_cluster['cluster'] = cluster
    return df_cluster

def get_photometry(filepath):
    data=Table.read(filepath)
    df_photometry=data.to_pandas()
    return df_photometry
    
def get_flux_cluster(df_cluster,df_photometry):
    df_photometry = df_photometry.merge(df_cluster, on= 'ID')
    df_flux = df_photometry[['ID','u_FLUX', 'gHSC_FLUX', 'rHSC_FLUX', 'iHSC_FLUX', 'zHSC_FLUX', 'cluster']]
    return df_flux
    
def train_classifier(dataset, classes):
    # Initialize mlflow run
    run_name = parameters.model_type+'_'
    if parameters.model_type == 'rdf':
        run_name = run_name+str(parameters.rdf_num_estimators)+'_'+str(parameters.rdf_max_depth)
        mlflow.log_param("rdf_num_estimators", parameters.rdf_num_estimators)
        mlflow.log_param("rdf_max_depth", parameters.rdf_max_depth)
    elif parameters.model_type == 'svm':
        run_name = run_name+str(parameters.svm_kernel)
        mlflow.log_param("svm_kernel", parameters.svm_kernel)
    run_name = run_name+'_'+str(parameters.num_clusters)
    mlflow.start_run(run_name=run_name)

    # log parameters
    if parameters.model_type == 'rdf':
        mlflow.log_param("rdf_num_estimators", parameters.rdf_num_estimators)
        mlflow.log_param("rdf_max_depth", parameters.rdf_max_depth)
    elif parameters.model_type == 'svm':
        mlflow.log_param("svm_kernel", parameters.svm_kernel)
    mlflow.log_param("num_clusters", parameters.num_clusters)
    mlflow.log_param("model_type", parameters.model_type)

    features = dataset[["u_FLUX","gHSC_FLUX","rHSC_FLUX","iHSC_FLUX","zHSC_FLUX"]]
    mlflow.log_param("features", features.columns)
    labels = dataset["cluster"]
    mlflow.log_param("labels", labels.name)
    
    X_data = features.to_numpy()
    Y_data= labels.to_numpy()
    
    #Create a scaler model that is fit on the input data.
    scaler = StandardScaler().fit(X_data)

    #Scale the numeric feature variables
    X_data = scaler.transform(X_data)

    #Convert target variable as a one-hot-encoding array
    Y_data = label_binarize(Y_data,classes=classes)
    
    #Split training and test data
    X_train,X_test,Y_train,Y_test = train_test_split( X_data, Y_data, train_size=0.75)
    mlflow.log_param("train_size", 0.75)

    X_test,X_val,Y_test,Y_val = train_test_split( X_test, Y_test, test_size=0.5)
    mlflow.log_param("test_size", 0.5)

    if parameters.model_type == 'rdf':
        classifier = OneVsRestClassifier(RandomForestClassifier(n_estimators=parameters.rdf_num_estimators, max_depth=parameters.rdf_max_depth))
    elif parameters.model_type == 'svm':
        classifier = OneVsRestClassifier(SVC(kernel=parameters.svm_kernel,probability=True))

    classifier.fit(X_train, Y_train)
    Y_test_hat = classifier.predict(X_test)
    score = accuracy_score(Y_test, Y_test_hat)
    mlflow.log_metric("accuracy_score", score)

    mlflow.end_run()

    return classifier, score

def run_pipeline(SFH_dirname,SFH_filename,FITS_filepath):
    parameters = set_pipeline_parameters()
    df_photometry = get_photometry(FITS_filepath)
    df_cluster = get_sfh_cluster(SFH_dirname,SFH_filename,parameters.num_clusters)
    df_flux = get_flux_cluster(df_cluster,df_photometry)
    classes = []
    for i in range(num_clusters):
        classes.append(i)
    classifier, score = train_classifier(df_flux, classes, parameters)
    return classifier, score
    