# Librerias EDA

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import f_classif, SelectKBest
import numpy as np
import json
from pickle import dump

# Librerias ML

from sklearn.linear_model import LinearRegression, Lasso, ElasticNetCV, Ridge

from sklearn.metrics import mean_squared_error, r2_score

#Recopilamos datos
data_url = "https://raw.githubusercontent.com/4GeeksAcademy/regularized-linear-regression-project-tutorial/main/demographic_health_data.csv"
sep = (",")

def DataCompiler(url, sep):
    data = pd.read_csv(url, sep = sep)

    #Guardamos el csv en local
    data.to_csv("../data/raw/raw_data.csv", sep=";")

    return data

data = DataCompiler(data_url, sep)

print(data)

def DataInfo(dataset):
    print(f"Dataset dimensions: {dataset.shape}")
    print(f"\nDataset information:\n{dataset.info()}")
    print(f"\nDataset nan-values: {dataset.isna().sum().sort_values(ascending=False)}")
    

DataInfo (data)

#Funcion para eliminar duplicados

#Columna identificadora del Dataset.

def EraseDuplicates(dataset):
    older_shape = dataset.shape
    id = "fips"
    
    if ("id" in locals()):
        dataset.drop(id , axis = 1, inplace = True)
                     
    if (dataset.duplicated().sum()):
        print(f"Total number of duplicates {dataset.duplicated().sum()}")
        print ("Erase duplicates...")
        dataset.drop_duplicates(inplace = True)
    else:
        print ("No coincidences.")
        pass
    
    print (f"The older dimension of dataset is {older_shape}, and the new dimension is {dataset.shape}.")
    
    return dataset

data = EraseDuplicates(data)

#Quierlo eliminar todos los datos que sean porcentajes
percent_data = []
columns = data.columns

for i in columns:
    if ("%" in str(i) or "Percent" in str(i) or "PCT" in str(i)):
        percent_data.append(i)



#Tambien quiero quitar todos los rates.
r_data = []

for i in columns:
    if("R" in str(i)):
        r_data.append(i)

r_data

#No quiero las prevalence, solo los numeros totales.
prevalence_data = []

for i in columns:
    if("prevalence" in str(i)):
        prevalence_data.append(i)

prevalence_data

#Tampoco quiero los AAMC
AAMC_data = []
for i in columns:
    if("AAMC" in str(i)):
        AAMC_data.append(i)

AAMC_data

#Los nombres no interesan si hay un FIP
names_data = []

for i in columns:
    if("NAME" in str(i)):
        names_data.append(i)

names_data

#Funcion para eliminar datos irrelevantes.

def EraseIrrelevants(dataset, lst):
    older_shape = data.shape
    print("Erase irrelevant´s dates...")
    dataset.drop(lst, axis = 1, inplace = True)
    print (f"The old dimension of dataset is {older_shape}, and the new dimension is {dataset.shape}.")
    return dataset

EraseIrrelevants(data, percent_data)
EraseIrrelevants(data, r_data)
EraseIrrelevants(data, prevalence_data)
EraseIrrelevants(data, AAMC_data)
EraseIrrelevants(data, names_data)
EraseIrrelevants(data, ["POP_ESTIMATE_2018", "N_POP_CHG_2018", "POVALL_2018", "MEDHHINC_2018", "Median_Household_Income_2018", "TOT_POP", "GQ_ESTIMATES_2018", "Total Population", "Population Aged 60+", "STATE_FIPS", "CNTY_FIPS", "county_pop2018_18 and older", "CI90LBINC_2018", "CI90UBINC_2018"])

#cambiamos el nombre de las columnas de las edades, por mania solamente.

data.rename( columns= {"0-9" : "child", "19-Oct" : "teenager", "20-29" : "young", "30-39" : "adult", "40-49" : "senior", "50-59" : "older_senior", "60-69" : "retired", "70-79" : "elder", "80+" : "late_elder"}, inplace=True)
data.rename( columns={"Less than a high school diploma 2014-18" : "less_highSchool", "High school diploma only 2014-18" : "highSchool", "Some college or associate's degree 2014-18" : "college", "Bachelor's degree or higher 2014-18" : "bachelor"}, inplace=True)
data.rename( columns={"White-alone pop" : "caucasian", "Black-alone pop" : "afrodescendant", "Asian-alone pop" : "asiatic", "Native American/American Indian-alone pop" : "american-native", "Hawaiian/Pacific Islander-alone pop" : "maoli", "Two or more races pop" : "mixted"}, inplace=True)
data.rename( columns={"Family Medicine/General Practice Primary Care (2019)" : "Family_Medicine_General_Practice_Primary_Care(2019)"}, inplace=True)

#Analisis sobre variables categoricas

def CategoricGraf(dataset):
    #Creamos la figura
    fig, axis = plt.subplots(1, 2, figsize=(15,5))

    #Creamos las graficas necesarias
    sns.histplot( ax = axis[0], data = dataset, x = "Urban_rural_code")
   

    #Mostramos el grafico.
    plt.tight_layout()
    plt.show()

CategoricGraf(data)

# Analisis sobre variables numericas

def NumericalGraf(dataset):
    #Creamos la figura
    fig, axis = plt.subplots(4, 3, figsize=(15,8), gridspec_kw={"height_ratios" : [6,1,6,1]})

    #Creamos las graficas necesarias
    sns.histplot( ax = axis[0,0], data = dataset[dataset["Obesity_number"] <= 15000], x = "Obesity_number", kde = True).set(xlabel = None)
    sns.boxplot( ax = axis[1,0], data = dataset[dataset["Obesity_number"] <= 15000], x = "Obesity_number")
    sns.histplot( ax = axis[0,1], data = dataset[dataset["Heart disease_number"] <= 5000], x = "Heart disease_number", kde = True).set(xlabel = None)
    sns.boxplot( ax = axis[1,1], data = dataset[dataset["Heart disease_number"] <= 5000], x = "Heart disease_number")
    sns.histplot( ax = axis[0,2], data = dataset[dataset["COPD_number"] <= 5000], x = "COPD_number", kde = True).set(xlabel = None)
    sns.boxplot( ax = axis[1,2], data = dataset[dataset["COPD_number"] <= 5000], x = "COPD_number")
    sns.histplot( ax = axis[2,0], data = dataset[dataset["diabetes_number"] <= 8000], x = "diabetes_number", kde = True).set(xlabel = None)
    sns.boxplot( ax = axis[3,0], data = dataset[dataset["diabetes_number"] <= 8000], x = "diabetes_number")
    sns.histplot( ax = axis[2,1], data = dataset[dataset["CKD_number"] <= 1500], x = "CKD_number", kde = True).set(xlabel = None)
    sns.boxplot( ax = axis[3,1], data = dataset[dataset["CKD_number"] <= 1500], x = "CKD_number")
    sns.histplot( ax = axis[2,2], data = dataset[dataset["anycondition_number"] <= 30000], x = "anycondition_number", kde = True).set(xlabel = None)
    sns.boxplot( ax = axis[3,2], data = dataset[dataset["anycondition_number"] <= 30000], x = "anycondition_number")
    
    plt.tight_layout()
    plt.show()

NumericalGraf(data)

etnic_list = ["caucasian", "afrodescendant", "american-native", "asiatic", "maoli","mixted"]
degree_list = ["less_highSchool", "highSchool", "college", "bachelor"]
employers_list = ["Employed_2018", "Unemployed_2018"]

def NumNumAnalysi(dataset, y, x_list, x_width, y_width):
    #Creamos la figura
    fig, axis = plt.subplots(x_width, y_width, figsize=(15,10))
    x_axis = 0
    y_axis = 0
    for i in range(len(x_list)):
        if (y_axis == 0):
            sns.regplot( ax = axis[x_axis,y_axis], data = dataset, x = x_list[i], y = y)
        else:
            sns.regplot( ax = axis[x_axis,y_axis], data = dataset, x = x_list[i], y = y).set(ylabel = None)

        if (y_axis < y_width - 1):
            sns.heatmap( data[[y,x_list[i]]].corr(), annot=True, fmt=".2f", ax = axis[x_axis + 1, y_axis], cbar=False, xticklabels = False)
        else:
            sns.heatmap( data[[y,x_list[i]]].corr(), annot = True, fmt = ".2f", ax = axis[x_axis + 1, y_axis], xticklabels = False) 
        
        y_axis = y_axis + 1
        if (y_axis == y_width):
            y_axis = 0
            x_axis = x_axis + 2
    
    plt.tight_layout()
    plt.show()

NumNumAnalysi(data, "Heart disease_number", etnic_list, 4, 3)
NumNumAnalysi(data, "Heart disease_number", degree_list, 4, 2)
NumNumAnalysi(data, "Heart disease_number", employers_list, 2, 2)

#Combinación Target/Pred

def CombTargPred(dataset):
    
    fig, axis = plt.subplots(3, 2, figsize = (10, 8))

    sns.regplot(ax = axis[0,0], data = dataset, x = "Urban_rural_code", y = "Heart disease_number")
    sns.regplot(ax = axis[0,1], data = dataset, x = "Urban_rural_code", y = "diabetes_number")
    sns.regplot(ax = axis[1,0], data = dataset, x = "Urban_rural_code", y = "anycondition_number")
    sns.regplot(ax = axis[1,1], data = dataset, x = "Urban_rural_code", y = "Obesity_number")
    sns.regplot(ax = axis[2,0], data = dataset, x = "Urban_rural_code", y = "COPD_number")
    sns.regplot(ax = axis[2,1], data = dataset, x = "Urban_rural_code", y = "CKD_number")
    

    plt.tight_layout()
    plt.show()

CombTargPred(data)

#Tabla de correlaciones
fig, axis = plt.subplots(figsize=(20,20))

sns.heatmap(data.corr(), annot=True, fmt=".2f")

plt.tight_layout()
plt.show()

# Comprobamos las metricas de la tabla.

data.describe()

target = "Heart disease_number"

#Creamos una funcion para transformar los outliers.

def SplitOutliers(dataset, target):
    
    dataset_with_outliers = dataset.copy()
    
    #Establecemos los limites.
    for i in dataset.columns:
        if (i == target):
            print(f"Target detected: {target}")
            pass
        
        #Esta parte la tengo que mejorar para poder clasificar los campos categoricos
        elif (i == "Urban_rural_code"):
            print(f"Categorical predictor: Urban_rural_code")
            pass

        else:
            stats = dataset[i].describe()
            iqr = stats["75%"] - stats["25%"]
            upper_limit = float(stats["75%"] + (2 * iqr))
            lower_limit = float(stats["25%"] - (2 * iqr))
            if (lower_limit < 0):
                lower_limit = 0

            #Ajustamos el outlier por encima.
            dataset[i] = dataset[i].apply(lambda x : upper_limit if (x > upper_limit) else x)

            #Ajustamos el outlier por debajo.
            dataset[i] = dataset[i].apply(lambda x : lower_limit if (x < lower_limit) else x)

            #Guardamos los límites en un json.
            with open (f"../data/interim/outerliers_{i}.json", "w") as j:
                json.dump({"upper_limit" : upper_limit, "lower_limit" : lower_limit}, j)
    
    return dataset_with_outliers, dataset

data_with_outliers, data_without_outliers = SplitOutliers (data, target)

#Comprobamos si existen valores faltantes.

data_with_outliers.isna().sum().sort_values()
data_without_outliers.isna().sum().sort_values()

# Primero dividimos los dataframes entre test y train

def SplitData(dataset, target):
    #Aislamos el target
    features = []
    for i in dataset.columns:
        if (i == target):
            pass
        else:
            features.append(i)
    
    x = dataset.drop(target, axis = 1)[features]
    y = dataset[target].squeeze()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42)

    return x_train, x_test, y_train, y_test

x_train_with_outliers, x_test_with_outliers, y_train, y_test = SplitData(data_with_outliers, target)
x_train_without_outliers, x_test_without_outliers, _, _ = SplitData(data_without_outliers, target)

y_train.to_csv("../data/processed/y_train.csv")
y_test.to_csv("../data/processed/y_test.csv")

#Tenemos que escalar los dataset con Normalizacion y con Escala mM (min-Max)

#Normalizacion
def StandardScaleData(dataset):
    #Aislamos el target
    features = []
    for i in dataset.columns:
        if (i == target):
            pass
        else:
            features.append(i)

    scaler = StandardScaler()
    scaler.fit(dataset)

    x_scaler = scaler.transform(dataset)
    x_scaler = pd.DataFrame(dataset, index = dataset.index, columns = features)
    
    if(dataset is x_train_with_outliers):
        dump(scaler, open("../data/interim/standar_scale_with_outliers.sav", "wb"))

    elif(dataset is x_train_without_outliers):
        dump(scaler, open("../data/interim/standar_scale_without_outliers.sav", "wb"))

    return x_scaler

x_train_with_outliers_standarscale = StandardScaleData(x_train_with_outliers)
x_train_without_outliers_standarscale = StandardScaleData(x_train_without_outliers)
x_test_with_outliers_standscale = StandardScaleData(x_test_with_outliers)
x_test_without_outliers_standscale = StandardScaleData(x_test_without_outliers)

#Escala mM
def MinMaxScaleData(dataset):
    #Aislamos el target
    features = []
    for i in dataset.columns:
        if (i == target):
            pass
        else:
            features.append(i)

    scaler = MinMaxScaler()
    scaler.fit(dataset)

    x_scaler = scaler.transform(dataset)
    x_scaler = pd.DataFrame(dataset, index = dataset.index, columns = features)

    if(dataset is x_train_with_outliers):
        dump(scaler, open("../data/interim/min-Max_Scale_with_outliers.sav", "wb"))

    elif(dataset is x_train_without_outliers):
        dump(scaler, open("../data/interim/min-Max_Scale_without_outliers.sav", "wb"))

    return x_scaler

x_train_with_outliers_mMScale = MinMaxScaleData(x_train_with_outliers)
x_train_without_outliers_mMScale = MinMaxScaleData(x_train_without_outliers)
x_test_with_outliers_mMScale = MinMaxScaleData(x_test_with_outliers)
x_test_without_outliers_mMScale = MinMaxScaleData(x_test_without_outliers)

#Seleccion de caracteristicas

def SelectFeaturesTrain(dataset, y, filename, k = 5):
    sel_model = SelectKBest(f_classif, k=k)
    sel_model.fit(dataset, y)
    col_name = sel_model.get_support()
    x_sel = pd.DataFrame(sel_model.transform(dataset), columns = dataset.columns.values[col_name])
    dump(sel_model, open(f"../data/interim/{filename}.sav", "wb"))
    train_cols = x_sel.columns
    return x_sel, train_cols

def SelectFeaturesTest(dataset, y, filename, train_cols, k = 5):
    dataset = pd.DataFrame(dataset[train_cols])
    sel_model = SelectKBest(f_classif, k=k)
    sel_model.fit(dataset, y)
    col_name = sel_model.get_support()
    x_sel = pd.DataFrame(sel_model.transform(dataset), columns = dataset.columns.values[col_name])
    dump(sel_model, open(f"../data/interim/{filename}.sav", "wb"))
    return x_sel

#Dataset sin normalizacion
x_train_sel_with_outliers, cols = SelectFeaturesTrain(x_train_with_outliers, y_train, "x_train_with_outliers")
x_test_sel_with_outliers = SelectFeaturesTest(x_test_with_outliers, y_test, "x_test_sel_with_outliers", cols)
x_train_sel_without_outliers, cols = SelectFeaturesTrain(x_train_without_outliers, y_train, "x_train_without_outliers")
x_test_sel_without_outliers = SelectFeaturesTest(x_test_with_outliers, y_test, "x_test_sel_without_outliers", cols)

#Dataset Normalizado
x_train_sel_with_outliers_standarscale, cols = SelectFeaturesTrain(x_train_with_outliers_standarscale, y_train, "x_train_with_outliers_standarscale")
x_test_sel_with_outliers_standarscale = SelectFeaturesTest(x_test_with_outliers, y_test, "x_test_sel_with_outliers_standarscale", cols)
x_train_sel_without_outliers_standarscale, cols = SelectFeaturesTrain(x_train_without_outliers_standarscale, y_train, "x_train_sel_without_outliers_standarscale")
x_test_sel_without_outliers_standarscale = SelectFeaturesTest(x_test_with_outliers, y_test, "x_test_sel_without_outliers_standarscale", cols)

#Train dataset Escalado min-Max
x_train_sel_with_outliers_mMScale, cols = SelectFeaturesTrain(x_train_with_outliers_mMScale, y_train, "x_test_with_outliers_mMScaler")
x_test_sel_with_outliers_mMScale = SelectFeaturesTest(x_test_with_outliers, y_test, "x_test_sel_with_outliers_mMScale", cols)
x_train_sel_without_outliers_mMScale, cols = SelectFeaturesTrain(x_train_without_outliers_mMScale, y_train, "x_train_without_outliers_mMScaler")
x_test_sel_without_outliers_mMScale = SelectFeaturesTest(x_test_with_outliers, y_test, "x_test_sel_without_outliers_mMScale", cols)

#Para acabar nos guardamos los datasets en un csv

def DataToCsv(dataset, filename):
    return dataset.to_csv(f"../data/processed/{filename}.csv")

DataToCsv(x_train_sel_with_outliers, "x_train_sel_with_outliers")
DataToCsv(x_test_sel_with_outliers, "x_test_sel_with_outliers")
DataToCsv(x_train_sel_without_outliers, "x_train_sel_without_outliers")
DataToCsv(x_test_sel_without_outliers, "x_test_sel_without_outliers")
DataToCsv(x_train_sel_with_outliers_standarscale, "x_train_sel_with_outliers_standarscale")
DataToCsv(x_test_sel_with_outliers_standarscale, "x_test_sel_with_outliers_standarscale")
DataToCsv(x_train_sel_without_outliers_standarscale, "x_train_sel_without_outliers_standarscale")
DataToCsv(x_test_sel_without_outliers_standarscale, "x_test_sel_without_outliers_standarscale")
DataToCsv(x_train_sel_with_outliers_mMScale, "x_train_sel_with_outliers_mMScale")
DataToCsv(x_test_sel_with_outliers_mMScale, "x_test_sel_with_outliers_mMScale")
DataToCsv(x_train_sel_without_outliers_mMScale, "x_train_sel_without_outliers_mMScale")
DataToCsv(x_test_sel_without_outliers_mMScale, "x_test_sel_without_outliers_mMScale")

########## MACHINE LEARNING ##########

traindfs = [ x_train_sel_with_outliers_standarscale, x_train_sel_without_outliers_standarscale, x_train_sel_with_outliers_mMScale, x_train_sel_without_outliers_mMScale]
testdfs = [ x_test_sel_with_outliers_standarscale, x_test_sel_without_outliers_standarscale, x_test_sel_with_outliers_mMScale, x_test_sel_without_outliers_mMScale]

def LinealRegresion (traindataset, testdataset):
    results = []
    models = []
    parameters = []

    for i in range(len(traindataset)):
        model = LinearRegression()
        traindf = traindataset[i]

        model.fit(traindf, y_train)
        y_train_predict = model.predict(traindf)
        y_test_predict = model.predict(testdataset[i])

        parameters.append({"Intercepter (a)" : float(model.intercept_), "Coeficient (b1 ~ b5)" : list(model.coef_)})
        result = {"index:" : i, "Parameters_train" : {"MSE" : float(mean_squared_error(y_train, y_train_predict)), "R2" : r2_score(y_train, y_train_predict)}, "Parameters_test" : {"MSE" : float(mean_squared_error(y_test, y_test_predict)), "R2" : r2_score(y_test, y_test_predict)}}
        results.append(result)
        models.append(model)

    with open ("../data/processed/lineal_regresion_parameters.json", "w") as j:
        json.dump( parameters, j)

    return results, models

lr_results, lr_models = LinealRegresion(traindfs, testdfs)

def LinealRegresionLasso (traindataset, testdataset):
    results = []
    models = []
    parameters = []

    for i in range(len(traindataset)):
        model = Lasso()
        traindf = traindataset[i]

        model.fit(traindf, y_train)
        y_train_predict = model.predict(traindf)
        y_test_predict = model.predict(testdataset[i])

        parameters.append({"Intercepter (a)" : float(model.intercept_), "Coeficient (b1 ~ b5)" : list(model.coef_)})
        result = {"index:" : i, "Parameters_train" : {"MSE" : float(mean_squared_error(y_train, y_train_predict)), "R2" : r2_score(y_train, y_train_predict)}, "Parameters_test" : {"MSE" : float(mean_squared_error(y_test, y_test_predict)), "R2" : r2_score(y_test, y_test_predict)}}
        results.append(result)
        models.append(model)

    with open ("../data/processed/lasso_parameters.json", "w") as j:
        json.dump( parameters, j)

    return results, models

lasso_results, lasso_models = LinealRegresionLasso(traindfs, testdfs)

def LinealRegresionRidge (traindataset, testdataset):
    results = []
    models = []
    parameters = []

    for i in range(len(traindataset)):
        model = Ridge()
        traindf = traindataset[i]

        model.fit(traindf, y_train)
        y_train_predict = model.predict(traindf)
        y_test_predict = model.predict(testdataset[i])

        parameters.append({"Intercepter (a)" : float(model.intercept_), "Coeficient (b1 ~ b5)" : list(model.coef_)})
        result = {"index:" : i, "Parameters_train" : {"MSE" : float(mean_squared_error(y_train, y_train_predict)), "R2" : r2_score(y_train, y_train_predict)}, "Parameters_test" : {"MSE" : float(mean_squared_error(y_test, y_test_predict)), "R2" : r2_score(y_test, y_test_predict)}}
        results.append(result)
        models.append(model)

    with open ("../data/processed/ridge_parameters.json", "w") as j:
        json.dump( parameters, j)

    return results, models

ridge_results, ridge_models = LinealRegresionRidge(traindfs, testdfs)

def LinealRegresionElasticCV (traindataset, testdataset):
    results = []
    models = []
    parameters = []

    for i in range(len(traindataset)):
        model = ElasticNetCV()
        traindf = traindataset[i]

        model.fit(traindf, y_train)
        y_train_predict = model.predict(traindf)
        y_test_predict = model.predict(testdataset[i])

        parameters.append({"Intercepter (a)" : float(model.intercept_), "Coeficient (b1 ~ b5)" : list(model.coef_)})
        result = {"index:" : i, "Parameters_train" : {"MSE" : float(mean_squared_error(y_train, y_train_predict)), "R2" : r2_score(y_train, y_train_predict)}, "Parameters_test" : {"MSE" : float(mean_squared_error(y_test, y_test_predict)), "R2" : r2_score(y_test, y_test_predict)}}
        results.append(result)
        models.append(model)

    with open ("../data/processed/elasticnetcv_parameters.json", "w") as j:
        json.dump( parameters, j)

    return results, models

encv_results, encv_models = LinealRegresionElasticCV(traindfs, testdfs)

#Creamos el diccionario de hyperparametros

hyperparameters = {"alpha" : np.linspace(1.0, 100.0, num = 10), "fit_intercept" : [True, False], "max_iter" : np.random.randint(low=10000, size=10), "solver" : ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga", "lbfgs"], "positive" : [True, False], "random_state" : np.random.randint(100, size=5)}

#Pasamos el modelo preentrenado con los hiperparametros (En este caso paso un grid pero podría pasar un RandomSearchCV)
grid = GridSearchCV(ridge_models[1], hyperparameters, scoring="r2")

grid.fit(x_train_sel_without_outliers_standarscale, y_train)

grid.best_params_

#Guardamos el modelo entrenado

clf = grid.best_estimator_

y_test_predict = clf.predict(x_test_sel_without_outliers_standarscale)

mse = mean_squared_error(y_test, y_test_predict)
r2 = r2_score(y_test, y_test_predict)
final_model_parameters = {"MSE" : mean_squared_error(y_test, y_test_predict), "R2" : r2_score(y_test, y_test_predict)}

print(final_model_parameters)
