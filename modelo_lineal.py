import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

COLUMNS=['SalePrice', 'OverallQual', '1stFlrSF', 'TotRmsAbvGrd', 'YearBuilt', 'LotFrontage']

class ModeloLineal():
    
    b0 = -5
    b1 = 8
    
    def __init__(self, data_path: str, training_size: float):
        self.data = np.load(data_path)
        n = len(self.data)
        self.n_trainning = int(n * training_size)
        
        self.data_df = pd.DataFrame(self.data, columns=COLUMNS)
        self.training_df = pd.DataFrame(self.data[0:self.n_trainning], columns=COLUMNS)
        self.testing_df = pd.DataFrame(self.data[self.n_trainning:n], columns=COLUMNS)
        
    
    def generar_dataframe(self):
        pass
    
    def mostrar_medias(self):
        for col in self.training_df.columns:
            print(f'Media {col}: ', self.training_df[col].mean())
            
    def mostrar_maximos(self):
        for col in self.training_df.columns:
            print(f'Máximo {col}: ', self.training_df[col].max())
    
    def mostrar_minimos(self):
        for col in self.training_df.columns:
            print(f'Mínimo {col}: ', self.training_df[col].min())
        
    def mostrar_rangos(self):
        for col in self.training_df.columns:
            print(f'Rango {col}: ', self.training_df[col].max() - self.training_df[col].min())
    
    def mostrar_desviaciones(self):
        for col in self.training_df.columns:
            print(f'Desviación Estandar {col}: ', self.training_df[col].std())
    
    def mostar_histogramas(self):
        for col in self.training_df.columns:
            fig, axs = plt.subplots()
            sns.histplot(self.training_df[col], kde=True, ax=axs)
            
    def mostar_correlaciones(self):
        for col in self.training_df.columns:
            if col != 'SalePrice':
                fig, axs = plt.subplots()
                axs.scatter(self.training_df[col], self.training_df['SalePrice'])
                # plt.title(np.corrcoef(training_df[col], training_df['SalePrice'])[0][1])
                axs.set_title(f'{col} : ' + str(np.corrcoef(self.training_df[col], self.training_df['SalePrice'])[0][1]))
                
    def get_trainning_arrays(self, variable_x):
        variables_x = self.training_df[variable_x]
        variables_y = self.training_df['SalePrice']
        
        return np.array(variables_x), np.array(variables_y)
    
    def get_testing_arrays(self, variable_x):
        variables_x = self.testing_df[variable_x]
        variables_y = self.testing_df['SalePrice']
        
        return np.array(variables_x), np.array(variables_y)
    
    def get_muestra_x(self, variable_x):
        variables_x = np.array(self.testing_df[variable_x])
        
        n = len(variables_x)
        size = random.uniform(0, 1)
        n_muestra = int(n * size)
        
        return variables_x[0:n_muestra]
    
    def set_betas(self, b0, b1):
        self.b0 = b0
        self.b1 = b1
        
    def entrar_modelo(self, vector_x: np.array, vector_y: np.array, epochs: int, imprimir_error_cada: int, alpha: float):
        constante_1 = np.empty(self.n_trainning)
        constante_1.fill(1)
        observacion = np.column_stack((vector_x, constante_1))
        b0 = self.b0
        b1 = self.b1
        parametros = np.array([b1, b0])
        y_real = vector_y
        modelo = []
        error_array = []
        
        for i in range(epochs):
            # Se calcula Y aproximada, el error y los gradientes de B0 y B1
            y_calculada = np.dot(observacion, parametros)
            error = np.mean(np.power(y_real - y_calculada, 2)) / 2
            
            gradiente_b0 = np.mean(y_real - y_calculada)
            gradiente_b1 = np.mean(np.dot(y_real - y_calculada, observacion))
            
            b0 = b0 - (alpha*gradiente_b0)
            b1 = b1 - (alpha*gradiente_b1)
            parametros = np.array([b1, b0])
            
            # Se almacena el modelo de la iteracion correspondiente
            modelo.append([b0, b1])
            error_array.append(error)
            
            # Se imprime el error si cumple la condicion
            if (i + 1) % imprimir_error_cada == 0:
                print(f'El error es de: {error}')
                
        self.modelo = modelo
        self.error_array = error_array
        return modelo, error_array
    
    def graficar_error(self):
        iteraciones = np.array([i for i, item in enumerate(self.error_array)])
        errores = np.array(self.error_array)
        
        plt.plot(iteraciones, errores)
        plt.xlabel('Iteraciones')
        plt.ylabel('Error')
        plt.show()
        
    def graficar_modelo(self, n: int):
        modelo_np = np.array(self.modelo)
        for i, row in enumerate(modelo_np):
            if (i+1) % n == 0:
                x = np.linspace(-5,10,10)
                y = row[1] * x + row[0]
                plt.plot(x, y, label=(i+1))
        plt.title('Modelo')
        plt.xlabel('x', color='#1C2833')
        plt.ylabel('y', color='#1C2833')
        plt.legend(loc='best')
        plt.grid()
        plt.show()
        
    def entrar_con_scikit(self, training_x: np.array, training_y: np.array, testing_x: np.array, testing_y: np.array):
        X_train = training_x.reshape(-1, 1)
        y_train = training_y.reshape(-1, 1)
        X_test = testing_x.reshape(-1, 1)
        y_test = testing_y.reshape(-1, 1)
        
        regr = LinearRegression()
        regr.fit(X_train, y_train)
        
        error = regr.score(X_test, y_test)
        b0 = regr.intercept_.reshape(1)[0]
        b1 = regr.coef_.reshape(1)[0]
        
        return [b0, b1], error
    
    def estimar_modelos(self, betas_manual: list, betas_scikit: list, vector_x: np.array):
        constante_1 = np.empty(len(vector_x))
        constante_1.fill(1)
        
        observacion = np.column_stack((vector_x, constante_1))
        parametros_manual = np.array(betas_manual)
        parametros_scikit = np.array(betas_scikit)
        
        y_manual = np.dot(observacion, parametros_manual)
        y_scikit = np.dot(observacion, parametros_scikit)
        y_promedio = (y_manual + y_scikit) / 2
        
        return y_manual, y_scikit, y_promedio
    
    def graficar_estimacion_modelos(self, vector_x: np.array, y_manual: np.array, y_scikit: np.array, y_promedio: np.array):
        plt.plot(vector_x, y_manual, label='Modelo Manual')
        plt.plot(vector_x, y_scikit, label='Modelo Scikit')
        plt.plot(vector_x, y_promedio, label='Modelo Promedio')
        plt.title('Estimación Modelos')
        plt.xlabel('x', color='#1C2833')
        plt.ylabel('y', color='#1C2833')
        plt.legend(loc='best')
        plt.grid()
        plt.show()
        