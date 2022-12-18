

import numpy as np

def producto(a,b):
    y = np.zeros(b.shape[1])
  
    for i in range(b.shape[1]):
        for j in range(len(a)):
             y[i] += a[j]*b[j,i]
    return y    
 

class Perceptron:
    ''' Mi clase Perceptron con su constructor que crea los pesos del perceptron y sus funciones de 
        activacion, predecir datos, entrenar, desenso de gradiente para entrenarlo'''

    def __init__(self, capas = None):

        '''Constructor que inicialisa las capas del perceptron, ve que las dimenciones cuadren, 
           guarda las dimenciones de entrada y salida del mismo'''

        if capas is None:
            self.capas = [np.matrix([[0.0],[0.0]])]  
        else:               
            self.capas = capas                                   #* Paso mi lista de matrices que son los pesos de las capas del Perceptron
        try:
            for j in range(len(capas)-1):
                if capas[j].shape[1] + 1 != capas[j+1].shape[0]:     #* Veo que las dimenciones de las capas cuadren
                    raise Exception("ERROR, las dimenciones de las capas no coinciden")
                
            self.dim_input  = self.capas[0].shape[0] - 1
            self.dim_output = self.capas[-1].shape[1]

        except Exception as e:
            print(e)
            print("capa[%0d].shape[1] +1 = %1d " % (j,capas[j].shape[1] + 1)) 
            print("capas[%0d].shape[0] = %2d"    % (j+1,capas[j+1].shape[0]))
            return None

    def __str__(self):

        '''Imprimo las capas y las dimenciones de entrada y salida del perceptron'''

        print("dim_input  = ", self.dim_input)
        print("dim_output = ", self.dim_output)
        return f"{self.capas}"

    def funcion_activacion(self, x):
        ''' Funcion de activacion sigmoide '''
        return  (1/(1 + np.exp(-x)))           
         
    def predecir_dato(self, x, capa = None):
        '''Aplico la siguiente rcursion # Y_{j+1} = [(Y_{j})@capas[j],1], 
           donde Y_0 es mi dato de entrada y regreso Y_{n-1}@capas[n] donde n es el numero de capas'''
        

        if len(x) != self.dim_input:
            print("ERROR, La dimencion de x no cuadra")
            return None

        capas = self.capas

        if capa is None: 
            capa = len(capas)
        elif capa > len(capas):
            print("No se tienen %0d capas " % (capa))
            capa = len(capas)

        y = x 
        # print('y = ', y)
        for j in range(0, capa):
            y = y + [1]
            y = producto(y, capas[j])
            y = list(map(self.funcion_activacion,y))
        # y.pop()
        return y[0]
    
    def error(self, datos):   
        '''Calculo el error de la funcion sum (1/2)*(predicion[dato] - clase[dato])**2 sumando sobre todos los datos'''
        error = 0 
        for dato in datos:
            error += (1/2)*(self.predecir_dato(dato[0])-dato[1])**2
        return error

    def presicion(self, datos):
        '''Calculo la presicion del la funcion prediccion del perceptron '''  
        verdadero_positivo = 0
        falso_positivo     = 0
        pos                = 0
        neg                = 0
        for dato in datos:
            prediccion = self.predecir_dato(dato[0])
            if (prediccion > 0.5) and (dato[1] == 1):   #dato[1] es la clase
                verdadero_positivo += 1
            elif (prediccion > 0.5) and (dato[1] == 0):
                falso_positivo += 1
            if dato[1] == 0:
                pos +=1
            else:
                neg += 1
        if pos + neg == 0:
            print("no mandaste datos")
            return None
        try:
            pres = verdadero_positivo/(verdadero_positivo + falso_positivo)
        except ZeroDivisionError:
            return neg/(pos + neg)
        return pres

    def exactitud(self, datos):
        '''Calculo la exactitud de la funcion predicion de mi perceptron'''
        ver_pos = 0  
        fal_pos = 0 
        ver_neg = 0 
        fal_neg = 0 
        
        for dato in datos:
            predicion = self.predecir_dato(dato[0])

            if predicion > 0.5:
                if dato[1] == 1:
                    ver_pos += 1
                else:
                    fal_pos += 1
            else:
                if dato[1] == 0:
                    ver_neg += 1
                else:
                    fal_pos += 1

        return (ver_neg + ver_pos)/(ver_neg + ver_pos + fal_pos + fal_neg)      

    def fit(self, datos, epocas = None, tam_paso = None):           #todo entrenamiento para cuando tenemos 1 capa, ignoro la primera y solo entreno la segunda 
        '''Funcion de entremamiento del perceptron'''
        
        # num_datos =  len(datos)
        # tam_lote  =  min(len(datos)/10, 10)
        # for epoca in range(epocas):
        

        if tam_paso is None:
            tam_paso = 0.1

        gradiente_mat0 = np.zeros(self.capas[-1].shape[0]*self.capas[-1].shape[1])
        gradiente_mat0 = gradiente_mat0.reshape(self.capas[-1].shape)

        for dato in datos:
            predicion = self.predecir_dato(dato[0])
            y         = dato[0] + [1]
            W         = self.capas[-1]
            for i in range(gradiente_mat0.shape[0]):
                for j in range(gradiente_mat0.shape[1]):
                    gradiente_mat0[i,j]  = 2*(predicion-dato[1])
                    gradiente_mat0[i,j] *= predicion*(1-predicion)
                    gradiente_mat0[i,j] *= y[i]
                    gradiente_mat0[i,j] *= W[i,0]
                    gradiente_mat0[i,j] *= -tam_paso
        
            M = self.capas[-1] #todo
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    M[i,j] += gradiente_mat0[i,j]




