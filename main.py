
import numpy as np

def producto(a,b):
    y = np.zeros(b.shape[1])
  
    for i in range(b.shape[1]):
        for j in range(len(a)):
             y[i] += a[j]*b[j,i]
    return y    
 

class Perceptron:
    ''' 
    Mi clase Perceptron con su constructor que crea los pesos del perceptron y sus funciones de 
    activacion, predecir datos, entrenar, desenso de gradiente para entrenarlo
    '''

    def __init__(self, capas = None):
        '''
        Constructor que inicialisa las capas del perceptron, ve que las dimenciones cuadren, 
        guarda las dimenciones de entrada y salida del mismo
        '''

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
        '''
        Imprimo las capas y las dimenciones de entrada y salida del perceptron
        '''

        print("dim_input  = ", self.dim_input)
        print("dim_output = ", self.dim_output)
        return f"{self.capas}"

    def funcion_activacion(self, x):
        ''' 
        Funcion de activacion sigmoide 
        '''
        aux = (1/(1 + np.exp(-x)))
        return aux             
         
    def predecir_dato(self, x, capa = None):
        '''
        Aplico la siguiente rcursion # Y_{j+1} = [(Y_{j})@capas[j],1], 
        donde Y_0 es mi dato de entrada y regreso Y_{n-1}@capas[n] donde n es el numero de capas
        '''
        
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
        for j in range(0, capa):
            y = y + [1]
            y = producto(y, capas[j])
            y = list(map(self.funcion_activacion,y))

        #todo checar 
        if capa != len(capas):
            return y + [1] 

        return y


    #todo def grad_dato(self, dato, i, j, k):
    #todo     pass


    def des_grad(self, datos, tam_paso): #todo falta optimizar
        '''
        Desenso de gradiente con tamaño de paso fijo y no estocastico
        '''

        capas = self.capas
        num_capas = len(capas)
        k = num_capas 

        for capa in reversed(capas):
            k -= 1 
            for i in range(capa.shape[0]):
                for j in range(capa.shape[1]):
                    grad = 0
                    for dato in datos:
                        grad_dato = 0
                        predicion = self.predecir_dato(dato[0])[0]

                        if k == num_capas-1:
                            grad_dato += predicion - dato[1]                                      
                            grad_dato *= predicion*(1-predicion)                                  
                            grad_dato *= self.predecir_dato(dato[0], k)[i]                        
                            grad_dato *= -tam_paso

                        elif k == num_capas - 2:                    
                            grad_dato += predicion - dato[1]                                                                
                            grad_dato *= predicion*(1-predicion)
                            grad_dato *= self.capas[num_capas-1][0,0]                               
                            x_0        = self.predecir_dato(dato[0], num_capas-1)[0]
                            grad_dato *= x_0*(1-x_0)
                            aux        = self.predecir_dato(dato[0], num_capas-2)
                            grad_dato *= aux[i]
                            grad_dato *= -tam_paso

                        else:
                            grad_dato += predicion - dato[1] 
                            k = num_capas-k
                            for s in range(k+1+2, num_capas): #todo n o n-1 falta probar cuando tiene mas capas                               
                                x_0s  = self.predecir_dato(dato[0], s)[0]
                                grad_dato *= x_0s*(1-x_0s)
                                grad_dato *= capas[s-1][0,0]

                            x_0k2   = self.predecir_dato(dato[0], k+2)[0]
                            x_0k1   = self.predecir_dato(dato[0], k+1)[j]
                            grad_dato   *= x_0k2*(1- x_0k2)
                            grad_dato   *= capas[k+1][j,0]
                            grad_dato   *= x_0k1*(1- x_0k1)
                            grad_dato   *= self.predecir_dato(dato[0], k)[i]

                        grad += grad_dato 
                    capa[i,j] += grad


    def fit(self, datos, epocas = None, tam_paso = None, historial = False):           #todo entrenamiento para cuando tenemos 1 capa, ignoro la primera y solo entreno la segunda 
        '''
        Funcion de entremamiento del perceptron, la cual se basa en la funcion de desenso de gradiente
        '''

        if tam_paso is None:
            tam_paso = 0.1
        #todo un for de conjutos de datos i.e datos = subsetdatos
        errores = self.des_grad(datos, tam_paso) 
        if historial is True:
            errores = [self.error(datos)]
            return errores

    def error(self, datos):   
        '''
        Calculo el error de la funcion sum (1/2)*(predicion[dato] - clase[dato])**2 sumando sobre todos los datos
        '''

        error = 0 
        for dato in datos:
            error += (1/2)*(self.predecir_dato(dato[0])[0]-dato[1])**2
        return error
        
    def presicion(self, datos):
        '''
        Calculo la presicion del la funcion prediccion del perceptron 
        '''  

        verdadero_positivo = 0
        falso_positivo     = 0
        pos                = 0
        neg                = 0
        for dato in datos:
            prediccion = self.predecir_dato(dato[0])[0]
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
        '''
        Calculo la exactitud de la funcion predicion de mi perceptron
        '''
        
        ver_pos = 0  
        fal_pos = 0 
        ver_neg = 0 
        fal_neg = 0 
        
        for dato in datos:
            predicion = self.predecir_dato(dato[0])[0]

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

