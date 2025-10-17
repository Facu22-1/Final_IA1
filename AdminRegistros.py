import os
import numpy as np
import pickle
from typing import List, Tuple, Dict

class AdminRegistros:
    """
    Administra la lectura/escritura de archivos y datos procesados.
    """
    def __init__(self, ruta_base_datos: str = "./datos"):
        self.ruta_base_datos = ruta_base_datos
        self.ruta_audios = os.path.join(ruta_base_datos, "audios")
        self.ruta_procesados = os.path.join(ruta_base_datos, "procesados")
        
        # Crear directorios si no existen
        os.makedirs(self.ruta_procesados, exist_ok=True)
        
        # Mapeo de etiquetas
        self.etiquetas = {
            "manzana": 0,
            "banana": 1,
            "naranja": 2,
            "pera": 3
        }
        self.etiquetas_inv = {v: k for k, v in self.etiquetas.items()}
    
    def listar_audios(self) -> List[Tuple[str, str]]:
        """
        Lista todos los archivos de audio con sus etiquetas.
        Retorna lista de tuplas (ruta_archivo, etiqueta).
        """
        archivos_etiquetados = []
        
        for archivo in os.listdir(self.ruta_audios):
            if archivo.endswith(('.wav', '.mp3', '.m4a')):
                # Extraer etiqueta del nombre del archivo
                nombre_sin_ext = os.path.splitext(archivo)[0]
                
                # Buscar qué fruta está en el nombre
                for fruta in self.etiquetas.keys():
                    if fruta in nombre_sin_ext.lower():
                        ruta_completa = os.path.join(self.ruta_audios, archivo)
                        archivos_etiquetados.append((ruta_completa, fruta))
                        break
        
        return archivos_etiquetados
    
    def guardar_caracteristicas(self, X: np.ndarray, y: np.ndarray, 
                               nombre_archivo: str = "caracteristicas_audio.pkl"):
        """
        Guarda las características extraídas y sus etiquetas.
        """
        ruta_guardado = os.path.join(self.ruta_procesados, nombre_archivo)
        
        datos = {
            'X': X,
            'y': y,
            'etiquetas': self.etiquetas,
            'n_caracteristicas': X.shape[1] if X.ndim > 1 else 1
        }
        
        with open(ruta_guardado, 'wb') as f:
            pickle.dump(datos, f)
        
        print(f"Características guardadas en: {ruta_guardado}")
        print(f"- Muestras: {X.shape[0]}")
        print(f"- Características por muestra: {X.shape[1]}")
    
    def cargar_caracteristicas(self, nombre_archivo: str = "caracteristicas_audio.pkl"):
        """
        Carga características previamente guardadas.
        """
        ruta_archivo = os.path.join(self.ruta_procesados, nombre_archivo)
        
        with open(ruta_archivo, 'rb') as f:
            datos = pickle.load(f)
        
        return datos['X'], datos['y']