import numpy as np
import librosa
import os
from typing import Optional, Tuple

class PreprocesadorAudio:
    def __init__(self, frecuencia_muestreo=22050):
        self.frecuencia_muestreo = frecuencia_muestreo
        self.duracion_objetivo = 1.5  # segundos (suficiente para una palabra)
        self.umbral_energia = 0.02  # umbral para detectar inicio de voz
        self.n_mfcc = 13  # Número de coeficientes MFCC
        self.n_mels = 40  # Número de filtros mel
        
    def detectar_segmento_voz(self, audio: np.ndarray) -> Tuple[int, int]:
        """
        Detecta dónde empieza y termina la voz en el audio.
        Retorna índices de inicio y fin.
        """
        # Calcular energía en ventanas cortas
        tamano_ventana = int(0.01 * self.frecuencia_muestreo)  # ventanas de 10ms
        energia = np.array([
            np.sum(audio[i:i+tamano_ventana]**2) 
            for i in range(0, len(audio)-tamano_ventana, tamano_ventana//2)
        ])
        
        # Normalizar energía
        if energia.max() > 0:
            energia = energia / energia.max()
        
        # Encontrar donde la energía supera el umbral
        indices_activos = np.where(energia > self.umbral_energia)[0]
        
        if len(indices_activos) == 0:
            # No se detectó voz
            return 0, len(audio)
        
        # Convertir índices de ventanas a índices de audio
        inicio = indices_activos[0] * (tamano_ventana // 2)
        fin = indices_activos[-1] * (tamano_ventana // 2) + tamano_ventana
        
        # Agregar pequeño margen
        margen = int(0.05 * self.frecuencia_muestreo)  # 50ms de margen
        inicio = max(0, inicio - margen)
        fin = min(len(audio), fin + margen)
        
        return inicio, fin
    
    def cargar_audio(self, ruta_archivo: str, detectar_voz: bool = True) -> np.ndarray:
        """
        Carga audio y opcionalmente centra el segmento de voz.
        
        Args:
            ruta_archivo: Ruta del archivo de audio
            detectar_voz: Si True, centra el audio en el segmento de voz detectado
        """
        audio, sr = librosa.load(ruta_archivo, sr=self.frecuencia_muestreo)
        
        if detectar_voz:
            inicio, fin = self.detectar_segmento_voz(audio)
            audio = audio[inicio:fin]
        
        # Ahora sí normalizar duración
        longitud_objetivo = int(self.duracion_objetivo * self.frecuencia_muestreo)
        
        if len(audio) < longitud_objetivo:
            # Centrar el audio con padding en ambos lados
            padding_total = longitud_objetivo - len(audio)
            padding_inicio = padding_total // 2
            padding_fin = padding_total - padding_inicio
            audio = np.pad(audio, (padding_inicio, padding_fin))
        else:
            # Si es más largo, tomar el centro
            centro = len(audio) // 2
            mitad_objetivo = longitud_objetivo // 2
            audio = audio[centro - mitad_objetivo : centro + mitad_objetivo]
            
        return audio
    
    def extraer_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """
        Extrae coeficientes MFCC del audio.
        Retorna media y desviación estándar de cada coeficiente.
        """
        mfccs = librosa.feature.mfcc(
            y=audio, 
            sr=self.frecuencia_muestreo,
            n_mfcc=self.n_mfcc,
            n_mels=self.n_mels
        )
        
        # Calcular estadísticas por cada coeficiente
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        
        return np.concatenate([mfcc_mean, mfcc_std])
    
    def extraer_caracteristicas_temporales(self, audio: np.ndarray) -> np.ndarray:
        """
        Extrae características del dominio temporal.
        """
        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)
        
        # Energía RMS
        rms = librosa.feature.rms(y=audio)[0]
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)
        
        return np.array([zcr_mean, zcr_std, rms_mean, rms_std])
    
    def extraer_caracteristicas_espectrales(self, audio: np.ndarray) -> np.ndarray:
        """
        Extrae características del dominio espectral.
        """
        # Centroide espectral
        centroide = librosa.feature.spectral_centroid(y=audio, sr=self.frecuencia_muestreo)[0]
        centroide_mean = np.mean(centroide)
        centroide_std = np.std(centroide)
        
        # Ancho de banda espectral
        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.frecuencia_muestreo)[0]
        bandwidth_mean = np.mean(bandwidth)
        bandwidth_std = np.std(bandwidth)
        
        # Roll-off espectral
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.frecuencia_muestreo)[0]
        rolloff_mean = np.mean(rolloff)
        rolloff_std = np.std(rolloff)
        
        return np.array([
            centroide_mean, centroide_std,
            bandwidth_mean, bandwidth_std,
            rolloff_mean, rolloff_std
        ])
    
    def extraer_vector_caracteristicas(self, audio: np.ndarray) -> np.ndarray:
        """
        Extrae el vector completo de características del audio.
        Este es el vector que usará KNN para clasificar.
        """
        mfcc_features = self.extraer_mfcc(audio)
        temporal_features = self.extraer_caracteristicas_temporales(audio)
        spectral_features = self.extraer_caracteristicas_espectrales(audio)
        
        # Concatenar todas las características
        vector_final = np.concatenate([
            mfcc_features,      # 26 características (13 medias + 13 std)
            temporal_features,  # 4 características
            spectral_features   # 6 características
        ])
        
        return vector_final  # Total: 36 características
    
    def procesar_archivo(self, ruta_archivo: str) -> np.ndarray:
        """
        Método principal: carga audio y extrae características.
        """
        audio = self.cargar_audio(ruta_archivo, detectar_voz=True)
        return self.extraer_vector_caracteristicas(audio)