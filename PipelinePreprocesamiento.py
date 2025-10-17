class PipelinePreprocesamiento:
    """
    Orquesta todo el preprocesamiento de la base de datos de audio.
    """
    def __init__(self):
        self.preprocesador = PreprocesadorAudio()
        self.admin = AdminRegistros()
        
    def procesar_base_datos(self, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Procesa todos los audios y extrae características.
        
        Returns:
            X: matriz de características (n_muestras, n_características)
            y: vector de etiquetas (n_muestras,)
        """
        archivos_etiquetados = self.admin.listar_audios()
        
        if not archivos_etiquetados:
            raise ValueError("No se encontraron archivos de audio en la ruta especificada")
        
        if verbose:
            print(f"Procesando {len(archivos_etiquetados)} archivos de audio...")
            print(f"Distribución de clases:")
            for fruta in self.admin.etiquetas.keys():
                count = sum(1 for _, etiq in archivos_etiquetados if etiq == fruta)
                print(f"  - {fruta}: {count} muestras")
        
        X = []
        y = []
        errores = []
        
        for i, (ruta_archivo, etiqueta) in enumerate(archivos_etiquetados):
            try:
                if verbose and i % 5 == 0:
                    print(f"  Procesando archivo {i+1}/{len(archivos_etiquetados)}...")
                
                # Extraer características
                caracteristicas = self.preprocesador.procesar_archivo(ruta_archivo)
                
                X.append(caracteristicas)
                y.append(self.admin.etiquetas[etiqueta])
                
            except Exception as e:
                errores.append((ruta_archivo, str(e)))
                if verbose:
                    print(f"  Error en {os.path.basename(ruta_archivo)}: {e}")
        
        if errores and verbose:
            print(f"\n⚠️ Se encontraron {len(errores)} errores durante el procesamiento")
        
        # Convertir a arrays numpy
        X = np.array(X)
        y = np.array(y)
        
        if verbose:
            print(f"\n✅ Procesamiento completado:")
            print(f"  - Forma de X: {X.shape}")
            print(f"  - Forma de y: {y.shape}")
        
        # Guardar características
        self.admin.guardar_caracteristicas(X, y)
        
        return X, y
    
    def procesar_audio_nuevo(self, ruta_audio: str) -> Tuple[np.ndarray, str]:
        """
        Procesa un audio nuevo (para predicción).
        """
        caracteristicas = self.preprocesador.procesar_archivo(ruta_audio)
        
        # Determinar etiqueta probable basada en el nombre (si existe)
        etiqueta = "desconocido"
        nombre_archivo = os.path.basename(ruta_audio).lower()
        
        for fruta in self.admin.etiquetas.keys():
            if fruta in nombre_archivo:
                etiqueta = fruta
                break
        
        return caracteristicas, etiqueta