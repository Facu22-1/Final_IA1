# Ejemplo de uso
if __name__ == "__main__":
    # Procesar toda la base de datos
    pipeline = PipelinePreprocesamiento()
    X, y = pipeline.procesar_base_datos(verbose=True)
    
    # Los datos están listos para KNN
    print(f"\nDatos listos para entrenamiento:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")