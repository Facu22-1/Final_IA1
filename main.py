# Todo junto en un script principal
def main():
    # 1. Preparar datos
    pipeline = PipelinePreprocesamiento()
    X, y = pipeline.procesar_base_datos()
    
    # 2. Dividir y evaluar
    evaluador = EvaluadorModelo(X, y)
    mejor_k, _ = evaluador.buscar_mejor_k()
    
    # 3. Entrenar modelo final
    modelo = KNN(k=mejor_k)
    modelo.entrenar(evaluador.X_train, evaluador.y_train)
    
    # 4. Evaluar
    evaluador.evaluar_modelo_final(modelo)
    
    return modelo, pipeline