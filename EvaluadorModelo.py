from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

class EvaluadorModelo:
    """
    Maneja la evaluación del modelo KNN usando herramientas de scikit-learn.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42):
        """
        Divide los datos usando train_test_split de sklearn.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y  # Mantiene proporciones de clases
        )
        
        self.etiquetas_nombres = ['manzana', 'banana', 'naranja', 'pera']
        self._mostrar_division()
    
    def _mostrar_division(self):
        """
        Muestra estadísticas de la división.
        """
        print("\n📊 División de datos (usando sklearn):")
        print(f"  - Entrenamiento: {len(self.y_train)} muestras")
        print(f"  - Prueba: {len(self.y_test)} muestras")
        
        print("\n  Distribución por clase:")
        for i, nombre in enumerate(self.etiquetas_nombres):
            n_train = np.sum(self.y_train == i)
            n_test = np.sum(self.y_test == i)
            print(f"    {nombre}: train={n_train}, test={n_test}")
    
    def buscar_mejor_k(self, valores_k: list = [3, 5, 7, 9, 11]) -> tuple:
        """
        Encuentra el mejor valor de k probando diferentes opciones.
        """
        print("\n🔍 Buscando mejor valor de k...")
        mejores_resultados = {}
        
        for k in valores_k:
            modelo = KNN(k=k)
            modelo.entrenar(self.X_train, self.y_train)
            
            # Evaluar en conjunto de prueba
            predicciones = modelo.predecir(self.X_test)
            precision = np.mean(predicciones == self.y_test)
            mejores_resultados[k] = precision
            
            print(f"  k={k}: {precision:.2%}")
        
        mejor_k = max(mejores_resultados.keys(), key=lambda k: mejores_resultados[k])
        mejor_precision = mejores_resultados[mejor_k]
        
        print(f"\n⭐ Mejor k: {mejor_k} (precisión: {mejor_precision:.2%})")
        
        return mejor_k, mejores_resultados
    
    def evaluar_modelo_final(self, modelo: KNN) -> dict:
        """
        Evaluación completa del modelo usando métricas de sklearn.
        """
        print("\n📈 Evaluación del modelo final:")
        
        # Predicciones
        predicciones = modelo.predecir(self.X_test)
        
        # Matriz de confusión
        matriz_conf = confusion_matrix(self.y_test, predicciones)
        
        # Reporte de clasificación
        reporte = classification_report(
            self.y_test, 
            predicciones, 
            target_names=self.etiquetas_nombres,
            output_dict=True
        )
        
        # Mostrar resultados
        print("\nMatriz de Confusión:")
        print("       manz  bana  nara  pera")
        for i, fila in enumerate(matriz_conf):
            print(f"{self.etiquetas_nombres[i]:6} {fila}")
        
        print("\nMétricas por clase:")
        for etiqueta in self.etiquetas_nombres:
            metrics = reporte[etiqueta]
            print(f"  {etiqueta}:")
            print(f"    - Precisión: {metrics['precision']:.2%}")
            print(f"    - Recall: {metrics['recall']:.2%}")
            print(f"    - F1-score: {metrics['f1-score']:.2%}")
        
        print(f"\nPrecisión general: {reporte['accuracy']:.2%}")
        
        return {
            'matriz_confusion': matriz_conf,
            'reporte': reporte,
            'precision_general': reporte['accuracy']
        }
    
    def validacion_cruzada_manual(self, k_vecinos: int = 5, n_folds: int = 5):
        """
        Validación cruzada usando nuestro KNN con StratifiedKFold de sklearn.
        """
        print(f"\n🔄 Validación cruzada ({n_folds} folds)...")
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        precisiones = []
        
        for i, (train_idx, test_idx) in enumerate(skf.split(self.X_train, self.y_train)):
            X_fold_train = self.X_train[train_idx]
            y_fold_train = self.y_train[train_idx]
            X_fold_test = self.X_train[test_idx]
            y_fold_test = self.y_train[test_idx]
            
            # Entrenar modelo
            modelo = KNN(k=k_vecinos)
            modelo.entrenar(X_fold_train, y_fold_train)
            
            # Evaluar
            predicciones = modelo.predecir(X_fold_test)
            precision = np.mean(predicciones == y_fold_test)
            precisiones.append(precision)
            
            print(f"  Fold {i+1}: {precision:.2%}")
        
        print(f"\nPrecisión media: {np.mean(precisiones):.2%} ± {np.std(precisiones):.2%}")
        
        return precisiones