import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_transfer_learning_diagram():
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Configuración de estilo
    box_props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', linewidth=1.5)
    arrow_props = dict(arrowstyle='->', connectionstyle='arc3', linewidth=2, color='gray')
    
    # --- BLOQUE 1: MODELO PRE-ENTRENADO (GOOGLE) ---
    ax.add_patch(patches.Rectangle((0.5, 3), 3, 2, facecolor='#e1f5fe', edgecolor='#01579b', linewidth=2))
    ax.text(2, 4.3, "Modelo Base\n(MobileNetV3)", ha='center', va='center', weight='bold', fontsize=12)
    ax.text(2, 3.6, "Entrenado con\nMillones de Imágenes\n(Conocimiento General)", ha='center', va='center', fontsize=10)

    # Flecha de transferencia
    ax.annotate('Transferencia de\nConocimiento', xy=(5.5, 3.5), xytext=(3.5, 3.5),
                arrowprops=dict(facecolor='black', shrink=0.05), ha='center', va='bottom', fontsize=10, weight='italic')

    # --- BLOQUE 2: TU PROTOTIPO (AJUSTE FINO) ---
    # Capas Congeladas/Heredadas
    ax.add_patch(patches.Rectangle((6, 4), 4, 1, facecolor='#f5f5f5', edgecolor='#757575', linewidth=2, linestyle='--'))
    ax.text(8, 4.5, "Capas de Extracción de Características\n(Bordes, Texturas, Formas)", ha='center', va='center', fontsize=9)

    # Capas de Clasificación (Fine-Tuning)
    ax.add_patch(patches.Rectangle((6, 2), 4, 1.5, facecolor='#fff3e0', edgecolor='#e65100', linewidth=2))
    ax.text(8, 2.75, "Capas de Clasificación Final\n(Ajuste Fino / Fine-Tuning)", ha='center', va='center', weight='bold', fontsize=11)
    ax.text(8, 2.2, "Entrenado con Espectrogramas\nde Elliot (N=1)", ha='center', va='center', fontsize=9, color='#e65100')

    # --- ENTRADA Y SALIDA ---
    # Entrada
    ax.text(0.5, 1.5, "Entrada: Espectrograma de Mel", ha='left', va='center', fontsize=10, style='italic')
    ax.annotate('', xy=(6, 2.75), xytext=(4, 1.5), arrowprops=arrow_props)
    
    # Salida
    ax.text(10.5, 2.75, "Predicción:\nGASES / SUEÑO", ha='left', va='center', weight='bold', color='green', fontsize=12)
    ax.annotate('', xy=(11, 2.75), xytext=(10, 2.75), arrowprops=arrow_props)

    # Ajustes finales
    ax.set_xlim(0, 13)
    ax.set_ylim(1, 6)
    ax.axis('off')
    plt.title("Esquema de Aprendizaje por Transferencia (Transfer Learning)", fontsize=14, weight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig("esquema_transfer_learning.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    create_transfer_learning_diagram()