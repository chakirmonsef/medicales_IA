# ============================================================
#  PROJET : Analyse d'Images Médicales avec l'IA
#  Niveau  : Lycée
#  Auteur  : Projet pédagogique IA & Médecine
# ============================================================

# ---- ÉTAPE 1 : Importer les bibliothèques nécessaires ----
# NumPy : pour manipuler des tableaux de nombres (les images sont des tableaux)
import numpy as np

# Matplotlib : pour afficher les images
import matplotlib.pyplot as plt

# TensorFlow / Keras : pour créer et entraîner le réseau de neurones
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# OS : pour gérer les fichiers
import os

print("✅ Bibliothèques chargées avec succès !")
print(f"   Version TensorFlow : {tf.__version__}")


# ============================================================
#  ÉTAPE 2 : Créer des données simulées (images médicales)
# ============================================================
# Dans la réalité, on utiliserait de vraies IRM ou scanners.
# Ici, on SIMULE deux types d'images :
#   - Classe 0 = image "normale" (pas de tumeur)
#   - Classe 1 = image "anormale" (présence d'une anomalie)

def creer_image_normale(taille=64):
    """
    Crée une image simulant un cerveau normal.
    L'image est grise avec de légères variations aléatoires.
    """
    image = np.random.normal(loc=100, scale=15, size=(taille, taille))
    # On s'assure que les valeurs restent entre 0 et 255 (niveaux de gris)
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def creer_image_anormale(taille=64):
    """
    Crée une image simulant un cerveau avec anomalie.
    Une zone plus claire (la 'tumeur') apparaît au centre.
    """
    image = np.random.normal(loc=100, scale=15, size=(taille, taille))

    # On ajoute une zone brillante au centre pour simuler une tumeur
    centre = taille // 2
    rayon = taille // 8
    for i in range(taille):
        for j in range(taille):
            # Calcul de la distance au centre
            distance = np.sqrt((i - centre)**2 + (j - centre)**2)
            if distance < rayon:
                image[i][j] += 80  # Zone plus lumineuse

    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


# ---- Générer le jeu de données ----
NB_IMAGES = 200  # 200 images par classe
TAILLE = 64      # Images de 64x64 pixels

print("\n📊 Génération des données d'entraînement...")

images_normales  = [creer_image_normale(TAILLE)  for _ in range(NB_IMAGES)]
images_anormales = [creer_image_anormale(TAILLE) for _ in range(NB_IMAGES)]

# Assembler toutes les images dans un tableau NumPy
X = np.array(images_normales + images_anormales, dtype=np.float32)
# Les labels : 0 = normal, 1 = anomalie
y = np.array([0] * NB_IMAGES + [1] * NB_IMAGES, dtype=np.float32)

# Normaliser les pixels entre 0 et 1 (important pour les réseaux de neurones)
X = X / 255.0

# Ajouter une dimension pour indiquer que c'est une image en niveaux de gris
# Format attendu par Keras : (nb_images, hauteur, largeur, canaux)
X = X[..., np.newaxis]

print(f"   Forme du tableau d'images : {X.shape}")
print(f"   Nombre de labels          : {len(y)}")


# ============================================================
#  ÉTAPE 3 : Diviser les données en ensemble d'entraînement
#            et ensemble de test
# ============================================================
# On mélange les données pour éviter les biais
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# 80 % des images servent à entraîner, 20 % à tester

print(f"\n🔀 Séparation des données :")
print(f"   Entraînement : {len(X_train)} images")
print(f"   Test         : {len(X_test)} images")


# ============================================================
#  ÉTAPE 4 : Construire le réseau de neurones convolutif (CNN)
# ============================================================
# Un CNN (Convolutional Neural Network) est parfait pour analyser
# des images. Il détecte automatiquement les formes et motifs.

print("\n🧠 Construction du réseau de neurones...")

modele = keras.Sequential([
    # --- Couche 1 : Détection de formes simples ---
    # 32 filtres de 3x3 pixels pour détecter contours et textures
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(TAILLE, TAILLE, 1)),
    # Réduit la taille de l'image de moitié (plus rapide à traiter)
    layers.MaxPooling2D((2, 2)),

    # --- Couche 2 : Détection de formes plus complexes ---
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # --- Couche 3 : Formes encore plus complexes ---
    layers.Conv2D(64, (3, 3), activation='relu'),

    # --- Aplatir les données 2D en 1D pour la classification ---
    layers.Flatten(),

    # --- Couche dense : prise de décision ---
    layers.Dense(64, activation='relu'),

    # --- Dropout : évite que le réseau "apprenne par coeur" ---
    layers.Dropout(0.5),

    # --- Sortie finale : 1 neurone → probabilité d'anomalie ---
    layers.Dense(1, activation='sigmoid')
])

# Afficher le résumé du modèle
modele.summary()


# ============================================================
#  ÉTAPE 5 : Compiler et entraîner le modèle
# ============================================================
# Compiler = choisir comment le réseau va apprendre
modele.compile(
    optimizer='adam',         # Algorithme d'optimisation
    loss='binary_crossentropy',  # Mesure de l'erreur
    metrics=['accuracy']      # On mesure la précision
)

print("\n🏋️ Entraînement du réseau de neurones...")
print("   (Chaque epoch = le réseau regarde toutes les images une fois)")

historique = modele.fit(
    X_train, y_train,
    epochs=10,           # 10 passages sur les données
    batch_size=32,       # Le réseau traite 32 images à la fois
    validation_split=0.1, # 10% des données d'entraînement pour valider
    verbose=1
)

print("\n✅ Entraînement terminé !")


# ============================================================
#  ÉTAPE 6 : Évaluer les performances du modèle
# ============================================================
print("\n📈 Évaluation sur les données de test...")

perte, precision = modele.evaluate(X_test, y_test, verbose=0)
print(f"   Précision du modèle : {precision * 100:.1f}%")
print(f"   Erreur (loss)       : {perte:.4f}")


# ============================================================
#  ÉTAPE 7 : Faire une prédiction sur une nouvelle image
# ============================================================
print("\n🔬 Test de prédiction sur une nouvelle image...")

# Créer une image de test inconnue (simulée)
nouvelle_image_normale   = creer_image_normale(TAILLE)
nouvelle_image_anormale  = creer_image_anormale(TAILLE)

def predire(image_brute):
    """
    Analyse une image médicale et retourne le diagnostic.
    """
    # Préparer l'image pour le réseau (même format que les données d'entraînement)
    img = image_brute.astype(np.float32) / 255.0
    img = img[np.newaxis, ..., np.newaxis]  # Ajouter les dimensions manquantes

    # Prédiction (valeur entre 0 et 1)
    probabilite = modele.predict(img, verbose=0)[0][0]

    diagnostic = "⚠️  ANOMALIE DÉTECTÉE" if probabilite > 0.5 else "✅ IMAGE NORMALE"
    confiance  = probabilite if probabilite > 0.5 else 1 - probabilite

    print(f"   Probabilité brute : {probabilite:.3f}")
    print(f"   Diagnostic        : {diagnostic}")
    print(f"   Confiance         : {confiance * 100:.1f}%")
    return probabilite

print("\n--- Image 1 (simulée normale) ---")
p1 = predire(nouvelle_image_normale)

print("\n--- Image 2 (simulée anormale) ---")
p2 = predire(nouvelle_image_anormale)


# ============================================================
#  ÉTAPE 8 : Visualiser les résultats
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
fig.suptitle("Analyse d'Images Médicales par Intelligence Artificielle",
             fontsize=14, fontweight='bold', color='#1a3c5e')

# --- Ligne 1 : Courbes d'apprentissage ---
axes[0, 0].plot(historique.history['accuracy'],    color='#2196F3', label='Entraînement')
axes[0, 0].plot(historique.history['val_accuracy'], color='#FF5722', label='Validation')
axes[0, 0].set_title('Précision au fil du temps', fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Précision')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(historique.history['loss'],     color='#4CAF50', label='Entraînement')
axes[0, 1].plot(historique.history['val_loss'], color='#FF9800', label='Validation')
axes[0, 1].set_title("Erreur (Loss) au fil du temps", fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Erreur')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# --- Résumé des performances ---
axes[0, 2].axis('off')
texte = (
    f"📊 RÉSULTATS DU MODÈLE\n\n"
    f"Précision : {precision * 100:.1f}%\n"
    f"Erreur    : {perte:.4f}\n\n"
    f"Images d'entraînement : {len(X_train)}\n"
    f"Images de test        : {len(X_test)}\n\n"
    f"Architecture : CNN\n"
    f"Epochs       : 10"
)
axes[0, 2].text(0.1, 0.5, texte, transform=axes[0, 2].transAxes,
                fontsize=11, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='#e8f4fd', alpha=0.8))

# --- Ligne 2 : Images analysées ---
axes[1, 0].imshow(nouvelle_image_normale, cmap='gray')
axes[1, 0].set_title('Image normale (test)', fontweight='bold', color='green')
axes[1, 0].axis('off')

axes[1, 1].imshow(nouvelle_image_anormale, cmap='gray')
axes[1, 1].set_title('Image anormale (test)', fontweight='bold', color='red')
axes[1, 1].axis('off')

# --- Barre de probabilité ---
categories = ['Normale', 'Anormale']
probs_img1 = [1 - p1, p1]
probs_img2 = [1 - p2, p2]
x = np.arange(len(categories))
largeur = 0.35

bars1 = axes[1, 2].bar(x - largeur/2, probs_img1, largeur, label='Image normale',
                        color=['#4CAF50', '#FFCDD2'])
bars2 = axes[1, 2].bar(x + largeur/2, probs_img2, largeur, label='Image anormale',
                        color=['#C8E6C9', '#F44336'])
axes[1, 2].set_title('Probabilités prédites', fontweight='bold')
axes[1, 2].set_xticks(x)
axes[1, 2].set_xticklabels(categories)
axes[1, 2].set_ylabel('Probabilité')
axes[1, 2].set_ylim(0, 1)
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3, axis='y')
axes[1, 2].axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Seuil (0.5)')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/resultats_analyse_medicale.png',
            dpi=150, bbox_inches='tight')
print("\n💾 Graphique sauvegardé : resultats_analyse_medicale.png")
plt.show()

print("\n" + "="*55)
print("  FIN DU PROGRAMME - Analyse d'Images Médicales par IA")
print("="*55)