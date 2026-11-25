"""Detección de grietas en imágenes de pavimento con OpenCV (visión clásica, sin entrenamiento)."""
import cv2
import numpy as np

# Umbrales de severidad según el % de área agrietada (ajustables)
UMBRAL_MEDIA = 0.8
UMBRAL_ALTA = 3.0


def clasificar_severidad(porcentaje):
    if porcentaje < UMBRAL_MEDIA:
        return "Baja"
    if porcentaje < UMBRAL_ALTA:
        return "Media"
    return "Alta"


def analizar_imagen(contenido: bytes, ancho_max: int = 900):
    """Recibe los bytes de una imagen y devuelve un diccionario con el resultado."""
    arr = np.frombuffer(contenido, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("No se pudo leer la imagen. Verifique que sea PNG o JPG válido.")

    # Redimensionar para uniformar el análisis
    h, w = img.shape[:2]
    if w > ancho_max:
        img = cv2.resize(img, (ancho_max, int(h * ancho_max / w)), interpolation=cv2.INTER_AREA)

    # 1. Preprocesamiento: gris + ecualización local + suavizado
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gris = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gris)
    gris = cv2.GaussianBlur(gris, (5, 5), 0)

    # 2. Black-hat: resalta estructuras oscuras y delgadas (grietas) sobre fondo claro
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    blackhat = cv2.morphologyEx(gris, cv2.MORPH_BLACKHAT, kernel)

    # 3. Umbralización automática (Otsu) con un mínimo de contraste
    _, mascara = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mascara[blackhat < 18] = 0

    # 4. Limpieza: cerrar pequeños huecos y descartar manchas pequeñas o compactas
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(mascara, connectivity=8)
    limpia = np.zeros_like(mascara)
    area_min = 0.0004 * mascara.size
    for i in range(1, n):
        x, y, ancho, alto, area = stats[i]
        alargada = max(ancho, alto) >= 3 * max(1, min(ancho, alto)) or area > 4 * area_min
        if area >= area_min and alargada:
            limpia[etiquetas == i] = 255

    # 5. Resultados
    porcentaje = 100.0 * np.count_nonzero(limpia) / limpia.size
    overlay = img.copy()
    overlay[limpia > 0] = (0, 0, 255)                      # grietas en rojo (BGR)
    overlay = cv2.addWeighted(overlay, 0.8, img, 0.2, 0)
    contornos, _ = cv2.findContours(limpia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return {
        "original": cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        "detectada": cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
        "mascara": limpia,
        "porcentaje": float(porcentaje),
        "num_grietas": len(contornos),
        "severidad": clasificar_severidad(porcentaje),
    }