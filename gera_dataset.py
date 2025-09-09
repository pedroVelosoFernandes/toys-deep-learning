import os
import io
import json
import random
import numpy as np
from glob import glob
from PIL import Image, ImageFilter, ImageEnhance, ImageChops
import cv2
from datetime import datetime

# ------------- Caminhos -------------
BASE_DIR = r"C:\UFCG\poker\implementacao_artificial"
THEME_BASE = os.path.join(BASE_DIR, "theme")   # theme1 ... theme5
BG_DIR = os.path.join(BASE_DIR, "backgrounds")
OCCLUDER_DIR = os.path.join(BASE_DIR, "occluders")  # opcional: pngs de chips/mao/hud com alpha
OUTPUT_DIR = os.path.join(BASE_DIR, "dataset")

# ------------- Config -------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

N_IMAGES = 2500
TRAIN_RATIO = 0.8
N_TRAIN, N_VAL = int(N_IMAGES * TRAIN_RATIO), N_IMAGES - int(N_IMAGES * TRAIN_RATIO)

EXTRA_ROT_PROB = 0.5
CENTRO_PROB = 0.9
CENTRO_ROT_PROB = 0.2
MAO_SKIP_PROB = 0.3
CENTRO_CROP_PROB = 0.5
RUIDO_PROB = 0.2
DISTORCAO_PROB = 0.1

# Artifatos de câmera (são leves e muito úteis)
JPEG_ARTIFACT_PROB = 0.6
DOWNSCALE_UPSCALE_PROB = 0.6
GAMMA_PROB = 0.5
VIGNETTE_PROB = 0.4
DEFOCUS_PROB = 0.25
OCCLUDER_PROB = 0.35  # se houver

ROT_LIMIT_DEG = 22  # manter 6↔9 íntegros
MIN_BOX_SIZE = 6     # px; evita caixas degeneradas

# ------------- IO & Classes -------------
with open(os.path.join(BASE_DIR, "posicoes", "posicao_cartas.json"), "r", encoding="utf-8") as f:
    pos_data = json.load(f)
results = pos_data["annotations"][0]["result"]
img_w, img_h = results[0]["original_width"], results[0]["original_height"]
positions_hand = [res["value"] for res in results]

with open(os.path.join(BASE_DIR, "posicoes", "posicao_cartas_centro.json"), "r", encoding="utf-8") as f:
    pos_data_centro = json.load(f)
results_centro = pos_data_centro["annotations"][0]["result"]
positions_centro = [res["value"] for res in results_centro]

classes = ['10_Clubs', '10_Diamonds', '10_Hearts', '10_Spades', '2_Clubs', '2_Diamonds', '2_Hearts', '2_Spades',
           '3_Clubs', '3_Diamonds', '3_Hearts', '3_Spades', '4_Clubs', '4_Diamonds', '4_Hearts', '4_Spades',
           '5_Clubs', '5_Diamonds', '5_Hearts', '5_Spades', '6_Clubs', '6_Diamonds', '6_Hearts', '6_Spades',
           '7_Clubs', '7_Diamonds', '7_Hearts', '7_Spades', '8_Clubs', '8_Diamonds', '8_Hearts', '8_Spades',
           '9_Clubs', '9_Diamonds', '9_Hearts', '9_Spades', 'A_Clubs', 'A_Diamonds', 'A_Hearts', 'A_Spades',
           'J_Clubs', 'J_Diamonds', 'J_Hearts', 'J_Spades', 'K_Clubs', 'K_Diamonds', 'K_Hearts', 'K_Spades',
           'Q_Clubs', 'Q_Diamonds', 'Q_Hearts', 'Q_Spades']
class_to_id = {name: i for i, name in enumerate(classes)}

# Saída
for split in ["train", "val"]:
    os.makedirs(os.path.join(OUTPUT_DIR, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, split, "labels"), exist_ok=True)
manifest_path = os.path.join(OUTPUT_DIR, "manifest.jsonl")

# ------------- Pré-cache de assets -------------
def precache_themes(theme_base):
    themes = {}
    for i in range(1, 6):
        tdir = os.path.join(theme_base, f"theme{i}")
        if not os.path.exists(tdir):
            continue
        for f in os.listdir(tdir):
            if f.endswith(".png"):
                name = os.path.splitext(f)[0]
                full = os.path.join(tdir, f)
                # Atraso de IO compensa carregar lazy; guardo só paths para economia de RAM
                themes.setdefault(name, []).append(full)
    return themes

def precache_images_in_dir(d):
    if not os.path.exists(d): return []
    return glob(os.path.join(d, "*.png")) + glob(os.path.join(d, "*.jpg")) + glob(os.path.join(d, "*.jpeg"))

themes = precache_themes(THEME_BASE)
background_paths = precache_images_in_dir(BG_DIR)
occluder_paths = precache_images_in_dir(OCCLUDER_DIR)

def get_card_path(card_name):
    return random.choice(themes[card_name])

# ------------- Utils geom/labels -------------
def overlap(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)

def can_place(new_box, labels):
    return all(not overlap(new_box, b) for _, b in labels)

def clamp(v, lo, hi): return max(lo, min(hi, v))

def clamp_box(x, y, w, h, W, H):
    x = clamp(x, 0, W)
    y = clamp(y, 0, H)
    w = clamp(w, 0, W - x)
    h = clamp(h, 0, H - y)
    return x, y, w, h

def to_yolo_format(cls_id, box, W, H):
    x, y, w, h = box
    if w <= 0 or h <= 0: return None
    cx = (x + w / 2) / W
    cy = (y + h / 2) / H
    return f"{cls_id} {cx} {cy} {w / W} {h / H}"

# ------------- Efeitos visuais -------------
def add_drop_shadow(card_rgba, radius=6, offset=(4, 6), opacity=0.35):
    """Gera sombra suave sem mismatch de tamanhos."""
    assert card_rgba.mode == "RGBA"
    w, h = card_rgba.size
    ox, oy = offset
    pad = radius  # margem para o blur

    # Canvas acomoda blur + offset (positivo ou negativo)
    canvas_w = w + 2 * pad + abs(ox)
    canvas_h = h + 2 * pad + abs(oy)
    out = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))

    # Máscara borrada a partir do alfa da carta (tamanho w x h)
    mask_small = card_rgba.split()[-1].filter(ImageFilter.GaussianBlur(radius))

    # Máscara de sombra no tamanho do canvas; cola a pequena no deslocamento
    mask_full = Image.new("L", (canvas_w, canvas_h), 0)
    shadow_x = pad + max(ox, 0)
    shadow_y = pad + max(oy, 0)
    mask_full.paste(mask_small, (shadow_x, shadow_y))

    # Camada preta com opacidade desejada aplicada via máscara
    shadow_layer = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, int(255 * opacity)))
    out = Image.composite(shadow_layer, out, mask_full)

    # Cola a carta no canvas (compensando offsets negativos)
    card_x = pad + max(-ox, 0)
    card_y = pad + max(-oy, 0)
    out.paste(card_rgba, (card_x, card_y), card_rgba)
    return out

def jitter_card(card_rgba, brightness=(0.95, 1.05), contrast=(0.95, 1.05),
                saturation=(0.95, 1.05), hue=(-0.02, 0.02)):
    """Jitter leve sem usar Image.fromarray(..., mode='HSV') (evita DeprecationWarning)."""
    rgb = card_rgba.convert("RGB")
    rgb = ImageEnhance.Brightness(rgb).enhance(random.uniform(*brightness))
    rgb = ImageEnhance.Contrast(rgb).enhance(random.uniform(*contrast))

    hsv = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2HSV).astype(np.float32)
    # Saturation
    s_scale = random.uniform(*saturation)
    hsv[..., 1] = np.clip(hsv[..., 1] * s_scale, 0, 255)
    # Hue (OpenCV usa [0,180) para H)
    dh = random.uniform(*hue) * 180.0
    hsv[..., 0] = (hsv[..., 0] + dh) % 180

    rgb2 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    out = Image.fromarray(rgb2).convert("RGBA")
    out.putalpha(card_rgba.split()[-1])  # mantém o alfa original
    return out

def apply_camera_artifacts(img):
    # downscale->upscale
    if random.random() < DOWNSCALE_UPSCALE_PROB:
        scale = random.uniform(0.6, 0.95)
        w = max(8, int(img.width * scale))
        h = max(8, int(img.height * scale))
        resample_down = random.choice([Image.NEAREST, Image.BILINEAR, Image.BICUBIC])
        resample_up   = random.choice([Image.NEAREST, Image.BILINEAR, Image.BICUBIC])
        img = img.resize((w, h), resample=resample_down).resize((img_w, img_h), resample=resample_up)

    # gamma
    if random.random() < GAMMA_PROB:
        gamma = random.uniform(0.8, 1.2)
        arr = np.array(img).astype(np.float32) / 255.0
        arr = np.power(arr, gamma)
        img = Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8))

    # defocus leve
    if random.random() < DEFOCUS_PROB:
        radius = random.uniform(0.5, 1.8)
        img = img.filter(ImageFilter.GaussianBlur(radius))

    # vinheta
    if random.random() < VIGNETTE_PROB:
        y, x = np.ogrid[:img_h, :img_w]
        cx, cy = img_w / 2, img_h / 2
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        mask = r / r.max()
        strength = random.uniform(0.15, 0.35)
        vign = 1 - strength * (mask ** 1.8)
        arr = np.array(img).astype(np.float32)
        arr[..., :3] = np.clip(arr[..., :3] * vign[..., None], 0, 255)
        img = Image.fromarray(arr.astype(np.uint8))

    # compressão JPEG
    if random.random() < JPEG_ARTIFACT_PROB:
        q = random.randint(35, 90)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=q)
        buf.seek(0)
        img = Image.open(buf).convert("RGBA")
        img = img.resize((img_w, img_h), Image.BILINEAR)

    return img

# ------------- Colagem e deck -------------
def place_card(bg, card_path, pos, labels, crop_height=False, allow_extra_crop=False, allow_rotation=False):
    card_img = Image.open(card_path).convert("RGBA")

    # crop do centro (flop/turn/river)
    if crop_height:
        crop_h = int(card_img.height * 7/12)
        card_img = card_img.crop((0, 0, card_img.width, crop_h))
    elif allow_extra_crop and random.random() < CENTRO_CROP_PROB:
        crop_h = random.randint(int(card_img.height * 0.6), card_img.height)
        card_img = card_img.crop((0, 0, card_img.width, crop_h))

    # leve jitter para integrar com fundo
    card_img = jitter_card(card_img)

    # rotação pequena sem virar 6↔9
    if allow_rotation and random.random() < CENTRO_ROT_PROB:
        angle = random.uniform(-ROT_LIMIT_DEG, ROT_LIMIT_DEG)
        card_img = card_img.rotate(angle, expand=True)

    # drop shadow (fica natural)
    card_img = add_drop_shadow(card_img, radius=6, offset=(4, 6), opacity=0.35)

    x = int(pos["x"] / 100 * img_w)
    y = int(pos["y"] / 100 * img_h)
    w = int(pos["width"] / 100 * img_w)
    h = int(pos["height"] / 100 * img_h)
    new_box = (x, y, w, h)

    if not can_place(new_box, labels):
        return None

    card_resized = card_img.resize((w, h), Image.BILINEAR)
    bg.paste(card_resized, (x, y), card_resized)

    card_name = os.path.splitext(os.path.basename(card_path))[0]
    cls_id = class_to_id[card_name]
    return cls_id, new_box

def add_extra_rotated_cards(bg, labels, used_cards):
    # objetos “soltos” decorativos (sem deck rígido)
    for _ in range(2):
        card_name = random.choice([c for c in classes if c not in used_cards])
        used_cards.add(card_name)
        card_path = get_card_path(card_name)
        card_img = Image.open(card_path).convert("RGBA")
        angle = random.uniform(-ROT_LIMIT_DEG, ROT_LIMIT_DEG)
        card_img = add_drop_shadow(card_img.rotate(angle, expand=True))
        w = random.randint(int(img_w * 0.05), int(img_w * 0.15))
        aspect = card_img.height / card_img.width
        h = int(w * aspect)
        x = random.randint(0, img_w - w)
        y = random.randint(0, img_h - h)
        new_box = (x, y, w, h)
        if not can_place(new_box, labels):
            continue
        card_resized = card_img.resize((w, h), Image.BILINEAR)
        bg.paste(card_resized, (x, y), card_resized)
        labels.append((class_to_id[card_name], new_box))

def maybe_add_occluder(bg, labels):
    if not occluder_paths or random.random() > OCCLUDER_PROB:
        return
    occ = Image.open(random.choice(occluder_paths)).convert("RGBA")
    scale = random.uniform(0.2, 0.6)
    w = int(occ.width * scale)
    h = int(occ.height * scale)
    if w < 8 or h < 8: return
    occ = occ.resize((w, h), Image.BILINEAR)
    x = random.randint(0, img_w - w)
    y = random.randint(0, img_h - h)
    bg.paste(occ, (x, y), occ)
    # não geramos caixa para o oclusor; serve só para “sujar” a cena

# ------------- Warping global e boxes -------------
def aplicar_distorcao_global(im, labels):
    arr = np.array(im)
    H, W = arr.shape[:2]
    src = np.float32([[0, 0], [W, 0], [0, H], [W, H]])
    jitter = int(0.06 * min(W, H))  # 6% do tamanho, mais seguro
    dst = src + np.random.randint(-jitter, jitter + 1, src.shape).astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(arr, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    new_labels = []
    for cls_id, (x, y, w, h) in labels:
        if w < MIN_BOX_SIZE or h < MIN_BOX_SIZE: 
            continue
        pts = np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]], dtype=np.float32).reshape(-1, 1, 2)
        pts_warped = cv2.perspectiveTransform(pts, M).reshape(-1, 2)
        x_min, y_min = float(pts_warped[:, 0].min()), float(pts_warped[:, 1].min())
        x_max, y_max = float(pts_warped[:, 0].max()), float(pts_warped[:, 1].max())
        nw, nh = x_max - x_min, y_max - y_min
        if nw < MIN_BOX_SIZE or nh < MIN_BOX_SIZE:
            continue
        xx, yy, nw, nh = clamp_box(int(x_min), int(y_min), int(nw), int(nh), W, H)
        if nw < MIN_BOX_SIZE or nh < MIN_BOX_SIZE:
            continue
        new_labels.append((cls_id, (xx, yy, nw, nh)))

    out = Image.fromarray(warped).convert("RGBA")
    return out, new_labels

# ------------- Dataset loop -------------
def generate_dataset(n_samples, split_name, start_idx=0):
    os.makedirs(os.path.join(OUTPUT_DIR, split_name, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, split_name, "labels"), exist_ok=True)

    with open(manifest_path, "a", encoding="utf-8") as mf:
        for idx in range(n_samples):
            bg_path = random.choice(background_paths)
            bg = Image.open(bg_path).convert("RGBA").resize((img_w, img_h), Image.BILINEAR)
            labels = []

            # deck sem reposição para a imagem (evita duplicar a MESMA carta na cena)
            deck = set(classes)
            used_cards = set()

            # mão (12 posições)
            for pos, card_name in zip(positions_hand, random.sample(list(deck), min(12, len(deck)))):
                if random.random() < MAO_SKIP_PROB:
                    continue
                deck.discard(card_name); used_cards.add(card_name)
                card_path = get_card_path(card_name)
                result = place_card(bg, card_path, pos, labels, crop_height=True)
                if result:
                    labels.append(result)

            # centro
            if random.random() < CENTRO_PROB:
                n_cent = random.randint(1, min(5, len(deck)))
                for pos in random.sample(positions_centro, n_cent):
                    card_name = random.choice(list(deck))
                    deck.discard(card_name); used_cards.add(card_name)
                    card_path = get_card_path(card_name)
                    result = place_card(bg, card_path, pos, labels,
                                        crop_height=False,
                                        allow_extra_crop=True,
                                        allow_rotation=True)
                    if result:
                        labels.append(result)

            # extras rotacionais (decorativos)
            if random.random() < EXTRA_ROT_PROB:
                add_extra_rotated_cards(bg, labels, used_cards)

            # oclusores plausíveis
            maybe_add_occluder(bg, labels)

            # augment leve antes do warp global
            if random.random() < RUIDO_PROB:
                # ruído gaussiano leve + clip
                arr = np.array(bg).astype(np.int16)
                noise = np.random.normal(0, 14, arr.shape).astype(np.int16)
                bg = Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))

            # warping global (perspectiva) com atualização de boxes
            labels_final = labels
            if random.random() < DISTORCAO_PROB:
                bg, labels_final = aplicar_distorcao_global(bg, labels)

            # artefatos de câmera (muito úteis)
            bg = apply_camera_artifacts(bg)

            # garante pelo menos uma label válida
            labels_final = [ (c, clamp_box(x, y, w, h, img_w, img_h)) 
                             for c, (x, y, w, h) in labels_final
                             if w >= MIN_BOX_SIZE and h >= MIN_BOX_SIZE ]
            if not labels_final:
                # force pelo menos uma carta simples para não desperdiçar amostra
                pos = random.choice(positions_centro)
                if deck:
                    cn = random.choice(list(deck))
                    cp = get_card_path(cn)
                    r = place_card(bg, cp, pos, [], crop_height=False)
                    if r:
                        labels_final = [r]

            # salvar
            img_name = f"{split_name}_{start_idx + idx}.png"
            txt_name = f"{split_name}_{start_idx + idx}.txt"
            img_out = os.path.join(OUTPUT_DIR, split_name, "images", img_name)
            txt_out = os.path.join(OUTPUT_DIR, split_name, "labels", txt_name)

            bg.save(img_out)
            # YOLO txt
            with open(txt_out, "w") as f:
                lines = []
                for c, b in labels_final:
                    line = to_yolo_format(c, b, img_w, img_h)
                    if line: lines.append(line)
                f.write("\n".join(lines))

            # manifesto (debug/reprodutibilidade)
            mf.write(json.dumps({
                "split": split_name,
                "index": start_idx + idx,
                "bg": os.path.basename(bg_path),
                "used_cards": sorted(list(used_cards)),
                "labels": [{"cls": int(c), "box": list(map(int, b))} for c, b in labels_final],
                "seed": SEED,
                "timestamp": datetime.now().isoformat(timespec="seconds")
            }, ensure_ascii=False) + "\n")

# ------------- Rodar -------------
generate_dataset(N_TRAIN, "train", 0)
generate_dataset(N_VAL, "val", 0)
print("✅ Dataset gerado com deck por imagem, sombras, artefatos de câmera e labels saneados.")
