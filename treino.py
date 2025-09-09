from ultralytics import YOLO

def start_training():
    """
    Função que carrega o modelo e inicia o treinamento.
    """
    # Carregar um modelo pré-treinado.
    model = YOLO(r'C:\UFCG\poker\implementacao_artificial\yolov8n.pt') 

    # Treinamento
    resultado = model.train(
        data="C:/UFCG/poker/implementacao_artificial/poker_data.yaml",  # YAML que você mostrou
        epochs=50,
        batch=-1,           # quantidade de épocas (ajuste conforme precisar)
        imgsz=640,            # tamanho da imagem (padrão 640x640)
        lr0=0.01,
        lrf=0.01,                 # cosine final LR fraction
        momentum=0.937,
        weight_decay=5e-4,
        warmup_epochs=3.0,
        cls=1.0,
        box=7.5,
        dfl=1.5,
        cache="ram",
        patience=50,
        seed=42,
        #batch=32,             # batch size (ajuste para caber na sua GPU/CPU)
        device=0,             # 0 para GPU (cuda:0), 'cpu' para rodar sem GPU
        #workers=16,            # número de threads para dataloader
        name="cartas_yolo8_2000_2", # nome da pasta de saída em runs/detect
        #exist_ok=True,         # sobrescreve se já existir
        #optimizer="AdamW",  # otimizador mais moderno
        #cos_lr=True         # usa cosine learning rate (treino mais estável)
    )
    print("Treinamento concluído!")
    # O melhor modelo estará em 'poker_training_results/run1/weights/best.pt'

# Este bloco garante que o código só será executado quando o script
# for rodado diretamente, e não quando for importado.
if __name__ == '__main__':
    start_training()
